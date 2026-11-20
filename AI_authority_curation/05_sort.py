#!/usr/bin/env python3
"""
File approved stubs from "Notices d'autorités à traiter" into the authority sets.

Reads ``plan_sort.csv`` from an unsorted run and, for each row whose ``approve``
is ``yes``, PATCHes the stub whole so that it follows its kind's template:

- resource class and template of the target kind (person 94/5, organisation
  96/7, subject 244/3, place 9/6, event 54/2), the kind's item set in place of
  set 12826, and ``dcterms:type`` → "Notice d'autorité" (67568) if missing;
- the cleaned ``new_title`` as ``dcterms:title``, and each name in
  ``alternatives`` as ``dcterms:alternative`` unless the record already
  carries it;
- for persons, ``birth_year`` / ``death_year`` as ``foaf:birthday`` /
  ``dcterms:date`` (year only), unless those fields are already set.

Documents' links then follow the kind: a stub filed as a place and cited
through ``dcterms:subject`` moves to ``dcterms:spatial`` on every linking
item, and a stub filed as anything else moves the other way. Annotations
travel with the link, and no property ends up naming the record twice.

A stub the plan lists as already filed elsewhere only leaves set 12826.
Stubs that duplicate an existing authority are not filed here: they are rows
of ``plan_merges.csv``, for ``04_merge.py``. Descriptions and roles are rows of
the usual plans, for ``03_apply.py`` — run this step first, so they land on a
filed record.

Every item is backed up before its PATCH and fetched again after it.

Usage:
    python AI_authority_curation/05_sort.py --run output/unsorted_20261017T190000 --dry-run
    python AI_authority_curation/05_sort.py --run output/unsorted_20261017T190000
"""
from __future__ import annotations

import argparse
import copy
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.console_utils import count_table  # noqa: E402
from common.iwac_config import AUTHORITY_RECORD_TYPE_ITEM_ID  # noqa: E402
from common.log_redaction import configure_logging  # noqa: E402
from common.omeka_client import OmekaClient  # noqa: E402
from common.omeka_text_updater import open_backup  # noqa: E402
from common.write_guard import WriteGuard, add_write_guard_args  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curation_core import (  # noqa: E402
    KINDS,
    SORT_TARGETS,
    TOPICAL_TERMS,
    UNSORTED_SET_ID,
    move_topical_links,
    normalize_name,
    record_names,
    repeated_links,
    topical_term,
)

logger = logging.getLogger(__name__)
console = Console()
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
APPROVED = {"yes", "y", "oui", "o"}


def value_count(item: Dict[str, Any]) -> int:
    return sum(len(v) for k, v in item.items()
               if ":" in k and not k.startswith("o:") and isinstance(v, list))


def newest_unsorted_run(output_dir: Path) -> Optional[Path]:
    runs = sorted(output_dir.glob("unsorted_*"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def file_stub(item: Dict[str, Any], row: Dict[str, str], props: Dict[str, int]) -> int:
    """Apply one plan row to a fetched stub, in place. Returns values added."""
    kind = KINDS[row["target"]]
    added = 0
    sets = [s for s in item.get("o:item_set", []) if s.get("o:id") != UNSORTED_SET_ID]
    if kind.item_set_id not in {s.get("o:id") for s in sets}:
        sets.append({"o:id": kind.item_set_id})
    item["o:item_set"] = sets
    if not row.get("model") and (row.get("note") or "").startswith("already in"):
        return added  # filed elsewhere already: it only leaves the unsorted set

    item["o:resource_class"] = {"o:id": kind.resource_class_id}
    item["o:resource_template"] = {"o:id": kind.template_id}
    types = item.setdefault("dcterms:type", [])
    if not any(v.get("value_resource_id") == AUTHORITY_RECORD_TYPE_ITEM_ID for v in types):
        types.append({"type": "customvocab:6", "property_id": props["dcterms:type"], "is_public": True,
                      "value_resource_id": AUTHORITY_RECORD_TYPE_ITEM_ID})
        added += 1

    new_title = (row.get("new_title") or "").strip()
    if new_title:
        titles = item.setdefault("dcterms:title", [])
        if titles:
            titles[0]["@value"] = new_title
        else:
            titles.append({"type": "literal", "property_id": props["dcterms:title"], "is_public": True,
                           "@value": new_title})
            added += 1

    have = {normalize_name(n) for n in record_names({**item, "o:title": new_title or item.get("o:title")})}
    for name in (n.strip() for n in (row.get("alternatives") or "").split("|")):
        if name and normalize_name(name) not in have:
            item.setdefault("dcterms:alternative", []).append(
                {"type": "literal", "property_id": props["dcterms:alternative"], "is_public": True,
                 "@value": name})
            have.add(normalize_name(name))
            added += 1

    if row["target"] == "persons":
        for column, term in (("birth_year", "foaf:birthday"), ("death_year", "dcterms:date")):
            year = (row.get(column) or "").strip()
            if year and not item.get(term):
                item[term] = [{"type": "numeric:timestamp", "property_id": props[term], "is_public": True,
                               "@value": year}]
                added += 1
    return added


def relocate_links(client: OmekaClient, aid: int, kind: str, topical_ids: Dict[str, int],
                   dry_run: bool, sink) -> Dict[str, int]:
    """Move documents' links to a filed stub into its kind's topical property.

    A stub filed as a place and cited through ``dcterms:subject`` is moved to
    ``dcterms:spatial``, a stub filed as anything else the other way. Each
    linking item is PATCHed whole and fetched again.
    """
    wrong = next(t for t in TOPICAL_TERMS if t != topical_term(kind))
    linking = client.get_items(**{"property[0][property]": topical_ids[wrong],
                                  "property[0][type]": "res", "property[0][text]": aid})
    report = {"links_moved": 0, "link_failures": 0, "linking_items": len(linking)}
    for listed in linking:
        item_id = int(listed["o:id"])
        item = client.get_item(item_id) if not dry_run else listed
        if not item:
            report["link_failures"] += 1
            continue
        original = copy.deepcopy(item)
        moved, dropped = move_topical_links(item, aid, kind, topical_ids)
        report["links_moved"] += moved + dropped
        if dry_run or not (moved or dropped):
            continue
        if sink is not None:
            sink(original)
        if not client.update_item(item_id, item):
            report["link_failures"] += 1
            continue
        after = client.get_item(item_id) or {}
        left = any(isinstance(v, dict) and v.get("value_resource_id") == aid for v in after.get(wrong, []))
        if left or repeated_links(after, [aid]) or value_count(after) != value_count(original) - dropped:
            report["link_failures"] += 1
    return report


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, help="Unsorted run folder (default: newest)")
    add_write_guard_args(parser)
    args = parser.parse_args(argv)

    run_dir = args.run or newest_unsorted_run(OUTPUT_DIR)
    plan = run_dir / "plan_sort.csv" if run_dir else None
    if plan is None or not plan.exists():
        console.print("[red]No plan_sort.csv found: build the plan of an unsorted run first.[/]")
        return 1
    with plan.open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if (r.get("approve") or "").strip().lower() in APPROVED]
    usable = []
    for row in rows:
        if row["status"] == "invalid" or row["target"] not in SORT_TARGETS:
            console.print(f"[yellow]Skipping {row['authority_id']}: approved but "
                          f"{'invalid' if row['status'] == 'invalid' else 'no target kind'}[/]")
        else:
            usable.append(row)
    if not usable:
        console.print("Nothing approved to file.")
        return 0

    guard = WriteGuard.from_args(args, default_backup_dir=run_dir / "backups")
    client = OmekaClient.from_env()
    props = {term: client.get_property_id(term) for term in
             ("dcterms:type", "dcterms:title", "dcterms:alternative", "foaf:birthday", "dcterms:date")}
    topical_ids = {term: client.get_property_id(term) for term in TOPICAL_TERMS}
    targets = sorted({r["target"] for r in usable})
    if not guard.confirm(console, action="File stubs into the authority sets", base_url=client.base_url,
                         item_count=len(usable),
                         details=[f"Into:          {', '.join(targets)}",
                                  "Documents' links follow the kind: places in dcterms:spatial, "
                                  "the rest in dcterms:subject."],
                         title="About to file authority records"):
        return 1

    results: List[Dict[str, Any]] = []
    with open_backup(guard.backup_dir if guard.backup_enabled else None,
                     label="authority_sort", dry_run=guard.dry_run) as sink:
        for row in usable:
            aid = int(row["authority_id"])
            item = client.get_item(aid)
            if not item:
                results.append({"authority_id": aid, "target": row["target"], "status": "not_found"})
                continue
            if UNSORTED_SET_ID not in {s.get("o:id") for s in item.get("o:item_set", [])}:
                results.append({"authority_id": aid, "target": row["target"], "status": "already moved"})
                continue
            original = copy.deepcopy(item)
            added = file_stub(item, row, props)
            if guard.dry_run:
                links = relocate_links(client, aid, row["target"], topical_ids, True, None)
                results.append({"authority_id": aid, "target": row["target"], "status": "would_file",
                                "values_added": added, **links})
                continue
            if sink is not None:
                sink(original)
            if not client.update_item(aid, item):
                results.append({"authority_id": aid, "target": row["target"], "status": "failed"})
                continue
            after = client.get_item(aid) or {}
            kind = KINDS[row["target"]]
            sets = {s.get("o:id") for s in after.get("o:item_set", [])}
            ok = (kind.item_set_id in sets and UNSORTED_SET_ID not in sets
                  and value_count(after) == value_count(original) + added
                  and (not row.get("new_title") or after.get("o:title") == row["new_title"].strip()))
            links = relocate_links(client, aid, row["target"], topical_ids, False, sink)
            ok = ok and not links["link_failures"]
            results.append({"authority_id": aid, "target": row["target"], "values_added": added, **links,
                            "status": "filed, verified" if ok else "filed, CHECK FAILED"})

    log_path = run_dir / f"sorted_{'dryrun_' if guard.dry_run else ''}{datetime.now():%Y%m%dT%H%M%S}.csv"
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["authority_id", "target", "status", "values_added",
                                                     "links_moved", "link_failures"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    tally: Dict[str, int] = {}
    for result in results:
        tally[result["status"]] = tally.get(result["status"], 0) + 1
    console.print(count_table(sorted(tally.items()), title=guard.mode_label))
    console.print(f"Log: [cyan]{log_path}[/]")
    return 0 if all(r["status"] in ("would_file", "filed, verified", "already moved") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
