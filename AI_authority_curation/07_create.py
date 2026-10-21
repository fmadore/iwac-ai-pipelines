#!/usr/bin/env python3
"""
Create the approved new authority records and add approved alternative names.

Reads a run folder from ``06_ner_plan.py``:

- ``plan_create.csv`` — each approved row becomes a new item in its kind's set,
  class and template (``curation_core.new_record_payload``): title,
  alternatives, ``dcterms:type`` → Notice d'autorité, the French description
  annotated ``iwac:descriptionModel`` → the reviewer, and its category and
  country as ``iwac:nerModel``-annotated links.
- ``plan_alternatives.csv`` — each approved row adds the names an existing
  record lacks to its ``dcterms:alternative``, by whole-item PATCH.

Before anything is created, the titles and alternatives of every authority
set and of set 12826 are read once (one search per title would take hours):
a row whose title is already there is skipped as ``exists``. A journal beside
the plan records each creation before and after the POST, so a re-run never
creates a record twice and a POST interrupted mid-way is reported, not
repeated. Every created item is fetched again and its values counted; every
PATCHed item is backed up first and checked after.

Usage:
    python AI_authority_curation/07_create.py --run output/ner_burkina_faso_20261021T100000 --dry-run
    python AI_authority_curation/07_create.py --run output/ner_burkina_faso_20261021T100000
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
from common.checkpoint import JsonCheckpoint, fingerprint  # noqa: E402
from common.console_utils import count_table, standard_progress  # noqa: E402
from common.iwac_config import (  # noqa: E402
    AUTHORITY_RECORD_TYPE_ITEM_ID,
    IWAC_NER_MODEL_PROPERTY_ID,
    item_api_url,
    model_annotation_value,
    resolve_property_ids,
)
from common.log_redaction import configure_logging  # noqa: E402
from common.omeka_client import OmekaClient  # noqa: E402
from common.omeka_link_updater import NER_MODEL_LABEL  # noqa: E402
from common.omeka_text_updater import open_backup  # noqa: E402
from common.write_guard import WriteGuard, add_write_guard_args  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curation_core import (  # noqa: E402
    DESCRIPTION_MODEL_TERM,
    KINDS,
    REVIEWER_MODEL_KEY,
    SORT_TARGETS,
    UNSORTED_SET_ID,
    description_problem,
    new_record_payload,
    normalize_name,
    record_names,
    split_names,
)

logger = logging.getLogger(__name__)
console = Console()
APPROVED = {"yes", "y", "oui", "o"}
DESCRIPTION_MODEL_LABEL = "AI Model - Description"
TERMS = ("dcterms:title", "dcterms:alternative", "dcterms:description", "dcterms:type",
         "dcterms:isPartOf", "dcterms:spatial")


def value_count(item: Dict[str, Any]) -> int:
    return sum(len(v) for k, v in item.items()
               if ":" in k and not k.startswith("o:") and isinstance(v, list))


def approved(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [r for r in csv.DictReader(handle) if (r.get("approve") or "").strip().lower() in APPROVED]


def name_index(client: OmekaClient) -> Dict[str, List[int]]:
    """Every folded title and alternative in the authority sets and set 12826."""
    index: Dict[str, List[int]] = {}
    set_ids = sorted({k.item_set_id for k in KINDS.values()} | {UNSORTED_SET_ID, 267})
    for set_id in set_ids:
        with console.status(f"Reading names in set {set_id}…"):
            for item in client.iter_items(set_id):
                for name in record_names(item):
                    index.setdefault(normalize_name(name), []).append(int(item["o:id"]))
    return index


def is_model_text(row: Dict[str, str]) -> bool:
    """True when the description is exactly the reviewer's draft: a curator's
    rewrite of ``description_fr`` is written without the model annotation."""
    return (row.get("model") == REVIEWER_MODEL_KEY
            and row["description_fr"].strip() == (row.get("draft_fr") or "").strip())


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="Run folder written by 06_ner_plan.py")
    parser.add_argument("--only", choices=("create", "alternatives"), help="Apply one plan only")
    add_write_guard_args(parser)
    args = parser.parse_args(argv)

    creates = [] if args.only == "alternatives" else approved(args.run / "plan_create.csv")
    alternatives = [] if args.only == "create" else approved(args.run / "plan_alternatives.csv")
    if not creates and not alternatives:
        console.print("Nothing approved to write.")
        return 0
    usable = []
    for row in creates:
        text = row["description_fr"].strip()
        problem = ("unknown kind" if row["kind"] not in SORT_TARGETS
                   else description_problem(text) if text else None)
        if problem:
            console.print(f"[yellow]Refusing {row['title']!r}: {problem}[/]")
        else:
            usable.append(row)
    creates = usable

    guard = WriteGuard.from_args(args, default_backup_dir=args.run / "backups")
    client = OmekaClient.from_env()
    props = {term: client.get_property_id(term) for term in TERMS}
    props[DESCRIPTION_MODEL_TERM] = resolve_property_ids(client, [DESCRIPTION_MODEL_TERM])[DESCRIPTION_MODEL_TERM]
    description_annotation = model_annotation_value(client.base_url, REVIEWER_MODEL_KEY,
                                                    props[DESCRIPTION_MODEL_TERM], DESCRIPTION_MODEL_LABEL)
    link_annotation = model_annotation_value(client.base_url, REVIEWER_MODEL_KEY,
                                             IWAC_NER_MODEL_PROPERTY_ID, NER_MODEL_LABEL)
    type_value = {"type": "customvocab:6", "is_public": True, "value_resource_name": "items",
                  "@id": item_api_url(client.base_url, AUTHORITY_RECORD_TYPE_ITEM_ID),
                  "value_resource_id": AUTHORITY_RECORD_TYPE_ITEM_ID}

    def link_value(rid: int) -> Dict[str, Any]:
        return {"type": "resource:item", "is_public": True, "value_resource_name": "items",
                "@id": item_api_url(client.base_url, rid), "value_resource_id": rid}

    if not guard.confirm(console, action="Create authority records and add alternative names",
                         base_url=client.base_url, item_count=len(creates) + len(alternatives),
                         details=[f"New records:   {len(creates)}",
                                  f"Alternatives:  {len(alternatives)} existing records"],
                         title="About to write authority records"):
        return 1

    index = name_index(client) if creates else {}
    results: List[Dict[str, Any]] = []
    journal = None if guard.dry_run else JsonCheckpoint.open(
        args.run / "create_journal.json", {"base_url": client.base_url, "plan": "plan_create.csv", "version": 1})
    created_log = None
    if not guard.dry_run and creates:
        created_log = (args.run / f"created_{datetime.now():%Y%m%dT%H%M%S}.csv").open("x", encoding="utf-8",
                                                                                      newline="")
        created_writer = csv.DictWriter(created_log, fieldnames=["title", "kind", "o:id"])
        created_writer.writeheader()

    with standard_progress(console) as progress:
        task = progress.add_task("Creating records", total=len(creates))
        for row in creates:
            progress.advance(task)
            title = row["title"].strip()
            clash = sorted(set(index.get(normalize_name(title), [])))
            if clash:
                results.append({"action": "create", "title": title, "status": f"exists: {clash[:3]}"})
                continue
            stamp = is_model_text(row)
            payload = new_record_payload(row, props, type_value=type_value,
                                         description_annotation=description_annotation if stamp else None,
                                         link_annotation=link_annotation, link_value=link_value)
            if guard.dry_run:
                results.append({"action": "create", "title": title, "status": "would_create",
                                "values": value_count(payload)})
                continue
            key = fingerprint({"kind": row["kind"], "title": normalize_name(title)})
            state = journal.entries.get(key)
            if state:
                status = ("already created" if state != "pending"
                          else "INTERRUPTED: check Omeka for this title before re-running")
                results.append({"action": "create", "title": title, "status": status, "o:id": state})
                continue
            # A failed POST stays "pending": a timeout can still have created
            # the item, so the next run reports it instead of creating again.
            journal.mark(key, "pending")
            result = client.create_item(payload)
            if not result:
                results.append({"action": "create", "title": title, "status": "failed"})
                continue
            new_id = int(result["o:id"])
            journal.mark(key, str(new_id))
            created_writer.writerow({"title": title, "kind": row["kind"], "o:id": new_id})
            created_log.flush()
            index.setdefault(normalize_name(title), []).append(new_id)
            after = client.get_item(new_id) or {}
            ok = after.get("o:title") == title and value_count(after) == value_count(payload)
            results.append({"action": "create", "title": title, "o:id": new_id,
                            "status": "created, verified" if ok else "created, CHECK FAILED"})
    if created_log is not None:
        created_log.close()

    with open_backup(guard.backup_dir if guard.backup_enabled else None,
                     label="authority_alternatives", dry_run=guard.dry_run) as sink:
        for row in alternatives:
            aid = int(row["authority_id"])
            item = client.get_item(aid)
            if not item:
                results.append({"action": "alternatives", "o:id": aid, "status": "not_found"})
                continue
            original = copy.deepcopy(item)
            have = {normalize_name(n) for n in record_names(item)}
            added = 0
            for name in split_names(row["alternatives"]):
                if normalize_name(name) not in have:
                    have.add(normalize_name(name))
                    item.setdefault("dcterms:alternative", []).append(
                        {"type": "literal", "property_id": props["dcterms:alternative"], "is_public": True,
                         "@value": name})
                    added += 1
            if not added:
                results.append({"action": "alternatives", "o:id": aid, "status": "unchanged"})
                continue
            if guard.dry_run:
                results.append({"action": "alternatives", "o:id": aid, "status": "would_update",
                                "values": added})
                continue
            if sink is not None:
                sink(original)
            if not client.update_item(aid, item):
                results.append({"action": "alternatives", "o:id": aid, "status": "failed"})
                continue
            after = client.get_item(aid) or {}
            ok = value_count(after) == value_count(original) + added
            results.append({"action": "alternatives", "o:id": aid, "values": added,
                            "status": "updated, verified" if ok else "updated, CHECK FAILED"})

    log_path = args.run / f"written_{'dryrun_' if guard.dry_run else ''}{datetime.now():%Y%m%dT%H%M%S}.csv"
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["action", "title", "o:id", "status", "values"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    tally: Dict[str, int] = {}
    for result in results:
        key = f"{result['action']}: {result['status'].split(':')[0]}"
        tally[key] = tally.get(key, 0) + 1
    console.print(count_table(sorted(tally.items()), title=guard.mode_label))
    console.print(f"Log: [cyan]{log_path}[/]")
    bad = [r for r in results if "FAILED" in r["status"] or r["status"] in ("failed", "not_found")
           or r["status"].startswith("INTERRUPTED")]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
