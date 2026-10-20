#!/usr/bin/env python3
"""
Write the approved descriptions, role links and Wikidata identifiers to Omeka.

Reads ``plan_descriptions.csv``, ``plan_roles.csv`` and ``plan_identifiers.csv``
from a run folder and applies every row whose ``approve`` is ``yes``, through
the shared write gate (``--dry-run``, confirmation, pre-write backup):

- A description becomes two ``dcterms:description`` literals, tagged
  ``@language`` ``fr`` and ``en``. A record that already holds a description is
  held back: this step fills gaps and never replaces anyone's text.
- A role becomes a ``dcterms:isPartOf`` link to the category item.
- A Wikidata match becomes a ``dcterms:identifier`` link
  (``https://www.wikidata.org/wiki/Q…``, labelled with the QID), the shape every
  authority already uses. A record that already has a Wikidata item is held
  back. Identifiers carry no model annotation: each is one click from the
  entity it names, and the curator approved it.

Provenance follows who wrote what. A description written exactly as the
reviewer drafted it is annotated ``iwac:descriptionModel`` → the reviewer's
model item, and a role the reviewer proposed is annotated ``iwac:nerModel``. A
description the curator edited or wrote, and a role the curator chose on a
``flagged`` row, are written without an annotation: the text or the choice is
then the curator's. ``invalid`` rows are never written.

Until ``iwac:descriptionModel`` is installed, a dry run reports what it would
annotate and a live run refuses.

Each write is checked by fetching the item again: the new values must be there
and every other value must still be.

Usage:
    python AI_authority_curation/03_apply.py --dry-run
    python AI_authority_curation/03_apply.py --run output/organisations_20261017T160000
    python AI_authority_curation/03_apply.py --only roles --dry-run
"""
from __future__ import annotations

import argparse
import copy
import csv
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.console_utils import count_table  # noqa: E402
from common.iwac_config import (  # noqa: E402
    IWAC_NER_MODEL_PROPERTY_ID,
    model_annotation_value,
    resolve_property_ids,
)
from common.log_redaction import configure_logging  # noqa: E402
from common.omeka_client import OmekaClient  # noqa: E402
from common.omeka_link_updater import (  # noqa: E402
    NER_MODEL_LABEL,
    ResourceLinkSpec,
    update_item_resource_links,
)
from common.omeka_text_updater import PropertyTarget, open_backup, update_item_text  # noqa: E402
from common.write_guard import WriteGuard, add_write_guard_args  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curation_core import (  # noqa: E402
    DESCRIPTION_MODEL_TERM,
    LINK_MODEL_TERM,
    REVIEWER_MODEL_KEY,
    description_problem,
)
from wikidata import ITEM_URL, record_qids  # noqa: E402

logger = logging.getLogger(__name__)
console = Console()
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
APPROVED = {"yes", "y", "oui", "o"}
DESCRIPTION_MODEL_LABEL = "AI Model - Description"


def newest_run(output_dir: Path) -> Optional[Path]:
    runs = sorted((p for p in output_dir.glob("*_*") if (p / "manifest.json").exists()),
                  key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def approved_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if (r.get("approve") or "").strip().lower() in APPROVED]
    skipped = [r for r in rows if r.get("status") == "invalid"]
    for row in skipped:
        console.print(f"[yellow]Skipping {row['authority_id']}: approved but invalid "
                      f"({row.get('problems')})[/]")
    return [r for r in rows if r.get("status") != "invalid"]


def value_count(item: Dict[str, Any]) -> int:
    return sum(len(v) for k, v in item.items()
               if ":" in k and not k.startswith("o:") and isinstance(v, list))


def is_reviewer_text(row: Dict[str, str]) -> bool:
    """True when the text about to be written is exactly the reviewer's draft.

    Authorship, not status, decides the annotation: a flagged draft the curator
    checked and approved unchanged is still the model's text.
    """
    return (row["status"] in ("ready", "flagged") and row.get("model") == REVIEWER_MODEL_KEY
            and bool(row["draft_fr"].strip())
            and row["description_fr"].strip() == row["draft_fr"].strip()
            and row["description_en"].strip() == row["draft_en"].strip())


def wikidata_value(qid: str, property_id: int) -> Dict[str, Any]:
    """A Wikidata link as every existing authority carries one."""
    return {"type": "uri", "property_id": property_id, "is_public": True,
            "@id": ITEM_URL.format(qid), "o:label": qid}


def add_wikidata(client: OmekaClient, aid: int, qid: str, property_id: int, guard: WriteGuard, sink) -> str:
    """Add one Wikidata identifier unless the record already has one."""
    item = client.get_item(aid)
    if not item:
        return "not_found"
    existing = record_qids(item)
    if existing:
        return f"held back: already has {', '.join(existing)}"
    if guard.dry_run:
        return "would_update"
    original = copy.deepcopy(item)
    item.setdefault("dcterms:identifier", []).append(wikidata_value(qid, property_id))
    if sink is not None:
        sink(original)
    if not client.update_item(aid, item):
        return "failed"
    after = client.get_item(aid) or {}
    ok = qid in record_qids(after) and value_count(after) == value_count(original) + 1
    return "updated, verified" if ok else "updated, CHECK FAILED"


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, help="Run folder (default: newest)")
    parser.add_argument("--only", choices=("descriptions", "roles", "identifiers"))
    add_write_guard_args(parser)
    args = parser.parse_args(argv)

    run_dir = args.run or newest_run(OUTPUT_DIR)
    if run_dir is None:
        console.print("[red]No run folder found.[/]")
        return 1
    guard = WriteGuard.from_args(args, default_backup_dir=run_dir / "backups")
    wanted = {args.only} if args.only else {"descriptions", "roles", "identifiers"}
    descriptions = approved_rows(run_dir / "plan_descriptions.csv") if "descriptions" in wanted else []
    roles = approved_rows(run_dir / "plan_roles.csv") if "roles" in wanted else []
    roles = [r for r in roles if (r.get("role_id") or "").strip().isdigit()]
    identifiers = approved_rows(run_dir / "plan_identifiers.csv") if "identifiers" in wanted else []
    identifiers = [r for r in identifiers if re.fullmatch(r"Q\d+", (r.get("qid") or "").strip())]
    if not descriptions and not roles and not identifiers:
        console.print("Nothing approved to write.")
        return 0

    client = OmekaClient.from_env()
    description_pid = client.get_property_id("dcterms:description")
    is_part_of_pid = client.get_property_id("dcterms:isPartOf")
    description_annotation = None
    if any(is_reviewer_text(r) for r in descriptions):
        try:
            pid = resolve_property_ids(client, [DESCRIPTION_MODEL_TERM])[DESCRIPTION_MODEL_TERM]
        except KeyError:
            message = (f"{DESCRIPTION_MODEL_TERM} is not installed. Upload v1.5 of iwac-vocabulary.ttl "
                       "first (Admin → Vocabularies → IWAC Ontology → Update).")
            if not guard.dry_run:
                console.print(f"[red]{message}[/]")
                return 1
            console.print(f"[yellow]{message} Dry run continues; nothing is sent.[/]")
            pid = -1
        description_annotation = model_annotation_value(client.base_url, REVIEWER_MODEL_KEY, pid,
                                                        DESCRIPTION_MODEL_LABEL)
    link_annotation = model_annotation_value(client.base_url, REVIEWER_MODEL_KEY,
                                             IWAC_NER_MODEL_PROPERTY_ID, NER_MODEL_LABEL)

    identifier_pid = client.get_property_id("dcterms:identifier")
    annotated = sum(is_reviewer_text(r) for r in descriptions)
    if not guard.confirm(
        console, action="Fill empty descriptions, add role links and Wikidata identifiers",
        base_url=client.base_url,
        item_count=len({r["authority_id"] for r in descriptions + roles + identifiers}),
        details=[f"Descriptions:  {len(descriptions)} ({annotated} annotated {DESCRIPTION_MODEL_TERM})",
                 f"Role links:    {len(roles)} (annotated {LINK_MODEL_TERM} when ready)",
                 f"Wikidata:      {len(identifiers)}"],
    ):
        return 1

    results: List[Dict[str, Any]] = []
    with open_backup(guard.backup_dir if guard.backup_enabled else None,
                     label="authority_curation", dry_run=guard.dry_run) as sink:
        for row in descriptions:
            aid = int(row["authority_id"])
            fr, en = row["description_fr"].strip(), row["description_en"].strip()
            problem = description_problem(fr) or (en and description_problem(en))
            if problem:
                results.append({"action": "description", "authority_id": aid, "status": f"refused: {problem}"})
                continue
            stamp = is_reviewer_text(row)
            targets = [
                PropertyTarget("dcterms:description", description_pid, "Description",
                               annotation_term=DESCRIPTION_MODEL_TERM if stamp else None,
                               annotation_value=description_annotation if stamp else None,
                               language=lang)
                for lang in ("fr", "en")
            ]
            before: Dict[str, int] = {}

            def check(item: Dict[str, Any], before=before) -> Optional[str]:
                before["count"] = value_count(item)
                if any((v.get("@value") or "").strip() for v in item.get("dcterms:description", [])
                       if isinstance(v, dict)):
                    return "already has a description"
                return None

            status = update_item_text(client, aid, fr, targets[0], dry_run=guard.dry_run,
                                      extra_values=[(targets[1], en)] if en else [],
                                      backup=sink, check=check)
            if status == "updated":
                after = client.get_item(aid) or {}
                texts = {v.get("@language"): v.get("@value") for v in after.get("dcterms:description", [])}
                expected = before.get("count", -1) + (2 if en else 1)
                ok = texts.get("fr") == fr and (not en or texts.get("en") == en) and value_count(after) == expected
                status = "updated, verified" if ok else "updated, CHECK FAILED"
            results.append({"action": "description", "authority_id": aid, "status": status,
                            "annotated": stamp})

        for row in roles:
            aid, role_id = int(row["authority_id"]), int(row["role_id"])
            stamp = row["status"] == "ready" and row.get("model") == REVIEWER_MODEL_KEY
            item = client.get_item(aid)
            if not item:
                results.append({"action": "role", "authority_id": aid, "status": "not_found"})
                continue
            count = value_count(item)
            spec = ResourceLinkSpec("dcterms:isPartOf", is_part_of_pid, [role_id], "Is Part Of",
                                    annotation_term=LINK_MODEL_TERM if stamp else None,
                                    annotation_value=link_annotation if stamp else None)
            outcome = update_item_resource_links(client, aid, [spec], dry_run=guard.dry_run,
                                                 on_pre_write=sink, item_data=item)
            status = outcome.status
            if status == "updated":
                after = client.get_item(aid) or {}
                linked = {v.get("value_resource_id") for v in after.get("dcterms:isPartOf", [])}
                ok = role_id in linked and value_count(after) == count + 1
                status = "updated, verified" if ok else "updated, CHECK FAILED"
            results.append({"action": "role", "authority_id": aid, "status": status, "annotated": stamp})

        for row in identifiers:
            aid, qid = int(row["authority_id"]), row["qid"].strip()
            results.append({"action": "wikidata", "authority_id": aid,
                            "status": add_wikidata(client, aid, qid, identifier_pid, guard, sink)})

    stamp_time = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = run_dir / f"applied_{'dryrun_' if guard.dry_run else ''}{stamp_time}.csv"
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["action", "authority_id", "status", "annotated"])
        writer.writeheader()
        writer.writerows(results)
    tally: Dict[str, int] = {}
    for result in results:
        key = f"{result['action']}: {result['status']}"
        tally[key] = tally.get(key, 0) + 1
    console.print(count_table(sorted(tally.items()), title=guard.mode_label))
    console.print(f"Log: [cyan]{log_path}[/]")
    failed = [r for r in results if "FAILED" in r["status"] or r["status"] in ("failed", "not_found")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
