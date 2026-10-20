#!/usr/bin/env python3
"""
Check the reviewers' verdicts and write the plan the curator approves.

Read-only. Every batch from ``01_collect.py`` is matched with the verdict file
its Haiku reviewer wrote, and each verdict is checked by
``curation_core``: written by ``claude-haiku-5-5``, one or two sentences in
French and in English, evidence drawn only from the documents the batch
supplied, a role from the categories offered. The review files:

  plan_descriptions.csv   one row per record that needed a description
  plan_roles.csv          one row per role proposed for a person
  plan_merges.csv         one row per candidate pair, and per stub that
                          duplicates an existing authority
  plan_identifiers.csv    one row per Wikidata item a reviewer matched
  plan_sort.csv           unsorted runs only: where each stub is filed

``status`` is ``ready`` (may be approved), ``flagged`` (the reviewer was not
sure, used general knowledge, or proposed a category that does not exist: your
call), ``invalid`` (broke a rule; never applied) or ``rejected`` (the reviewer
found the pair to be two entities). Put ``yes`` in ``approve`` for each row
to write. A description edited in the CSV is written without the model
annotation, since the text is then no longer the model's.

Usage:
    python AI_authority_curation/02_build_plan.py              # newest run
    python AI_authority_curation/02_build_plan.py --run output/organisations_20261017T160000
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.console_utils import count_table  # noqa: E402
from common.log_redaction import configure_logging  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curation_core import (  # noqa: E402
    REVIEWER_MODEL_KEY,
    check_describe_batch,
    check_duplicate_batch,
    check_triage_batch,
)

console = Console()
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

DESCRIPTION_COLUMNS = ["approve", "status", "authority_id", "title", "url", "description_fr",
                       "description_en", "draft_fr", "draft_en", "evidence", "problems", "note",
                       "batch", "model"]
ROLE_COLUMNS = ["approve", "status", "authority_id", "title", "url", "role_id", "role_title",
                "new_role", "evidence", "problems", "batch", "model"]
MERGE_COLUMNS = ["approve", "status", "survivor_id", "survivor_title", "survivor_url", "loser_id",
                 "loser_title", "loser_url", "score", "candidate_reason", "reviewer_reason",
                 "evidence", "problems", "batch", "model"]
SORT_COLUMNS = ["approve", "status", "authority_id", "current_title", "url", "target", "new_title",
                "alternatives", "birth_year", "death_year", "problems", "note", "batch", "model"]
IDENTIFIER_COLUMNS = ["approve", "status", "authority_id", "title", "url", "qid", "wikidata_label",
                      "wikidata_description", "wikidata_url", "reason", "problems", "batch", "model"]


def newest_run(output_dir: Path) -> Optional[Path]:
    runs = sorted((p for p in output_dir.glob("*_*") if (p / "manifest.json").exists()),
                  key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"batch": path.stem, "model": "", "results": "unparseable"}  # fails validation


def _ids(ids: List[int]) -> str:
    return " ".join(str(i) for i in ids)


def write_csv(path: Path, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, help="Run folder from 01_collect.py (default: newest)")
    args = parser.parse_args(argv)

    run_dir = args.run or newest_run(OUTPUT_DIR)
    if run_dir is None or not (run_dir / "manifest.json").exists():
        console.print("[red]No run folder with a manifest.json found.[/]")
        return 1
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    role_titles = {int(r["id"]): r["title"] for r in manifest.get("role_options", [])}

    descriptions: List[Dict[str, Any]] = []
    roles: List[Dict[str, Any]] = []
    merges: List[Dict[str, Any]] = []
    identifiers: List[Dict[str, Any]] = []
    sorts: List[Dict[str, Any]] = [
        {"approve": "", "status": "ready", "authority_id": a["id"], "current_title": a["title"],
         "url": "", "target": a["target"], "new_title": "", "alternatives": "", "birth_year": "",
         "death_year": "", "problems": "", "batch": "",
         "note": f"already in the {a['target']} set: only leaves the unsorted set", "model": ""}
        for a in manifest.get("already_filed", [])
    ]
    missing_files = 0

    for name in manifest["batches"]:
        batch = json.loads((run_dir / "batches" / f"{name}.json").read_text(encoding="utf-8"))
        verdict_file = load_json(run_dir / "verdicts" / f"{name}.json")
        missing_files += verdict_file is None
        model = (verdict_file or {}).get("model", "")

        if batch["task"] == "triage":
            entries = {int(a["authority"]["id"]): a for a in batch["authorities"]}
            for row in check_triage_batch(batch, verdict_file):
                entry = entries[row.authority_id]
                record, verdict = entry["authority"], row.verdict
                if row.action == "merge" and verdict:
                    survivor = next((c for c in entry["candidates"] if int(c["id"]) == verdict.duplicate_of),
                                    {"title": "", "url": ""})
                    merges.append({
                        "approve": "", "status": row.status,
                        "survivor_id": verdict.duplicate_of, "survivor_title": survivor["title"],
                        "survivor_url": survivor["url"], "loser_id": row.authority_id,
                        "loser_title": record["title"], "loser_url": record["url"],
                        "score": survivor.get("score", ""), "candidate_reason": survivor.get("reason", ""),
                        "reviewer_reason": verdict.note, "evidence": _ids(verdict.evidence),
                        "problems": "; ".join(row.problems), "batch": name, "model": model,
                    })
                else:
                    sorts.append({
                        "approve": "", "status": row.status, "authority_id": row.authority_id,
                        "current_title": record["title"], "url": record["url"],
                        "target": verdict.target if verdict else "",
                        "new_title": verdict.title if verdict else "",
                        "alternatives": " | ".join(verdict.alternatives) if verdict else "",
                        "birth_year": (verdict.birth_year or "") if verdict else "",
                        "death_year": (verdict.death_year or "") if verdict else "",
                        "problems": "; ".join(row.problems), "note": verdict.note if verdict else "",
                        "batch": name, "model": model,
                    })

        if batch["task"] in ("describe", "triage"):
            records = {int(a["authority"]["id"]): a["authority"] for a in batch["authorities"]}
            offered = {int(a["authority"]["id"]): {c["qid"]: c for c in a.get("wikidata_candidates", [])}
                       for a in batch["authorities"]}
            for row in check_describe_batch(batch, verdict_file):
                record = records.get(row.authority_id, {})
                verdict = row.verdict
                if row.wikidata_status != "none":
                    picked = offered.get(row.authority_id, {}).get(verdict.wikidata if verdict else "", {})
                    identifiers.append({
                        "approve": "", "status": row.wikidata_status, "authority_id": row.authority_id,
                        "title": record.get("title", ""), "url": record.get("url", ""),
                        "qid": (verdict.wikidata or "") if verdict else "",
                        "wikidata_label": picked.get("label", ""),
                        "wikidata_description": picked.get("description_fr") or picked.get("description_en", ""),
                        "wikidata_url": picked.get("url", ""),
                        "reason": verdict.wikidata_reason if verdict else "",
                        "problems": "; ".join(row.wikidata_problems), "batch": name, "model": model,
                    })
                if row.description_status != "none":
                    descriptions.append({
                        "approve": "", "status": row.description_status,
                        "authority_id": row.authority_id, "title": record.get("title", ""),
                        "url": record.get("url", ""),
                        "description_fr": verdict.description_fr if verdict else "",
                        "description_en": verdict.description_en if verdict else "",
                        "draft_fr": verdict.description_fr if verdict else "",
                        "draft_en": verdict.description_en if verdict else "",
                        "evidence": _ids(verdict.evidence) if verdict else "",
                        "problems": "; ".join(row.description_problems),
                        "note": verdict.note if verdict else "", "batch": name, "model": model,
                    })
                if row.role_status != "none":
                    roles.append({
                        "approve": "", "status": row.role_status,
                        "authority_id": row.authority_id, "title": record.get("title", ""),
                        "url": record.get("url", ""),
                        "role_id": verdict.role_id if verdict and verdict.role_id else "",
                        "role_title": role_titles.get(verdict.role_id, "") if verdict and verdict.role_id else "",
                        "new_role": (verdict.new_role or "") if verdict else "",
                        "evidence": _ids(verdict.role_evidence) if verdict else "",
                        "problems": "; ".join(row.role_problems), "batch": name, "model": model,
                    })
        elif batch["task"] == "duplicates":
            sides = {}
            for pair in batch["pairs"]:
                key = tuple(sorted((int(pair["a"]["id"]), int(pair["b"]["id"]))))
                sides[key] = pair
            for row in check_duplicate_batch(batch, verdict_file):
                pair = sides[row.pair]
                records = {int(pair["a"]["id"]): pair["a"], int(pair["b"]["id"]): pair["b"]}
                verdict = row.verdict
                survivor = verdict.survivor_id if verdict and verdict.survivor_id in records else row.pair[0]
                loser = row.pair[1] if survivor == row.pair[0] else row.pair[0]
                merges.append({
                    "approve": "", "status": row.status,
                    "survivor_id": survivor, "survivor_title": records[survivor]["title"],
                    "survivor_url": records[survivor]["url"],
                    "loser_id": loser, "loser_title": records[loser]["title"],
                    "loser_url": records[loser]["url"],
                    "score": pair["score"], "candidate_reason": pair["candidate_reason"],
                    "reviewer_reason": verdict.reason if verdict else "",
                    "evidence": _ids(verdict.evidence) if verdict else "",
                    "problems": "; ".join(row.problems), "batch": name, "model": model,
                })

    order = {"ready": 0, "flagged": 1, "invalid": 2, "rejected": 3}
    for rows in (descriptions, roles, merges, sorts, identifiers):
        rows.sort(key=lambda r: order.get(r["status"], 9))
    write_csv(run_dir / "plan_descriptions.csv", DESCRIPTION_COLUMNS, descriptions)
    write_csv(run_dir / "plan_roles.csv", ROLE_COLUMNS, roles)
    write_csv(run_dir / "plan_merges.csv", MERGE_COLUMNS, merges)
    write_csv(run_dir / "plan_identifiers.csv", IDENTIFIER_COLUMNS, identifiers)
    if manifest["kind"] == "unsorted":
        write_csv(run_dir / "plan_sort.csv", SORT_COLUMNS, sorts)

    for label, rows in (("Filing", sorts), ("Descriptions", descriptions), ("Roles", roles),
                        ("Wikidata", identifiers), ("Merges", merges)):
        counts = Counter(r["status"] for r in rows)
        if rows:
            console.print(count_table([(status, counts[status]) for status in
                                       ("ready", "flagged", "invalid", "rejected") if counts[status]],
                                      title=label))
    proposals = Counter(r["new_role"].strip() for r in roles if r["new_role"].strip())
    if proposals:
        console.print(count_table(proposals.most_common(),
                                  title="Proposed new role categories (create the useful ones, then re-collect)"))
    if missing_files:
        console.print(f"[yellow]{missing_files} batch(es) have no verdict file yet: "
                      f"their rows are marked invalid.[/]")
    console.print(f"Plan written to [cyan]{run_dir}[/]. Only rows written by {REVIEWER_MODEL_KEY} "
                  "can be ready; mark the ones to write with approve = yes.")
    return 1 if missing_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
