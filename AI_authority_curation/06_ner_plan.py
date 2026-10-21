#!/usr/bin/env python3
"""
Turn the review of NER leftovers into plans the write steps can apply.

``AI_NER/02`` leaves the entities it could not match in ``*_unreconciled_*``
files. The ``iwac-data`` skill's ``ner_leftovers`` scripts cluster them, have
Claude Haiku 5.5 reviewers judge each cluster from article snippets, and write
``proposals.json``. This step reads that file and writes a run folder:

  plan_create.csv        new records, for 07_create.py
  plan_alternatives.csv  names to add to existing records, for 07_create.py
  plan_sort.csv          stubs of set 12826 to file, for 05_sort.py
  plan_descriptions.csv  descriptions drafted for those stubs, for 03_apply.py
  plan_roles.csv         categories proposed for those stubs, for 03_apply.py
  plan_merges.csv        duplicate pairs, for 04_merge.py — never pre-approved

It writes nothing to Omeka. It reads the person, organisation and place sets
once to turn the reviewers' category and country labels into the items that
already stand for them; a label with no item is kept as text and linked to
nothing. Rows are pre-approved unless something needs the curator: a record
that may already exist, low confidence, a stub filed as a mere theme, any
merge.

Apply in this order: 05_sort.py, 03_apply.py, 07_create.py, then 04_merge.py
once the curator has approved merges.

Usage:
    python AI_authority_curation/06_ner_plan.py --proposals path/to/proposals.json --label burkina_faso
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.console_utils import count_table  # noqa: E402
from common.log_redaction import configure_logging  # noqa: E402
from common.omeka_client import OmekaClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curation_core import (  # noqa: E402
    ALTERNATIVE_COLUMNS,
    CREATE_COLUMNS,
    KINDS,
    NER_KIND,
    REVIEWER_MODEL_KEY,
    fold,
)

logger = logging.getLogger(__name__)
console = Console()
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SITE = "https://islam.zmo.de/s/afrique_ouest/item/"
#: "Pays": the category every country record is part of.
COUNTRY_CATEGORY_ID = 15835

SORT_COLUMNS = ["approve", "status", "authority_id", "current_title", "url", "target", "new_title",
                "alternatives", "birth_year", "death_year", "problems", "note", "batch", "model"]
DESCRIPTION_COLUMNS = ["approve", "status", "authority_id", "title", "url", "description_fr",
                       "description_en", "draft_fr", "draft_en", "evidence", "problems", "note",
                       "batch", "model"]
ROLE_COLUMNS = ["approve", "status", "authority_id", "title", "url", "role_id", "role_title",
                "new_role", "evidence", "problems", "batch", "model"]
MERGE_COLUMNS = ["approve", "status", "survivor_id", "survivor_title", "survivor_url", "loser_id",
                 "loser_title", "loser_url", "score", "candidate_reason", "reviewer_reason",
                 "evidence", "problems", "batch", "model"]


def label_maps(records: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Category labels and country names (folded) → the items that stand for them.

    Categories are whatever authority records already point at through
    ``dcterms:isPartOf``; countries are the places that are part of "Pays".
    """
    categories: Dict[str, int] = {}
    countries: Dict[str, int] = {}
    for record in records:
        parents = [v for v in record.get("dcterms:isPartOf", []) if isinstance(v, dict)]
        for value in parents:
            rid, title = value.get("value_resource_id"), value.get("display_title")
            if rid and title:
                categories.setdefault(fold(title), int(rid))
        if any(v.get("value_resource_id") == COUNTRY_CATEGORY_ID for v in parents):
            countries.setdefault(fold(record.get("o:title") or ""), int(record["o:id"]))
    return categories, countries


def needs_curator(row: Dict[str, Any]) -> bool:
    flags = row.get("signalements") or ""
    return row.get("confiance") == "low" or "needs_curator" in flags


def create_rows(new: List[Dict[str, Any]], categories: Dict[str, int],
                countries: Dict[str, int]) -> List[Dict[str, Any]]:
    rows = []
    for r in new:
        kind = NER_KIND.get(r["type"])
        exists = r.get("existe déjà ?") or ""
        exists = "" if exists.startswith("(") else exists
        problems = []
        if not kind:
            problems.append("no kind")
        if exists:
            problems.append("may already exist")
        category_id = categories.get(fold(r.get("catégorie") or ""), "")
        if category_id == countries.get(fold(r.get("pays") or "")):
            category_id = ""  # a country named as category: it is the country link
        rows.append({
            "approve": "" if problems or needs_curator(r) else "yes",
            "status": "check: " + ", ".join(problems) if problems else ("flagged" if needs_curator(r) else "ready"),
            "kind": kind or "", "title": r["titre proposé"], "alternatives": r["alternatifs"],
            "description_fr": r.get("description proposée") or "",
            "draft_fr": r.get("description proposée") or "",
            "category": r.get("catégorie") or "", "category_id": category_id,
            "country": r.get("pays") or "", "country_id": countries.get(fold(r.get("pays") or ""), ""),
            "mentions": r["mentions"], "confidence": r.get("confiance") or "",
            "flags": r.get("signalements") or "", "exists": exists, "evidence": r.get("articles") or "",
            "note": "; ".join(x for x in (r.get("note"), r.get("règle pays")) if x),
            "model": REVIEWER_MODEL_KEY,
        })
    return rows


def alternative_rows(existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{
        "approve": "" if needs_curator(r) or "match_not_offered" in (r.get("signalements") or "") else "yes",
        "status": "flagged" if needs_curator(r) else "ready",
        "authority_id": r["id"], "title": r["titre actuel"], "url": SITE + str(r["id"]),
        "kind": r["type actuel"], "current_alternatives": r["alternatifs actuels"],
        "alternatives": r["alternatifs à ajouter"], "mentions": r["mentions"],
        "confidence": r.get("confiance") or "", "flags": r.get("signalements") or "",
        "evidence": r.get("articles") or "", "note": r.get("note") or "", "model": REVIEWER_MODEL_KEY,
    } for r in existing if r.get("alternatifs à ajouter")]


def stub_rows(stubs: List[Dict[str, Any]], categories: Dict[str, int]) -> Tuple[List, List, List]:
    sort, descriptions, roles = [], [], []
    for r in stubs:
        kind = NER_KIND.get(r.get("classer comme") or "")
        theme = kind == "subjects"
        ready = bool(kind) and not needs_curator(r) and not theme
        url = SITE + str(r["id"])
        sort.append({
            "approve": "yes" if ready else "", "status": "ready" if kind else "invalid",
            "authority_id": r["id"], "current_title": r["titre actuel"], "url": url, "target": kind or "",
            "new_title": r.get("titre proposé") or "", "alternatives": r.get("alternatifs à ajouter") or "",
            "birth_year": "", "death_year": "",
            "problems": "a common theme: file only if the subject vocabulary should hold it" if theme else "",
            "note": r.get("note") or "", "batch": "", "model": REVIEWER_MODEL_KEY,
        })
        text = (r.get("description proposée") or "").strip()
        if text:
            descriptions.append({
                "approve": "yes" if ready else "", "status": "ready", "authority_id": r["id"],
                "title": r.get("titre proposé") or r["titre actuel"], "url": url,
                "description_fr": text, "description_en": "", "draft_fr": text, "draft_en": "",
                "evidence": r.get("articles") or "", "problems": "", "note": "", "batch": "",
                "model": REVIEWER_MODEL_KEY,
            })
        role_id = categories.get(fold(r.get("catégorie") or ""))
        if role_id:
            roles.append({
                "approve": "yes" if ready else "", "status": "ready", "authority_id": r["id"],
                "title": r.get("titre proposé") or r["titre actuel"], "url": url, "role_id": role_id,
                "role_title": r["catégorie"], "new_role": "", "evidence": r.get("articles") or "",
                "problems": "", "batch": "", "model": REVIEWER_MODEL_KEY,
            })
    return sort, descriptions, roles


def merge_rows(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The classified record survives a stub; otherwise the first id does."""
    rows = []
    for p in pairs:
        a, b = (p["id A"], p["titre A"], p["type A"]), (p["id B"], p["titre B"], p["type B"])
        survivor, loser = (b, a) if a[2] == "stub" and b[2] != "stub" else (a, b)
        rows.append({
            "approve": "", "status": "ready" if p.get("vraisemblance") == "plausible" else "flagged",
            "survivor_id": survivor[0], "survivor_title": survivor[1], "survivor_url": SITE + str(survivor[0]),
            "loser_id": loser[0], "loser_title": loser[1], "loser_url": SITE + str(loser[0]),
            "score": "", "candidate_reason": "NER leftover review",
            "reviewer_reason": p.get("note") or "", "evidence": "",
            "problems": "" if p.get("vraisemblance") == "plausible" else p.get("vraisemblance", ""),
            "batch": "", "model": REVIEWER_MODEL_KEY,
        })
    return rows


def write_csv(path: Path, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--proposals", type=Path, required=True, help="proposals.json from the review")
    parser.add_argument("--label", default="ner", help="Run folder label, e.g. the country (default: ner)")
    args = parser.parse_args(argv)

    proposals = json.loads(args.proposals.read_text(encoding="utf-8"))
    client = OmekaClient.from_env()
    records: List[Dict[str, Any]] = []
    for key in ("persons", "organisations", "places"):
        with console.status(f"Reading {key} for category and country items…"):
            records += client.get_items(KINDS[key].item_set_id)
    categories, countries = label_maps(records)

    sort, descriptions, roles = stub_rows(proposals["ebauches"], categories)
    plans = {
        "plan_create.csv": (CREATE_COLUMNS, create_rows(proposals["nouvelles"], categories, countries)),
        "plan_alternatives.csv": (ALTERNATIVE_COLUMNS, alternative_rows(proposals["alternatifs"])),
        "plan_sort.csv": (SORT_COLUMNS, sort),
        "plan_descriptions.csv": (DESCRIPTION_COLUMNS, descriptions),
        "plan_roles.csv": (ROLE_COLUMNS, roles),
        "plan_merges.csv": (MERGE_COLUMNS, merge_rows(proposals["doublons"])),
    }
    run_dir = OUTPUT_DIR / f"ner_{args.label}_{datetime.now():%Y%m%dT%H%M%S}"
    run_dir.mkdir(parents=True)
    for name, (columns, rows) in plans.items():
        write_csv(run_dir / name, columns, rows)
    (run_dir / "manifest.json").write_text(json.dumps({
        "kind": "ner_leftovers", "source": str(args.proposals.resolve()), "model": REVIEWER_MODEL_KEY,
        "created": datetime.now().isoformat(timespec="seconds"),
        "plans": {name: len(rows) for name, (_, rows) in plans.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    console.print(count_table(
        [(name, f"{sum(r['approve'] == 'yes' for r in rows)} approved / {len(rows)}")
         for name, (_, rows) in plans.items()], title="Plans"))
    console.print(f"Run: [cyan]{run_dir}[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
