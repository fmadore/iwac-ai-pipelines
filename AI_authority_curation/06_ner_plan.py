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
  duplicates_to_examine.csv  the reviewers' remarks on possible duplicates

No merge plan is written. A reviewer judging one term sometimes remarked
that other records looked alike; on 2026-10-21 turning those remarks into
pairs merged an event into its place ("Affrontements intercommunautaires
Yirgou (2019)" / the village), an arrest into the person arrested, the FAIB
into the AIMB and Nas-Air into Air Afrique. Duplicates are the curator's to
find and confirm; ``04_merge.py`` applies a ``plan_merges.csv`` the curator
writes.

It writes nothing to Omeka. It reads the person, organisation and place sets
once to turn the reviewers' category and country labels into the items that
already stand for them; a label with no item is kept as text and linked to
nothing. Rows are pre-approved unless something needs the curator: a record
that may already exist, low confidence, a stub filed as a mere theme.

Apply in this order: 05_sort.py, 03_apply.py, 07_create.py, then reconcile
again (AI_NER/02): its ambiguous-terms files show the duplicates worth
examining. Merge only those the curator has confirmed.

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
EXAMINE_COLUMNS = ["record_a", "title_a", "kind_a", "url_a", "record_b", "title_b", "kind_b", "url_b",
                   "reviewer_note"]


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


def examine_rows(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The reviewers' duplicate remarks as they were made, for the curator to
    examine. Not a merge plan: no survivor, no approval column."""
    return [{
        "record_a": p["id A"], "title_a": p["titre A"], "kind_a": p["type A"], "url_a": SITE + str(p["id A"]),
        "record_b": p["id B"], "title_b": p["titre B"], "kind_b": p["type B"], "url_b": SITE + str(p["id B"]),
        "reviewer_note": p.get("note") or "",
    } for p in pairs]


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
        "duplicates_to_examine.csv": (EXAMINE_COLUMNS, examine_rows(proposals["doublons"])),
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
        [(name, f"{sum(r.get('approve') == 'yes' for r in rows)} approved / {len(rows)}")
         for name, (_, rows) in plans.items()], title="Plans"))
    console.print(f"Run: [cyan]{run_dir}[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
