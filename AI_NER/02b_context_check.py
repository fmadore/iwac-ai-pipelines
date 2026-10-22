#!/usr/bin/env python3
"""
Check each article's reconciled entities against the article itself.

Step 02 matches names to authority records without reading the article, so a
shared name is taken for the same entity: three persons called Abdoulaye
Ouédraogo, the health ministries of six countries, a village and the clashes
named after it. This step sends the model one call per article — its text
and every entry the reconciliation produced or could not settle — and lets it
choose, for each entry, among the candidate records or none:

- ``match``: the record step 02 chose by name; kept unless the text says otherwise;
- ``ambiguous``: several records share the name; step 02 left it unlinked;
- ``approximate``: the fuzzy candidates of an unreconciled term.

The model may only choose among the ids it was given. The output
``*_reconciled_checked.csv`` has the same columns as the reconciled CSV, so
``03_Omeka_update.py --input`` writes it like any other, and its provenance
still traces back to the NER checkpoint. Every decision, with the model's
reason, is logged to ``*_context_decisions.csv`` for the curator.

Decisions are appended to ``*_reconciled_checked.decisions.jsonl`` as they
arrive, keyed by a fingerprint of the article's text and entries, so an
interrupted run resumes and a changed reconciliation re-checks only what
changed.

``--limit N`` checks a seeded random sample of N articles (a pilot); the
checked CSV then holds only those articles, so a pilot can never be written
to Omeka in place of the full run by mistake.

Usage:
    python AI_NER/02b_context_check.py --input output/x_reconciled.csv --limit 100
    python AI_NER/02b_context_check.py --input output/x_reconciled.csv --workers 8
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, Field
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.checkpoint import JsonCheckpoint, fingerprint, read_checkpoint_context, sha256_text  # noqa: E402
from common.console_utils import count_table, standard_progress  # noqa: E402
from common.llm_provider import (  # noqa: E402
    TEXT_EXTENDED_MODELS,
    LLMConfig,
    build_llm_client,
    get_model_option,
    summary_from_option,
)
from common.log_redaction import configure_logging  # noqa: E402
from common.omeka_client import OmekaClient  # noqa: E402
from common.reconciliation import resolve_term  # noqa: E402
from common.reconciliation_cli import (  # noqa: E402
    SPATIAL_COLUMN,
    SPATIAL_OUTPUT_COLUMN,
    SUBJECT_COLUMN,
    SUBJECT_OUTPUT_COLUMN,
    AuthorityDicts,
    build_dictionaries,
    checkpoint_country,
)
from common.run_context import model_context  # noqa: E402

logger = logging.getLogger(__name__)
console = Console()
SCRIPT_DIR = Path(__file__).resolve().parent
PROMPT_PATH = SCRIPT_DIR / "context_check_prompt.md"
MAX_TEXT_CHARS = 24_000
MAX_APPROXIMATE = 3
#: Item set → record type shown to the model.
SET_KIND = {266: "person", 854: "organisation", 268: "place", 1: "subject", 2: "event", 267: "notice"}
csv.field_size_limit(10**9)


class EntryDecision(BaseModel):
    n: int = Field(description="The entry number")
    choice: Optional[int] = Field(description="Id of the chosen candidate, or null for no link")
    reason: str = Field(description="At most 15 words: what in the article decides it")


class ArticleDecisions(BaseModel):
    decisions: List[EntryDecision]


@dataclass
class Entry:
    n: int
    term: str
    where: str            # "subject" or "place": the column a choice is written to
    status: str           # "match", "ambiguous" or "approximate"
    candidates: List[int] = field(default_factory=list)


def article_entries(row: Dict[str, str], dicts: AuthorityDicts,
                    approximate: Dict[str, List[int]]) -> List[Entry]:
    """The entries of one article, exactly as step 02 resolved its terms.

    A subject term step 02 sent to the place column (a university, a mosque)
    is a ``place`` entry here too.
    """
    entries: List[Entry] = []

    def add(term: str, where: str, status: str, ids: Iterable[Any]) -> None:
        entries.append(Entry(len(entries) + 1, term, where, status, [int(i) for i in ids]))

    for column, where in ((SPATIAL_COLUMN, "place"), (SUBJECT_COLUMN, "subject")):
        for term in dict.fromkeys(t.strip() for t in (row.get(column) or "").split("|") if t.strip()):
            if where == "place":
                found, ambiguous = resolve_term(term, dicts.spatial, dicts.spatial_ambiguous)
            else:
                found, ambiguous = resolve_term(term, dicts.subject, dicts.subject_ambiguous)
                if found is None and not ambiguous:
                    place, place_ambiguous = resolve_term(term, dicts.spatial, dicts.spatial_ambiguous)
                    if place is not None and not place_ambiguous:
                        add(term, "place", "match", [place])
                        continue
            if found is not None:
                add(term, where, "match", [found])
            elif ambiguous:
                add(term, where, "ambiguous", ambiguous)
            elif approximate.get(term.lower()):
                add(term, where, "approximate", approximate[term.lower()][:MAX_APPROXIMATE])
    return entries


def reconciled_ids(row: Dict[str, str]) -> Dict[str, List[int]]:
    return {where: [int(i) for i in (row.get(col) or "").split("|") if i.strip()]
            for where, col in (("place", SPATIAL_OUTPUT_COLUMN), ("subject", SUBJECT_OUTPUT_COLUMN))}


def apply_decisions(entries: Sequence[Entry], decisions: Sequence[EntryDecision]) -> List[Dict[str, Any]]:
    """One outcome per entry. A missing decision keeps a match and links
    nothing otherwise; a choice outside the entry's candidates is refused the
    same way, so the model can never introduce an id it was not offered."""
    by_n = {d.n: d for d in decisions}
    outcomes = []
    for entry in entries:
        decision = by_n.get(entry.n)
        default = entry.candidates[0] if entry.status == "match" else None
        if decision is None:
            choice, note = default, "no decision returned"
        elif decision.choice is not None and decision.choice not in entry.candidates:
            choice, note = default, f"refused id {decision.choice} (not a candidate)"
        else:
            choice, note = decision.choice, decision.reason
        proposed = entry.candidates[0] if entry.status == "match" else None
        change = ("keep" if choice == proposed and choice is not None
                  else "drop" if proposed is not None and choice is None
                  else "replace" if proposed is not None
                  else "add" if choice is not None else "none")
        outcomes.append({"n": entry.n, "term": entry.term, "field": entry.where, "status": entry.status,
                         "proposed": proposed, "choice": choice, "change": change, "reason": note})
    return outcomes


def checked_ids(outcomes: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
    result: Dict[str, List[int]] = {"place": [], "subject": []}
    for outcome in outcomes:
        if outcome["choice"] is not None and outcome["choice"] not in result[outcome["field"]]:
            result[outcome["field"]].append(outcome["choice"])
    return result


def record_card(item: Dict[str, Any]) -> Dict[str, Any]:
    sets = [s.get("o:id") for s in item.get("o:item_set", [])]
    kind = next((SET_KIND[s] for s in sets if s in SET_KIND), "unsorted")
    country = [v.get("display_title") for v in item.get("dcterms:spatial", []) if v.get("display_title")]
    if kind == "place":
        country = [v.get("display_title") for v in item.get("dcterms:isPartOf", []) if v.get("display_title")]
    descriptions = item.get("dcterms:description", [])
    description = next((v.get("@value") for v in descriptions if v.get("@language") == "fr"),
                       next((v.get("@value") for v in descriptions), ""))
    return {"title": item.get("o:title"), "type": kind, "country": ", ".join(country[:2]),
            "description": (description or "")[:300],
            "alternatives": [v.get("@value") for v in item.get("dcterms:alternative", [])][:5]}


def user_prompt(row: Dict[str, str], country: Optional[str], entries: Sequence[Entry],
                cards: Dict[int, Dict[str, Any]]) -> str:
    text = (row.get("bibo:content") or "")[:MAX_TEXT_CHARS]
    payload = {
        "article": {"id": row["o:id"], "title": row.get("Title", ""), "newspaper_country": country or "unknown",
                    "text": text},
        "entries": [{"n": e.n, "term": e.term, "field": e.where, "status": e.status,
                     "candidates": [{"id": i, **cards.get(i, {"title": "(record not found)"})}
                                    for i in e.candidates]} for e in entries],
    }
    return json.dumps(payload, ensure_ascii=False)


def load_approximate(base: Path) -> Dict[str, List[int]]:
    """Fuzzy candidates per unreconciled term, best first, from step 02's files."""
    found: Dict[str, List[int]] = {}
    for tag in ("spatial", "subject_and_topic"):
        path = Path(f"{base}_potential_reconciliation_{tag}.csv")
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                ids = found.setdefault(r["Unreconciled Value"].lower(), [])
                if r["Item ID"].isdigit() and int(r["Item ID"]) not in ids:
                    ids.append(int(r["Item ID"]))
    return found


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="*_reconciled.csv from step 02")
    parser.add_argument("--model", choices=TEXT_EXTENDED_MODELS,
                        help="Checker model (default: the model of the NER run)")
    parser.add_argument("--limit", type=int, help="Check a seeded random sample of N articles (pilot)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Discard decisions made under another model/prompt")
    args = parser.parse_args(argv)

    input_path: Path = args.input
    if not input_path.stem.endswith("_reconciled"):
        console.print("[red]--input must be a *_reconciled.csv from step 02[/]")
        return 1
    base = input_path.with_name(input_path.stem[: -len("_reconciled")])
    source = base.with_suffix(input_path.suffix)
    ner_context = read_checkpoint_context(source)
    model_key = args.model or ner_context.get("model_key")
    if not model_key:
        console.print("[red]No NER checkpoint names a model: pass --model[/]")
        return 1
    option = get_model_option(model_key, allowed_keys=TEXT_EXTENDED_MODELS)
    config = LLMConfig(reasoning_effort="medium", text_verbosity="low", thinking_level="minimal")
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    country = checkpoint_country(source)

    with input_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames, rows = list(reader.fieldnames or []), list(reader)
    if args.limit:
        rows = sorted(random.Random(args.seed).sample(rows, min(args.limit, len(rows))), key=lambda r: int(r["o:id"]))

    client = OmekaClient.from_env()
    dicts = build_dictionaries(client, extended_matching=True, country=country)
    approximate = load_approximate(base)
    work = []
    mismatched = 0
    for row in rows:
        entries = article_entries(row, dicts, approximate)
        matched = {"place": [], "subject": []}
        for e in entries:
            if e.status == "match":
                matched[e.where].append(e.candidates[0])
        mismatched += any(set(matched[w]) != set(reconciled_ids(row)[w]) for w in matched)
        work.append((row, entries))
    if mismatched:
        console.print(f"[yellow]{mismatched} article(s) whose matches differ from the CSV: the authorities "
                      "changed since step 02. Re-run step 02 first.[/]")
        return 1

    ids = sorted({i for _, entries in work for e in entries for i in e.candidates})
    cards = {i: record_card(item) for i, item in client.get_items_by_ids(ids).items()}

    out_csv = input_path.with_name(f"{input_path.stem}_checked.csv")
    journal_path = out_csv.with_suffix(".decisions.jsonl")
    run = model_context(option, config)
    JsonCheckpoint.open(out_csv.with_suffix(".checkpoint.json"),
                        {"pipeline": "ner-context-check-v1", **run, "prompt_sha256": sha256_text(prompt)},
                        reset=args.force)
    if args.force and journal_path.exists():
        journal_path.unlink()
    done: Dict[str, Dict[str, Any]] = {}
    if journal_path.exists():
        for line in journal_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done[str(record["o:id"])] = record

    llm = build_llm_client(option, config=config)
    lock = threading.Lock()
    results: Dict[str, Dict[str, Any]] = {}
    pending = []
    for row, entries in work:
        key = fingerprint({"text": sha256_text(row.get("bibo:content") or ""),
                           "entries": [e.__dict__ for e in entries]})
        record = done.get(row["o:id"])
        if not entries:
            results[row["o:id"]] = {"o:id": row["o:id"], "fingerprint": key, "outcomes": []}
        elif record and record["fingerprint"] == key:
            results[row["o:id"]] = record
        else:
            pending.append((row, entries, key))

    console.print(f"Checker: {summary_from_option(option)} · country {country or 'none'} · "
                  f"{len(work)} articles, {len(pending)} to check, {len(work) - len(pending)} already decided")

    def check(row, entries, key):
        decisions = llm.generate_structured(prompt, user_prompt(row, country, entries, cards), ArticleDecisions)
        record = {"o:id": row["o:id"], "fingerprint": key,
                  "outcomes": apply_decisions(entries, decisions.decisions)}
        with lock:
            with journal_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    failed = 0
    with standard_progress(console) as progress, ThreadPoolExecutor(max_workers=args.workers) as pool:
        task = progress.add_task("Checking articles", total=len(pending))
        futures = {pool.submit(check, *job): job[0]["o:id"] for job in pending}
        for future in as_completed(futures):
            try:
                record = future.result()
                results[record["o:id"]] = record
            except Exception as exc:  # one article's failure must not stop the run
                failed += 1
                logger.error("Article %s failed: %s", futures[future], exc)
            progress.advance(task)

    decisions_path = base.with_name(f"{base.name}_context_decisions.csv")
    tally: Dict[str, int] = {}
    with out_csv.open("w", encoding="utf-8", newline="") as out, \
            decisions_path.open("w", encoding="utf-8", newline="") as log:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        log_writer = csv.DictWriter(log, fieldnames=["o:id", "title", "n", "term", "field", "status", "proposed",
                                                     "proposed_title", "choice", "choice_title", "change", "reason"])
        log_writer.writeheader()
        for row, _ in work:
            record = results.get(row["o:id"])
            if record is None:
                continue  # failed: left out, so step 03 cannot write unchecked links
            ids = checked_ids(record["outcomes"])
            writer.writerow({**row, SPATIAL_OUTPUT_COLUMN: "|".join(map(str, ids["place"])),
                             SUBJECT_OUTPUT_COLUMN: "|".join(map(str, ids["subject"]))})
            for o in record["outcomes"]:
                tally[f"{o['status']}: {o['change']}"] = tally.get(f"{o['status']}: {o['change']}", 0) + 1
                log_writer.writerow({"o:id": row["o:id"], "title": row.get("Title", "")[:120], **o,
                                     "proposed_title": cards.get(o["proposed"], {}).get("title", "") if o["proposed"] else "",
                                     "choice_title": cards.get(o["choice"], {}).get("title", "") if o["choice"] else ""})
    console.print(count_table(sorted(tally.items()), title="Decisions"))
    if llm.usage.requests:
        console.print(f"Usage: {llm.usage.summary()}")
    console.print(f"Checked CSV: [cyan]{out_csv}[/]\nDecisions:   [cyan]{decisions_path}[/]")
    if failed:
        console.print(f"[red]{failed} article(s) failed and were left out; re-run to retry them.[/]")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
