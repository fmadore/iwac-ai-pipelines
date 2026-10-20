#!/usr/bin/env python3
"""
Prepare authority records for review: what needs work, and what the archive says.

Read-only. For one authority kind it finds the records with no
``dcterms:description`` (and, for persons, no role category in
``dcterms:isPartOf``), gathers quotations from the documents that link to each
one, lists pairs of records that look like one entity spelled two ways, and
writes the batches the Haiku reviewers of the ``iwac-authority-curation``
skill read:

  output/<kind>_<timestamp>/
    manifest.json             what was collected, from which records
    batches/describe_NNN.json records to describe (and, for persons, to type)
    batches/duplicates_NNN.json candidate duplicate pairs
    no_evidence.csv           records no document links to: nothing to describe from
    verdicts/                 where each reviewer writes its batch's verdict

Quotations come from ``bibo:content`` of the authenticated item, so they include
text that is not public on the site; the output folder is gitignored.

``--kind unsorted`` works on "Notices d'autorités à traiter" (set 12826)
instead: ``batches/triage_NNN.json`` carry each stub with its evidence and the
existing authorities whose names are closest, for a reviewer to file it, clean
its title and describe it. Stubs are taken most-linked first, from a cached
count (``output/unsorted_link_counts.json``, ``--recount`` to refresh); each run
skips stubs an earlier run already batched, so repeating the command continues
the pass. Stubs already in an authority set are only listed (``already_filed``),
and stubs nothing links to go to ``unlinked_stubs.csv``.

Usage:
    python AI_authority_curation/01_collect.py --kind organisations --limit 40
    python AI_authority_curation/01_collect.py --kind persons --task describe --limit 30
    python AI_authority_curation/01_collect.py --kind organisations --task duplicates
    python AI_authority_curation/01_collect.py --kind persons --ids 125261,43062
    python AI_authority_curation/01_collect.py --kind unsorted --limit 60
    python AI_authority_curation/01_collect.py --kind unsorted --limit 60 --recount
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.console_utils import key_value_table, standard_progress  # noqa: E402
from common.iwac_config import AUTHORITY_ITEM_SET_ID, item_page_url  # noqa: E402
from common.log_redaction import configure_logging  # noqa: E402
from common.omeka_client import OmekaClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curation_core import (  # noqa: E402
    KINDS,
    SORT_TARGETS,
    NameIndex,
    duplicate_candidates,
    find_snippets,
    name_variants,
    record_names,
)
from wikidata import WikidataSearch, record_qids  # noqa: E402

#: Wikidata search answers, kept between runs.
WIKIDATA_CACHE = "wikidata_cache.json"

logger = logging.getLogger(__name__)
console = Console()

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
#: Four at a time: twelve concurrent requests tripped TLS handshake timeouts on
#: this server (omeka-curation notes, 2026-10-11).
MAX_WORKERS = 4
SUMMARY_CHARS = 600


def _values(record: Dict[str, Any], term: str) -> List[Dict[str, Any]]:
    return [v for v in record.get(term, []) or [] if isinstance(v, dict)]


def _text(value: Dict[str, Any]) -> str:
    return str(value.get("@value") or value.get("display_title") or value.get("o:label")
               or value.get("@id") or "")


def summarize_record(record: Dict[str, Any], base_url: str, counts: Dict[str, int]) -> Dict[str, Any]:
    """The record as the reviewer sees it: names, what it already says, how linked."""
    names = record_names(record)
    out: Dict[str, Any] = {
        "id": int(record["o:id"]),
        "title": names[0] if names else "",
        "alternatives": names[1:],
        "url": item_page_url(base_url, int(record["o:id"])),
        "linked_documents": counts,
    }
    for term in ("dcterms:description", "dcterms:isPartOf", "dcterms:spatial", "dcterms:relation",
                 "foaf:birthday", "dcterms:date", "dcterms:identifier"):
        values = [_text(v) for v in _values(record, term)]
        if values:
            out[term] = values
    return out


def linked_filter(property_id: int, authority_id: int) -> Dict[str, Any]:
    return {"property[0][property]": property_id, "property[0][type]": "res",
            "property[0][text]": authority_id}


def count_links(client: OmekaClient, authority_id: int, prop_ids: Dict[str, int]) -> Dict[str, int]:
    return {term: client.count_items(**linked_filter(pid, authority_id)) for term, pid in prop_ids.items()}


def gather_evidence(
    client: OmekaClient,
    record: Dict[str, Any],
    prop_ids: Dict[str, int],
    counts: Dict[str, int],
    max_docs: int,
) -> List[Dict[str, Any]]:
    """Up to *max_docs* documents linking to *record*, with quotations naming it.

    Half come newest-first and half oldest-first, so a record linked across
    fifty years is not described from its last five. A document whose text never
    names the record (OCR damage, a link made from the summary) contributes its
    AI summary instead, marked as such.
    """
    names = name_variants(record)
    seen: Dict[int, Dict[str, Any]] = {}
    for term, pid in prop_ids.items():
        if not counts.get(term):
            continue
        for order in ("desc", "asc"):
            if len(seen) >= max_docs:
                break
            page = client.list_page(1, per_page=max(1, max_docs // 2), sort_by="dcterms:date",
                                    sort_order=order, **linked_filter(pid, int(record["o:id"])))
            for doc in page:
                if len(seen) >= max_docs:
                    break
                did = int(doc["o:id"])
                if did in seen:
                    continue
                content = " ".join(_text(v) for v in _values(doc, "bibo:content"))
                snippets = find_snippets(content, names)
                entry: Dict[str, Any] = {
                    "item_id": did,
                    "title": doc.get("o:title") or "",
                    "date": next((_text(v) for v in _values(doc, "dcterms:date")), ""),
                    "publisher": next((_text(v) for v in _values(doc, "dcterms:publisher")), ""),
                    "linked_as": term,
                }
                if snippets:
                    entry["quotes"] = snippets
                else:
                    summary = next((_text(v) for v in _values(doc, "bibo:shortDescription")
                                    if v.get("@language") in (None, "fr")), "")
                    summary = summary or next((_text(v) for v in _values(doc, "dcterms:abstract")), "")
                    if summary:
                        entry["summary_not_a_quote"] = summary[:SUMMARY_CHARS]
                seen[did] = entry
    return list(seen.values())


def role_categories(client: OmekaClient, persons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The role categories persons are filed under, read live, most used first.

    ``dcterms:isPartOf`` on a person holds roles (Imam, Journaliste …) and
    affiliations (a newspaper, an association) alike; the roles are the targets
    that are "Notice d'autorité" items in set 267.
    """
    usage: Counter = Counter()
    for person in persons:
        for value in _values(person, "dcterms:isPartOf"):
            if value.get("value_resource_id"):
                usage[int(value["value_resource_id"])] += 1
    targets = client.get_items_by_ids(usage)
    roles = [
        {"id": tid, "title": item.get("o:title"), "people": usage[tid]}
        for tid, item in targets.items()
        if AUTHORITY_ITEM_SET_ID in {s.get("o:id") for s in item.get("o:item_set", [])}
    ]
    return sorted(roles, key=lambda r: (-r["people"], r["title"] or ""))


def write_batches(run_dir: Path, prefix: str, entries: List[Any], size: int,
                  extra: Dict[str, Any], key: str) -> List[str]:
    names = []
    for start in range(0, len(entries), size):
        name = f"{prefix}_{start // size + 1:03d}"
        payload = {"batch": name, **extra, key: entries[start:start + size]}
        (run_dir / "batches" / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        names.append(name)
    return names


#: Linked-document counts for every stub of the unsorted set, kept between runs:
#: ranking 3,000 stubs costs a request per stub per property, about a quarter
#: of an hour, and the order barely moves from one run to the next.
COUNTS_FILE = "unsorted_link_counts.json"


def stub_link_counts(client: OmekaClient, stubs: List[Dict[str, Any]], prop_ids: Dict[str, int],
                     cache: Path, recount: bool) -> Dict[int, Dict[str, int]]:
    """Linked documents per stub and property, from the cache where it has them.

    Only stubs the cache lacks are counted, so a set that grows costs only its
    new stubs. ``--recount`` starts over, for when links have moved (after
    merges, or a NER upload).
    """
    counts: Dict[int, Dict[str, int]] = {}
    if cache.exists() and not recount:
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if stored.get("terms") == list(prop_ids):
            counts = {int(k): v for k, v in stored["counts"].items()}
    missing = [r for r in stubs if int(r["o:id"]) not in counts]
    if missing:
        with standard_progress(console) as progress, ThreadPoolExecutor(MAX_WORKERS) as executor:
            task = progress.add_task(f"[cyan]Counting linked documents for {len(missing)} stubs…",
                                     total=len(missing))

            def counted(record):
                result = count_links(client, int(record["o:id"]), prop_ids)
                progress.advance(task)
                return int(record["o:id"]), result

            counts.update(executor.map(counted, missing))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"counted": datetime.now().isoformat(timespec="seconds"),
                                     "terms": list(prop_ids), "counts": counts}), encoding="utf-8")
    return {int(r["o:id"]): counts[int(r["o:id"])] for r in stubs}


def counts_stamp(cache: Path) -> str:
    try:
        return json.loads(cache.read_text(encoding="utf-8")).get("counted", "")
    except (OSError, ValueError):
        return ""


def previously_batched(output_dir: Path, current: Path) -> set:
    """Ids of stubs any earlier unsorted run already sent to a reviewer."""
    seen = set()
    for batch in output_dir.glob("unsorted_*/batches/triage_*.json"):
        if batch.parents[1] == current:
            continue
        for entry in json.loads(batch.read_text(encoding="utf-8")).get("authorities", []):
            seen.add(int(entry["authority"]["id"]))
    return seen


def used_qids(records: Iterable[Dict[str, Any]]) -> Dict[str, Tuple[int, str]]:
    """Wikidata items already on an authority: ``{qid: (id, title)}``."""
    used: Dict[str, Tuple[int, str]] = {}
    for record in records:
        for qid in record_qids(record):
            used.setdefault(qid, (int(record["o:id"]), record.get("o:title") or ""))
    return used


def attach_wikidata(entry: Dict[str, Any], record: Dict[str, Any], search: Optional[WikidataSearch],
                    used: Dict[str, Tuple[int, str]]) -> None:
    """Offer Wikidata candidates for a record that has no Wikidata item yet.

    A candidate already on another authority is marked ``used_by``: it is a
    duplicate signal, never a second copy of the identifier.
    """
    if search is None or record_qids(record):
        return
    names = name_variants(record)
    words = [n for n in names if not (" " not in n.strip() and n.strip().isupper())]  # bare acronyms add noise
    candidates = search.candidates((words or names)[:3])
    for candidate in candidates:
        owner = used.get(candidate["qid"])
        if owner and owner[0] != int(record["o:id"]):
            candidate["used_by"] = {"id": owner[0], "title": owner[1]}
    entry["wikidata_candidates"] = candidates
    entry["needs"] = [*entry["needs"], "wikidata"]


def collect_triage(args, client: OmekaClient, records: List[Dict[str, Any]],
                   prop_ids: Dict[str, int], run_dir: Path, manifest: Dict[str, Any]) -> None:
    """Batches for the unsorted set: where each stub goes, and what it may duplicate.

    Stubs that already sit in one of the five authority sets are listed in the
    manifest as ``already_filed``: they need only leaving the unsorted set, and
    no reviewer. The rest are taken most-linked first, so the stubs the archive
    uses most become real authorities first. Stubs an earlier run already sent
    to a reviewer are skipped (``--revisit`` to include them), and stubs no
    document links to are listed in ``unlinked_stubs.csv`` instead of batched
    (``--include-unlinked``): they are candidates for deletion as much as for
    filing, and that is the curator's call.
    """
    target_sets = {KINDS[k].item_set_id: k for k in SORT_TARGETS}
    already = [r for r in records if {s["o:id"] for s in r.get("o:item_set", [])} & set(target_sets)]
    manifest["already_filed"] = [
        {"id": int(r["o:id"]), "title": r.get("o:title"),
         "target": next(target_sets[s["o:id"]] for s in r["o:item_set"] if s["o:id"] in target_sets)}
        for r in already]
    filed_ids = {a["id"] for a in manifest["already_filed"]}

    console.print("[cyan]Fetching the five authority sets to compare names…[/]")
    authorities: Dict[int, Dict[str, Any]] = {}
    set_of: Dict[int, str] = {}
    for key in SORT_TARGETS:
        for record in client.get_items(KINDS[key].item_set_id):
            authorities[int(record["o:id"])] = record
            set_of[int(record["o:id"])] = key
    index = NameIndex(authorities.values())
    roles = role_categories(client, [r for r in authorities.values() if set_of[int(r["o:id"])] == "persons"])
    manifest["role_options"] = roles
    search = None if args.no_wikidata else WikidataSearch(args.output_dir / WIKIDATA_CACHE)
    qids = used_qids(authorities.values())

    wanted = [int(x) for x in args.ids.split(",")] if args.ids else None
    stubs = [r for r in records if int(r["o:id"]) not in filed_ids]
    stub_counts = stub_link_counts(client, stubs, prop_ids, args.output_dir / COUNTS_FILE, args.recount)
    manifest["counts_from"] = counts_stamp(args.output_dir / COUNTS_FILE)
    reviewed = set() if args.revisit or wanted else previously_batched(args.output_dir, run_dir)
    total = {rid: sum(c.values()) for rid, c in stub_counts.items()}

    pool = [r for r in stubs if (wanted is None or int(r["o:id"]) in wanted)
            and int(r["o:id"]) not in reviewed]
    unlinked = [r for r in pool if not total[int(r["o:id"])]]
    if not (args.include_unlinked or wanted):
        pool = [r for r in pool if total[int(r["o:id"])]]
    pool.sort(key=lambda r: (-total[int(r["o:id"])], int(r["o:id"])))
    if args.limit:
        pool = pool[:args.limit]
    with (run_dir / "unlinked_stubs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["o:id", "title"])
        writer.writerows([r["o:id"], r.get("o:title")] for r in unlinked)
    manifest.update({"stubs": len(stubs), "previously_reviewed": len(reviewed & set(total)),
                     "unlinked": len(unlinked), "ranked_by": "linked documents"})

    with standard_progress(console) as progress, ThreadPoolExecutor(MAX_WORKERS) as executor:
        task = progress.add_task("[cyan]Reading the documents…", total=len(pool))

        def entry_for(record):
            rid = int(record["o:id"])
            counts = stub_counts[rid]
            candidates = []
            for other, score, reason in index.candidates(record):
                summary = summarize_record(authorities[other], client.base_url, {})
                candidates.append({**summary, "set": set_of[other], "score": score, "reason": reason})
            entry = {
                "authority": summarize_record(record, client.base_url, counts),
                "needs": ["sort", "description", "role"],
                "candidates": candidates,
                "evidence": gather_evidence(client, record, prop_ids, counts, args.docs),
            }
            attach_wikidata(entry, record, search, qids)
            progress.advance(task)
            return entry

        entries = list(executor.map(entry_for, pool))
    if search is not None:
        search.save()
    extra = {"task": "triage", "kind": "unsorted", "targets": list(SORT_TARGETS),
             "role_options": [{"id": r["id"], "title": r["title"]} for r in roles]}
    manifest["batches"] += write_batches(run_dir, "triage", entries, args.batch_size, extra, "authorities")
    manifest["triaged"] = len(entries)
    manifest["links_range"] = [total[int(pool[0]["o:id"])], total[int(pool[-1]["o:id"])]] if pool else []


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", required=True, choices=sorted(KINDS))
    parser.add_argument("--task", nargs="+", choices=("describe", "duplicates", "wikidata"),
                        default=["describe", "duplicates"],
                        help="wikidata: only reconcile records that have no Wikidata item yet")
    parser.add_argument("--no-wikidata", action="store_true",
                        help="Do not offer Wikidata candidates to the reviewers")
    parser.add_argument("--ids", help="Comma-separated authority ids instead of the whole set")
    parser.add_argument("--limit", type=int, help="At most this many records, most-linked first")
    parser.add_argument("--recount", action="store_true",
                        help="--kind unsorted: count every stub's links again instead of using the cache")
    parser.add_argument("--revisit", action="store_true",
                        help="--kind unsorted: include stubs an earlier run already sent to a reviewer")
    parser.add_argument("--include-unlinked", action="store_true",
                        help="--kind unsorted: also batch stubs no document links to")
    parser.add_argument("--max-pairs", type=int, default=200, help="At most this many duplicate pairs")
    parser.add_argument("--docs", type=int, default=8, help="Documents read per record (default 8)")
    parser.add_argument("--batch-size", type=int, default=6, help="Records per describe batch (default 6)")
    parser.add_argument("--pair-batch-size", type=int, default=10, help="Pairs per duplicates batch")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    kind = KINDS[args.kind]
    client = OmekaClient.from_env()
    prop_ids = {term: client.get_property_id(term) for term in kind.evidence_terms}
    if None in prop_ids.values():
        console.print(f"[red]Could not resolve {prop_ids}[/]")
        return 1

    console.print(f"[cyan]Fetching item set {kind.item_set_id} ({kind.key})…[/]")
    records = client.get_items(kind.item_set_id)
    by_id = {int(r["o:id"]): r for r in records}

    if kind.key == "unsorted":
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        run_dir = args.output_dir / f"unsorted_{stamp}"
        (run_dir / "batches").mkdir(parents=True)
        (run_dir / "verdicts").mkdir()
        manifest: Dict[str, Any] = {"kind": "unsorted", "item_set_id": kind.item_set_id, "created": stamp,
                                    "records_in_set": len(records), "tasks": ["triage"],
                                    "evidence_terms": list(prop_ids), "batches": []}
        collect_triage(args, client, records, prop_ids, run_dir, manifest)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
        console.print(key_value_table([
            ("Run folder", str(run_dir)),
            ("Records in set", len(records)),
            ("Already filed elsewhere", len(manifest["already_filed"])),
            ("Stubs linked from no document", manifest["unlinked"]),
            ("Skipped, reviewed in an earlier run", manifest["previously_reviewed"] or None),
            ("To triage (most-linked first)", manifest["triaged"]),
            ("Linked documents, first to last", " → ".join(map(str, manifest["links_range"])) or None),
            ("Link counts from", manifest["counts_from"]),
            ("Batches", len(manifest["batches"])),
        ], title="Collected"))
        return 0
    wanted = [int(x) for x in args.ids.split(",")] if args.ids else None
    if wanted:
        missing = [i for i in wanted if i not in by_id]
        if missing:
            console.print(f"[red]Not in item set {kind.item_set_id}: {missing}[/]")
            return 1

    roles = role_categories(client, records) if kind.key == "persons" else []
    role_ids = {r["id"] for r in roles}

    def needs(record: Dict[str, Any]) -> List[str]:
        out = [] if _values(record, "dcterms:description") else ["description"]
        if kind.key == "persons" and not any(
                v.get("value_resource_id") in role_ids for v in _values(record, "dcterms:isPartOf")):
            out.append("role")
        return out

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = args.output_dir / f"{kind.key}_{stamp}"
    (run_dir / "batches").mkdir(parents=True)
    (run_dir / "verdicts").mkdir()
    manifest: Dict[str, Any] = {
        "kind": kind.key, "item_set_id": kind.item_set_id, "created": stamp,
        "records_in_set": len(records), "tasks": args.task, "evidence_terms": list(prop_ids),
        "role_options": roles, "batches": [],
    }

    search = None if args.no_wikidata else WikidataSearch(args.output_dir / WIKIDATA_CACHE)
    qids = used_qids(records)

    for task_name in [t for t in ("describe", "wikidata") if t in args.task]:
        if task_name == "describe":
            pool = [by_id[i] for i in wanted] if wanted else [r for r in records if needs(r)]
        else:  # reconcile records that have no Wikidata item, described or not
            pool = [by_id[i] for i in wanted] if wanted else [r for r in records if not record_qids(r)]
        with standard_progress(console) as progress, ThreadPoolExecutor(MAX_WORKERS) as executor:
            task = progress.add_task("[cyan]Counting linked documents…", total=len(pool))

            def counted(record, task=task):
                result = count_links(client, int(record["o:id"]), prop_ids)
                progress.advance(task)
                return result

            counts = dict(zip((int(r["o:id"]) for r in pool), executor.map(counted, pool), strict=True))
        linked = [r for r in pool if sum(counts[int(r["o:id"])].values())]
        unlinked = [r for r in pool if not sum(counts[int(r["o:id"])].values())]
        linked.sort(key=lambda r: -sum(counts[int(r["o:id"])].values()))
        if args.limit:
            linked = linked[:args.limit]

        if task_name == "describe":
            with (run_dir / "no_evidence.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["o:id", "title", "needs"])
                for record in unlinked:
                    writer.writerow([record["o:id"], record.get("o:title"), " ".join(needs(record))])

        entries = []
        with standard_progress(console) as progress, ThreadPoolExecutor(MAX_WORKERS) as executor:
            task = progress.add_task("[cyan]Reading the documents…", total=len(linked))

            def entry_for(record, task_name=task_name, task=task, counts=counts):
                rid = int(record["o:id"])
                entry = {
                    "authority": summarize_record(record, client.base_url, counts[rid]),
                    "needs": (needs(record) or ["description"]) if task_name == "describe" else [],
                    "evidence": gather_evidence(client, record, prop_ids, counts[rid],
                                                args.docs if task_name == "describe" else min(args.docs, 4)),
                }
                attach_wikidata(entry, record, search, qids)
                progress.advance(task)
                return entry

            entries = [e for e in executor.map(entry_for, linked) if e["needs"]]
        if search is not None:
            search.save()
        extra = {"task": "describe", "kind": kind.key}
        if roles:
            extra["role_options"] = [{"id": r["id"], "title": r["title"]} for r in roles]
        manifest["batches"] += write_batches(run_dir, task_name, entries,
                                             args.batch_size if task_name == "describe" else 10,
                                             extra, "authorities")
        manifest["described" if task_name == "describe" else "reconciled"] = len(entries)
        if task_name == "describe":
            manifest["no_evidence"] = len(unlinked)

    if "duplicates" in args.task:
        pairs = duplicate_candidates(records, strip_honorifics=kind.key == "persons")
        if wanted:
            pairs = [p for p in pairs if p.a in wanted or p.b in wanted]
        pairs = pairs[:args.max_pairs]
        involved = {i for p in pairs for i in (p.a, p.b)}
        with standard_progress(console) as progress, ThreadPoolExecutor(MAX_WORKERS) as executor:
            task = progress.add_task("[cyan]Reading evidence for candidate pairs…", total=len(involved))

            def side(rid):
                record = by_id[rid]
                counts = count_links(client, rid, prop_ids)
                result = {**summarize_record(record, client.base_url, counts),
                          "evidence": gather_evidence(client, record, prop_ids, counts, 3)}
                progress.advance(task)
                return rid, result

            sides = dict(executor.map(side, sorted(involved)))
        pair_entries = [{"a": sides[p.a], "b": sides[p.b], "score": p.score, "candidate_reason": p.reason}
                        for p in pairs]
        manifest["batches"] += write_batches(run_dir, "duplicates", pair_entries, args.pair_batch_size,
                                             {"task": "duplicates", "kind": kind.key}, "pairs")
        manifest["pairs"] = len(pair_entries)

    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    console.print(key_value_table([
        ("Run folder", str(run_dir)),
        ("Records in set", len(records)),
        ("To describe", manifest.get("described")),
        ("To reconcile with Wikidata", manifest.get("reconciled")),
        ("No linked documents", manifest.get("no_evidence")),
        ("Candidate pairs", manifest.get("pairs")),
        ("Role categories offered", len(roles) or None),
        ("Batches", len(manifest["batches"])),
    ], title="Collected"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
