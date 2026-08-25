#!/usr/bin/env python3
"""
merge_shards.py
===============

Merge the JSONL shards an offline annotation run produced, and report what
never succeeded.

``annotate_offline.py`` writes one object per *annotation*, not per article: a
failed item is retried on a later run and appends a second record, so an item id
appears once per attempt. Merging is therefore not concatenation. The rule, as
stated in that script's header, is to key on ``item_id`` and keep the last
record without an ``analysis_error``.

What this adds is the other half of that rule. Earlier failures are kept on
purpose, because the failure rate is a finding about the model rather than noise
to be cleaned up — but a finding nobody writes down is not kept, it is merely
not deleted. So this script also emits a **failure log**: every item that has no
successful record, how many times it was attempted, and which fault it hit each
time. That file is the evidence for a claim like "these 145 failed four times
for the same reason", which is otherwise buried in three 6 MB shards.

An item with no success is still written to the merged file, carrying its last
``analysis_error``. Downstream, a record with that key set is not an annotation
and must not be written to Omeka; keeping it makes the gap visible instead of
turning it into a silently absent row.

Usage
-----
    python serving/merge_shards.py --shards 'cache/qwen38_full/full-s*.jsonl' \\
        --output cache/qwen38_full/qwen38_merged.jsonl

    # Report only — read the shards, write nothing:
    python serving/merge_shards.py --shards 'work/full-s*.jsonl' --dry-run

Prompt fingerprints are checked, not assumed. Records made under two different
prompts are two different instruments, and merging them into one file would
hide that; the script refuses unless ``--allow-mixed-prompts`` says otherwise.
"""
import sys
import json
import glob
import argparse
import collections
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.console_utils import count_table, key_value_table

from rich.console import Console

console = Console()


def _error_of(record: Dict[str, Any]) -> Optional[str]:
    """The record's ``analysis_error``, or None if it succeeded."""
    return (record.get("result") or {}).get("analysis_error")


def _error_class(message: str) -> str:
    """Collapse an error message to the fault it represents.

    A Pydantic message runs to several lines and quotes the offending input, so
    the raw string is nearly unique per item and counts nothing. What identifies
    the fault is the validator's own clause — everything before the ``;`` that
    introduces the explanation, or before the ``[type=...]`` machine detail.
    """
    flat = " ".join((message or "").split())
    if not flat:
        return "unknown"
    if "Value error," in flat:
        clause = flat.split("Value error,", 1)[1]
        clause = clause.split(";")[0].split("[type=")[0].strip()
        return f"ValidationError: {clause}"
    return flat.split(":")[0] if ":" in flat else flat[:70]


def load_shards(patterns: List[str]) -> Tuple[List[Path], Dict[int, List[Dict[str, Any]]]]:
    """Read every shard into ``{item_id: [attempt, ...]}`` in append order."""
    paths: List[Path] = []
    for pattern in patterns:
        matched = sorted(Path(p) for p in glob.glob(pattern))
        if not matched:
            console.print(f"[yellow]No shard matched[/yellow] {pattern}")
        paths.extend(matched)

    attempts: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    console.print(f"[red]{path.name}:{line_no} is not JSON[/red] — {exc}")
                    continue
                item_id = record.get("item_id")
                if item_id is None:
                    console.print(f"[red]{path.name}:{line_no} has no item_id[/red]")
                    continue
                attempts[int(item_id)].append(record)
    return paths, attempts


def resolve(attempts: Dict[int, List[Dict[str, Any]]]):
    """Split items into the last success, or every attempt when none succeeded."""
    merged: Dict[int, Dict[str, Any]] = {}
    failed: Dict[int, List[Dict[str, Any]]] = {}
    for item_id, records in attempts.items():
        successes = [r for r in records if not _error_of(r)]
        if successes:
            merged[item_id] = successes[-1]
        else:
            merged[item_id] = records[-1]
            failed[item_id] = records
    return merged, failed


def build_failure_log(failed: Dict[int, List[Dict[str, Any]]], *, prompt: Optional[str]) -> Dict[str, Any]:
    """The evidence file: per item, how often it was tried and how it failed."""
    items = []
    for item_id in sorted(failed):
        records = failed[item_id]
        faults = [_error_class(_error_of(r) or "") for r in records]
        items.append(
            {
                "item_id": item_id,
                "attempts": len(records),
                "language": records[-1].get("language"),
                "fault": faults[-1],
                "same_fault_every_attempt": len(set(faults)) == 1,
                "faults": faults,
                "last_error": " ".join((_error_of(records[-1]) or "").split())[:400],
            }
        )
    by_fault = collections.Counter(entry["fault"] for entry in items)
    by_attempts = collections.Counter(entry["attempts"] for entry in items)
    return {
        "prompt_fingerprint": prompt,
        "model": next(
            (r[-1].get("model") for r in failed.values() if r), None
        ),
        "total_failed": len(items),
        "min_attempts": min((e["attempts"] for e in items), default=0),
        "faults": dict(by_fault.most_common()),
        "attempts_histogram": {str(k): v for k, v in sorted(by_attempts.items())},
        "items": items,
    }


def rounds_of(attempts: Dict[int, List[Dict[str, Any]]]) -> List[Tuple[int, int, int]]:
    """Reconstruct per-round (attempted, failed) from append order.

    Each retry re-attempts a subset of the previous round, so a round boundary
    is where an item id recurs. This is inference from file layout rather than a
    recorded fact — the records carry no timestamp — but it is what shows
    whether retrying is still recovering anything.
    """
    per_round: Dict[int, List[int]] = collections.defaultdict(lambda: [0, 0])
    for records in attempts.values():
        for index, record in enumerate(records):
            bucket = per_round[index]
            bucket[0] += 1
            if _error_of(record):
                bucket[1] += 1
    return [(rnd, vals[0], vals[1]) for rnd, vals in sorted(per_round.items())]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge offline annotation shards and log what never succeeded."
    )
    parser.add_argument(
        "--shards",
        nargs="+",
        required=True,
        help="Shard paths or globs, e.g. 'cache/qwen38_full/full-s*.jsonl'",
    )
    parser.add_argument("--output", help="Merged JSONL to write (omit with --dry-run)")
    parser.add_argument(
        "--failure-log",
        help="Where to write the failure evidence JSON "
        "(default: alongside --output as <stem>_failures.json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report only; write nothing"
    )
    parser.add_argument(
        "--allow-mixed-prompts",
        action="store_true",
        help="Merge records made under different prompt fingerprints (they are "
        "different instruments; the default is to refuse)",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.output:
        parser.error("--output is required unless --dry-run is given")

    paths, attempts = load_shards(args.shards)
    if not attempts:
        console.print("[red]No records read — nothing to merge.[/red]")
        return 1

    prompts = {
        r.get("prompt") for records in attempts.values() for r in records
    }
    if len(prompts) > 1 and not args.allow_mixed_prompts:
        console.print(
            f"[red]Shards mix {len(prompts)} prompt fingerprints:[/red] "
            f"{sorted(str(p) for p in prompts)}\n"
            "These are different instruments. Re-run with --allow-mixed-prompts "
            "only if you know why they belong in one file."
        )
        return 1

    merged, failed = resolve(attempts)
    total_attempts = sum(len(r) for r in attempts.values())
    succeeded = len(merged) - len(failed)

    console.print(
        key_value_table(
            [
                ("Shards read", str(len(paths))),
                ("Annotation attempts", f"{total_attempts:,}"),
                ("Unique items", f"{len(merged):,}"),
                ("Succeeded", f"{succeeded:,} ({succeeded / len(merged):.2%})"),
                ("Never succeeded", f"{len(failed):,}"),
                ("Prompt fingerprint", ", ".join(sorted(str(p) for p in prompts))),
            ],
            title="Merge",
        )
    )

    rounds = rounds_of(attempts)
    if len(rounds) > 1:
        console.print(
            count_table(
                [
                    (
                        f"Round {rnd}" if rnd else "First pass",
                        f"{tried:,} attempted / {bad:,} failed",
                    )
                    for rnd, tried, bad in rounds
                ],
                title="Retry convergence",
            )
        )

    log = build_failure_log(failed, prompt=next(iter(prompts), None))
    if failed:
        console.print(
            count_table(
                [(fault, str(n)) for fault, n in log["faults"].items()],
                title=f"Residual faults ({log['total_failed']} items, "
                f"min {log['min_attempts']} attempts each)",
            )
        )

    if args.dry_run:
        console.print("[yellow]--dry-run: nothing written.[/yellow]")
        return 0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item_id in sorted(merged):
            handle.write(json.dumps(merged[item_id], ensure_ascii=False) + "\n")
    console.print(f"[green]Merged →[/green] {output} ({len(merged):,} lines)")

    log_path = (
        Path(args.failure_log)
        if args.failure_log
        else output.with_name(f"{output.stem}_failures.json")
    )
    log_path.write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    console.print(f"[green]Failure log →[/green] {log_path} ({log['total_failed']:,} items)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
