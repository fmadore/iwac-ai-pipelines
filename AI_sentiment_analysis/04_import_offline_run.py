#!/usr/bin/env python3
"""
04_import_offline_run.py
========================

Seed the run cache from an offline annotation run, so ``01`` writes those
answers to Omeka instead of asking for them again.

A member served from your own GPU is annotated in two places. The model runs on
the cluster, unattended, against a corpus shipped as plain JSON — see
``serving/annotate_offline.py`` — because a tunnel cannot be held open for a
slot the scheduler picks at 3 a.m. The Omeka write happens here, on the machine
that has the credentials. Nothing carries the answers from one to the other
until this script does.

The alternative is to let ``01`` re-annotate: the pipeline would work unchanged,
and on a self-hosted member the marginal cost is queue time rather than money.
It would also be wrong. Re-annotation is not reproducible on this panel — the
temperature is the vendor's and it is 1.0 — so a second pass produces different
labels for some articles, and the ones already published would silently change.
The answers that were measured, reported and reasoned about are the ones on the
cluster. This imports those.

What it will not do
-------------------
**Import an answer that a different instrument produced.** A cache record's
identity is ``(model_id, reasoning, prompt)``, and ``01`` looks up all three: a
record whose prompt fingerprint or reasoning level disagrees with the live panel
configuration is not a cheaper copy of the answer ``01`` would get, it is an
answer to a different question. Every record is checked against the live config
and the run aborts on a mismatch rather than importing a subset, because a
partial import is the one outcome that looks like success and is not.

**Import a failure.** Records carrying an ``analysis_error`` are counted and
skipped. Caching them would make a failed item look answered — the cache stores
successes only, precisely so a resume retries what never worked.

Usage
-----
    python AI_sentiment_analysis/04_import_offline_run.py \\
        --input cache/qwen38_full/qwen38_merged.jsonl --model qwen3_8_27b --dry-run

    python AI_sentiment_analysis/04_import_offline_run.py \\
        --input cache/qwen38_full/qwen38_merged.jsonl --model qwen3_8_27b

Then run ``01`` for that member. Every imported item is served from cache, so
the run is a write pass and asks the model nothing:

    python AI_sentiment_analysis/01_sentiment_analysis.py --models qwen3_8_27b --dry-run

This script touches no server. It reads a local JSONL and appends to the local
cache; ``01`` remains the only thing that writes to Omeka.
"""
import sys
import json
import argparse
import collections
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.console_utils import count_table, key_value_table
from common.llm_provider import get_model_option
from common.log_redaction import install_credential_redaction

# This script needs no credentials — it reads a local JSONL and appends to a
# local cache. Redaction is installed anyway, because "this entry point happens
# not to touch a key today" is a property that changes without anyone noticing,
# and the cost of being wrong is a key in a log file.
install_credential_redaction()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sentiment_core import (  # noqa: E402
    PANEL,
    PANEL_REASONING_EFFECTIVE,
    RESULT_FIELD_SUFFIXES,
    load_system_prompt,
    prompt_fingerprint,
)
from sentiment_cache import SentimentCache  # noqa: E402

console = Console()

CACHE_DIR_NAME = "cache"
CACHE_FILE_NAME = "sentiment_v2.jsonl"


def expected_identity(member_key: str) -> Dict[str, str]:
    """What a cache record for this member must carry to be found by ``01``.

    Read from the same places ``01`` reads them, rather than restated here: the
    registry for the model id, :data:`PANEL_REASONING_EFFECTIVE` for the depth
    actually requested of this model, and the prompt file for the fingerprint.
    A copy would be free to drift, and drift here means silently re-annotating a
    corpus that is already annotated.
    """
    member = PANEL[member_key]
    return {
        "model_id": get_model_option(member.registry_key).model,
        "reasoning": PANEL_REASONING_EFFECTIVE[member_key],
        "prompt": prompt_fingerprint(load_system_prompt()),
    }


def record_identity(record: Dict[str, Any]) -> Dict[str, str]:
    """The same three fields as an offline annotation record names them."""
    return {
        "model_id": str(record.get("model") or ""),
        "reasoning": str(record.get("reasoning_effort") or ""),
        "prompt": str(record.get("prompt") or ""),
    }


def read_offline(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                console.print(f"[red]{path.name}:{line_no} is not JSON[/red] — {exc}")
    return records


def partition(
    records: List[Dict[str, Any]], expected: Dict[str, str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[Tuple, int]]:
    """Split into importable, failed, and mismatched-by-identity."""
    importable: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    mismatched: Dict[Tuple, int] = collections.Counter()

    for record in records:
        identity = record_identity(record)
        if identity != expected:
            differing = tuple(
                (field, identity[field]) for field in expected
                if identity[field] != expected[field]
            )
            mismatched[differing] += 1
            continue
        result = record.get("result") or {}
        if result.get("analysis_error"):
            failed.append(record)
        else:
            importable.append(record)
    return importable, failed, mismatched


def clean_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the schema's own fields.

    ``analysis_error`` rides along on every offline record, null on the ones
    that worked. Storing a null under that key would leave the cache holding a
    field whose only readers test it for truthiness — harmless today, and
    exactly the sort of thing that is read as a sentinel later.
    """
    return {
        field: result[field]
        for field in RESULT_FIELD_SUFFIXES
        if field in result
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed the sentiment cache from an offline annotation run. "
                    "Writes nothing to Omeka."
    )
    parser.add_argument("--input", required=True,
                        help="Merged offline JSONL (see serving/merge_shards.py)")
    parser.add_argument("--model", required=True, choices=sorted(PANEL),
                        help="Panel member these annotations belong to")
    parser.add_argument("--cache", default=None,
                        help=f"Cache file to append to (default: {CACHE_DIR_NAME}/{CACHE_FILE_NAME})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be imported and write nothing")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.is_absolute():
        candidate = Path(__file__).resolve().parent / source
        source = candidate if candidate.exists() else source
    if not source.exists():
        console.print(f"[red]✗[/] No such file: {source}")
        return 2

    expected = expected_identity(args.model)
    records = read_offline(source)
    if not records:
        console.print("[red]✗[/] No records read.")
        return 2

    importable, failed, mismatched = partition(records, expected)

    console.print(key_value_table(
        [
            ("Source", str(source)),
            ("Panel member", f"{PANEL[args.model].label}  →  iwac:{PANEL[args.model].property_prefix}*"),
            ("Model id", expected["model_id"]),
            ("Reasoning", expected["reasoning"]),
            ("Prompt fingerprint", expected["prompt"]),
        ],
        title="Offline import",
    ))

    if mismatched:
        console.print(Panel(
            "These records were produced by a different instrument:\n\n  "
            + "\n  ".join(
                f"{count:,} record(s) — "
                + ", ".join(f"{field}={value!r} (expected {expected[field]!r})"
                            for field, value in differing)
                for differing, count in mismatched.most_common()
            )
            + "\n\nNothing was imported. A record whose prompt or reasoning "
              "level disagrees with the live panel is an answer to a different "
              "question, not a cheaper copy of this one.",
            title="Identity mismatch — refusing to import", border_style="red",
        ))
        return 1

    cache_path = (
        Path(args.cache) if args.cache
        else Path(__file__).resolve().parent / CACHE_DIR_NAME / CACHE_FILE_NAME
    )
    cache = SentimentCache(path=cache_path)
    report = cache.load()

    already = sum(
        1 for record in importable
        if cache.has(record["item_id"], args.model, **expected)
    )
    to_write = len(importable) - already

    console.print(count_table(
        [
            ("Records read", f"{len(records):,}"),
            ("Importable", f"{len(importable):,}"),
            ("…already cached", f"[dim]{already:,}[/]"),
            ("…to import", f"[green]{to_write:,}[/]"),
            ("Skipped — carried an analysis_error", f"[yellow]{len(failed):,}[/]"),
            ("Cache records loaded", f"[dim]{report.records:,}[/]"),
        ],
        title="Plan",
    ))

    if failed:
        console.print(
            f"[dim]The {len(failed):,} failures stay out of the cache, so they are "
            f"not counted as answered. If they are a retired gap rather than a "
            f"repairable one, the failure log beside the merged JSONL is what "
            f"records why.[/]"
        )

    if args.dry_run:
        console.print("[yellow]--dry-run: nothing written.[/]")
        return 0

    if not to_write:
        console.print("[green]✓[/] Nothing to import — the cache already has these.")
        return 0

    written = 0
    with cache:
        for record in importable:
            if cache.has(record["item_id"], args.model, **expected):
                continue
            cache.put(
                record["item_id"],
                args.model,
                clean_result(record.get("result") or {}),
                **expected,
            )
            written += 1

    console.print(f"[green]✓[/] Imported {written:,} annotations into {cache_path}")
    console.print(
        "[dim]Next: run 01 for this member. Every imported item is served from "
        "cache, so it writes to Omeka without asking the model anything:\n"
        f"  python AI_sentiment_analysis/01_sentiment_analysis.py --models {args.model} --dry-run[/]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
