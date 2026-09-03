#!/usr/bin/env python3
"""
Assign French subject and spatial keywords to scholarly references with an LLM.

Reads the ``items_*.csv`` that step 01 exported, sends each reference's text
to the chosen model with the rules in ``02_enrichment_prompt.md``, and writes
an enriched CSV — every original column plus ``Subject AI`` and ``Spatial AI``
(pipe-separated) — for step 03 to reconcile against the authority records.

Items whose ``bibo:content`` is empty get empty keyword columns. Items that
already carry both subject and spatial links are skipped as well, unless
``--reindex`` is passed; when an item carries only one of the two, its existing
links are resolved to names through ``index_*.csv`` and handed to the model so
the new keywords complement rather than repeat them.

Output is durable: each row is flushed as it is written, and a checkpoint
beside the CSV records the model, prompt and input it was made with. Re-running
resumes where it stopped; a run with a different model or prompt refuses to
append to the old file unless ``--force`` is passed.

Usage:
    python 02_enrich_references.py                              # newest items_*.csv, default model
    python 02_enrich_references.py --input output/items_78405_20260902.csv
    python 02_enrich_references.py --model gpt-5.6-luna
    python 02_enrich_references.py --reindex                    # also items that already have both
    python 02_enrich_references.py --force                      # discard an output made differently
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.checkpoint import (  # noqa: E402
    CheckpointMismatch,
    JsonCheckpoint,
    checkpoint_path_for,
    load_csv_ids,
    sha256_text,
)
from common.console_utils import count_table, key_value_table, standard_progress  # noqa: E402
from common.llm_provider import (  # noqa: E402
    DEFAULT_TEXT_MODEL_KEY,
    LEGACY_CLI_MODEL_KEYS,
    TEXT_EXTENDED_MODELS,
    BaseLLMClient,
    LLMConfig,
    build_llm_client,
    get_model_option,
    summary_from_option,
)
from common.log_redaction import install_credential_redaction  # noqa: E402
from common.retry import retry_with_backoff  # noqa: E402

# Credentials ride in provider headers; keep them out of anything an SDK
# decides to log.
install_credential_redaction()

console = Console()

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
PROMPT_PATH = SCRIPT_DIR / "02_enrichment_prompt.md"

ALLOWED_MODEL_KEYS = TEXT_EXTENDED_MODELS
SUBJECT_COLUMN = "Subject AI"
SPATIAL_COLUMN = "Spatial AI"
EXISTING_SUBJECT_COLUMN = "Existing Subject IDs"
EXISTING_SPATIAL_COLUMN = "Existing Spatial IDs"
CONTENT_COLUMN = "bibo:content"

#: Terms the prompt forbids because the whole collection is about them; dropped
#: again here so a model that ignores the rule cannot flood the authority file.
GENERIC_TERMS = {"islam", "musulmans", "musulman", "musulmane", "musulmanes"}


class ReferenceKeywords(BaseModel):
    """What the model returns for one reference — the two keyword lists."""

    subjects: List[str] = Field(
        default_factory=list,
        description="5 to 8 French thematic keywords (persons, organisations, themes, events)",
    )
    spatial: List[str] = Field(
        default_factory=list,
        description="French place names mentioned or implied, most specific first",
    )


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def find_latest_items_csv(output_dir: Path = OUTPUT_DIR) -> Optional[Path]:
    """The newest ``items_*.csv`` from step 01, ignoring enriched/reconciled ones."""
    candidates = [
        path for path in output_dir.glob("items_*.csv")
        if not path.name.startswith("items_enriched_")
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def load_index_titles(output_dir: Path = OUTPUT_DIR) -> Dict[str, str]:
    """``id -> title`` for every authority term step 01 exported.

    Both index files share one namespace of Omeka ids, so one lookup serves
    subject and spatial columns alike. Missing files just mean existing links
    cannot be named and are left out of the prompt.
    """
    titles: Dict[str, str] = {}
    for name in ("index_subject.csv", "index_spatial.csv"):
        path = output_dir / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("id") and row.get("title"):
                    titles[row["id"].strip()] = row["title"].strip()
    return titles


def resolve_names(ids_field: str, titles: Mapping[str, str]) -> List[str]:
    """Turn a pipe-separated id list into the titles the index knows."""
    return [
        titles[token.strip()]
        for token in (ids_field or "").split("|")
        if token.strip() in titles
    ]


def read_items(path: Path) -> List[Dict[str, str]]:
    csv.field_size_limit(10 * 1024 * 1024)  # sys.maxsize overflows C long on Windows
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ["o:id", "Title", EXISTING_SUBJECT_COLUMN, EXISTING_SPATIAL_COLUMN, CONTENT_COLUMN]
        missing = [column for column in required if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name} lacks column(s): {', '.join(missing)}")
        return list(reader)


# ---------------------------------------------------------------------------
# The model call
# ---------------------------------------------------------------------------

def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_user_prompt(
    title: str,
    content: str,
    existing_subjects: Sequence[str],
    existing_spatial: Sequence[str],
) -> str:
    lines = [f"TITLE: {title}".strip()]
    if existing_subjects:
        lines.append(f"EXISTING SUBJECTS: {', '.join(existing_subjects)}")
    if existing_spatial:
        lines.append(f"EXISTING SPATIAL: {', '.join(existing_spatial)}")
    lines.append("")
    lines.append("TEXT TO ANALYZE:")
    lines.append(content)
    return "\n".join(lines)


def clean_terms(terms: Iterable[str], existing: Iterable[str] = ()) -> List[str]:
    """Strip, drop blanks, generic terms and repeats (case-insensitively)."""
    seen = {term.strip().casefold() for term in existing}
    cleaned: List[str] = []
    for term in terms:
        text = " ".join(str(term).split())
        key = text.casefold()
        if not text or key in GENERIC_TERMS or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


def enrich_reference(
    llm_client: BaseLLMClient,
    system_prompt: str,
    row: Mapping[str, str],
    titles: Mapping[str, str],
) -> ReferenceKeywords:
    """One structured call, with the item's existing links named for the model."""
    existing_subjects = resolve_names(row.get(EXISTING_SUBJECT_COLUMN, ""), titles)
    existing_spatial = resolve_names(row.get(EXISTING_SPATIAL_COLUMN, ""), titles)
    user_prompt = build_user_prompt(
        row.get("Title", ""), row.get(CONTENT_COLUMN, ""), existing_subjects, existing_spatial
    )

    @retry_with_backoff(max_retries=3, base_delay=5.0)
    def _call() -> ReferenceKeywords:
        return llm_client.generate_structured(system_prompt, user_prompt, ReferenceKeywords)

    result = _call()
    return ReferenceKeywords(
        subjects=clean_terms(result.subjects, existing_subjects),
        spatial=clean_terms(result.spatial, existing_spatial),
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def enriched_path_for(input_path: Path) -> Path:
    """``items_78405_20260902.csv`` -> ``items_enriched_78405_20260902.csv``.

    Step 03 finds its input by the ``items_enriched_`` prefix, so the name is
    part of the contract between the two steps.
    """
    stem = input_path.stem
    if stem.startswith("items_"):
        stem = stem[len("items_"):]
    return input_path.with_name(f"items_enriched_{stem}{input_path.suffix}")


def needs_enrichment(row: Mapping[str, str], *, reindex: bool) -> bool:
    if not (row.get(CONTENT_COLUMN) or "").strip():
        return False
    if reindex:
        return True
    has_both = bool((row.get(EXISTING_SUBJECT_COLUMN) or "").strip()) and bool(
        (row.get(EXISTING_SPATIAL_COLUMN) or "").strip()
    )
    return not has_both


def write_keyword_summary(enriched_csv: Path, output_dir: Path = OUTPUT_DIR) -> Path:
    """``keyword_summary_<timestamp>.csv``: Term, Type, Count over the whole file."""
    counts: Counter = Counter()
    with enriched_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for column, kind in ((SUBJECT_COLUMN, "subject"), (SPATIAL_COLUMN, "spatial")):
                for term in (row.get(column) or "").split("|"):
                    if term.strip():
                        counts[(term.strip(), kind)] += 1
    path = output_dir / f"keyword_summary_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Term", "Type", "Count"])
        for (term, kind), count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([term, kind, count])
    return path


def run(
    llm_client: BaseLLMClient,
    system_prompt: str,
    rows: Sequence[Mapping[str, str]],
    titles: Mapping[str, str],
    output_path: Path,
    *,
    fieldnames: Sequence[str],
    resume: bool,
    reindex: bool,
) -> Counter:
    """Write every row to *output_path*, enriching the ones that need it.

    Rows that need no model call are written straight away with empty keyword
    columns, so the finished file holds every input row and step 03 can run on
    it as a whole. Each row is flushed as it is written.
    """
    stats: Counter = Counter()
    mode = "a" if resume else "w"
    with output_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        if not resume:
            writer.writeheader()
            handle.flush()
        with standard_progress(console) as progress:
            task = progress.add_task("[cyan]Assigning keywords...", total=len(rows))
            for row in rows:
                out = dict(row)
                out.setdefault(SUBJECT_COLUMN, "")
                out.setdefault(SPATIAL_COLUMN, "")
                if not needs_enrichment(row, reindex=reindex):
                    stats["skipped"] += 1
                else:
                    try:
                        keywords = enrich_reference(llm_client, system_prompt, row, titles)
                    except Exception as exc:  # the row is left out, so a re-run retries it
                        stats["failed"] += 1
                        console.print(f"[red]✗[/] item {row.get('o:id')}: {exc}")
                        progress.update(task, advance=1)
                        continue
                    out[SUBJECT_COLUMN] = "|".join(keywords.subjects)
                    out[SPATIAL_COLUMN] = "|".join(keywords.spatial)
                    stats["enriched"] += 1
                    stats["subject_terms"] += len(keywords.subjects)
                    stats["spatial_terms"] += len(keywords.spatial)
                writer.writerow(out)
                handle.flush()
                progress.update(task, advance=1)
    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assign French subject and spatial keywords to references with an LLM.",
    )
    parser.add_argument(
        "--input", type=Path,
        help="items_*.csv from step 01 (default: the newest one in output/).",
    )
    parser.add_argument(
        "--model", choices=ALLOWED_MODEL_KEYS + LEGACY_CLI_MODEL_KEYS, default=DEFAULT_TEXT_MODEL_KEY,
        help=f"Text model (default: {DEFAULT_TEXT_MODEL_KEY}).",
    )
    parser.add_argument(
        "--reindex", action="store_true",
        help="Also process items that already carry both subject and spatial links.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Replace an existing output whose model, prompt or input differs.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    console.print(Panel(
        "[bold]Reference Indexing — Step 2[/bold]\n"
        "Assign Subject AI and Spatial AI keywords with an LLM",
        title="Enrich References",
        border_style="cyan",
    ))

    input_path = args.input or find_latest_items_csv()
    if input_path is None or not input_path.is_file():
        console.print(f"[red]✗[/] No items_*.csv found in {OUTPUT_DIR}. Run 01_fetch_references.py first.")
        return 1

    try:
        model_option = get_model_option(args.model, allowed_keys=ALLOWED_MODEL_KEYS)
        # "minimal" is a request for the shallowest level the model offers; the
        # adapter snaps it to what the model accepts. No temperature: that is
        # the vendor's, and lives in MODEL_REGISTRY.
        llm_client = build_llm_client(
            model_option, config=LLMConfig(reasoning_effort="medium", thinking_level="minimal")
        )
        rows = read_items(input_path)
    except ValueError as exc:
        console.print(f"[red]✗[/] {exc}")
        return 1

    system_prompt = load_prompt()
    titles = load_index_titles()
    output_path = enriched_path_for(input_path)
    fieldnames = list(rows[0].keys()) if rows else []
    for column in (SUBJECT_COLUMN, SPATIAL_COLUMN):
        if column not in fieldnames:
            fieldnames.append(column)

    context = {
        "pipeline": "reference-enrichment-v1",
        "model_key": model_option.key,
        "model_id": model_option.model,
        "prompt_sha256": sha256_text(system_prompt),
        "input": input_path.name,
        "reindex": args.reindex,
    }
    try:
        checkpoint_path = checkpoint_path_for(output_path)
        if output_path.exists() and not checkpoint_path.exists() and not args.force:
            raise CheckpointMismatch(
                f"Existing output has no provenance checkpoint: {output_path}. Use --force to replace it."
            )
        JsonCheckpoint.open(checkpoint_path, context, reset=args.force)
        done = set() if args.force else load_csv_ids(output_path, "o:id")
    except CheckpointMismatch as exc:
        console.print(f"[red]✗[/] {exc}")
        return 1
    pending = [row for row in rows if (row.get("o:id") or "").strip() not in done]
    resume = bool(done) and output_path.exists()

    console.print(key_value_table([
        ("Input", input_path.name),
        ("Output", output_path.name),
        ("Model", summary_from_option(model_option)),
        ("Items", len(rows)),
        ("Already done", len(done) or None),
        ("To process", sum(1 for row in pending if needs_enrichment(row, reindex=args.reindex))),
        ("Existing links named", f"{len(titles)} authority titles" if titles else "no index files"),
        ("Mode", "reindex all" if args.reindex else "skip items with both link sets"),
    ]))
    console.print()

    if not pending:
        console.print("[green]✓[/] Output is already complete for this model and prompt.")
        return 0

    stats = run(
        llm_client, system_prompt, pending, titles, output_path,
        fieldnames=fieldnames, resume=resume, reindex=args.reindex,
    )
    summary_path = write_keyword_summary(output_path)

    console.print()
    console.print(count_table([
        ("Enriched", stats["enriched"]),
        ("Skipped (no text / already indexed)", stats["skipped"]),
        ("Failed", stats["failed"]),
        ("Subject terms", stats["subject_terms"]),
        ("Spatial terms", stats["spatial_terms"]),
        ("Model usage", llm_client.usage.summary() if llm_client.usage.requests else None),
    ], title="Step 2 Summary"))
    console.print(Panel(
        f"[green]✓[/] Enriched CSV: [cyan]{output_path.name}[/]\n"
        f"Keyword summary: [cyan]{summary_path.name}[/]\n\n"
        "Next: python 03_reconcile_metadata.py",
        title="Step 2 Complete",
        border_style="green",
    ))
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
