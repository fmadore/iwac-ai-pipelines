#!/usr/bin/env python3
"""
Extract the works a publication cites, and write them to ``bibo:cites``.

Reads the ``apparatus`` blocks of the JSON sidecars produced by
``02_mistral_blocks_processor.py`` — the footnotes and the bibliography, already
separated from the body by the layout model — and turns them into one
``bibo:cites`` literal per distinct cited work.

Why not Mistral's ``document_annotation_format``, which does this in one call:
it costs more and returns less. Annotated pages bill at $5/1000 against $4/1000
for plain OCR, and the annotation runs *on top of* the OCR rather than instead
of it, so there is no cheaper "citations only" mode. Measured on item 4987 it
also under-extracts as the document grows — 10 cited works from the full 33
pages against 20 from the first 8. The apparatus blocks are already paid for by
step 02 (170 of them on item 5071), so this step only has to read them.

The same measurement is why the apparatus is sent in chunks rather than whole:
a single request over 20,000 characters of notes loses citations the same way.

``bibo:cites`` is declared on all four reference templates — Thesis, Journal
article, Book chapter, Book — and was populated on zero items before this step
existed.

Usage:
    python 04_extract_citations.py --item-id 5071 --dry-run
    python 04_extract_citations.py --item-id 5071
    python 04_extract_citations.py --model gpt-5.6-luna --limit 5
    python 04_extract_citations.py --extract-only    # write JSON, touch nothing

Requirements:
    - OCR_Results/<item_id>.json from step 02
    - The provider key for the chosen model, and Omeka credentials
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.console_utils import key_value_table, standard_progress
from common.iwac_config import BIBO_CITES_PROPERTY_ID
from common.llm_provider import LLMConfig, build_llm_client, get_model_option
from common.llm_registry import TEXT_ECONOMY_MODELS
from common.log_redaction import install_credential_redaction
from common.omeka_client import OmekaClient
from common.write_guard import WriteGuard, add_write_guard_args

console = Console()

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "OCR_Results"
OUTPUT_DIR = SCRIPT_DIR / "output"
LOG_DIR = SCRIPT_DIR / "log"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=LOG_DIR / "citations.log",
)
install_credential_redaction()

PROMPT_PATH = SCRIPT_DIR / "citation_extraction_prompt.md"
CITES_TERM = "bibo:cites"

#: Characters of apparatus per request. Small enough that the model does not
#: start dropping entries, large enough that ``Ibid.`` usually still has its
#: antecedent in the same chunk — the two pressures point in opposite
#: directions, and this is where they were balanced.
CHUNK_CHARS = 6000

#: Blocks of apparatus carried into the next chunk, so a short form near a
#: boundary can still be resolved against what preceded it.
CHUNK_OVERLAP_BLOCKS = 3

#: This stage overrides ``DEFAULT_TEXT_MODEL_KEY``. The shared default routes
#: through OpenRouter, and ``OPENROUTER_PROVIDER_PREFS`` narrows the eligible
#: backends to those advertising ``json_schema`` — which is what this stage
#: needs, and also what leaves it queueing behind everyone else wanting the
#: same thing. Measured here on 2026-08-18: individual chunk requests took 47
#: minutes and then 2 hours, against seconds for the same work on a first-party
#: endpoint. ``AI_summary`` pins Luna over the same default for the same class
#: of reason, throughput rather than quality.
DEFAULT_CITATION_MODEL_KEY = "gpt-5.6-luna"

#: A hung route must fail this chunk, not the run. The shared 300 s default is
#: sized for magazine-issue consolidation emitting a whole table of contents;
#: an apparatus chunk is a fraction of that, and a chunk that dies is retried
#: by re-running the step, since results are written per item.
CITATION_TIMEOUT_SECONDS = 180.0


class Citation(BaseModel):
    """One work cited by the publication."""

    raw: str = Field(description="The citation as printed, lightly cleaned")
    authors: List[str] = Field(default_factory=list)
    title: Optional[str] = None
    container: Optional[str] = Field(
        default=None, description="Journal, edited volume, newspaper or archive"
    )
    year: Optional[str] = None
    kind: Optional[str] = None
    cited_on_pages: List[int] = Field(default_factory=list)


class CitationList(BaseModel):
    citations: List[Citation] = Field(default_factory=list)


def apparatus_chunks(sidecar: Dict[str, Any]) -> List[str]:
    """Group the sidecar's apparatus blocks into prompt-sized chunks."""
    fragments: List[str] = []
    for page in sidecar.get("pages", []):
        page_number = page.get("index", 0) + 1
        for block in page.get("blocks", []):
            if block.get("role") != "apparatus":
                continue
            content = (block.get("content") or "").strip()
            if content:
                fragments.append(f"[page {page_number}] {content}")

    chunks: List[str] = []
    current: List[str] = []
    size = 0
    for fragment in fragments:
        if current and size + len(fragment) > CHUNK_CHARS:
            chunks.append("\n".join(current))
            current = current[-CHUNK_OVERLAP_BLOCKS:]
            size = sum(len(f) for f in current)
        current.append(fragment)
        size += len(fragment)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _fold(text: str) -> str:
    """Lower-case and strip accents, which a scanned apparatus prints both ways."""
    folded = unicodedata.normalize("NFKD", text.lower())
    return " ".join(
        "".join(ch for ch in folded if not unicodedata.combining(ch)).split()
    )


def _work_key(citation: Citation) -> str:
    """Author and title alone — the work, regardless of how fully it was cited."""
    basis = " ".join(
        part for part in (" ".join(citation.authors), citation.title or "") if part
    ).strip()
    return _fold(basis)


def _dedup_key(citation: Citation) -> str:
    """Fold a citation to a comparison key.

    The year is part of the key, because it is often the only thing separating
    two editions of one title. That alone would over-split — an apparatus cites
    a work fully once and briefly after, so the same book appears with and
    without its year — which is what :func:`_absorb_undated` then repairs.

    Citations with neither author nor title (archival files, interviews) fall
    back to ``raw``: there is nothing else to compare them on.
    """
    work = _work_key(citation)
    if not work:
        return _fold(citation.raw)
    return f"{work} {_fold(citation.year or '')}".strip()


def _merge_into(existing: Citation, citation: Citation) -> None:
    """Fold *citation* into *existing*, keeping the fullest of each field."""
    if len(citation.raw) > len(existing.raw):
        existing.raw = citation.raw
    for field in ("title", "container", "year", "kind"):
        if not getattr(existing, field) and getattr(citation, field):
            setattr(existing, field, getattr(citation, field))
    if not existing.authors and citation.authors:
        existing.authors = citation.authors
    existing.cited_on_pages = sorted(
        set(existing.cited_on_pages) | set(citation.cited_on_pages)
    )


def _absorb_undated(citations: List[Citation]) -> List[Citation]:
    """Fold a year-less citation into its dated twin, when there is only one.

    "GRESH (A.), L'Arabie-Saoudite" and "GRESH (A.), L'Arabie-Saoudite, 1983"
    are one work, and keeping them apart would double-count it. But where the
    same author and title carry *two* years, the undated citation genuinely
    cannot be assigned to either edition — so it is left standing on its own
    rather than attached to whichever happened to be seen first.
    """
    dated: Dict[str, List[Citation]] = {}
    for citation in citations:
        work = _work_key(citation)
        if work and citation.year:
            dated.setdefault(work, []).append(citation)

    out: List[Citation] = []
    for citation in citations:
        work = _work_key(citation)
        if work and not citation.year:
            candidates = dated.get(work, [])
            if len(candidates) == 1:
                _merge_into(candidates[0], citation)
                continue
        out.append(citation)
    return out


def merge_citations(batches: List[CitationList]) -> List[Citation]:
    """Merge overlapping chunk results into one work per entry."""
    merged: Dict[str, Citation] = {}
    for batch in batches:
        for citation in batch.citations:
            if not (citation.raw or "").strip():
                continue
            key = _dedup_key(citation)
            existing = merged.get(key)
            if existing is None:
                merged[key] = citation.model_copy(deep=True)
                continue
            # Keep the fullest form seen, and union the page lists: the overlap
            # between chunks exists precisely so a work cited across a boundary
            # is seen twice.
            _merge_into(existing, citation)
    return _absorb_undated(list(merged.values()))


def extract_for_item(llm_client, sidecar: Dict[str, Any], prompt: str) -> List[Citation]:
    """Run the extraction over one document's apparatus."""
    chunks = apparatus_chunks(sidecar)
    if not chunks:
        return []

    batches: List[CitationList] = []
    with standard_progress(console) as progress:
        task = progress.add_task(f"[cyan]{len(chunks)} chunks", total=len(chunks))
        for chunk in chunks:
            try:
                batches.append(
                    llm_client.generate_structured(prompt, chunk, CitationList)
                )
            except Exception as exc:
                console.print(f"  [yellow]⚠[/] a chunk failed: {exc}")
                logging.error("Chunk failed: %s", exc, exc_info=True)
            progress.update(task, advance=1)

    return merge_citations(batches)


def cites_values(citations: List[Citation]) -> List[Dict[str, Any]]:
    """Build the ``bibo:cites`` value objects for a whole-item PATCH."""
    return [
        {
            "type": "literal",
            "property_id": BIBO_CITES_PROPERTY_ID,
            "property_label": "cites",
            # Same reasoning as bibo:content: the apparatus of a copyrighted
            # work is part of that work.
            "is_public": False,
            "@value": citation.raw.strip(),
        }
        for citation in citations
        if citation.raw.strip()
    ]


def _values_match(stored: Any, wanted: List[Dict[str, Any]]) -> bool:
    """True when the stored values already say what *wanted* would write.

    Omeka echoes every value back with server-added keys (``@annotation``,
    ``value_resource_name`` …), so an exact list comparison never matches and
    every re-run would re-PATCH and re-dump a backup. Compare only the keys
    this pipeline sends, in order.
    """
    if not isinstance(stored, list) or len(stored) != len(wanted):
        return False
    return all(
        isinstance(have, dict) and all(have.get(key) == value for key, value in want.items())
        for have, want in zip(stored, wanted, strict=True)
    )


def write_citations(
    client: OmekaClient,
    item_id: int,
    citations: List[Citation],
    guard: WriteGuard,
) -> str:
    """PATCH one item's ``bibo:cites``, preserving every other property.

    The whole item is fetched and sent back: Omeka treats RDF properties as one
    block and deletes anything absent from the payload.
    """
    item_data = client.get_item(item_id)
    if not item_data:
        return "not_found"

    values = cites_values(citations)
    if _values_match(item_data.get(CITES_TERM), values):
        return "unchanged"

    if guard.dry_run:
        return "would_update"

    guard.dump_backup([item_data], label=f"citations_{item_id}")
    item_data[CITES_TERM] = values
    return "updated" if client.update_item(item_id, item_data) else "failed"


def _preview_table(citations: List[Citation], limit: int = 10) -> Table:
    table = Table(title=f"Cited works ({len(citations)})", box=box.ROUNDED)
    table.add_column("Kind", style="dim", width=10)
    table.add_column("Year", width=6)
    table.add_column("Citation", style="cyan")
    table.add_column("Pages", justify="right", width=6)
    for citation in citations[:limit]:
        table.add_row(
            citation.kind or "—",
            citation.year or "—",
            citation.raw[:78],
            str(len(citation.cited_on_pages)) or "—",
        )
    return table


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract cited works from publication apparatus into bibo:cites.",
    )
    parser.add_argument(
        "--item-id", type=int, action="append", dest="item_ids",
        help="Restrict to these item ids. Repeatable.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N documents.")
    parser.add_argument(
        "--model", default=None,
        help=f"Text model key (default: {DEFAULT_CITATION_MODEL_KEY}).",
    )
    parser.add_argument(
        "--extract-only", action="store_true",
        help="Write the citation JSON and stop — Omeka is not touched.",
    )
    parser.add_argument(
        "--re-extract", action="store_true",
        help="Call the model again even where a citation JSON already exists.",
    )
    add_write_guard_args(parser, default_backup_dir=OUTPUT_DIR)
    args = parser.parse_args()

    console.print(Panel(
        "[bold]Extract cited works from scholarly apparatus[/]\n\n"
        "Reads the footnote and bibliography blocks isolated by step 02 and\n"
        "writes one bibo:cites literal per distinct work.",
        title="📚 Citation Extractor",
        border_style="cyan",
    ))
    console.print()

    load_dotenv()
    if not PROMPT_PATH.exists():
        console.print(f"[red]✗[/] Prompt not found: {PROMPT_PATH}")
        return 1
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    sidecars = sorted(RESULTS_DIR.glob("*.json"))
    sidecars = [p for p in sidecars if p.stem.isdigit()]
    if args.item_ids:
        wanted = {str(i) for i in args.item_ids}
        sidecars = [p for p in sidecars if p.stem in wanted]
    if args.limit:
        sidecars = sidecars[: args.limit]

    if not sidecars:
        console.print("[yellow]⚠[/] No sidecars found. Run 02 first.")
        return 1

    model_option = get_model_option(
        args.model or DEFAULT_CITATION_MODEL_KEY, allowed_keys=TEXT_ECONOMY_MODELS
    )
    # No temperature: MODEL_REGISTRY holds each vendor's recommendation.
    config = LLMConfig(
        reasoning_effort="low",
        thinking_level="minimal",
        request_timeout_seconds=CITATION_TIMEOUT_SECONDS,
    )
    # Built on first use: writing a reviewed set back to Omeka should not
    # require a provider key for a model it never calls.
    _client_cache: List[Any] = []

    def llm() -> Any:
        if not _client_cache:
            _client_cache.append(build_llm_client(model_option, config=config))
        return _client_cache[0]

    guard = WriteGuard.from_args(args, default_backup_dir=OUTPUT_DIR)
    console.print(key_value_table([
        ("Model", f"{model_option.label} ({model_option.model})"),
        ("Sidecars", len(sidecars)),
        ("Target property", f"{CITES_TERM} (id {BIBO_CITES_PROPERTY_ID})"),
        ("Mode", "extract only — no Omeka write" if args.extract_only else guard.mode_label),
    ]))
    console.print()

    try:
        client = None if args.extract_only else OmekaClient.from_env()
    except ValueError as exc:
        console.print(f"[red]✗[/] {exc}")
        return 1

    OUTPUT_DIR.mkdir(exist_ok=True)
    results: Dict[int, List[Citation]] = {}

    for path in sidecars:
        item_id = int(path.stem)
        console.print()
        console.rule(f"[bold]📄 item {item_id}[/]")

        # Extraction is not deterministic — the same apparatus yielded 95 works
        # on one run and 88 on the next — so a stored result is reused rather
        # than regenerated. Otherwise the set an operator reviewed is not the
        # set that gets written, and every dry run costs another extraction.
        cached_path = OUTPUT_DIR / f"citations_{item_id}.json"
        if cached_path.exists() and not args.re_extract:
            citations = [
                Citation(**entry)
                for entry in json.loads(cached_path.read_text(encoding="utf-8"))
            ]
            console.print(
                f"  [dim]Reusing {cached_path.name} "
                f"({len(citations)} works) — pass --re-extract to redo[/]"
            )
        else:
            sidecar = json.loads(path.read_text(encoding="utf-8"))
            citations = extract_for_item(llm(), sidecar, prompt)
            if citations:
                cached_path.write_text(
                    json.dumps(
                        [c.model_dump() for c in citations], ensure_ascii=False, indent=2
                    ),
                    encoding="utf-8",
                )
            logging.info("item %s: %d cited works", item_id, len(citations))

        results[item_id] = citations
        if not citations:
            console.print("  [yellow]⚠[/] no citable work found in the apparatus")
            continue

        console.print(_preview_table(citations))

    total = sum(len(v) for v in results.values())
    console.print()
    console.print(key_value_table([
        ("Documents", len(results)),
        ("Cited works", total),
    ], title="Extraction", value_style="cyan"))

    if args.extract_only or not total:
        console.print(f"\n[green]✓[/] Citation JSON in [cyan]{OUTPUT_DIR}[/]")
        return 0

    if not guard.confirm(
        console,
        action=f"Write {total} bibo:cites literals",
        base_url=client.base_url,
        item_count=len([v for v in results.values() if v]),
        details=[f"Property:  {CITES_TERM} (id {BIBO_CITES_PROPERTY_ID}), is_public=False"],
    ):
        return 1

    stats: Dict[str, int] = {}
    for item_id, citations in results.items():
        if not citations:
            continue
        status = write_citations(client, item_id, citations, guard)
        stats[status] = stats.get(status, 0) + 1
        console.print(f"  item {item_id}: [cyan]{status}[/] ({len(citations)} works)")

    console.print()
    console.print(key_value_table(list(stats.items()), title="Write", value_style="cyan"))
    return 0 if not stats.get("failed") else 1


if __name__ == "__main__":
    sys.exit(main())
