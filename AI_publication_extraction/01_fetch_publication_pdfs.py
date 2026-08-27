#!/usr/bin/env python3
"""
Discover scholarly references on Omeka and download their PDFs.

Unlike the newspaper OCR pipeline, this step takes no item-set argument: the
"references" population is defined by resource *class*, and the nine classes are
declared once in ``common.iwac_config.REFERENCE_RESOURCE_CLASSES``. The script
sweeps all nine, keeps the items that have a PDF and no ``bibo:content`` yet,
and downloads those PDFs into ``PDF/`` named ``<item_id>.pdf``.

Why class and not template: template 10 carries both ``Book`` (40) and
``EditedBook`` (52), so a template filter would over- and under-select at once.

Why ``o:media_type`` and not the filename: 43 of the reference items carry a
cover image as their *first* media and the PDF further down the list, and some
sources have no ``.pdf`` suffix at all. Selecting on the declared media type is
what makes the backlog count come out right (47 items, not 90).

Usage:
    python 01_fetch_publication_pdfs.py                  # the whole backlog
    python 01_fetch_publication_pdfs.py --list           # report only, download nothing
    python 01_fetch_publication_pdfs.py --item-id 5312   # one item (pilot runs)
    python 01_fetch_publication_pdfs.py --limit 5
    python 01_fetch_publication_pdfs.py --include-processed   # ignore existing text

Requirements:
    - Environment variables: OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.console_utils import key_value_table, standard_progress
from common.downloader import stream_download
from common.iwac_config import REFERENCE_RESOURCE_CLASSES
from common.log_redaction import install_credential_redaction
from common.omeka_client import OmekaClient

console = Console()

SCRIPT_DIR = Path(__file__).resolve().parent
PDF_DIR = SCRIPT_DIR / "PDF"
OUTPUT_DIR = SCRIPT_DIR / "output"
LOG_DIR = SCRIPT_DIR / "log"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=LOG_DIR / "fetch.log",
)
install_credential_redaction()

#: Large scans are the norm here (one thesis is 276 MB), so the per-file
#: timeout is far longer than the shared default.
DOWNLOAD_TIMEOUT = 600


@dataclass
class Candidate:
    """One reference item and the PDFs attached to it."""

    item_id: int
    resource_class: str
    title: str
    has_text: bool
    pdf_urls: List[str]
    pdf_bytes: int
    num_pages: Optional[int]

    @property
    def megabytes(self) -> float:
        return self.pdf_bytes / 1e6


def _first_value(item: Dict[str, Any], term: str) -> str:
    for value in item.get(term) or []:
        text = value.get("@value")
        if text:
            return str(text)
    return ""


def _int_value(item: Dict[str, Any], term: str) -> Optional[int]:
    raw = _first_value(item, term)
    try:
        return int(raw.strip())
    except (ValueError, AttributeError):
        return None


def _has_text(item: Dict[str, Any]) -> bool:
    return any((v.get("@value") or "").strip() for v in item.get("bibo:content") or [])


def items_by_class(client: OmekaClient, class_id: int) -> List[Dict[str, Any]]:
    """Every item of one resource class, paginated.

    ``OmekaClient.get_items`` is shaped around item sets, but ``requests`` drops
    query parameters whose value is ``None``, so passing no item set turns it
    into a plain class query — verified live against all nine classes, which
    return exactly their documented counts and nothing of another class.

    A ``get_items_by_class`` method on the shared client would read better, but
    ``omeka_client.py`` is not modified without sign-off: every pipeline in the
    repo depends on it.
    """
    return client.get_items(None, resource_class_id=class_id)


def discover(
    client: OmekaClient,
    *,
    item_ids: Optional[List[int]] = None,
    include_processed: bool = False,
) -> List[Candidate]:
    """Sweep the reference classes and return the items worth downloading.

    Every media of every item is resolved, because the PDF is often not the
    first one. That is one API call per media, so the sweep is the slow part of
    this step — not the downloads.
    """
    items: List[Dict[str, Any]] = []
    if item_ids:
        for item_id in item_ids:
            item = client.get_item(item_id)
            if item:
                items.append(item)
            else:
                console.print(f"[yellow]⚠[/] Item {item_id} not found — skipped")
    else:
        with standard_progress(console) as progress:
            task = progress.add_task(
                "[cyan]Sweeping reference classes...", total=len(REFERENCE_RESOURCE_CLASSES)
            )
            for class_id in REFERENCE_RESOURCE_CLASSES:
                items.extend(items_by_class(client, class_id))
                progress.update(task, advance=1)

    candidates: List[Candidate] = []
    media_cache: Dict[str, Dict[str, Any]] = {}

    with standard_progress(console) as progress:
        task = progress.add_task("[cyan]Resolving media...", total=len(items))
        for item in items:
            urls: List[str] = []
            size = 0
            for media in item.get("o:media") or []:
                url = media.get("@id")
                if not url:
                    continue
                if url not in media_cache:
                    media_cache[url] = client.get_resource(url) or {}
                data = media_cache[url]
                source = str(data.get("o:source") or "")
                is_pdf = (
                    data.get("o:media_type") == "application/pdf"
                    or source.lower().endswith(".pdf")
                )
                if is_pdf:
                    original = data.get("o:original_url") or source
                    if original:
                        urls.append(original)
                        size += int(data.get("o:size") or 0)

            class_id = (item.get("o:resource_class") or {}).get("o:id")
            candidates.append(
                Candidate(
                    item_id=item["o:id"],
                    resource_class=REFERENCE_RESOURCE_CLASSES.get(class_id, str(class_id)),
                    title=_first_value(item, "dcterms:title"),
                    has_text=_has_text(item),
                    pdf_urls=urls,
                    pdf_bytes=size,
                    num_pages=_int_value(item, "bibo:numPages"),
                )
            )
            progress.update(task, advance=1)

    with_pdf = [c for c in candidates if c.pdf_urls]
    if include_processed:
        return with_pdf
    return [c for c in with_pdf if not c.has_text]


def _summary_table(candidates: List[Candidate]) -> Table:
    table = Table(title="Backlog by resource class", box=box.ROUNDED)
    table.add_column("Class", style="cyan")
    table.add_column("Items", justify="right")
    table.add_column("PDF size", justify="right")
    table.add_column("Known pages", justify="right")

    by_class: Dict[str, List[Candidate]] = {}
    for candidate in candidates:
        by_class.setdefault(candidate.resource_class, []).append(candidate)

    for name in sorted(by_class, key=lambda k: -len(by_class[k])):
        group = by_class[name]
        pages = sum(c.num_pages or 0 for c in group)
        table.add_row(
            name,
            str(len(group)),
            f"{sum(c.megabytes for c in group):,.0f} MB",
            f"{pages:,}" if pages else "—",
        )
    return table


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover scholarly references on Omeka and download their PDFs.",
    )
    parser.add_argument(
        "--item-id", type=int, action="append", dest="item_ids",
        help="Restrict to one item id. Repeatable. Skips the class sweep.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Download at most N documents (smallest first, so a pilot is cheap).",
    )
    parser.add_argument(
        "--include-processed", action="store_true",
        help="Also take items that already carry bibo:content (re-extraction).",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_only",
        help="Report what would be downloaded and exit.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PDF_DIR,
        help=f"Where PDFs are written (default: {PDF_DIR}).",
    )
    args = parser.parse_args()

    console.print(Panel(
        "[bold]Discover scholarly references and download their PDFs[/]\n\n"
        "Sweeps the nine reference resource classes, keeps items with a PDF "
        "and no extracted text, and downloads them as <item_id>.pdf.",
        title="📚 Publication PDF Fetcher",
        border_style="cyan",
    ))
    console.print()

    load_dotenv()
    try:
        client = OmekaClient.from_env()
    except ValueError as exc:
        console.print(f"[red]✗[/] {exc}")
        return 1

    console.print(key_value_table([
        ("Omeka URL", client.base_url),
        ("Resource classes", ", ".join(str(c) for c in REFERENCE_RESOURCE_CLASSES)),
        ("Scope", f"item {args.item_ids}" if args.item_ids else "all reference classes"),
        ("Filter", "all with a PDF" if args.include_processed else "no bibo:content yet"),
        ("Output", str(args.output_dir)),
    ]))
    console.print()

    candidates = discover(
        client, item_ids=args.item_ids, include_processed=args.include_processed
    )
    if not candidates:
        console.print("[yellow]⚠[/] Nothing to download — no matching item has a PDF.")
        return 0

    # Smallest first: a pilot run should hit the cheap documents, and a run that
    # is interrupted has then produced the most results per minute spent.
    candidates.sort(key=lambda c: c.pdf_bytes)
    if args.limit:
        candidates = candidates[: args.limit]

    console.print(_summary_table(candidates))
    console.print()
    total_mb = sum(c.megabytes for c in candidates)
    known = [c.num_pages for c in candidates if c.num_pages]
    est_pages = sum(known) + (len(candidates) - len(known)) * (
        sorted(known)[len(known) // 2] if known else 0
    )
    console.print(key_value_table([
        ("Documents", len(candidates)),
        ("Total download", f"{total_mb:,.0f} MB"),
        ("Pages (bibo:numPages)", f"{sum(known):,} known on {len(known)} items"),
        ("Estimated OCR cost", f"~${est_pages * 4 / 1000:,.2f}" if est_pages else "unknown"),
    ], title="Scope"))
    console.print()

    OUTPUT_DIR.mkdir(exist_ok=True)
    manifest_path = OUTPUT_DIR / "candidates.json"
    manifest_path.write_text(
        json.dumps([asdict(c) for c in candidates], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    console.print(f"[dim]Manifest: {manifest_path}[/]")

    if args.list_only:
        for candidate in candidates:
            console.print(
                f"  [cyan]{candidate.item_id}[/] "
                f"[dim]{candidate.megabytes:>7,.1f} MB[/] "
                f"{candidate.resource_class:<20} {candidate.title[:60]}"
            )
        console.print("\n[dim]--list: nothing downloaded.[/]")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = skipped = failed = 0

    console.print()
    console.rule("[bold cyan]Downloading")
    for candidate in candidates:
        for index, url in enumerate(candidate.pdf_urls, start=1):
            name = f"{candidate.item_id}.pdf"
            if len(candidate.pdf_urls) > 1:
                name = f"{candidate.item_id}_{index}.pdf"
            path = args.output_dir / name

            if path.exists():
                console.print(f"  [dim]•[/] {name} already downloaded — skipped")
                skipped += 1
                continue

            console.print(
                f"  [cyan]↓[/] {name} [dim]({candidate.megabytes:,.1f} MB)[/] "
                f"{candidate.title[:50]}"
            )
            result = stream_download(
                url, path, timeout=DOWNLOAD_TIMEOUT, logger=logging.getLogger(__name__)
            )
            if result:
                downloaded += 1
            else:
                console.print(f"    [red]✗[/] download failed for item {candidate.item_id}")
                failed += 1

    console.print()
    console.print(key_value_table([
        ("Downloaded", downloaded),
        ("Already present", skipped),
        ("Failed", failed),
    ], title="Summary", value_style="cyan"))
    console.print(f"\n[green]✓[/] PDFs in [cyan]{args.output_dir}[/]")
    console.print("[dim]Next: python 02_mistral_blocks_processor.py[/]")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
