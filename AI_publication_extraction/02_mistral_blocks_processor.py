#!/usr/bin/env python3
"""
Structured OCR for scholarly publications, via Mistral Document AI (OCR 4.1).

Each PDF in ``PDF/`` is sent to the OCR endpoint with ``include_blocks=True``,
which returns paragraph-level blocks carrying a structural label — ``text``,
``title``, ``table``, ``references``, ``header``, ``footer`` and the rest — in
reading order. Two files come back out per document:

    OCR_Results/<item_id>.txt    plain text for bibo:content (step 03)
    OCR_Results/<item_id>.json   the typed blocks, with bboxes and roles

The JSON sidecar is the point of this pipeline. It is what a later citation pass
reads: the footnotes and the bibliography are already separated from the body
there, so extracting cited works never has to guess where the apparatus starts.

Why this is not a flag on ``AI_ocr_extraction/02_mistral_ocr_processor.py``:
that script drops every page's header and footer from page 2 on, which is right
for a newspaper's running head and destructive here. On a 33-page article from
the Cahiers du CERLESHS, the page feet held 53 substantive footnotes — 7,388
characters, including full citations — against 32 folio numbers. So a foot is
kept unless it is a folio number or repeats across the document; see
``common.mistral_ocr.classify_blocks``.

Oversized scans are handled rather than skipped: Mistral rejects uploads over
50 MB, and four documents in the backlog exceed that (up to 276 MB). They are
split by page range, OCR'd part by part, and stitched back with their original
page numbers.

Usage:
    python 02_mistral_blocks_processor.py
    python 02_mistral_blocks_processor.py --item-id 5312    # pilot on one document
    python 02_mistral_blocks_processor.py --limit 3
    python 02_mistral_blocks_processor.py --force           # redo completed documents
    python 02_mistral_blocks_processor.py --rpm 30          # proactive throttling

Requirements:
    - MISTRAL_API_KEY in the environment / .env
    - PDFs in PDF/ (named <item_id>.pdf by 01_fetch_publication_pdfs.py)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.console_utils import key_value_table
from common.log_redaction import install_credential_redaction
from common.mistral_ocr import (
    MISTRAL_OCR_MODEL,
    MistralOcrClient,
    classify_blocks,
    render_plain_text,
)
from common.rate_limiter import QuotaExhaustedError

console = Console()

SCRIPT_DIR = Path(__file__).resolve().parent
PDF_DIR = SCRIPT_DIR / "PDF"
RESULTS_DIR = SCRIPT_DIR / "OCR_Results"
LOG_DIR = SCRIPT_DIR / "log"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=LOG_DIR / "ocr_blocks.log",
)
install_credential_redaction()

#: Bumped when the block→text rules change, so ``--force`` is not the only way
#: to tell an old sidecar from one this version would produce.
SIDECAR_VERSION = 1


def build_sidecar(
    pdf_path: Path, result, blocks, text: str, elapsed: float
) -> Dict:
    """Assemble the JSON written beside the plain text."""
    role_counts = Counter(b.role for b in blocks)
    type_counts = Counter(b.type for b in blocks)

    pages = []
    by_page: Dict[int, List] = {}
    for block in blocks:
        by_page.setdefault(block.page_index, []).append(block)

    for raw in result.pages:
        index = raw.get("index", 0)
        pages.append({
            "index": index,
            "dimensions": raw.get("dimensions"),
            "tables": raw.get("tables") or [],
            "hyperlinks": raw.get("hyperlinks") or [],
            "blocks": [
                {
                    "type": b.type,
                    "role": b.role,
                    "content": b.content,
                    "bbox": b.bbox,
                }
                for b in by_page.get(index, [])
            ],
        })

    return {
        "sidecar_version": SIDECAR_VERSION,
        "source_pdf": pdf_path.name,
        "source_bytes": pdf_path.stat().st_size,
        "model": result.model,
        "pages_processed": result.pages_processed,
        "upload_parts": result.parts,
        "warnings": result.warnings,
        "elapsed_seconds": round(elapsed, 2),
        "text_characters": len(text),
        "block_roles": dict(role_counts),
        "block_types": dict(type_counts),
        "pages": pages,
    }


def process_one(
    client: MistralOcrClient, pdf_path: Path, results_dir: Path
) -> Optional[Dict]:
    """OCR one PDF and write its ``.txt`` and ``.json``. Returns the sidecar."""
    console.print()
    console.rule(f"[bold]📄 {pdf_path.name}[/]")
    size_mb = pdf_path.stat().st_size / 1e6
    console.print(f"  [dim]Size:[/] {size_mb:,.1f} MB")

    started = time.time()
    result = client.process_pdf(pdf_path)
    elapsed = time.time() - started

    for warning in result.warnings:
        console.print(f"  [yellow]⚠[/] {warning}")

    blocks = classify_blocks(result.pages)
    text = render_plain_text(blocks)

    if not text.strip():
        console.print("  [red]✗[/] OCR produced no text — nothing written")
        logging.error("No text extracted from %s", pdf_path.name)
        return None

    sidecar = build_sidecar(pdf_path, result, blocks, text, elapsed)

    (results_dir / f"{pdf_path.stem}.txt").write_text(text, encoding="utf-8")
    (results_dir / f"{pdf_path.stem}.json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    roles = sidecar["block_roles"]
    console.print(
        f"  [green]✓[/] {result.pages_processed} pages in {elapsed:,.1f}s — "
        f"{len(text):,} characters"
    )
    console.print(
        f"  [dim]Blocks:[/] {roles.get('body', 0)} body · "
        f"[cyan]{roles.get('apparatus', 0)} apparatus[/] · "
        f"{roles.get('furniture', 0)} furniture dropped"
    )
    logging.info(
        "%s: %d pages, %d chars, roles=%s",
        pdf_path.name, result.pages_processed, len(text), roles,
    )
    return sidecar


def _block_table(sidecars: List[Dict]) -> Table:
    """Aggregate what the structural labels actually found across the run."""
    totals: Counter = Counter()
    for sidecar in sidecars:
        for page in sidecar["pages"]:
            for block in page["blocks"]:
                totals[(block["type"], block["role"])] += len(block["content"] or "")

    table = Table(title="Structure found", box=box.ROUNDED)
    table.add_column("Block type", style="cyan")
    table.add_column("Role", style="dim")
    table.add_column("Characters", justify="right")
    for (btype, role), chars in sorted(totals.items(), key=lambda kv: -kv[1]):
        style = "green" if role == "apparatus" else None
        table.add_row(btype, role, f"{chars:,}", style=style)
    return table


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Structured OCR for scholarly PDFs via Mistral Document AI.",
    )
    parser.add_argument(
        "--item-id", type=int, action="append", dest="item_ids",
        help="Process only these item ids (matched against <id>.pdf). Repeatable.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N PDFs.")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process documents that already have results.",
    )
    parser.add_argument("--rpm", type=int, default=None, help="Requests per minute limit.")
    parser.add_argument(
        "--model", default=MISTRAL_OCR_MODEL,
        help=f"OCR model id (default: pinned {MISTRAL_OCR_MODEL}).",
    )
    parser.add_argument(
        "--pdf-dir", type=Path, default=PDF_DIR, help=f"Source directory (default: {PDF_DIR})."
    )
    args = parser.parse_args()

    console.print(Panel(
        "[bold]Structured OCR for scholarly publications[/]\n"
        "Mistral Document AI with block extraction — body, footnotes and\n"
        "bibliography separated into a JSON sidecar beside the plain text.",
        title="📚 Publication Blocks Processor",
        border_style="cyan",
    ))
    console.print()

    load_dotenv()
    import os

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        console.print("[red]✗[/] MISTRAL_API_KEY not found in the environment!")
        return 1

    RESULTS_DIR.mkdir(exist_ok=True)
    pdf_files = sorted(args.pdf_dir.glob("*.pdf"))
    if args.item_ids:
        wanted = {str(i) for i in args.item_ids}
        pdf_files = [p for p in pdf_files if p.stem.split("_")[0] in wanted]
    if not args.force:
        pdf_files = [p for p in pdf_files if not (RESULTS_DIR / f"{p.stem}.json").exists()]
    if args.limit:
        pdf_files = pdf_files[: args.limit]

    console.print(key_value_table([
        ("Model", args.model),
        ("Source", str(args.pdf_dir)),
        ("Output", str(RESULTS_DIR)),
        ("Documents", len(pdf_files)),
        ("Rate limit", f"{args.rpm} rpm" if args.rpm else "none (paid tier)"),
    ]))
    console.print()

    if not pdf_files:
        console.print(
            "[yellow]⚠[/] Nothing to process. "
            "Run 01_fetch_publication_pdfs.py first, or pass --force to redo."
        )
        return 0

    client = MistralOcrClient(
        api_key, model=args.model, requests_per_minute=args.rpm,
        logger=logging.getLogger(__name__),
    )

    sidecars: List[Dict] = []
    failed = 0
    quota_hit = False
    started = time.time()

    for pdf_path in pdf_files:
        try:
            sidecar = process_one(client, pdf_path, RESULTS_DIR)
            if sidecar:
                sidecars.append(sidecar)
            else:
                failed += 1
        except QuotaExhaustedError as exc:
            console.print(f"\n[red]✗ Quota exhausted[/] — stopping. {exc}")
            logging.error("Quota exhausted on %s: %s", pdf_path.name, exc)
            quota_hit = True
            break
        except Exception as exc:
            console.print(f"  [red]✗[/] {pdf_path.name} failed: {exc}")
            logging.error("Failed on %s: %s", pdf_path.name, exc, exc_info=True)
            failed += 1

    console.print()
    console.rule("[bold cyan]Summary")
    total_pages = sum(s["pages_processed"] for s in sidecars)
    console.print(key_value_table([
        ("Documents processed", len(sidecars)),
        ("Failed", failed),
        ("Pages", f"{total_pages:,}"),
        ("Characters", f"{sum(s['text_characters'] for s in sidecars):,}"),
        ("Billed (approx.)", f"~${total_pages * 4 / 1000:,.2f}"),
        ("Elapsed", f"{time.time() - started:,.1f}s"),
    ], title="Run", value_style="cyan"))

    if sidecars:
        console.print()
        console.print(_block_table(sidecars))
        console.print(
            f"\n[green]✓[/] Results in [cyan]{RESULTS_DIR}[/] "
            "([dim].txt for step 03, .json for the citation pass[/])"
        )
        console.print("[dim]Next: python 03_omeka_content_updater.py --dry-run[/]")

    if quota_hit:
        return 1
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
