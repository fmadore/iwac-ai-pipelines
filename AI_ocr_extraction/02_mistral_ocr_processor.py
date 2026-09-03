"""
AI-Powered PDF OCR using Mistral Document AI (OCR 4.1)

Sends each PDF to Mistral's dedicated OCR endpoint and saves clean plain text —
one ``.txt`` per PDF, named by Omeka item id — for the downstream content
updater (``03_omeka_content_updater.py``).

Unlike a chat model, the OCR endpoint ingests a whole multi-page PDF at once
and returns one entry per page, so there is no page splitting or base64
encoding here. Transport — upload, signed URL, retry, quota handling, the
50 MB split — lives in ``common/mistral_ocr.py`` and is shared with
``AI_publication_extraction``; what differs between the two pipelines is only
what they keep of each page:

* Here (newspaper articles), running heads and footers are dropped from page 2
  onwards and kept on page 1, where they hold the byline and citation rather
  than a repeated header.
* ``AI_publication_extraction`` keeps footers unless they are folio numbers or
  repeat across the document, because on scholarly PDFs a page foot is
  overwhelmingly a footnote.

The model id is the pinned release from ``common/mistral_ocr.py`` — never the
rolling ``mistral-ocr-latest`` alias — because step 03 stamps it into an
``iwac:ocrModel`` annotation, and a run that cannot name its model cannot be
cited.

Usage:
    python 02_mistral_ocr_processor.py
    python 02_mistral_ocr_processor.py --rpm 30   # optional proactive throttling

Requirements:
    - MISTRAL_API_KEY in the environment / .env
    - PDFs in the PDF/ directory (named <item_id>.pdf by 01_omeka_pdf_downloader.py)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.console_utils import count_table, key_value_table, print_file_table, standard_progress
from common.log_redaction import install_credential_redaction
from common.mistral_ocr import MISTRAL_OCR_MODEL, MistralOcrClient, markdown_to_plain_text
from common.rate_limiter import QuotaExhaustedError

console = Console()

SCRIPT_DIR = Path(__file__).resolve().parent
PDF_DIR = SCRIPT_DIR / "PDF"
RESULTS_DIR = SCRIPT_DIR / "OCR_Results"
LOG_DIR = SCRIPT_DIR / "log"


def render_pages(pages: Sequence[Dict[str, Any]]) -> str:
    """Join OCR pages into the plain text written to ``bibo:content``.

    Page 1 keeps its header and footer — on a newspaper scan they carry the
    byline and the journal citation. Later pages drop them, since there they
    are the running head and the folio number. Page markers follow the
    convention the Gemini path established: none on the first page,
    ``--- Page N ---`` before every other one. A page with no text contributes
    nothing rather than a placeholder — a placeholder would be uploaded to
    the archive as if it were content.
    """
    rendered: List[str] = []
    for position, page in enumerate(pages, start=1):
        body = page.get("markdown") or ""
        if position == 1:
            head = (page.get("header") or "").strip()
            foot = (page.get("footer") or "").strip()
            body = "\n\n".join(part for part in (head, body, foot) if part)
        text = markdown_to_plain_text(body)
        if not text:
            continue
        rendered.append(text if position == 1 else f"\n\n--- Page {position} ---\n\n{text}")
    return "".join(rendered).strip()


def process_pdf(client: MistralOcrClient, pdf_path: Path, output_dir: Path) -> bool:
    """OCR one PDF and write ``<stem>.txt``. Returns True when text was written.

    Raises:
        QuotaExhaustedError: propagated so the batch stops instead of burning
            retries on every remaining file.
    """
    console.print()
    console.rule(f"[bold]📄 {pdf_path.name}[/]")
    console.print(f"  [dim]Size:[/] {pdf_path.stat().st_size / (1024 * 1024):.2f} MB")

    result = client.process_pdf(pdf_path)
    for warning in result.warnings:
        console.print(f"  [yellow]⚠[/] {warning}")

    text = render_pages(result.pages)
    if not text:
        console.print("  [red]✗[/] OCR produced no text — nothing written")
        logging.error("No text extracted from %s", pdf_path.name)
        return False

    output_file = output_dir / f"{pdf_path.stem}.txt"
    output_file.write_text(text, encoding="utf-8")
    empty_pages = sum(1 for page in result.pages if not (page.get("markdown") or "").strip())
    console.print(
        f"  [green]✓[/] {result.pages_processed} pages"
        + (f" ({empty_pages} without text)" if empty_pages else "")
    )
    console.print(f"  [dim]Output:[/] {output_file.name} ({len(text):,} characters)")
    logging.info("%s: %d pages (%d empty)", pdf_path.name, result.pages_processed, empty_pages)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OCR every PDF in PDF/ through Mistral Document AI into OCR_Results/.",
    )
    parser.add_argument(
        "--rpm", type=int, default=None,
        help="Requests per minute to space calls at (default: none — paid tier).",
    )
    return parser


def main() -> int:
    """Batch-process every PDF in PDF/ through Mistral OCR."""
    args = build_parser().parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=LOG_DIR / "ocr_mistral.log",
    )
    # Credentials ride in Omeka query strings and provider headers; keep them
    # out of anything urllib3 or an SDK decides to log.
    install_credential_redaction()

    console.print(Panel(
        "[bold]AI-Powered PDF OCR using Mistral Document AI[/bold]\n"
        "Dedicated OCR endpoint — one call per PDF, plain-text output",
        title="📄 Mistral OCR Processor",
        border_style="cyan",
    ))

    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        console.print("[red]✗[/] MISTRAL_API_KEY not found in environment variables!")
        return 1

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    RESULTS_DIR.mkdir(exist_ok=True)
    console.print(key_value_table([
        ("Model", MISTRAL_OCR_MODEL),
        ("Input", str(PDF_DIR)),
        ("Output", str(RESULTS_DIR)),
        ("Documents", len(pdf_files)),
        ("Rate limit", f"{args.rpm} rpm" if args.rpm else "none (paid tier)"),
    ], title="Configuration"))

    if not pdf_files:
        console.print("\n[red]✗[/] No PDF files found in the PDF directory!")
        return 1
    console.print()
    print_file_table(console, pdf_files, title=f"📚 PDF Files to Process ({len(pdf_files)})")

    client = MistralOcrClient(
        api_key, requests_per_minute=args.rpm, logger=logging.getLogger(__name__),
    )

    processed = failed = 0
    quota_hit = False
    start = time.time()
    console.print()
    console.rule("[bold cyan]📄 Processing PDFs[/]")
    with standard_progress(console) as progress:
        task = progress.add_task("[cyan]Processing PDFs...", total=len(pdf_files))
        for pdf_path in pdf_files:
            progress.update(task, description=f"[cyan]Processing {pdf_path.name}...")
            try:
                if process_pdf(client, pdf_path, RESULTS_DIR):
                    processed += 1
                else:
                    failed += 1
            except QuotaExhaustedError as exc:
                # Results for already-completed PDFs are on disk; stop here.
                failed += 1
                quota_hit = True
                console.print(Panel(
                    "[red bold]Mistral API quota exhausted — stopping all processing.[/]\n"
                    "Results for already-processed PDFs have been saved.\n"
                    "Wait for your quota to reset or upgrade your plan.",
                    title="Quota Exhausted",
                    border_style="red",
                ))
                logging.error("Quota exhausted on %s: %s", pdf_path.name, exc)
                progress.update(task, advance=1)
                break
            except Exception as exc:
                failed += 1
                console.print(f"[red]✗[/] Failed to process {pdf_path.name}: {exc}")
                logging.error("Failed to process %s: %s", pdf_path.name, exc, exc_info=True)
            progress.update(task, advance=1)

    elapsed_minutes = (time.time() - start) / 60
    console.print()
    console.print(count_table([
        ("Total PDFs", len(pdf_files)),
        ("Successful", processed),
        ("Failed", failed),
        ("Processing time", f"{elapsed_minutes:.1f} minutes"),
    ], title="📈 Processing Summary"))
    logging.info("Processing complete. %d/%d PDFs in %.1f min", processed, len(pdf_files), elapsed_minutes)
    if processed:
        console.print("[dim]Next: python 03_omeka_content_updater.py --dry-run --model mistral-ocr-4-1[/]")

    if quota_hit:
        return 1
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠[/] Process interrupted by user")
        sys.exit(130)
