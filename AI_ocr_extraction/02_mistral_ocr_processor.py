"""
AI-Powered PDF OCR using Mistral OCR 4 (Document AI)

Sends each PDF to Mistral's dedicated OCR endpoint in a single call and saves
clean plain text — one ``.txt`` per PDF, named by Omeka item id — for the
downstream content updater (``03_omeka_content_updater.py``).

The endpoint is addressed as ``mistral-ocr-latest``, which resolves server-side
to the newest OCR release (OCR 4 as of June 2026). Unlike a chat model, the OCR
endpoint ingests a whole multi-page PDF at once and returns a ``pages[]`` array
of Markdown — so there is no need to split, count, or base64-encode pages.

Pipeline per PDF:
    1. Upload the PDF (Files API) and get a short-lived signed URL.
    2. Run OCR in one ``client.ocr.process`` call, with header/footer
       extraction so repeated running heads (page numbers, author/title) are
       split out of the body — dropped on pages 2+, kept on the first page
       where they hold real front matter (byline, journal citation).
    3. Strip Mistral's Markdown to plain text (matching the Gemini OCR path,
       which the project standardises on for ``bibo:content``).
    4. Join pages with ``--- Page N ---`` markers and write ``<stem>.txt``.

Usage:
    python 02_mistral_ocr_processor.py
    python 02_mistral_ocr_processor.py --rpm 30   # optional proactive throttling

Requirements:
    - MISTRAL_API_KEY in the environment / .env
    - PDFs in the PDF/ directory (named <item_id>.pdf by 01_omeka_pdf_downloader.py)
    - mistralai package
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Rich console output (consistent with the rest of the pipeline)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Shared proactive throttler + quota handling (provider-agnostic; only spaces
# requests when --rpm is set) and the shared exponential-backoff retry decorator.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.rate_limiter import RateLimiter, QuotaExhaustedError, is_mistral_quota_exhausted
from common.retry import retry_with_backoff
from common.console_utils import standard_progress
from common.mistral_ocr import markdown_to_plain_text
from common.log_redaction import install_credential_redaction

try:
    from mistralai.client import Mistral
    from mistralai.client.errors import SDKError
except ImportError as exc:
    raise RuntimeError("mistralai package is required. Install with: pip install mistralai") from exc

console = Console()

# Set up logging configuration for tracking OCR operations and errors
script_dir = Path(__file__).parent
log_dir = script_dir / "log"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "ocr_mistral.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=log_file,
)
# Credentials ride in Omeka query strings and provider headers; keep them
# out of anything urllib3 or an SDK decides to log.
install_credential_redaction()

# `-latest` resolves to the newest OCR model server-side (OCR 4 as of 2026-06).
MISTRAL_OCR_MODEL = "mistral-ocr-latest"

# HTTP statuses worth retrying (transient rate limit / server errors).
TRANSIENT_STATUS = {429, 500, 502, 503, 504}


def _status_code(error: Exception) -> Optional[int]:
    """Best-effort extraction of an HTTP status code from a Mistral SDK error.

    Mistral's ``SDKError`` exposes the response as ``raw_response`` (an
    httpx.Response), so the status lives at ``error.raw_response.status_code``
    rather than on a ``.code`` attribute.
    """
    resp = getattr(error, "raw_response", None)
    code = getattr(resp, "status_code", None)
    if code is None:
        code = getattr(error, "status_code", None)
    return code


def _is_retryable(error: Exception) -> bool:
    """Predicate for ``retry_with_backoff``: retry transient errors only.

    Quota exhaustion is never retried — the account quota is dead, so it is
    surfaced immediately (and converted to ``QuotaExhaustedError`` by the
    caller). SDK errors are retried only for transient HTTP statuses;
    unexpected errors (network hiccups, parsing) are always retried.
    """
    if is_mistral_quota_exhausted(error):
        return False
    if isinstance(error, SDKError):
        return _status_code(error) in TRANSIENT_STATUS
    return True


# Markdown -> plain text normalisation now lives in ``common/mistral_ocr.py``,
# shared with ``AI_publication_extraction``, which needs the identical rules so
# that ``bibo:content`` reads the same whichever pipeline produced it.


class MistralOCRProcessor:
    """OCR pipeline backed by Mistral's dedicated Document AI endpoint.

    One ``client.ocr.process`` call handles an entire multi-page PDF; the
    response carries one Markdown entry per page, which we normalise to plain
    text before saving.
    """

    def __init__(self, api_key: str, requests_per_minute: Optional[int] = None):
        self.client = Mistral(api_key=api_key)
        self.model = MISTRAL_OCR_MODEL
        self.rate_limiter = RateLimiter(requests_per_minute, logger=logging.getLogger(__name__))

    def _ocr_document(self, pdf_path: Path):
        """Upload a PDF and run OCR, returning the list of page objects (or None).

        Retries transient HTTP errors (429 / 5xx) with exponential backoff and
        jitter via ``common.retry.retry_with_backoff``. The uploaded file is
        deleted from Mistral storage afterwards.

        Raises:
            QuotaExhaustedError: When the account quota is exhausted, or a 429
                still persists after all retries — the whole batch should stop.
        """
        uploaded_id: Optional[str] = None
        signed_url: Optional[str] = None

        @retry_with_backoff(max_retries=3, base_delay=5.0, is_retryable=_is_retryable)
        def _run_ocr():
            nonlocal uploaded_id, signed_url

            # Upload once; reuse the signed URL across retries.
            if uploaded_id is None:
                uploaded = self.client.files.upload(
                    file={"file_name": pdf_path.name, "content": pdf_path.read_bytes()},
                    purpose="ocr",
                )
                if not uploaded or not uploaded.id:
                    raise RuntimeError("File upload returned no id")
                uploaded_id = uploaded.id
                signed_url = self.client.files.get_signed_url(file_id=uploaded_id).url

            self.rate_limiter.wait()
            response = self.client.ocr.process(
                model=self.model,
                document={"type": "document_url", "document_url": signed_url},
                extract_header=True,   # split running heads/footers out of
                extract_footer=True,   # page.markdown (see process_pdf)
            )
            if not response.pages:
                raise RuntimeError("OCR response contained no pages")
            return response.pages

        try:
            return _run_ocr()

        except SDKError as e:
            code = _status_code(e)
            if is_mistral_quota_exhausted(e) or code == 429:
                # Account quota dead, or a 429 that survived every retry:
                # further PDFs would only burn retries — stop the batch.
                console.print("  [red]✗ Quota exhausted[/] — stopping the batch")
                logging.error("Mistral quota exhausted on %s (HTTP %s): %s", pdf_path.name, code, e)
                raise QuotaExhaustedError(str(e)) from e
            console.print(f"  [yellow]⚠[/] Mistral API error (HTTP {code}) — skipping this PDF")
            logging.error("Mistral OCR error for %s (HTTP %s): %s", pdf_path.name, code, e)
            return None

        except Exception as e:
            console.print(f"  [red]✗[/] OCR failed after retries: {e}")
            logging.error("Unexpected OCR error for %s: %s", pdf_path.name, e, exc_info=True)
            return None

        finally:
            # Best-effort cleanup so uploads don't accumulate in Mistral storage.
            if uploaded_id is not None:
                try:
                    self.client.files.delete(file_id=uploaded_id)
                except Exception:
                    pass

    def process_pdf(self, pdf_path: Path, output_dir: Path) -> bool:
        """OCR a single PDF and write ``<stem>.txt``. Returns True on success."""
        console.print()
        console.rule(f"[bold]📄 {pdf_path.name}[/]")

        if not pdf_path.exists():
            console.print(f"[red]✗[/] PDF file not found: {pdf_path}")
            logging.error("PDF file not found: %s", pdf_path)
            return False

        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        console.print(f"  [dim]Size:[/] {size_mb:.2f} MB")

        pages = self._ocr_document(pdf_path)
        if not pages:
            console.print("  [red]✗[/] OCR failed — no output file created")
            return False

        # Normalise each page to plain text and join with page markers.
        # Running heads/footers were extracted into page.header/.footer and are
        # dropped — except on the first page, where they hold front matter
        # (byline, journal citation) rather than a repeated running head.
        empty_pages = 0
        rendered: List[str] = []
        for i, page in enumerate(pages, 1):
            body = getattr(page, "markdown", "") or ""
            if i == 1:
                head = (getattr(page, "header", "") or "").strip()
                foot = (getattr(page, "footer", "") or "").strip()
                body = "\n\n".join(s for s in (head, body, foot) if s)
            text = markdown_to_plain_text(body)
            if not text:
                text = f"[Empty page {i}]"
                empty_pages += 1
            rendered.append(text if i == 1 else f"\n\n--- Page {i} ---\n\n{text}")

        output_file = output_dir / f"{pdf_path.stem}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("".join(rendered))

        output_size = output_file.stat().st_size
        total = len(pages)
        if empty_pages:
            console.print(
                f"  [yellow]⚠[/] {total - empty_pages}/{total} pages with text "
                f"({empty_pages} empty)"
            )
        else:
            console.print(f"  [green]✓[/] {total} pages")
        console.print(f"  [dim]Output:[/] {output_file.name} ({output_size:,} bytes)")
        logging.info("PDF %s: %d pages (%d empty)", pdf_path.name, total, empty_pages)
        return True


def main():
    """Batch-process every PDF in PDF/ through Mistral OCR."""
    console.print(
        Panel(
            "[bold]AI-Powered PDF OCR using Mistral OCR 4[/bold]\n"
            "Dedicated Document AI endpoint — one call per PDF, plain-text output",
            title="📄 Mistral OCR Processor",
            border_style="cyan",
        )
    )

    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        console.print("[red]✗[/] MISTRAL_API_KEY not found in environment variables!")
        return
    console.print("[green]✓[/] API Key loaded successfully")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rpm", type=int, default=None, help="Requests per minute limit")
    known_args, _ = parser.parse_known_args()

    # Set up directory paths
    pdf_dir = script_dir / "PDF"
    output_dir = script_dir / "OCR_Results"
    output_dir.mkdir(exist_ok=True)

    # Display configuration
    config_table = Table(title="Configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Model", f"{MISTRAL_OCR_MODEL} (OCR 4)")
    config_table.add_row("Output format", "Plain text")
    config_table.add_row("Rate limit", f"{known_args.rpm} rpm" if known_args.rpm else "none (paid tier)")
    console.print(config_table)

    processor = MistralOCRProcessor(api_key, requests_per_minute=known_args.rpm)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        console.print("\n[red]✗[/] No PDF files found in the PDF directory!")
        return

    total_pdfs = len(pdf_files)
    files_table = Table(title=f"📚 PDF Files to Process ({total_pdfs})", box=box.ROUNDED)
    files_table.add_column("#", style="dim", width=4)
    files_table.add_column("Filename", style="cyan")
    files_table.add_column("Size", justify="right", style="green")
    for idx, pdf_file in enumerate(pdf_files, 1):
        files_table.add_row(str(idx), pdf_file.name, f"{pdf_file.stat().st_size / (1024 * 1024):.2f} MB")
    console.print()
    console.print(files_table)

    processed = 0
    failed = 0
    start = time.time()

    console.print()
    console.rule("[bold cyan]📄 Processing PDFs[/]")
    with standard_progress(console) as progress:
        task = progress.add_task("[cyan]Processing PDFs...", total=total_pdfs)
        for pdf_path in pdf_files:
            progress.update(task, description=f"[cyan]Processing {pdf_path.name}...")
            try:
                if processor.process_pdf(pdf_path, output_dir):
                    processed += 1
                else:
                    failed += 1
            except QuotaExhaustedError:
                # Results for already-completed PDFs are on disk; stop here
                # instead of burning retries on every remaining file.
                failed += 1
                console.print(Panel(
                    "[red bold]Mistral API quota exhausted — stopping all processing.[/]\n"
                    "Results for already-processed PDFs have been saved.\n"
                    "Wait for your quota to reset or upgrade your plan.",
                    title="Quota Exhausted",
                    border_style="red",
                ))
                logging.error("Quota exhausted — aborting remaining PDFs.")
                progress.update(task, advance=1)
                break
            except Exception as e:
                failed += 1
                console.print(f"[red]✗[/] Failed to process {pdf_path.name}: {e}")
                logging.error("Failed to process %s: %s", pdf_path.name, e, exc_info=True)
            progress.update(task, advance=1)

    elapsed = time.time() - start
    success_rate = (processed / total_pdfs * 100) if total_pdfs else 0

    summary_table = Table(title="📈 Processing Summary", box=box.ROUNDED)
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Total PDFs", str(total_pdfs))
    summary_table.add_row("[green]Successful[/]", f"[green]{processed}[/]")
    summary_table.add_row("[red]Failed[/]", f"[red]{failed}[/]")
    summary_table.add_row("Processing Time", f"{elapsed / 60:.1f} minutes")
    summary_table.add_row("Success Rate", f"{success_rate:.1f}%")
    console.print()
    console.print(summary_table)
    logging.info("Processing complete. %d/%d PDFs in %.1f min", processed, total_pdfs, elapsed / 60)

    console.print(
        Panel(
            f"[green]✓[/] Processed {processed}/{total_pdfs} PDFs successfully",
            title="✨ OCR Complete",
            border_style="green",
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠[/] Process interrupted by user")
        logging.info("Process interrupted by user")
    except Exception as e:
        console.print(f"\n[red]✗[/] An error occurred: {e}")
        logging.error("An error occurred: %s", e, exc_info=True)
