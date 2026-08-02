"""Islamic Magazine Article Extraction Pipeline using Mistral Document AI

This script implements a two-step pipeline to extract and consolidate articles
from an Islamic magazine using Mistral's native PDF understanding with OCR.

Supported stages:
- Mistral OCR (Step 1: page-by-page extraction specialized for documents)
- Repository default text model (Step 2: consolidation with structured outputs)

Step 1: Page-by-page extraction with OCR
- Uploads PDF once to Mistral
- Analyzes each page individually with Mistral OCR
- Uses document_annotation_format so OCR and structured extraction happen in a
  single request per page, rather than ocr.process followed by chat.parse
- Extracts exact titles and generates brief summaries
- Detects continuation indicators

Step 2: Magazine-level consolidation
- Merges articles fragmented across multiple pages
- Uses structured outputs for consistent article index format
- Produces a global summary per article
- Lists all associated pages

Output formats:
- JSON files for programmatic access (step1_consolidated.json, final_index.json)
- Markdown files for human readability (step1_consolidated.md, final_index.md)

Robustness mechanisms:
- Automatic retry on error with exponential backoff (max 3 attempts)
- Progressive result saving with JSON caching (cache kept until step 2 succeeds)
- Resumption possible from already processed files
- Quota-aware error handling: quota/billing exhaustion (and persistent 429s)
  stop the whole batch; optional --rpm throttling for rate limits
- Uploaded files are always cleaned up from the Mistral cloud, even on failure

API Best Practices:
- Uses client.chat.parse() with Pydantic response_format for structured outputs
- Uses Mistral OCR for document processing
- Uses the repository default text model for cost-effective consolidation

Usage:
    python 02_Mistral_generate_summaries_issue.py
    python 02_Mistral_generate_summaries_issue.py --rpm 5

Requirements:
    - MISTRAL_API_KEY environment variable
    - mistralai Python package
    - PDF files in PDF/ directory
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Rich console output helpers (the console itself is shared with the pipeline module)
from rich.panel import Panel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.pdf_utils import get_pdf_page_count  # noqa: E402
from common.llm_provider import DEFAULT_TEXT_MODEL_KEY, ModelOption, get_model_option  # noqa: E402
from common.rate_limiter import RateLimiter, QuotaExhaustedError, is_mistral_quota_exhausted  # noqa: E402
from common.retry import retry_with_backoff  # noqa: E402

# Shared magazine-extraction building blocks (models, prompts, step skeletons)
from magazine_extraction import (  # noqa: E402
    PageExtraction,
    build_text_consolidator,
    console,
    load_extraction_prompt,
    run_extraction_pipeline,
    run_magazine_batch,
)

try:
    from mistralai.client import Mistral
    from mistralai.extra import response_format_from_pydantic_model
except ImportError as exc:
    raise RuntimeError("mistralai package is required. Install with: pip install mistralai") from exc

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds (exponential backoff via common.retry)

# Mistral multimodal extraction model
MISTRAL_OCR = "mistral-ocr-latest"  # For OCR API endpoint
MISTRAL_OCR_TIMEOUT_MS = 600_000

# ------------------------------------------------------------------
# Error Helpers
# ------------------------------------------------------------------
def _http_status(error: Exception) -> Optional[int]:
    """Best-effort HTTP status code from a Mistral SDK exception."""
    return getattr(error, "status_code", None) or getattr(error, "code", None)


def _raise_for_quota(error: Exception) -> None:
    """Convert an unambiguous quota/billing error into QuotaExhaustedError."""
    if is_mistral_quota_exhausted(error):
        raise QuotaExhaustedError(str(error)) from error


def _raise_if_persistent_429(error: Exception) -> None:
    """After retries are exhausted, treat a still-429 error as quota exhaustion.

    A 429 that survives exponential backoff means the account is being
    throttled hard enough that continuing the batch is pointless.
    """
    if _http_status(error) == 429:
        raise QuotaExhaustedError(
            f"Persistent rate limiting (429) after {MAX_RETRIES} attempts: {error}"
        ) from error

# ------------------------------------------------------------------
# PDF Processing with Mistral
# ------------------------------------------------------------------
def upload_pdf_to_mistral(client: Mistral, pdf_path: Path) -> tuple[str, str]:
    """
    Upload a PDF file to Mistral and get the file ID and signed URL.

    Args:
        client: Mistral client
        pdf_path: Path to the PDF file

    Returns:
        Tuple of (file_id, signed_url) for use in API calls
    """
    try:
        console.print(f"\n[cyan]⬆ Uploading PDF to Mistral:[/] {pdf_path.name}")
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        console.print(f"[dim]📊 PDF size: {file_size_mb:.2f} MB[/]")

        # Upload file
        with open(pdf_path, 'rb') as f:
            uploaded_file = client.files.upload(
                file={
                    "file_name": pdf_path.name,
                    "content": f,
                },
                purpose="ocr"
            )

        file_id = uploaded_file.id
        console.print(f"[green]✓[/] PDF uploaded: [dim]{file_id}[/]")
        logging.info(f"Uploaded PDF {pdf_path.name} with ID: {file_id}")

        # Get signed URL for OCR access
        signed_url_obj = client.files.get_signed_url(file_id=file_id)
        signed_url = signed_url_obj.url
        console.print("[green]✓[/] Signed URL obtained")
        logging.info(f"Got signed URL for file {file_id}")

        return file_id, signed_url

    except Exception as e:
        logging.error(f"Error uploading PDF to Mistral: {e}")
        raise

# ------------------------------------------------------------------
# AI Generation Functions with Retry
# ------------------------------------------------------------------
@retry_with_backoff(max_retries=MAX_RETRIES, base_delay=RETRY_DELAY)
def generate_page_extraction_mistral(client: Mistral, signed_url: str, page_num: int,
                                    prompt: str, rate_limiter: RateLimiter) -> Optional[PageExtraction]:
    """
    Generate structured page extraction with the Mistral OCR API.

    Uses ``document_annotation_format`` so OCR and structured extraction happen
    in a SINGLE request. This used to be two: ``ocr.process`` to get markdown,
    then ``chat.parse`` to turn that markdown into a PageExtraction — which
    doubled both the latency and the rate-limiter pressure per page, and threw
    away the layout information the OCR model had by flattening to text first.

    Args:
        client: Mistral client
        signed_url: Signed URL for the uploaded PDF
        page_num: Page number to analyze (1-indexed for display, 0-indexed for API)
        prompt: Extraction prompt, passed as the annotation prompt
        rate_limiter: Shared rate limiter (wait() called before the API request)

    Returns:
        PageExtraction object or None

    Raises:
        QuotaExhaustedError: When the API quota/billing limit is exhausted
            (never retried by the decorator).
    """
    try:
        rate_limiter.wait()
        response = client.ocr.process(
            model=MISTRAL_OCR,
            pages=[page_num - 1],  # API uses 0-indexed pages
            document={
                "type": "document_url",
                "document_url": signed_url,
            },
            document_annotation_format=response_format_from_pydantic_model(PageExtraction),
            document_annotation_prompt=prompt,
            include_image_base64=False,
        )

        if not response:
            raise RuntimeError("Invalid OCR response structure")

        annotation = getattr(response, "document_annotation", None)
        if not annotation:
            raise RuntimeError(f"No document annotation returned for page {page_num}")

        # The annotation comes back as a JSON string matching the schema.
        parsed = (
            annotation
            if isinstance(annotation, PageExtraction)
            else PageExtraction.model_validate_json(annotation)
            if isinstance(annotation, str)
            else PageExtraction.model_validate(annotation)
        )

        logging.info(f"Page {page_num}: extracted {len(parsed.articles)} article(s)")

        # Ensure page number is set correctly
        parsed.page_number = page_num
        return parsed

    except QuotaExhaustedError:
        raise
    except Exception as e:
        _raise_for_quota(e)
        logging.error(f"Generation error for page {page_num}: {e}")
        raise

# ------------------------------------------------------------------
# Main Pipeline
# ------------------------------------------------------------------
def _validate_pdf(pdf_path: Path) -> None:
    """Reject missing or non-PDF inputs before creating provider state."""
    if not pdf_path.is_file():
        raise ValueError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Input must be a PDF file, got: {pdf_path}")


def _mistral_client_from_env() -> Mistral:
    """Build the OCR client with a finite request deadline."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not found in environment variables")
    return Mistral(api_key=api_key, timeout_ms=MISTRAL_OCR_TIMEOUT_MS)


def _delete_mistral_upload(client: Mistral, file_id: Optional[str]) -> None:
    """Best-effort cleanup of a temporary Mistral cloud upload."""
    if file_id is None:
        return
    try:
        client.files.delete(file_id=file_id)
        console.print("[dim]🗑 Uploaded file deleted from Mistral cloud[/]")
        logging.info(f"Deleted file {file_id} from Mistral cloud")
    except Exception as exc:
        logging.warning(f"Failed to delete file {file_id}: {exc}")


def _run_uploaded_magazine(
    client: Mistral,
    signed_url: str,
    pdf_path: Path,
    output_dir: Path,
    magazine_id: str,
    extraction_prompt: str,
    model_step2: ModelOption,
    rate_limiter: RateLimiter,
) -> Path:
    """Run both extraction stages after the PDF upload is ready."""
    with console.status("[cyan]Reading PDF structure...", spinner="dots"):
        total_pages = get_pdf_page_count(pdf_path)
    console.print(f"[green]✓[/] PDF has [bold]{total_pages}[/] pages")
    console.print("[dim]📄 Will process each page with Mistral Document AI[/]")
    output_dir.mkdir(parents=True, exist_ok=True)

    def extract_page(page_num: int) -> Optional[PageExtraction]:
        try:
            return generate_page_extraction_mistral(
                client, signed_url, page_num, extraction_prompt, rate_limiter
            )
        except QuotaExhaustedError:
            raise
        except Exception as exc:
            _raise_if_persistent_429(exc)
            raise

    final_file = run_extraction_pipeline(
        extract_page=extract_page,
        consolidate=build_text_consolidator(model_step2),
        total_pages=total_pages,
        output_dir=output_dir,
        magazine_id=magazine_id,
        step1_model_label="Mistral OCR",
        step2_model_label=model_step2.label,
        schema_note="Pydantic schema",
    )
    logging.info(f"Pipeline complete for magazine {magazine_id}")
    logging.info(f"Final index: {final_file}")
    return final_file


def process_magazine(model_step2: ModelOption, pdf_path: Path, output_dir: Path, magazine_id: str = None,
                     rate_limiter: Optional[RateLimiter] = None):
    """
    Complete pipeline to process a magazine PDF using Mistral's Document AI.

    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory
        magazine_id: Magazine identifier (optional)
        rate_limiter: Shared RateLimiter (None = no throttling)
    """
    magazine_id = magazine_id or pdf_path.stem
    rate_limiter = rate_limiter or RateLimiter(requests_per_minute=None)

    logging.info(f"Processing magazine: {magazine_id}")
    logging.info(f"Input: {pdf_path}")
    logging.info(f"Output: {output_dir}")

    _validate_pdf(pdf_path)
    client = _mistral_client_from_env()
    extraction_prompt = load_extraction_prompt()

    file_id = None
    try:
        file_id, signed_url = upload_pdf_to_mistral(client, pdf_path)
        return _run_uploaded_magazine(
            client,
            signed_url,
            pdf_path,
            output_dir,
            magazine_id,
            extraction_prompt,
            model_step2,
            rate_limiter,
        )
    except QuotaExhaustedError:
        raise  # let main() stop the whole batch
    except Exception as e:
        logging.error(f"Error processing PDF {pdf_path}: {e}")
        raise
    finally:
        _delete_mistral_upload(client, file_id)

# ------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------
def main() -> int:
    """Main entry point of the script."""
    try:
        load_dotenv()

        parser = argparse.ArgumentParser(
            description="Islamic magazine article extraction to a table of contents (Mistral)."
        )
        parser.add_argument(
            "--rpm", type=int, default=None,
            help="Requests per minute limit (e.g. 5 for free tier). "
                 "Omit for no throttling (paid tiers).",
        )
        args = parser.parse_args()

        intro_panel = Panel(
            "[bold cyan]Islamic Magazine Article Extraction Pipeline (Mistral)[/]\n\n"
            "[dim]Using Mistral Document AI with OCR capabilities[/]\n\n"
            "📖 [white]Step 1:[/] Page-by-page extraction [dim](Mistral OCR)[/]\n"
            f"📊 [white]Step 2:[/] Magazine-level consolidation "
            f"[dim]({DEFAULT_TEXT_MODEL_KEY})[/]",
            title="🚀 Pipeline Started", border_style="cyan", padding=(1, 2),
        )

        logging.info("=== Magazine Article Extraction Pipeline (Mistral) ===")
        model_step2 = get_model_option(DEFAULT_TEXT_MODEL_KEY)

        # Rate limiter shared across the whole batch (None = no throttling)
        rate_limiter = RateLimiter(requests_per_minute=args.rpm)
        if args.rpm:
            console.print(f"[cyan]⏱[/] Rate limiting: {args.rpm} requests/minute")

        return run_magazine_batch(
            lambda pdf_path, output_dir, magazine_id: process_magazine(
                model_step2, pdf_path, output_dir, magazine_id,
                rate_limiter=rate_limiter,
            ),
            script_dir=Path(__file__).resolve().parent,
            intro_panel=intro_panel,
            api_key_env=("MISTRAL_API_KEY", "OPENROUTER_API_KEY"),
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠[/] Process interrupted by user")
        logging.info("Process interrupted by user")
        return 1
    except Exception as e:
        console.print(f"\n[red]✗ Pipeline failed:[/] {e}")
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    sys.exit(main())
