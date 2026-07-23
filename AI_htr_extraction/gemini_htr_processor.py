"""
AI-Powered HTR (Handwritten Text Recognition) Script using Google Gemini

This script performs high-precision transcription of HANDWRITTEN documents (French, Arabic, or multilingual)
using Google's Gemini vision model with direct PDF processing. It is engineered for research-grade,
archival-quality extraction while preserving correct typography, reading order, and structural semantics.

Usage:
    python gemini_htr_processor.py
    python gemini_htr_processor.py --model gemini-pro-latest --language arabic
    python gemini_htr_processor.py --pdf-dir /path/to/pdfs --rpm 5

Requirements:
    - Environment variable: GEMINI_API_KEY
    - PDF files in the PDF/ directory (or --pdf-dir)
    - HTR system prompt files (htr_system_prompt_french.md, htr_system_prompt_arabic.md, etc.)
"""

import argparse
import os
import random
import time
import logging
from pathlib import Path
from typing import Optional
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from dotenv import load_dotenv

# Add repo root to path for shared imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.gemini_utils import (
    build_generation_config,
    delete_uploaded_file,
    extract_text_from_response,
    get_thinking_level,
    upload_and_wait_active,
)
from common.pdf_utils import extract_pdf_page, get_pdf_page_count
from common.rate_limiter import RateLimiter, QuotaExhaustedError, is_quota_exhausted

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Initialize rich console
console = Console()

# Set up logging configuration for tracking HTR operations and errors
# Save log file in a dedicated log directory
script_dir = Path(__file__).parent
log_dir = script_dir / 'log'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / 'htr_gemini.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file
)

# Transient Gemini API errors worth retrying with backoff. Anything else —
# in particular programming errors — must raise immediately instead of being
# retried (a NameError was once retried with backoff here for months).
RETRYABLE_API_CODES = (429, 500, 503)


class GeminiHTR:
    """
    A high-precision HTR system using Google's Gemini model with native PDF processing.

    This class implements a page-by-page HTR pipeline that:
    1. Extracts individual pages from PDF documents
    2. Sends each page directly to Gemini for processing
    3. Leverages Gemini's native document understanding
    4. Applies sophisticated text extraction and formatting rules
    5. Handles uncertainty and quality control per page

    The system is designed for academic research and archival purposes, with emphasis on:
    - Maintaining precise reading order and layout relationships
    - Preserving language-specific typography and formatting
    - Handling document structure (columns, zones, captions)
    - Processing pages individually for better control and error recovery
    """

    def __init__(self, api_key: str, model_name: str, language: str = "french", requests_per_minute: Optional[int] = None):
        """
        Initialize the GeminiHTR system with API credentials and model name.

        Args:
            api_key (str): Google Gemini API key for authentication
            model_name (str): The Gemini model name to use
            language (str): Language of the manuscripts ("french", "arabic", or "multilingual")
            requests_per_minute: Optional RPM limit for proactive throttling (None = no throttling)
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.language = language
        self.rate_limiter = RateLimiter(requests_per_minute, logger=logging.getLogger(__name__))
        self.system_instruction = self._get_system_instruction()
        self.generation_config = self._setup_generation_config()

    def _setup_generation_config(self):
        """
        Configure generation parameters for optimal HTR performance.

        HIGH media resolution matters for handwriting where fine stroke detail
        and letter disambiguation are critical (ULTRA_HIGH isn't supported at
        the global config level).

        Returns:
            types.GenerateContentConfig: Configured generation config
        """
        console.print(f"[cyan]🧠[/] Using thinking level '{get_thinking_level(self.model_name)}' for {self.model_name}")
        console.print(f"[cyan]🖼[/]  Using media resolution 'HIGH' for {self.model_name}")

        return build_generation_config(
            self.model_name,
            system_instruction=self.system_instruction,
            max_output_tokens=65535,
            media_resolution="HIGH",
        )

    def _get_system_instruction(self):
        """
        Get the specialized system instructions for handwritten text recognition.

        Loads the appropriate system prompt based on the selected language:
        - French: htr_system_prompt_french.md
        - Arabic: htr_system_prompt_arabic.md
        - Multilingual: htr_system_prompt_multilingual.md (auto-detects language)

        Returns:
            str: Detailed system instruction for HTR processing
        """
        # Select the appropriate prompt file based on language
        if self.language == "arabic":
            prompt_file = Path(__file__).parent / "htr_system_prompt_arabic.md"
        elif self.language == "multilingual":
            prompt_file = Path(__file__).parent / "htr_system_prompt_multilingual.md"
        else:  # default to french
            prompt_file = Path(__file__).parent / "htr_system_prompt_french.md"

        # Fallback to old naming convention if new files don't exist
        if not prompt_file.exists():
            prompt_file = Path(__file__).parent / "htr_system_prompt.md"

        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logging.error(f"System prompt file not found: {prompt_file}")
            raise FileNotFoundError(f"HTR system prompt file not found at {prompt_file}") from None
        except Exception as e:
            logging.error(f"Error reading system prompt file: {e}")
            raise

    def _language_desc(self) -> str:
        """Human-readable language description used in user prompts."""
        if self.language == "multilingual":
            return "text (detect language automatically)"
        if self.language == "arabic":
            return "Arabic"
        return "French"

    def _build_user_prompt(self) -> str:
        """Build the per-page user prompt (the document part comes first)."""
        return (
            f"This is a legitimate handwritten text transcription (HTR) request for academic research and archival preservation. "
            f"Transcribe ALL handwritten {self._language_desc()} text with exact wording, spacing rules, accents, and WITHOUT summarizing or omitting any zones."
        )

    def _extract_response_text(self, response, pdf_content, page_num: int) -> str:
        """Validate a Gemini response and extract its text.

        Uses ``extract_text_from_response`` (which safely skips thought-parts)
        plus a local RECITATION check that falls back to alternative prompts.

        Args:
            response: The Gemini response to validate.
            pdf_content: The PDF Part or uploaded File (reused by the
                RECITATION fallback prompts).
            page_num: Page number for logging (1-indexed).

        Returns:
            The extracted text.

        Raises:
            RuntimeError: When the response is empty, blocked, or every
                RECITATION fallback failed.
        """
        if not response.candidates:
            raise RuntimeError("No candidates in Gemini response")

        candidate = response.candidates[0]

        if not candidate.content or not candidate.content.parts:
            if candidate.finish_reason == types.FinishReason.RECITATION:
                console.print(f"  └─ [yellow]⚠[/] Page {page_num}: Copyright detection triggered, trying alternative...")
                result = self._try_alternative_prompts(pdf_content, page_num)
                if result:
                    return result
                raise RuntimeError("All copyright retry strategies failed")
            raise RuntimeError(f"No valid response. Finish reason: {candidate.finish_reason}")

        text_content = extract_text_from_response(response)
        if not text_content:
            raise RuntimeError("Empty text response from Gemini")

        return text_content

    def process_pdf_page_inline(self, page_bytes: bytes, page_num: int) -> Optional[str]:
        """
        Process a single PDF page inline by sending bytes directly.

        Args:
            page_bytes (bytes): PDF page as bytes
            page_num (int): Page number (for logging, 1-indexed)

        Returns:
            Optional[str]: Extracted text or None if failed
        """
        try:
            console.print(f"  └─ [cyan]📄[/] Processing page {page_num} inline...")

            # Create PDF part from page bytes
            pdf_part = types.Part.from_bytes(
                data=page_bytes,
                mime_type='application/pdf'
            )

            console.print(f"  └─ [cyan]🤖[/] Generating HTR text for page {page_num}...")

            # Following Google's best practice: put prompt AFTER the document
            self.rate_limiter.wait()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[pdf_part, self._build_user_prompt()],  # Document first, then prompt
                config=self.generation_config
            )

            text_content = self._extract_response_text(response, pdf_part, page_num)

            console.print(f"  └─ [green]✅[/] Page {page_num} HTR complete")
            return text_content

        except QuotaExhaustedError:
            raise
        except genai_errors.APIError as e:
            if is_quota_exhausted(e):
                raise QuotaExhaustedError(str(e)) from e
            console.print(f"  └─ [red]❌[/] Page {page_num} inline processing failed: {e}")
            logging.error(f"Page {page_num} inline processing failed: {e}")
            return None
        except (TimeoutError, ConnectionError, RuntimeError) as e:
            # Programming errors deliberately propagate instead of being swallowed.
            console.print(f"  └─ [red]❌[/] Page {page_num} inline processing failed: {e}")
            logging.error(f"Page {page_num} inline processing failed: {e}")
            return None

    def process_pdf_page_upload(self, page_bytes: bytes, page_num: int) -> Optional[str]:
        """
        Process a single PDF page using File API upload (as fallback).

        Only transient failures are retried with backoff: API errors with
        code 429/500/503, timeouts, and connection errors. Everything else
        (bad requests, empty responses, programming errors) fails fast.
        The upload is always deleted afterwards, on success or failure.

        Args:
            page_bytes (bytes): PDF page as bytes
            page_num (int): Page number (for logging, 1-indexed)

        Returns:
            Optional[str]: Extracted text or None if failed
        """
        max_retries = 3
        base_delay = 5

        for attempt in range(max_retries):
            uploaded_file = None
            retryable = False
            try:
                try:
                    console.print(f"  └─ [cyan]⬆[/]  Uploading page {page_num} to Gemini...")
                    uploaded_file = upload_and_wait_active(
                        self.client,
                        page_bytes,
                        mime_type='application/pdf',
                        max_wait=60,
                        poll_interval=1.0,
                    )

                    console.print(f"  └─ [cyan]🤖[/] Generating HTR text for page {page_num}...")

                    # Following Google's best practice: put prompt AFTER the document
                    self.rate_limiter.wait()
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[uploaded_file, self._build_user_prompt()],  # Document first, then prompt
                        config=self.generation_config
                    )

                    text_content = self._extract_response_text(response, uploaded_file, page_num)

                    console.print(f"  └─ [green]✅[/] Page {page_num} HTR complete")
                    return text_content

                finally:
                    # Uploads are never left behind — delete after each attempt.
                    if uploaded_file is not None:
                        delete_uploaded_file(self.client, uploaded_file)

            except QuotaExhaustedError:
                raise
            except genai_errors.APIError as e:
                if is_quota_exhausted(e):
                    raise QuotaExhaustedError(str(e)) from e
                retryable = getattr(e, "code", 0) in RETRYABLE_API_CODES
                console.print(f"  └─ [red]❌[/] Page {page_num} error (attempt {attempt + 1}/{max_retries}): {e}")
                logging.error(f"Page {page_num} processing error (attempt {attempt + 1}): {e}", exc_info=True)
            except (TimeoutError, ConnectionError) as e:
                retryable = True
                console.print(f"  └─ [red]❌[/] Page {page_num} error (attempt {attempt + 1}/{max_retries}): {e}")
                logging.error(f"Page {page_num} processing error (attempt {attempt + 1}): {e}", exc_info=True)
            except RuntimeError as e:
                # Empty/blocked responses and upload failures: not worth retrying.
                console.print(f"  └─ [red]❌[/] Page {page_num} failed: {e}")
                logging.error(f"Page {page_num} processing failed: {e}", exc_info=True)
                return None

            if not retryable:
                return None
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                console.print(f"  └─ [yellow]🔄[/] Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
            else:
                console.print(f"  └─ [red]❌[/] Page {page_num} max retries reached.")

        return None

    def _try_alternative_prompts(self, pdf_content, page_num: int) -> Optional[str]:
        """
        Try alternative prompts when copyright detection is triggered.

        Args:
            pdf_content: PDF Part or File object
            page_num (int): Page number (for logging, 1-indexed)

        Returns:
            Optional[str]: Extracted text or None if all strategies failed
        """
        language_desc = self._language_desc()
        if self.language == "multilingual":
            language_desc = "text (automatically detecting language)"

        alternative_prompts = [
            (
                "Academic Fair Use Request",
                "This is a legitimate academic research request for historical document preservation and scholarly analysis. "
                "Under fair use principles, please perform HTR text extraction from this historical handwritten page. "
                "The purpose is archival preservation and academic research, not commercial reproduction. "
                f"Please transcribe all visible handwritten {language_desc} text while maintaining original formatting and structure."
            ),
            (
                "Educational HTR Request",
                "Please assist with educational HTR processing of this historical handwritten document page. "
                f"Transcribe the handwritten {language_desc} text content for research and educational purposes. "
                "Focus on accuracy and completeness of the text transcription."
            ),
            (
                "Technical HTR Analysis",
                f"Perform technical handwritten text recognition analysis on this {language_desc} document page. "
                "Output the detected text content with preserved formatting. "
                "This is for document digitization and preservation purposes."
            )
        ]

        for strategy_name, alternative_prompt in alternative_prompts:
            try:
                console.print(f"  └─ [yellow]🔄[/] Page {page_num}: Trying {strategy_name}...")
                self.rate_limiter.wait()
                retry_response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[pdf_content, alternative_prompt],  # Document first, then prompt
                    config=self.generation_config
                )

                text_content = extract_text_from_response(retry_response)
                if text_content:
                    console.print(f"  └─ [green]✅[/] Page {page_num} complete (using {strategy_name})")
                    return text_content

                if retry_response.candidates:
                    retry_finish_reason = retry_response.candidates[0].finish_reason
                else:
                    retry_finish_reason = 'Unknown'
                console.print(f"  └─ [yellow]⚠[/] Page {page_num}: {strategy_name} failed. Finish reason: {retry_finish_reason}")

            except genai_errors.APIError as e:
                if is_quota_exhausted(e):
                    raise QuotaExhaustedError(str(e)) from e
                console.print(f"  └─ [yellow]⚠[/] Page {page_num}: {strategy_name} error: {e}")
                continue
            except (TimeoutError, ConnectionError) as e:
                console.print(f"  └─ [yellow]⚠[/] Page {page_num}: {strategy_name} error: {e}")
                continue

        return None

    def process_pdf(self, pdf_path: Path, output_dir: Path) -> None:
        """
        Process a PDF file page-by-page and save results to a text file.

        This method:
        1. Gets the page count from the PDF
        2. Extracts and processes each page individually
        3. Combines results with page markers
        4. Saves to output file

        Args:
            pdf_path (Path): Path to the PDF file to process
            output_dir (Path): Directory to save the output text file
        """
        try:
            console.print()
            console.rule(f"[bold]📄 Processing PDF: {pdf_path.name}", style="cyan")

            # Verify PDF exists
            if not pdf_path.exists():
                console.print(f"[red]❌[/] PDF file not found: [cyan]{pdf_path}[/]")
                logging.error(f"PDF file not found: {pdf_path}")
                return

            file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
            console.print(f"[cyan]📊[/] PDF size: {file_size_mb:.2f} MB")

            # Get page count
            console.print("\n[cyan]🔄[/] Reading PDF structure...")
            total_pages = get_pdf_page_count(pdf_path)
            console.print(f"[green]✅[/] PDF has {total_pages} pages")

            # Create output file
            output_file = output_dir / f"{pdf_path.stem}.txt"
            console.print(f"\n[cyan]📝[/] Output will be saved to: [cyan]{output_file}[/]")

            # Track processing statistics
            successful_pages = 0
            failed_pages = []
            quota_exhausted = False

            # Process each page
            with open(output_file, 'w', encoding='utf-8') as f:
                for page_idx in range(total_pages):
                    page_num = page_idx + 1  # 1-indexed for display

                    console.print()
                    console.rule(f"[dim]📃 Processing page {page_num}/{total_pages}[/]", style="dim")

                    try:
                        # Extract single page as PDF bytes
                        console.print(f"  └─ [cyan]📄[/] Extracting page {page_num}...")
                        page_bytes = extract_pdf_page(pdf_path, page_idx)
                        page_size_mb = len(page_bytes) / (1024 * 1024)

                        # Process page (try inline first, then upload if needed)
                        text = None
                        if page_size_mb < 20:
                            console.print(f"  └─ [cyan]📄[/] Page size: {page_size_mb:.2f} MB - trying inline...")
                            text = self.process_pdf_page_inline(page_bytes, page_num)

                        # Fallback to upload if inline failed or page too large
                        if not text:
                            if page_size_mb < 20:
                                console.print("  └─ [yellow]⚠[/] Inline failed, falling back to upload...")
                            else:
                                console.print(f"  └─ [cyan]📄[/] Page size: {page_size_mb:.2f} MB - using upload...")
                            text = self.process_pdf_page_upload(page_bytes, page_num)

                        if text and text.strip():
                            # Special handling for first page - no header, no extra newlines
                            if page_num == 1:
                                f.write(text)
                            else:
                                # For subsequent pages, add page marker and newlines
                                f.write(f"\n\n--- Page {page_num} ---\n\n")
                                f.write(text)

                            successful_pages += 1
                            console.print(f"[green]✅[/] Successfully processed page {page_num}")
                        else:
                            failed_pages.append(page_num)
                            console.print(f"[red]❌[/] Failed to process page {page_num}")
                            # Add a placeholder for failed pages
                            if page_num == 1:
                                f.write(f"[ERROR: Failed to process page {page_num}]")
                            else:
                                f.write(f"\n\n--- Page {page_num} ---\n\n[ERROR: Failed to process page {page_num}]")

                    except QuotaExhaustedError:
                        remaining = total_pages - page_idx
                        console.print(
                            f"\n[red]❌[/] Quota exhausted! Completed {successful_pages}/{total_pages} pages, "
                            f"{remaining} remaining — stopping early"
                        )
                        logging.error(f"Quota exhausted during {pdf_path.name} at page {page_num}. {successful_pages} pages completed, {remaining} remaining.")
                        quota_exhausted = True
                        break

                    except Exception as e:
                        failed_pages.append(page_num)
                        console.print(f"[red]❌[/] Error processing page {page_num}: {e}")
                        logging.error(f"Error processing page {page_num} of {pdf_path}: {e}")
                        # Add error placeholder
                        if page_num == 1:
                            f.write(f"[ERROR: Failed to process page {page_num}: {str(e)}]")
                        else:
                            f.write(f"\n\n--- Page {page_num} ---\n\n[ERROR: Failed to process page {page_num}: {str(e)}]")

            # Report processing statistics
            console.print()
            console.rule(f"[bold]📊 Processing Summary for {pdf_path.name}", style="cyan")

            success_rate = (successful_pages / total_pages) * 100 if total_pages > 0 else 0
            output_size = output_file.stat().st_size

            summary_table = Table(box=box.ROUNDED)
            summary_table.add_column("Metric", style="dim")
            summary_table.add_column("Value", style="green")
            summary_table.add_row("Total pages", str(total_pages))
            summary_table.add_row("Successfully processed", str(successful_pages))
            if quota_exhausted:
                skipped = total_pages - successful_pages - len(failed_pages)
                summary_table.add_row("Skipped (quota exhausted)", str(skipped))
            summary_table.add_row("Failed pages", f"[red]{len(failed_pages)}[/]" if failed_pages else "0")
            if failed_pages:
                summary_table.add_row("Failed page numbers", str(failed_pages))
            summary_table.add_row("Success rate", f"{success_rate:.1f}%")
            summary_table.add_row("Output file size", f"{output_size:,} bytes")
            console.print(summary_table)
            console.print()

            # Log the results
            logging.info(f"PDF {pdf_path.name}: {successful_pages}/{total_pages} pages successful ({success_rate:.1f}%)")
            if failed_pages:
                logging.warning(f"PDF {pdf_path.name}: Failed pages: {failed_pages}")

            # Re-raise after saving partial results so main() can stop
            if quota_exhausted:
                raise QuotaExhaustedError("Daily quota exhausted")

        except QuotaExhaustedError:
            raise  # let main() handle it
        except Exception as e:
            console.print(f"\n[red]❌[/] Error processing PDF [cyan]{pdf_path}[/]: {e}")
            logging.error(f"Error processing PDF {pdf_path}: {e}", exc_info=True)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Page-by-page handwritten text recognition (HTR) using Google Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gemini_htr_processor.py
  python gemini_htr_processor.py --model gemini-pro-latest --language arabic
  python gemini_htr_processor.py --pdf-dir /path/to/pdfs --rpm 5
        """
    )
    parser.add_argument(
        "--model",
        choices=["gemini-flash-latest", "gemini-pro-latest"],
        default=None,
        help="Model to use for HTR (default: interactive selection)"
    )
    parser.add_argument(
        "--language",
        choices=["french", "arabic", "multilingual"],
        default=None,
        help="Manuscript language (default: interactive selection)"
    )
    parser.add_argument(
        "--pdf-dir",
        default=None,
        help="Directory containing PDF files (default: PDF/ next to this script)"
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=None,
        help="Rate limit: maximum requests per minute (default: no limit)"
    )
    return parser.parse_args()


def select_language_interactive() -> str:
    """Interactively select the manuscript language if not provided via CLI."""
    console.print("\n[bold]Please choose the manuscript language:[/]")
    console.print("1: French handwritten manuscripts")
    console.print("2: Arabic handwritten manuscripts")
    console.print("3: Multilingual/Other languages (AI will auto-detect)")

    language_choice = ""
    while language_choice not in ["1", "2", "3"]:
        language_choice = console.input("[bold]Enter your choice (1, 2, or 3):[/] ").strip()
        if language_choice not in ["1", "2", "3"]:
            console.print("[red]❌[/] Invalid choice. Please enter 1, 2, or 3.")

    if language_choice == "1":
        return "french"
    if language_choice == "2":
        return "arabic"
    return "multilingual"


def select_model_interactive() -> str:
    """Interactively select the Gemini model if not provided via CLI."""
    console.print("\n[bold]Please choose the Gemini model to use:[/]")
    console.print("1: gemini-flash-latest (Faster, good for most cases)")
    console.print("2: gemini-pro-latest (More powerful, more accurate but slower)")

    model_choice = ""
    while model_choice not in ["1", "2"]:
        model_choice = console.input("[bold]Enter your choice (1 or 2):[/] ").strip()
        if model_choice not in ["1", "2"]:
            console.print("[red]❌[/] Invalid choice. Please enter 1 or 2.")

    if model_choice == "1":
        return "gemini-flash-latest"
    return "gemini-pro-latest"


def main():
    """
    Main function to orchestrate the page-by-page PDF HTR process.

    Handles CLI flags and interactive fallbacks for language and model
    selection, then batch-processes PDFs. Each PDF is processed page-by-page
    for better control and error recovery.
    """
    args = parse_args()

    console.print(Panel(
        "Process handwritten PDFs page-by-page without image conversion",
        title="🚀 Page-by-Page PDF HTR using Google Gemini",
        border_style="cyan"
    ))

    # Load environment variables from .env file
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]❌[/] GEMINI_API_KEY not found in environment variables!")
        return
    console.print("[green]✅[/] API Key loaded successfully")

    # Language selection (CLI or interactive)
    if args.language:
        selected_language = args.language
    else:
        selected_language = select_language_interactive()
    console.print(f"[green]✅[/] Using language: [cyan]{selected_language.capitalize()}[/]")

    # Model selection (CLI or interactive)
    if args.model:
        selected_model_name = args.model
    else:
        selected_model_name = select_model_interactive()
    console.print(f"[green]✅[/] Using model: [cyan]{selected_model_name}[/]")

    # Set up directory paths
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else script_dir / "PDF"
    output_dir = script_dir / "OCR_Results"
    output_dir.mkdir(exist_ok=True)

    # Display configuration
    console.print()
    config_table = Table(title="⚙️ Configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Model", selected_model_name)
    config_table.add_row("Language", selected_language.capitalize())
    config_table.add_row("PDF Directory", str(pdf_dir))
    config_table.add_row("Output Directory", str(output_dir))
    config_table.add_row("Rate Limit", f"{args.rpm} RPM" if args.rpm else "None")
    console.print(config_table)

    # Initialize the HTR processor with selected language and model
    console.print("\n[cyan]🔧[/] Initializing Gemini HTR Processor...")
    htr = GeminiHTR(api_key, selected_model_name, selected_language, requests_per_minute=args.rpm)

    # Find all PDF files to process
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        console.print(f"\n[red]❌[/] No PDF files found in [cyan]{pdf_dir}[/]!")
        return

    total_pdfs = len(pdf_files)
    console.print(f"\n[cyan]📚[/] Found {total_pdfs} PDF files to process")

    # Track overall statistics
    overall_stats = {
        'total_pdfs': total_pdfs,
        'processed_pdfs': 0,
        'failed_pdfs': 0,
        'total_size_mb': 0,
        'processing_start': time.time()
    }

    # Process each PDF file sequentially
    for idx, pdf_path in enumerate(pdf_files, 1):
        console.print(f"\n[cyan]📊[/] Progress: PDF {idx}/{total_pdfs} ({(idx / total_pdfs * 100):.1f}%)")

        try:
            htr.process_pdf(pdf_path, output_dir)

            # Check if output file has content
            output_file = output_dir / f"{pdf_path.stem}.txt"
            if output_file.exists() and output_file.stat().st_size > 100:
                overall_stats['processed_pdfs'] += 1
                overall_stats['total_size_mb'] += pdf_path.stat().st_size / (1024 * 1024)
                logging.info(f"Successfully processed {pdf_path.name}")
            else:
                overall_stats['failed_pdfs'] += 1
                logging.warning(f"Output file for {pdf_path.name} is empty or very small")

        except QuotaExhaustedError:
            console.print("\n[red bold]❌ API quota exhausted — stopping all processing.[/]")
            console.print("[red]Partial results (if any) have been saved.[/]")
            console.print("[red]Wait for your quota to reset or upgrade your plan.[/]")
            logging.error("Quota exhausted — aborting remaining PDFs.")
            break

        except Exception as e:
            overall_stats['failed_pdfs'] += 1
            console.print(f"[red]❌[/] Failed to process [cyan]{pdf_path.name}[/]: {e}")
            logging.error(f"Failed to process {pdf_path.name}: {e}")

    # Calculate processing time
    processing_time = time.time() - overall_stats['processing_start']

    # Print final summary
    console.print()
    console.rule("[bold]📈 Final Processing Summary", style="cyan")

    final_table = Table(box=box.ROUNDED)
    final_table.add_column("Metric", style="dim")
    final_table.add_column("Value", style="green")
    final_table.add_row("Total PDFs found", str(overall_stats['total_pdfs']))
    final_table.add_row("Successfully processed", str(overall_stats['processed_pdfs']))
    final_table.add_row(
        "Failed to process",
        f"[red]{overall_stats['failed_pdfs']}[/]" if overall_stats['failed_pdfs'] > 0 else "0",
    )
    final_table.add_row("Total size processed", f"{overall_stats['total_size_mb']:.2f} MB")
    final_table.add_row("Processing time", f"{processing_time / 60:.1f} minutes")
    if overall_stats['total_pdfs'] > 0:
        success_rate = (overall_stats['processed_pdfs'] / overall_stats['total_pdfs']) * 100
        final_table.add_row("Overall success rate", f"{success_rate:.1f}%")
    console.print(final_table)

    # Log final summary
    logging.info(f"Processing complete. {overall_stats['processed_pdfs']}/{overall_stats['total_pdfs']} PDFs processed successfully in {processing_time/60:.1f} minutes")

    console.print("\n[green]✨ Direct PDF HTR Process Complete! ✨[/]\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠ Process interrupted by user[/]")
        logging.info("Process interrupted by user")
    except Exception:
        console.print("\n[red]❌ An error occurred:[/]")
        console.print_exception()
        logging.error("An unexpected error occurred", exc_info=True)
