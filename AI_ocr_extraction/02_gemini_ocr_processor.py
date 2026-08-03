"""
AI-Powered Direct PDF Processing using Google Gemini / Gemma (Page-by-Page)

This script performs high-precision OCR on PDF documents by sending them directly to Gemini
page-by-page, without converting to images first. This leverages Gemini's native PDF understanding
while maintaining precise control over page-by-page extraction.

The page loop, retry policy, finish-reason handling and batch driver live in
``common/gemini_page_processor.py``, shared with AI_htr_extraction. This file
supplies only what is specific to printed-page OCR: the system prompt, the user
request, and the generation settings.

Usage:
    python 02_gemini_ocr_processor.py
    python 02_gemini_ocr_processor.py --model gemini-pro --rpm 5

Requirements:
    - Environment variable: GEMINI_API_KEY
    - PDF files in the PDF/ directory
    - OCR system prompt in ocr_system_prompt.md

Model Selection:
    - Uses shared LLM provider with three options (all via the Gemini API):
        * Gemini Flash — faster, cost-effective, uses MINIMAL thinking
        * Gemini Pro — higher quality, uses LOW thinking
        * Gemma 4 31B    — open-weights flagship, uses MINIMAL thinking
          (Gemma 4 accepts only MINIMAL or HIGH; MINIMAL is used for OCR speed.)

Advantages over image-based approach:
    - No Poppler dependency needed
    - Better document structure understanding
    - Processes images, diagrams, and tables natively
    - Simpler pipeline with fewer conversions
    - Page-by-page processing for better control
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

console = Console()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.console_utils import count_table, key_value_table, print_file_table, standard_progress
from common.gemini_page_processor import GeminiPageProcessor, PagePolicy, process_pdf_batch
from common.gemini_utils import build_generation_config, build_gemini_client, get_thinking_level
from common.llm_provider import GEMINI_DOCUMENT_MODELS, LLMConfig, get_model_option, summary_from_option
from common.rate_limiter import RateLimiter
from common.log_redaction import install_credential_redaction

# Set up logging configuration for tracking OCR operations and errors
script_dir = Path(__file__).resolve().parent
log_dir = script_dir / 'log'
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_dir / 'ocr_gemini_pdf.log',
)
# Credentials ride in Omeka query strings and provider headers; keep them
# out of anything urllib3 or an SDK decides to log.
install_credential_redaction()

# User request sent alongside each page; the system instruction carries the
# detailed OCR rules.
OCR_USER_PROMPT = (
    "Please perform complete OCR transcription of this single page. "
    "Extract all visible text maintaining original formatting and structure."
)

# Archival newspaper scans are dense and often degraded. ULTRA_HIGH is only
# accepted per-Part (GenerateContentConfig caps at HIGH), which is why the
# resolution is set on the policy rather than in the generation config.
OCR_MEDIA_RESOLUTION = "ULTRA_HIGH"


def load_system_instruction() -> str:
    """Load the OCR system prompt that carries the detailed extraction rules."""
    prompt_file = script_dir / "ocr_system_prompt.md"
    try:
        return prompt_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        logging.error(f"System prompt file not found: {prompt_file}")
        raise FileNotFoundError(f"OCR system prompt file not found at {prompt_file}") from None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Page-by-page OCR of PDFs using Google Gemini / Gemma."
    )
    parser.add_argument(
        "--model", choices=GEMINI_DOCUMENT_MODELS, default=None,
        help="Model to use (default: interactive selection)",
    )
    parser.add_argument(
        "--pdf-dir", type=Path, default=None,
        help="Directory containing PDF files (default: PDF/ next to this script)",
    )
    parser.add_argument(
        "--rpm", type=int, default=None,
        help="Requests per minute limit (e.g. 5 for free tier)",
    )
    return parser.parse_args()


def main():
    """Orchestrate the page-by-page PDF OCR process."""
    args = parse_args()

    console.print(Panel(
        "[bold]AI-Powered PDF OCR using Google Gemini[/bold]\n"
        "Processes PDFs page-by-page with native document understanding",
        title="📄 Gemini OCR Processor",
        border_style="cyan",
    ))

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]✗[/] GEMINI_API_KEY not found in environment variables!")
        return
    console.print("[green]✓[/] API Key loaded successfully")

    console.print()
    console.rule("[bold cyan]🤖 Model Selection[/]")
    model_option = get_model_option(args.model, allowed_keys=GEMINI_DOCUMENT_MODELS)

    # OCR runs at the model's default temperature: Google recommends sending none
    # for Gemini 3, because a lowered one can make the model loop — on a page scan
    # that shows up as the same line repeating until max_output_tokens. Minimal
    # thinking for speed. All Gemini 3 / Gemma 4 models use thinking_level (it
    # cannot be disabled):
    #   Gemini Flash: MINIMAL — fastest, sufficient for OCR
    #   Gemini Pro:   LOW     — Pro does not accept MINIMAL
    #   Gemma 4:      MINIMAL — only MINIMAL or HIGH accepted; MINIMAL for speed
    llm_config = LLMConfig(
        thinking_level="LOW" if "pro" in model_option.model.lower() else "MINIMAL",
    )

    console.print(key_value_table([
        ("Model", summary_from_option(model_option)),
        ("Thinking Level", get_thinking_level(model_option.model, llm_config.thinking_level)),
        ("Media Resolution", OCR_MEDIA_RESOLUTION),
        ("Rate Limit", f"{args.rpm} RPM" if args.rpm else "None"),
    ]))

    pdf_dir = args.pdf_dir or (script_dir / "PDF")
    output_dir = script_dir / "OCR_Results"

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        console.print(f"[red]✗[/] No PDF files found in {pdf_dir}!")
        return

    console.print()
    print_file_table(console, pdf_files, title=f"📚 PDF Files to Process ({len(pdf_files)})")

    processor = GeminiPageProcessor(
        build_gemini_client(api_key),
        model_option.model,
        build_generation_config(
            model_option.model,
            thinking_level=llm_config.thinking_level,
            system_instruction=load_system_instruction(),
        ),
        PagePolicy(user_prompt=OCR_USER_PROMPT, media_resolution=OCR_MEDIA_RESOLUTION),
        rate_limiter=RateLimiter(args.rpm, logger=logging.getLogger(__name__)),
        console=console,
        logger=logging.getLogger(__name__),
    )
    console.print("[green]✓[/] Processor initialized")

    console.print()
    console.rule("[bold cyan]📄 Processing PDFs[/]")

    with standard_progress(console) as progress:
        batch = process_pdf_batch(processor, pdf_files, output_dir, console=console, progress=progress)

    if batch.quota_exhausted:
        console.print(Panel(
            "[red bold]API quota exhausted — stopping all processing.[/]\n"
            "Partial results (if any) have been saved.\n"
            "Wait for your quota to reset or upgrade your plan.",
            title="Quota Exhausted",
            border_style="red",
        ))

    success_rate = (batch.processed / len(pdf_files) * 100) if pdf_files else 0
    console.print()
    console.print(count_table([
        ("Total PDFs", str(len(pdf_files))),
        ("[green]Successful[/]", f"[green]{batch.processed}[/]"),
        ("[red]Failed[/]", f"[red]{batch.failed}[/]"),
        ("Total Size", f"{batch.total_size_mb:.2f} MB"),
        ("Processing Time", f"{batch.elapsed_seconds / 60:.1f} minutes"),
        ("Success Rate", f"{success_rate:.1f}%"),
    ], title="📈 Processing Summary"))

    logging.info(
        "Processing complete. %d/%d PDFs processed successfully in %.1f minutes",
        batch.processed, len(pdf_files), batch.elapsed_seconds / 60,
    )

    console.print(Panel(
        f"[green]✓[/] Processed {batch.processed}/{len(pdf_files)} PDFs successfully",
        title="✨ OCR Complete",
        border_style="green",
    ))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠[/] Process interrupted by user")
        logging.info("Process interrupted by user")
    except Exception as e:
        console.print(f"\n[red]✗[/] An error occurred: {e}")
        logging.error(f"An error occurred: {e}", exc_info=True)
