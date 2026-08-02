"""
AI-Powered HTR (Handwritten Text Recognition) Script using Google Gemini

This script performs high-precision transcription of HANDWRITTEN documents (French, Arabic, or
multilingual) using Google's Gemini vision model with direct PDF processing. It is engineered for
research-grade, archival-quality extraction while preserving correct typography, reading order,
and structural semantics.

The page loop, retry policy, finish-reason handling and batch driver live in
``common/gemini_page_processor.py``, shared with AI_ocr_extraction. This file
supplies what is specific to handwriting: the per-language system prompts, the
user request, and the RECITATION fallback strategies.

Usage:
    python gemini_htr_processor.py
    python gemini_htr_processor.py --model gemini-pro --language arabic
    python gemini_htr_processor.py --pdf-dir /path/to/pdfs --rpm 5

Requirements:
    - Environment variable: GEMINI_API_KEY
    - PDF files in the PDF/ directory (or --pdf-dir)
    - HTR system prompt files (htr_system_prompt_french.md, htr_system_prompt_arabic.md, etc.)
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google.genai import errors as genai_errors
from rich.console import Console
from rich.panel import Panel

console = Console()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.console_utils import count_table, key_value_table, print_file_table, standard_progress
from common.gemini_page_processor import GeminiPageProcessor, PagePolicy, process_pdf_batch
from common.gemini_utils import (
    build_generation_config,
    build_gemini_client,
    extract_text_from_response,
    get_thinking_level,
)
from common.llm_provider import get_model_option, summary_from_option
from common.rate_limiter import QuotaExhaustedError, RateLimiter, is_quota_exhausted

script_dir = Path(__file__).resolve().parent
log_dir = script_dir / 'log'
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_dir / 'htr_gemini.log',
)
LOGGER = logging.getLogger(__name__)

# Handwriting needs the finest stroke detail available. ULTRA_HIGH is only
# accepted per-Part — GenerateContentConfig caps at HIGH.
HTR_MEDIA_RESOLUTION = "ULTRA_HIGH"

LANGUAGES = {
    "french": ("French", "htr_system_prompt_french.md"),
    "arabic": ("Arabic", "htr_system_prompt_arabic.md"),
    "multilingual": ("text (detect language automatically)", "htr_system_prompt_multilingual.md"),
}

# Gemini models available for HTR, keyed by llm_provider registry key.
HTR_MODELS = ["gemini-flash", "gemini-pro"]


def load_system_instruction(language: str) -> str:
    """Load the language-specific HTR system prompt."""
    _, filename = LANGUAGES[language]
    prompt_file = script_dir / filename

    # Fallback to the old naming convention if the per-language file is absent.
    if not prompt_file.exists():
        prompt_file = script_dir / "htr_system_prompt.md"

    try:
        return prompt_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        LOGGER.error(f"System prompt file not found: {prompt_file}")
        raise FileNotFoundError(f"HTR system prompt file not found at {prompt_file}") from None


def build_user_prompt(language: str) -> str:
    """The per-page request; the document part is sent before it."""
    description, _ = LANGUAGES[language]
    return (
        "This is a legitimate handwritten text transcription (HTR) request for academic research "
        "and archival preservation. "
        f"Transcribe ALL handwritten {description} text with exact wording, spacing rules, accents, "
        "and WITHOUT summarizing or omitting any zones."
    )


def build_recitation_fallback(client, model_name: str, generation_config,
                              language: str, rate_limiter: RateLimiter):
    """Return an ``on_blocked`` hook that retries with reframed prompts.

    Gemini's copyright detector fires on some archival handwriting. Restating
    the archival/educational purpose usually clears it; if none of the three
    framings works the page is skipped rather than silently dropped.
    """
    description, _ = LANGUAGES[language]
    if language == "multilingual":
        description = "text (automatically detecting language)"

    strategies = [
        (
            "Academic Fair Use Request",
            "This is a legitimate academic research request for historical document preservation and "
            "scholarly analysis. Under fair use principles, please perform HTR text extraction from this "
            "historical handwritten page. The purpose is archival preservation and academic research, not "
            "commercial reproduction. "
            f"Please transcribe all visible handwritten {description} text while maintaining original "
            "formatting and structure."
        ),
        (
            "Educational HTR Request",
            "Please assist with educational HTR processing of this historical handwritten document page. "
            f"Transcribe the handwritten {description} text content for research and educational purposes. "
            "Focus on accuracy and completeness of the text transcription."
        ),
        (
            "Technical HTR Analysis",
            f"Perform technical handwritten text recognition analysis on this {description} document page. "
            "Output the detected text content with preserved formatting. "
            "This is for document digitization and preservation purposes."
        ),
    ]

    def on_blocked(page_content, page_num: int) -> Optional[str]:
        for name, prompt in strategies:
            try:
                console.print(f"  └─ [yellow]🔄[/] Page {page_num}: trying {name}...")
                rate_limiter.wait()
                response = client.models.generate_content(
                    model=model_name,
                    contents=[page_content, prompt],  # document first, then prompt
                    config=generation_config,
                )
                text = extract_text_from_response(response)
                if text:
                    console.print(f"  └─ [green]✅[/] Page {page_num} complete (using {name})")
                    return text

                reason = response.candidates[0].finish_reason if response.candidates else "Unknown"
                console.print(f"  └─ [yellow]⚠[/] Page {page_num}: {name} failed. Finish reason: {reason}")

            except genai_errors.APIError as exc:
                if is_quota_exhausted(exc):
                    raise QuotaExhaustedError(str(exc)) from exc
                console.print(f"  └─ [yellow]⚠[/] Page {page_num}: {name} error: {exc}")
            except (TimeoutError, ConnectionError) as exc:
                console.print(f"  └─ [yellow]⚠[/] Page {page_num}: {name} error: {exc}")
        return None

    return on_blocked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Page-by-page handwritten text recognition (HTR) using Google Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gemini_htr_processor.py
  python gemini_htr_processor.py --model gemini-pro --language arabic
  python gemini_htr_processor.py --pdf-dir /path/to/pdfs --rpm 5
        """,
    )
    parser.add_argument(
        "--model", choices=HTR_MODELS, default=None,
        help="Model to use for HTR (default: interactive selection)",
    )
    parser.add_argument(
        "--language", choices=list(LANGUAGES), default=None,
        help="Manuscript language (default: interactive selection)",
    )
    parser.add_argument(
        "--pdf-dir", default=None,
        help="Directory containing PDF files (default: PDF/ next to this script)",
    )
    parser.add_argument(
        "--rpm", type=int, default=None,
        help="Rate limit: maximum requests per minute (default: no limit)",
    )
    return parser.parse_args()


def select_language_interactive() -> str:
    """Interactively select the manuscript language if not provided via CLI."""
    console.print("\n[bold]Please choose the manuscript language:[/]")
    console.print("1: French handwritten manuscripts")
    console.print("2: Arabic handwritten manuscripts")
    console.print("3: Multilingual/Other languages (AI will auto-detect)")

    choices = {"1": "french", "2": "arabic", "3": "multilingual"}
    while True:
        choice = console.input("[bold]Enter your choice (1, 2, or 3):[/] ").strip()
        if choice in choices:
            return choices[choice]
        console.print("[red]❌[/] Invalid choice. Please enter 1, 2, or 3.")


def main():
    """Orchestrate the page-by-page PDF HTR process."""
    args = parse_args()

    console.print(Panel(
        "Process handwritten PDFs page-by-page without image conversion",
        title="🚀 Page-by-Page PDF HTR using Google Gemini",
        border_style="cyan",
    ))

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]❌[/] GEMINI_API_KEY not found in environment variables!")
        return
    console.print("[green]✅[/] API Key loaded successfully")

    language = args.language or select_language_interactive()
    console.print(f"[green]✅[/] Using language: [cyan]{language.capitalize()}[/]")

    model_option = get_model_option(args.model, allowed_keys=HTR_MODELS)
    console.print(f"[green]✅[/] Using model: [cyan]{model_option.model}[/]")

    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else script_dir / "PDF"
    output_dir = script_dir / "OCR_Results"

    console.print()
    console.print(key_value_table([
        ("Model", summary_from_option(model_option)),
        ("Language", language.capitalize()),
        ("Thinking Level", get_thinking_level(model_option.model)),
        ("Media Resolution", HTR_MEDIA_RESOLUTION),
        ("PDF Directory", str(pdf_dir)),
        ("Output Directory", str(output_dir)),
        ("Rate Limit", f"{args.rpm} RPM" if args.rpm else "None"),
    ], title="⚙️ Configuration"))

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        console.print(f"\n[red]❌[/] No PDF files found in [cyan]{pdf_dir}[/]!")
        return

    console.print()
    print_file_table(console, pdf_files, title=f"📚 PDF Files to Process ({len(pdf_files)})")

    client = build_gemini_client(api_key)
    rate_limiter = RateLimiter(args.rpm, logger=LOGGER)
    generation_config = build_generation_config(
        model_option.model,
        system_instruction=load_system_instruction(language),
        max_output_tokens=65535,
    )

    processor = GeminiPageProcessor(
        client,
        model_option.model,
        generation_config,
        PagePolicy(
            user_prompt=build_user_prompt(language),
            media_resolution=HTR_MEDIA_RESOLUTION,
            on_blocked=build_recitation_fallback(
                client, model_option.model, generation_config, language, rate_limiter
            ),
        ),
        rate_limiter=rate_limiter,
        console=console,
        logger=LOGGER,
        verbose=True,
    )

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

    console.print()
    console.rule("[bold]📈 Final Processing Summary", style="cyan")
    success_rate = (batch.processed / len(pdf_files) * 100) if pdf_files else 0
    console.print(count_table([
        ("Total PDFs found", str(len(pdf_files))),
        ("[green]Successfully processed[/]", f"[green]{batch.processed}[/]"),
        ("[red]Failed to process[/]", f"[red]{batch.failed}[/]"),
        ("Total size processed", f"{batch.total_size_mb:.2f} MB"),
        ("Processing time", f"{batch.elapsed_seconds / 60:.1f} minutes"),
        ("Overall success rate", f"{success_rate:.1f}%"),
    ]))

    LOGGER.info(
        "Processing complete. %d/%d PDFs processed successfully in %.1f minutes",
        batch.processed, len(pdf_files), batch.elapsed_seconds / 60,
    )

    console.print("\n[green]✨ Direct PDF HTR Process Complete! ✨[/]\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠ Process interrupted by user[/]")
        LOGGER.info("Process interrupted by user")
    except Exception:
        console.print("\n[red]❌ An error occurred:[/]")
        console.print_exception()
        LOGGER.error("An unexpected error occurred", exc_info=True)
