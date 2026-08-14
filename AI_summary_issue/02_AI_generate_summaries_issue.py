"""Islamic Magazine Article Extraction Pipeline (2-Step Process)

This script implements a two-step pipeline to extract and consolidate articles
from an Islamic magazine using Gemini's native PDF understanding with structured outputs.

Supported extraction profiles:
- standard: Gemini Pro (step 1, per page)
- light: Gemini Flash (step 1, per page)

Both profiles use the repository's default text model for step-2 consolidation.

Step 1: Page-by-page extraction (high-performance model)
- Extracts individual pages using pypdf (document parsed once per magazine)
- Sends each page to Gemini using Part.from_bytes (native PDF understanding)
- Uses structured outputs (Pydantic models) for guaranteed JSON schema compliance
- Identifies articles present on the page with typed data extraction
- Extracts exact titles and generates brief summaries
- Detects continuation indicators

Step 2: Magazine-level consolidation (fast model)
- Merges articles fragmented across multiple pages
- Eliminates duplicates with the fast model
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
- Quota-aware error handling: daily quota exhaustion stops the whole batch,
  optional --rpm throttling for free-tier rate limits

API Best Practices:
- Uses system_instruction in GenerateContentConfig for prompts
- Uses response_mime_type='application/json' for structured outputs
- Uses response_schema with Pydantic models for type-safe extraction
- Uses ThinkingConfig with thinking_level for all Gemini 3 models
- Passes PDF bytes via Part.from_bytes for native processing

Usage:
    python 02_AI_generate_summaries_issue.py
    python 02_AI_generate_summaries_issue.py --profile light --rpm 5
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple
from dotenv import load_dotenv

# Rich console output helpers (the console itself is shared with the pipeline module)
from rich.panel import Panel
from rich.table import Table
from rich import box

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.llm_provider import (  # noqa: E402
    DEFAULT_TEXT_MODEL_KEY,
    LLMConfig,
    ModelOption,
    get_model_option,
    summary_from_option,
)
from common.gemini_utils import (  # noqa: E402
    build_generation_config,
    build_gemini_client,
    extract_text_from_response,
)
from common.rate_limiter import RateLimiter, QuotaExhaustedError, is_quota_exhausted  # noqa: E402
from common.retry import retry_with_backoff  # noqa: E402
from common.log_redaction import install_credential_redaction

# Shared magazine-extraction building blocks (models, prompts, step skeletons)
from magazine_extraction import (  # noqa: E402
    PageExtraction,
    PdfPageSource,
    console,
    build_text_consolidator,
    load_extraction_prompt,
    run_extraction_pipeline,
    run_magazine_batch,
)

# Import Gemini types for PDF processing
try:
    from google import genai
    from google.genai import types
    from google.genai import errors as genai_errors
except ImportError as exc:
    raise RuntimeError("google-genai package is required for PDF processing") from exc

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Credentials ride in Omeka query strings and provider headers; keep them
# out of anything urllib3 or an SDK decides to log.
install_credential_redaction()
load_dotenv()

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# ------------------------------------------------------------------
# Client Initialization
# ------------------------------------------------------------------
def get_model_pair(profile: str = "standard") -> Tuple[ModelOption, ModelOption]:
    """Get the multimodal extraction and text consolidation model pair.

    Profiles:
        - "standard": Gemini Pro for per-page extraction (best quality).
        - "light": Gemini Flash for per-page extraction (cheaper/faster).

    Returns:
        Tuple of (model_step1, model_step2).
    """
    if profile == "light":
        step1_option = get_model_option("gemini-3.7-flash")   # per-page extraction
    else:
        step1_option = get_model_option("gemini-pro")
    step2_option = get_model_option(DEFAULT_TEXT_MODEL_KEY)

    # Display model configuration in a table
    model_table = Table(title=f"🤖 Model Configuration ({profile})", box=box.ROUNDED, show_header=True, header_style="bold cyan")
    model_table.add_column("Step", style="white", width=12)
    model_table.add_column("Purpose", style="dim")
    model_table.add_column("Model", style="green")
    model_table.add_row("Step 1", "Page extraction", summary_from_option(step1_option))
    model_table.add_row("Step 2", "Consolidation", summary_from_option(step2_option))
    console.print(model_table)
    console.print()

    logging.info(f"Selected models: Step 1={summary_from_option(step1_option)}, Step 2={summary_from_option(step2_option)}")

    return step1_option, step2_option

def choose_profile() -> str:
    """Interactively ask the user which model profile to run.

    Accepts 1/a for the standard profile and 2/b for the light profile.
    """
    table = Table(title="Select model profile", box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("#", style="cyan", justify="center")
    table.add_column("Profile", style="green")
    table.add_column("Description", style="white")
    table.add_row("1 / a", "standard", "Gemini Pro per page - best extraction quality")
    table.add_row("2 / b", "light", "Gemini Flash per page - cheaper / faster")
    console.print(table)

    mapping = {"1": "standard", "a": "standard", "2": "light", "b": "light"}
    while True:
        choice = console.input("\n[bold]Profile (1/a = standard, 2/b = light) [1]:[/] ").strip().lower()
        if not choice:
            return "standard"
        if choice in mapping:
            return mapping[choice]
        console.print("[red]Invalid choice - enter 1, 2, a, or b.[/]")


# ------------------------------------------------------------------
# AI Generation Functions with Retry
# ------------------------------------------------------------------
@retry_with_backoff(max_retries=MAX_RETRIES, base_delay=RETRY_DELAY)
def generate_with_gemini(client: genai.Client, model_name: str, page_bytes: bytes,
                        page_num: int, config: types.GenerateContentConfig,
                        rate_limiter: RateLimiter) -> Optional[PageExtraction]:
    """
    Generate structured page extraction with Gemini for a single PDF page.

    Args:
        client: Gemini client
        model_name: Model name to use
        page_bytes: Single PDF page as bytes
        page_num: Page number (for logging)
        config: Generation config (includes system_instruction and response_schema)
        rate_limiter: Shared rate limiter (wait() called before the API request)

    Returns:
        PageExtraction object or None

    Raises:
        QuotaExhaustedError: When the daily API quota is exhausted (not retried).
    """
    try:
        # Create PDF part from page bytes
        pdf_part = types.Part.from_bytes(
            data=page_bytes,
            mime_type='application/pdf'
        )

        rate_limiter.wait()
        # Generate content with single page PDF
        response = client.models.generate_content(
            model=model_name,
            contents=[pdf_part, f"Analysez la page {page_num} de ce document PDF."],
            config=config
        )

        text_content = extract_text_from_response(response)
        if not text_content:
            finish_reason = response.candidates[0].finish_reason if response.candidates else None
            raise RuntimeError(f"Empty or invalid Gemini response. Finish reason: {finish_reason}")

        # Parse JSON response into Pydantic model
        extraction = PageExtraction.model_validate_json(text_content)
        extraction.page_number = page_num  # Ensure page number is set
        return extraction

    except genai_errors.APIError as e:
        if is_quota_exhausted(e):
            raise QuotaExhaustedError(str(e)) from e
        logging.error(f"Generation error for page {page_num}: {e}")
        raise  # Let the retry decorator handle it
    except Exception as e:
        logging.error(f"Generation error for page {page_num}: {e}")
        raise  # Let the retry decorator handle it

# ------------------------------------------------------------------
# Main Pipeline
# ------------------------------------------------------------------
def process_magazine(model_step1: ModelOption, model_step2: ModelOption,
                    pdf_path: Path, output_dir: Path, magazine_id: str = None,
                    rate_limiter: Optional[RateLimiter] = None):
    """
    Complete pipeline to process a magazine PDF using Gemini's native PDF understanding.

    Args:
        model_step1: Model option for step 1 (Gemini Pro)
        model_step2: Model option for step 2 (Gemini Flash)
        pdf_path: Path to PDF file
        output_dir: Output directory
        magazine_id: Magazine identifier (optional)
        rate_limiter: Shared RateLimiter (None = no throttling)
    """
    # Determine magazine identifier
    if magazine_id is None:
        magazine_id = pdf_path.stem
    if rate_limiter is None:
        rate_limiter = RateLimiter(requests_per_minute=None)

    logging.info(f"Processing magazine: {magazine_id}")
    logging.info(f"Input: {pdf_path}")
    logging.info(f"Output: {output_dir}")

    # Verify PDF exists
    if not pdf_path.exists() or not pdf_path.is_file():
        raise ValueError(f"PDF file not found: {pdf_path}")

    if pdf_path.suffix.lower() != '.pdf':
        raise ValueError(f"Input must be a PDF file, got: {pdf_path}")

    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment variables")

    # Initialize Gemini client
    client = build_gemini_client(api_key)

    # Configure for each step - all Gemini 3 models use thinking_level.
    # Step 1 (per-page extraction) is the quality-critical step - Pro (standard)
    # or Flash (light) - so it gets LOW thinking.
    # Neither step sets a temperature: Google recommends sending none for
    # Gemini 3, since lowering it can send the model into a loop.
    config_step1 = LLMConfig(thinking_level="LOW")

    # Load the extraction prompt
    extraction_prompt = load_extraction_prompt()

    # Get PDF info and display in panel
    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)

    pdf_info = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    pdf_info.add_column("Key", style="dim")
    pdf_info.add_column("Value")
    pdf_info.add_row("📄 File", pdf_path.name)
    pdf_info.add_row("📊 Size", f"{file_size_mb:.2f} MB")
    pdf_info.add_row("📁 Output", str(output_dir))

    console.print(Panel(pdf_info, title=f"[bold]Magazine: {magazine_id}[/]", border_style="blue"))

    try:
        # Parse the PDF once for the whole magazine (page count + per-page bytes)
        with console.status("[cyan]Reading PDF structure...", spinner="dots"):
            page_source = PdfPageSource(pdf_path)
            total_pages = len(page_source)
        console.print(f"[green]✓[/] PDF has [bold]{total_pages}[/] pages")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Shared generation config for step 1 (structured output, thinking level)
        step1_gen_config = build_generation_config(
            model_step1.model,
            thinking_level=config_step1.thinking_level,
            system_instruction=extraction_prompt,
            max_output_tokens=8192,
            response_schema=PageExtraction,
        )
        logging.info(f"Using Gemini 3 with thinking_level={config_step1.thinking_level} for step 1")

        def extract_page(page_num: int) -> Optional[PageExtraction]:
            """Provider callable for the shared step-1 loop."""
            page_bytes = page_source.page_bytes(page_num - 1)  # 0-indexed for pypdf
            return generate_with_gemini(
                client, model_step1.model, page_bytes, page_num, step1_gen_config, rate_limiter
            )

        consolidate = build_text_consolidator(model_step2)

        # Step 1 (page loop) + step 2 (consolidation) via the shared skeleton.
        # The page_*.json cache is deleted only after step 2 succeeds.
        final_file = run_extraction_pipeline(
            extract_page=extract_page,
            consolidate=consolidate,
            total_pages=total_pages,
            output_dir=output_dir,
            magazine_id=magazine_id,
            step1_model_label=summary_from_option(model_step1),
            step2_model_label=summary_from_option(model_step2),
            schema_note="JSON schema",
        )

        logging.info(f"Pipeline complete for magazine {magazine_id}")
        logging.info(f"Final index: {final_file}")

        return final_file

    except QuotaExhaustedError:
        raise  # let main() stop the whole batch
    except Exception as e:
        console.print(f"[red]✗[/] Error processing PDF: {e}")
        logging.error(f"Error processing PDF {pdf_path}: {e}")
        raise

# ------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------
def main() -> int:
    """Main entry point of the script."""
    try:
        load_dotenv()

        # Parse CLI args. When neither --profile nor --light is given, the user
        # is prompted interactively (see choose_profile()).
        parser = argparse.ArgumentParser(
            description="Islamic magazine article extraction to a table of contents."
        )
        parser.add_argument(
            "--profile", choices=["standard", "light"], default=None,
            help="standard = Gemini Pro per page; light = Gemini Flash per page. "
                 f"Both consolidate with {DEFAULT_TEXT_MODEL_KEY}. "
                 "Omit to choose interactively.",
        )
        parser.add_argument(
            "--light", action="store_true", help="Shortcut for --profile light.",
        )
        parser.add_argument(
            "--rpm", type=int, default=None,
            help="Requests per minute limit (e.g. 5 for free tier). "
                 "Omit for no throttling (paid tiers).",
        )
        args = parser.parse_args()

        # Resolve the model profile: a CLI flag wins, otherwise ask interactively.
        if args.light:
            profile = "light"
        elif args.profile:
            profile = args.profile
        else:
            profile = choose_profile()

        step1_label = "Gemini Flash" if profile == "light" else "Gemini Pro"
        intro_panel = Panel(
            "[bold cyan]Islamic Magazine Article Extraction Pipeline[/]\n\n"
            f"[dim]Using Gemini's native PDF understanding — profile: {profile}[/]\n\n"
            f"📖 [white]Step 1:[/] Page-by-page extraction [dim]({step1_label})[/]\n"
            f"📊 [white]Step 2:[/] Magazine-level consolidation "
            f"[dim]({DEFAULT_TEXT_MODEL_KEY})[/]",
            title="🚀 Pipeline Started", border_style="cyan", padding=(1, 2),
        )

        logging.info("=== Magazine Article Extraction Pipeline ===")

        model_step1, model_step2 = get_model_pair(profile)

        # Rate limiter shared across the whole batch (None = no throttling)
        rate_limiter = RateLimiter(requests_per_minute=args.rpm)
        if args.rpm:
            console.print(f"[cyan]⏱[/] Rate limiting: {args.rpm} requests/minute")

        return run_magazine_batch(
            lambda pdf_path, output_dir, magazine_id: process_magazine(
                model_step1, model_step2, pdf_path, output_dir, magazine_id,
                rate_limiter=rate_limiter,
            ),
            script_dir=Path(__file__).resolve().parent,
            intro_panel=intro_panel,
            api_key_env=("GEMINI_API_KEY", "OPENROUTER_API_KEY"),
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
