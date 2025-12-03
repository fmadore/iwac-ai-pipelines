"""Islamic Magazine Article Extraction Pipeline (2-Step Process)

This script implements a two-step pipeline to extract and consolidate articles
from an Islamic magazine using Gemini's native PDF understanding with structured outputs.

Supported models:
- Gemini 3 Pro (step 1) - High-quality extraction with thinking_level
- Gemini 2.5 Flash (step 2) - Fast consolidation with thinking_budget

Step 1: Page-by-page extraction (high-performance model)
- Extracts individual pages using PyPDF2
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
- Progressive result saving with JSON caching
- Resumption possible from already processed files

API Best Practices:
- Uses system_instruction in GenerateContentConfig for prompts
- Uses response_mime_type='application/json' for structured outputs
- Uses response_schema with Pydantic models for type-safe extraction
- Uses ThinkingConfig with thinking_level for Gemini 3 Pro
- Uses ThinkingConfig with thinking_budget for Gemini 2.5 Flash
- Passes PDF bytes via Part.from_bytes for native processing

Usage:
    python 02_AI_generate_summaries_issue.py
"""

import os
import sys
import io
import json
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
from PyPDF2 import PdfReader, PdfWriter
from pydantic import BaseModel, Field

# Rich console for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich import box

console = Console()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from common.llm_provider import (  # noqa: E402
    LLMConfig,
    ModelOption,
    get_model_option,
    summary_from_option,
)

# ------------------------------------------------------------------
# Pydantic Models for Structured Outputs
# ------------------------------------------------------------------

class PageArticle(BaseModel):
    """An article found on a single page."""
    titre: str = Field(description="Titre exact de l'article tel qu'imprimé sur la page")
    continuation: Optional[str] = Field(default=None, description="Indication de continuation: 'suite page X' ou null si aucune")
    resume: str = Field(description="Résumé bref de 2-3 phrases du contenu visible sur cette page")

class PageExtraction(BaseModel):
    """Extraction result for a single PDF page."""
    page_number: int = Field(description="Numéro de la page analysée")
    is_cover: bool = Field(default=False, description="True si c'est une page de couverture")
    has_articles: bool = Field(default=True, description="True si des articles sont présents")
    articles: List[PageArticle] = Field(default_factory=list, description="Liste des articles trouvés sur cette page")
    other_content: Optional[str] = Field(default=None, description="Description des autres contenus non-articles (publicités, annonces, etc.)")

class ConsolidatedArticle(BaseModel):
    """A consolidated article after merging fragmented pages."""
    titre: str = Field(description="Titre exact complet de l'article")
    pages: str = Field(description="Numéros de pages, ex: '1-3' ou '1, 3, 5'")
    resume: str = Field(description="Résumé global consolidé de 4-6 phrases")

class MagazineIndex(BaseModel):
    """Final consolidated index of all articles in the magazine."""
    articles: List[ConsolidatedArticle] = Field(description="Liste des articles consolidés du magazine")

# Import Gemini types for PDF processing
try:
    from google import genai
    from google.genai import types
except ImportError:
    raise RuntimeError("google-genai package is required for PDF processing")

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
RETRY_BACKOFF = 2  # multiplier for exponential backoff

# ------------------------------------------------------------------
# Prompt Loading
# ------------------------------------------------------------------
def load_extraction_prompt() -> str:
    """Load the prompt for step 1 (page-by-page extraction).
    
    Note: With structured outputs, the prompt focuses on instructions
    while the schema defines the output format.
    """
    script_dir = Path(__file__).parent
    prompt_file = script_dir / 'summary_prompt_issue.md'
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt template not found: {prompt_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to read prompt template {prompt_file}: {e}")

def load_consolidation_prompt() -> str:
    """Load the prompt for step 2 (consolidation).
    
    Note: With structured outputs, the prompt focuses on instructions
    while the schema defines the output format.
    """
    script_dir = Path(__file__).parent
    prompt_file = script_dir / 'consolidation_prompt_issue.md'
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Consolidation prompt template not found: {prompt_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to read consolidation prompt template {prompt_file}: {e}")

# ------------------------------------------------------------------
# Client Initialization
# ------------------------------------------------------------------
def get_model_pair() -> Tuple[ModelOption, ModelOption]:
    """Get the Gemini model pair for the pipeline.
    
    Returns:
        Tuple of (model_step1, model_step2) where step1 is Pro and step2 is Flash.
    """
    step1_option = get_model_option("gemini-pro")
    step2_option = get_model_option("gemini-flash")
    
    # Display model configuration in a table
    model_table = Table(title="🤖 Model Configuration", box=box.ROUNDED, show_header=True, header_style="bold cyan")
    model_table.add_column("Step", style="white", width=12)
    model_table.add_column("Purpose", style="dim")
    model_table.add_column("Model", style="green")
    model_table.add_row("Step 1", "Page extraction", summary_from_option(step1_option))
    model_table.add_row("Step 2", "Consolidation", summary_from_option(step2_option))
    console.print(model_table)
    console.print()
    
    logging.info(f"Selected models: Step 1={summary_from_option(step1_option)}, Step 2={summary_from_option(step2_option)}")
    
    return step1_option, step2_option

# ------------------------------------------------------------------
# PDF Page Extraction (PyPDF2)
# ------------------------------------------------------------------
def extract_pdf_page(pdf_path: Path, page_number: int) -> bytes:
    """
    Extract a single page from a PDF as bytes.
    
    Args:
        pdf_path: Path to the PDF file
        page_number: Page number to extract (0-indexed)
        
    Returns:
        PDF bytes containing only the specified page
    """
    try:
        reader = PdfReader(str(pdf_path))
        writer = PdfWriter()
        
        # Add the specific page
        writer.add_page(reader.pages[page_number])
        
        # Write to bytes
        output_buffer = io.BytesIO()
        writer.write(output_buffer)
        output_buffer.seek(0)
        
        return output_buffer.getvalue()
        
    except Exception as e:
        logging.error(f"Error extracting page {page_number + 1} from {pdf_path}: {e}")
        raise

def get_pdf_page_count(pdf_path: Path) -> int:
    """
    Get the number of pages in a PDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Number of pages in the PDF
    """
    try:
        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception as e:
        logging.error(f"Error reading PDF page count from {pdf_path}: {e}")
        raise

# ------------------------------------------------------------------
# AI Generation Functions with Retry
# ------------------------------------------------------------------
def retry_on_error(max_retries: int = MAX_RETRIES, delay: float = RETRY_DELAY):
    """
    Decorator for automatic retry on error.
    
    Args:
        max_retries: Maximum number of attempts
        delay: Initial delay between attempts (with exponential backoff)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logging.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= RETRY_BACKOFF
                    else:
                        logging.error(f"All {max_retries} attempts failed.")
            
            raise last_exception
        return wrapper
    return decorator

@retry_on_error(max_retries=MAX_RETRIES, delay=RETRY_DELAY)
def generate_with_gemini(client: genai.Client, model_name: str, page_bytes: bytes, 
                        page_num: int, config: types.GenerateContentConfig) -> Optional[PageExtraction]:
    """
    Generate structured page extraction with Gemini for a single PDF page.
    
    Args:
        client: Gemini client
        model_name: Model name to use
        page_bytes: Single PDF page as bytes
        page_num: Page number (for logging)
        config: Generation config (includes system_instruction and response_schema)
        
    Returns:
        PageExtraction object or None
    """
    try:
        # Create PDF part from page bytes
        pdf_part = types.Part.from_bytes(
            data=page_bytes,
            mime_type='application/pdf'
        )
        
        # Generate content with single page PDF
        response = client.models.generate_content(
            model=model_name,
            contents=[pdf_part, f"Analysez la page {page_num} de ce document PDF."],
            config=config
        )
        
        # Validate response
        if not response.candidates:
            raise Exception("No candidates in Gemini response")
        
        candidate = response.candidates[0]
        
        if not candidate.content or not candidate.content.parts:
            finish_reason = candidate.finish_reason
            raise Exception(f"No valid response. Finish reason: {finish_reason}")
        
        text_content = response.text.strip()
        if not text_content:
            raise Exception("Empty text response from Gemini")
        
        # Parse JSON response into Pydantic model
        extraction = PageExtraction.model_validate_json(text_content)
        extraction.page_number = page_num  # Ensure page number is set
        return extraction
        
    except Exception as e:
        logging.error(f"Generation error for page {page_num}: {e}")
        raise  # Let the retry decorator handle it

@retry_on_error(max_retries=MAX_RETRIES, delay=RETRY_DELAY)
def generate_consolidation_with_gemini(client: genai.Client, model_name: str,
                                      user_content: str, config: types.GenerateContentConfig) -> Optional[MagazineIndex]:
    """
    Generate structured consolidation response with Gemini.
    
    Args:
        client: Gemini client
        model_name: Model name to use
        user_content: The extracted content to consolidate (JSON)
        config: Generation config (includes system_instruction and response_schema)
        
    Returns:
        MagazineIndex object or None
    """
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=user_content,
            config=config
        )
        
        # Validate response
        if not response.candidates:
            raise Exception("No candidates in Gemini response")
        
        candidate = response.candidates[0]
        
        if not candidate.content or not candidate.content.parts:
            finish_reason = candidate.finish_reason
            raise Exception(f"No valid response. Finish reason: {finish_reason}")
        
        text_content = response.text.strip()
        if not text_content:
            raise Exception("Empty text response from Gemini")
        
        # Parse JSON response into Pydantic model
        return MagazineIndex.model_validate_json(text_content)
        
    except Exception as e:
        logging.error(f"Consolidation generation error: {e}")
        raise  # Let the retry decorator handle it

def format_extraction_to_markdown(extraction: PageExtraction) -> str:
    """Convert a PageExtraction to markdown format for intermediate files."""
    lines = [f"## Page : {extraction.page_number}\n"]
    
    if extraction.is_cover:
        lines.append("Page de couverture du magazine.\n")
        return "\n".join(lines)
    
    if not extraction.has_articles:
        lines.append("Aucun article identifié sur cette page.\n")
        if extraction.other_content:
            lines.append(f"\n### Autres contenus\n{extraction.other_content}\n")
        return "\n".join(lines)
    
    for i, article in enumerate(extraction.articles, 1):
        lines.append(f"\n### Article {i}")
        lines.append(f"- Titre exact : \"{article.titre}\"")
        lines.append(f"- Continuation : {article.continuation or 'aucune'}")
        lines.append(f"- Résumé :\n  {article.resume}")
    
    if extraction.other_content:
        lines.append(f"\n### Autres contenus\n{extraction.other_content}")
    
    return "\n".join(lines)

def format_index_to_markdown(index: MagazineIndex) -> str:
    """Convert a MagazineIndex to the final markdown format."""
    lines = ["# Index des articles du magazine\n"]
    
    for article in index.articles:
        lines.append(f"\n## {article.titre}")
        lines.append(f"- Pages : {article.pages}")
        lines.append(f"- Résumé :")
        lines.append(f"  {article.resume}")
    
    return "\n".join(lines)

# ------------------------------------------------------------------
# Pipeline Functions
# ------------------------------------------------------------------
def step1_extract_pages(client: genai.Client, model_option: ModelOption, llm_config: LLMConfig,
                        pdf_path: Path, total_pages: int, extraction_prompt: str, 
                        output_dir: Path, magazine_id: str) -> Tuple[Path, List[PageExtraction]]:
    """
    Step 1: Page-by-page extraction with high-performance Gemini model.
    Uses structured outputs for guaranteed JSON schema compliance.
    Progressive saving to allow resumption in case of interruption.
    
    Args:
        client: Gemini client
        model_option: Model option for step 1
        llm_config: LLM configuration
        pdf_path: Path to the PDF file
        total_pages: Total number of pages in the PDF
        extraction_prompt: System prompt for extraction
        output_dir: Output directory
        magazine_id: Magazine identifier
        
    Returns:
        Tuple of (consolidated markdown file path, list of PageExtraction objects)
    """
    step1_dir = output_dir / "step1_page_extractions"
    step1_dir.mkdir(parents=True, exist_ok=True)
    
    all_extractions: List[PageExtraction] = []
    all_markdown: List[str] = []
    model_name = summary_from_option(model_option)
    
    # Setup generation config with structured output
    config_kwargs = {
        "system_instruction": extraction_prompt,
        "temperature": llm_config.temperature or 0.2,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",  # Enable JSON mode
        "response_schema": PageExtraction,  # Pydantic model for structured output
    }
    
    # Handle thinking config based on model type
    is_gemini_3 = "gemini-3" in model_option.model.lower()
    if is_gemini_3:
        thinking_level = llm_config.thinking_level or model_option.default_thinking_level or "low"
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
        logging.info(f"Using Gemini 3 with thinking_level={thinking_level}")
    else:
        thinking_budget = llm_config.thinking_budget
        if thinking_budget is None:
            thinking_budget = model_option.default_thinking_budget
        if thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
            logging.info(f"Using Gemini 2.5 with thinking_budget={thinking_budget}")
    
    gen_config = types.GenerateContentConfig(**config_kwargs)
    
    console.print(f"\n[bold cyan]Step 1:[/] Processing {total_pages} pages with [green]{model_name}[/]")
    console.print("[dim]Using structured outputs (JSON schema)[/]")
    logging.info(f"Step 1: Processing {total_pages} pages with {model_name} (structured output)...")
    
    success_count = 0
    cached_count = 0
    error_count = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False
    ) as progress:
        task = progress.add_task("[cyan]Extracting articles...", total=total_pages)
        
        for page_num in range(1, total_pages + 1):
            page_json_file = step1_dir / f"page_{page_num:03d}.json"
            page_md_file = step1_dir / f"page_{page_num:03d}.md"
            
            # Check if page has already been processed (JSON cache)
            if page_json_file.exists():
                logging.info(f"Page {page_num} already processed, loading from cache...")
                try:
                    with open(page_json_file, 'r', encoding='utf-8') as f:
                        extraction = PageExtraction.model_validate_json(f.read())
                    all_extractions.append(extraction)
                    all_markdown.append(format_extraction_to_markdown(extraction))
                    cached_count += 1
                    progress.update(task, advance=1, description=f"[dim]Page {page_num}/{total_pages} (cached)[/]")
                    continue
                except Exception as e:
                    logging.warning(f"Failed to load cached page {page_num}, re-processing: {e}")
            
            progress.update(task, description=f"[cyan]Page {page_num}/{total_pages}[/]")
            
            # Extract single page from PDF
            try:
                page_idx = page_num - 1  # 0-indexed for PyPDF2
                page_bytes = extract_pdf_page(pdf_path, page_idx)
                
                # Generate structured extraction with automatic retry
                extraction = generate_with_gemini(
                    client, model_option.model, page_bytes, page_num, gen_config
                )
                
                if extraction:
                    all_extractions.append(extraction)
                    markdown = format_extraction_to_markdown(extraction)
                    all_markdown.append(markdown)
                    
                    # Save both JSON and markdown for debugging/resumption
                    with open(page_json_file, 'w', encoding='utf-8') as f:
                        f.write(extraction.model_dump_json(indent=2))
                    with open(page_md_file, 'w', encoding='utf-8') as f:
                        f.write(markdown)
                    
                    logging.info(f"Page {page_num} processed and saved")
                    success_count += 1
                else:
                    logging.error(f"No extraction generated for page {page_num}")
                    # Create a placeholder extraction
                    placeholder = PageExtraction(
                        page_number=page_num,
                        has_articles=False,
                        other_content="Error processing this page."
                    )
                    all_extractions.append(placeholder)
                    all_markdown.append(format_extraction_to_markdown(placeholder))
                    error_count += 1
                    
            except Exception as e:
                logging.error(f"Failed to extract or process page {page_num} after retries: {e}")
                placeholder = PageExtraction(
                    page_number=page_num,
                    has_articles=False,
                    other_content=f"Error: {str(e)}"
                )
                all_extractions.append(placeholder)
                all_markdown.append(format_extraction_to_markdown(placeholder))
                error_count += 1
            
            progress.update(task, advance=1)
    
    # Display step 1 summary
    step1_summary = Table(box=box.SIMPLE, show_header=False)
    step1_summary.add_column("Status", style="bold")
    step1_summary.add_column("Count", justify="right")
    step1_summary.add_row("[green]✓ Processed[/]", str(success_count))
    if cached_count > 0:
        step1_summary.add_row("[dim]↺ Cached[/]", str(cached_count))
    if error_count > 0:
        step1_summary.add_row("[red]✗ Errors[/]", str(error_count))
    console.print(step1_summary)
    
    # Save consolidated JSON (for step 2)
    consolidated_json_file = output_dir / f"{magazine_id}_step1_consolidated.json"
    with open(consolidated_json_file, 'w', encoding='utf-8') as f:
        json.dump([e.model_dump() for e in all_extractions], f, ensure_ascii=False, indent=2)
    
    # Save consolidated markdown (for human review)
    consolidated_md_file = output_dir / f"{magazine_id}_step1_consolidated.md"
    with open(consolidated_md_file, 'w', encoding='utf-8') as f:
        f.write(f"# Page-by-page extraction - Magazine {magazine_id}\n\n")
        f.write('\n---\n'.join(all_markdown))
    
    console.print(f"[green]✓[/] Step 1 complete → [dim]{consolidated_json_file.name}[/]")
    logging.info(f"Step 1 complete. Consolidated files: {consolidated_json_file}, {consolidated_md_file}")
    
    # Cleanup individual files
    for f in step1_dir.glob('page_*.*'):
        f.unlink()
    try:
        step1_dir.rmdir()
    except OSError:
        pass
    
    return consolidated_json_file, all_extractions

def step2_consolidate(client: genai.Client, model_option: ModelOption, llm_config: LLMConfig,
                     step1_file: Path, extractions: List[PageExtraction], 
                     output_dir: Path, magazine_id: str) -> Path:
    """
    Step 2: Magazine-level consolidation with fast Gemini model.
    Uses structured outputs for guaranteed JSON schema compliance.
    
    Args:
        client: Gemini client
        model_option: Model option for step 2
        llm_config: LLM configuration
        step1_file: Step 1 consolidated JSON file
        extractions: List of PageExtraction objects from step 1
        output_dir: Output directory
        magazine_id: Magazine identifier
        
    Returns:
        Path to the final consolidated file
    """
    model_name = summary_from_option(model_option)
    console.print(f"\n[bold cyan]Step 2:[/] Consolidating articles with [green]{model_name}[/]")
    console.print("[dim]Using structured outputs (JSON schema)[/]")
    logging.info(f"Step 2: Consolidating articles at magazine level with {model_name} (structured output)...")
    
    # Prepare extracted content as JSON for the model
    # This provides structured input that's easier to consolidate
    extracted_json = json.dumps([e.model_dump() for e in extractions], ensure_ascii=False, indent=2)
    
    # Load the consolidation prompt as system instruction
    consolidation_prompt = load_consolidation_prompt()
    # Remove the {extracted_content} placeholder - we'll pass JSON directly
    system_prompt = consolidation_prompt.split('{extracted_content}')[0].strip()
    # Add instruction about JSON input/output
    system_prompt += "\n\nL'entrée est fournie au format JSON structuré. Consolidez les articles et retournez le résultat au format JSON selon le schéma fourni."
    
    # Setup generation config with structured output
    config_kwargs = {
        "system_instruction": system_prompt,
        "temperature": llm_config.temperature or 0.3,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",  # Enable JSON mode
        "response_schema": MagazineIndex,  # Pydantic model for structured output
    }
    
    # Handle thinking config based on model type
    is_gemini_3 = "gemini-3" in model_option.model.lower()
    if is_gemini_3:
        thinking_level = llm_config.thinking_level or model_option.default_thinking_level or "low"
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
    else:
        thinking_budget = llm_config.thinking_budget
        if thinking_budget is None:
            thinking_budget = model_option.default_thinking_budget
        if thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    
    gen_config = types.GenerateContentConfig(**config_kwargs)
    
    # Generate consolidation with automatic retry
    try:
        with console.status("[cyan]Generating article index...", spinner="dots"):
            index = generate_consolidation_with_gemini(
                client, model_option.model, extracted_json, gen_config
            )
        
        if not index:
            console.print("[red]✗[/] Failed to generate consolidated output")
            logging.error("Failed to generate consolidated output")
            raise RuntimeError("Step 2 consolidation failed - no output generated")
        
        # Save the final result as JSON
        final_json_file = output_dir / f"{magazine_id}_final_index.json"
        with open(final_json_file, 'w', encoding='utf-8') as f:
            f.write(index.model_dump_json(indent=2))
        
        # Save the final result as markdown (for human readability)
        final_md_file = output_dir / f"{magazine_id}_final_index.md"
        with open(final_md_file, 'w', encoding='utf-8') as f:
            f.write(format_index_to_markdown(index))
        
        console.print(f"[green]✓[/] Step 2 complete → [dim]{final_md_file.name}[/]")
        logging.info(f"Step 2 complete. Final index: {final_md_file}")
        return final_md_file
        
    except Exception as e:
        console.print(f"[red]✗[/] Step 2 failed: {e}")
        logging.error(f"Step 2 failed after retries: {e}")
        raise

# ------------------------------------------------------------------
# Main Pipeline
# ------------------------------------------------------------------
def process_magazine(model_step1: ModelOption, model_step2: ModelOption,
                    pdf_path: Path, output_dir: Path, magazine_id: str = None):
    """
    Complete pipeline to process a magazine PDF using Gemini's native PDF understanding.
    
    Args:
        model_step1: Model option for step 1 (Gemini Pro)
        model_step2: Model option for step 2 (Gemini Flash)
        pdf_path: Path to PDF file
        output_dir: Output directory
        magazine_id: Magazine identifier (optional)
    """
    # Determine magazine identifier
    if magazine_id is None:
        magazine_id = pdf_path.stem
    
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
    client = genai.Client(api_key=api_key)
    
    # Configure for each step based on model type
    # Step 1: Gemini 3 Pro uses thinking_level, Gemini 2.5 uses thinking_budget
    is_step1_gemini_3 = "gemini-3" in model_step1.model.lower()
    if is_step1_gemini_3:
        config_step1 = LLMConfig(
            thinking_level="low",  # Use low thinking for faster processing
            temperature=0.2
        )
    else:
        config_step1 = LLMConfig(
            thinking_budget=500,  # Gemini 2.5 uses thinking_budget
            temperature=0.2
        )
    
    # Step 2: Gemini 2.5 Flash with thinking disabled for simple consolidation
    config_step2 = LLMConfig(
        thinking_budget=0,
        temperature=0.3
    )
    
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
        # Get page count
        with console.status("[cyan]Reading PDF structure...", spinner="dots"):
            total_pages = get_pdf_page_count(pdf_path)
        console.print(f"[green]✓[/] PDF has [bold]{total_pages}[/] pages")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Page-by-page extraction (Gemini Pro) with structured output
        # Returns both the consolidated file and the list of extractions
        step1_file, extractions = step1_extract_pages(
            client, model_step1, config_step1, pdf_path, total_pages,
            extraction_prompt, output_dir, magazine_id
        )
        
        # Step 2: Consolidation (Gemini Flash) with structured output
        final_file = step2_consolidate(
            client, model_step2, config_step2, step1_file, extractions, output_dir, magazine_id
        )
        
        logging.info(f"Pipeline complete for magazine {magazine_id}")
        logging.info(f"Final index: {final_file}")
        
        return final_file
        
    except Exception as e:
        console.print(f"[red]✗[/] Error processing PDF: {e}")
        logging.error(f"Error processing PDF {pdf_path}: {e}")
        raise

# ------------------------------------------------------------------
# User Interaction
# ------------------------------------------------------------------
def get_input_pdfs(script_dir: Path) -> list[Path]:
    """Get the list of all PDFs to process."""
    # Use the default PDF folder
    default_pdf_dir = script_dir / "PDF"
    
    if not default_pdf_dir.exists():
        console.print(f"[red]✗[/] PDF folder does not exist: {default_pdf_dir}")
        raise FileNotFoundError(f"PDF folder does not exist: {default_pdf_dir}")
    
    # Get all PDFs
    pdf_files = sorted(list(default_pdf_dir.glob('*.pdf')))
    
    if not pdf_files:
        console.print(f"[red]✗[/] No PDF files found in {default_pdf_dir}")
        raise FileNotFoundError(f"No PDF files found in {default_pdf_dir}")
    
    logging.info(f"{len(pdf_files)} PDF file(s) found in {default_pdf_dir}")
    return pdf_files

# ------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------
def main():
    """Main entry point of the script."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Verify Gemini API key
        if not os.getenv("GEMINI_API_KEY"):
            console.print(Panel(
                "[red]GEMINI_API_KEY not found in environment variables![/]\n\n"
                "Please set your API key in a .env file or environment.",
                title="✗ Configuration Error",
                border_style="red"
            ))
            return
        
        script_dir = Path(__file__).parent
        
        # Display welcome banner
        intro_text = (
            "[bold cyan]Islamic Magazine Article Extraction Pipeline[/]\n\n"
            "[dim]Using Gemini's native PDF understanding[/]\n\n"
            "📖 [white]Step 1:[/] Page-by-page extraction [dim](Gemini Pro)[/]\n"
            "📊 [white]Step 2:[/] Magazine-level consolidation [dim](Gemini Flash)[/]"
        )
        console.print(Panel(intro_text, title="🚀 Pipeline Started", border_style="cyan", padding=(1, 2)))
        
        logging.info("=== Magazine Article Extraction Pipeline ===")
        
        # Model selection (Gemini only)
        model_step1, model_step2 = get_model_pair()
        
        # Get the list of PDFs to process
        pdf_files = get_input_pdfs(script_dir)
        
        # Display file list
        file_table = Table(title=f"📚 Found {len(pdf_files)} PDF file(s)", box=box.ROUNDED)
        file_table.add_column("#", style="dim", width=4)
        file_table.add_column("Filename", style="white")
        file_table.add_column("Size", justify="right", style="cyan")
        
        for i, pdf_path in enumerate(pdf_files, 1):
            size_mb = pdf_path.stat().st_size / (1024 * 1024)
            file_table.add_row(str(i), pdf_path.name, f"{size_mb:.1f} MB")
        
        console.print(file_table)
        console.print()
        
        # Process each PDF
        success_count = 0
        error_count = 0
        
        for i, pdf_path in enumerate(pdf_files, 1):
            console.rule(f"[bold]PDF {i}/{len(pdf_files)}[/]", style="blue")
            
            logging.info(f"Processing PDF {i}/{len(pdf_files)}: {pdf_path.name}")
            
            # Use the filename as magazine ID
            magazine_id = pdf_path.stem
            
            # Define the output directory
            output_dir = script_dir / "Magazine_Extractions" / magazine_id
            
            try:
                # Execute the pipeline for this PDF
                process_magazine(model_step1, model_step2, pdf_path, output_dir, magazine_id)
                success_count += 1
                console.print(f"\n[green]✓[/] PDF {i}/{len(pdf_files)} completed: [bold]{pdf_path.name}[/]")
            except Exception as e:
                error_count += 1
                console.print(f"\n[red]✗[/] PDF {i}/{len(pdf_files)} failed: {pdf_path.name}")
                logging.error(f"Failed to process {pdf_path.name}: {e}")
        
        # Final summary
        console.print()
        summary_table = Table(box=box.ROUNDED, title="🏁 Pipeline Complete", title_style="bold green")
        summary_table.add_column("Status", style="bold")
        summary_table.add_column("Count", justify="right")
        summary_table.add_row("[green]✓ Processed[/]", str(success_count))
        if error_count > 0:
            summary_table.add_row("[red]✗ Failed[/]", str(error_count))
        summary_table.add_row("[cyan]Total[/]", str(len(pdf_files)))
        console.print(summary_table)
        
        logging.info(f"Pipeline completed: {success_count} success, {error_count} errors")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠[/] Process interrupted by user")
        logging.info("Process interrupted by user")
    except Exception as e:
        console.print(f"\n[red]✗ Pipeline failed:[/] {e}")
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
