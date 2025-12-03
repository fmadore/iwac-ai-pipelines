"""Islamic Magazine Article Extraction Pipeline using Mistral Document AI

This script implements a two-step pipeline to extract and consolidate articles
from an Islamic magazine using Mistral's native PDF understanding with OCR.

Supported models:
- Mistral OCR (Step 1: page-by-page extraction specialized for documents)
- Mistral Small (Step 2: cost-effective text consolidation with structured outputs)

Step 1: Page-by-page extraction with OCR
- Uploads PDF once to Mistral
- Analyzes each page individually with Mistral OCR
- Uses structured outputs (Pydantic models) for guaranteed JSON schema compliance
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
- Automatic retry on error (max 3 attempts)
- Progressive result saving with JSON caching
- Resumption possible from already processed files

API Best Practices:
- Uses client.chat.parse() with Pydantic response_format for structured outputs
- Uses Mistral OCR for document processing
- Uses Mistral Small for cost-effective consolidation

Usage:
    python 02_Mistral_generate_summaries_issue.py
    
Requirements:
    - MISTRAL_API_KEY environment variable
    - mistralai Python package
    - PDF files in PDF/ directory
"""

import os
import sys
import io
import json
import base64
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
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

try:
    from mistralai import Mistral
except ImportError:
    raise RuntimeError("mistralai package is required. Install with: pip install mistralai")

# ------------------------------------------------------------------
# Pydantic Models for Structured Outputs
# ------------------------------------------------------------------

class PageArticle(BaseModel):
    """An article found on a single page."""
    titre: str = Field(description="Titre exact de l'article tel qu'imprimé sur la page")
    auteurs: Optional[List[str]] = Field(default=None, description="Liste des auteurs de l'article (ex: ['Jean Dupont', 'La Rédaction']). Null si non mentionné.")
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
    auteurs: Optional[List[str]] = Field(default=None, description="Liste des auteurs de l'article. Null si non mentionné.")
    pages: str = Field(description="Numéros de pages, ex: '1-3' ou '1, 3, 5'")
    resume: str = Field(description="Résumé global consolidé de 4-6 phrases")

class MagazineIndex(BaseModel):
    """Final consolidated index of all articles in the magazine."""
    articles: List[ConsolidatedArticle] = Field(description="Liste des articles consolidés du magazine")

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
RETRY_BACKOFF = 2  # multiplier for exponential backoff

# Mistral models
MISTRAL_OCR = "mistral-ocr-latest"  # For OCR API endpoint
MISTRAL_SMALL = "mistral-small-latest"  # For chat completion consolidation

# ------------------------------------------------------------------
# Prompt Loading
# ------------------------------------------------------------------
def load_extraction_prompt() -> str:
    """Load the prompt for step 1 (page-by-page extraction)."""
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
        print(f"\n⬆️  Uploading PDF to Mistral: {pdf_path.name}")
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"📊 PDF size: {file_size_mb:.2f} MB")
        
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
        print(f"✅ PDF uploaded: {file_id}")
        logging.info(f"Uploaded PDF {pdf_path.name} with ID: {file_id}")
        
        # Get signed URL for OCR access
        print("🔗 Getting signed URL for OCR access...")
        signed_url_obj = client.files.get_signed_url(file_id=file_id)
        signed_url = signed_url_obj.url
        print(f"✅ Signed URL obtained")
        logging.info(f"Got signed URL for file {file_id}")
        
        return file_id, signed_url
        
    except Exception as e:
        logging.error(f"Error uploading PDF to Mistral: {e}")
        raise

def get_pdf_page_count(pdf_path: Path) -> int:
    """
    Get the number of pages in a PDF using PyPDF2.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Number of pages in the PDF
    """
    try:
        from PyPDF2 import PdfReader
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
def generate_page_extraction_mistral(client: Mistral, signed_url: str, page_num: int,
                                    prompt: str) -> Optional[PageExtraction]:
    """
    Generate structured page extraction with Mistral OCR API.
    
    Args:
        client: Mistral client
        signed_url: Signed URL for the uploaded PDF
        page_num: Page number to analyze (1-indexed for display, 0-indexed for API)
        prompt: Extraction prompt (used as system instruction)
        
    Returns:
        PageExtraction object or None
    """
    try:
        # Call Mistral OCR API with signed URL and page number (0-indexed)
        response = client.ocr.process(
            model=MISTRAL_OCR,
            pages=[page_num - 1],  # API uses 0-indexed pages
            document={
                "type": "document_url",
                "document_url": signed_url
            },
            include_image_base64=False
        )
        
        # Extract OCR text from response
        if not response or not hasattr(response, 'pages'):
            raise Exception("Invalid OCR response structure")
        
        # Get text from the page
        page_text = ""
        if len(response.pages) > 0:
            page_data = response.pages[0]
            
            # Mistral OCR returns markdown text
            if hasattr(page_data, 'markdown') and page_data.markdown:
                page_text = page_data.markdown
                logging.info(f"  - Extracted {len(page_text)} chars from markdown")
            else:
                logging.error(f"  - No markdown attribute or empty content")
        
        if not page_text:
            raise Exception("No text extracted from OCR")
        
        logging.info(f"Page {page_num}: Extracted {len(page_text)} characters via OCR")
        
        # Now use chat.parse() with structured output to analyze the OCR text
        chat_response = client.chat.parse(
            model=MISTRAL_SMALL,
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": f"Analysez le contenu de la page {page_num}:\n\n{page_text}"
                }
            ],
            response_format=PageExtraction,
            temperature=0.2
        )
        
        if not chat_response.choices:
            raise Exception("No choices in chat response")
        
        # Get the parsed structured response
        parsed = chat_response.choices[0].message.parsed
        if parsed is None:
            raise Exception("No parsed response from Mistral")
        
        # Ensure page number is set correctly
        parsed.page_number = page_num
        return parsed
        
    except Exception as e:
        logging.error(f"Generation error for page {page_num}: {e}")
        raise

@retry_on_error(max_retries=MAX_RETRIES, delay=RETRY_DELAY)
def generate_consolidation_mistral(client: Mistral, system_prompt: str, 
                                   extracted_json: str) -> Optional[MagazineIndex]:
    """
    Generate structured consolidation with Mistral Small using chat.parse().
    
    Args:
        client: Mistral client
        system_prompt: System prompt for consolidation
        extracted_json: JSON string of extracted page data
        
    Returns:
        MagazineIndex object or None
    """
    try:
        response = client.chat.parse(
            model=MISTRAL_SMALL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Consolidez les articles extraits ci-dessous:\n\n{extracted_json}"
                }
            ],
            response_format=MagazineIndex,
            temperature=0.3
        )
        
        if not response.choices:
            raise Exception("No choices in Mistral response")
        
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise Exception("No parsed response from Mistral")
        
        return parsed
        
    except Exception as e:
        logging.error(f"Consolidation generation error: {e}")
        raise

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
        lines.append(f"- Titre : {article.titre}")
        if article.auteurs:
            lines.append(f"- Auteur(s) : {', '.join(article.auteurs)}")
        lines.append(f"- Résumé :\n  {article.resume}")
    
    if extraction.other_content:
        lines.append(f"\n### Autres contenus\n{extraction.other_content}")
    
    return "\n".join(lines)

def format_index_to_markdown(index: MagazineIndex) -> str:
    """Convert a MagazineIndex to the final markdown format."""
    lines = ["# Index des articles du magazine\n"]
    
    for article in index.articles:
        lines.append(f"\n## {article.titre}")
        if article.auteurs:
            lines.append(f"- Auteur(s) : {', '.join(article.auteurs)}")
        lines.append(f"- Pages : {article.pages}")
        lines.append(f"- Résumé :")
        lines.append(f"  {article.resume}")
    
    return "\n".join(lines)

# ------------------------------------------------------------------
# Pipeline Functions
# ------------------------------------------------------------------
def step1_extract_pages(client: Mistral, signed_url: str, total_pages: int,
                        extraction_prompt: str, output_dir: Path, magazine_id: str) -> Tuple[Path, List[PageExtraction]]:
    """
    Step 1: Page-by-page extraction with Mistral OCR and structured outputs.
    Progressive saving to allow resumption in case of interruption.
    
    Args:
        client: Mistral client
        signed_url: Signed URL for the uploaded PDF
        total_pages: Total number of pages in the PDF
        extraction_prompt: Prompt template for extraction
        output_dir: Output directory
        magazine_id: Magazine identifier
        
    Returns:
        Tuple of (consolidated JSON file path, list of PageExtraction objects)
    """
    step1_dir = output_dir / "step1_page_extractions"
    step1_dir.mkdir(parents=True, exist_ok=True)
    
    all_extractions: List[PageExtraction] = []
    all_markdown: List[str] = []
    
    console.print(f"\n[bold cyan]Step 1:[/] Processing {total_pages} pages with [green]Mistral OCR[/]")
    console.print("[dim]Using structured outputs (Pydantic schema)[/]")
    logging.info(f"Step 1: Processing {total_pages} pages with Mistral OCR (structured output)...")
    
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
            
            # Generate structured extraction with automatic retry using Mistral
            try:
                extraction = generate_page_extraction_mistral(
                    client, signed_url, page_num, extraction_prompt
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
                    placeholder = PageExtraction(
                        page_number=page_num,
                        has_articles=False,
                        other_content="Error processing this page."
                    )
                    all_extractions.append(placeholder)
                    all_markdown.append(format_extraction_to_markdown(placeholder))
                    error_count += 1
                    
            except Exception as e:
                logging.error(f"Failed to process page {page_num} after retries: {e}")
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

def step2_consolidate(client: Mistral, step1_file: Path, extractions: List[PageExtraction],
                     output_dir: Path, magazine_id: str) -> Path:
    """
    Step 2: Magazine-level consolidation with Mistral Small and structured outputs.
    
    Args:
        client: Mistral client
        step1_file: Step 1 consolidated JSON file
        extractions: List of PageExtraction objects from step 1
        output_dir: Output directory
        magazine_id: Magazine identifier
        
    Returns:
        Path to the final consolidated file
    """
    console.print(f"\n[bold cyan]Step 2:[/] Consolidating articles with [green]Mistral Small[/]")
    console.print("[dim]Using structured outputs (Pydantic schema)[/]")
    logging.info("Step 2: Consolidating articles at magazine level with Mistral Small (structured output)...")
    
    # Prepare extracted content as JSON for the model
    extracted_json = json.dumps([e.model_dump() for e in extractions], ensure_ascii=False, indent=2)
    
    # Load the consolidation prompt as system instruction
    consolidation_prompt = load_consolidation_prompt()
    # Remove the {extracted_content} placeholder - we'll pass JSON directly
    system_prompt = consolidation_prompt.split('{extracted_content}')[0].strip()
    # Add instruction about JSON input/output
    system_prompt += "\n\nL'entrée est fournie au format JSON structuré. Consolidez les articles et retournez le résultat au format JSON selon le schéma fourni."
    
    # Generate consolidation with automatic retry
    try:
        with console.status("[cyan]Generating article index...", spinner="dots"):
            index = generate_consolidation_mistral(client, system_prompt, extracted_json)
        
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
def process_magazine(pdf_path: Path, output_dir: Path, magazine_id: str = None):
    """
    Complete pipeline to process a magazine PDF using Mistral's Document AI.
    
    Args:
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
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not found in environment variables")
    
    # Initialize Mistral client
    client = Mistral(api_key=api_key)
    
    # Load the extraction prompt
    extraction_prompt = load_extraction_prompt()
    
    try:
        # Upload PDF to Mistral and get signed URL
        file_id, signed_url = upload_pdf_to_mistral(client, pdf_path)
        
        # Get page count
        print("\n🔄 Reading PDF structure...")
        total_pages = get_pdf_page_count(pdf_path)
        print(f"✅ PDF has {total_pages} pages")
        print("📄 Will process each page with Mistral Document AI")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Page-by-page extraction (Mistral OCR)
        step1_file, extractions = step1_extract_pages(
            client, signed_url, total_pages, extraction_prompt, output_dir, magazine_id
        )
        
        # Step 2: Consolidation (Mistral Small)
        final_file = step2_consolidate(
            client, step1_file, extractions, output_dir, magazine_id
        )
        
        # Cleanup: delete uploaded file from Mistral cloud
        try:
            print("\n🗑️  Cleaning up uploaded file...")
            client.files.delete(file_id=file_id)
            print("✅ File deleted from Mistral cloud")
            logging.info(f"Deleted file {file_id} from Mistral cloud")
        except Exception as e:
            logging.warning(f"Failed to delete file {file_id}: {e}")
        
        logging.info(f"✓ Pipeline complete for magazine {magazine_id}")
        logging.info(f"✓ Final index: {final_file}")
        
        return final_file
        
    except Exception as e:
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
        raise FileNotFoundError(f"PDF folder does not exist: {default_pdf_dir}")
    
    # Get all PDFs
    pdf_files = sorted(list(default_pdf_dir.glob('*.pdf')))
    
    if not pdf_files:
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
        
        # Verify Mistral API key
        if not os.getenv("MISTRAL_API_KEY"):
            print("❌ MISTRAL_API_KEY not found in environment variables!")
            return
        
        script_dir = Path(__file__).parent
        
        print("\n🚀 Starting Magazine Article Extraction Pipeline (Mistral)")
        print("="*60)
        print("📖 Using Mistral Document AI with OCR capabilities")
        print("Step 1: Page-by-page extraction (Mistral OCR)")
        print("Step 2: Magazine-level consolidation (Mistral Small)")
        print("="*60)
        
        logging.info("=== Magazine Article Extraction Pipeline (Mistral) ===")
        logging.info("Step 1: Page-by-page extraction (Mistral OCR)")
        logging.info("Step 2: Magazine-level consolidation (Mistral Small)")
        logging.info("")
        
        # Get the list of PDFs to process
        pdf_files = get_input_pdfs(script_dir)
        
        print(f"\n📚 Found {len(pdf_files)} PDF file(s) to process\n")
        
        # Process each PDF
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\n{'='*60}")
            print(f"📊 Processing PDF {i}/{len(pdf_files)}: {pdf_path.name}")
            print(f"{'='*60}")
            
            logging.info(f"\n{'='*60}")
            logging.info(f"Processing PDF {i}/{len(pdf_files)}: {pdf_path.name}")
            logging.info(f"{'='*60}")
            
            # Use the filename as magazine ID
            magazine_id = pdf_path.stem
            logging.info(f"Magazine ID: {magazine_id}")
            
            # Define the output directory
            output_dir = script_dir / "Magazine_Extractions" / magazine_id
            
            # Execute the pipeline for this PDF
            process_magazine(pdf_path, output_dir, magazine_id)
            
            print(f"\n✅ PDF {i}/{len(pdf_files)} completed: {pdf_path.name}")
            logging.info(f"PDF {i}/{len(pdf_files)} completed: {pdf_path.name}")
        
        print(f"\n{'='*60}")
        print(f"✨ Pipeline completed successfully - {len(pdf_files)} magazine(s) processed")
        print(f"{'='*60}\n")
        
        logging.info(f"\n{'='*60}")
        logging.info(f"=== Pipeline completed successfully - {len(pdf_files)} magazine(s) processed ===")
        logging.info(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        logging.info("Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
