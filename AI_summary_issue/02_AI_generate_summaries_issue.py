"""Islamic Magazine Article Extraction Pipeline (2-Step Process)

This script implements a two-step pipeline to extract and consolidate articles
from an Islamic magazine using Gemini's native PDF understanding.

Supported models:
- Gemini: Pro (step 1) + Flash (step 2)

Step 1: Page-by-page extraction (high-performance model)
- Uploads PDF once to Gemini's File API
- Analyzes each page individually with native PDF understanding
- Identifies articles present on the page
- Extracts exact titles and generates brief summaries
- Detects continuation indicators

Step 2: Magazine-level consolidation (fast model)
- Merges articles fragmented across multiple pages
- Eliminates duplicates with the fast model
- Produces a global summary per article
- Lists all associated pages

Robustness mechanisms:
- Automatic retry on error (max 3 attempts)
- Progressive result saving
- Resumption possible from already processed files

Advantages of native PDF processing:
- No PyPDF2 dependency needed
- Better document structure understanding
- Processes images, diagrams, and tables natively
- Simpler pipeline with fewer conversions

Usage:
    python 02_AI_generate_summaries_issue.py
"""

import os
import sys
import io
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
from PyPDF2 import PdfReader, PdfWriter

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
    """Load the prompt for step 1 (page-by-page extraction)."""
    script_dir = Path(__file__).parent
    prompt_file = script_dir / 'summary_prompt_issue.md'
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        if '{text}' not in content and '{page_number}' not in content:
            logging.warning("Prompt template missing required placeholders.")
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt template not found: {prompt_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to read prompt template {prompt_file}: {e}")

def load_consolidation_prompt() -> str:
    """Load the prompt for step 2 (consolidation)."""
    script_dir = Path(__file__).parent
    prompt_file = script_dir / 'consolidation_prompt_issue.md'
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        if '{extracted_content}' not in content:
            logging.warning("Consolidation prompt template missing '{extracted_content}' placeholder.")
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
    print("\n=== Model Selection for Pipeline ===")
    print("Step 1 (page-by-page extraction): Gemini Pro")
    print("Step 2 (consolidation): Gemini Flash\n")
    
    step1_option = get_model_option("gemini-pro")
    step2_option = get_model_option("gemini-flash")
    
    logging.info(f"\nSelected models: Gemini (Pro + Flash)")
    logging.info(f"  Step 1: {summary_from_option(step1_option)}")
    logging.info(f"  Step 2: {summary_from_option(step2_option)}")
    
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
                        page_num: int, prompt: str, config) -> Optional[str]:
    """
    Generate a response with Gemini for a single PDF page with automatic retry.
    
    Args:
        client: Gemini client
        model_name: Model name to use
        page_bytes: Single PDF page as bytes
        page_num: Page number (for logging)
        prompt: Prompt template to use
        config: Generation config
        
    Returns:
        Generated response or None
    """
    try:
        # Create PDF part from page bytes
        pdf_part = types.Part.from_bytes(
            data=page_bytes,
            mime_type='application/pdf'
        )
        
        # Generate content with single page PDF and prompt
        # Following Google's best practice: put document first, then prompt
        response = client.models.generate_content(
            model=model_name,
            contents=[pdf_part, prompt],
            config=config
        )
        
        # Validate response
        if not response.candidates:
            raise Exception("No candidates in Gemini response")
        
        candidate = response.candidates[0]
        
        if not candidate.content or not candidate.content.parts:
            finish_reason = candidate.finish_reason
            raise Exception(f"No valid response. Finish reason: {finish_reason}")
        
        text_content = response.text.replace('\xa0', ' ').strip().replace('*', '')
        if not text_content:
            raise Exception("Empty text response from Gemini")
        
        return text_content
        
    except Exception as e:
        logging.error(f"Generation error for page {page_num}: {e}")
        raise  # Let the retry decorator handle it

@retry_on_error(max_retries=MAX_RETRIES, delay=RETRY_DELAY)
def generate_consolidation_with_gemini(client: genai.Client, model_name: str,
                                      prompt: str, config) -> Optional[str]:
    """
    Generate consolidation response with Gemini with automatic retry.
    
    Args:
        client: Gemini client
        model_name: Model name to use
        prompt: Full prompt with extracted content
        config: Generation config
        
    Returns:
        Generated response or None
    """
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt],
            config=config
        )
        
        # Validate response
        if not response.candidates:
            raise Exception("No candidates in Gemini response")
        
        candidate = response.candidates[0]
        
        if not candidate.content or not candidate.content.parts:
            finish_reason = candidate.finish_reason
            raise Exception(f"No valid response. Finish reason: {finish_reason}")
        
        text_content = response.text.replace('\xa0', ' ').strip().replace('*', '')
        if not text_content:
            raise Exception("Empty text response from Gemini")
        
        return text_content
        
    except Exception as e:
        logging.error(f"Consolidation generation error: {e}")
        raise  # Let the retry decorator handle it

# ------------------------------------------------------------------
# Pipeline Functions
# ------------------------------------------------------------------
def step1_extract_pages(client: genai.Client, model_option: ModelOption, llm_config: LLMConfig,
                        pdf_path: Path, total_pages: int, extraction_prompt: str, 
                        output_dir: Path, magazine_id: str) -> Path:
    """
    Step 1: Page-by-page extraction with high-performance Gemini model.
    Progressive saving to allow resumption in case of interruption.
    
    Args:
        client: Gemini client
        model_option: Model option for step 1
        llm_config: LLM configuration
        pdf_path: Path to the PDF file
        total_pages: Total number of pages in the PDF
        extraction_prompt: Prompt template for extraction
        output_dir: Output directory
        magazine_id: Magazine identifier
        
    Returns:
        Path to the step 1 consolidated file
    """
    step1_dir = output_dir / "step1_page_extractions"
    step1_dir.mkdir(parents=True, exist_ok=True)
    
    all_extractions = []
    model_name = summary_from_option(model_option)
    
    # Setup generation config
    thinking_budget = llm_config.thinking_budget
    if thinking_budget is None:
        thinking_budget = model_option.default_thinking_budget
    
    config_kwargs = {
        "temperature": llm_config.temperature or 0.2,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    
    gen_config = types.GenerateContentConfig(**config_kwargs)
    
    logging.info(f"Step 1: Processing {total_pages} pages with {model_name}...")
    
    for page_num in tqdm(range(1, total_pages + 1), desc="Extracting articles per page"):
        page_file = step1_dir / f"page_{page_num:03d}.md"
        
        # Check if page has already been processed
        if page_file.exists():
            logging.info(f"Page {page_num} already processed, loading from cache...")
            with open(page_file, 'r', encoding='utf-8') as f:
                extraction = f.read()
            all_extractions.append(f"\n{extraction}\n")
            continue
        
        # Extract single page from PDF
        try:
            page_idx = page_num - 1  # 0-indexed for PyPDF2
            page_bytes = extract_pdf_page(pdf_path, page_idx)
            
            # Generate extraction with automatic retry using Gemini
            extraction = generate_with_gemini(
                client, model_option.model, page_bytes, page_num,
                extraction_prompt, gen_config
            )
            
            if extraction:
                # Add page number at the beginning of the AI response
                extraction_with_page = f"## Page : {page_num}\n\n{extraction}"
                all_extractions.append(f"\n{extraction_with_page}\n")
                
                # Save the individual extraction immediately
                with open(page_file, 'w', encoding='utf-8') as f:
                    f.write(extraction_with_page)
                logging.info(f"✓ Page {page_num} processed and saved")
            else:
                logging.error(f"✗ No extraction generated for page {page_num}")
                # Create a placeholder to avoid blocking the pipeline
                placeholder = f"## Page : {page_num}\n\nError processing this page.\n"
                all_extractions.append(f"\n{placeholder}\n")
                with open(page_file, 'w', encoding='utf-8') as f:
                    f.write(placeholder)
        except Exception as e:
            logging.error(f"✗ Failed to extract or process page {page_num} after retries: {e}")
            placeholder = f"## Page : {page_num}\n\nError: {str(e)}\n"
            all_extractions.append(f"\n{placeholder}\n")
            with open(page_file, 'w', encoding='utf-8') as f:
                f.write(placeholder)
    
    # Consolidate all extractions into a single file
    consolidated_file = output_dir / f"{magazine_id}_step1_consolidated.md"
    with open(consolidated_file, 'w', encoding='utf-8') as f:
        f.write(f"# Page-by-page extraction - Magazine {magazine_id}\n\n")
        f.write('\n---\n'.join(all_extractions))
    
    logging.info(f"Step 1 complete. Consolidated file: {consolidated_file}")
    
    # Delete individual files to save space
    logging.info("Cleaning up individual page files...")
    for page_file in step1_dir.glob('page_*.md'):
        page_file.unlink()
    logging.info(f"Deleted {len(list(step1_dir.glob('page_*.md')))} individual page files")
    
    # Optional: remove directory if empty
    try:
        step1_dir.rmdir()
        logging.info(f"Removed empty directory: {step1_dir}")
    except OSError:
        # Directory not empty, keep it
        pass
    
    return consolidated_file

def step2_consolidate(client: genai.Client, model_option: ModelOption, llm_config: LLMConfig,
                     step1_file: Path, output_dir: Path, magazine_id: str) -> Path:
    """
    Step 2: Magazine-level consolidation with fast Gemini model.
    
    Args:
        client: Gemini client
        model_option: Model option for step 2
        llm_config: LLM configuration
        step1_file: Step 1 consolidated file
        output_dir: Output directory
        magazine_id: Magazine identifier
        
    Returns:
        Path to the final consolidated file
    """
    model_name = summary_from_option(model_option)
    logging.info(f"Step 2: Consolidating articles at magazine level with {model_name}...")
    
    # Read the step 1 consolidated file
    with open(step1_file, 'r', encoding='utf-8') as f:
        extracted_content = f.read()
    
    # Load the consolidation prompt
    consolidation_prompt = load_consolidation_prompt()
    full_prompt = consolidation_prompt.replace('{extracted_content}', extracted_content)
    
    # Setup generation config
    thinking_budget = llm_config.thinking_budget
    if thinking_budget is None:
        thinking_budget = model_option.default_thinking_budget
    
    config_kwargs = {
        "temperature": llm_config.temperature or 0.3,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    
    gen_config = types.GenerateContentConfig(**config_kwargs)
    
    # Generate consolidation with automatic retry
    try:
        consolidated = generate_consolidation_with_gemini(
            client, model_option.model, full_prompt, gen_config
        )
        
        if not consolidated:
            logging.error("Failed to generate consolidated output")
            raise RuntimeError("Step 2 consolidation failed - no output generated")
        
        # Save the final result
        final_file = output_dir / f"{magazine_id}_final_index.md"
        with open(final_file, 'w', encoding='utf-8') as f:
            f.write(consolidated)
        
        logging.info(f"✓ Step 2 complete. Final index: {final_file}")
        return final_file
        
    except Exception as e:
        logging.error(f"✗ Step 2 failed after retries: {e}")
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
    
    # Configure for each step
    # Step 1: High-performance model with thinking enabled for detailed extraction
    config_step1 = LLMConfig(
        thinking_budget=500,
        temperature=0.2
    )
    
    # Step 2: Fast model with thinking disabled for simple consolidation
    config_step2 = LLMConfig(
        thinking_budget=0,
        temperature=0.3
    )
    
    # Load the extraction prompt
    extraction_prompt = load_extraction_prompt()
    
    # Get PDF info
    print("\n" + "="*50)
    print(f"📄 Processing PDF: {pdf_path.name}")
    print("="*50)
    
    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    print(f"📊 PDF size: {file_size_mb:.2f} MB")
    
    try:
        # Get page count
        print("🔄 Reading PDF structure...")
        total_pages = get_pdf_page_count(pdf_path)
        print(f"✅ PDF has {total_pages} pages")
        print("📄 Will extract and process each page individually")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Page-by-page extraction (Gemini Pro)
        # Each page will be extracted and sent individually to Gemini
        step1_file = step1_extract_pages(
            client, model_step1, config_step1, pdf_path, total_pages,
            extraction_prompt, output_dir, magazine_id
        )
        
        # Step 2: Consolidation (Gemini Flash)
        final_file = step2_consolidate(
            client, model_step2, config_step2, step1_file, output_dir, magazine_id
        )
        
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
        
        # Verify Gemini API key
        if not os.getenv("GEMINI_API_KEY"):
            print("❌ GEMINI_API_KEY not found in environment variables!")
            return
        
        script_dir = Path(__file__).parent
        
        print("\n🚀 Starting Magazine Article Extraction Pipeline")
        print("="*60)
        print("📖 Using Gemini's native PDF understanding")
        print("Step 1: Page-by-page extraction (Gemini Pro)")
        print("Step 2: Magazine-level consolidation (Gemini Flash)")
        print("="*60)
        
        logging.info("=== Magazine Article Extraction Pipeline ===")
        logging.info("Step 1: Page-by-page extraction (Gemini Pro)")
        logging.info("Step 2: Magazine-level consolidation (Gemini Flash)")
        logging.info("")
        
        # Model selection (Gemini only)
        model_step1, model_step2 = get_model_pair()
        
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
            process_magazine(model_step1, model_step2, pdf_path, output_dir, magazine_id)
            
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
