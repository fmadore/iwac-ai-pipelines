"""Islamic Magazine Article Extraction Pipeline (2-Step Process)

This script implements a two-step pipeline to extract and consolidate articles
from an Islamic magazine using page-by-page PDF or OCR text files.

Supported models:
- Gemini: Pro (step 1) + Flash (step 2)
- OpenAI: GPT-5.1 full (step 1) + GPT-5.1 mini (step 2)

Step 1: Page-by-page extraction (high-performance model)
- Analyzes each page individually with the most capable model
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

Usage:
    python 02_AI_generate_summaries_issue.py
"""

import os
import sys
import re
import json
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
import PyPDF2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from common.llm_provider import (  # noqa: E402
    BaseLLMClient,
    LLMConfig,
    ModelOption,
    build_llm_client,
    get_model_option,
    summary_from_option,
)

try:
    from google import genai
    from google.genai import types, errors
except ImportError:
    genai = None
    types = None
    errors = None

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
    """Allow the user to select the model pair for the pipeline.
    
    Returns:
        Tuple of (model_step1, model_step2) where step1 is the high-performance model
        and step2 is the fast model.
    """
    print("\n=== Model Selection for Pipeline ===")
    print("Step 1 (page-by-page extraction): High-performance model")
    print("Step 2 (consolidation): Fast model\n")
    
    # Define available pairs
    model_pairs = {
        "gemini": {
            "step1": "gemini-pro",
            "step2": "gemini-flash",
            "name": "Gemini (Pro + Flash)"
        },
        "openai": {
            "step1": "openai-5.1",
            "step2": "openai",
            "name": "OpenAI (GPT-5.1 full + mini)"
        }
    }
    
    print("Available model pairs:")
    print("  1) Gemini (Pro + Flash) - Pro for extraction, Flash for consolidation")
    print("  2) OpenAI (GPT-5.1 full + mini) - Full for extraction, mini for consolidation")
    
    while True:
        choice = input("\nChoose model pair (1 or 2): ").strip()
        if choice == "1":
            pair_key = "gemini"
            break
        elif choice == "2":
            pair_key = "openai"
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    pair = model_pairs[pair_key]
    step1_option = get_model_option(pair["step1"])
    step2_option = get_model_option(pair["step2"])
    
    logging.info(f"\nSelected models: {pair['name']}")
    logging.info(f"  Step 1: {summary_from_option(step1_option)}")
    logging.info(f"  Step 2: {summary_from_option(step2_option)}")
    
    return step1_option, step2_option

# ------------------------------------------------------------------
# PDF Processing
# ------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: Path) -> Dict[int, str]:
    """
    Extract text from each page of a PDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary {page_number: text}
    """
    pages_text = {}
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                pages_text[page_num + 1] = text  # Numbering starts from 1
                
        logging.info(f"Extracted text from {total_pages} pages in {pdf_path.name}")
        return pages_text
    except Exception as e:
        logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return {}

def load_txt_files_as_pages(txt_dir: Path) -> Dict[int, str]:
    """
    Load numbered TXT files as pages.
    Expects files named like: page_1.txt, page_2.txt, etc.
    Or simply all .txt files in alphabetical order.
    
    Args:
        txt_dir: Directory containing TXT files
        
    Returns:
        Dictionary {page_number: text}
    """
    pages_text = {}
    try:
        txt_files = sorted([f for f in txt_dir.glob('*.txt')])
        if not txt_files:
            logging.warning(f"No TXT files found in {txt_dir}")
            return {}
        
        for idx, txt_file in enumerate(txt_files, start=1):
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Try to extract page number from filename
            match = re.search(r'page[_\s]?(\d+)', txt_file.stem, re.IGNORECASE)
            if match:
                page_num = int(match.group(1))
            else:
                page_num = idx  # Use alphabetical order
            
            pages_text[page_num] = text
        
        logging.info(f"Loaded {len(pages_text)} pages from TXT files in {txt_dir}")
        return pages_text
    except Exception as e:
        logging.error(f"Error loading TXT files from {txt_dir}: {e}")
        return {}

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
def generate_with_llm(llm_client: BaseLLMClient, prompt: str) -> Optional[str]:
    """
    Generate a response with the LLM client with automatic retry.
    
    Args:
        llm_client: LLM client (OpenAI or Gemini)
        prompt: Prompt to send
        
    Returns:
        Generated response or None
    """
    if not prompt.strip():
        return None
    
    try:
        response = llm_client.generate(
            system_prompt="",  # The prompt is complete in user_prompt
            user_prompt=prompt
        )
        
        if response:
            return response.strip().replace('*', '')
        
        logging.error("Model returned empty response.")
        return None
    except Exception as e:
        logging.error(f"Generation error: {e}")
        raise  # Let the retry decorator handle it

# ------------------------------------------------------------------
# Pipeline Functions
# ------------------------------------------------------------------
def step1_extract_pages(llm_client: BaseLLMClient, pages_text: Dict[int, str], 
                        extraction_prompt: str, output_dir: Path, magazine_id: str,
                        model_name: str) -> Path:
    """
    Step 1: Page-by-page extraction with high-performance model.
    Progressive saving to allow resumption in case of interruption.
    
    Args:
        llm_client: Configured LLM client
        pages_text: Dictionary {page_num: text}
        extraction_prompt: Prompt template for extraction
        output_dir: Output directory
        magazine_id: Magazine identifier
        model_name: Model name for logging
        
    Returns:
        Path to the step 1 consolidated file
    """
    step1_dir = output_dir / "step1_page_extractions"
    step1_dir.mkdir(parents=True, exist_ok=True)
    
    all_extractions = []
    
    logging.info(f"Step 1: Processing {len(pages_text)} pages with {model_name}...")
    
    for page_num in tqdm(sorted(pages_text.keys()), desc="Extracting articles per page"):
        page_file = step1_dir / f"page_{page_num:03d}.md"
        
        # Check if page has already been processed
        if page_file.exists():
            logging.info(f"Page {page_num} already processed, loading from cache...")
            with open(page_file, 'r', encoding='utf-8') as f:
                extraction = f.read()
            all_extractions.append(f"\n{extraction}\n")
            continue
        
        page_text = pages_text[page_num]
        
        # Prepare the prompt with text (no page number)
        prompt = extraction_prompt.replace('{text}', page_text)
        
        # Generate extraction with automatic retry
        try:
            extraction = generate_with_llm(llm_client, prompt)
            
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
            logging.error(f"✗ Failed to process page {page_num} after retries: {e}")
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

def step2_consolidate(llm_client: BaseLLMClient, step1_file: Path, 
                     output_dir: Path, magazine_id: str, model_name: str) -> Path:
    """
    Step 2: Magazine-level consolidation with fast model.
    
    Args:
        llm_client: Configured LLM client
        step1_file: Step 1 consolidated file
        output_dir: Output directory
        magazine_id: Magazine identifier
        model_name: Model name for logging
        
    Returns:
        Path to the final consolidated file
    """
    logging.info(f"Step 2: Consolidating articles at magazine level with {model_name}...")
    
    # Read the step 1 consolidated file
    with open(step1_file, 'r', encoding='utf-8') as f:
        extracted_content = f.read()
    
    # Load the consolidation prompt
    consolidation_prompt = load_consolidation_prompt()
    full_prompt = consolidation_prompt.replace('{extracted_content}', extracted_content)
    
    # Generate consolidation with automatic retry
    try:
        consolidated = generate_with_llm(llm_client, full_prompt)
        
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
                    input_path: Path, output_dir: Path, magazine_id: str = None):
    """
    Complete pipeline to process a magazine.
    
    Args:
        model_step1: Model option for step 1 (high-performance)
        model_step2: Model option for step 2 (fast)
        input_path: Path to PDF or TXT directory
        output_dir: Output directory
        magazine_id: Magazine identifier (optional)
    """
    # Determine magazine identifier
    if magazine_id is None:
        magazine_id = input_path.stem
    
    logging.info(f"Processing magazine: {magazine_id}")
    logging.info(f"Input: {input_path}")
    logging.info(f"Output: {output_dir}")
    
    # Configure LLM clients for each step
    # Step 1: High-performance model with medium reasoning for detailed extraction
    config_step1 = LLMConfig(
        reasoning_effort="medium",
        text_verbosity="medium",
        thinking_budget=500,
        temperature=0.2
    )
    llm_client_step1 = build_llm_client(model_step1, config=config_step1)
    
    # Step 2: Fast model with low reasoning for simple consolidation
    config_step2 = LLMConfig(
        reasoning_effort="low",
        text_verbosity="low",
        thinking_budget=0,
        temperature=0.3
    )
    llm_client_step2 = build_llm_client(model_step2, config=config_step2)
    
    # Load the extraction prompt
    extraction_prompt = load_extraction_prompt()
    
    # Extract text page by page
    if input_path.is_file() and input_path.suffix.lower() == '.pdf':
        logging.info("Input is PDF file - extracting text...")
        pages_text = extract_text_from_pdf(input_path)
    elif input_path.is_dir():
        logging.info("Input is directory - loading TXT files...")
        pages_text = load_txt_files_as_pages(input_path)
    else:
        raise ValueError(f"Invalid input path: {input_path}. Must be PDF file or directory with TXT files.")
    
    if not pages_text:
        raise RuntimeError("No pages extracted from input")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Page-by-page extraction (high-performance model)
    step1_file = step1_extract_pages(
        llm_client_step1, pages_text, extraction_prompt, output_dir, magazine_id,
        summary_from_option(model_step1)
    )
    
    # Step 2: Consolidation (fast model)
    final_file = step2_consolidate(
        llm_client_step2, step1_file, output_dir, magazine_id,
        summary_from_option(model_step2)
    )
    
    logging.info(f"✓ Pipeline complete for magazine {magazine_id}")
    logging.info(f"✓ Final index: {final_file}")
    
    return final_file

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
        script_dir = Path(__file__).parent
        
        logging.info("=== Magazine Article Extraction Pipeline ===")
        logging.info("Step 1: Page-by-page extraction (high-performance model)")
        logging.info("Step 2: Magazine-level consolidation (fast model)")
        logging.info("")
        
        # Model selection
        model_step1, model_step2 = get_model_pair()
        
        # Get the list of PDFs to process
        pdf_files = get_input_pdfs(script_dir)
        
        # Process each PDF
        for i, pdf_path in enumerate(pdf_files, 1):
            logging.info(f"\n{'='*60}")
            logging.info(f"Processing PDF {i}/{len(pdf_files)}: {pdf_path.name}")
            logging.info(f"{'='*60}")
            
            # Use the filename as Omeka ID
            omeka_id = pdf_path.stem
            logging.info(f"Omeka ID: {omeka_id}")
            
            # Define the output directory
            output_dir = script_dir / "Magazine_Extractions" / omeka_id
            
            # Execute the pipeline for this PDF
            process_magazine(model_step1, model_step2, pdf_path, output_dir, omeka_id)
            
            logging.info(f"PDF {i}/{len(pdf_files)} completed: {pdf_path.name}")
        
        logging.info(f"\n{'='*60}")
        logging.info(f"=== Pipeline completed successfully - {len(pdf_files)} magazine(s) processed ===")
        logging.info(f"{'='*60}")
        
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
