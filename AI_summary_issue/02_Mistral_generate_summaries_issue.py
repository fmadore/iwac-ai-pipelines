"""Islamic Magazine Article Extraction Pipeline using Mistral Document AI

This script implements a two-step pipeline to extract and consolidate articles
from an Islamic magazine using Mistral's native PDF understanding with OCR.

Supported models:
- Mistral OCR (Step 1: page-by-page extraction specialized for documents)
- Mistral Small (Step 2: cost-effective text consolidation)

Step 1: Page-by-page extraction with annotations
- Uploads PDF once to Mistral
- Analyzes each page individually with Pixtral Large
- Uses annotations to identify article boundaries and structure
- Extracts exact titles and generates brief summaries
- Detects continuation indicators

Step 2: Magazine-level consolidation
- Merges articles fragmented across multiple pages
- Eliminates duplicates with Mistral Large
- Produces a global summary per article
- Lists all associated pages

Robustness mechanisms:
- Automatic retry on error (max 3 attempts)
- Progressive result saving
- Resumption possible from already processed files

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
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

try:
    from mistralai import Mistral
except ImportError:
    raise RuntimeError("mistralai package is required. Install with: pip install mistralai")

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
                                    prompt: str) -> Optional[str]:
    """
    Generate page extraction with Mistral OCR API using document understanding.
    
    Args:
        client: Mistral client
        signed_url: Signed URL for the uploaded PDF
        page_num: Page number to analyze (1-indexed for display, 0-indexed for API)
        prompt: Extraction prompt (used in post-processing)
        
    Returns:
        Generated extraction text or None
    """
    try:
        # Call Mistral OCR API with signed URL and page number (0-indexed)
        # The OCR API returns raw OCR text, we'll need to process it
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
                logging.error(f"  - Available: {[attr for attr in dir(page_data) if not attr.startswith('_')]}")
        
        if not page_text:
            raise Exception("No text extracted from OCR")
        
        logging.info(f"Page {page_num}: Extracted {len(page_text)} characters via OCR")
        
        # Now use chat completion to analyze the OCR text with our prompt
        page_prompt = f"{prompt}\n\nPage Content:\n{page_text}"
        
        chat_response = client.chat.complete(
            model=MISTRAL_SMALL,  # Use Small model for analysis
            messages=[
                {
                    "role": "user",
                    "content": page_prompt
                }
            ]
        )
        
        if not chat_response.choices:
            raise Exception("No choices in chat response")
        
        text_content = chat_response.choices[0].message.content.strip()
        if not text_content:
            raise Exception("Empty text response from Mistral")
        
        return text_content
        
    except Exception as e:
        logging.error(f"Generation error for page {page_num}: {e}")
        raise

@retry_on_error(max_retries=MAX_RETRIES, delay=RETRY_DELAY)
def generate_consolidation_mistral(client: Mistral, prompt: str) -> Optional[str]:
    """
    Generate consolidation with Mistral Small (text-only, cost-effective).
    
    Args:
        client: Mistral client
        prompt: Full consolidation prompt with extracted content
        
    Returns:
        Generated consolidation text or None
    """
    try:
        response = client.chat.complete(
            model=MISTRAL_SMALL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        if not response.choices:
            raise Exception("No choices in Mistral response")
        
        text_content = response.choices[0].message.content.strip()
        if not text_content:
            raise Exception("Empty text response from Mistral")
        
        return text_content
        
    except Exception as e:
        logging.error(f"Consolidation generation error: {e}")
        raise

# ------------------------------------------------------------------
# Pipeline Functions
# ------------------------------------------------------------------
def step1_extract_pages(client: Mistral, signed_url: str, total_pages: int,
                        extraction_prompt: str, output_dir: Path, magazine_id: str) -> Path:
    """
    Step 1: Page-by-page extraction with Mistral OCR.
    Progressive saving to allow resumption in case of interruption.
    
    Args:
        client: Mistral client
        signed_url: Signed URL for the uploaded PDF
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
    
    logging.info(f"Step 1: Processing {total_pages} pages with Mistral OCR...")
    
    for page_num in tqdm(range(1, total_pages + 1), desc="Extracting articles per page"):
        page_file = step1_dir / f"page_{page_num:03d}.md"
        
        # Check if page has already been processed
        if page_file.exists():
            logging.info(f"Page {page_num} already processed, loading from cache...")
            with open(page_file, 'r', encoding='utf-8') as f:
                extraction = f.read()
            all_extractions.append(f"\n{extraction}\n")
            continue
        
        # Generate extraction with automatic retry using Mistral
        try:
            extraction = generate_page_extraction_mistral(
                client, signed_url, page_num, extraction_prompt
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
    
    # Optional: remove directory if empty
    try:
        step1_dir.rmdir()
        logging.info(f"Removed empty directory: {step1_dir}")
    except OSError:
        pass
    
    return consolidated_file

def step2_consolidate(client: Mistral, step1_file: Path, output_dir: Path,
                     magazine_id: str) -> Path:
    """
    Step 2: Magazine-level consolidation with Mistral Small.
    
    Args:
        client: Mistral client
        step1_file: Step 1 consolidated file
        output_dir: Output directory
        magazine_id: Magazine identifier
        
    Returns:
        Path to the final consolidated file
    """
    logging.info("Step 2: Consolidating articles at magazine level with Mistral Small...")
    
    # Read the step 1 consolidated file
    with open(step1_file, 'r', encoding='utf-8') as f:
        extracted_content = f.read()
    
    # Load the consolidation prompt
    consolidation_prompt = load_consolidation_prompt()
    full_prompt = consolidation_prompt.replace('{extracted_content}', extracted_content)
    
    # Generate consolidation with automatic retry
    try:
        consolidated = generate_consolidation_mistral(client, full_prompt)
        
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
        step1_file = step1_extract_pages(
            client, signed_url, total_pages, extraction_prompt, output_dir, magazine_id
        )
        
        # Step 2: Consolidation (Mistral Small)
        final_file = step2_consolidate(
            client, step1_file, output_dir, magazine_id
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
