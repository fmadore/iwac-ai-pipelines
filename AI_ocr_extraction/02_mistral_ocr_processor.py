"""
AI-Powered PDF OCR Processing using Mistral Document AI

This script performs high-precision OCR on PDF documents using Mistral's dedicated
OCR endpoint (mistral-ocr-latest). Unlike Gemini's multimodal approach, Mistral 
provides a specialized Document AI service optimized specifically for OCR tasks.

Usage:
    python 02_mistral_ocr_processor.py

Requirements:
    - Environment variable: MISTRAL_API_KEY
    - PDF files in the PDF/ directory
    - mistralai package

Key Features:
    - Uses dedicated mistral-ocr-latest model
    - Processes entire PDFs or specific page ranges
    - Returns text in Markdown format with preserved structure
    - Supports base64 encoded PDFs or file upload with signed URLs
    - Handles complex layouts including multi-column text and tables

Note:
    Mistral Document AI is a specialized OCR endpoint, not a chat/generation API.
    It does not accept system prompts - the OCR model extracts text as-is.
    This is different from Gemini which uses its multimodal capabilities.
"""

import os
import sys
import time
import base64
import logging
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
from PyPDF2 import PdfReader, PdfWriter
import io

# Import Mistral client
try:
    from mistralai import Mistral
except ImportError:
    raise RuntimeError("mistralai package is required. Install with: pip install mistralai")

# Set up logging configuration for tracking OCR operations and errors
script_dir = Path(__file__).parent
log_dir = script_dir / 'log'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / 'ocr_mistral.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file
)

# Mistral OCR model
MISTRAL_OCR_MODEL = "mistral-ocr-latest"


class MistralOCRProcessor:
    """
    A high-precision OCR system using Mistral's dedicated Document AI endpoint.
    
    This class implements an OCR pipeline that:
    1. Encodes PDF documents to base64 or uploads via File API
    2. Sends documents to Mistral's specialized OCR endpoint
    3. Extracts text in Markdown format with preserved structure
    4. Handles multi-page documents with page-level control
    
    Key differences from Gemini processor:
    - Uses dedicated OCR endpoint (client.ocr.process), not chat/generate
    - No system prompts - OCR model extracts text without instructions
    - Returns structured response with pages[].markdown format
    - Simpler API with built-in document handling
    """

    def __init__(self, api_key: str, process_page_by_page: bool = False):
        """
        Initialize the MistralOCRProcessor with API credentials.
        
        Args:
            api_key (str): Mistral API key for authentication
            process_page_by_page (bool): If True, process each page separately.
                                          If False, process entire PDF at once.
        """
        self.client = Mistral(api_key=api_key)
        self.model = MISTRAL_OCR_MODEL
        self.process_page_by_page = process_page_by_page
        
        print(f"🤖 Initialized Mistral OCR with model: {self.model}")
        print(f"📄 Processing mode: {'page-by-page' if process_page_by_page else 'whole document'}")

    def _encode_pdf_to_base64(self, pdf_path: Path) -> str:
        """
        Encode a PDF file to base64 string.
        
        Args:
            pdf_path (Path): Path to the PDF file
            
        Returns:
            str: Base64 encoded string of the PDF
        """
        try:
            with open(pdf_path, 'rb') as pdf_file:
                return base64.b64encode(pdf_file.read()).decode('utf-8')
        except Exception as e:
            logging.error(f"Error encoding PDF to base64: {e}")
            raise

    def _encode_page_to_base64(self, page_bytes: bytes) -> str:
        """
        Encode PDF page bytes to base64 string.
        
        Args:
            page_bytes (bytes): PDF page as bytes
            
        Returns:
            str: Base64 encoded string
        """
        return base64.b64encode(page_bytes).decode('utf-8')

    def extract_pdf_page(self, pdf_path: Path, page_number: int) -> bytes:
        """
        Extract a single page from a PDF as bytes.
        
        Args:
            pdf_path (Path): Path to the PDF file
            page_number (int): Page number to extract (0-indexed)
            
        Returns:
            bytes: PDF bytes containing only the specified page
        """
        try:
            reader = PdfReader(str(pdf_path))
            writer = PdfWriter()
            writer.add_page(reader.pages[page_number])
            
            output_buffer = io.BytesIO()
            writer.write(output_buffer)
            output_buffer.seek(0)
            
            return output_buffer.getvalue()
            
        except Exception as e:
            logging.error(f"Error extracting page {page_number + 1} from {pdf_path}: {e}")
            raise

    def get_pdf_page_count(self, pdf_path: Path) -> int:
        """
        Get the number of pages in a PDF.
        
        Args:
            pdf_path (Path): Path to the PDF file
            
        Returns:
            int: Number of pages in the PDF
        """
        try:
            reader = PdfReader(str(pdf_path))
            return len(reader.pages)
        except Exception as e:
            logging.error(f"Error reading PDF page count from {pdf_path}: {e}")
            raise

    def process_pdf_base64(self, pdf_path: Path, pages: Optional[List[int]] = None) -> Optional[str]:
        """
        Process a PDF using base64 encoding (no file upload required).
        
        This method encodes the entire PDF to base64 and sends it as a data URL.
        Suitable for PDFs under ~50MB.
        
        Args:
            pdf_path (Path): Path to the PDF file
            pages (Optional[List[int]]): Specific pages to process (0-indexed).
                                          If None, processes all pages.
            
        Returns:
            Optional[str]: Extracted text in Markdown format, or None if failed
        """
        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries):
            try:
                print(f"  └─ 📄 Encoding PDF to base64...")
                base64_pdf = self._encode_pdf_to_base64(pdf_path)
                
                # Create data URL for the PDF
                document_url = f"data:application/pdf;base64,{base64_pdf}"
                
                print(f"  └─ 🤖 Sending to Mistral OCR...")
                
                # Build OCR request
                ocr_params = {
                    "model": self.model,
                    "document": {
                        "type": "document_url",
                        "document_url": document_url
                    }
                }
                
                # Add page specification if provided
                if pages is not None:
                    ocr_params["pages"] = pages
                    print(f"  └─ 📃 Processing pages: {[p + 1 for p in pages]}")
                
                # Call OCR endpoint
                ocr_response = self.client.ocr.process(**ocr_params)
                
                # Extract text from response
                if not ocr_response.pages:
                    raise Exception("No pages in OCR response")
                
                # Combine all pages with markdown content
                text_parts = []
                for i, page in enumerate(ocr_response.pages):
                    if page.markdown:
                        text_parts.append(page.markdown)
                    else:
                        text_parts.append(f"[Empty page {i + 1}]")
                
                print(f"  └─ ✅ OCR complete - {len(ocr_response.pages)} pages processed")
                return "\n\n".join(text_parts)
                
            except Exception as e:
                print(f"  └─ ❌ Error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                logging.error(f"OCR processing error (attempt {attempt + 1}): {e}", exc_info=True)
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"  └─ 🔄 Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"  └─ ❌ Max retries reached.")
        
        return None

    def process_pdf_upload(self, pdf_path: Path) -> Optional[str]:
        """
        Process a PDF using file upload with signed URL.
        
        This method uploads the PDF to Mistral's file storage first,
        gets a signed URL, then processes. Better for larger files.
        
        Args:
            pdf_path (Path): Path to the PDF file
            
        Returns:
            Optional[str]: Extracted text in Markdown format, or None if failed
        """
        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries):
            try:
                print(f"  └─ ⬆️ Uploading PDF to Mistral...")
                
                # Upload the file
                uploaded_file = self.client.files.upload(
                    file={
                        "file_name": pdf_path.name,
                        "content": open(pdf_path, "rb"),
                    },
                    purpose="ocr"
                )
                
                if not uploaded_file or not uploaded_file.id:
                    raise Exception("Failed to upload PDF to Mistral")
                
                print(f"  └─ 🔗 Getting signed URL...")
                signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id)
                
                print(f"  └─ 🤖 Processing with Mistral OCR...")
                ocr_response = self.client.ocr.process(
                    model=self.model,
                    document={
                        "type": "document_url",
                        "document_url": signed_url.url,
                    }
                )
                
                # Extract text from response
                if not ocr_response.pages:
                    raise Exception("No pages in OCR response")
                
                # Combine all pages
                text_parts = []
                for i, page in enumerate(ocr_response.pages):
                    if page.markdown:
                        text_parts.append(page.markdown)
                    else:
                        text_parts.append(f"[Empty page {i + 1}]")
                
                print(f"  └─ ✅ OCR complete - {len(ocr_response.pages)} pages processed")
                return "\n\n".join(text_parts)
                
            except Exception as e:
                print(f"  └─ ❌ Error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                logging.error(f"OCR upload processing error (attempt {attempt + 1}): {e}", exc_info=True)
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"  └─ 🔄 Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"  └─ ❌ Max retries reached.")
        
        return None

    def process_page_base64(self, page_bytes: bytes, page_num: int) -> Optional[str]:
        """
        Process a single PDF page using base64 encoding.
        
        Args:
            page_bytes (bytes): PDF page as bytes
            page_num (int): Page number (for logging, 1-indexed)
            
        Returns:
            Optional[str]: Extracted text or None if failed
        """
        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries):
            try:
                print(f"  └─ 📄 Encoding page {page_num} to base64...")
                base64_page = self._encode_page_to_base64(page_bytes)
                document_url = f"data:application/pdf;base64,{base64_page}"
                
                print(f"  └─ 🤖 Processing page {page_num} with Mistral OCR...")
                
                ocr_response = self.client.ocr.process(
                    model=self.model,
                    document={
                        "type": "document_url",
                        "document_url": document_url
                    }
                )
                
                if not ocr_response.pages:
                    raise Exception("No pages in OCR response")
                
                # For single page, we expect exactly one page
                text = ocr_response.pages[0].markdown
                if not text:
                    text = f"[Empty page {page_num}]"
                
                print(f"  └─ ✅ Page {page_num} OCR complete")
                return text
                
            except Exception as e:
                print(f"  └─ ❌ Page {page_num} error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                logging.error(f"Page {page_num} OCR error (attempt {attempt + 1}): {e}", exc_info=True)
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"  └─ 🔄 Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"  └─ ❌ Page {page_num} max retries reached.")
        
        return None

    def process_pdf(self, pdf_path: Path, output_dir: Path) -> None:
        """
        Process a PDF file and save results to a text file.
        
        This method handles the complete OCR pipeline:
        1. Reads the PDF
        2. Processes via base64 encoding (or upload for large files)
        3. Saves extracted text to output file
        
        Args:
            pdf_path (Path): Path to the PDF file to process
            output_dir (Path): Directory to save the output text file
        """
        try:
            print("\n" + "="*50)
            print(f"📄 Processing PDF: {pdf_path.name}")
            print("="*50)
            
            # Verify PDF exists
            if not pdf_path.exists():
                print(f"❌ PDF file not found: {pdf_path}")
                logging.error(f"PDF file not found: {pdf_path}")
                return
            
            file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
            print(f"📊 PDF size: {file_size_mb:.2f} MB")
            
            # Get page count for logging
            total_pages = self.get_pdf_page_count(pdf_path)
            print(f"📃 Total pages: {total_pages}")
            
            # Create output file
            output_file = output_dir / f"{pdf_path.stem}.txt"
            print(f"📝 Output will be saved to: {output_file}")
            
            # Track processing
            successful_pages = 0
            failed_pages = []
            
            if self.process_page_by_page:
                # Process each page individually
                print("\n🔄 Processing page-by-page...")
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    for page_idx in range(total_pages):
                        page_num = page_idx + 1
                        
                        print(f"\n--- Page {page_num}/{total_pages} ---")
                        
                        try:
                            page_bytes = self.extract_pdf_page(pdf_path, page_idx)
                            text = self.process_page_base64(page_bytes, page_num)
                            
                            if text and text.strip():
                                if page_num == 1:
                                    f.write(text)
                                else:
                                    f.write(f"\n\n--- Page {page_num} ---\n\n")
                                    f.write(text)
                                successful_pages += 1
                            else:
                                failed_pages.append(page_num)
                                if page_num == 1:
                                    f.write(f"[ERROR: Failed to process page {page_num}]")
                                else:
                                    f.write(f"\n\n--- Page {page_num} ---\n\n[ERROR: Failed to process page {page_num}]")
                                    
                        except Exception as e:
                            failed_pages.append(page_num)
                            logging.error(f"Error processing page {page_num}: {e}")
                            if page_num == 1:
                                f.write(f"[ERROR: Failed to process page {page_num}: {str(e)}]")
                            else:
                                f.write(f"\n\n--- Page {page_num} ---\n\n[ERROR: {str(e)}]")
            else:
                # Process entire PDF at once
                print("\n🔄 Processing entire document...")
                
                # Choose method based on file size
                # Use upload for files > 20MB to avoid base64 overhead
                if file_size_mb > 20:
                    print(f"  └─ 📦 Large file - using upload method")
                    text = self.process_pdf_upload(pdf_path)
                else:
                    print(f"  └─ 📄 Using base64 encoding method")
                    text = self.process_pdf_base64(pdf_path)
                
                if text and text.strip():
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                    successful_pages = total_pages
                else:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write("[ERROR: Failed to process document]")
                    failed_pages = list(range(1, total_pages + 1))
            
            # Report statistics
            print("\n" + "="*50)
            print(f"📊 Processing Summary for {pdf_path.name}")
            print("="*50)
            print(f"Total pages: {total_pages}")
            print(f"Successfully processed: {successful_pages}")
            print(f"Failed pages: {len(failed_pages)}")
            if failed_pages:
                print(f"Failed page numbers: {failed_pages}")
            
            success_rate = (successful_pages / total_pages) * 100 if total_pages > 0 else 0
            print(f"Success rate: {success_rate:.1f}%")
            
            output_size = output_file.stat().st_size
            print(f"Output file size: {output_size:,} bytes")
            print("="*50 + "\n")
            
            logging.info(f"PDF {pdf_path.name}: {successful_pages}/{total_pages} pages successful ({success_rate:.1f}%)")
            if failed_pages:
                logging.warning(f"PDF {pdf_path.name}: Failed pages: {failed_pages}")
            
        except Exception as e:
            print(f"\n❌ Error processing PDF {pdf_path}: {e}")
            logging.error(f"Error processing PDF {pdf_path}: {e}", exc_info=True)


def main():
    """
    Main function to orchestrate the Mistral OCR process.
    
    Handles user interaction, configuration, and batch processing of PDFs.
    """
    print("\n🚀 Starting Mistral Document AI OCR Process")
    print("="*50)
    print("📖 This script uses Mistral's dedicated OCR endpoint")
    print("="*50)
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("❌ MISTRAL_API_KEY not found in environment variables!")
        return
    print("✅ API Key loaded successfully")
    
    # Ask user for processing mode
    print("\n📋 Processing Mode Selection")
    print("="*50)
    print("  1) Process entire PDF at once (faster, recommended)")
    print("  2) Process page-by-page (more control, slower)")
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            process_page_by_page = False
            break
        elif choice == "2":
            process_page_by_page = True
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    print(f"✅ Mode: {'Page-by-page' if process_page_by_page else 'Whole document'}")
    
    # Set up directory paths
    script_dir = Path(__file__).parent
    pdf_dir = script_dir / "PDF"
    output_dir = script_dir / "OCR_Results"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize the OCR processor
    print("\n🔧 Initializing Mistral OCR Processor...")
    processor = MistralOCRProcessor(api_key, process_page_by_page=process_page_by_page)
    
    # Find all PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("\n❌ No PDF files found in the PDF directory!")
        return
    
    total_pdfs = len(pdf_files)
    print(f"\n📚 Found {total_pdfs} PDF files to process")
    
    # Track statistics
    overall_stats = {
        'total_pdfs': total_pdfs,
        'processed_pdfs': 0,
        'failed_pdfs': 0,
        'total_size_mb': 0,
        'processing_start': time.time()
    }
    
    # Process each PDF
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n📊 Progress: PDF {idx}/{total_pdfs} ({(idx/total_pdfs*100):.1f}%)")
        
        try:
            processor.process_pdf(pdf_path, output_dir)
            
            output_file = output_dir / f"{pdf_path.stem}.txt"
            if output_file.exists() and output_file.stat().st_size > 100:
                overall_stats['processed_pdfs'] += 1
                overall_stats['total_size_mb'] += pdf_path.stat().st_size / (1024 * 1024)
                logging.info(f"Successfully processed {pdf_path.name}")
            else:
                overall_stats['failed_pdfs'] += 1
                logging.warning(f"Output file for {pdf_path.name} is empty or very small")
                
        except Exception as e:
            overall_stats['failed_pdfs'] += 1
            print(f"❌ Failed to process {pdf_path.name}: {e}")
            logging.error(f"Failed to process {pdf_path.name}: {e}")
    
    # Calculate processing time
    processing_time = time.time() - overall_stats['processing_start']
    
    # Print summary
    print("\n" + "="*60)
    print("📈 FINAL PROCESSING SUMMARY")
    print("="*60)
    print(f"Total PDFs found: {overall_stats['total_pdfs']}")
    print(f"Successfully processed: {overall_stats['processed_pdfs']}")
    print(f"Failed to process: {overall_stats['failed_pdfs']}")
    print(f"Total size processed: {overall_stats['total_size_mb']:.2f} MB")
    print(f"Processing time: {processing_time/60:.1f} minutes")
    
    if overall_stats['total_pdfs'] > 0:
        success_rate = (overall_stats['processed_pdfs'] / overall_stats['total_pdfs']) * 100
        print(f"Overall success rate: {success_rate:.1f}%")
    
    print("="*60)
    logging.info(f"Processing complete. {overall_stats['processed_pdfs']}/{overall_stats['total_pdfs']} PDFs processed in {processing_time/60:.1f} minutes")
    
    print("\n✨ Mistral Document AI OCR Process Complete! ✨\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
        logging.info("Process interrupted by user")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        logging.error(f"An error occurred: {e}", exc_info=True)
