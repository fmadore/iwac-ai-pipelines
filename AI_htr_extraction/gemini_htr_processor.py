"""
AI-Powered HTR (Handwritten Text Recognition) Script using Google Gemini

This script performs high-precision transcription of HANDWRITTEN documents (French, Arabic, or multilingual)
using Google's Gemini vision model with direct PDF processing. It is engineered for research-grade,
archival-quality extraction while preserving correct typography, reading order, and structural semantics.

Usage:
    python gemini_htr_processor.py

Requirements:
    - Environment variable: GEMINI_API_KEY
    - PDF files in the PDF/ directory
    - HTR system prompt files (htr_system_prompt_french.md, htr_system_prompt_arabic.md, etc.)
    - PyPDF2 for page extraction
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv
from PyPDF2 import PdfReader, PdfWriter
import io

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

    def __init__(self, api_key: str, model_name: str, language: str = "french"):
        """
        Initialize the GeminiHTR system with API credentials and model name.
        
        Args:
            api_key (str): Google Gemini API key for authentication
            model_name (str): The Gemini model name to use
            language (str): Language of the manuscripts ("french", "arabic", or "multilingual")
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.language = language
        self.generation_config = self._setup_generation_config()
        
    def _setup_generation_config(self):
        """
        Configure generation parameters for optimal HTR performance.
        
        The configuration focuses on:
        - Lower temperature for more consistent output
        - High top_p and top_k for reliable text recognition
        - Sufficient output tokens for long documents
        - Model-appropriate thinking budget
        
        Returns:
            types.GenerateContentConfig: Configured generation config
        """
        # Set thinking budget based on model capabilities
        if "2.5-pro" in self.model_name.lower():
            # Gemini 2.5 Pro requires thinking mode (minimum budget 128)
            thinking_budget = 128  # Minimum for Pro model
            print(f"🧠 Using thinking budget {thinking_budget} for {self.model_name}")
        else:
            # Gemini 2.5 Flash can disable thinking for simple tasks
            thinking_budget = 0  # Disable thinking for faster HTR
            print(f"🧠 Disabling thinking mode for {self.model_name}")
        
        return types.GenerateContentConfig(
            temperature=0.1,      # Lower temperature for more consistent output
            top_p=0.95,          # High top_p for reliable text recognition
            top_k=40,            # Balanced top_k for good candidate selection
            max_output_tokens=65535,  # Support for long documents
            response_mime_type="text/plain",  # Ensure text output
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                )
            ]
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
            raise FileNotFoundError(f"HTR system prompt file not found at {prompt_file}")
        except Exception as e:
            logging.error(f"Error reading system prompt file: {e}")
            raise
    
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
            print(f"  └─ 📄 Processing page {page_num} inline...")
            
            # Create PDF part from page bytes
            pdf_part = types.Part.from_bytes(
                data=page_bytes,
                mime_type='application/pdf'
            )
            
            print(f"  └─ 🤖 Generating HTR text for page {page_num}...")
            
            # Following Google's best practice: put prompt AFTER the document
            if self.language == "multilingual":
                language_desc = "text (detect language automatically)"
            elif self.language == "arabic":
                language_desc = "Arabic"
            else:
                language_desc = "French"
            
            combined_prompt = (
                self._get_system_instruction() + "\n\n" +
                f"This is a legitimate handwritten text transcription (HTR) request for academic research and archival preservation. "
                f"Transcribe ALL handwritten {language_desc} text with exact wording, spacing rules, accents, and WITHOUT summarizing or omitting any zones."
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[pdf_part, combined_prompt],  # Document first, then prompt
                config=self.generation_config
            )
            
            # Validate response
            if not response.candidates:
                raise Exception("No candidates in Gemini response")
            
            candidate = response.candidates[0]
            
            if not candidate.content or not candidate.content.parts:
                finish_reason = candidate.finish_reason
                if finish_reason == types.FinishReason.RECITATION:
                    print(f"  └─ ⚠️ Page {page_num}: Copyright detection triggered, trying alternative...")
                    return self._try_alternative_prompts(pdf_part, page_num)
                else:
                    raise Exception(f"No valid response. Finish reason: {finish_reason}")
            
            text_content = response.text.replace('\xa0', ' ').strip()
            if not text_content:
                raise Exception("Empty text response from Gemini")
            
            print(f"  └─ ✅ Page {page_num} HTR complete")
            return text_content
            
        except Exception as e:
            print(f"  └─ ❌ Page {page_num} inline processing failed: {str(e)}")
            logging.error(f"Page {page_num} inline processing failed: {e}")
            return None

    def process_pdf_page_upload(self, page_bytes: bytes, page_num: int) -> Optional[str]:
        """
        Process a single PDF page using File API upload (as fallback).
        
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
                print(f"  └─ ⬆️  Uploading page {page_num} to Gemini...")
                
                # Upload page bytes using BytesIO
                page_io = io.BytesIO(page_bytes)
                pdf_file = self.client.files.upload(
                    file=page_io,
                    config=dict(mime_type='application/pdf')
                )
                
                if not pdf_file or not pdf_file.name:
                    raise Exception("Failed to upload page to Gemini")
                
                print(f"  └─ ⏳ Waiting for page {page_num} processing...")
                processing_attempts = 0
                max_processing_time = 60
                
                while processing_attempts < max_processing_time:
                    file = self.client.files.get(name=pdf_file.name)
                    if file.state.name == "ACTIVE":
                        break
                    elif file.state.name == "FAILED":
                        raise Exception(f"File processing failed: {file.state.name}")
                    elif file.state.name not in ["PROCESSING"]:
                        raise Exception(f"Unexpected file state: {file.state.name}")
                    
                    processing_attempts += 1
                    time.sleep(1)
                
                if processing_attempts >= max_processing_time:
                    raise TimeoutError(f"File processing timed out after {max_processing_time} seconds")
                
                print(f"  └─ 🤖 Generating HTR text for page {page_num}...")
                
                # Following Google's best practice: put prompt AFTER the document
                if self.language == "multilingual":
                    language_desc = "text (detect language automatically)"
                elif self.language == "arabic":
                    language_desc = "Arabic"
                else:
                    language_desc = "French"
                
                combined_prompt = (
                    self._get_system_instruction() + "\n\n" +
                    f"This is a legitimate handwritten text transcription (HTR) request for academic research and archival preservation. "
                    f"Transcribe ALL handwritten {language_desc} text with exact wording, spacing rules, accents, and WITHOUT summarizing or omitting any zones."
                )
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[pdf_file, combined_prompt],  # Document first, then prompt
                    config=self.generation_config
                )
                
                # Validate response
                if not response.candidates:
                    raise Exception("No candidates in Gemini response")
                
                candidate = response.candidates[0]
                
                if not candidate.content or not candidate.content.parts:
                    finish_reason = candidate.finish_reason
                    if finish_reason == types.FinishReason.RECITATION:
                        print(f"  └─ ⚠️ Page {page_num}: Copyright detection, trying alternatives...")
                        result = self._try_alternative_prompts(pdf_file, page_num)
                        if result:
                            return result
                        raise Exception("All copyright retry strategies failed")
                    else:
                        raise Exception(f"No valid response. Finish reason: {finish_reason}")
                
                text_content = response.text.replace('\xa0', ' ').strip()
                if not text_content:
                    raise Exception("Empty text response from Gemini")
                
                print(f"  └─ ✅ Page {page_num} HTR complete")
                return text_content
                
            except Exception as e:
                print(f"  └─ ❌ Page {page_num} error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                logging.error(f"Page {page_num} processing error (attempt {attempt + 1}): {e}", exc_info=True)
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"  └─ 🔄 Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"  └─ ❌ Page {page_num} max retries reached.")
        
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
        if self.language == "multilingual":
            language_desc = "text (automatically detecting language)"
        elif self.language == "arabic":
            language_desc = "Arabic"
        else:
            language_desc = "French"
        
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
                print(f"  └─ 🔄 Page {page_num}: Trying {strategy_name}...")
                retry_response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[pdf_content, alternative_prompt],  # Document first, then prompt
                    config=self.generation_config
                )
                
                if (retry_response.candidates and 
                    retry_response.candidates[0].content and 
                    retry_response.candidates[0].content.parts):
                    text_content = retry_response.text.replace('\xa0', ' ').strip()
                    if text_content:
                        print(f"  └─ ✅ Page {page_num} complete (using {strategy_name})")
                        return text_content
                    else:
                        print(f"  └─ ⚠️ Page {page_num}: {strategy_name} returned empty response")
                else:
                    retry_finish_reason = retry_response.candidates[0].finish_reason if retry_response.candidates else 'Unknown'
                    print(f"  └─ ⚠️ Page {page_num}: {strategy_name} failed. Finish reason: {retry_finish_reason}")
                    
            except Exception as e:
                print(f"  └─ ⚠️ Page {page_num}: {strategy_name} error: {str(e)}")
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
            print("\n" + "="*50)
            print(f"� Processing PDF: {pdf_path.name}")
            print("="*50)
            
            # Verify PDF exists
            if not pdf_path.exists():
                print(f"❌ PDF file not found: {pdf_path}")
                logging.error(f"PDF file not found: {pdf_path}")
                return
            
            file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
            print(f"📊 PDF size: {file_size_mb:.2f} MB")
            
            # Get page count
            print("\n🔄 Reading PDF structure...")
            total_pages = self.get_pdf_page_count(pdf_path)
            print(f"✅ PDF has {total_pages} pages")
            
            # Create output file
            output_file = output_dir / f"{pdf_path.stem}.txt"
            print(f"\n📝 Output will be saved to: {output_file}")
            
            # Track processing statistics
            successful_pages = 0
            failed_pages = []
            
            # Process each page
            with open(output_file, 'w', encoding='utf-8') as f:
                for page_idx in range(total_pages):
                    page_num = page_idx + 1  # 1-indexed for display
                    
                    print("\n" + "-"*40)
                    print(f"📃 Processing page {page_num}/{total_pages}")
                    print("-"*40)
                    
                    try:
                        # Extract single page as PDF bytes
                        print(f"  └─ 📄 Extracting page {page_num}...")
                        page_bytes = self.extract_pdf_page(pdf_path, page_idx)
                        page_size_mb = len(page_bytes) / (1024 * 1024)
                        
                        # Process page (try inline first, then upload if needed)
                        text = None
                        if page_size_mb < 20:
                            print(f"  └─ 📄 Page size: {page_size_mb:.2f} MB - trying inline...")
                            text = self.process_pdf_page_inline(page_bytes, page_num)
                        
                        # Fallback to upload if inline failed or page too large
                        if not text:
                            if page_size_mb < 20:
                                print(f"  └─ ⚠️ Inline failed, falling back to upload...")
                            else:
                                print(f"  └─ 📄 Page size: {page_size_mb:.2f} MB - using upload...")
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
                            print(f"✅ Successfully processed page {page_num}")
                        else:
                            failed_pages.append(page_num)
                            print(f"❌ Failed to process page {page_num}")
                            # Add a placeholder for failed pages
                            if page_num == 1:
                                f.write(f"[ERROR: Failed to process page {page_num}]")
                            else:
                                f.write(f"\n\n--- Page {page_num} ---\n\n[ERROR: Failed to process page {page_num}]")
                    
                    except Exception as e:
                        failed_pages.append(page_num)
                        print(f"❌ Error processing page {page_num}: {e}")
                        logging.error(f"Error processing page {page_num} of {pdf_path}: {e}")
                        # Add error placeholder
                        if page_num == 1:
                            f.write(f"[ERROR: Failed to process page {page_num}: {str(e)}]")
                        else:
                            f.write(f"\n\n--- Page {page_num} ---\n\n[ERROR: Failed to process page {page_num}: {str(e)}]")
            
            # Report processing statistics
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
            
            # Validate output file
            output_size = output_file.stat().st_size
            print(f"Output file size: {output_size:,} bytes")
            print("="*50 + "\n")
            
            # Log the results
            logging.info(f"PDF {pdf_path.name}: {successful_pages}/{total_pages} pages successful ({success_rate:.1f}%)")
            if failed_pages:
                logging.warning(f"PDF {pdf_path.name}: Failed pages: {failed_pages}")
            
        except Exception as e:
            print(f"\n❌ Error processing PDF {pdf_path}: {e}")
            logging.error(f"Error processing PDF {pdf_path}: {e}", exc_info=True)

def main():
    """
    Main function to orchestrate the page-by-page PDF HTR process.
    
    Handles user interaction, language and model selection, and batch processing of PDFs.
    Each PDF is processed page-by-page for better control and error recovery.
    """
    print("\n🚀 Starting Page-by-Page PDF HTR Process")
    print("="*50)
    print("📖 This script processes handwritten PDFs page-by-page without image conversion")
    print("="*50)
    
    # Load environment variables from .env file
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables!")
        return
    print("✅ API Key loaded successfully")

    # Language selection
    print("\nPlease choose the manuscript language:")
    print("1: French handwritten manuscripts")
    print("2: Arabic handwritten manuscripts")
    print("3: Multilingual/Other languages (AI will auto-detect)")
    
    # Get valid language choice from user
    language_choice = ""
    while language_choice not in ["1", "2", "3"]:
        language_choice = input("Enter your choice (1, 2, or 3): ")
        if language_choice not in ["1", "2", "3"]:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

    # Map choice to language
    if language_choice == "1":
        selected_language = "french"
    elif language_choice == "2":
        selected_language = "arabic"
    else:
        selected_language = "multilingual"
    
    print(f"✅ Using language: {selected_language.capitalize()}")

    # Present model selection options to user
    print("\nPlease choose the Gemini model to use:")
    print("1: gemini-2.5-flash (Faster, good for most cases)")
    print("2: gemini-2.5-pro (More powerful, more accurate but slower)")
    
    # Get valid model choice from user
    model_choice = ""
    while model_choice not in ["1", "2"]:
        model_choice = input("Enter your choice (1 or 2): ")
        if model_choice not in ["1", "2"]:
            print("❌ Invalid choice. Please enter 1 or 2.")

    # Map choice to model name
    if model_choice == "1":
        selected_model_name = "gemini-2.5-flash"
    else:
        selected_model_name = "gemini-2.5-pro"
    
    print(f"✅ Using model: {selected_model_name}")
    
    # Set up directory paths
    script_dir = Path(__file__).parent
    pdf_dir = script_dir / "PDF"
    output_dir = script_dir / "OCR_Results"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize the HTR processor with selected language and model
    print("\n🔧 Initializing Gemini HTR Processor...")
    htr = GeminiHTR(api_key, selected_model_name, selected_language)
    
    # Find all PDF files to process
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("\n❌ No PDF files found in the PDF directory!")
        return
    
    total_pdfs = len(pdf_files)
    print(f"\n📚 Found {total_pdfs} PDF files to process")
    
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
        print(f"\n📊 Progress: PDF {idx}/{total_pdfs} ({(idx/total_pdfs*100):.1f}%)")
        
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
                
        except Exception as e:
            overall_stats['failed_pdfs'] += 1
            print(f"❌ Failed to process {pdf_path.name}: {e}")
            logging.error(f"Failed to process {pdf_path.name}: {e}")

    # Calculate processing time
    processing_time = time.time() - overall_stats['processing_start']
    
    # Print final summary
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
    
    # Log final summary
    logging.info(f"Processing complete. {overall_stats['processed_pdfs']}/{overall_stats['total_pdfs']} PDFs processed successfully in {processing_time/60:.1f} minutes")

    print("\n✨ Direct PDF HTR Process Complete! ✨\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
        logging.info("Process interrupted by user")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        logging.error(f"An error occurred: {e}", exc_info=True)
