"""
AI-Powered OCR Script using Google Gemini

This script performs high-precision OCR on PDF documents using Google's Gemini AI model.
It's specialized for French newspaper articles and produces research-grade text extraction
with proper formatting, typography, and layout preservation.

Usage:
    python 02_gemini_ocr_processor.py

Requirements:
    - Environment variable: GEMINI_API_KEY
    - PDF files in the PDF/ directory
    - Poppler PDF utilities installed
    - OCR system prompt in ocr_system_prompt.md
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Optional
from google import genai
from google.genai import types
from pdf2image import convert_from_path
from PIL import Image
import tempfile
from tqdm import tqdm
from dotenv import load_dotenv
import warnings
from PIL import Image, ImageFile

# Configure PIL to handle large images safely for legitimate OCR use cases
Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb check for our legitimate use case
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle potentially truncated image files

# Set up logging configuration for tracking OCR operations and errors
# Save log file in a dedicated log directory
script_dir = Path(__file__).parent
log_dir = script_dir / 'log'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / 'ocr_gemini.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file
)

# Suppress PIL DecompressionBombWarning for our legitimate high-DPI newspaper images
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)

class GeminiOCR:
    """
    A high-precision OCR system using Google's Gemini model, specialized for French newspaper articles.
    
    This class implements a complete OCR pipeline that:
    1. Converts PDF documents to images
    2. Processes images through Gemini's vision model
    3. Applies sophisticated text extraction and formatting rules
    4. Handles uncertainty and quality control
    
    The system is designed for academic research and archival purposes, with emphasis on:
    - Maintaining precise reading order and layout relationships
    - Preserving French typography and formatting
    - Handling document structure (columns, zones, captions)
    - Documenting uncertainty in a standardized way
    """

    def __init__(self, api_key: str, model_name: str):
        """
        Initialize the GeminiOCR system with API credentials and model name.
        
        Args:
            api_key (str): Google Gemini API key for authentication
            model_name (str): The Gemini model name to use
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.generation_config = self._setup_generation_config()
        
    def _setup_generation_config(self):
        """
        Configure generation parameters for optimal OCR performance.
        
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
            thinking_budget = 0  # Disable thinking for faster OCR
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
        Get the specialized system instructions for French newspaper OCR.
        
        Loads the system prompt from the ocr_system_prompt.md file for better maintainability.
        
        Returns:
            str: Detailed system instruction for OCR processing
        """
        try:
            prompt_file = Path(__file__).parent / "ocr_system_prompt.md"
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logging.error(f"System prompt file not found: {prompt_file}")
            raise FileNotFoundError(f"OCR system prompt file not found at {prompt_file}")
        except Exception as e:
            logging.error(f"Error reading system prompt file: {e}")
            raise
    
    def pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        """
        Convert a PDF document to a list of high-quality images for OCR processing.
        
        This method handles:
        - PDF existence and readability verification
        - Poppler dependency checking
        - Optimal image conversion settings
        - Error handling and logging
        
        Args:
            pdf_path (Path): Path to the PDF file to be processed
            
        Returns:
            List[Image.Image]: List of PIL Image objects, one per page
                             Returns empty list if conversion fails
        
        Technical Details:
        - Uses pdf2image with poppler backend
        - 300 DPI for archival-quality scanning and optimal text recognition
        - Grayscale conversion for better text recognition and noise reduction
        - Multi-threaded processing for performance
        - 180-second timeout for large documents
        """
        try:
            # Verify PDF file existence and accessibility
            if not pdf_path.exists():
                logging.error(f"PDF file not found: {pdf_path}")
                return []
                
            logging.info(f"PDF file exists and size is: {pdf_path.stat().st_size} bytes")
            
            # Verify Poppler installation (required for PDF conversion)
            poppler_path = r"C:\Program Files\poppler\Library\bin"
            if not os.path.exists(poppler_path):
                logging.error(f"Poppler dependency not found at: {poppler_path}")
                return []
                
            logging.info("Initiating PDF conversion...")
            
            # Convert PDF to images with optimized parameters for OCR quality
            images = convert_from_path(
                pdf_path,
                poppler_path=poppler_path,
                dpi=300,              # High DPI for optimal text recognition, especially for poor scans
                fmt='png',            # Lossless format for quality
                thread_count=2,        # Parallel processing
                use_pdftocairo=True,   # Preferred converter
                timeout=180,           # Extended timeout for large files
                first_page=1,
                last_page=None,        # Process all pages
                strict=False,          # More permissive processing
                grayscale=True         # Better for text recognition, reduces noise from colored backgrounds
            )
            
            logging.info(f"Successfully converted {len(images)} pages to images")
            return images
            
        except Exception as e:
            # Handle specific error cases with appropriate logging
            if "timeout" in str(e).lower():
                logging.error(
                    "PDF conversion timeout - possible PDF/A format or complexity issue",
                    exc_info=True
                )
            else:
                logging.error(f"PDF conversion error: {str(e)}", exc_info=True)
            return []

    def _get_image_size_mb(self, image: Image.Image) -> float:
        """Calculate the approximate size of an image in MB when saved as PNG."""
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        size_bytes = buffer.tell()
        return size_bytes / (1024 * 1024)

    def _optimize_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Optimize image for OCR while maintaining quality for text recognition."""
        width, height = image.size
        max_dimension = 2048  # API recommended max
        
        # Check if image is too large for API
        if width > max_dimension or height > max_dimension:
            logging.info(f"Resizing large image from {width}x{height}")
            # Use high-quality resampling to preserve text clarity
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            logging.info(f"Resized to {image.size}")
        
        # Check file size and reduce quality only if absolutely necessary
        size_mb = self._get_image_size_mb(image)
        if size_mb > 18:  # Close to 20MB API limit
            logging.warning(f"Image size {size_mb:.1f}MB is very large, applying compression")
            # Apply minimal compression while preserving text quality
            import io
            buffer = io.BytesIO()
            # Use high quality JPEG compression as last resort
            temp_image = image.convert('RGB') if image.mode != 'RGB' else image
            temp_image.save(buffer, format='JPEG', quality=95, optimize=True)
            image = Image.open(buffer)
            new_size_mb = self._get_image_size_mb(image)
            logging.info(f"Compressed image from {size_mb:.1f}MB to {new_size_mb:.1f}MB")
        
        return image

    def process_image_inline(self, image: Image.Image) -> Optional[str]:
        """Process smaller images using inline data instead of file upload."""
        try:
            print("  └─ 🔄 Processing image inline...")
            
            # Convert image to bytes
            import io
            from google.genai import types
            
            buffer = io.BytesIO()
            image.save(buffer, format='PNG', optimize=True)
            image_bytes = buffer.getvalue()
            
            # Create image part
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/png'
            )
            
            # Generate response
            print("  └─ 🤖 Generating OCR text (inline)...")
            combined_prompt = (
                self._get_system_instruction() + "\n\n" +
                "Please perform complete OCR transcription of this document image. "
                "Extract all visible text maintaining original formatting and structure."
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[combined_prompt, image_part],
                config=self.generation_config
            )
            
            # Validate response and handle RECITATION errors
            if not response.candidates:
                raise Exception("No candidates in Gemini response")
            
            candidate = response.candidates[0]
            
            # Check for RECITATION errors in inline processing
            if not candidate.content or not candidate.content.parts:
                finish_reason = candidate.finish_reason
                if finish_reason == types.FinishReason.RECITATION:
                    print("  └─ ⚠️ Inline processing hit copyright detection, trying alternative...")
                    # Try a simpler prompt for inline processing
                    simple_prompt = (
                        "Please extract the text content from this document image for educational purposes. "
                        "Focus on accurate transcription of all visible text."
                    )
                    
                    retry_response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[simple_prompt, image_part],
                        config=self.generation_config
                    )
                    
                    if (retry_response.candidates and 
                        retry_response.candidates[0].content and 
                        retry_response.candidates[0].content.parts):
                        text_content = retry_response.text.replace('\xa0', ' ').strip()
                        if text_content:
                            print("  └─ ✅ OCR complete (inline with alternative prompt)")
                            return text_content
                    
                    # If still failing, let it fall back to file upload
                    raise Exception("Inline copyright retry failed")
                else:
                    raise Exception(f"No valid response from Gemini. Finish reason: {finish_reason}")
            
            text_content = response.text.replace('\xa0', ' ').strip()
            if not text_content:
                raise Exception("Empty text response from Gemini")
            
            print("  └─ ✅ OCR complete (inline)")
            return text_content
            
        except Exception as e:
            print(f"  └─ ❌ Inline processing failed: {str(e)}")
            logging.error(f"Inline image processing failed: {e}")
            return None

    def process_image(self, image: Image.Image) -> Optional[str]:
        """Process a single image with Gemini, choosing optimal method based on size."""
        # Optimize image for OCR while preserving quality
        image = self._optimize_image_for_ocr(image)

        # Check image size to decide processing method
        image_size_mb = self._get_image_size_mb(image)
        
        # Use inline processing for smaller images (< 15MB) for better efficiency
        if image_size_mb < 15:
            print(f"  └─ 📏 Image size: {image_size_mb:.1f}MB - using inline processing")
            result = self.process_image_inline(image)
            if result:
                return result
            else:
                print("  └─ ⚠️ Inline processing failed, falling back to file upload")
        else:
            print(f"  └─ 📏 Image size: {image_size_mb:.1f}MB - using file upload")
        
        # Use file upload method (original method with improvements)
        return self.process_image_upload(image)

    def process_image_upload(self, image: Image.Image) -> Optional[str]:
        """Process a single image with Gemini using file upload with improved error handling."""
        max_retries = 5  # Increased retries
        base_delay = 2   # Base delay in seconds

        for attempt in range(max_retries):
            tmp_file_path = None
            try:
                # Save image to temporary file with better error handling
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_file_path = tmp.name
                    print("  └─ 💾 Saving temporary image...")
                    image.save(tmp_file_path, optimize=True, quality=95)
                    
                    # Validate file was created and has content
                    if not os.path.exists(tmp_file_path) or os.path.getsize(tmp_file_path) == 0:
                        raise Exception("Failed to create temporary image file")
                    
                    # Upload to Gemini with validation
                    print("  └─ ⬆️  Uploading to Gemini...")
                    image_file = self.client.files.upload(file=tmp_file_path)
                    
                    if not image_file or not image_file.name:
                        raise Exception("Failed to upload image to Gemini")
                    
                    # Wait for processing with better timeout handling
                    print("  └─ ⏳ Processing file...")
                    processing_attempts = 0
                    max_processing_time = 60  # Maximum wait time
                    
                    while processing_attempts < max_processing_time:
                        file = self.client.files.get(name=image_file.name)
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
                    
                    # Generate response with improved prompt
                    print("  └─ 🤖 Generating OCR text...")
                    combined_prompt = (
                        self._get_system_instruction() + "\n\n" +
                        "Please perform complete OCR transcription of this document image. "
                        "Extract all visible text maintaining original formatting and structure."
                    )
                    
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[combined_prompt, image_file],
                        config=self.generation_config
                    )
                    
                    # Improved response validation
                    if not response.candidates:
                        raise Exception("No candidates in Gemini response")
                    
                    candidate = response.candidates[0]
                    
                    if not candidate.content or not candidate.content.parts:
                        finish_reason = candidate.finish_reason
                        safety_info = ""
                        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                            safety_info = f" Safety ratings: {candidate.safety_ratings}"
                        
                        # Handle RECITATION errors (copyright detection) with multiple strategies
                        if finish_reason == types.FinishReason.RECITATION:
                            print("  └─ ⚠️ Copyright detection triggered, trying alternative approaches...")
                            
                            # Strategy 1: Academic fair use prompt
                            alternative_prompts = [
                                (
                                    "Academic Fair Use Request",
                                    "This is a legitimate academic research request for historical document preservation and scholarly analysis. "
                                    "Under fair use principles, please perform OCR text extraction from this historical newspaper document. "
                                    "The purpose is archival preservation and academic research, not commercial reproduction. "
                                    "Please extract all visible text while maintaining original formatting and structure."
                                ),
                                (
                                    "Educational OCR Request", 
                                    "Please assist with educational OCR processing of this historical document image. "
                                    "Extract the text content for research and educational purposes. "
                                    "Focus on accuracy and completeness of the text transcription."
                                ),
                                (
                                    "Technical OCR Analysis",
                                    "Perform technical optical character recognition analysis on this document image. "
                                    "Output the detected text content with preserved formatting. "
                                    "This is for document digitization and preservation purposes."
                                )
                            ]
                            
                            for strategy_name, alternative_prompt in alternative_prompts:
                                try:
                                    print(f"  └─ 🔄 Trying {strategy_name}...")
                                    retry_response = self.client.models.generate_content(
                                        model=self.model_name,
                                        contents=[alternative_prompt, image_file],
                                        config=self.generation_config
                                    )
                                    
                                    if (retry_response.candidates and 
                                        retry_response.candidates[0].content and 
                                        retry_response.candidates[0].content.parts):
                                        text_content = retry_response.text.replace('\xa0', ' ').strip()
                                        if text_content:
                                            print(f"  └─ ✅ OCR complete (using {strategy_name})")
                                            return text_content
                                        else:
                                            print(f"  └─ ⚠️ {strategy_name} returned empty response")
                                    else:
                                        retry_finish_reason = retry_response.candidates[0].finish_reason if retry_response.candidates else 'Unknown'
                                        print(f"  └─ ⚠️ {strategy_name} failed. Finish reason: {retry_finish_reason}")
                                        
                                except Exception as e:
                                    print(f"  └─ ⚠️ {strategy_name} error: {str(e)}")
                                    continue
                            
                            # If all strategies failed, raise the original error
                            raise Exception(f"All copyright retry strategies failed. Original finish reason: {finish_reason}")
                            
                        else:
                            raise Exception(f"Response lacks content. Finish reason: {finish_reason}.{safety_info}")
                    
                    # Extract and validate text
                    text_content = response.text.replace('\xa0', ' ').strip()
                    if not text_content:
                        raise Exception("Empty text response from Gemini")
                    
                    print("  └─ ✅ OCR complete")
                    return text_content
                
            except Exception as e:
                print(f"  └─ ❌ Error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                logging.error(f"Error processing image (attempt {attempt + 1}): {e}", exc_info=True)
                
                # Exponential backoff for retries
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"  └─ 🔄 Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print("  └─ ❌ Max retries reached. Skipping this image.")
                    
            finally:
                # Ensure temporary file cleanup
                if tmp_file_path and os.path.exists(tmp_file_path):
                    try:
                        os.unlink(tmp_file_path)
                    except Exception as cleanup_error:
                        logging.warning(f"Failed to cleanup temporary file {tmp_file_path}: {cleanup_error}")
        
        return None

    def process_pdf(self, pdf_path: Path, output_dir: Path) -> None:
        """Process a PDF file and save results to a text file with improved validation and reporting."""
        try:
            print("\n" + "="*50)
            print(f"📄 Processing PDF: {pdf_path.name}")
            print("="*50)
            
            # Add poppler to PATH temporarily
            poppler_path = r"C:\Program Files\poppler\Library\bin"
            os.environ["PATH"] = poppler_path + os.pathsep + os.environ["PATH"]
            
            # Convert PDF to images
            print("\n🔄 Converting PDF to images...")
            images = self.pdf_to_images(pdf_path)
            if not images:
                print("❌ No images were extracted from the PDF!")
                logging.error(f"No images extracted from {pdf_path}")
                return
            print(f"✅ Successfully extracted {len(images)} pages from PDF")

            # Create output text file
            output_file = output_dir / f"{pdf_path.stem}.txt"
            print(f"\n📝 Output will be saved to: {output_file}")
            
            # Track processing statistics
            total_pages = len(images)
            successful_pages = 0
            failed_pages = []
            
            # Process each page with better error tracking
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, image in enumerate(images, 1):
                    print("\n" + "-"*40)
                    print(f"📃 Processing page {i}/{total_pages}")
                    print("-"*40)
                    
                    text = self.process_image(image)
                    if text and text.strip():  # Validate we got actual content
                        # Special handling for first page - no header, no extra newlines
                        if i == 1:
                            f.write(text)
                        else:
                            # For subsequent pages, add page marker and newlines
                            f.write(f"\n\n--- Page {i} ---\n\n")
                            f.write(text)
                        
                        successful_pages += 1
                        print(f"✅ Successfully processed page {i}")
                    else:
                        failed_pages.append(i)
                        print(f"❌ Failed to process page {i}")
                        # Add a placeholder for failed pages to maintain structure
                        if i == 1:
                            f.write(f"[ERROR: Failed to process page {i}]")
                        else:
                            f.write(f"\n\n--- Page {i} ---\n\n[ERROR: Failed to process page {i}]")
            
            # Report processing statistics
            print("\n" + "="*50)
            print(f"📊 Processing Summary for {pdf_path.name}")
            print("="*50)
            print(f"Total pages: {total_pages}")
            print(f"Successfully processed: {successful_pages}")
            print(f"Failed pages: {len(failed_pages)}")
            if failed_pages:
                print(f"Failed page numbers: {failed_pages}")
            
            success_rate = (successful_pages / total_pages) * 100
            print(f"Success rate: {success_rate:.1f}%")
            
            # Log the results
            logging.info(f"PDF {pdf_path.name}: {successful_pages}/{total_pages} pages successful ({success_rate:.1f}%)")
            if failed_pages:
                logging.warning(f"PDF {pdf_path.name}: Failed pages: {failed_pages}")
            
            # Validate output file has content
            if output_file.exists() and output_file.stat().st_size > 0:
                print(f"✅ Output file created successfully: {output_file.stat().st_size} bytes")
            else:
                print("⚠️ Warning: Output file is empty or missing!")
                logging.warning(f"Empty or missing output file for {pdf_path.name}")
            
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error processing PDF {pdf_path}: {e}")
            logging.error(f"Error processing PDF {pdf_path}: {e}", exc_info=True)

def main():
    """
    Main function to orchestrate the OCR process.
    
    Handles user interaction, model selection, and batch processing of PDFs.
    """
    print("\\n🚀 Starting OCR Process")
    print("="*50)
    
    # Load environment variables from .env file
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables!")
        return
    print("✅ API Key loaded successfully")

    # Present model selection options to user
    print("\\nPlease choose the Gemini model to use:")
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
    pdf_dir = script_dir / "PDF"  # Input directory for PDFs
    output_dir = script_dir / "OCR_Results"  # Output directory for text files
    output_dir.mkdir(exist_ok=True)  # Create output directory if it doesn't exist
    
    # Initialize the OCR processor with selected model
    print("\\n🔧 Initializing Gemini OCR...")
    ocr = GeminiOCR(api_key, selected_model_name)
    
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
        'empty_outputs': 0
    }
    
    # Process each PDF file sequentially
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n📊 Progress: PDF {idx}/{total_pdfs} ({(idx/total_pdfs*100):.1f}%)")
        
        try:
            ocr.process_pdf(pdf_path, output_dir)
            
            # Check if output file has content before marking as successful
            output_file = output_dir / f"{pdf_path.stem}.txt"
            if output_file.exists() and output_file.stat().st_size > 100:  # At least 100 bytes
                overall_stats['processed_pdfs'] += 1
                logging.info(f"Successfully processed {pdf_path.name}")
            else:
                overall_stats['empty_outputs'] += 1
                overall_stats['failed_pdfs'] += 1  # Empty output counts as failure
                logging.warning(f"Output file for {pdf_path.name} is empty or very small - counting as failed")
                
        except Exception as e:
            overall_stats['failed_pdfs'] += 1
            print(f"❌ Failed to process {pdf_path.name}: {e}")
            logging.error(f"Failed to process {pdf_path.name}: {e}")

    # Print final summary
    print("\n" + "="*60)
    print("📈 FINAL PROCESSING SUMMARY")
    print("="*60)
    print(f"Total PDFs found: {overall_stats['total_pdfs']}")
    print(f"Successfully processed: {overall_stats['processed_pdfs']}")
    print(f"Failed to process: {overall_stats['failed_pdfs']}")
    print(f"  └─ Empty/small outputs: {overall_stats['empty_outputs']}")
    
    if overall_stats['total_pdfs'] > 0:
        success_rate = (overall_stats['processed_pdfs'] / overall_stats['total_pdfs']) * 100
        print(f"Overall success rate: {success_rate:.1f}%")
    
    print("="*60)
    
    # Log final summary
    logging.info(f"Processing complete. {overall_stats['processed_pdfs']}/{overall_stats['total_pdfs']} PDFs processed successfully")

    print("\n✨ OCR Process Complete! ✨\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        logging.error(f"An error occurred: {e}")
