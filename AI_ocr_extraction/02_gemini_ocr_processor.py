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
        - Balanced temperature for accuracy vs. creativity
        - High top_p and top_k for reliable text recognition
        - Sufficient output tokens for long documents
        - Thinking budget disabled for direct responses
        
        Returns:
            types.GenerateContentConfig: Configured generation config
        """
        return types.GenerateContentConfig(
            temperature=0.2,      # Lower temperature for more consistent output
            top_p=0.95,          # High top_p for reliable text recognition
            top_k=40,            # Balanced top_k for good candidate selection
            max_output_tokens=65535,  # Support for long documents
            response_mime_type="text/plain",  # Ensure text output
            thinking_config=types.ThinkingConfig(thinking_budget=-1),  # Disable thinking mode for both models
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
        - Grayscale conversion for better text recognition
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
            
            # Convert PDF to images with optimized parameters
            images = convert_from_path(
                pdf_path,
                poppler_path=poppler_path,
                dpi=300,              # Archival-quality resolution for optimal OCR
                fmt='png',            # Lossless format for quality
                thread_count=2,        # Parallel processing
                use_pdftocairo=True,   # Preferred converter
                timeout=180,           # Extended timeout for large files
                first_page=1,
                last_page=None,        # Process all pages
                strict=False,          # More permissive processing
                grayscale=True         # Better for text recognition
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

    def process_image(self, image: Image.Image) -> Optional[str]:
        """Process a single image with Gemini."""
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                # Save image to temporary file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    print("  └─ 💾 Saving temporary image...")
                    image.save(tmp.name)
                    
                    # Upload to Gemini
                    print("  └─ ⬆️  Uploading to Gemini...")
                    image_file = self.client.files.upload(file=tmp.name)
                    
                    # Wait for processing
                    print("  └─ ⏳ Processing file...")
                    attempts = 0
                    while attempts < 30:
                        file = self.client.files.get(name=image_file.name)
                        if file.state.name == "ACTIVE":
                            break
                        elif file.state.name != "PROCESSING":
                            raise Exception(f"File processing failed: {file.state.name}")
                        
                        attempts += 1
                        time.sleep(2)
                    
                    if attempts >= 30:
                        raise TimeoutError("File processing timed out after 60 seconds")
                    
                    # Generate response
                    print("  └─ 🤖 Generating OCR text...")
                    # Combine system instruction with user prompt
                    combined_prompt = (
                        self._get_system_instruction() + "\n\n" +
                        "This is a legitimate OCR request for academic research. "
                        "Please transcribe this French newspaper article completely..."
                    )
                    initial_response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[combined_prompt, image_file],
                        config=self.generation_config
                    )
                    
                    cleaned_text = None # Default to no text

                    if not initial_response.candidates:
                        print("  └─ ❌ Error: No candidates in Gemini response.")
                        logging.error(f"No candidates in Gemini response. Full response: {initial_response}")
                    elif not initial_response.candidates[0].content or not initial_response.candidates[0].content.parts:
                        candidate = initial_response.candidates[0]
                        finish_reason = candidate.finish_reason
                        safety_ratings_log = f"Safety Ratings: {candidate.safety_ratings}" if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings else "Safety Ratings: N/A"
                        
                        print(f"  └─ ⚠️ Initial response lacks content parts. Finish Reason: {finish_reason}. {safety_ratings_log}")
                        logging.warning(f"Initial response lacks content parts. Finish Reason: {finish_reason}. Candidate: {candidate}. {safety_ratings_log}")

                        if finish_reason == 4:  # Copyright detection (user's interpretation)
                            print("  └─ ⚠️ Copyright detection (finish_reason 4) triggered. Attempting with modified prompt...")
                            # Combine system instruction with copyright retry prompt
                            copyright_retry_prompt = (
                                self._get_system_instruction() + "\n\n" +
                                "This is a fair use OCR request for academic research and archival purposes. "
                                "Please perform text extraction of this historical newspaper content..."
                            )
                            response_after_copyright_retry = self.client.models.generate_content(
                                model=self.model_name,
                                contents=[copyright_retry_prompt, image_file],
                                config=self.generation_config
                            )
                            if response_after_copyright_retry.candidates and \
                               response_after_copyright_retry.candidates[0].content and \
                               response_after_copyright_retry.candidates[0].content.parts:
                                cleaned_text = response_after_copyright_retry.text.replace('\xa0', ' ').rstrip()
                                print("  └─ ✅ OCR complete (after copyright retry)")
                            else:
                                retry_candidate = response_after_copyright_retry.candidates[0] if response_after_copyright_retry.candidates else None
                                retry_finish_reason = retry_candidate.finish_reason if retry_candidate else "UNKNOWN"
                                retry_safety_ratings = f"Safety Ratings: {retry_candidate.safety_ratings}" if retry_candidate and hasattr(retry_candidate, 'safety_ratings') and retry_candidate.safety_ratings else "Safety Ratings: N/A"
                                print(f"  └─ ❌ Failed after copyright retry. Finish Reason: {retry_finish_reason}. {retry_safety_ratings}")
                                logging.error(f"Failed after copyright retry. Finish Reason: {retry_finish_reason}. Candidate: {retry_candidate}. {retry_safety_ratings}")
                        elif finish_reason == 2: # SAFETY
                            print(f"  └─ ⚠️ Safety block (finish_reason 2) on initial response. No text extracted. {safety_ratings_log}")
                            logging.warning(f"Safety block (finish_reason 2). Candidate: {candidate}. {safety_ratings_log}")
                        # else: Other finish reasons (MAX_TOKENS, RECITATION, etc.) or no specific handling, cleaned_text remains None
                    else:
                        # Success path for the initial response
                        cleaned_text = initial_response.text.replace('\xa0', ' ').rstrip()
                        print("  └─ ✅ OCR complete (initial)")
                    
                # Clean up temporary file
                os.unlink(tmp.name) # Original placement from user's code
                
                # Return the cleaned_text (which might be None if issues occurred)
                return cleaned_text
                
            except Exception as e:
                print(f"  └─ ❌ Error: {str(e)}")
                logging.error(f"Error processing image with Gemini: {e}", exc_info=True)
                
                # Check for specific retryable errors
                if "503" in str(e) or "read operation timed out" in str(e) or attempt < max_retries - 1:
                    print(f"  └─ 🔄 Retrying... ({attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    print("  └─ ❌ Max retries reached. Skipping this image.")
                    return None

    def process_pdf(self, pdf_path: Path, output_dir: Path) -> None:
        """Process a PDF file and save results to a text file."""
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
            
            # Process each page
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, image in enumerate(images, 1):
                    print("\n" + "-"*40)
                    print(f"📃 Processing page {i}/{len(images)}")
                    print("-"*40)
                    
                    text = self.process_image(image)
                    if text:
                        # Special handling for first page - no header, no extra newlines
                        if i == 1:
                            f.write(text)
                        else:
                            # For subsequent pages, add page marker and newlines
                            f.write(f"\n\n--- Page {i} ---\n\n")
                            f.write(text)
                        
                        print(f"✅ Successfully processed page {i}")
                    else:
                        print(f"❌ Failed to process page {i}")
                        
            print("\n" + "="*50)
            print(f"✅ Completed processing {pdf_path.name}")
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
    print("2: gemini-2.5-pro (More powerful, potentially more accurate but slower)")
    
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
    
    # Process each PDF file sequentially
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n📊 Progress: PDF {idx}/{total_pdfs} ({(idx/total_pdfs*100):.1f}%)")
        ocr.process_pdf(pdf_path, output_dir)

    print("\n✨ OCR Process Complete! ✨\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        logging.error(f"An error occurred: {e}")
