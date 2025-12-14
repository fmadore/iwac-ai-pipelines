"""
AI-Powered Direct PDF Processing using Google Gemini (Page-by-Page)

This script performs high-precision OCR on PDF documents by sending them directly to Gemini
page-by-page, without converting to images first. This leverages Gemini's native PDF understanding
while maintaining precise control over page-by-page extraction.

Usage:
    python 02_gemini_ocr_processor.py

Requirements:
    - Environment variable: GEMINI_API_KEY
    - PDF files in the PDF/ directory
    - OCR system prompt in ocr_system_prompt.md
    - PyPDF2 for page extraction
    - Shared llm_provider module for model selection

Model Selection:
    - Uses shared LLM provider with Gemini Flash and Gemini Pro options
    - Flash: Faster, cost-effective, thinking disabled for speed
    - Pro: Higher quality, extended thinking enabled

Advantages over image-based approach:
    - No Poppler dependency needed
    - Better document structure understanding
    - Processes images, diagrams, and tables natively
    - Simpler pipeline with fewer conversions
    - Page-by-page processing for better control
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from rich.progress import Progress as ProgressType
from dotenv import load_dotenv
from PyPDF2 import PdfReader, PdfWriter
import io

# Rich console output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich import box

console = Console()

# Import shared LLM provider for model selection (but use genai.Client directly for multimodal)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.llm_provider import get_model_option, summary_from_option, LLMConfig

# Import Gemini types for PDF processing
try:
    from google import genai
    from google.genai import types
except ImportError:
    raise RuntimeError("google-genai package is required for PDF processing")

# Set up logging configuration for tracking OCR operations and errors
script_dir = Path(__file__).parent
log_dir = script_dir / 'log'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / 'ocr_gemini_pdf.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file
)

class GeminiPDFProcessor:
    """
    A high-precision OCR system using Google's Gemini model with native PDF processing.
    
    This class implements a page-by-page OCR pipeline that:
    1. Extracts individual pages from PDF documents
    2. Sends each page directly to Gemini for processing
    3. Leverages Gemini's native document understanding
    4. Applies sophisticated text extraction and formatting rules
    5. Handles uncertainty and quality control per page
    
    The system is designed for academic research and archival purposes, with emphasis on:
    - Maintaining precise reading order and layout relationships
    - Preserving French typography and formatting
    - Handling document structure (columns, zones, captions)
    - Processing pages individually for better control and error recovery
    """

    def __init__(self, api_key: str, model_option, llm_config: LLMConfig):
        """
        Initialize the GeminiPDFProcessor with API credentials and model configuration.
        
        Args:
            api_key (str): Google Gemini API key for authentication
            model_option: ModelOption from llm_provider
            llm_config (LLMConfig): LLM configuration for generation parameters
        
        Note: This script uses genai.Client() directly instead of the shared llm_provider
        because it requires multimodal capabilities (PDF uploads, inline processing) that
        the text-only wrapper doesn't expose.
        """
        # Use genai.Client directly for multimodal PDF processing capabilities
        self.client = genai.Client(api_key=api_key)
        self.model_option = model_option
        self.model_name = model_option.model
        self.llm_config = llm_config
        self.generation_config = self._setup_generation_config()
        
    def _setup_generation_config(self):
        """
        Configure generation parameters for optimal OCR performance.
        
        The configuration focuses on:
        - Lower temperature for more consistent output
        - High top_p and top_k for reliable text recognition
        - Sufficient output tokens for long documents
        - Model-appropriate thinking configuration:
          - Gemini 3 Pro: uses thinking_level ("low" or "high")
          - Gemini 2.5 Flash: uses thinking_budget (0 to disable, or positive value)
        
        Returns:
            types.GenerateContentConfig: Configured generation config
        """
        # Check if this is a Gemini 3 model (uses thinking_level instead of thinking_budget)
        is_gemini_3 = "gemini-3" in self.model_name.lower()
        
        # Load system instruction from file (modern API pattern)
        system_instruction = self._get_system_instruction()
        
        # Use latest SDK with system_instruction in config (per Context7 docs)
        config_kwargs = {
            "system_instruction": system_instruction,  # Use proper system_instruction parameter
            "temperature": self.llm_config.temperature or 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 65535,
            "response_mime_type": "text/plain",
            "safety_settings": [
                types.SafetySetting(
                    category='HARM_CATEGORY_HARASSMENT',
                    threshold='BLOCK_NONE'
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_HATE_SPEECH',
                    threshold='BLOCK_NONE'
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                    threshold='BLOCK_NONE'
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_DANGEROUS_CONTENT',
                    threshold='BLOCK_NONE'
                )
            ]
        }
        
        # Configure thinking based on model type
        if is_gemini_3:
            # Gemini 3 Pro uses thinking_level ("low" or "high"), cannot be disabled
            thinking_level = self.llm_config.thinking_level or self.model_option.default_thinking_level or "low"
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
            console.print(f"  [cyan]🧠 Thinking:[/] level='{thinking_level}' for {self.model_name}")
        else:
            # Gemini 2.5 series uses thinking_budget
            thinking_budget = self.llm_config.thinking_budget
            if thinking_budget is None:
                thinking_budget = self.model_option.default_thinking_budget
            
            if thinking_budget is not None:
                if thinking_budget == 0:
                    console.print(f"  [cyan]🧠 Thinking:[/] disabled for {self.model_name}")
                else:
                    console.print(f"  [cyan]🧠 Thinking:[/] budget={thinking_budget} for {self.model_name}")
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
        
        return types.GenerateContentConfig(**config_kwargs)
    
    def _extract_text_from_response(self, response) -> str:
        """
        Safely extract text from response, ignoring thought traces.
        
        Args:
            response: The generation response object
            
        Returns:
            str: The extracted text content
        """
        if not response.candidates:
            return ""
            
        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            return ""
            
        text_parts = []
        for part in candidate.content.parts:
            # Only include parts that have text (skips thought_signature etc.)
            if hasattr(part, 'text') and part.text:
                text_parts.append(part.text)
                
        return "".join(text_parts).replace('\xa0', ' ').strip()
    
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
            # Create PDF part from page bytes
            pdf_part = types.Part.from_bytes(
                data=page_bytes,
                mime_type='application/pdf'
            )
            
            # System instruction is in the config, just provide user request with document
            user_prompt = (
                "Please perform complete OCR transcription of this single page. "
                "Extract all visible text maintaining original formatting and structure."
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[pdf_part, user_prompt],  # Document first, then user request
                config=self.generation_config  # Contains system_instruction
            )
            
            # Validate response
            if not response.candidates:
                raise Exception("No candidates in Gemini response")
            
            candidate = response.candidates[0]
            
            if not candidate.content or not candidate.content.parts:
                finish_reason = candidate.finish_reason
                raise Exception(f"No valid response. Finish reason: {finish_reason}")
            
            text_content = self._extract_text_from_response(response)
            if not text_content:
                raise Exception("Empty text response from Gemini")
            
            return text_content
            
        except Exception as e:
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
                
                # Save page bytes to a temporary file for upload (SDK requires file path)
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                    tmp_file.write(page_bytes)
                    tmp_path = tmp_file.name
                
                try:
                    # Upload using file path with UploadFileConfig (per Context7 docs)
                    pdf_file = self.client.files.upload(
                        file=tmp_path,
                        config=types.UploadFileConfig(
                            mime_type='application/pdf',
                            display_name=f'page_{page_num}.pdf'
                        )
                    )
                finally:
                    # Clean up temp file
                    os.unlink(tmp_path)
                
                if not pdf_file or not pdf_file.name:
                    raise Exception("Failed to upload page to Gemini")
                
                processing_attempts = 0
                max_processing_time = 60
                
                # Poll file state using string comparison (per Context7 docs: FileState enum)
                while processing_attempts < max_processing_time:
                    file = self.client.files.get(name=pdf_file.name)
                    # Check state - can be string or enum, handle both
                    state = file.state if isinstance(file.state, str) else getattr(file.state, 'name', str(file.state))
                    if state == 'ACTIVE':
                        break
                    elif state == 'FAILED':
                        raise Exception(f"File processing failed: {state}")
                    elif state not in ['PROCESSING', 'STATE_UNSPECIFIED']:
                        raise Exception(f"Unexpected file state: {state}")
                    
                    processing_attempts += 1
                    time.sleep(1)
                
                if processing_attempts >= max_processing_time:
                    raise TimeoutError(f"File processing timed out after {max_processing_time} seconds")
                
                # System instruction is in the config, just provide user request with document
                user_prompt = (
                    "Please perform complete OCR transcription of this single page. "
                    "Extract all visible text maintaining original formatting and structure."
                )
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[pdf_file, user_prompt],  # Document first, then user request
                    config=self.generation_config  # Contains system_instruction
                )
                
                # Validate response
                if not response.candidates:
                    raise Exception("No candidates in Gemini response")
                
                candidate = response.candidates[0]
                
                if not candidate.content or not candidate.content.parts:
                    finish_reason = candidate.finish_reason
                    raise Exception(f"No valid response. Finish reason: {finish_reason}")
                
                text_content = self._extract_text_from_response(response)
                if not text_content:
                    raise Exception("Empty text response from Gemini")
                
                return text_content
                
            except Exception as e:
                logging.error(f"Page {page_num} processing error (attempt {attempt + 1}): {e}", exc_info=True)
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
        
        return None

    def process_pdf(self, pdf_path: Path, output_dir: Path, progress: Optional[Progress] = None) -> None:
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
            progress (Optional[Progress]): Rich progress bar for page tracking
        """
        try:
            console.print()
            console.rule(f"[bold]📄 {pdf_path.name}[/]")
            
            # Verify PDF exists
            if not pdf_path.exists():
                console.print(f"[red]✗[/] PDF file not found: {pdf_path}")
                logging.error(f"PDF file not found: {pdf_path}")
                return
            
            file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
            console.print(f"  [dim]Size:[/] {file_size_mb:.2f} MB")
            
            # Get page count
            total_pages = self.get_pdf_page_count(pdf_path)
            console.print(f"  [dim]Pages:[/] {total_pages}")
            
            # Create output file
            output_file = output_dir / f"{pdf_path.stem}.txt"
            console.print(f"  [dim]Output:[/] {output_file.name}")
            
            # Track processing statistics
            successful_pages = 0
            failed_pages = []
            
            # Process each page with nested progress
            with open(output_file, 'w', encoding='utf-8') as f:
                # Create page progress bar if parent progress exists
                page_task = None
                if progress:
                    page_task = progress.add_task(f"[dim]  Pages", total=total_pages, visible=True)
                
                for page_idx in range(total_pages):
                    page_num = page_idx + 1  # 1-indexed for display
                    
                    try:
                        # Extract single page as PDF bytes
                        page_bytes = self.extract_pdf_page(pdf_path, page_idx)
                        page_size_mb = len(page_bytes) / (1024 * 1024)
                        
                        # Process page (try inline first, then upload if needed)
                        text = None
                        if page_size_mb < 20:
                            text = self.process_pdf_page_inline(page_bytes, page_num)
                        
                        # Fallback to upload if inline failed or page too large
                        if not text:
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
                        else:
                            failed_pages.append(page_num)
                            # Add a placeholder for failed pages
                            if page_num == 1:
                                f.write(f"[ERROR: Failed to process page {page_num}]")
                            else:
                                f.write(f"\n\n--- Page {page_num} ---\n\n[ERROR: Failed to process page {page_num}]")
                    
                    except Exception as e:
                        failed_pages.append(page_num)
                        logging.error(f"Error processing page {page_num} of {pdf_path}: {e}")
                        # Add error placeholder
                        if page_num == 1:
                            f.write(f"[ERROR: Failed to process page {page_num}: {str(e)}]")
                        else:
                            f.write(f"\n\n--- Page {page_num} ---\n\n[ERROR: Failed to process page {page_num}: {str(e)}]")
                    
                    # Update page progress
                    if progress and page_task is not None:
                        progress.update(page_task, advance=1)
                
                # Remove page task when done
                if progress and page_task is not None:
                    progress.remove_task(page_task)
            
            # Report processing statistics
            success_rate = (successful_pages / total_pages) * 100 if total_pages > 0 else 0
            output_size = output_file.stat().st_size
            
            # Show completion status
            if failed_pages:
                console.print(f"  [yellow]⚠[/] {successful_pages}/{total_pages} pages ([red]failed: {failed_pages}[/])")
            else:
                console.print(f"  [green]✓[/] {successful_pages}/{total_pages} pages ({success_rate:.0f}%)")
            
            console.print(f"  [dim]Output size:[/] {output_size:,} bytes")
            
            # Log the results
            logging.info(f"PDF {pdf_path.name}: {successful_pages}/{total_pages} pages successful ({success_rate:.1f}%)")
            if failed_pages:
                logging.warning(f"PDF {pdf_path.name}: Failed pages: {failed_pages}")
            
        except Exception as e:
            console.print(f"[red]✗[/] Error processing PDF {pdf_path}: {e}")
            logging.error(f"Error processing PDF {pdf_path}: {e}", exc_info=True)

def main():
    """
    Main function to orchestrate the page-by-page PDF OCR process.
    
    Handles user interaction, model selection, and batch processing of PDFs.
    Each PDF is processed page-by-page for better control and error recovery.
    """
    # Welcome banner
    console.print(Panel(
        "[bold]AI-Powered PDF OCR using Google Gemini[/bold]\n"
        "Processes PDFs page-by-page with native document understanding",
        title="📄 Gemini OCR Processor",
        border_style="cyan"
    ))
    
    # Load environment variables from .env file
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]✗[/] GEMINI_API_KEY not found in environment variables!")
        return
    console.print("[green]✓[/] API Key loaded successfully")

    # Use shared LLM provider for model selection (Gemini models only)
    console.print()
    console.rule("[bold cyan]🤖 Model Selection[/]")
    model_option = get_model_option(None, allowed_keys=["gemini-flash", "gemini-pro"])
    
    # Configure LLM for OCR task based on model type
    # OCR benefits from low temperature and minimal thinking for speed
    is_gemini_3 = "gemini-3" in model_option.model.lower()
    
    if is_gemini_3:
        # Gemini 3 Pro uses thinking_level ("low" or "high"), cannot be disabled
        llm_config = LLMConfig(
            temperature=0.1,  # Low temperature for consistent OCR
            thinking_level="low"  # Use low thinking for faster OCR
        )
    else:
        # Gemini 2.5 Flash uses thinking_budget (0 to disable)
        llm_config = LLMConfig(
            temperature=0.1,  # Low temperature for consistent OCR
            thinking_budget=0  # Disable thinking for Flash
        )
    
    # Display configuration table
    config_table = Table(title="Configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Model", summary_from_option(model_option))
    config_table.add_row("Temperature", str(llm_config.temperature))
    if is_gemini_3:
        config_table.add_row("Thinking Level", llm_config.thinking_level or "low")
    else:
        config_table.add_row("Thinking Budget", str(llm_config.thinking_budget))
    console.print(config_table)
    
    # Set up directory paths
    script_dir = Path(__file__).parent
    pdf_dir = script_dir / "PDF"
    output_dir = script_dir / "OCR_Results"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize the PDF processor
    console.print()
    with console.status("[cyan]Initializing Gemini PDF Processor..."):
        processor = GeminiPDFProcessor(api_key, model_option, llm_config)
    console.print("[green]✓[/] Processor initialized")
    
    # Find all PDF files to process
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        console.print("[red]✗[/] No PDF files found in the PDF directory!")
        return
    
    total_pdfs = len(pdf_files)
    
    # Display files table
    console.print()
    files_table = Table(title=f"📚 PDF Files to Process ({total_pdfs})", box=box.ROUNDED)
    files_table.add_column("#", style="dim", width=4)
    files_table.add_column("Filename", style="cyan")
    files_table.add_column("Size", justify="right", style="green")
    for idx, pdf_file in enumerate(pdf_files, 1):
        size_mb = pdf_file.stat().st_size / (1024 * 1024)
        files_table.add_row(str(idx), pdf_file.name, f"{size_mb:.2f} MB")
    console.print(files_table)
    
    # Track overall statistics
    overall_stats = {
        'total_pdfs': total_pdfs,
        'processed_pdfs': 0,
        'failed_pdfs': 0,
        'total_size_mb': 0,
        'processing_start': time.time()
    }
    
    # Process each PDF file with progress tracking
    console.print()
    console.rule("[bold cyan]📄 Processing PDFs[/]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        pdf_task = progress.add_task("[cyan]Processing PDFs...", total=total_pdfs)
        
        for idx, pdf_path in enumerate(pdf_files, 1):
            progress.update(pdf_task, description=f"[cyan]Processing {pdf_path.name}...")
            
            try:
                processor.process_pdf(pdf_path, output_dir, progress)
                
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
                console.print(f"[red]✗[/] Failed to process {pdf_path.name}: {e}")
                logging.error(f"Failed to process {pdf_path.name}: {e}")
            
            progress.update(pdf_task, advance=1)

    # Calculate processing time
    processing_time = time.time() - overall_stats['processing_start']
    success_rate = (overall_stats['processed_pdfs'] / overall_stats['total_pdfs']) * 100 if overall_stats['total_pdfs'] > 0 else 0
    
    # Print final summary table
    console.print()
    summary_table = Table(title="📈 Processing Summary", box=box.ROUNDED)
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Total PDFs", str(overall_stats['total_pdfs']))
    summary_table.add_row("[green]Successful[/]", f"[green]{overall_stats['processed_pdfs']}[/]")
    summary_table.add_row("[red]Failed[/]", f"[red]{overall_stats['failed_pdfs']}[/]")
    summary_table.add_row("Total Size", f"{overall_stats['total_size_mb']:.2f} MB")
    summary_table.add_row("Processing Time", f"{processing_time/60:.1f} minutes")
    summary_table.add_row("Success Rate", f"{success_rate:.1f}%")
    console.print(summary_table)
    
    # Log final summary
    logging.info(f"Processing complete. {overall_stats['processed_pdfs']}/{overall_stats['total_pdfs']} PDFs processed successfully in {processing_time/60:.1f} minutes")

    console.print(Panel(
        f"[green]✓[/] Processed {overall_stats['processed_pdfs']}/{overall_stats['total_pdfs']} PDFs successfully",
        title="✨ OCR Complete",
        border_style="green"
    ))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠[/] Process interrupted by user")
        logging.info("Process interrupted by user")
    except Exception as e:
        console.print(f"\n[red]✗[/] An error occurred: {e}")
        logging.error(f"An error occurred: {e}", exc_info=True)
