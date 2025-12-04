"""
OCR Text Correction Pipeline using LLM Provider

This script uses the shared LLM provider to correct and improve OCR text output.
It processes text files containing raw OCR output, applies AI-powered corrections,
and saves the improved versions while handling large texts through chunking.

Supports multiple models via --model flag:
- gemini-flash (default): Gemini 2.5 Flash - fast, cost-effective
- gemini-pro: Gemini 3 Pro - highest quality
- gpt-5-mini: OpenAI GPT-5 mini - cost-optimized
- mistral-large: Mistral Large 3 - multimodal MoE
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich import box

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.llm_provider import (
    LLMConfig,
    build_llm_client,
    get_model_option,
    summary_from_option,
)

# Load environment variables from .env file
load_dotenv()

# Initialize Rich console
console = Console()

# Allowed models for this pipeline
ALLOWED_MODELS = ["gemini-flash", "gemini-pro", "gpt-5-mini", "gpt-5.1", "mistral-large", "ministral-14b"]


def get_system_instruction() -> str:
    """
    Load the OCR correction system prompt from markdown file.
    
    Returns:
        str: System instruction for OCR correction
    """
    script_dir = Path(__file__).resolve().parent
    prompt_file = script_dir / "ocr_correction_prompt.md"
    
    try:
        return prompt_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        console.print("[yellow]⚠[/] Prompt file not found, using fallback instruction")
        return (
            "You are an expert OCR correction assistant. "
            "Correct OCR errors, fix formatting, and improve text structure "
            "while preserving the original meaning and historical authenticity."
        )


def correct_text_with_llm(client, text: str, system_prompt: str) -> str:
    """
    Use LLM to correct OCR errors and improve text formatting.
    
    Args:
        client: Initialized LLM client from llm_provider
        text: Raw OCR text to be corrected
        system_prompt: System instruction for OCR correction
        
    Returns:
        str: Corrected and improved text
    """
    if not text or not text.strip():
        return text
    
    user_prompt = (
        "Here is the OCR text that needs correction. "
        "Analyze the structure, fix errors, and ensure proper formatting:\n\n"
        f"{text}"
    )
    
    try:
        return client.generate(system_prompt, user_prompt)
    except Exception as e:
        console.print(f"[red]✗[/] Error during API call: {e}")
        return text  # Return original text if correction fails

def split_text(text: str, max_chars: int = 200000) -> list[str]:
    """
    Split long texts into manageable chunks for LLM processing.
    
    This function implements intelligent text splitting that:
    - Respects sentence boundaries where possible
    - Handles extremely long sentences by splitting on word boundaries
    - Ensures chunks stay within character limits
    
    Args:
        text: Input text to be split
        max_chars: Maximum characters per chunk (default: 200000)
        
    Returns:
        List of text chunks ready for processing
    """
    # If text is shorter than limit, return as single chunk
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Split into sentences (roughly) by splitting on periods followed by spaces
    sentences = text.replace(". ", ".|").split("|")
    
    for sentence in sentences:
        sentence = sentence.strip()
        sentence_length = len(sentence)
        
        # If adding this sentence would exceed limit, save current chunk and start new one
        if current_length + sentence_length > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Handle sentences that are longer than max_chars
            if sentence_length > max_chars:
                # Split long sentence into smaller pieces
                words = sentence.split()
                temp_chunk = []
                temp_length = 0
                
                for word in words:
                    word_length = len(word) + 1  # +1 for space
                    if temp_length + word_length > max_chars:
                        if temp_chunk:
                            chunks.append(" ".join(temp_chunk))
                            temp_chunk = []
                            temp_length = 0
                    temp_chunk.append(word)
                    temp_length += word_length
                
                if temp_chunk:
                    chunks.append(" ".join(temp_chunk))
            else:
                current_chunk.append(sentence)
                current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add any remaining text
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def process_file(
    client,
    input_file_path: Path,
    output_file_path: Path,
    system_prompt: str,
    max_length: int,
) -> tuple[bool, str]:
    """
    Process a single text file through the OCR correction pipeline.
    
    Args:
        client: Initialized LLM client
        input_file_path: Path to input text file
        output_file_path: Path where corrected text will be saved
        system_prompt: System instruction for OCR correction
        max_length: Maximum text length before chunking is required
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        original_text = input_file_path.read_text(encoding="utf-8")
        
        if not original_text.strip():
            return False, "Empty file"
        
        if len(original_text) > max_length:
            chunks = split_text(original_text, max_length)
            corrected_chunks = [
                correct_text_with_llm(client, chunk, system_prompt)
                for chunk in chunks
            ]
            corrected_text = " ".join(corrected_chunks)
        else:
            corrected_text = correct_text_with_llm(client, original_text, system_prompt)
        
        output_file_path.write_text(corrected_text, encoding="utf-8")
        return True, "Success"
        
    except Exception as e:
        return False, str(e)


def process_txt_files(
    client,
    input_dir: Path,
    output_dir: Path,
    system_prompt: str,
    max_length: int = 200000,
) -> tuple[int, int]:
    """
    Process all text files in a directory through the OCR correction pipeline.
    
    Args:
        client: Initialized LLM client
        input_dir: Directory containing input text files
        output_dir: Directory where corrected files will be saved
        system_prompt: System instruction for OCR correction
        max_length: Maximum text length before chunking is required
        
    Returns:
        Tuple of (success_count, error_count)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    txt_files = sorted(input_dir.glob("*.txt"))
    
    if not txt_files:
        console.print("[yellow]⚠[/] No .txt files found in input directory")
        return 0, 0
    
    success_count = 0
    error_count = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing files...", total=len(txt_files))
        
        for txt_file in txt_files:
            output_file = output_dir / txt_file.name
            success, message = process_file(
                client, txt_file, output_file, system_prompt, max_length
            )
            
            if success:
                success_count += 1
            else:
                error_count += 1
                console.print(f"[red]✗[/] {txt_file.name}: {message}")
            
            progress.update(task, advance=1, description=f"Processing {txt_file.name}")
    
    return success_count, error_count


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Correct OCR text using AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=ALLOWED_MODELS,
        default=None,
        help="AI model to use for correction (default: interactive selection)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Input directory containing .txt files (default: ./TXT)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for corrected files (default: ./Corrected_TXT)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=200000,
        help="Maximum text length before chunking (default: 200000)",
    )
    return parser.parse_args()


def main():
    """
    Main execution function that orchestrates the OCR correction process.
    
    Sets up directories, initializes the LLM client, and processes all text files
    while providing rich console output for progress and status.
    """
    args = parse_args()
    
    # Display welcome banner
    console.print(
        Panel(
            "[bold]OCR Text Correction Pipeline[/bold]\n\n"
            "Corrects OCR errors in text files using AI models.\n"
            "Preserves historical authenticity while fixing technical artifacts.",
            title="📝 OCR Correction",
            border_style="cyan",
        )
    )
    
    # Get script directory for default paths
    script_dir = Path(__file__).resolve().parent
    input_dir = args.input_dir or (script_dir / "TXT")
    output_dir = args.output_dir or (script_dir / "Corrected_TXT")
    
    # Get model selection
    try:
        model_option = get_model_option(args.model, allowed_keys=ALLOWED_MODELS)
    except ValueError as e:
        console.print(f"[red]✗[/] {e}")
        sys.exit(1)
    
    # Configure LLM based on provider
    # For Gemini Flash: disable thinking for faster, cheaper processing
    # For other models: use appropriate defaults
    if model_option.key == "gemini-flash":
        config = LLMConfig(temperature=0.2, thinking_budget=0)  # Disable thinking
    elif model_option.key == "gemini-pro":
        config = LLMConfig(temperature=0.2, thinking_level="low")  # Minimal thinking
    else:
        config = LLMConfig(temperature=0.2)  # Default for OpenAI/Mistral
    
    # Display configuration
    console.rule("[bold cyan]Configuration")
    
    config_table = Table(box=box.ROUNDED, show_header=False)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Model", summary_from_option(model_option))
    config_table.add_row("Input Directory", str(input_dir))
    config_table.add_row("Output Directory", str(output_dir))
    config_table.add_row("Max Chunk Length", f"{args.max_length:,} chars")
    if model_option.key == "gemini-flash":
        config_table.add_row("Thinking", "Disabled (thinking_budget=0)")
    elif model_option.key == "gemini-pro":
        config_table.add_row("Thinking Level", "low")
    console.print(config_table)
    console.print()
    
    # Validate input directory
    if not input_dir.exists():
        console.print(f"[red]✗[/] Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Count files
    txt_files = list(input_dir.glob("*.txt"))
    console.print(f"[cyan]📁[/] Found [bold]{len(txt_files)}[/bold] text files to process")
    console.print()
    
    if not txt_files:
        console.print("[yellow]⚠[/] No files to process. Exiting.")
        sys.exit(0)
    
    # Initialize LLM client
    console.print("[cyan]🔌[/] Initializing LLM client...")
    try:
        client = build_llm_client(model_option, config=config)
    except RuntimeError as e:
        console.print(f"[red]✗[/] Failed to initialize client: {e}")
        sys.exit(1)
    
    # Load system prompt
    system_prompt = get_system_instruction()
    console.print(f"[green]✓[/] System prompt loaded ({len(system_prompt):,} chars)")
    console.print()
    
    # Process files
    console.rule("[bold cyan]Processing Files")
    success_count, error_count = process_txt_files(
        client, input_dir, output_dir, system_prompt, args.max_length
    )
    
    # Display summary
    console.print()
    console.rule("[bold cyan]Summary")
    
    summary_table = Table(box=box.ROUNDED, show_header=False)
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value")
    summary_table.add_row("Total Files", str(len(txt_files)))
    summary_table.add_row("Successful", f"[green]{success_count}[/]")
    summary_table.add_row("Failed", f"[red]{error_count}[/]" if error_count else "[green]0[/]")
    summary_table.add_row("Output Location", str(output_dir))
    console.print(summary_table)
    
    if error_count == 0:
        console.print(
            Panel(
                f"[green]✓ All {success_count} files corrected successfully![/]",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[yellow]⚠ Completed with {error_count} error(s)[/]",
                border_style="yellow",
            )
        )


if __name__ == "__main__":
    main()
