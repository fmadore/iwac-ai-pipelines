"""
ALTO XML OCR Correction Pipeline

This script corrects OCR text in ALTO XML files while preserving all coordinates
and layout information. It extracts text from TextBlock elements (for better context),
sends them to an LLM for correction, and writes back the corrected text to the original
String elements while keeping HPOS, VPOS, WIDTH, HEIGHT untouched.

The approach:
1. Parse ALTO XML and group TextLines by their parent TextBlock (for paragraph context)
2. Send each block's lines to LLM for correction using structured output
3. Re-tokenize corrected text and map back to original String elements
4. Update only CONTENT attributes, preserving all coordinate data

Supports multiple models via --model flag:
- gemini-flash (default): Gemini 2.5 Flash - fast, cost-effective
- gemini-pro: Gemini 3 Pro - highest quality
- gpt-5-mini: OpenAI GPT-5 mini - cost-optimized
- mistral-large: Mistral Large 3 - multimodal MoE
"""

import argparse
import copy
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich import box
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

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.llm_provider import (
    LLMConfig,
    build_llm_client,
    get_model_option,
    summary_from_option,
)

# Load environment variables
load_dotenv()

# Initialize Rich console
console = Console()

# Allowed models for this pipeline
ALLOWED_MODELS = [
    "gemini-flash",
    "gemini-pro",
    "gpt-5-mini",
    "gpt-5.1",
    "mistral-large",
    "ministral-14b",
]

# ALTO XML namespaces (common versions)
ALTO_NAMESPACES = {
    "alto2": "http://www.loc.gov/standards/alto/ns-v2#",
    "alto3": "http://www.loc.gov/standards/alto/ns-v3#",
    "alto4": "http://www.loc.gov/standards/alto/ns-v4#",
}


# --- Pydantic Models for Structured Output ---


class CorrectedLine(BaseModel):
    """A single corrected text line with token mapping."""

    line_index: int = Field(description="The 0-based index of the line being corrected")
    original_tokens: list[str] = Field(
        description="The original tokens (words) from the line, in order"
    )
    corrected_tokens: list[str] = Field(
        description="The corrected tokens, must be same length as original_tokens"
    )
    # TODO: Consider adding flagged_tokens field to capture tokens that need
    # manual review (e.g., merged words like "ilest" that cannot be split to
    # "il est" due to coordinate preservation constraints). This would allow
    # exporting a CSV of problematic tokens for post-processing.
    # See: https://github.com/fmadore/iwac-ai-pipelines/issues/XXX
    # flagged_tokens: list[str] = Field(
    #     default_factory=list,
    #     description="Tokens needing manual review (merged/split words that couldn't be corrected)"
    # )


class LineCorrectionBatch(BaseModel):
    """Batch of corrected lines with strict token alignment."""

    lines: list[CorrectedLine] = Field(
        description="List of corrected lines with token-level alignment"
    )


# --- ALTO Parsing Utilities ---


@dataclass
class StringElement:
    """Represents an ALTO String element with its attributes."""

    element: ET.Element
    content: str
    hpos: Optional[float] = None
    vpos: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None


@dataclass
class TextLineData:
    """Represents an ALTO TextLine with its String children."""

    element: ET.Element
    strings: list[StringElement]
    full_text: str  # Concatenated text from all String elements
    line_id: str = ""  # ALTO ID attribute if present


@dataclass
class TextBlockData:
    """Represents an ALTO TextBlock with its TextLine children."""

    element: ET.Element
    lines: list[TextLineData]
    block_id: str = ""  # ALTO ID attribute if present
    full_text: str = ""  # Concatenated text from all lines (for context)


def detect_alto_namespace(root: ET.Element) -> Optional[str]:
    """Detect the ALTO namespace from the root element."""
    # Check the tag for namespace
    if root.tag.startswith("{"):
        ns_end = root.tag.index("}")
        return root.tag[1:ns_end]

    # Check common namespaces in attributes
    for ns_uri in ALTO_NAMESPACES.values():
        if ns_uri in str(ET.tostring(root)):
            return ns_uri

    return None


def parse_alto_xml(file_path: Path) -> tuple[ET.ElementTree, list[TextBlockData], str]:
    """
    Parse an ALTO XML file and extract TextBlocks with their TextLines.

    Grouping by TextBlock provides better context for OCR correction,
    as the LLM can see full paragraphs instead of isolated lines.

    Args:
        file_path: Path to the ALTO XML file

    Returns:
        Tuple of (ElementTree, list of TextBlockData, namespace URI)
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Detect namespace
    ns = detect_alto_namespace(root)
    ns_prefix = f"{{{ns}}}" if ns else ""

    text_blocks = []

    # Find all TextBlock elements
    for text_block in root.iter(f"{ns_prefix}TextBlock"):
        block_id = text_block.get("ID", "")
        lines = []
        block_text_parts = []

        # Extract all TextLine elements within this TextBlock
        for text_line in text_block.iter(f"{ns_prefix}TextLine"):
            line_id = text_line.get("ID", "")
            strings = []
            text_parts = []

            # Extract all String elements within this TextLine
            for string_elem in text_line.iter(f"{ns_prefix}String"):
                content = string_elem.get("CONTENT", "")
                string_data = StringElement(
                    element=string_elem,
                    content=content,
                    hpos=_safe_float(string_elem.get("HPOS")),
                    vpos=_safe_float(string_elem.get("VPOS")),
                    width=_safe_float(string_elem.get("WIDTH")),
                    height=_safe_float(string_elem.get("HEIGHT")),
                )
                strings.append(string_data)
                text_parts.append(content)

            if strings:
                line_full_text = " ".join(text_parts)
                lines.append(
                    TextLineData(
                        element=text_line,
                        strings=strings,
                        full_text=line_full_text,
                        line_id=line_id,
                    )
                )
                block_text_parts.append(line_full_text)

        if lines:
            text_blocks.append(
                TextBlockData(
                    element=text_block,
                    lines=lines,
                    block_id=block_id,
                    full_text=" ".join(block_text_parts),
                )
            )

    return tree, text_blocks, ns or ""


def _safe_float(value: Optional[str]) -> Optional[float]:
    """Safely convert a string to float."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# --- LLM Correction Logic ---


def get_system_prompt() -> str:
    """
    Load the ALTO OCR correction system prompt from markdown file.

    Returns:
        str: System instruction for ALTO OCR correction
    """
    script_dir = Path(__file__).resolve().parent
    prompt_file = script_dir / "alto_correction_prompt.md"

    try:
        return prompt_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        console.print("[yellow]⚠[/] Prompt file not found, using fallback instruction")
        return _get_fallback_prompt()


def _get_fallback_prompt() -> str:
    """Fallback prompt if the markdown file is not found."""
    return """You are an expert OCR correction assistant for ALTO XML documents.

## Your Task
Correct OCR errors in text lines while maintaining STRICT TOKEN ALIGNMENT.

## Critical Rules

1. **PRESERVE TOKEN COUNT**: The number of corrected tokens MUST EXACTLY MATCH the number of original tokens.
   - If original has 5 words, corrected MUST have 5 words
   - Never merge or split tokens
   - Never add or remove tokens

2. **Token Mapping**: Each original token maps 1:1 to a corrected token:
   - Token 0 → Corrected Token 0
   - Token 1 → Corrected Token 1
   - etc.

3. **What to Correct**:
   - OCR character substitutions (e.g., "0" → "O", "rn" → "m", "l" → "I")
   - Obvious misspellings caused by OCR (e.g., "tlie" → "the")
   - Wrong diacritics from OCR errors (e.g., "café" misread as "cafÃ©")

4. **What to PRESERVE**:
   - Historical spellings (e.g., "connexion", "shew", "to-day")
   - Period-appropriate vocabulary and grammar
   - Author's original language style
   - If unsure, keep the original

5. **If a token cannot be corrected**, return it unchanged.

## Example

Input line: "Tlie gouvernernent lias decided"
Original tokens: ["Tlie", "gouvernernent", "lias", "decided"]
Corrected tokens: ["The", "gouvernement", "has", "decided"]

Note: "gouvernement" is kept as French spelling, not changed to English "government".

## Output Format

Return structured JSON with exact token alignment for each line."""


def correct_lines_batch(
    client,
    lines: list[TextLineData],
    system_prompt: str,
    batch_size: int = 20,
) -> dict[int, list[str]]:
    """
    Correct a batch of text lines using structured LLM output.

    Args:
        client: LLM client
        lines: List of TextLineData to correct
        system_prompt: System prompt for the LLM
        batch_size: Number of lines to process per API call (used for very large blocks)

    Returns:
        Dictionary mapping line index to corrected token list
    """
    corrections = {}

    for batch_start in range(0, len(lines), batch_size):
        batch_end = min(batch_start + batch_size, len(lines))
        batch = lines[batch_start:batch_end]

        # Build the user prompt with line data
        user_prompt_parts = ["Correct the following OCR text lines:\n"]

        for i, line_data in enumerate(batch):
            global_idx = batch_start + i
            tokens = [s.content for s in line_data.strings]
            user_prompt_parts.append(
                f"\nLine {global_idx}:\n"
                f"  Text: \"{line_data.full_text}\"\n"
                f"  Tokens ({len(tokens)}): {tokens}"
            )

        user_prompt = "\n".join(user_prompt_parts)

        try:
            result = client.generate_structured(
                system_prompt,
                user_prompt,
                LineCorrectionBatch,
            )

            # Process results
            for corrected_line in result.lines:
                idx = corrected_line.line_index
                if batch_start <= idx < batch_end:
                    original_count = len(batch[idx - batch_start].strings)
                    corrected_count = len(corrected_line.corrected_tokens)

                    if corrected_count == original_count:
                        corrections[idx] = corrected_line.corrected_tokens
                    else:
                        # Token count mismatch - keep original
                        console.print(
                            f"[yellow]⚠[/] Line {idx}: token count mismatch "
                            f"(expected {original_count}, got {corrected_count}), keeping original"
                        )

        except Exception as e:
            console.print(f"[red]✗[/] Error processing batch {batch_start}-{batch_end}: {e}")
            # Continue with next batch

    return corrections


def correct_block(
    client,
    block: TextBlockData,
    system_prompt: str,
    max_lines_per_request: int = 50,
) -> dict[int, list[str]]:
    """
    Correct all lines in a TextBlock, sending the full block for context.

    This provides better OCR correction because the LLM can see the full
    paragraph context, including word breaks across lines (e.g., "rache-" + "ter").

    Args:
        client: LLM client
        block: TextBlockData containing all lines in the block
        system_prompt: System prompt for the LLM
        max_lines_per_request: Maximum lines to send in one request (for very large blocks)

    Returns:
        Dictionary mapping line index (within block) to corrected token list
    """
    lines = block.lines
    corrections = {}

    # For small blocks, send all at once
    if len(lines) <= max_lines_per_request:
        # Build the user prompt with full block context
        user_prompt_parts = [
            f"Correct the following OCR text block (Block ID: {block.block_id or 'unknown'}).\n",
            f"Full block text for context: \"{block.full_text}\"\n",
            "\nLines to correct:\n"
        ]

        for i, line_data in enumerate(lines):
            tokens = [s.content for s in line_data.strings]
            user_prompt_parts.append(
                f"\nLine {i}:\n"
                f"  Text: \"{line_data.full_text}\"\n"
                f"  Tokens ({len(tokens)}): {tokens}"
            )

        user_prompt = "\n".join(user_prompt_parts)

        try:
            result = client.generate_structured(
                system_prompt,
                user_prompt,
                LineCorrectionBatch,
            )

            # Process results
            for corrected_line in result.lines:
                idx = corrected_line.line_index
                if 0 <= idx < len(lines):
                    original_count = len(lines[idx].strings)
                    corrected_count = len(corrected_line.corrected_tokens)

                    if corrected_count == original_count:
                        corrections[idx] = corrected_line.corrected_tokens
                    else:
                        console.print(
                            f"[yellow]⚠[/] Block {block.block_id}, Line {idx}: token count mismatch "
                            f"(expected {original_count}, got {corrected_count}), keeping original"
                        )

        except Exception as e:
            console.print(f"[red]✗[/] Error processing block {block.block_id}: {e}")

    else:
        # For very large blocks, fall back to batch processing
        corrections = correct_lines_batch(client, lines, system_prompt, max_lines_per_request)

    return corrections


def apply_corrections_to_alto(
    text_blocks: list[TextBlockData],
    all_corrections: dict[str, dict[int, list[str]]],
    ns: str,
) -> int:
    """
    Apply corrections to the ALTO XML String elements.

    Args:
        text_blocks: List of TextBlockData with references to XML elements
        all_corrections: Dictionary mapping block_id to {line_index: corrected_tokens}
        ns: ALTO namespace URI

    Returns:
        Number of String elements updated
    """
    updated_count = 0

    for block in text_blocks:
        block_key = block.block_id or id(block)
        if block_key not in all_corrections:
            continue

        corrections = all_corrections[block_key]

        for line_idx, corrected_tokens in corrections.items():
            if line_idx >= len(block.lines):
                continue

            line_data = block.lines[line_idx]

            # Verify token count matches
            if len(corrected_tokens) != len(line_data.strings):
                continue

            # Apply corrections to each String element
            for string_data, new_content in zip(line_data.strings, corrected_tokens):
                old_content = string_data.content
                if old_content != new_content:
                    string_data.element.set("CONTENT", new_content)
                    updated_count += 1

    return updated_count


# --- File Processing ---


def process_alto_file(
    client,
    input_path: Path,
    output_path: Path,
    system_prompt: str,
    max_lines_per_request: int = 50,
) -> tuple[bool, int, int, int, str]:
    """
    Process a single ALTO XML file.

    Args:
        client: LLM client
        input_path: Path to input ALTO XML
        output_path: Path for corrected output
        system_prompt: System prompt for LLM
        max_lines_per_request: Max lines per API request for large blocks

    Returns:
        Tuple of (success, blocks_processed, lines_processed, strings_updated, message)
    """
    try:
        # Parse ALTO XML
        tree, text_blocks, ns = parse_alto_xml(input_path)

        if not text_blocks:
            return True, 0, 0, 0, "No text blocks found"

        # Count total lines
        total_lines = sum(len(block.lines) for block in text_blocks)

        # Get corrections for each block
        all_corrections = {}
        for block in text_blocks:
            block_key = block.block_id or id(block)
            corrections = correct_block(client, block, system_prompt, max_lines_per_request)
            if corrections:
                all_corrections[block_key] = corrections

        # Apply corrections to XML
        updated_count = apply_corrections_to_alto(text_blocks, all_corrections, ns)

        # Write corrected XML
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(output_path, encoding="unicode", xml_declaration=True)

        return True, len(text_blocks), total_lines, updated_count, "Success"

    except ET.ParseError as e:
        return False, 0, 0, 0, f"XML parse error: {e}"
    except Exception as e:
        return False, 0, 0, 0, str(e)


def process_alto_files(
    client,
    input_dir: Path,
    output_dir: Path,
    system_prompt: str,
    max_lines_per_request: int = 50,
) -> tuple[int, int, int, int, int]:
    """
    Process all ALTO XML files in a directory.

    Args:
        client: LLM client
        input_dir: Directory containing ALTO XML files
        output_dir: Directory for corrected files
        system_prompt: System prompt for LLM
        max_lines_per_request: Max lines per API request for large blocks

    Returns:
        Tuple of (success_count, error_count, total_blocks, total_lines, total_strings_updated)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find ALTO XML files (common extensions)
    alto_files = sorted(
        list(input_dir.glob("*.xml"))
        + list(input_dir.glob("*.alto"))
        + list(input_dir.glob("*.ALTO"))
    )

    if not alto_files:
        console.print("[yellow]⚠[/] No ALTO XML files found in input directory")
        return 0, 0, 0, 0, 0

    success_count = 0
    error_count = 0
    total_blocks = 0
    total_lines = 0
    total_strings = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing ALTO files...", total=len(alto_files))

        for alto_file in alto_files:
            output_file = output_dir / alto_file.name
            success, blocks, lines, strings, message = process_alto_file(
                client, alto_file, output_file, system_prompt, max_lines_per_request
            )

            if success:
                success_count += 1
                total_blocks += blocks
                total_lines += lines
                total_strings += strings
                if strings > 0:
                    console.print(
                        f"[green]✓[/] {alto_file.name}: {blocks} blocks, {lines} lines, {strings} corrections"
                    )
            else:
                error_count += 1
                console.print(f"[red]✗[/] {alto_file.name}: {message}")

            progress.update(task, advance=1, description=f"Processing {alto_file.name}")

    return success_count, error_count, total_blocks, total_lines, total_strings


# --- CLI ---


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Correct OCR text in ALTO XML files while preserving coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with default model (interactive selection)
  python 02_correct_alto_xml.py

  # Use Gemini Flash (fast, no thinking)
  python 02_correct_alto_xml.py --model gemini-flash

  # Custom directories and max lines per request
  python 02_correct_alto_xml.py --input-dir ./ALTO --output-dir ./ALTO_Corrected --max-lines 40
""",
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
        help="Input directory containing ALTO XML files (default: ./ALTO)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for corrected files (default: ./ALTO_Corrected)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=50,
        help="Maximum lines per API request for large blocks (default: 50)",
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Display welcome banner
    console.print(
        Panel(
            "[bold]ALTO XML OCR Correction Pipeline[/bold]\n\n"
            "Corrects OCR errors in ALTO XML files while preserving\n"
            "all coordinate data (HPOS, VPOS, WIDTH, HEIGHT).\n\n"
            "Processes by TextBlock for better paragraph context.\n"
            "Uses structured output for reliable token-level alignment.",
            title="📄 ALTO Correction",
            border_style="cyan",
        )
    )

    # Get script directory for default paths
    script_dir = Path(__file__).resolve().parent
    input_dir = args.input_dir or (script_dir / "ALTO")
    output_dir = args.output_dir or (script_dir / "ALTO_Corrected")

    # Get model selection
    try:
        model_option = get_model_option(args.model, allowed_keys=ALLOWED_MODELS)
    except ValueError as e:
        console.print(f"[red]✗[/] {e}")
        sys.exit(1)

    # Configure LLM based on provider
    if model_option.key == "gemini-flash":
        config = LLMConfig(temperature=0.1, thinking_budget=0)  # Disable thinking
    elif model_option.key == "gemini-pro":
        config = LLMConfig(temperature=0.1, thinking_level="low")  # Minimal thinking
    else:
        config = LLMConfig(temperature=0.1)  # Low temp for consistency

    # Display configuration
    console.rule("[bold cyan]Configuration")

    config_table = Table(box=box.ROUNDED, show_header=False)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Model", summary_from_option(model_option))
    config_table.add_row("Input Directory", str(input_dir))
    config_table.add_row("Output Directory", str(output_dir))
    config_table.add_row("Max Lines/Request", f"{args.max_lines} (for large blocks)")
    if model_option.key == "gemini-flash":
        config_table.add_row("Thinking", "Disabled (thinking_budget=0)")
    elif model_option.key == "gemini-pro":
        config_table.add_row("Thinking Level", "low")
    console.print(config_table)
    console.print()

    # Validate input directory
    if not input_dir.exists():
        console.print(f"[red]✗[/] Input directory not found: {input_dir}")
        console.print(
            "\n[dim]Create an ALTO/ directory with your ALTO XML files, or use --input-dir[/]"
        )
        sys.exit(1)

    # Count files
    alto_files = (
        list(input_dir.glob("*.xml"))
        + list(input_dir.glob("*.alto"))
        + list(input_dir.glob("*.ALTO"))
    )
    console.print(f"[cyan]📁[/] Found [bold]{len(alto_files)}[/bold] ALTO XML files to process")
    console.print()

    if not alto_files:
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
    system_prompt = get_system_prompt()
    console.print(f"[green]✓[/] System prompt loaded ({len(system_prompt):,} chars)")
    console.print()

    # Process files
    console.rule("[bold cyan]Processing Files")
    success_count, error_count, total_blocks, total_lines, total_strings = process_alto_files(
        client, input_dir, output_dir, system_prompt, args.max_lines
    )

    # Display summary
    console.print()
    console.rule("[bold cyan]Summary")

    summary_table = Table(box=box.ROUNDED, show_header=False)
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value")
    summary_table.add_row("Total Files", str(len(alto_files)))
    summary_table.add_row("Successful", f"[green]{success_count}[/]")
    summary_table.add_row(
        "Failed", f"[red]{error_count}[/]" if error_count else "[green]0[/]"
    )
    summary_table.add_row("Text Blocks Processed", f"{total_blocks:,}")
    summary_table.add_row("Text Lines Processed", f"{total_lines:,}")
    summary_table.add_row("Strings Corrected", f"{total_strings:,}")
    summary_table.add_row("Output Location", str(output_dir))
    console.print(summary_table)

    if error_count == 0:
        console.print(
            Panel(
                f"[green]✓ All {success_count} files processed successfully![/]\n"
                f"[dim]{total_strings:,} String elements corrected across {total_blocks:,} blocks ({total_lines:,} lines)[/]",
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
