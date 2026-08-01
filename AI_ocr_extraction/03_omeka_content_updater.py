"""
Database Update Script for OCR Results

This script processes OCR text files and updates corresponding Omeka S items
with the extracted text content. It preserves all existing metadata while
adding or updating the bibo:content property, and annotates each value with
the OCR model used (iwac:ocrModel).

The write step lives in ``common/omeka_text_updater.py``, shared with the
summary, OCR-correction and transcription updaters — so unchanged items are
skipped, ``--dry-run`` works, and the confirmation gate behaves identically
across all four.

Usage:
    python 03_omeka_content_updater.py            # prompts, then updates live
    python 03_omeka_content_updater.py --dry-run  # fetch + report only, writes nothing
    python 03_omeka_content_updater.py --model gemini-flash --yes

Requirements:
    - Environment variables: OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL
    - OCR_Results directory with .txt files named after item IDs
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

# Initialize rich console
console = Console()

# Shared Omeka client and IWAC instance configuration
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.console_utils import key_value_table
from common.omeka_client import OmekaClient
from common.omeka_text_updater import PropertyTarget, run_text_updates, updates_from_directory
from common.iwac_config import (
    AI_MODEL_ITEMS,
    BIBO_CONTENT_PROPERTY_ID,
    IWAC_OCR_MODEL_PROPERTY_ID,
    model_annotation_value,
    select_model_key,
)

CONTENT_TERM = "bibo:content"
OCR_MODEL_TERM = "iwac:ocrModel"


def main() -> int:
    """Upload OCR text to Omeka, annotated with the model that produced it."""
    parser = argparse.ArgumentParser(
        description="Update Omeka S items with OCR extracted text (preserves existing metadata)."
    )
    parser.add_argument(
        "--model", choices=list(AI_MODEL_ITEMS),
        help="OCR model used for extraction. Prompts interactively when omitted.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch each item and report what would change, but write nothing.",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the interactive confirmation before writing.",
    )
    args = parser.parse_args()

    console.print(Panel(
        "[bold]Update Omeka S items with OCR extracted text[/]\n\n"
        "This script reads OCR text files and updates the corresponding "
        "Omeka S items with the bibo:content property and OCR model annotation.",
        title="Omeka S Content Updater",
        border_style="cyan"
    ))
    console.print()

    try:
        client = OmekaClient.from_env()
    except ValueError as e:
        console.print(f"[red]✗[/] {e}")
        return 1

    # Which model produced this OCR? Recorded as an iwac:ocrModel annotation.
    model_key = args.model or select_model_key(default="gemini-3.6-flash")
    if model_key is None:
        return 1
    ocr_model_value = model_annotation_value(
        client.base_url, model_key, IWAC_OCR_MODEL_PROPERTY_ID, "AI Model - OCR"
    )
    target = PropertyTarget(
        term=CONTENT_TERM,
        property_id=BIBO_CONTENT_PROPERTY_ID,
        property_label="content",
        annotation_term=OCR_MODEL_TERM,
        annotation_value=ocr_model_value,
    )
    console.print()

    ocr_folder = Path(__file__).resolve().parent / "OCR_Results"
    console.print(key_value_table([
        ("OCR Results Folder", str(ocr_folder)),
        ("Omeka URL", client.base_url),
        ("OCR Model", ocr_model_value["display_title"]),
        ("Mode", "DRY RUN — no writes" if args.dry_run else "LIVE update"),
    ]))
    console.print()

    if not ocr_folder.exists():
        console.print(f"[red]✗[/] Error: OCR_Results folder not found: {ocr_folder}")
        return 1

    # strip=False: OCR text is stored verbatim, leading/trailing layout included.
    updates = updates_from_directory(ocr_folder, strip=False)
    if not updates:
        console.print("[yellow]⚠[/] No .txt files found in OCR_Results directory.")
        return 1

    console.print(f"[green]✓[/] Found [cyan]{len(updates)}[/] text files to process.")
    console.print()

    stats = run_text_updates(
        client, updates, target,
        console=console,
        dry_run=args.dry_run,
        require_confirmation=not args.yes,
        extra_confirm_lines=[f"Source folder:    {ocr_folder}"],
        description="Updating OCR content...",
    )
    if not stats:
        return 1  # operator declined

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
