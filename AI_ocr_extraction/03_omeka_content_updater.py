"""
Database Update Script for OCR Results

This script processes OCR text files and updates corresponding Omeka S items
with the extracted text content. It preserves all existing metadata while
adding or updating the bibo:content property with the OCR results, and
annotates each value with the OCR model used (iwac:ocrModel).

SAFETY:
    - ``--dry-run`` fetches each item and reports what would change without
      PATCHing anything.
    - In live mode, an interactive confirmation is required before any write.
    (Same pattern as AI_summary_issue/03_update_omeka_toc.py.)

Usage:
    python 03_omeka_content_updater.py            # prompts, then updates live
    python 03_omeka_content_updater.py --dry-run  # fetch + report only, writes nothing

Requirements:
    - Environment variables: OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL
    - OCR_Results directory with .txt files named after item IDs
"""

import argparse
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich import box

# Initialize rich console
console = Console()

# Shared Omeka client and IWAC instance configuration
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.omeka_client import OmekaClient
from common.iwac_config import (
    AI_MODEL_ITEMS,
    BIBO_CONTENT_PROPERTY_ID,
    IWAC_OCR_MODEL_PROPERTY_ID,
    model_annotation_value,
)


def select_ocr_model():
    """
    Prompt user to select the OCR model used for extraction.

    Returns:
        Optional[str]: The selected AI_MODEL_ITEMS key, or None if invalid.
    """
    choices = {str(i): key for i, key in enumerate(AI_MODEL_ITEMS, start=1)}

    console.print("[bold]Select the OCR model used for extraction:[/]")
    for number, key in choices.items():
        model = AI_MODEL_ITEMS[key]
        console.print(f"  [cyan]{number}[/]. {model['display_title']} (item {model['item_id']})")

    choice = console.input("\nEnter choice: ").strip()

    if choice in choices:
        model_key = choices[choice]
        console.print(f"[green]✓[/] Selected: [bold]{AI_MODEL_ITEMS[model_key]['display_title']}[/]")
        return model_key
    else:
        console.print("[red]✗[/] Invalid choice.")
        return None


def apply_ocr_content(item_data, new_content, ocr_model_value):
    """
    Set bibo:content on a fetched item *in place*, preserving other metadata.

    Uses OmekaClient.upsert_property_value to replace the first existing
    literal (or append a new one), then attaches the iwac:ocrModel value
    annotation to that literal.

    Args:
        item_data (dict): The full item JSON-LD representation
        new_content (str): The new OCR text content
        ocr_model_value (dict): The iwac:ocrModel annotation value object
    """
    OmekaClient.upsert_property_value(
        item_data,
        "bibo:content",
        BIBO_CONTENT_PROPERTY_ID,
        new_content,
        property_label="content",
    )

    # Annotate the literal we just wrote with the OCR model used.
    for value in item_data["bibo:content"]:
        if (
            isinstance(value, dict)
            and value.get("property_id") == BIBO_CONTENT_PROPERTY_ID
            and value.get("type", "literal") == "literal"
        ):
            value["@annotation"] = {"iwac:ocrModel": [dict(ocr_model_value)]}
            break


def update_item_with_new_content(client: OmekaClient, item_id, new_content, ocr_model_value, dry_run=False):
    """
    Update an Omeka S item with new bibo:content while preserving all other metadata.
    Adds an @annotation with iwac:ocrModel to the bibo:content value.

    Args:
        client: OmekaClient instance
        item_id (str): The ID of the item to update
        new_content (str): The new OCR text content to add/update
        ocr_model_value (dict): The iwac:ocrModel annotation value object
        dry_run (bool): When True, fetch and report only — never PATCH

    Returns:
        str: One of 'updated', 'would_update', 'not_found', or 'failed'.
    """
    item_data = client.get_item(int(item_id))
    if not item_data:
        console.print(f"  [yellow]⚠[/] No data found for item {item_id}. Skipping.")
        return "not_found"

    apply_ocr_content(item_data, new_content, ocr_model_value)

    if dry_run:
        return "would_update"

    return "updated" if client.update_item(int(item_id), item_data) else "failed"


def main():
    """
    Main function to process OCR results and update Omeka S database.

    Reads all text files from the OCR_Results directory and updates the
    corresponding Omeka S items with the extracted text content.
    """
    parser = argparse.ArgumentParser(
        description="Update Omeka S items with OCR extracted text (preserves existing metadata)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch each item and report what would change, but write nothing.",
    )
    args = parser.parse_args()

    # Display welcome banner
    console.print(Panel(
        "[bold]Update Omeka S items with OCR extracted text[/]\n\n"
        "This script reads OCR text files and updates the corresponding "
        "Omeka S items with the bibo:content property and OCR model annotation.",
        title="Omeka S Content Updater",
        border_style="cyan"
    ))
    console.print()

    # Initialize shared Omeka client
    try:
        client = OmekaClient.from_env()
    except ValueError as e:
        console.print(f"[red]✗[/] {e}")
        return

    # Select OCR model for annotation
    model_key = select_ocr_model()
    if model_key is None:
        return
    ocr_model_value = model_annotation_value(
        client.base_url, model_key, IWAC_OCR_MODEL_PROPERTY_ID, "AI Model - OCR"
    )
    console.print()

    # Set up directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ocr_folder = os.path.join(script_dir, "OCR_Results")

    # Display configuration
    config_table = Table(title="Configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("OCR Results Folder", ocr_folder)
    config_table.add_row("Omeka URL", client.base_url)
    config_table.add_row("OCR Model", ocr_model_value['display_title'])
    config_table.add_row("Mode", "DRY RUN — no writes" if args.dry_run else "LIVE update")
    console.print(config_table)
    console.print()

    # Verify that the OCR results directory exists
    if not os.path.exists(ocr_folder):
        console.print(f"[red]✗[/] Error: OCR_Results folder not found: {ocr_folder}")
        return

    # Find all text files in the OCR results directory
    txt_files = [f for f in os.listdir(ocr_folder) if f.endswith('.txt')]

    if not txt_files:
        console.print("[yellow]⚠[/] No .txt files found in OCR_Results directory.")
        return

    console.print(f"[green]✓[/] Found [cyan]{len(txt_files)}[/] text files to process.")
    console.print()

    # Confirm before touching live data
    console.print(Panel(
        f"Items to update:  {len(txt_files)}\n"
        f"Omeka:            {client.base_url}\n"
        f"Annotation model: {ocr_model_value['display_title']} (item {ocr_model_value['value_resource_id']})\n"
        f"Property written: bibo:content (id {BIBO_CONTENT_PROPERTY_ID})\n"
        f"Mode:             {'DRY RUN — no writes' if args.dry_run else 'LIVE update'}",
        title="About to update Omeka",
        border_style="cyan" if args.dry_run else "yellow",
    ))
    if not args.dry_run:
        confirm = console.input("\n[bold]Proceed with updating these items? [y/N]:[/] ").strip().lower()
        if confirm not in ("y", "yes"):
            console.print("[yellow]Aborted — no changes made.[/]")
            return
    console.print()
    console.rule("[bold blue]Updating Omeka S Items")
    console.print()

    # Process each text file and update corresponding Omeka S item
    stats = {"updated": 0, "would_update": 0, "not_found": 0, "failed": 0}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Updating items...", total=len(txt_files))

        for txt_file in txt_files:
            # Extract item ID from filename (assumes filename format: {item_id}.txt)
            item_id = os.path.splitext(txt_file)[0]
            txt_path = os.path.join(ocr_folder, txt_file)

            # Read the OCR text content
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Update the Omeka S item with the OCR content + annotation
                result = update_item_with_new_content(
                    client, item_id, content, ocr_model_value, dry_run=args.dry_run
                )
                if result == "failed":
                    console.print(f"  [red]✗[/] PATCH failed for item {item_id} (see log)")
                stats[result] += 1

            except Exception as e:
                console.print(f"  [red]✗[/] Error processing file {txt_file}: {e}")
                stats["failed"] += 1

            progress.update(task, advance=1)

    # Display summary
    console.print()
    console.rule("[bold blue]Summary")
    console.print()

    summary_table = Table(box=box.ROUNDED)
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Count", justify="right")
    if args.dry_run:
        summary_table.add_row("[green]Would Update[/]", f"[green]{stats['would_update']}[/]")
    else:
        summary_table.add_row("[green]Successfully Updated[/]", f"[green]{stats['updated']}[/]")
    summary_table.add_row("[yellow]Not Found (skipped)[/]", f"[yellow]{stats['not_found']}[/]")
    summary_table.add_row("[red]Failed[/]", f"[red]{stats['failed']}[/]")
    summary_table.add_row("Total Files", str(len(txt_files)))
    console.print(summary_table)

    if args.dry_run:
        console.print("\n[green]✓[/] Dry run completed — no changes were made.")
    else:
        console.print("\n[green]✓[/] Database update process completed.")

if __name__ == "__main__":
    main()
