"""
Database Update Script for OCR Results

This script processes OCR text files and updates corresponding Omeka S items
with the extracted text content. It preserves all existing metadata while
adding or updating the bibo:content property with the OCR results, and
annotates each value with the OCR model used (iwac:ocrModel).

Usage:
    python 03_omeka_content_updater.py

Requirements:
    - Environment variables: OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL
    - OCR_Results directory with .txt files named after item IDs
"""

import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich import box

# Initialize rich console
console = Console()

# Shared Omeka client
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.omeka_client import OmekaClient

# Property IDs
BIBO_CONTENT_PROPERTY_ID = 91  # bibo:content
IWAC_OCR_MODEL_PROPERTY_ID = 312  # iwac:ocrModel

# Available OCR models for annotation
OCR_MODELS = {
    "1": {
        "name": "Gemini 3.1 pro",
        "item_id": 78536,
        "data": {
            "type": "resource:item",
            "property_id": IWAC_OCR_MODEL_PROPERTY_ID,
            "property_label": "AI Model - OCR",
            "is_public": True,
            "@id": "https://islam.zmo.de/api/items/78536",
            "value_resource_id": 78536,
            "value_resource_name": "items",
            "url": None,
            "display_title": "Gemini 3.1 pro"
        }
    }
}


def select_ocr_model():
    """
    Prompt user to select the OCR model used for extraction.

    Returns:
        dict: The selected OCR model annotation data, or None if invalid.
    """
    console.print("[bold]Select the OCR model used for extraction:[/]")
    for key, model in OCR_MODELS.items():
        console.print(f"  [cyan]{key}[/]. {model['name']} (item {model['item_id']})")

    choice = console.input("\nEnter choice: ").strip()

    if choice in OCR_MODELS:
        selected = OCR_MODELS[choice]
        console.print(f"[green]✓[/] Selected: [bold]{selected['name']}[/]")
        return selected['data']
    else:
        console.print("[red]✗[/] Invalid choice.")
        return None


def update_item_with_new_content(client: OmekaClient, item_id, new_content, ocr_model_data):
    """
    Update an Omeka S item with new bibo:content while preserving all other metadata.
    Adds an @annotation with iwac:ocrModel to the bibo:content value.

    Args:
        client: OmekaClient instance
        item_id (str): The ID of the item to update
        new_content (str): The new OCR text content to add/update
        ocr_model_data (dict): The OCR model annotation data

    Returns:
        bool: True if update succeeded, False otherwise
    """
    item_data = client.get_item(int(item_id))
    if not item_data:
        console.print(f"  [yellow]⚠[/] No data found for item {item_id}. Skipping.")
        return False

    annotation = {'iwac:ocrModel': [ocr_model_data.copy()]}
    bibo_content_key = 'bibo:content'

    if bibo_content_key in item_data:
        # Update existing bibo:content
        updated = False
        for i, value in enumerate(item_data[bibo_content_key]):
            if value.get('type') == 'literal':
                item_data[bibo_content_key][i]['@value'] = new_content
                item_data[bibo_content_key][i]['@annotation'] = annotation
                updated = True
                break

        if not updated:
            item_data[bibo_content_key].append({
                "type": "literal",
                "property_id": BIBO_CONTENT_PROPERTY_ID,
                "property_label": "content",
                "is_public": True,
                "@value": new_content,
                "@annotation": annotation
            })
    else:
        item_data[bibo_content_key] = [{
            "type": "literal",
            "property_id": BIBO_CONTENT_PROPERTY_ID,
            "property_label": "content",
            "is_public": True,
            "@value": new_content,
            "@annotation": annotation
        }]

    return client.update_item(int(item_id), item_data)

def main():
    """
    Main function to process OCR results and update Omeka S database.

    Reads all text files from the OCR_Results directory and updates the
    corresponding Omeka S items with the extracted text content.
    """
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
    ocr_model_data = select_ocr_model()
    if ocr_model_data is None:
        return
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
    config_table.add_row("OCR Model", ocr_model_data['display_title'])
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
    console.rule("[bold blue]Updating Omeka S Items")
    console.print()

    # Process each text file and update corresponding Omeka S item
    success_count = 0
    failed_count = 0
    skipped_count = 0

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
                result = update_item_with_new_content(client, item_id, content, ocr_model_data)
                if result:
                    success_count += 1
                else:
                    skipped_count += 1

            except Exception as e:
                console.print(f"  [red]✗[/] Error processing file {txt_file}: {e}")
                failed_count += 1

            progress.update(task, advance=1)

    # Display summary
    console.print()
    console.rule("[bold blue]Summary")
    console.print()

    summary_table = Table(box=box.ROUNDED)
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Count", justify="right")
    summary_table.add_row("[green]Successfully Updated[/]", f"[green]{success_count}[/]")
    summary_table.add_row("[yellow]Skipped[/]", f"[yellow]{skipped_count}[/]")
    summary_table.add_row("[red]Failed[/]", f"[red]{failed_count}[/]")
    summary_table.add_row("Total Files", str(len(txt_files)))
    console.print(summary_table)

    console.print(f"\n[green]✓[/] Database update process completed.")

if __name__ == "__main__":
    main()
