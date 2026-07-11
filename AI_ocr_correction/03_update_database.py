"""
This script updates Omeka S items with corrected OCR text content.
It reads corrected text files and updates the corresponding items in the Omeka S database
while preserving existing metadata and handling the bibo:content property appropriately.
"""

import argparse
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)

# Get the directory of the current script and set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# Shared Omeka client
sys.path.insert(0, os.path.join(script_dir, '..'))
from common.omeka_client import OmekaClient

# Directory containing the corrected text files (relative to script location)
DEFAULT_TXT_DIRECTORY = os.path.join(script_dir, 'Corrected_TXT')

BIBO_CONTENT_PROPERTY_ID = 91  # bibo:content in a stock Omeka S install

console = Console()


def update_item_with_new_content(client: OmekaClient, item_id, new_content):
    """
    Update an Omeka S item with new OCR content while preserving existing metadata.

    Args:
        client: Shared Omeka S client
        item_id (str): ID of the Omeka S item to update
        new_content (str): New OCR content to add to the item

    Returns:
        bool: True if update was successful, False otherwise
    """
    item_data = client.get_item(int(item_id))
    if not item_data:
        console.print(f"[yellow]⚠[/] No data found for item {item_id}. Skipping update.")
        return False

    # Omeka S JSON-LD keys property values by term (e.g. 'bibo:content'),
    # not by a generic 'value' key — writing anywhere else is silently dropped.
    values = item_data.setdefault('bibo:content', [])

    for value in values:
        if value.get('property_id') == BIBO_CONTENT_PROPERTY_ID:
            value['@value'] = new_content
            break
    else:
        values.append({
            "type": "literal",
            "property_id": BIBO_CONTENT_PROPERTY_ID,
            "property_label": "content",
            "is_public": True,
            "@value": new_content,
        })

    return client.update_item(int(item_id), item_data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload corrected OCR text to Omeka S (bibo:content)")
    parser.add_argument(
        "--txt-dir",
        default=DEFAULT_TXT_DIRECTORY,
        help="Directory of corrected .txt files named <item_id>.txt",
    )
    args = parser.parse_args()

    console.print(Panel.fit("[bold]OCR Correction — Omeka S Database Update[/]", border_style="cyan"))

    if not os.path.isdir(args.txt_dir):
        console.print(f"[red]✗[/] Corrected-text directory not found: {args.txt_dir}")
        return 1

    try:
        client = OmekaClient.from_env()
    except ValueError as exc:
        console.print(f"[red]✗[/] {exc}")
        return 1

    txt_files = sorted(f for f in os.listdir(args.txt_dir) if f.endswith('.txt'))
    if not txt_files:
        console.print(f"[yellow]⚠[/] No .txt files found in {args.txt_dir}")
        return 1

    successful_updates = 0
    failed_updates = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Updating items...", total=len(txt_files))

        for filename in txt_files:
            item_id = os.path.splitext(filename)[0]  # Extract item ID from filename
            file_path = os.path.join(args.txt_dir, filename)
            progress.update(task, description=f"[cyan]Updating item {item_id}...")

            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    new_content = file.read().strip()
                if not new_content:
                    console.print(f"[yellow]⚠[/] {filename} is empty — skipped")
                    failed_updates.append(filename)
                elif update_item_with_new_content(client, item_id, new_content):
                    successful_updates += 1
                else:
                    failed_updates.append(filename)
            except Exception as e:
                console.print(f"[red]✗[/] Error processing file {filename}: {e}")
                failed_updates.append(filename)

            progress.update(task, advance=1)

    console.print(f"\n[green]✓[/] Successfully updated: {successful_updates}/{len(txt_files)} items")
    if failed_updates:
        console.print(f"[red]✗[/] Failed updates ({len(failed_updates)}):")
        for failed_file in failed_updates:
            console.print(f"  - {failed_file}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
