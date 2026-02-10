"""
This script retrieves OCR text from items in an Omeka S database and saves them as individual text files.
It handles pagination, concurrent processing, and includes error handling and logging functionality.
"""

import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich import box

# Initialize Rich console
console = Console()

# Shared Omeka client
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.omeka_client import OmekaClient

# Output directory is set relative to the script's location for portability
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TXT")
MAX_WORKERS = 5  # Maximum number of concurrent threads for processing

# Configure logging to track script execution and errors
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_and_save_content(item, output_dir):
    """
    Extract OCR text from an item and save it to a file.

    Args:
        item (dict): Item data containing OCR text
        output_dir (str): Directory where text files will be saved

    Returns:
        tuple: (item_id, success_status, skipped)
    """
    item_id = item["o:id"]
    file_name = os.path.join(output_dir, f"{item_id}.txt")

    extracted_text = item.get("extracttext:extracted_text", [])
    content_values = [content["@value"] for content in extracted_text if "@value" in content]
    content_text = "\n".join(content_values)

    # Skip creating file if there's no content
    if not content_text.strip():
        return item_id, True, True  # skipped=True

    try:
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(content_text)
        return item_id, True, False  # skipped=False
    except IOError as e:
        logging.error(f"Error writing file for item {item_id}: {e}")
        return item_id, False, False


def process_items(items, output_dir):
    """
    Process multiple items concurrently using a thread pool.

    Args:
        items (list): List of items to process
        output_dir (str): Directory where text files will be saved

    Returns:
        tuple: (success_count, skipped_count, error_count)
    """
    success_count = 0
    skipped_count = 0
    error_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Extracting OCR text...", total=len(items))

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_item = {executor.submit(extract_and_save_content, item, output_dir): item for item in items}
            for future in as_completed(future_to_item):
                item_id, success, skipped = future.result()
                if skipped:
                    skipped_count += 1
                elif success:
                    success_count += 1
                else:
                    error_count += 1
                progress.update(task, advance=1)

    return success_count, skipped_count, error_count


def main():
    """
    Main execution function that orchestrates the OCR text extraction process.

    Prompts for item set ID, initializes the client, and processes all items
    while providing progress feedback through rich console output.
    """
    # Welcome banner
    console.print(Panel(
        "Extract OCR text from Omeka S items and save as individual text files.",
        title="OCR Text Extractor",
        border_style="cyan"
    ))

    # Initialize shared Omeka client
    try:
        client = OmekaClient.from_env()
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        return

    # Get item set ID from user
    console.print()
    item_set_id = int(console.input("[cyan]Enter the item set ID:[/] "))
    console.print()

    # Create the OUTPUT_DIR in the same folder as the script
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Display configuration
    config_table = Table(title="Configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Item Set ID", str(item_set_id))
    config_table.add_row("Output Directory", OUTPUT_DIR)
    config_table.add_row("Max Workers", str(MAX_WORKERS))
    console.print(config_table)
    console.print()

    # Fetch items
    console.rule("[bold cyan]Fetching Items")
    with console.status("[cyan]Fetching items from Omeka S...", spinner="dots"):
        items = client.get_items(item_set_id)
    console.print(f"[green]{chr(10003)}[/] Fetched [bold]{len(items)}[/] items from Omeka S")
    console.print()

    if not items:
        console.print("[yellow]No items found in this item set.[/]")
        return

    # Process items
    console.rule("[bold cyan]Extracting Content")
    success_count, skipped_count, error_count = process_items(items, OUTPUT_DIR)
    console.print()

    # Summary
    console.rule("[bold cyan]Summary")
    summary_table = Table(box=box.ROUNDED)
    summary_table.add_column("Status", style="bold")
    summary_table.add_column("Count", justify="right")
    summary_table.add_row("[green]Files saved[/]", str(success_count))
    summary_table.add_row("[yellow]Skipped (no text)[/]", str(skipped_count))
    summary_table.add_row("[red]Errors[/]", str(error_count))
    summary_table.add_row("[dim]Total items[/]", str(len(items)))
    console.print(summary_table)

    console.print()
    console.print(Panel(
        f"Text files saved to: [cyan]{OUTPUT_DIR}[/]",
        title="Complete",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
