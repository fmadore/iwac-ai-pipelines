"""
Database Update Script for OCR Results

This script processes OCR text files and updates corresponding Omeka S items
with the extracted text content. It preserves all existing metadata while
adding or updating the bibo:content property with the OCR results.

Usage:
    python 03_update_database.py

Requirements:
    - Environment variables: OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL
    - OCR_Results directory with .txt files named after item IDs
"""

import os
import requests

# Load environment variables from .env if present
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich import box

# Initialize rich console
console = Console()


# Load API credentials from environment variables (now loaded from .env if present)
base_url = os.environ.get('OMEKA_BASE_URL')
key_identity = os.environ.get('OMEKA_KEY_IDENTITY')
key_credential = os.environ.get('OMEKA_KEY_CREDENTIAL')

def get_full_item_data(item_id):
    """
    Retrieve complete item data from Omeka S API.
    
    Args:
        item_id (str): The ID of the item to retrieve
        
    Returns:
        dict or None: Complete item data including all properties, or None if failed
    """
    url = f"{base_url}/items/{item_id}"
    params = {
        'key_identity': key_identity,
        'key_credential': key_credential
    }
    headers = {'Content-Type': 'application/json'}
    
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        return response.json()
    return None

def update_item_with_new_content(item_id, new_content):
    """
    Update an Omeka S item with new bibo:content while preserving all other metadata.
    
    This function retrieves the current item data, updates or adds the bibo:content
    property with the new OCR text, and sends the updated data back to the API.
    
    Args:
        item_id (str): The ID of the item to update
        new_content (str): The new OCR text content to add/update
        
    Returns:
        bool: True if update succeeded, False otherwise
    """
    # Retrieve current item data to preserve existing properties
    item_data = get_full_item_data(item_id)
    if not item_data:
        console.print(f"  [yellow]⚠[/] No data found for item {item_id}. Skipping.")
        return False

    # Ensure the 'value' key exists for storing property values
    if 'value' not in item_data:
        item_data['value'] = []

    # Search for existing bibo:content property and update it
    bibo_content_found = False
    for value in item_data['value']:
        # Property ID 91 corresponds to bibo:content (this may vary by installation)
        if value.get('property_id') == 91:
            value['@value'] = new_content  # Update existing content
            bibo_content_found = True
            break

    # If bibo:content doesn't exist, create a new property entry
    if not bibo_content_found:
        item_data['value'].append({
            "type": "literal",
            "property_id": 91,  # bibo:content property ID
            "property_label": "content",
            "is_public": True,
            "@value": new_content
        })

    # Send updated item data back to the API
    url = f"{base_url}/items/{item_id}"
    params = {
        'key_identity': key_identity,
        'key_credential': key_credential
    }
    headers = {'Content-Type': 'application/json'}
    
    response = requests.patch(url, json=item_data, params=params, headers=headers)
    if response.status_code == 200:
        return True
    else:
        console.print(f"  [red]✗[/] Failed to update item {item_id}: {response.text}")
        return False

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
        "Omeka S items with the bibo:content property.",
        title="📤 Omeka S Content Updater",
        border_style="cyan"
    ))
    console.print()
    
    # Verify that all required environment variables are set
    if not all([base_url, key_identity, key_credential]):
        console.print("[red]✗[/] Error: Missing required environment variables.")
        console.print("  Please set OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, and OMEKA_KEY_CREDENTIAL.")
        return

    # Set up directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ocr_folder = os.path.join(script_dir, "OCR_Results")

    # Display configuration
    config_table = Table(title="⚙️ Configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("OCR Results Folder", ocr_folder)
    config_table.add_row("Omeka URL", base_url)
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
                    
                # Update the Omeka S item with the OCR content
                result = update_item_with_new_content(item_id, content)
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
