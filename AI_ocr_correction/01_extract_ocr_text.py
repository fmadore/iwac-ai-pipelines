"""
This script retrieves OCR text from items in an Omeka S database and saves them as individual text files.
It handles pagination, concurrent processing, and includes error handling and logging functionality.
"""

import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich import box

# Load environment variables
load_dotenv()

# Initialize Rich console
console = Console()

# Configuration constants
BASE_URL = os.getenv('OMEKA_BASE_URL')  # Base URL for the Omeka S API
KEY_IDENTITY = os.getenv('OMEKA_KEY_IDENTITY')  # API key identity
KEY_CREDENTIAL = os.getenv('OMEKA_KEY_CREDENTIAL')  # API key credential

# Output directory is set relative to the script's location for portability
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TXT")
ITEMS_PER_PAGE = 100  # Number of items to fetch per API request
MAX_WORKERS = 5  # Maximum number of concurrent threads for processing

# Configure logging to track script execution and errors
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


class ItemFetcher:
    """
    A class to handle fetching and processing items from the Omeka S API.
    
    This class manages API connections, item retrieval, and content extraction
    with built-in retry mechanisms and concurrent processing capabilities.
    """

    def __init__(self, base_url, output_dir):
        """
        Initialize the ItemFetcher with API endpoint and output location.
        
        Args:
            base_url (str): The base URL of the Omeka S API
            output_dir (str): Directory where extracted text files will be saved
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.session = self.create_session()
        # Add authentication parameters
        self.auth_params = {
            'key_identity': KEY_IDENTITY,
            'key_credential': KEY_CREDENTIAL
        }

    @staticmethod
    def create_session():
        """
        Create a requests session with retry mechanism for robust API communication.
        
        Returns:
            requests.Session: Configured session object with retry capabilities
        """
        session = requests.Session()
        # Configure automatic retries for failed requests
        retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    def fetch_items(self, item_set_id):
        """
        Fetch all items from a specific item set using pagination.
        
        Args:
            item_set_id (int): ID of the Omeka S item set to fetch
            
        Returns:
            list: List of items retrieved from the API
        """
        endpoint = f"{self.base_url}/items"
        params = {
            **self.auth_params,  # Include authentication parameters
            "item_set_id": item_set_id,
            "per_page": ITEMS_PER_PAGE,
            "page": 1
        }
        items = []

        with console.status("[cyan]Fetching items from Omeka S...", spinner="dots") as status:
            while True:
                try:
                    response = self.session.get(endpoint, params=params)
                    response.raise_for_status()
                    page_items = response.json()
                    if not page_items:
                        break
                    items.extend(page_items)
                    status.update(f"[cyan]Fetching items... Page {params['page']} ({len(items)} items)")
                    params["page"] += 1
                except requests.RequestException as e:
                    console.print(f"[red]✗[/] Error fetching page {params['page']}: {e}")
                    break

        return items

    def extract_and_save_content(self, item):
        """
        Extract OCR text from an item and save it to a file.
        
        Args:
            item (dict): Item data containing OCR text
            
        Returns:
            tuple: (item_id, success_status, skipped)
        """
        item_id = item["o:id"]
        file_name = os.path.join(self.output_dir, f"{item_id}.txt")

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

    def process_items(self, items):
        """
        Process multiple items concurrently using a thread pool.
        
        Args:
            items (list): List of items to process
            
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
                future_to_item = {executor.submit(self.extract_and_save_content, item): item for item in items}
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
    
    Prompts for item set ID, initializes the fetcher, and processes all items
    while providing progress feedback through rich console output.
    """
    # Welcome banner
    console.print(Panel(
        "Extract OCR text from Omeka S items and save as individual text files.",
        title="📄 OCR Text Extractor",
        border_style="cyan"
    ))
    
    # Get item set ID from user
    console.print()
    item_set_id = int(console.input("[cyan]Enter the item set ID:[/] "))
    console.print()

    # Create the OUTPUT_DIR in the same folder as the script
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Display configuration
    config_table = Table(title="⚙️ Configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Item Set ID", str(item_set_id))
    config_table.add_row("Output Directory", OUTPUT_DIR)
    config_table.add_row("Items per Page", str(ITEMS_PER_PAGE))
    config_table.add_row("Max Workers", str(MAX_WORKERS))
    console.print(config_table)
    console.print()

    fetcher = ItemFetcher(BASE_URL, OUTPUT_DIR)

    # Fetch items
    console.rule("[bold cyan]Fetching Items")
    items = fetcher.fetch_items(item_set_id)
    console.print(f"[green]✓[/] Fetched [bold]{len(items)}[/] items from Omeka S")
    console.print()

    if not items:
        console.print("[yellow]⚠[/] No items found in this item set.")
        return

    # Process items
    console.rule("[bold cyan]Extracting Content")
    success_count, skipped_count, error_count = fetcher.process_items(items)
    console.print()

    # Summary
    console.rule("[bold cyan]Summary")
    summary_table = Table(box=box.ROUNDED)
    summary_table.add_column("Status", style="bold")
    summary_table.add_column("Count", justify="right")
    summary_table.add_row("[green]✓[/] Files saved", str(success_count))
    summary_table.add_row("[yellow]○[/] Skipped (no text)", str(skipped_count))
    summary_table.add_row("[red]✗[/] Errors", str(error_count))
    summary_table.add_row("[dim]Total items[/]", str(len(items)))
    console.print(summary_table)
    
    console.print()
    console.print(Panel(
        f"Text files saved to: [cyan]{OUTPUT_DIR}[/]",
        title="✨ Complete",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
