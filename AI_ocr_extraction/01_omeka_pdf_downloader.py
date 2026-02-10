"""
PDF Download Script for Omeka S Collections

This script downloads PDF files from Omeka S digital collections using the API.
It processes item sets, finds PDF media attachments, and downloads them locally
with proper error handling and concurrent processing.

Usage:
    python 01_PDF_download.py

Requirements:
    - Environment variables: OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL
    - Valid Omeka S item set ID
"""

import os
import sys
import requests
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple, Any

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


class PDFDownloader:
    """
    Handles PDF downloading from Omeka S media attachments.

    Processes items to find PDF media, downloads files, and manages local storage.
    """

    def __init__(self, omeka_client: OmekaClient, pdf_folder: Path):
        """
        Initialize the PDF downloader.

        Args:
            omeka_client (OmekaClient): Configured Omeka S client
            pdf_folder (Path): Directory to save downloaded PDFs
        """
        self.client = omeka_client
        self.pdf_folder = pdf_folder

    @staticmethod
    def download_pdf(url: str, pdf_path: Path) -> Optional[Path]:
        """
        Download a PDF file from a URL to local storage.

        Args:
            url (str): URL of the PDF file to download
            pdf_path (Path): Local path where the PDF should be saved

        Returns:
            Optional[Path]: Path to downloaded file or None if download failed
        """
        try:
            # Stream download to handle large files efficiently
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()

                # Write file in chunks to avoid memory issues
                with open(pdf_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            return pdf_path
        except requests.RequestException as e:
            logging.error(f"Failed to download {url}: {e}")
            return None

    @staticmethod
    def create_valid_filename(item_data: Dict[str, Any]) -> str:
        """
        Create a valid filename for a PDF based on item data.

        Args:
            item_data (Dict[str, Any]): Omeka S item data

        Returns:
            str: Valid filename for the PDF
        """
        return f"{item_data['o:id']}.pdf"

    def process_item(self, item: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """
        Process a single Omeka S item to find and download PDF files.

        Args:
            item (Dict[str, Any]): Omeka S item data

        Returns:
            Optional[Tuple[str, str]]: Tuple of (item_id, downloaded_files_paths) or None if failed
        """
        try:
            # Get detailed item data including media attachments
            item_data = self.client.get_item(item['o:id'])
            pdf_urls = []

            # Search through media attachments for PDF files
            if 'o:media' in item_data:
                for media in item_data['o:media']:
                    # Get detailed media data
                    media_data = self.client.get_resource(media['@id'])

                    if media_data and 'o:source' in media_data:
                        source = media_data['o:source']

                        # Check if this media is a PDF file
                        if source.lower().endswith('.pdf'):
                            # Prefer original URL if available, fallback to source
                            pdf_urls.append(media_data.get('o:original_url', source))

            if not pdf_urls:
                logging.warning(f"No PDF URLs found for item {item['o:id']}")
                return None

            # Download all PDF files associated with this item
            downloaded_files = []
            for index, pdf_url in enumerate(pdf_urls):
                pdf_filename = self.create_valid_filename(item_data)

                # Handle multiple PDFs per item by adding index suffix
                if len(pdf_urls) > 1:
                    pdf_filename = pdf_filename.replace('.pdf', f'_{index + 1}.pdf')

                pdf_path = self.pdf_folder / pdf_filename

                # Attempt to download the PDF
                downloaded_pdf_path = self.download_pdf(pdf_url, pdf_path)
                if downloaded_pdf_path:
                    downloaded_files.append(str(downloaded_pdf_path))
                    logging.info(f"Downloaded PDF: {downloaded_pdf_path}")

            return item_data['o:id'], '|'.join(downloaded_files)

        except Exception as e:
            logging.error(f"Error processing item {item['o:id']}: {e}")
            return None

def setup_logging(script_dir: Path) -> None:
    """
    Configure logging for the PDF download process.

    Args:
        script_dir (Path): Directory where the log file should be created
    """
    # Create log directory if it doesn't exist
    log_dir = script_dir / 'log'
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / 'pdf_download.log'
    logging.basicConfig(
        level=logging.INFO,
        filename=log_file,
        filemode='a',  # Append to existing log file
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def download_pdfs_from_item_set(item_set_id: str, pdf_folder: Path, max_workers: int = 2) -> None:
    """
    Download all PDFs from a specific Omeka S item set.

    Args:
        item_set_id (str): ID of the Omeka S item set to process
        pdf_folder (Path): Directory to save downloaded PDFs
        max_workers (int): Maximum number of concurrent download threads (default: 2)
    """
    # Create PDF directory if it doesn't exist
    pdf_folder.mkdir(parents=True, exist_ok=True)

    # Initialize shared Omeka client and downloader
    try:
        client = OmekaClient.from_env()
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        return
    downloader = PDFDownloader(client, pdf_folder)

    # Display configuration
    config_table = Table(title="Configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Item Set ID", item_set_id)
    config_table.add_row("Output Folder", str(pdf_folder))
    config_table.add_row("Max Workers", str(max_workers))
    config_table.add_row("Omeka URL", client.base_url)
    console.print(config_table)
    console.print()

    # Retrieve all items from the specified item set
    with console.status("[cyan]Fetching items from Omeka S...", spinner="dots"):
        items = client.get_items(int(item_set_id))

    if not items:
        console.print("[red]No items found for item set[/]", item_set_id)
        logging.error(f"No items found for item set {item_set_id}")
        return

    console.print(f"[green]{chr(10003)}[/] Found [cyan]{len(items)}[/] items in item set")
    console.print()
    console.rule("[bold blue]Downloading PDFs")
    console.print()

    # Process items concurrently with thread pool
    results = []
    failed_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Downloading PDFs...", total=len(items))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            futures = {executor.submit(downloader.process_item, item): item for item in items}

            # Collect results
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                else:
                    failed_count += 1
                progress.update(task, advance=1)

    # Display summary
    console.print()
    console.rule("[bold blue]Summary")
    console.print()

    summary_table = Table(box=box.ROUNDED)
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Count", justify="right")
    summary_table.add_row("[green]PDFs Downloaded[/]", f"[green]{len(results)}[/]")
    summary_table.add_row("[red]Failed/Skipped[/]", f"[red]{failed_count}[/]")
    summary_table.add_row("Total Items", str(len(items)))
    console.print(summary_table)

    logging.info(f"Total PDFs downloaded: {len(results)}")
    console.print(f"\n[green]{chr(10003)}[/] Download complete! PDFs saved to: [cyan]{pdf_folder}[/]")

if __name__ == "__main__":
    # Initialize script directory and logging
    script_dir = Path(__file__).parent
    setup_logging(script_dir)

    # Display welcome banner
    console.print(Panel(
        "[bold]Download PDF files from Omeka S digital collections[/]\n\n"
        "This script retrieves all PDF media attachments from an item set "
        "and saves them locally for OCR processing.",
        title="Omeka S PDF Downloader",
        border_style="cyan"
    ))
    console.print()

    # Set up PDF storage directory
    pdf_folder = script_dir / "PDF"

    # Get item set ID from user input
    from rich.prompt import Prompt
    item_set_id = Prompt.ask("[cyan]Enter the Omeka S item set ID[/]")
    console.print()

    # Start the download process
    download_pdfs_from_item_set(item_set_id, pdf_folder, max_workers=2)
