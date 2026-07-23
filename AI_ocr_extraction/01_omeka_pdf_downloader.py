"""
PDF Download Script for Omeka S Collections

This script downloads PDF files from Omeka S digital collections using the API.
It processes item sets, finds PDF media attachments, and downloads them locally
with proper error handling and concurrent processing.

The downloading logic lives in ``common/pdf_downloader.py``; this script is a
thin interactive entry point for the OCR extraction pipeline.

Usage:
    python 01_omeka_pdf_downloader.py

Requirements:
    - Environment variables: OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL
    - Valid Omeka S item set ID
"""

import logging
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# Initialize rich console
console = Console()

# Shared Omeka client and PDF downloader
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.omeka_client import OmekaClient
from common.pdf_downloader import download_pdfs_from_item_set


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
    item_set_id = Prompt.ask("[cyan]Enter the Omeka S item set ID[/]")
    console.print()

    # Initialize shared Omeka client
    try:
        client = OmekaClient.from_env()
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)

    # Start the download process
    download_pdfs_from_item_set(client, item_set_id, pdf_folder, max_workers=2, console=console)
