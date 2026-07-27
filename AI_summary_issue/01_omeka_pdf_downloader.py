"""
PDF Download Script for Omeka S Collections

Downloads PDF media from Omeka S items in a given item set, restricted to items
whose resource class is **bibo:Issue** (periodical issues). The class filter is
applied server-side via ``resource_class_id``, with a defensive ``@type`` backstop,
so PDFs are only ever fetched from bibo:Issue items.

The downloading logic lives in ``common/pdf_downloader.py``; this script is a
thin entry point for the magazine-issue pipeline.

Usage:
    python 01_omeka_pdf_downloader.py

Requirements:
    - Environment variables: OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL
    - Valid Omeka S item set ID
"""

import logging
import sys
from pathlib import Path

from rich.console import Console

# Initialize rich console
console = Console()

# Shared Omeka client and PDF downloader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.omeka_client import OmekaClient
from common.pdf_downloader import download_pdfs_from_item_set

# Only download PDFs from items whose resource class is bibo:Issue (verified id).
BIBO_ISSUE_CLASS_ID = 60  # bibo:Issue


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

    # Set up PDF storage directory
    pdf_folder = script_dir / "PDF"

    # Get item set ID from user input
    item_set_id = input("Enter the Omeka S item set ID: ")

    # Initialize shared Omeka client
    try:
        client = OmekaClient.from_env()
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)

    # Start the download process (bibo:Issue items only)
    download_pdfs_from_item_set(
        client,
        item_set_id,
        pdf_folder,
        resource_class_id=BIBO_ISSUE_CLASS_ID,
        required_class_term="bibo:Issue",
        max_workers=2,
        console=console,
    )
