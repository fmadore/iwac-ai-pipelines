"""
Shared PDF downloader for Omeka S pipelines.

Extracted from the near-identical ``01_omeka_pdf_downloader.py`` scripts in
``AI_ocr_extraction/`` and ``AI_summary_issue/``. Provides:

- :class:`PDFDownloader` — finds PDF media on an item and downloads them
  (streaming, with a ``.part`` temp file so truncated downloads are never
  mistaken for complete ones on re-run).
- :func:`download_pdfs_from_item_set` — the full item-set loop with
  concurrent downloads, rich progress/output, and failure counting.

Usage:
    from common.omeka_client import OmekaClient
    from common.pdf_downloader import download_pdfs_from_item_set

    client = OmekaClient.from_env()
    stats = download_pdfs_from_item_set(client, item_set_id, output_dir)
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import requests
from rich import box
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from common.omeka_client import OmekaClient

LOGGER = logging.getLogger(__name__)


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
    def download_pdf(url: str, pdf_path: Path, timeout: int = 30) -> Optional[Path]:
        """
        Download a PDF file from a URL to local storage.

        Streams to a ``.part`` temp file and renames on success, so an
        interrupted download can never be mistaken for a completed file on
        the next run.

        Args:
            url (str): URL of the PDF file to download
            pdf_path (Path): Local path where the PDF should be saved
            timeout (int): Request timeout in seconds

        Returns:
            Optional[Path]: Path to downloaded file or None if download failed
        """
        part_path = pdf_path.with_suffix(pdf_path.suffix + '.part')
        try:
            # Stream download to handle large files efficiently
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()

                # Write file in chunks to avoid memory issues
                with open(part_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                expected = response.headers.get('Content-Length')
                if expected is not None and part_path.stat().st_size != int(expected):
                    raise requests.RequestException(
                        f"Incomplete download: got {part_path.stat().st_size} of {expected} bytes"
                    )

            part_path.rename(pdf_path)
            return pdf_path

        except (requests.RequestException, OSError) as e:
            LOGGER.error("Failed to download %s: %s", url, e)
            part_path.unlink(missing_ok=True)
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
            Optional[Tuple[str, str]]: Tuple of (item_id, downloaded_files_paths)
                or None if nothing was downloaded / the item failed
        """
        item_id = item.get('o:id')
        try:
            # Get detailed item data including media attachments
            item_data = self.client.get_item(item_id)
            if not item_data:
                LOGGER.error("Could not fetch item %s from Omeka", item_id)
                return None

            pdf_urls = []

            # Search through media attachments for PDF files
            for media in item_data.get('o:media', []):
                # Get detailed media data
                media_data = self.client.get_resource(media['@id'])

                if media_data and 'o:source' in media_data:
                    source = media_data['o:source']

                    # Check if this media is a PDF file
                    if source.lower().endswith('.pdf'):
                        # Prefer original URL if available, fallback to source
                        pdf_urls.append(media_data.get('o:original_url', source))

            if not pdf_urls:
                LOGGER.warning("No PDF URLs found for item %s", item_id)
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
                    LOGGER.info("Downloaded PDF: %s", downloaded_pdf_path)
                else:
                    LOGGER.error("Failed to download %s for item %s", pdf_url, item_id)

            if not downloaded_files:
                # Every download for this item failed — report it as a failure.
                return None

            return item_data['o:id'], '|'.join(downloaded_files)

        except Exception as e:
            LOGGER.error("Error processing item %s: %s", item_id, e)
            return None


def download_pdfs_from_item_set(
    client: OmekaClient,
    item_set_id: Union[str, int],
    output_dir: Path,
    *,
    resource_class_id: Optional[int] = None,
    required_class_term: Optional[str] = None,
    max_workers: int = 2,
    console: Optional[Console] = None,
) -> Dict[str, int]:
    """
    Download all PDFs from a specific Omeka S item set.

    Args:
        client: Configured Omeka S client
        item_set_id: ID of the Omeka S item set to process
        output_dir: Directory to save downloaded PDFs
        resource_class_id: Optional server-side resource class filter
            (e.g. bibo:Issue) passed to the items query
        required_class_term: Optional ``@type`` term (e.g. ``"bibo:Issue"``)
            used as a defensive client-side backstop on top of
            *resource_class_id*
        max_workers: Maximum number of concurrent download threads
        console: Optional rich console (a new one is created if omitted)

    Returns:
        Dict with ``downloaded``, ``failed``, and ``total_items`` counts.
    """
    console = console or Console()
    stats = {"downloaded": 0, "failed": 0, "total_items": 0}

    # Create PDF directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    downloader = PDFDownloader(client, output_dir)

    # Display configuration
    config_table = Table(title="Configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Item Set ID", str(item_set_id))
    config_table.add_row("Output Folder", str(output_dir))
    config_table.add_row("Max Workers", str(max_workers))
    config_table.add_row("Omeka URL", client.base_url)
    if resource_class_id is not None:
        class_label = f"{resource_class_id}"
        if required_class_term:
            class_label += f" ({required_class_term})"
        config_table.add_row("Resource Class Filter", class_label)
    console.print(config_table)
    console.print()

    # Retrieve all items from the specified item set
    extra_params: Dict[str, Any] = {}
    if resource_class_id is not None:
        extra_params["resource_class_id"] = resource_class_id

    with console.status("[cyan]Fetching items from Omeka S...", spinner="dots"):
        items = client.get_items(int(item_set_id), **extra_params)

    if not items:
        console.print(f"[red]No items found for item set {item_set_id}[/]")
        LOGGER.error("No items found for item set %s", item_set_id)
        return stats

    # Defensive backstop: keep only items whose @type confirms the class.
    if required_class_term:
        all_count = len(items)
        items = [item for item in items if required_class_term in item.get("@type", [])]
        dropped = all_count - len(items)
        if dropped:
            console.print(
                f"[yellow]⚠[/] Skipped [cyan]{dropped}[/] item(s) returned "
                f"without a {required_class_term} @type"
            )
            LOGGER.warning("Skipped %d item(s) returned without a %s @type", dropped, required_class_term)
        if not items:
            console.print(f"[red]No {required_class_term} items found for item set {item_set_id}[/]")
            LOGGER.error("No %s items found in item set %s", required_class_term, item_set_id)
            return stats

    stats["total_items"] = len(items)
    item_label = f"{required_class_term} items" if required_class_term else "items"
    console.print(f"[green]{chr(10003)}[/] Found [cyan]{len(items)}[/] {item_label} in item set")
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

    stats["downloaded"] = len(results)
    stats["failed"] = failed_count

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

    LOGGER.info("Total PDFs downloaded: %d (failed/skipped: %d)", len(results), failed_count)
    console.print(f"\n[green]{chr(10003)}[/] Download complete! PDFs saved to: [cyan]{output_dir}[/]")

    return stats
