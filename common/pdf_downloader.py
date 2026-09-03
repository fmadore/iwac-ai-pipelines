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
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from rich import box
from rich.console import Console
from rich.table import Table

from common.console_utils import key_value_table, standard_progress
from common.downloader import stream_download
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

        Thin wrapper over :func:`common.downloader.stream_download`, which
        handles the ``.part`` temp file and the truncation check.

        Args:
            url (str): URL of the PDF file to download
            pdf_path (Path): Local path where the PDF should be saved
            timeout (int): Request timeout in seconds

        Returns:
            Optional[Path]: Path to downloaded file or None if download failed
        """
        return stream_download(url, pdf_path, timeout=timeout, logger=LOGGER)

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
            item_data = self.client.get_item(item_id)
            if not item_data:
                LOGGER.error("Could not fetch item %s from Omeka", item_id)
                return None
            pdf_urls = self._find_pdf_urls(item_data)
            if not pdf_urls:
                LOGGER.warning("No PDF URLs found for item %s", item_id)
                return None
            downloaded_files = self._download_pdfs(item_data, pdf_urls)
            if not downloaded_files:
                return None
            return item_data['o:id'], '|'.join(downloaded_files)
        except Exception as e:
            LOGGER.error("Error processing item %s: %s", item_id, e)
            return None

    def _find_pdf_urls(self, item_data: Dict[str, Any]) -> list[str]:
        """Resolve original URLs for PDF media attached to an item."""
        pdf_urls = []
        for media in item_data.get("o:media", []):
            media_data = self.client.get_resource(media["@id"])
            if not media_data or "o:source" not in media_data:
                continue
            source = media_data["o:source"]
            if source.lower().endswith(".pdf"):
                pdf_urls.append(media_data.get("o:original_url", source))
        return pdf_urls

    def _download_pdfs(
        self,
        item_data: Dict[str, Any],
        pdf_urls: list[str],
    ) -> list[str]:
        """Download all discovered PDFs using deterministic item-based names."""
        downloaded_files = []
        for index, pdf_url in enumerate(pdf_urls, start=1):
            pdf_filename = self.create_valid_filename(item_data)
            if len(pdf_urls) > 1:
                pdf_filename = pdf_filename.replace(".pdf", f"_{index}.pdf")
            pdf_path = self.pdf_folder / pdf_filename
            downloaded = self.download_pdf(pdf_url, pdf_path)
            if downloaded:
                downloaded_files.append(str(downloaded))
                LOGGER.info("Downloaded PDF: %s", downloaded)
            else:
                LOGGER.error(
                    "Failed to download %s for item %s", pdf_url, item_data["o:id"],
                )
        return downloaded_files


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

    class_label = None
    if resource_class_id is not None:
        class_label = f"{resource_class_id}" + (f" ({required_class_term})" if required_class_term else "")
    console.print(key_value_table([
        ("Item Set ID", str(item_set_id)),
        ("Output Folder", str(output_dir)),
        ("Max Workers", str(max_workers)),
        ("Omeka URL", client.base_url),
        ("Resource Class Filter", class_label),
    ]))
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

    with standard_progress(console) as progress:
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


# ---------------------------------------------------------------------------
# Entry point shared by the pipelines' ``01_omeka_pdf_downloader.py`` scripts
# ---------------------------------------------------------------------------

def run_cli(
    argv: Optional[Sequence[str]],
    *,
    pipeline_dir: Path,
    description: str,
    resource_class_id: Optional[int] = None,
    required_class_term: Optional[str] = None,
) -> int:
    """Parse ``--item-set-id`` (or ask), then download into ``<pipeline>/PDF/``.

    One implementation for the OCR and magazine pipelines, which differ only in
    whether the items are restricted to a resource class (``bibo:Issue``).
    """
    import argparse

    from rich.console import Console
    from rich.panel import Panel

    from common.log_redaction import install_credential_redaction

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--item-set-id", type=int, help="Omeka S item set to download from (asked when omitted).")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent downloads (default: 2).")
    parser.add_argument(
        "--output-dir", type=Path, default=pipeline_dir / "PDF",
        help="Where the PDFs go (default: <pipeline>/PDF).",
    )
    args = parser.parse_args(argv)

    log_dir = pipeline_dir / "log"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, filename=log_dir / "pdf_download.log", filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # Credentials ride in Omeka query strings; keep them out of anything
    # urllib3 decides to log.
    install_credential_redaction()

    console = Console()
    console.print(Panel(description, title="Omeka S PDF Downloader", border_style="cyan"))
    console.print()

    item_set_id = args.item_set_id
    if item_set_id is None:
        try:
            item_set_id = int(console.input("[cyan]Enter the Omeka S item set ID:[/] ").strip())
        except (ValueError, EOFError, KeyboardInterrupt):
            console.print("[red]✗[/] An item set id is required.")
            return 1
        console.print()

    try:
        client = OmekaClient.from_env()
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        return 1

    stats = download_pdfs_from_item_set(
        client, item_set_id, args.output_dir,
        resource_class_id=resource_class_id, required_class_term=required_class_term,
        max_workers=args.workers, console=console,
    )
    return 0 if stats.get("failed", 0) == 0 else 1
