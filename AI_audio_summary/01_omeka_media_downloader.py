#!/usr/bin/env python3
"""
Audio/Video Media Download Script for Omeka S Collections

This script downloads audio and video files from Omeka S digital collections using the API.
It can process either an entire item set or a single item, finding media attachments
and downloading them locally with proper error handling and concurrent processing.

Usage:
    python 01_omeka_media_downloader.py

Requirements:
    - Environment variables: OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL
    - Valid Omeka S item set ID or item ID

Supported formats:
    Audio: .mp3, .wav, .m4a, .flac, .ogg, .aac, .wma
    Video: .mp4, .mkv, .avi, .mov, .wmv, .flv, .webm, .m4v, .mpeg, .mpg, .3gp
"""

import os
import re
import sys
import logging
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich import box

# Initialize rich console
console = Console()

# Script directory for relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()

# Shared Omeka client
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.omeka_client import OmekaClient

# Supported media formats
AUDIO_EXTENSIONS: Set[str] = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma'}
VIDEO_EXTENSIONS: Set[str] = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpeg', '.mpg', '.3gp'}
SUPPORTED_EXTENSIONS: Set[str] = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


class MediaDownloader:
    """
    Handles audio/video media downloading from Omeka S media attachments.

    Processes items to find audio and video media, downloads files,
    and manages local storage with proper naming conventions.
    """

    def __init__(self, omeka_client: OmekaClient, media_folder: Path):
        """
        Initialize the media downloader.

        Args:
            omeka_client (OmekaClient): Configured Omeka S client
            media_folder (Path): Directory to save downloaded media files
        """
        self.client = omeka_client
        self.media_folder = media_folder

    @staticmethod
    def is_supported_media(filename: str) -> bool:
        """
        Check if a filename has a supported audio or video extension.

        Args:
            filename (str): Name of the file to check

        Returns:
            bool: True if the file extension is supported, False otherwise
        """
        if not filename:
            return False
        extension = Path(filename).suffix.lower()
        return extension in SUPPORTED_EXTENSIONS

    @staticmethod
    def get_media_type(filename: str) -> str:
        """
        Determine if a file is audio or video based on its extension.

        Args:
            filename (str): Name of the file to check

        Returns:
            str: 'audio', 'video', or 'unknown'
        """
        if not filename:
            return 'unknown'
        extension = Path(filename).suffix.lower()
        if extension in AUDIO_EXTENSIONS:
            return 'audio'
        elif extension in VIDEO_EXTENSIONS:
            return 'video'
        return 'unknown'

    @staticmethod
    def download_file(url: str, file_path: Path, timeout: int = 300) -> Optional[Path]:
        """
        Download a media file from a URL to local storage.

        Uses streaming to handle large files efficiently and avoid memory issues.

        Args:
            url (str): URL of the media file to download
            file_path (Path): Local path where the file should be saved
            timeout (int): Request timeout in seconds (default: 300 for large media files)

        Returns:
            Optional[Path]: Path to downloaded file or None if download failed
        """
        try:
            # Stream download to handle large files efficiently
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()

                # Write file in chunks to avoid memory issues
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            return file_path

        except requests.Timeout:
            logging.error(f"Timeout downloading {url}")
            return None
        except requests.RequestException as e:
            logging.error(f"Failed to download {url}: {e}")
            return None

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize a filename by removing or replacing invalid characters.

        Args:
            filename (str): Original filename

        Returns:
            str: Sanitized filename safe for filesystem use
        """
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip(' .')
        # Limit length to avoid filesystem issues
        if len(sanitized) > 200:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:200 - len(ext)] + ext
        return sanitized

    def create_filename(self, item_data: Dict[str, Any], media_data: Dict[str, Any],
                        index: int = 0, total_media: int = 1) -> str:
        """
        Create a filename for a media file using the original source filename from Omeka S.

        Uses the 'o:source' field to preserve the original naming and ordering.
        Falls back to item ID if source unavailable.

        Args:
            item_data (Dict[str, Any]): Omeka S item data
            media_data (Dict[str, Any]): Omeka S media data
            index (int): Index of this media file (0-based) if multiple files per item
            total_media (int): Total number of media files for this item

        Returns:
            str: Valid filename for the media file
        """
        # Use the original source filename to preserve correct ordering
        original_source = media_data.get('o:source', '')

        if original_source:
            return self.sanitize_filename(original_source)

        # Fallback: use item ID and media ID if no source available
        item_id = item_data.get('o:id', 'unknown')
        media_id = media_data.get('o:id', index + 1)

        # Determine file extension
        extension = ''
        original_url = media_data.get('o:original_url', '')
        if original_url:
            extension = Path(original_url).suffix.lower()

        if not extension:
            media_type = media_data.get('o:media_type', '')
            if media_type.startswith('video/'):
                extension = '.mp4'
            elif media_type.startswith('audio/'):
                extension = '.mp3'

        return f"{item_id}-{media_id}{extension}"

    def process_item(self, item: Dict[str, Any]) -> Optional[Tuple[str, List[str]]]:
        """
        Process a single Omeka S item to find and download audio/video media files.

        Args:
            item (Dict[str, Any]): Omeka S item data (basic info with o:id)

        Returns:
            Optional[Tuple[str, List[str]]]: Tuple of (item_id, list of downloaded file paths)
                or None if no media was found or download failed
        """
        item_id = item.get('o:id')

        try:
            # Get detailed item data including media attachments
            item_data = self.client.get_item(item_id)
            media_urls = []

            # Search through media attachments for supported audio/video files
            if 'o:media' in item_data:
                for media in item_data['o:media']:
                    # Get detailed media data
                    media_data = self.client.get_resource(media['@id'])

                    if media_data and 'o:source' in media_data:
                        source = media_data.get('o:source', '')

                        # Check if this media is a supported audio/video file
                        if self.is_supported_media(source):
                            # Prefer original URL if available
                            download_url = media_data.get('o:original_url')
                            if download_url:
                                media_urls.append((download_url, media_data))

            if not media_urls:
                logging.info(f"No audio/video media found for item {item_id}")
                return None

            # Download all media files associated with this item
            downloaded_files = []
            total_media = len(media_urls)

            for index, (media_url, media_data) in enumerate(media_urls):
                # Create descriptive filename
                filename = self.create_filename(item_data, media_data, index, total_media)
                file_path = self.media_folder / filename

                # Skip if file already exists
                if file_path.exists():
                    logging.info(f"File already exists, skipping: {filename}")
                    downloaded_files.append(str(file_path))
                    continue

                # Get media type for logging
                media_type = self.get_media_type(media_data.get('o:source', ''))

                # Attempt to download the media file
                logging.info(f"Downloading {media_type}: {filename}")
                downloaded_path = self.download_file(media_url, file_path)

                if downloaded_path:
                    downloaded_files.append(str(downloaded_path))
                    logging.info(f"Downloaded {media_type}: {downloaded_path}")
                else:
                    logging.error(f"Failed to download: {filename}")

            if downloaded_files:
                return str(item_id), downloaded_files
            return None

        except Exception as e:
            logging.error(f"Error processing item {item_id}: {e}")
            return None


def setup_logging(script_dir: Path) -> None:
    """
    Configure logging for the media download process.

    Args:
        script_dir (Path): Directory where the log file should be created
    """
    # Create log directory if it doesn't exist
    log_dir = script_dir / 'log'
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / 'media_download.log'

    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def download_media_from_item_set(item_set_id: str, media_folder: Path,
                                  max_workers: int = 2) -> List[Tuple[str, List[str]]]:
    """
    Download all audio/video media from a specific Omeka S item set.

    Args:
        item_set_id (str): ID of the Omeka S item set to process
        media_folder (Path): Directory to save downloaded media files
        max_workers (int): Maximum number of concurrent download threads (default: 2)

    Returns:
        List[Tuple[str, List[str]]]: List of (item_id, downloaded_files) tuples
    """
    # Create media directory if it doesn't exist
    media_folder.mkdir(parents=True, exist_ok=True)

    # Initialize shared Omeka client and downloader
    client = OmekaClient.from_env()
    downloader = MediaDownloader(client, media_folder)

    # Retrieve all items from the specified item set
    with console.status(f"[cyan]Fetching items from item set {item_set_id}...[/]"):
        items = client.get_items(int(item_set_id))

    if not items:
        console.print(f"[yellow]No items found for item set {item_set_id}[/]")
        logging.warning(f"No items found for item set {item_set_id}")
        return []

    console.print(f"[green]{chr(10003)}[/] Found [cyan]{len(items)}[/] item(s) to process\n")

    # Process items concurrently with thread pool
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Downloading media...", total=len(items))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            futures = {executor.submit(downloader.process_item, item): item for item in items}

            # Collect results with progress bar
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                progress.update(task, advance=1)

    return results


def download_media_from_single_item(item_id: str, media_folder: Path) -> Optional[Tuple[str, List[str]]]:
    """
    Download all audio/video media from a single Omeka S item.

    Args:
        item_id (str): ID of the Omeka S item to process
        media_folder (Path): Directory to save downloaded media files

    Returns:
        Optional[Tuple[str, List[str]]]: Tuple of (item_id, downloaded_files) or None if failed
    """
    # Create media directory if it doesn't exist
    media_folder.mkdir(parents=True, exist_ok=True)

    # Initialize shared Omeka client and downloader
    client = OmekaClient.from_env()
    downloader = MediaDownloader(client, media_folder)

    console.print(f"\n[cyan]Processing item {item_id}...[/]")

    # Create item dict in the format expected by process_item
    item = {'o:id': item_id}

    with console.status("[cyan]Downloading media...[/]"):
        return downloader.process_item(item)


def get_download_choice() -> Tuple[str, str]:
    """
    Prompt user to choose between downloading from an item set or a single item.

    Returns:
        Tuple[str, str]: Tuple of (choice_type, id) where choice_type is 'item_set' or 'item'
    """
    # Display welcome banner
    console.print(Panel(
        "Download audio and video files from Omeka S digital collections",
        title="Omeka S Audio/Video Media Downloader",
        border_style="cyan"
    ))

    # Display supported formats table
    formats_table = Table(title="Supported Formats", box=box.ROUNDED)
    formats_table.add_column("Type", style="cyan")
    formats_table.add_column("Extensions", style="green")
    formats_table.add_row("Audio", ", ".join(sorted(AUDIO_EXTENSIONS)))
    formats_table.add_row("Video", ", ".join(sorted(VIDEO_EXTENSIONS)))
    console.print(formats_table)

    # Display options
    console.print("\n[bold]Download Options:[/]")
    console.print("  [cyan]1.[/] Download from an item set (multiple items)")
    console.print("  [cyan]2.[/] Download from a single item")
    console.print()

    while True:
        choice = console.input("[bold]Select option (1 or 2):[/] ").strip()
        if choice == '1':
            item_set_id = console.input("[bold]Enter the Omeka S item set ID:[/] ").strip()
            if item_set_id:
                return 'item_set', item_set_id
            console.print("[red]Error: Item set ID cannot be empty.[/]")
        elif choice == '2':
            item_id = console.input("[bold]Enter the Omeka S item ID:[/] ").strip()
            if item_id:
                return 'item', item_id
            console.print("[red]Error: Item ID cannot be empty.[/]")
        else:
            console.print("[red]Invalid choice. Please enter 1 or 2.[/]")


def print_summary(results: List[Tuple[str, List[str]]], media_folder: Path) -> None:
    """
    Print a summary of the download results.

    Args:
        results: List of (item_id, downloaded_files) tuples
        media_folder: Path where media files were saved
    """
    console.print()
    console.rule("[bold]Download Summary", style="cyan")

    if not results:
        console.print("[yellow]No media files were downloaded.[/]")
        return

    total_files = sum(len(files) for _, files in results)
    total_items = len(results)

    # Count by type
    audio_count = 0
    video_count = 0
    for _, files in results:
        for file_path in files:
            ext = Path(file_path).suffix.lower()
            if ext in AUDIO_EXTENSIONS:
                audio_count += 1
            elif ext in VIDEO_EXTENSIONS:
                video_count += 1

    # Create summary table
    summary_table = Table(title="Results", box=box.ROUNDED)
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Items with media", str(total_items))
    summary_table.add_row("Total files downloaded", str(total_files))
    if audio_count:
        summary_table.add_row("Audio files", str(audio_count))
    if video_count:
        summary_table.add_row("Video files", str(video_count))
    summary_table.add_row("Output folder", str(media_folder))
    console.print(summary_table)

    # Create files table
    files_table = Table(title="Downloaded Files", box=box.ROUNDED)
    files_table.add_column("Item ID", style="cyan")
    files_table.add_column("Filename", style="green")
    files_table.add_column("Type", style="dim")

    for item_id, files in results:
        for file_path in files:
            filename = Path(file_path).name
            ext = Path(file_path).suffix.lower()
            media_type = "Audio" if ext in AUDIO_EXTENSIONS else "Video"
            files_table.add_row(str(item_id), filename, media_type)

    console.print(files_table)


def main():
    """
    Main function to run the audio/video media download script.

    Prompts user for download type (item set or single item) and processes accordingly.
    """
    # Initialize logging
    setup_logging(SCRIPT_DIR)

    # Set up media storage directory
    media_folder = SCRIPT_DIR / "Audio"

    try:
        # Get user choice
        choice_type, target_id = get_download_choice()

        # Display configuration
        console.print()
        config_table = Table(title="Configuration", box=box.ROUNDED)
        config_table.add_column("Setting", style="dim")
        config_table.add_column("Value", style="green")
        config_table.add_row("Mode", "Item Set" if choice_type == 'item_set' else "Single Item")
        config_table.add_row("Target ID", target_id)
        config_table.add_row("Output Folder", str(media_folder))
        console.print(config_table)
        console.print()

        # Process based on choice
        if choice_type == 'item_set':
            results = download_media_from_item_set(target_id, media_folder, max_workers=2)
        else:
            result = download_media_from_single_item(target_id, media_folder)
            results = [result] if result else []

        # Print summary
        print_summary(results, media_folder)

        console.print(f"\n[green]{chr(10003)}[/] Download complete. Total items with media: [cyan]{len(results)}[/]")
        logging.info(f"Download complete. Total items with media: {len(results)}")

    except ValueError as e:
        console.print(f"\n[red]Configuration Error:[/] {e}")
        logging.error(f"Configuration error: {e}")
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Download cancelled by user.[/]")
        logging.info("Download cancelled by user")
    except Exception as e:
        console.print(f"\n[red]Unexpected error:[/] {e}")
        logging.exception(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
