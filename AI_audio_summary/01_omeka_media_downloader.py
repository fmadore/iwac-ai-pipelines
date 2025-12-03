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
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Script directory for relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()

# Supported media formats
AUDIO_EXTENSIONS: Set[str] = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma'}
VIDEO_EXTENSIONS: Set[str] = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpeg', '.mpg', '.3gp'}
SUPPORTED_EXTENSIONS: Set[str] = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


@dataclass
class OmekaConfig:
    """
    Configuration container for Omeka S API credentials and settings.
    
    Attributes:
        base_url (str): Base URL of the Omeka S instance
        key_identity (str): API key identity for authentication
        key_credential (str): API key credential for authentication
    """
    base_url: str
    key_identity: str
    key_credential: str


class OmekaClient:
    """
    Client for interacting with Omeka S API.
    
    Handles API authentication, pagination, and data retrieval from Omeka S collections.
    """
    
    def __init__(self, config: OmekaConfig):
        """
        Initialize the Omeka S client with configuration.
        
        Args:
            config (OmekaConfig): Configuration object with API credentials
        """
        self.config = config
        self.headers = {'Content-Type': 'application/json'}
        
        # Normalize base URL to ensure consistent API endpoint construction
        base = config.base_url.rstrip('/')
        if base.endswith('/api'):
            base = base[:-4]  # Remove existing /api suffix
        self.base_url = f"{base}/api"
        
    def _get_auth_params(self) -> Dict[str, str]:
        """
        Get authentication parameters for API requests.
        
        Returns:
            Dict[str, str]: Dictionary containing authentication parameters
        """
        return {
            'key_identity': self.config.key_identity,
            'key_credential': self.config.key_credential
        }
        
    def get_items(self, item_set_id: str, per_page: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve all items from a specific Omeka S item set.
        
        Handles pagination automatically to fetch all items in the collection.
        
        Args:
            item_set_id (str): ID of the Omeka S item set to retrieve
            per_page (int): Number of items to fetch per API request (default: 100)
            
        Returns:
            List[Dict[str, Any]]: List of all items in the item set
        """
        url = f"{self.base_url}/items"
        params = {
            **self._get_auth_params(),
            'item_set_id': item_set_id,
            'per_page': per_page,
            'page': 1
        }
        
        all_items = []
        
        # Paginate through all items in the collection
        while True:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            page_items = response.json()
            if not page_items:
                break  # No more items to fetch
                
            all_items.extend(page_items)
            logging.info(f"Retrieved {len(page_items)} items from page {params['page']}")
            
            # Check if we've reached the last page
            if len(page_items) < params['per_page']:
                break
                
            params['page'] += 1  # Move to next page
            
        return all_items
    
    def get_item_data(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve detailed data for a specific Omeka S item.
        
        Args:
            item_id (str): ID of the item to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Item data or None if request fails
            
        Raises:
            requests.HTTPError: If the API request fails
        """
        url = f"{self.base_url}/items/{item_id}"
        response = requests.get(url, params=self._get_auth_params(), headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_media_data(self, media_url: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve media data from a media URL.
        
        Args:
            media_url (str): URL of the media resource
            
        Returns:
            Optional[Dict[str, Any]]: Media data or None if request fails
            
        Raises:
            requests.HTTPError: If the API request fails
        """
        response = requests.get(media_url, params=self._get_auth_params(), headers=self.headers)
        response.raise_for_status()
        return response.json()


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
                
                # Get file size if available for progress tracking
                total_size = int(response.headers.get('content-length', 0))
                
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
        
        Uses the 'o:source' field (e.g., 'iwac-video-0000001-1.mp4') to preserve
        the original naming and ordering. Falls back to item ID if source unavailable.
        
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
            item_data = self.client.get_item_data(item_id)
            media_urls = []
            
            # Search through media attachments for supported audio/video files
            if 'o:media' in item_data:
                for media in item_data['o:media']:
                    # Get detailed media data
                    media_data = self.client.get_media_data(media['@id'])
                    
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
    
    Sets up both file and console logging handlers.
    
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


def load_config() -> OmekaConfig:
    """
    Load configuration from environment variables.
    
    Returns:
        OmekaConfig: Configuration object with API credentials
        
    Raises:
        ValueError: If required environment variables are missing
    """
    load_dotenv()  # Load .env file
    
    base_url = os.getenv('OMEKA_BASE_URL', '')
    key_identity = os.getenv('OMEKA_KEY_IDENTITY', '')
    key_credential = os.getenv('OMEKA_KEY_CREDENTIAL', '')
    
    if not all([base_url, key_identity, key_credential]):
        raise ValueError(
            "Missing required environment variables. Please set:\n"
            "  - OMEKA_BASE_URL\n"
            "  - OMEKA_KEY_IDENTITY\n"
            "  - OMEKA_KEY_CREDENTIAL"
        )
    
    return OmekaConfig(
        base_url=base_url,
        key_identity=key_identity,
        key_credential=key_credential
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
    
    # Initialize Omeka client and downloader
    config = load_config()
    client = OmekaClient(config)
    downloader = MediaDownloader(client, media_folder)
    
    # Retrieve all items from the specified item set
    print(f"\nFetching items from item set {item_set_id}...")
    items = client.get_items(item_set_id)
    
    if not items:
        print(f"No items found for item set {item_set_id}")
        logging.warning(f"No items found for item set {item_set_id}")
        return []
    
    print(f"Found {len(items)} item(s) to process\n")
        
    # Process items concurrently with thread pool
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        futures = {executor.submit(downloader.process_item, item): item for item in items}
        
        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading media"):
            result = future.result()
            if result:
                results.append(result)
                
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
    
    # Initialize Omeka client and downloader
    config = load_config()
    client = OmekaClient(config)
    downloader = MediaDownloader(client, media_folder)
    
    print(f"\nProcessing item {item_id}...")
    
    # Create item dict in the format expected by process_item
    item = {'o:id': item_id}
    
    return downloader.process_item(item)


def get_download_choice() -> Tuple[str, str]:
    """
    Prompt user to choose between downloading from an item set or a single item.
    
    Returns:
        Tuple[str, str]: Tuple of (choice_type, id) where choice_type is 'item_set' or 'item'
    """
    print("\n" + "=" * 60)
    print("OMEKA S AUDIO/VIDEO MEDIA DOWNLOADER")
    print("=" * 60)
    print("\nSupported formats:")
    print(f"  Audio: {', '.join(sorted(AUDIO_EXTENSIONS))}")
    print(f"  Video: {', '.join(sorted(VIDEO_EXTENSIONS))}")
    print("\nDownload options:")
    print("  1. Download from an item set (multiple items)")
    print("  2. Download from a single item")
    print()
    
    while True:
        choice = input("Select option (1 or 2): ").strip()
        if choice == '1':
            item_set_id = input("Enter the Omeka S item set ID: ").strip()
            if item_set_id:
                return 'item_set', item_set_id
            print("Error: Item set ID cannot be empty.")
        elif choice == '2':
            item_id = input("Enter the Omeka S item ID: ").strip()
            if item_id:
                return 'item', item_id
            print("Error: Item ID cannot be empty.")
        else:
            print("Invalid choice. Please enter 1 or 2.")


def print_summary(results: List[Tuple[str, List[str]]], media_folder: Path) -> None:
    """
    Print a summary of the download results.
    
    Args:
        results: List of (item_id, downloaded_files) tuples
        media_folder: Path where media files were saved
    """
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    
    if not results:
        print("No media files were downloaded.")
        return
    
    total_files = sum(len(files) for _, files in results)
    total_items = len(results)
    
    print(f"Items with media: {total_items}")
    print(f"Total files downloaded: {total_files}")
    print(f"Output folder: {media_folder}")
    
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
    
    if audio_count:
        print(f"  - Audio files: {audio_count}")
    if video_count:
        print(f"  - Video files: {video_count}")
    
    print("\nDownloaded files:")
    for item_id, files in results:
        print(f"\n  Item {item_id}:")
        for file_path in files:
            print(f"    - {Path(file_path).name}")


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
        
        # Process based on choice
        if choice_type == 'item_set':
            results = download_media_from_item_set(target_id, media_folder, max_workers=2)
        else:
            result = download_media_from_single_item(target_id, media_folder)
            results = [result] if result else []
        
        # Print summary
        print_summary(results, media_folder)
        
        logging.info(f"Download complete. Total items with media: {len(results)}")
        
    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        logging.error(f"Configuration error: {e}")
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        logging.info("Download cancelled by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        logging.exception(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
