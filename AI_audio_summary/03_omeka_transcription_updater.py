#!/usr/bin/env python3
"""
Omeka S Transcription Updater

This script processes transcription text files and updates corresponding Omeka S items
with the transcribed content. It matches files to items using the dcterms:identifier
property and can join multiple transcription segments into a single content field.

File naming convention:
    - Single file: {identifier}_transcription.txt
    - Multiple segments: {identifier}-{segment}_transcription.txt
    
The script will automatically detect and join segments in numerical order.

Usage:
    python 03_omeka_transcription_updater.py

Requirements:
    - Environment variables: OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL
    - Transcriptions directory with .txt files following the naming convention
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich import box

# Initialize rich console
console = Console()

# Script directory for relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()

# Property IDs (may vary by Omeka S installation)
BIBO_CONTENT_PROPERTY_ID = 91  # bibo:content
DCTERMS_IDENTIFIER_PROPERTY_ID = 10  # dcterms:identifier


@dataclass
class OmekaConfig:
    """Configuration container for Omeka S API credentials."""
    base_url: str
    key_identity: str
    key_credential: str


class OmekaClient:
    """Client for interacting with Omeka S API."""
    
    def __init__(self, config: OmekaConfig):
        """Initialize the Omeka S client with configuration."""
        self.config = config
        self.headers = {'Content-Type': 'application/json'}
        
        # Normalize base URL
        base = config.base_url.rstrip('/')
        if base.endswith('/api'):
            base = base[:-4]
        self.base_url = f"{base}/api"
        
    def _get_auth_params(self) -> Dict[str, str]:
        """Get authentication parameters for API requests."""
        return {
            'key_identity': self.config.key_identity,
            'key_credential': self.config.key_credential
        }
    
    def search_item_by_identifier(self, identifier: str) -> Optional[Dict]:
        """
        Search for an Omeka S item by its dcterms:identifier value.
        
        Args:
            identifier: The identifier value to search for
            
        Returns:
            Item data dict or None if not found
        """
        url = f"{self.base_url}/items"
        params = {
            **self._get_auth_params(),
            'property[0][property]': DCTERMS_IDENTIFIER_PROPERTY_ID,
            'property[0][type]': 'eq',
            'property[0][text]': identifier,
            'per_page': 1
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            items = response.json()
            
            if items:
                return items[0]
            return None
            
        except requests.RequestException as e:
            logging.error(f"Error searching for item with identifier '{identifier}': {e}")
            return None
    
    def get_item(self, item_id: int) -> Optional[Dict]:
        """
        Retrieve complete item data from Omeka S API.
        
        Args:
            item_id: The ID of the item to retrieve
            
        Returns:
            Complete item data or None if failed
        """
        url = f"{self.base_url}/items/{item_id}"
        
        try:
            response = requests.get(url, params=self._get_auth_params(), headers=self.headers)
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            logging.error(f"Error retrieving item {item_id}: {e}")
            return None
    
    def update_item_content(self, item_id: int, content: str) -> bool:
        """
        Update an Omeka S item with new bibo:content while preserving all other metadata.
        
        Args:
            item_id: The ID of the item to update
            content: The transcription content to add/update
            
        Returns:
            True if update succeeded, False otherwise
        """
        # Retrieve current item data
        item_data = self.get_item(item_id)
        if not item_data:
            logging.error(f"Could not retrieve item {item_id} for update")
            return False
        
        # Check if bibo:content already exists and update it
        bibo_content_key = 'bibo:content'
        
        if bibo_content_key in item_data:
            # Update existing bibo:content
            # Find and update the literal value
            updated = False
            for i, value in enumerate(item_data[bibo_content_key]):
                if value.get('type') == 'literal':
                    item_data[bibo_content_key][i]['@value'] = content
                    updated = True
                    break
            
            if not updated:
                # Add new literal value
                item_data[bibo_content_key].append({
                    "type": "literal",
                    "property_id": BIBO_CONTENT_PROPERTY_ID,
                    "property_label": "content",
                    "is_public": True,
                    "@value": content
                })
        else:
            # Create new bibo:content property
            item_data[bibo_content_key] = [{
                "type": "literal",
                "property_id": BIBO_CONTENT_PROPERTY_ID,
                "property_label": "content",
                "is_public": True,
                "@value": content
            }]
        
        # Send updated item data back to API
        url = f"{self.base_url}/items/{item_id}"
        
        try:
            response = requests.patch(
                url, 
                json=item_data, 
                params=self._get_auth_params(), 
                headers=self.headers
            )
            response.raise_for_status()
            logging.info(f"Successfully updated item {item_id} with transcription")
            return True
            
        except requests.RequestException as e:
            logging.error(f"Failed to update item {item_id}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logging.error(f"Response: {e.response.text}")
            return False


class TranscriptionProcessor:
    """Processes transcription files and matches them to Omeka items."""
    
    def __init__(self, transcriptions_folder: Path):
        """
        Initialize the processor.
        
        Args:
            transcriptions_folder: Path to the folder containing transcription files
        """
        self.transcriptions_folder = transcriptions_folder
        
        # Regex patterns for filename parsing
        # Pattern: {identifier}-{segment}_transcription.txt or {identifier}_transcription.txt
        self.segment_pattern = re.compile(r'^(.+?)-(\d+)_transcription\.txt$')
        self.single_pattern = re.compile(r'^(.+?)_transcription\.txt$')
    
    def parse_filename(self, filename: str) -> Tuple[Optional[str], Optional[int]]:
        """
        Parse a transcription filename to extract identifier and segment number.
        
        Args:
            filename: The transcription filename
            
        Returns:
            Tuple of (identifier, segment_number) where segment_number is None for single files
        """
        # Try segmented pattern first
        match = self.segment_pattern.match(filename)
        if match:
            return match.group(1), int(match.group(2))
        
        # Try single file pattern
        match = self.single_pattern.match(filename)
        if match:
            return match.group(1), None
        
        return None, None
    
    def get_transcription_groups(self) -> Dict[str, List[Tuple[Path, Optional[int]]]]:
        """
        Group transcription files by their base identifier.
        
        Returns:
            Dictionary mapping identifiers to lists of (file_path, segment_number) tuples
        """
        groups = defaultdict(list)
        
        if not self.transcriptions_folder.exists():
            logging.warning(f"Transcriptions folder not found: {self.transcriptions_folder}")
            return groups
        
        for file_path in self.transcriptions_folder.iterdir():
            if not file_path.suffix == '.txt':
                continue
            
            identifier, segment = self.parse_filename(file_path.name)
            if identifier:
                groups[identifier].append((file_path, segment))
            else:
                logging.warning(f"Could not parse filename: {file_path.name}")
        
        # Sort each group by segment number
        for identifier in groups:
            groups[identifier].sort(key=lambda x: x[1] if x[1] is not None else 0)
        
        return groups
    
    def read_and_join_transcriptions(self, files: List[Tuple[Path, Optional[int]]]) -> str:
        """
        Read and join multiple transcription files.
        
        Args:
            files: List of (file_path, segment_number) tuples, sorted by segment
            
        Returns:
            Combined transcription content
        """
        contents = []
        
        for file_path, segment in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                if content:
                    if len(files) > 1 and segment is not None:
                        # Add separator for multiple segments
                        contents.append(f"\n{'='*60}\n[Part {segment}]\n{'='*60}\n\n{content}")
                    else:
                        contents.append(content)
                        
            except Exception as e:
                logging.error(f"Error reading file {file_path}: {e}")
        
        return '\n'.join(contents).strip()


def setup_logging(log_folder: Path) -> None:
    """Configure logging with file and console handlers."""
    log_folder.mkdir(exist_ok=True)
    log_file = log_folder / 'transcription_update.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def load_config() -> OmekaConfig:
    """Load configuration from environment variables."""
    load_dotenv()
    
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


def print_summary(results: Dict[str, str]) -> None:
    """Print a summary of the update results."""
    console.print()
    console.rule("[bold]Update Summary", style="cyan")
    
    success_count = sum(1 for status in results.values() if status == 'success')
    not_found_count = sum(1 for status in results.values() if status == 'not_found')
    failed_count = sum(1 for status in results.values() if status == 'failed')
    
    # Create summary table
    summary_table = Table(title="📊 Results", box=box.ROUNDED)
    summary_table.add_column("Status", style="dim")
    summary_table.add_column("Count", justify="right")
    summary_table.add_row("Total identifiers processed", str(len(results)))
    summary_table.add_row("[green]Successfully updated[/]", f"[green]{success_count}[/]")
    summary_table.add_row("[yellow]Item not found[/]", f"[yellow]{not_found_count}[/]")
    summary_table.add_row("[red]Failed to update[/]", f"[red]{failed_count}[/]")
    console.print(summary_table)
    
    if not_found_count > 0:
        console.print()
        not_found_table = Table(title="[yellow]⚠ Identifiers Not Found in Omeka[/]", box=box.ROUNDED)
        not_found_table.add_column("Identifier", style="yellow")
        for identifier, status in results.items():
            if status == 'not_found':
                not_found_table.add_row(identifier)
        console.print(not_found_table)
    
    if failed_count > 0:
        console.print()
        failed_table = Table(title="[red]✗ Failed Updates[/]", box=box.ROUNDED)
        failed_table.add_column("Identifier", style="red")
        for identifier, status in results.items():
            if status == 'failed':
                failed_table.add_row(identifier)
        console.print(failed_table)


def main():
    """Main function to process transcriptions and update Omeka S items."""
    # Setup
    setup_logging(SCRIPT_DIR / 'log')
    transcriptions_folder = SCRIPT_DIR / 'Transcriptions'
    
    # Display welcome banner
    console.print(Panel(
        "Process transcription files and update Omeka S items with transcribed content",
        title="📝 Omeka S Transcription Updater",
        border_style="cyan"
    ))
    
    try:
        # Load configuration
        config = load_config()
        client = OmekaClient(config)
        processor = TranscriptionProcessor(transcriptions_folder)
        
        # Get grouped transcription files
        with console.status("[cyan]Scanning transcription files...[/]"):
            groups = processor.get_transcription_groups()
        
        if not groups:
            console.print(f"\n[yellow]⚠[/] No transcription files found in: [cyan]{transcriptions_folder}[/]")
            return
        
        # Display files table
        files_table = Table(title="📁 Identifiers to Process", box=box.ROUNDED)
        files_table.add_column("Identifier", style="cyan")
        files_table.add_column("Segments", justify="right", style="green")
        
        for identifier, files in groups.items():
            file_count = len(files)
            segment_info = f"{file_count} segment(s)" if file_count > 1 else "1 file"
            files_table.add_row(identifier, segment_info)
        
        console.print(files_table)
        console.print(f"\n[bold]Total:[/] [cyan]{len(groups)}[/] unique identifier(s)")
        
        # Confirm before proceeding
        console.print("\n[dim]This will update the bibo:content property for matching Omeka S items.[/]")
        confirm = console.input("[bold]Continue? (y/n):[/] ").strip().lower()
        if confirm != 'y':
            console.print("[yellow]⚠[/] Operation cancelled.")
            return
        
        console.print()
        console.rule("[bold]Processing Transcriptions", style="cyan")
        console.print()
        
        results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Updating items...", total=len(groups))
            
            for identifier, files in groups.items():
                logging.info(f"Processing identifier: {identifier}")
                
                # Search for the item in Omeka
                item = client.search_item_by_identifier(identifier)
                
                if not item:
                    logging.warning(f"No item found with identifier: {identifier}")
                    results[identifier] = 'not_found'
                    progress.update(task, advance=1)
                    continue
                
                item_id = item.get('o:id')
                logging.info(f"Found item {item_id} for identifier {identifier}")
                
                # Read and join transcription content
                content = processor.read_and_join_transcriptions(files)
                
                if not content:
                    logging.warning(f"No content found for identifier: {identifier}")
                    results[identifier] = 'failed'
                    progress.update(task, advance=1)
                    continue
                
                # Update the item
                if client.update_item_content(item_id, content):
                    results[identifier] = 'success'
                else:
                    results[identifier] = 'failed'
                
                progress.update(task, advance=1)
        
        # Print summary
        print_summary(results)
        
        success_count = sum(1 for status in results.values() if status == 'success')
        console.print(f"\n[green]✓[/] Transcription update process completed. [cyan]{success_count}[/] item(s) updated.")
        logging.info("Transcription update process completed")
        
    except ValueError as e:
        console.print(f"\n[red]✗ Configuration Error:[/] {e}")
        logging.error(f"Configuration error: {e}")
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠[/] Operation cancelled by user.")
        logging.info("Operation cancelled by user")
    except Exception as e:
        console.print(f"\n[red]✗ Unexpected error:[/] {e}")
        logging.exception(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
