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
    python 03_omeka_transcription_updater.py --dry-run

Requirements:
    - Environment variables: OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL
    - Transcriptions directory with .txt files following the naming convention
"""

import argparse
import re
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Initialize rich console
console = Console()

# Script directory for relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()

# Shared Omeka client
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.iwac_config import BIBO_CONTENT_PROPERTY_ID, DCTERMS_IDENTIFIER_PROPERTY_ID
from common.omeka_client import OmekaClient
from common.omeka_text_updater import PropertyTarget, TextUpdate, run_text_updates
from common.log_redaction import install_credential_redaction

CONTENT_TARGET = PropertyTarget(
    term='bibo:content',
    property_id=BIBO_CONTENT_PROPERTY_ID,
    property_label='content',
)


def search_item_by_identifier(client: OmekaClient, identifier: str) -> Optional[Dict]:
    """
    Search for an Omeka S item by its dcterms:identifier value.

    Uses the shared client's session for retry-capable HTTP requests.

    Args:
        client: OmekaClient instance
        identifier: The identifier value to search for

    Returns:
        Item data dict or None if not found
    """
    items = client.search_items_by_property(DCTERMS_IDENTIFIER_PROPERTY_ID, identifier, per_page=1)
    return items[0] if items else None


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
    # Credentials ride in Omeka query strings and provider headers; keep them
    # out of anything urllib3 or an SDK decides to log.
    install_credential_redaction()


def resolve_updates(
    client: OmekaClient,
    processor: "TranscriptionProcessor",
    groups: Dict[str, List[Tuple[Path, Optional[int]]]],
) -> List[TextUpdate]:
    """Resolve each identifier to an item and join its transcription segments.

    Unlike the other text updaters, files here are named after a
    dcterms:identifier rather than an item ID, so the lookup happens up front.
    Unresolved identifiers keep ``item_id=None`` and are reported as
    ``not_found`` by the shared runner.
    """
    updates: List[TextUpdate] = []
    with console.status("[cyan]Matching identifiers to Omeka items...[/]"):
        for identifier, files in groups.items():
            item = search_item_by_identifier(client, identifier)
            if not item:
                logging.warning(f"No item found with identifier: {identifier}")
                updates.append(TextUpdate(label=identifier, item_id=None, text=""))
                continue

            item_id = item.get('o:id')
            logging.info(f"Found item {item_id} for identifier {identifier}")
            updates.append(TextUpdate(
                label=identifier,
                item_id=int(item_id),
                text=processor.read_and_join_transcriptions(files),
            ))
    return updates


def main() -> int:
    """Main function to process transcriptions and update Omeka S items."""
    parser = argparse.ArgumentParser(
        description="Update Omeka S items with audio transcriptions (bibo:content)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch each item and report what would change, but write nothing.",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the interactive confirmation before writing.",
    )
    args = parser.parse_args()

    setup_logging(SCRIPT_DIR / 'log')
    transcriptions_folder = SCRIPT_DIR / 'Transcriptions'

    console.print(Panel(
        "Process transcription files and update Omeka S items with transcribed content",
        title="Omeka S Transcription Updater",
        border_style="cyan"
    ))

    try:
        client = OmekaClient.from_env()
        processor = TranscriptionProcessor(transcriptions_folder)

        with console.status("[cyan]Scanning transcription files...[/]"):
            groups = processor.get_transcription_groups()

        if not groups:
            console.print(f"\n[yellow]No transcription files found in: [cyan]{transcriptions_folder}[/][/]")
            return 0

        files_table = Table(title="Identifiers to Process", box=box.ROUNDED)
        files_table.add_column("Identifier", style="cyan")
        files_table.add_column("Segments", justify="right", style="green")
        for identifier, files in groups.items():
            file_count = len(files)
            files_table.add_row(identifier, f"{file_count} segment(s)" if file_count > 1 else "1 file")
        console.print(files_table)
        console.print(f"\n[bold]Total:[/] [cyan]{len(groups)}[/] unique identifier(s)")

        updates = resolve_updates(client, processor, groups)
        unresolved = sum(1 for u in updates if u.item_id is None)
        if unresolved:
            console.print(f"[yellow]⚠[/] {unresolved} identifier(s) had no matching Omeka item")

        stats = run_text_updates(
            client, updates, CONTENT_TARGET,
            console=console,
            dry_run=args.dry_run,
            require_confirmation=not args.yes,
            extra_confirm_lines=[f"Source folder:    {transcriptions_folder}"],
            description="Updating transcriptions...",
        )
        if not stats:
            return 1  # operator declined

        logging.info("Transcription update process completed")
        return 0 if stats["failed"] == 0 else 1

    except ValueError as e:
        console.print(f"\n[red]Configuration Error:[/] {e}")
        logging.error(f"Configuration error: {e}")
        return 1
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Operation cancelled by user.[/]")
        logging.info("Operation cancelled by user")
        return 1
    except Exception as e:
        console.print(f"\n[red]Unexpected error:[/] {e}")
        logging.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
