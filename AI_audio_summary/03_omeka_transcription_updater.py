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

Only the *body* of each file is uploaded. Every transcription file opens with the
metadata header ``segments.write_transcription()`` writes — ``Transcription of:``,
``Generated using:``, Voxtral's ``Language:``/``Diarization:`` lines and the
``=`` * 50 separator — and ``bibo:content`` is the archive's full-text field,
exported to Hugging Face as ``OCR`` and indexed for search, so a header left
inside it is indexed as though a speaker had said it. ``segments.read_body()``
splits it off, per file rather than once: a recording that arrived as several
media files is several transcription files, each with its own header, and this
step joins them under one identifier.

The header is not lost by being stripped — it is where the
``iwac:transcriptionModel`` annotation below is read from, and it stays on disk.

Each value written carries an ``iwac:transcriptionModel`` annotation naming the
model that produced it, so a transcript's provenance survives outside the file
header on disk. ``--model`` is deliberately optional: three of the four models
that can fill ``Transcriptions/`` have no Omeka authority item to point at —
``voxtral-mini-2602`` has none yet, and ``gemini-pro-latest`` /
``gemini-flash-lite-latest`` are rolling aliases which deliberately have none,
because a run through one cannot state which release answered it. Requiring
``--model`` would make this step unusable for all three. Silence does not mean
"no provenance" either, though: with neither flag the model is read off the
transcripts' own ``Generated using:`` header, and the run stops when that header
names something no annotation can cite.

Usage:
    python 03_omeka_transcription_updater.py --dry-run
    python 03_omeka_transcription_updater.py --model gemini-3.7-flash
    python 03_omeka_transcription_updater.py --no-model-annotation --yes

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
from collections import Counter, defaultdict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Initialize rich console
console = Console()

# Script directory for relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()

# Shared Omeka client, then this pipeline's own directory for the sibling
# format module. The latter is implicit only while this file is the entry point;
# importing it any other way — a test — would fail on `segments`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.iwac_config import (
    AI_MODEL_ITEMS,
    BIBO_CONTENT_PROPERTY_ID,
    DCTERMS_IDENTIFIER_PROPERTY_ID,
    IWAC_TRANSCRIPTION_MODEL_PROPERTY_ID,
    model_annotation_value,
    select_model_key,
)
from common.omeka_client import OmekaClient
from common.omeka_text_updater import PropertyTarget, TextUpdate, run_text_updates
from common.log_redaction import install_credential_redaction

from segments import GENERATOR_FIELD, read_body, read_header

CONTENT_TERM = 'bibo:content'
TRANSCRIPTION_MODEL_TERM = 'iwac:transcriptionModel'

#: Stands in for a transcription file whose header records no generator — one
#: written before the field existed, or hand-edited.
UNRECORDED_GENERATOR = 'unrecorded'


def content_target(annotation_value: Optional[Dict] = None) -> PropertyTarget:
    """The ``bibo:content`` target, carrying provenance when one was asserted."""
    return PropertyTarget(
        term=CONTENT_TERM,
        property_id=BIBO_CONTENT_PROPERTY_ID,
        property_label='content',
        annotation_term=TRANSCRIPTION_MODEL_TERM if annotation_value else None,
        annotation_value=annotation_value,
    )


def annotation_key_for(generator: str) -> Optional[str]:
    """The ``AI_MODEL_ITEMS`` key a ``Generated using:`` header names, if any.

    Headers are written as ``"<vendor> <model id>"`` — ``"Google
    gemini-3.7-flash"``, ``"Mistral voxtral-mini-2602"`` — and for every model
    that has an authority item the id *is* the annotation key. Returns ``None``
    for the ones that do not, which is most of what this pipeline can produce.
    """
    model_id = generator.split(' ')[-1].strip() if generator else ''
    return model_id if model_id in AI_MODEL_ITEMS else None


def count_generators(
    groups: Dict[str, List[Tuple[Path, Optional[int]]]],
) -> "Counter[str]":
    """Tally the ``Generated using:`` header across every transcription file.

    Every file is read, not one per identifier: a recording that arrived as
    several media files is several transcription files, and nothing stops 02
    having produced one of them and 02b another.
    """
    counts: "Counter[str]" = Counter()
    for files in groups.values():
        for file_path, _ in files:
            generator = read_header(file_path).get(GENERATOR_FIELD, '').strip()
            counts[generator or UNRECORDED_GENERATOR] += 1
    return counts


def resolve_model_key(
    counts: "Counter[str]",
    *,
    requested: Optional[str],
    skip: bool,
    assume_yes: bool,
) -> Tuple[Optional[str], bool]:
    """Decide which model this batch's ``iwac:transcriptionModel`` names.

    Returns ``(model_key, ok)``. A *model_key* of ``None`` with *ok* true is a
    run that writes content and no provenance — all this step can do for a model
    with no authority item. *ok* false means the operator has to say something
    more explicit before anything is written.
    """
    if skip:
        console.print(
            f"[dim]Writing bibo:content with no {TRANSCRIPTION_MODEL_TERM} "
            "annotation (--no-model-annotation).[/]"
        )
        return None, True

    if len(counts) > 1:
        # Stricter than AI_youtube_transcription/03, which only warns. One
        # annotation is written for the whole batch, so a mixed folder
        # attributes every transcript to whichever model is chosen — and --yes
        # skips the confirmation panel the warning would have been read on. This
        # folder accumulates across 02 and 02b runs, so mixing is not exotic.
        console.print("[red]✗[/] Transcripts in this folder came from more than one model:")
        for generator, count in counts.most_common():
            console.print(f"    [dim]{count:>4} × {generator}[/]")
        console.print(
            "[red]  One annotation is written for the whole batch, so uploading them "
            "together would attribute all of them to one model.[/]"
        )
        console.print(
            "[dim]  Move each model's transcripts into their own folder and run this step "
            "once per folder, or pass --no-model-annotation to write no provenance.[/]"
        )
        return None, False

    generator = next(iter(counts))
    inferred = annotation_key_for(generator)

    if requested:
        expected = AI_MODEL_ITEMS[requested]['display_title']
        if inferred and inferred != requested:
            console.print(
                f"[yellow]⚠[/] Transcripts record [cyan]{generator}[/] but the annotation "
                f"will name [cyan]{expected}[/]."
            )
        elif inferred is None:
            # The expected shape for a rolling alias, and the reason a mismatch
            # is a warning rather than a refusal: only the operator can say
            # which release answered "gemini-pro-latest" on the day it ran.
            console.print(
                f"[dim]Transcripts record {generator}, which no annotation can name; "
                f"asserting {expected} as asked.[/]"
            )
        return requested, True

    if inferred is None:
        console.print(
            f"[red]✗[/] Transcripts record [cyan]{generator}[/], which has no entry in "
            "AI_MODEL_ITEMS, so no annotation can name it."
        )
        console.print(
            "[dim]  Pass --no-model-annotation to upload the text without provenance, or "
            "--model <key> to assert the pinned release behind a rolling alias.[/]"
        )
        return None, False

    if assume_yes:
        # Not a guess: the transcriber wrote this header itself, which is better
        # evidence than an unattended run has any other way of getting.
        console.print(
            f"[green]✓[/] Annotating as [cyan]{AI_MODEL_ITEMS[inferred]['display_title']}[/], "
            f"read from the transcripts' header ([dim]{generator}[/])."
        )
        return inferred, True

    console.print(f"[dim]Transcripts record {generator}.[/]")
    try:
        chosen = select_model_key(default=inferred)
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]No answer on stdin — aborted, nothing written.[/]")
        return None, False
    return chosen, chosen is not None


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
        Read and join multiple transcription files, header stripped.

        Only the body of each file is kept (see the module docstring), and every
        file in the group is split, not just the first — one header per segment
        file, so stripping once would leave the rest inline.

        Args:
            files: List of (file_path, segment_number) tuples, sorted by segment

        Returns:
            Combined transcription content
        """
        contents = []

        for file_path, segment in files:
            try:
                content = read_body(file_path)

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
        "--model", choices=list(AI_MODEL_ITEMS),
        help="AI model that produced the transcriptions. Read from the "
             "transcripts' header when omitted; pass it to assert the pinned "
             "release behind a rolling alias such as gemini-pro-latest.",
    )
    parser.add_argument(
        "--no-model-annotation", action="store_true",
        help="Upload the text with no iwac:transcriptionModel annotation. The "
             "only option for a model with no Omeka authority item — Voxtral today.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch each item and report what would change, but write nothing.",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the interactive confirmation before writing.",
    )
    parser.add_argument(
        "--backup-dir", type=Path, default=None,
        help="Where each item's pre-write JSON is dumped before its PATCH "
             "(default: <pipeline>/backups). The only route back from a bulk overwrite.",
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="Do not dump pre-write payloads. Not recommended.",
    )
    args = parser.parse_args()

    if args.model and args.no_model_annotation:
        parser.error("--model and --no-model-annotation contradict each other.")

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

        # Settled before the identifier lookups: a folder that cannot be
        # attributed should stop here, not after a few hundred searches against
        # a live archive.
        model_key, resolved = resolve_model_key(
            count_generators(groups),
            requested=args.model,
            skip=args.no_model_annotation,
            assume_yes=args.yes,
        )
        if not resolved:
            return 1

        annotation = None
        if model_key:
            model = AI_MODEL_ITEMS[model_key]
            annotation = model_annotation_value(
                client.base_url,
                model_key,
                IWAC_TRANSCRIPTION_MODEL_PROPERTY_ID,
                'AI Model - Transcription',
            )
            logging.info(
                "Annotating with %s -> %s (item %s)",
                TRANSCRIPTION_MODEL_TERM, model['display_title'], model['item_id'],
            )

        updates = resolve_updates(client, processor, groups)
        unresolved = sum(1 for u in updates if u.item_id is None)
        if unresolved:
            console.print(f"[yellow]⚠[/] {unresolved} identifier(s) had no matching Omeka item")

        backup_dir = None if args.no_backup else (args.backup_dir or SCRIPT_DIR / 'backups')

        stats = run_text_updates(
            client, updates, content_target(annotation),
            console=console,
            dry_run=args.dry_run,
            require_confirmation=not args.yes,
            extra_confirm_lines=[f"Source folder:    {transcriptions_folder}"],
            description="Updating transcriptions...",
            backup_dir=backup_dir,
            backup_label="audio_transcriptions",
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
