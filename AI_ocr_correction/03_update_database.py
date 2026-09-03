"""
This script updates Omeka S items with corrected OCR text content.

It reads corrected text files and updates the corresponding items in the Omeka S
database while preserving existing metadata and handling the bibo:content
property appropriately.

No ``iwac:ocrModel`` annotation is written here: correction rewrites text that
another model originally OCR'd, and ``upsert_property_value`` mutates the
existing literal in place, so the original OCR provenance is preserved rather
than overwritten with the correction model's name.

The write step lives in ``common/omeka_text_updater.py``, shared with the
summary, OCR-extraction and transcription updaters.

Usage:
    python 03_update_database.py --dry-run   # preview first
    python 03_update_database.py             # pre-write payloads go to backups/
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

# Shared Omeka client
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.iwac_config import BIBO_CONTENT_PROPERTY_ID
from common.omeka_client import OmekaClient
from common.omeka_text_updater import PropertyTarget, run_text_updates, updates_from_directory
from common.log_redaction import install_credential_redaction
from common.write_guard import WriteGuard, add_write_guard_args

# Credentials ride in Omeka query strings and provider headers; keep them
# out of anything urllib3 or an SDK decides to log.
install_credential_redaction()

PIPELINE_DIR = Path(__file__).resolve().parent
# Directory containing the corrected text files (relative to script location)
DEFAULT_TXT_DIRECTORY = PIPELINE_DIR / 'Corrected_TXT'
# Correction overwrites the whole bibo:content literal, so the pre-write dump
# is the only copy of the text a run replaces.
BACKUP_DIR = PIPELINE_DIR / 'backups'

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload corrected OCR text to Omeka S (bibo:content)")
    parser.add_argument(
        "--txt-dir",
        type=Path,
        default=DEFAULT_TXT_DIRECTORY,
        help="Directory of corrected .txt files named <item_id>.txt",
    )
    add_write_guard_args(parser, default_backup_dir=BACKUP_DIR)
    args = parser.parse_args()
    guard = WriteGuard.from_args(args, default_backup_dir=BACKUP_DIR)
    backup_dir = guard.backup_dir if guard.backup_enabled else None

    console.print(Panel.fit("[bold]OCR Correction — Omeka S Database Update[/]", border_style="cyan"))

    if not args.txt_dir.is_dir():
        console.print(f"[red]✗[/] Corrected-text directory not found: {args.txt_dir}")
        return 1

    try:
        client = OmekaClient.from_env()
    except ValueError as exc:
        console.print(f"[red]✗[/] {exc}")
        return 1

    updates = updates_from_directory(args.txt_dir)
    if not updates:
        console.print(f"[yellow]⚠[/] No .txt files found in {args.txt_dir}")
        return 1

    target = PropertyTarget(
        term='bibo:content',
        property_id=BIBO_CONTENT_PROPERTY_ID,
        property_label='content',
    )

    stats = run_text_updates(
        client, updates, target,
        console=console,
        dry_run=guard.dry_run,
        require_confirmation=not guard.assume_yes,
        extra_confirm_lines=[f"Source folder:    {args.txt_dir}"],
        description="Updating corrected text...",
        backup_dir=backup_dir,
        backup_label="corrected_content",
    )
    if not stats:
        return 1  # operator declined

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
