#!/usr/bin/env python3
"""
Write extracted publication text back to Omeka, with OCR provenance.

Reads ``OCR_Results/<item_id>.txt`` and PATCHes each item's ``bibo:content``,
annotating the value with ``iwac:ocrModel`` → the model that produced it. The
whole item is fetched and sent back, never a trimmed payload: Omeka treats RDF
properties as one block and drops anything absent.

This is the first provenance stamped on this corpus. All 425 ``bibo:content``
values currently on the reference classes carry no ``@annotation`` at all, so
nothing records which tool read them — that gap is exactly what the
``iwac:ocrModel`` annotation closes here.

Unlike its sibling in ``AI_ocr_extraction``, this step defaults to dumping the
pre-write payloads: these are books and theses whose extracted text is expensive
to reproduce, and the backup is the only route back from a bad overwrite.

Usage:
    python 03_omeka_content_updater.py --dry-run          # report, write nothing
    python 03_omeka_content_updater.py                    # prompts, then writes
    python 03_omeka_content_updater.py --item-id 5312     # one item
    python 03_omeka_content_updater.py --model mistral-ocr-4-1 --yes

Requirements:
    - Environment variables: OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL
    - OCR_Results/ populated by 02_mistral_blocks_processor.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.console_utils import key_value_table
from common.iwac_config import (
    AI_MODEL_ITEMS,
    BIBO_CONTENT_PROPERTY_ID,
    IWAC_OCR_MODEL_PROPERTY_ID,
    model_annotation_value,
    select_model_key,
)
from common.log_redaction import install_credential_redaction
from common.write_guard import add_write_guard_args
from common.omeka_client import OmekaClient
from common.omeka_text_updater import (
    PropertyTarget,
    TextUpdate,
    run_text_updates,
    updates_from_directory,
)

console = Console()

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "OCR_Results"
BACKUP_DIR = SCRIPT_DIR / "output"

CONTENT_TERM = "bibo:content"
OCR_MODEL_TERM = "iwac:ocrModel"
DEFAULT_MODEL_KEY = "mistral-ocr-4-1"

install_credential_redaction()


def _sidecar_note(updates: List[TextUpdate]) -> str:
    """One line on what the structural pass found, for the confirmation panel.

    An operator approving a bulk write should see that the apparatus was
    separated as intended before the PATCHes go out, not afterwards in a log.
    """
    apparatus = body = 0
    for update in updates:
        path = RESULTS_DIR / f"{update.item_id}.json"
        if not path.exists():
            continue
        try:
            roles = json.loads(path.read_text(encoding="utf-8")).get("block_roles", {})
        except (json.JSONDecodeError, OSError):
            continue
        apparatus += roles.get("apparatus", 0)
        body += roles.get("body", 0)
    if not (apparatus or body):
        return "Sidecars:         none found"
    return f"Blocks:           {body:,} body, {apparatus:,} apparatus (notes/bibliography)"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update Omeka S reference items with extracted publication text.",
    )
    parser.add_argument(
        "--model", choices=list(AI_MODEL_ITEMS), default=None,
        help=f"OCR model that produced the text (default: prompts, {DEFAULT_MODEL_KEY} preselected).",
    )
    parser.add_argument(
        "--item-id", type=int, action="append", dest="item_ids",
        help="Restrict the write to these item ids. Repeatable.",
    )
    add_write_guard_args(parser, default_backup_dir=BACKUP_DIR)
    args = parser.parse_args()

    console.print(Panel(
        "[bold]Write extracted publication text to Omeka S[/]\n\n"
        "PATCHes bibo:content on reference items and annotates each value\n"
        "with iwac:ocrModel — the first OCR provenance on this corpus.",
        title="📚 Publication Content Updater",
        border_style="cyan",
    ))
    console.print()

    load_dotenv()
    try:
        client = OmekaClient.from_env()
    except ValueError as exc:
        console.print(f"[red]✗[/] {exc}")
        return 1

    model_key = args.model or select_model_key(default=DEFAULT_MODEL_KEY)
    if model_key is None:
        return 1
    ocr_model_value = model_annotation_value(
        client.base_url, model_key, IWAC_OCR_MODEL_PROPERTY_ID, "AI Model - OCR"
    )
    target = PropertyTarget(
        term=CONTENT_TERM,
        property_id=BIBO_CONTENT_PROPERTY_ID,
        property_label="content",
        annotation_term=OCR_MODEL_TERM,
        annotation_value=ocr_model_value,
        # Never public. These are copyrighted monographs, theses and journal
        # articles; only 7 of the 423 reference texts already on the archive are
        # public, and a value created without this flag would default to public
        # and publish a whole book. It is also what the Hugging Face export
        # reads as ``OCR_is_public`` when deciding to mask a row's full text.
        is_public=False,
    )
    console.print()

    if not RESULTS_DIR.exists():
        console.print(f"[red]✗[/] Results folder not found: {RESULTS_DIR}")
        return 1

    # strip=False: OCR text is stored verbatim, leading/trailing layout included.
    updates = updates_from_directory(RESULTS_DIR, strip=False)
    if args.item_ids:
        wanted = set(args.item_ids)
        updates = [u for u in updates if u.item_id in wanted]

    if not updates:
        console.print("[yellow]⚠[/] No matching .txt files in OCR_Results.")
        return 1

    backup_dir = None if args.no_backup else args.backup_dir
    console.print(key_value_table([
        ("Results folder", str(RESULTS_DIR)),
        ("Omeka URL", client.base_url),
        ("OCR model", ocr_model_value["display_title"]),
        ("Documents", len(updates)),
        ("Mode", "DRY RUN — no writes" if args.dry_run else "LIVE update"),
        ("Backup", str(backup_dir) if backup_dir else "disabled"),
    ]))
    console.print()

    stats = run_text_updates(
        client, updates, target,
        console=console,
        dry_run=args.dry_run,
        require_confirmation=not args.yes,
        extra_confirm_lines=[
            f"Source folder:    {RESULTS_DIR}",
            _sidecar_note(updates),
        ],
        description="Writing publication text...",
        backup_dir=backup_dir,
        backup_label="publication_content",
    )
    if not stats:
        return 1  # operator declined

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
