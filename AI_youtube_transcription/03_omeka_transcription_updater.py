#!/usr/bin/env python3
"""
Write the YouTube transcriptions back to Omeka S as ``bibo:content``.

Two things differ from ``AI_audio_summary/03_omeka_transcription_updater.py``,
both deliberate:

* **The file's header is stripped, not uploaded.** ``bibo:content`` is the
  archive's full-text field, exported to Hugging Face as ``OCR`` and indexed for
  search. "Generated using: Google gemini-3.7-flash" inside it would be indexed
  as though it were something a speaker said. The header stays on disk, where it
  is auditable, and provenance goes into a value annotation instead.

* **Provenance is recorded.** Each value carries an ``iwac:transcriptionModel``
  annotation naming the model that produced it (property 315, "AI Model -
  Transcription"; this pipeline was the first to write it, and
  ``AI_audio_summary/03`` does too since 2026-08-27). This is why step 02 offers
  only *pinned* model ids: an annotation naming ``gemini-flash-latest`` would
  assert a version the run cannot confirm.

Transcripts are named ``<item_id>.txt``, so no identifier lookup is needed.
Incomplete transcripts — where a window failed and the header records
``Chunks: 2/3`` — are refused by default: a transcript missing its middle third
looks complete once it is a single Omeka value.

The write itself is ``common/omeka_text_updater.py``: the whole item is fetched
and PATCHed back, unchanged items are skipped, and every pre-write payload is
dumped to ``backups/`` first.

Usage:
    python 03_omeka_transcription_updater.py --dry-run
    python 03_omeka_transcription_updater.py --model gemini-3.5-flash-lite
    python 03_omeka_transcription_updater.py --model gemini-3.5-flash-lite --yes
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from rich.console import Console

# Add repo root to path for shared imports, and this pipeline's own directory
# for the sibling format module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.iwac_config import (
    AI_MODEL_ITEMS,
    IWAC_TRANSCRIPTION_MODEL_PROPERTY_ID,
    model_annotation_value,
    select_model_key,
)
from common.log_redaction import install_credential_redaction
from common.omeka_client import OmekaClient
from common.omeka_text_updater import PropertyTarget, TextUpdate, run_text_updates

from youtube_source import HEADER_GENERATOR, looping_reason, read_transcript

install_credential_redaction()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

console = Console()

SCRIPT_DIR = Path(__file__).parent.resolve()
TRANSCRIPTIONS_DIR = SCRIPT_DIR / "Transcriptions"

CONTENT_TERM = "bibo:content"
TRANSCRIPTION_MODEL_TERM = "iwac:transcriptionModel"

#: Step 02's default, offered as this step's default answer so the two agree
#: unless step 02 was run with --model.
DEFAULT_MODEL_KEY = "gemini-3.5-flash-lite"


def collect_updates(
    directory: Path, *, include_incomplete: bool
) -> Tuple[List[TextUpdate], List[Tuple[int, str]]]:
    """Build one update per transcript, returning the ones held back too.

    Files whose stem is not an item id are ignored — the language report written
    beside them is not a transcript, and neither is anything a human dropped in.
    """
    updates: List[TextUpdate] = []
    held_back: List[Tuple[int, str]] = []

    for path in sorted(directory.glob("*.txt")):
        if not path.stem.isdigit():
            continue
        item_id = int(path.stem)
        transcript = read_transcript(path)

        if not transcript.body.strip():
            held_back.append((item_id, "empty transcript"))
            continue
        # Checked here as well as in 02, and not overridable by
        # --include-incomplete: a transcript that loops is not a partial
        # transcript, it is 35,000 words of one clause wearing the right shape.
        # Three of the first 44 videos produced one, each reporting Chunks: 1/1.
        loop = looping_reason(transcript.body)
        if loop:
            held_back.append((item_id, f"degenerate repeating output ({loop})"))
            continue
        if not transcript.complete and not include_incomplete:
            done, total = transcript.chunks_done, transcript.chunks_total
            held_back.append((item_id, f"incomplete ({done}/{total} windows)"))
            continue

        updates.append(TextUpdate(
            label=path.name,
            item_id=item_id,
            text=transcript.body,
            metadata={"generator": transcript.header.get(HEADER_GENERATOR, "")},
        ))
    return updates, held_back


def report_generators(updates: List[TextUpdate], model_key: str) -> None:
    """Warn when the transcripts were not all made by the annotated model.

    The annotation is one value for the whole batch, so a directory holding two
    models' output would attribute both to whichever was chosen — silently.
    """
    seen: Dict[str, int] = {}
    for update in updates:
        generator = update.metadata.get("generator") or "unrecorded"
        seen[generator] = seen.get(generator, 0) + 1
    if len(seen) > 1:
        console.print("[yellow]⚠[/] Transcripts in this folder came from more than one model:")
        for generator, count in sorted(seen.items(), key=lambda pair: -pair[1]):
            console.print(f"    [dim]{count:>4} × {generator}[/]")
        console.print(
            "[yellow]  → One iwac:transcriptionModel annotation is written for the whole "
            "batch, so run them separately or the provenance will be wrong.[/]"
        )
        return

    generator = next(iter(seen), "")
    expected = AI_MODEL_ITEMS[model_key]["display_title"]
    if generator and generator != "unrecorded" and model_key not in generator:
        console.print(
            f"[yellow]⚠[/] Transcripts record [cyan]{generator}[/] but the annotation "
            f"will name [cyan]{expected}[/]."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload YouTube transcriptions to Omeka S (bibo:content) with "
                    "iwac:transcriptionModel provenance.",
    )
    parser.add_argument(
        "--model", choices=list(AI_MODEL_ITEMS),
        help="AI model that produced the transcriptions. Prompts when omitted.",
    )
    parser.add_argument(
        "--transcriptions-dir", type=Path, default=TRANSCRIPTIONS_DIR,
        help=f"Folder of <item_id>.txt transcripts (default: {TRANSCRIPTIONS_DIR.name}/).",
    )
    parser.add_argument(
        "--include-incomplete", action="store_true",
        help="Also upload transcripts whose header records a failed window. "
             "A gap is invisible once the text is a single Omeka value.",
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

    if not args.transcriptions_dir.exists():
        console.print(
            f"[red]✗[/] Transcriptions folder not found: [cyan]{args.transcriptions_dir}[/]\n"
            "[dim]Run 02_AI_transcribe_youtube.py first.[/]"
        )
        return 1

    try:
        client = OmekaClient.from_env()
    except ValueError as exc:
        console.print(f"[red]Configuration Error:[/] {exc}")
        return 1

    # Resolve at runtime: property IDs differ between Omeka S installations, and
    # writing with the wrong one files the transcript under another property.
    content_property_id = client.get_property_id(CONTENT_TERM)
    if content_property_id is None:
        console.print(f"[red]✗[/] Could not resolve property ID for {CONTENT_TERM} — aborting.")
        return 1

    updates, held_back = collect_updates(
        args.transcriptions_dir, include_incomplete=args.include_incomplete
    )
    if held_back:
        console.print(f"[yellow]⚠[/] {len(held_back)} transcript(s) held back:")
        for item_id, reason in held_back[:10]:
            console.print(f"    [dim]{item_id}: {reason}[/]")
        if len(held_back) > 10:
            console.print(f"    [dim]… and {len(held_back) - 10} more[/]")
        # Only offered when it would actually help: a looping transcript is
        # refused whatever the flags say, and suggesting otherwise invites an
        # operator to try the one thing that cannot work.
        if any("incomplete" in reason for _, reason in held_back):
            console.print("[dim]  Pass --include-incomplete to upload the incomplete ones anyway.[/]")
        if any("repeating" in reason for _, reason in held_back):
            console.print(
                "[dim]  The repeating ones cannot be uploaded: re-transcribe them with "
                "a stronger model (--model gemini-3.7-flash) into their own folder.[/]"
            )
    if not updates:
        console.print(f"[yellow]No transcripts to upload from {args.transcriptions_dir}[/]")
        return 0

    model_key = args.model or select_model_key(default=DEFAULT_MODEL_KEY)
    if model_key is None:
        return 1
    report_generators(updates, model_key)

    model = AI_MODEL_ITEMS[model_key]
    target = PropertyTarget(
        term=CONTENT_TERM,
        property_id=content_property_id,
        property_label="content",
        annotation_term=TRANSCRIPTION_MODEL_TERM,
        annotation_value=model_annotation_value(
            client.base_url,
            model_key,
            IWAC_TRANSCRIPTION_MODEL_PROPERTY_ID,
            "AI Model - Transcription",
        ),
    )
    logging.info(
        "Annotating with %s -> %s (item %s)",
        TRANSCRIPTION_MODEL_TERM, model["display_title"], model["item_id"],
    )

    backup_dir = None if args.no_backup else (args.backup_dir or SCRIPT_DIR / "backups")

    confirm_lines = [f"Source folder:    {args.transcriptions_dir}"]
    if held_back:
        confirm_lines.append(f"Held back:        {len(held_back)} incomplete")

    stats = run_text_updates(
        client, updates, target,
        console=console,
        dry_run=args.dry_run,
        require_confirmation=not args.yes,
        extra_confirm_lines=confirm_lines,
        description="Updating transcriptions...",
        backup_dir=backup_dir,
        backup_label="youtube_transcriptions",
    )
    if not stats:
        return 1  # operator declined

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
