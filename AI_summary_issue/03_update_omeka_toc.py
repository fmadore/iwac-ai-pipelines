"""
Update Omeka S items with the AI-generated table of contents via the REST API.

Scans Magazine_Extractions/ for *_final_index.json files produced by
02_AI_generate_summaries_issue.py, prompts for the annotation model, and writes
`dcterms:tableOfContents` to each item — annotated with the model that produced it
(`iwac:summaryModel` value annotation).

SAFETY — preserves existing metadata:
    Uses the shared Omeka text updater, which GETs the full representation,
    modifies ONLY dcterms:tableOfContents in place, and skips an unchanged
    value. Every other property is therefore sent back unchanged.

Usage:
    python 03_update_omeka_toc.py --dry-run  # fetch + report only, writes nothing
    python 03_update_omeka_toc.py            # prompts for the model, then updates live
    python 03_update_omeka_toc.py --model deepseek-v4-flash-0731 --yes

Writes are gated like every other write step: --dry-run, a pre-write payload
dump to backups/ (the only route back), and a confirmation before the first PATCH.

Requirements:
    OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL in .env
"""

import argparse
import json
import os
import sys
from pathlib import Path

from rich.console import Console

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.omeka_client import OmekaClient  # noqa: E402
from common.iwac_config import (  # noqa: E402
    AI_MODEL_ITEMS,
    DCTERMS_TABLE_OF_CONTENTS_PROPERTY_ID,
    IWAC_SUMMARY_MODEL_PROPERTY_ID,
    model_annotation_value,
    select_model_key,
)
from common.llm_provider import DEFAULT_TEXT_MODEL_KEY  # noqa: E402
from common.omeka_text_updater import (  # noqa: E402
    PropertyTarget,
    TextUpdate,
    run_text_updates,
)
from common.log_redaction import install_credential_redaction
from common.write_guard import WriteGuard, add_write_guard_args  # noqa: E402

# Credentials ride in Omeka query strings and provider headers; keep them
# out of anything urllib3 or an SDK decides to log.
install_credential_redaction()

console = Console()
BACKUP_DIR = Path(SCRIPT_DIR) / "backups"


def format_article_toc(article: dict) -> str:
    """Format a single article from a final_index.json into a TOC line."""
    title = article["titre"]
    pages = article["pages"]
    authors = article.get("auteurs")
    resume = article.get("resume", "")
    header = (
        f"p. {pages} : {title} ({', '.join(authors)})" if authors else f"p. {pages} : {title}"
    )
    return f"{header}\n{resume}"


def load_from_extractions(extractions_dir: Path) -> list:
    """Scan Magazine_Extractions/ for *_final_index.json files."""
    toc_entries = []
    if not extractions_dir.exists():
        return toc_entries
    for item_dir in sorted(extractions_dir.iterdir()):
        if not item_dir.is_dir():
            continue
        try:
            item_id = int(item_dir.name)
        except ValueError:
            continue
        index_files = list(item_dir.glob("*_final_index.json"))
        if not index_files:
            continue
        with open(index_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        articles = data.get("articles", [])
        if not articles:
            continue
        toc_text = "\n\n".join(format_article_toc(a) for a in articles)
        toc_entries.append({"item_id": item_id, "table_of_contents": toc_text})
    return toc_entries


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Update Omeka S items with the AI table of contents (preserves existing metadata)."
    )
    parser.add_argument(
        "--model", choices=list(AI_MODEL_ITEMS),
        help=(
            "AI model that consolidated the index. Prompts interactively when "
            f"omitted, with {DEFAULT_TEXT_MODEL_KEY} preselected."
        ),
    )
    add_write_guard_args(parser, default_backup_dir=BACKUP_DIR)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    guard = WriteGuard.from_args(args, default_backup_dir=BACKUP_DIR)
    backup_dir = guard.backup_dir if guard.backup_enabled else None

    extractions_dir = Path(SCRIPT_DIR) / "Magazine_Extractions"
    toc_entries = load_from_extractions(extractions_dir)
    if not toc_entries:
        console.print("[yellow]No entries found in Magazine_Extractions/.[/]")
        return 0

    try:
        client = OmekaClient.from_env()
    except ValueError as e:
        console.print(f"[red]✗[/] {e}")
        return 1

    # Asked rather than defaulted: an unattended run must not stamp whatever
    # the registry default happens to be on upload day.
    selected_key = args.model or select_model_key(default=DEFAULT_TEXT_MODEL_KEY)
    if selected_key is None:
        return 1
    model_value = model_annotation_value(
        client.base_url, selected_key, IWAC_SUMMARY_MODEL_PROPERTY_ID, "AI Model - Summary"
    )
    target = PropertyTarget(
        term="dcterms:tableOfContents",
        property_id=DCTERMS_TABLE_OF_CONTENTS_PROPERTY_ID,
        property_label="Table Of Contents",
        annotation_term="iwac:summaryModel",
        annotation_value=model_value,
    )
    updates = [
        TextUpdate(
            label=f"item {entry['item_id']}",
            item_id=entry["item_id"],
            text=entry["table_of_contents"],
        )
        for entry in toc_entries
    ]
    stats = run_text_updates(
        client,
        updates,
        target,
        console=console,
        dry_run=guard.dry_run,
        require_confirmation=not guard.assume_yes,
        extra_confirm_lines=[f"Source folder:    {extractions_dir}"],
        description="Updating tables of contents...",
        backup_dir=backup_dir,
        backup_label="table_of_contents",
    )
    if not stats:
        return 1  # operator declined
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
