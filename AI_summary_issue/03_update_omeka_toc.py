"""
Update Omeka S items with table of contents extracted by the issue-indexing agent.

Reads a JSON file mapping item IDs to table of contents text and updates
each item's dcterms:tableOfContents field in Omeka S.

Usage:
    python 03_update_omeka_toc.py --input toc_results.json
    python 03_update_omeka_toc.py --input toc_results.json --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.omeka_client import OmekaClient

console = Console()

# dcterms:tableOfContents property ID in Omeka S
TABLE_OF_CONTENTS_PROPERTY_ID = 18

# iwac:summaryModel property ID and Claude Opus 4.6 item for annotation
SUMMARY_MODEL_PROPERTY_ID = 313
CLAUDE_OPUS_ITEM_ID = 78528
CLAUDE_OPUS_API_URL = f"https://islam.zmo.de/api/items/{CLAUDE_OPUS_ITEM_ID}"
CLAUDE_OPUS_DISPLAY_TITLE = "Claude Opus 4.6"


def load_toc_results(input_path: Path) -> list:
    """Load TOC results from JSON file.

    Expected format:
        [{"item_id": 123, "table_of_contents": "p. 1-3 : ..."}]
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)



def update_items(client: OmekaClient, toc_entries: list, dry_run: bool = False) -> dict:
    """Update Omeka items with table of contents.

    Fetches each item first, modifies only the dcterms:tableOfContents field
    on the full item object, then PATCHes the whole thing back.  This preserves
    all existing metadata (same pattern as AI_summary/03_omeka_update_summaries.py).

    Returns a summary dict with counts of modified, skipped, and errored items.
    """
    stats = {"modified": 0, "skipped": 0, "errors": 0}

    for entry in toc_entries:
        item_id = entry["item_id"]
        toc_text = entry.get("table_of_contents", "").strip()

        if not toc_text:
            console.print(f"  [yellow]Skipped[/] item {item_id} -- empty table of contents")
            stats["skipped"] += 1
            continue

        if dry_run:
            console.print(f"  [cyan]Would update[/] item {item_id} ({len(toc_text)} chars)")
            stats["modified"] += 1
            continue

        # Fetch full existing item to preserve ALL metadata
        item_data = client.get_item(item_id)
        if not item_data:
            console.print(f"  [red]x[/] Could not fetch item {item_id}")
            stats["errors"] += 1
            continue

        # Build the TOC value with annotation
        toc_value = {
            "type": "literal",
            "property_id": TABLE_OF_CONTENTS_PROPERTY_ID,
            "property_label": "Table Of Contents",
            "is_public": True,
            "@annotation": {
                "iwac:summaryModel": [
                    {
                        "type": "resource:item",
                        "property_id": SUMMARY_MODEL_PROPERTY_ID,
                        "property_label": "AI Model - Summary",
                        "is_public": True,
                        "@id": CLAUDE_OPUS_API_URL,
                        "value_resource_id": CLAUDE_OPUS_ITEM_ID,
                        "value_resource_name": "items",
                        "url": None,
                        "display_title": CLAUDE_OPUS_DISPLAY_TITLE,
                    }
                ]
            },
            "@value": toc_text,
        }

        # Set or replace dcterms:tableOfContents on the full item
        item_data["dcterms:tableOfContents"] = [toc_value]

        success = client.update_item(item_id, item_data)

        if success:
            console.print(f"  [green]OK[/] Updated item {item_id}")
            stats["modified"] += 1
        else:
            console.print(f"  [red]FAIL[/] Failed to update item {item_id}")
            stats["errors"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Update Omeka S items with table of contents from JSON."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to JSON file with TOC results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without updating Omeka",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Error:[/] File not found: {input_path}")
        sys.exit(1)

    # Load TOC results
    toc_entries = load_toc_results(input_path)
    if not toc_entries:
        console.print("[yellow]No entries found in input file.[/]")
        sys.exit(0)

    # Display config
    mode = "[yellow]DRY RUN[/]" if args.dry_run else "[green]LIVE[/]"
    console.print(Panel(
        f"Input: {input_path}\nItems: {len(toc_entries)}\nMode: {mode}",
        title="Update Omeka — Table of Contents",
    ))

    # Initialize client and update
    client = OmekaClient.from_env()
    stats = update_items(client, toc_entries, dry_run=args.dry_run)

    # Summary
    summary = Table(title="Summary")
    summary.add_column("Status", style="bold")
    summary.add_column("Count", justify="right")
    summary.add_row("[green]Modified[/]", str(stats["modified"]))
    summary.add_row("[yellow]Skipped[/]", str(stats["skipped"]))
    summary.add_row("[red]Errors[/]", str(stats["errors"]))
    console.print(summary)


if __name__ == "__main__":
    main()
