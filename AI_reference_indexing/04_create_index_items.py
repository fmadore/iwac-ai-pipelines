#!/usr/bin/env python3
"""
Batch-create new authority items in Omeka S from reviewed unreconciled terms.

The user reviews the unreconciled CSV (from step 3) and adds an "Action" column
with values "create" or "skip". This script creates Omeka items for "create" rows.

Each authority type has its own item set, resource template, and resource class:
  - subject (topics)   → item set 1,   template 3, class 244
  - spatial            → item set 268, template 6, class 9
  - association (orgs) → item set 854, template 7, class 96
  - individu (people)  → item set 266, template 5, class 94
  - event              → item set 2,   template 2, class 54

Usage:
    python 04_create_index_items.py --input-csv output/..._unreconciled_subject.csv --type subject
    python 04_create_index_items.py --input-csv output/..._unreconciled_spatial.csv --type spatial
    python 04_create_index_items.py --input-csv output/..._unreconciled_subject.csv --type association
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

console = Console()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.omeka_client import OmekaClient  # noqa: E402
from common.iwac_config import (  # noqa: E402
    AUTHORITY_RECORD_TYPE_ITEM_ID,
    DCTERMS_TITLE_PROPERTY_ID,
    DCTERMS_TYPE_PROPERTY_ID,
    item_api_url,
)
from common.console_utils import standard_progress  # noqa: E402
from common.write_guard import WriteGuard, add_write_guard_args  # noqa: E402
from common.log_redaction import install_credential_redaction

# Credentials ride in Omeka query strings and provider headers; keep them
# out of anything urllib3 or an SDK decides to log.
install_credential_redaction()

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Authority type → (item_set_id, resource_template_id, resource_class_id)
AUTHORITY_TYPE_CONFIG = {
    "subject":     {"item_set": 1,   "resource_template": 3, "resource_class": 244},
    "spatial":     {"item_set": 268, "resource_template": 6, "resource_class": 9},
    "association": {"item_set": 854, "resource_template": 7, "resource_class": 96},
    "individu":    {"item_set": 266, "resource_template": 5, "resource_class": 94},
    "event":       {"item_set": 2,   "resource_template": 2, "resource_class": 54},
}


def build_item_payload(term: str, authority_type: str, base_url: str) -> dict:
    """Build the Omeka S JSON payload for a new authority item.

    Uses the correct item set, resource template, and resource class
    based on the authority type. All authority items also get
    dcterms:type → "Notice d'autorité" (linked authority-record type item,
    customvocab:6). ``base_url`` is the OmekaClient API base URL, used to
    build the linked item's ``@id``.
    """
    config = AUTHORITY_TYPE_CONFIG[authority_type]
    return {
        "o:item_set": [{"o:id": config["item_set"]}],
        "o:resource_class": {"o:id": config["resource_class"]},
        "o:resource_template": {"o:id": config["resource_template"]},
        "dcterms:title": [
            {
                "type": "literal",
                "property_id": DCTERMS_TITLE_PROPERTY_ID,
                "property_label": "Title",
                "is_public": True,
                "@value": term,
            }
        ],
        "dcterms:type": [
            {
                "type": "customvocab:6",
                "property_id": DCTERMS_TYPE_PROPERTY_ID,
                "property_label": "Type",
                "is_public": True,
                "@id": item_api_url(base_url, AUTHORITY_RECORD_TYPE_ITEM_ID),
                "value_resource_id": AUTHORITY_RECORD_TYPE_ITEM_ID,
                "value_resource_name": "items",
            }
        ],
    }


def read_reviewed_rows(input_csv: str) -> list[dict[str, str]]:
    """Read the reviewed CSV, raising when the reviewer's columns are absent."""
    with open(input_csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "Action" not in fieldnames:
            raise ValueError("CSV missing 'Action' column. Add 'create' or 'skip' per row.")
        if "Unreconciled Value" not in fieldnames:
            raise ValueError("CSV missing 'Unreconciled Value' column.")
        return list(reader)


def partition_by_action(rows: Sequence[Mapping[str, str]]) -> tuple[list, list]:
    """Split reviewed rows into the create and skip buckets."""
    def action(row: Mapping[str, str]) -> str:
        return row.get("Action", "").strip().lower()

    return (
        [row for row in rows if action(row) == "create"],
        [row for row in rows if action(row) == "skip"],
    )


def create_authority_items(
    client: OmekaClient,
    to_create: Sequence[Mapping[str, str]],
    authority_type: str,
    *,
    guard: WriteGuard,
) -> tuple[list[dict[str, str]], int]:
    """Create one item per reviewed term; return the mapping and error count."""
    created: list[dict[str, str]] = []
    errors = 0
    with standard_progress(console) as progress:
        task = progress.add_task(
            "[cyan]Checking authority items...[/]" if guard.dry_run
            else "[cyan]Creating authority items...[/]",
            total=len(to_create),
        )
        for row in to_create:
            term = row["Unreconciled Value"].strip()
            payload = build_item_payload(term, authority_type, client.base_url)
            if guard.dry_run:
                console.print(f"  [cyan]would create:[/] {term}")
            else:
                result = client.create_item(payload)
                if result:
                    new_id = result.get("o:id", "?")
                    created.append({"term": term, "o:id": str(new_id)})
                    console.print(f"  [green]{chr(10003)}[/] Created: {term} → ID {new_id}")
                else:
                    errors += 1
                    console.print(f"  [red]{chr(10007)}[/] Failed: {term}")
            progress.update(task, advance=1)
    return created, errors


def main():
    parser = argparse.ArgumentParser(description="Create new index items from reviewed unreconciled CSV")
    parser.add_argument("--input-csv", required=True, help="Path to reviewed unreconciled CSV with Action column")
    parser.add_argument(
        "--type", required=True,
        choices=list(AUTHORITY_TYPE_CONFIG.keys()),
        help="Authority type (determines item set, resource template, and class)",
    )
    add_write_guard_args(parser, default_backup_dir=Path(OUTPUT_DIR))
    args = parser.parse_args()
    guard = WriteGuard.from_args(args, default_backup_dir=Path(OUTPUT_DIR))

    type_config = AUTHORITY_TYPE_CONFIG[args.type]

    console.print(Panel(
        "[bold]Reference Indexing — Step 4[/bold]\n"
        "Batch-create new authority items in Omeka S",
        title="Create Index Items",
        border_style="cyan",
    ))

    client = OmekaClient.from_env()

    if not os.path.exists(args.input_csv):
        console.print(f"[red]✗[/] File not found: {args.input_csv}")
        return

    try:
        rows = read_reviewed_rows(args.input_csv)
    except (OSError, ValueError) as exc:
        console.print(f"[red]✗[/] {exc}")
        return

    to_create, to_skip = partition_by_action(rows)

    config_table = Table(title="Configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Input file", os.path.basename(args.input_csv))
    config_table.add_row("Type", args.type)
    config_table.add_row("Target item set", str(type_config["item_set"]))
    config_table.add_row("Resource template", str(type_config["resource_template"]))
    config_table.add_row("Resource class", str(type_config["resource_class"]))
    config_table.add_row("Terms to create", str(len(to_create)))
    config_table.add_row("Terms to skip", str(len(to_skip)))
    config_table.add_row("No action specified", str(len(rows) - len(to_create) - len(to_skip)))
    console.print(config_table)
    console.print()

    if not to_create:
        console.print("[yellow]No terms marked 'create'. Nothing to do.[/]")
        return

    if not guard.confirm(
        console,
        action=f"Create {args.type} authority items",
        base_url=client.base_url,
        item_count=len(to_create),
        details=[f"Item set:      {type_config['item_set']}"],
        title="About to create Omeka items",
    ):
        return

    created, errors = create_authority_items(client, to_create, args.type, guard=guard)

    # Write output mapping
    if created:
        date_tag = datetime.now().strftime("%Y%m%d")
        out_filename = f"newly_created_items_{args.type}_{date_tag}.csv"
        out_path = os.path.join(OUTPUT_DIR, out_filename)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["term", "o:id"])
            writer.writeheader()
            writer.writerows(created)

        console.print(f"\n[green]✓[/] Mapping saved to: {out_filename}")

    # Summary
    console.print()
    summary = Table(title="Creation Summary", box=box.ROUNDED)
    summary.add_column("Metric", style="dim")
    summary.add_column("Count", justify="right")
    summary.add_row("Created", f"[green]{len(created)}[/]")
    summary.add_row("Errors", f"[red]{errors}[/]" if errors else "[dim]0[/]")
    summary.add_row("Skipped", f"[dim]{len(to_skip)}[/]")
    console.print(summary)

    console.print()
    if guard.dry_run:
        console.print(Panel(
            f"[cyan]Dry run — nothing was created.[/]\n\n"
            f"A live run would create [cyan]{len(to_create)}[/] {args.type} "
            f"authority items in set {type_config['item_set']}",
            title="Step 4 Dry Run",
            border_style="cyan",
        ))
        return
    console.print(Panel(
        f"[green]✓[/] Created [cyan]{len(created)}[/] new {args.type} authority items in set {type_config['item_set']}",
        title="Step 4 Complete",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
