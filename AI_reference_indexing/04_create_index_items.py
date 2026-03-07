#!/usr/bin/env python3
"""
Batch-create new authority items in Omeka S from reviewed unreconciled terms.

The user reviews the unreconciled CSV (from step 3) and adds an "Action" column
with values "create" or "skip". This script creates Omeka items for "create" rows.

Usage:
    python 04_create_index_items.py --input-csv output/items_enriched_..._unreconciled_subject.csv \\
                                    --item-set-id 854 --type subject
    python 04_create_index_items.py --input-csv output/items_enriched_..._unreconciled_spatial.csv \\
                                    --item-set-id 268 --type spatial
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeElapsedColumn,
)
from rich import box

console = Console()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from common.omeka_client import OmekaClient  # noqa: E402

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def build_item_payload(term: str, item_set_id: int) -> dict:
    """Build the Omeka S JSON payload for a new authority item."""
    return {
        "o:item_set": [{"o:id": item_set_id}],
        "dcterms:title": [
            {
                "type": "literal",
                "property_id": 1,
                "property_label": "Title",
                "is_public": True,
                "@value": term,
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Create new index items from reviewed unreconciled CSV")
    parser.add_argument("--input-csv", required=True, help="Path to reviewed unreconciled CSV with Action column")
    parser.add_argument("--item-set-id", required=True, type=int, help="Target authority item set ID")
    parser.add_argument("--type", required=True, choices=["subject", "spatial"], help="Type of authority")
    args = parser.parse_args()

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

    # Read and validate CSV
    with open(args.input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "Action" not in (reader.fieldnames or []):
            console.print("[red]✗[/] CSV missing 'Action' column. Add 'create' or 'skip' per row.")
            return
        if "Unreconciled Value" not in (reader.fieldnames or []):
            console.print("[red]✗[/] CSV missing 'Unreconciled Value' column.")
            return
        rows = list(reader)

    to_create = [r for r in rows if r.get("Action", "").strip().lower() == "create"]
    to_skip = [r for r in rows if r.get("Action", "").strip().lower() == "skip"]

    config_table = Table(title="Configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Input file", os.path.basename(args.input_csv))
    config_table.add_row("Target item set", str(args.item_set_id))
    config_table.add_row("Type", args.type)
    config_table.add_row("Terms to create", str(len(to_create)))
    config_table.add_row("Terms to skip", str(len(to_skip)))
    config_table.add_row("No action specified", str(len(rows) - len(to_create) - len(to_skip)))
    console.print(config_table)
    console.print()

    if not to_create:
        console.print("[yellow]No terms marked 'create'. Nothing to do.[/]")
        return

    # Create items
    created = []
    errors = 0

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Creating authority items...", total=len(to_create))

        for row in to_create:
            term = row["Unreconciled Value"].strip()
            payload = build_item_payload(term, args.item_set_id)
            result = client.create_item(payload)

            if result:
                new_id = result.get("o:id", "?")
                created.append({"term": term, "o:id": str(new_id)})
                console.print(f"  [green]✓[/] Created: {term} → ID {new_id}")
            else:
                errors += 1
                console.print(f"  [red]✗[/] Failed: {term}")

            progress.update(task, advance=1)

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
    console.print(Panel(
        f"[green]✓[/] Created [cyan]{len(created)}[/] new {args.type} authority items in set {args.item_set_id}",
        title="Step 4 Complete",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
