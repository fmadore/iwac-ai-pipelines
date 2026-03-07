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

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeElapsedColumn,
)
from rich import box

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

console = Console()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from common.omeka_client import OmekaClient  # noqa: E402

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Authority type → (item_set_id, resource_template_id, resource_class_id)
AUTHORITY_TYPE_CONFIG = {
    "subject":     {"item_set": 1,   "resource_template": 3, "resource_class": 244},
    "spatial":     {"item_set": 268, "resource_template": 6, "resource_class": 9},
    "association": {"item_set": 854, "resource_template": 7, "resource_class": 96},
    "individu":    {"item_set": 266, "resource_template": 5, "resource_class": 94},
    "event":       {"item_set": 2,   "resource_template": 2, "resource_class": 54},
}


def build_item_payload(term: str, authority_type: str) -> dict:
    """Build the Omeka S JSON payload for a new authority item.

    Uses the correct item set, resource template, and resource class
    based on the authority type. All authority items also get
    dcterms:type → "Notice d'autorité" (linked item 67568, customvocab:6).
    """
    config = AUTHORITY_TYPE_CONFIG[authority_type]
    return {
        "o:item_set": [{"o:id": config["item_set"]}],
        "o:resource_class": {"o:id": config["resource_class"]},
        "o:resource_template": {"o:id": config["resource_template"]},
        "dcterms:title": [
            {
                "type": "literal",
                "property_id": 1,
                "property_label": "Title",
                "is_public": True,
                "@value": term,
            }
        ],
        "dcterms:type": [
            {
                "type": "customvocab:6",
                "property_id": 8,
                "property_label": "Type",
                "is_public": True,
                "@id": "https://islam.zmo.de/api/items/67568",
                "value_resource_id": 67568,
                "value_resource_name": "items",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Create new index items from reviewed unreconciled CSV")
    parser.add_argument("--input-csv", required=True, help="Path to reviewed unreconciled CSV with Action column")
    parser.add_argument(
        "--type", required=True,
        choices=list(AUTHORITY_TYPE_CONFIG.keys()),
        help="Authority type (determines item set, resource template, and class)",
    )
    args = parser.parse_args()

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
            payload = build_item_payload(term, args.type)
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
        f"[green]✓[/] Created [cyan]{len(created)}[/] new {args.type} authority items in set {type_config['item_set']}",
        title="Step 4 Complete",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
