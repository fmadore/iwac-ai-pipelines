#!/usr/bin/env python3
"""
Fetch items and export authority index files for reference indexing.

Outputs:
  - output/items_{ids}_{date}.csv          — items with bibo:content
  - output/index_subject.csv               — subject/topic authority terms
  - output/index_spatial.csv               — spatial authority terms

Usage:
    python 01_fetch_references.py --item-set-id 123
    python 01_fetch_references.py --item-set-id 123,456
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

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

# Authority item set IDs (same as NER pipeline)
SPATIAL_AUTHORITY_ITEM_SETS = ["268"]
SUBJECT_AUTHORITY_ITEM_SETS = ["854", "2", "266"]
TOPIC_AUTHORITY_ITEM_SETS = ["1"]

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_value(item: Dict[str, Any], prop: str) -> str:
    """Extract a property value from an Omeka item, preferring French."""
    if prop == "o:id":
        return str(item.get("o:id", ""))
    values = item.get(prop, [])
    if not values:
        return ""
    fr = next((v.get("@value", "") for v in values if v.get("@language") == "fr"), None)
    return fr if fr is not None else values[0].get("@value", "")


def get_resource_ids(item: Dict[str, Any], prop: str) -> str:
    """Extract pipe-separated resource IDs from a linked property."""
    entries = item.get(prop, [])
    if not entries or not isinstance(entries, list):
        return ""
    ids = []
    for entry in entries:
        if isinstance(entry, dict) and "value_resource_id" in entry:
            ids.append(str(entry["value_resource_id"]))
    return "|".join(ids)


def export_authority_index(
    client: OmekaClient,
    item_set_ids: List[str],
    output_path: str,
    label: str,
) -> int:
    """Export authority terms to a CSV index file.

    Columns: id, title, alternatives
    Returns number of terms exported.
    """
    rows = []
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]Fetching {label} authorities...", total=None)
        for item_set_id in item_set_ids:
            progress.update(task, description=f"[cyan]Fetching {label} set {item_set_id}...")
            try:
                items = client.get_items(int(item_set_id))
            except Exception as e:
                console.print(f"[red]✗[/] Error fetching set {item_set_id}: {e}")
                continue
            for item in items:
                item_id = str(item["o:id"])
                title = get_value(item, "dcterms:title")
                alternatives = []
                for alt in item.get("dcterms:alternative", []):
                    v = alt.get("@value", "").strip()
                    if v:
                        alternatives.append(v)
                rows.append({
                    "id": item_id,
                    "title": title,
                    "alternatives": "|".join(alternatives),
                })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title", "alternatives"])
        writer.writeheader()
        writer.writerows(rows)

    console.print(f"[green]✓[/] {label} index: {len(rows)} terms → {os.path.basename(output_path)}")
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch items and authority indices for reference indexing")
    parser.add_argument("--item-set-id", required=True, help="Comma-separated Omeka item set IDs")
    args = parser.parse_args()

    console.print(Panel(
        "[bold]Reference Indexing — Step 1[/bold]\n"
        "Fetch items and export authority index files",
        title="Fetch References",
        border_style="cyan",
    ))

    client = OmekaClient.from_env()
    item_set_ids = [s.strip() for s in args.item_set_id.split(",") if s.strip()]

    # Display config
    config_table = Table(title="Configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Item set IDs", ", ".join(item_set_ids))
    config_table.add_row("Omeka URL", client.base_url)
    config_table.add_row("Output dir", OUTPUT_DIR)
    console.print(config_table)
    console.print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Fetch items ---
    console.rule("[bold cyan]Fetching Items")
    all_items: List[Dict[str, Any]] = []
    for sid in item_set_ids:
        with console.status(f"[cyan]Fetching items from set {sid}...", spinner="dots"):
            items = client.get_items(int(sid))
        console.print(f"  [green]✓[/] Set {sid}: [bold]{len(items)}[/] items")
        all_items.extend(items)

    if not all_items:
        console.print("[red]✗[/] No items found.")
        return

    # Build items CSV
    date_tag = datetime.now().strftime("%Y%m%d")
    ids_tag = "_".join(item_set_ids)
    items_filename = f"items_{ids_tag}_{date_tag}.csv"
    items_path = os.path.join(OUTPUT_DIR, items_filename)

    fieldnames = ["o:id", "Title", "Existing Subject IDs", "Existing Spatial IDs", "bibo:content"]
    rows_written = 0

    with open(items_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in all_items:
            content = get_value(item, "bibo:content")
            writer.writerow({
                "o:id": get_value(item, "o:id"),
                "Title": get_value(item, "dcterms:title"),
                "Existing Subject IDs": get_resource_ids(item, "dcterms:subject"),
                "Existing Spatial IDs": get_resource_ids(item, "dcterms:spatial"),
                "bibo:content": content,
            })
            rows_written += 1

    console.print(f"[green]✓[/] Items CSV: {rows_written} rows → {items_filename}")
    console.print()

    # --- Export authority indices ---
    console.rule("[bold cyan]Exporting Authority Indices")

    subject_sets = SUBJECT_AUTHORITY_ITEM_SETS + TOPIC_AUTHORITY_ITEM_SETS
    subject_path = os.path.join(OUTPUT_DIR, "index_subject.csv")
    export_authority_index(client, subject_sets, subject_path, "Subject/Topic")

    spatial_path = os.path.join(OUTPUT_DIR, "index_spatial.csv")
    export_authority_index(client, SPATIAL_AUTHORITY_ITEM_SETS, spatial_path, "Spatial")

    # Summary
    console.print()
    console.print(Panel(
        f"[green]✓[/] Fetch complete!\n\n"
        f"Items: [cyan]{items_filename}[/]\n"
        f"Subject index: [cyan]index_subject.csv[/]\n"
        f"Spatial index: [cyan]index_spatial.csv[/]",
        title="Step 1 Complete",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
