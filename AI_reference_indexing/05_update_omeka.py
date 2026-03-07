#!/usr/bin/env python3
"""
Update Omeka S items with reconciled subject and spatial resource links.

Reads the reconciled CSV from step 3 and optionally merges newly created item
mappings from step 4. For each item, deduplicates against existing links and
PATCHes via OmekaClient.

Usage:
    python 05_update_omeka.py
    python 05_update_omeka.py --new-subject output/newly_created_items_subject_20260307.csv
    python 05_update_omeka.py --new-spatial output/newly_created_items_spatial_20260307.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, Optional, Set

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

# Property IDs in Omeka S
SPATIAL_PROPERTY_ID = 40   # dcterms:spatial
SUBJECT_PROPERTY_ID = 3    # dcterms:subject


def load_newly_created_mapping(csv_path: str) -> Dict[str, str]:
    """Load term→id mapping from a newly_created_items CSV."""
    mapping = {}
    if not csv_path or not os.path.exists(csv_path):
        return mapping
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            term = row.get("term", "").strip()
            oid = row.get("o:id", "").strip()
            if term and oid:
                mapping[term.lower()] = oid
    return mapping


def find_latest_reconciled_csv() -> Optional[str]:
    """Find the most recent *_reconciled.csv in the output directory."""
    try:
        candidates = [
            f for f in os.listdir(OUTPUT_DIR)
            if f.endswith("_reconciled.csv")
            and "_ambiguous_authorities" not in f
            and "_unreconciled" not in f
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)))
    except OSError:
        return None


def get_existing_resource_ids(item_data: dict, prop: str) -> Set[int]:
    """Extract existing resource IDs from an item's property."""
    ids = set()
    for entry in item_data.get(prop, []):
        if isinstance(entry, dict) and "value_resource_id" in entry:
            try:
                ids.add(int(entry["value_resource_id"]))
            except (ValueError, TypeError):
                pass
    return ids


def add_resource_links(
    item_data: dict,
    prop: str,
    property_id: int,
    property_label: str,
    ids_str: str,
    existing_ids: Set[int],
) -> int:
    """Add resource links to item_data, skipping duplicates. Returns count added."""
    if not ids_str:
        return 0

    if prop not in item_data or not isinstance(item_data[prop], list):
        item_data[prop] = []

    added = 0
    for raw_id in ids_str.split("|"):
        raw_id = raw_id.strip()
        if not raw_id:
            continue
        try:
            int_id = int(raw_id)
        except ValueError:
            continue
        if int_id in existing_ids:
            continue
        item_data[prop].append({
            "type": "resource:item",
            "property_id": property_id,
            "property_label": property_label,
            "is_public": True,
            "value_resource_id": int_id,
            "value_resource_name": "items",
        })
        existing_ids.add(int_id)
        added += 1
    return added


def main():
    parser = argparse.ArgumentParser(description="Update Omeka items with reconciled metadata links")
    parser.add_argument("--new-subject", default=None, help="CSV with newly created subject items (term,o:id)")
    parser.add_argument("--new-spatial", default=None, help="CSV with newly created spatial items (term,o:id)")
    args = parser.parse_args()

    console.print(Panel(
        "[bold]Reference Indexing — Step 5[/bold]\n"
        "Update Omeka S items with reconciled subject and spatial links",
        title="Update Omeka",
        border_style="cyan",
    ))

    client = OmekaClient.from_env()

    # Find reconciled CSV
    latest = find_latest_reconciled_csv()
    if not latest:
        console.print("[red]✗[/] No *_reconciled.csv found in output/")
        return

    input_path = os.path.join(OUTPUT_DIR, latest)

    # Load newly created mappings (optional)
    new_subject_map = load_newly_created_mapping(args.new_subject) if args.new_subject else {}
    new_spatial_map = load_newly_created_mapping(args.new_spatial) if args.new_spatial else {}

    config_table = Table(title="Configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Input file", latest)
    config_table.add_row("Omeka URL", client.base_url)
    if new_subject_map:
        config_table.add_row("New subject terms", str(len(new_subject_map)))
    if new_spatial_map:
        config_table.add_row("New spatial terms", str(len(new_spatial_map)))
    console.print(config_table)
    console.print()

    # Read rows — reconciled CSV may contain large bibo:content fields
    csv.field_size_limit(sys.maxsize)
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            console.print("[red]✗[/] CSV is empty")
            return
        required = ["o:id"]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            console.print(f"[red]✗[/] Missing columns: {', '.join(missing)}")
            return
        rows = list(reader)

    stats = {
        "total": len(rows),
        "modified": 0,
        "skipped": 0,
        "errors": 0,
        "spatial_added": 0,
        "subject_added": 0,
    }

    console.rule("[bold cyan]Processing Items")

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Updating Omeka items...", total=len(rows))

        for row in rows:
            item_id = row.get("o:id", "").strip()
            if not item_id:
                stats["skipped"] += 1
                progress.update(task, advance=1)
                continue

            spatial_ids = row.get("Spatial AI Reconciled ID", "")
            subject_ids = row.get("Subject AI Reconciled ID", "")

            # If we have unreconciled terms and newly created mappings, resolve them
            if new_spatial_map:
                spatial_raw = row.get("Spatial AI", "")
                if spatial_raw:
                    extra = []
                    for term in spatial_raw.split("|"):
                        t = term.strip().lower()
                        if t in new_spatial_map:
                            extra.append(new_spatial_map[t])
                    if extra:
                        existing = spatial_ids.split("|") if spatial_ids else []
                        spatial_ids = "|".join(existing + extra)

            if new_subject_map:
                subject_raw = row.get("Subject AI", "")
                if subject_raw:
                    extra = []
                    for term in subject_raw.split("|"):
                        t = term.strip().lower()
                        if t in new_subject_map:
                            extra.append(new_subject_map[t])
                    if extra:
                        existing = subject_ids.split("|") if subject_ids else []
                        subject_ids = "|".join(existing + extra)

            if not spatial_ids and not subject_ids:
                stats["skipped"] += 1
                progress.update(task, advance=1)
                continue

            item_data = client.get_item(int(item_id))
            if not item_data:
                stats["errors"] += 1
                progress.update(task, advance=1)
                continue

            existing_spatial = get_existing_resource_ids(item_data, "dcterms:spatial")
            existing_subject = get_existing_resource_ids(item_data, "dcterms:subject")

            s_added = add_resource_links(
                item_data, "dcterms:spatial", SPATIAL_PROPERTY_ID,
                "Spatial Coverage", spatial_ids, existing_spatial,
            )
            subj_added = add_resource_links(
                item_data, "dcterms:subject", SUBJECT_PROPERTY_ID,
                "Subject", subject_ids, existing_subject,
            )

            if s_added or subj_added:
                if client.update_item(int(item_id), item_data):
                    stats["modified"] += 1
                    stats["spatial_added"] += s_added
                    stats["subject_added"] += subj_added
                else:
                    stats["errors"] += 1
            else:
                stats["skipped"] += 1

            progress.update(task, advance=1)

    # Summary
    console.print()
    summary = Table(title="Update Summary", box=box.ROUNDED)
    summary.add_column("Metric", style="dim")
    summary.add_column("Count", justify="right")
    summary.add_row("Total items processed", str(stats["total"]))
    summary.add_row("Items modified", f"[green]{stats['modified']}[/]")
    summary.add_row("Items skipped (no changes)", f"[dim]{stats['skipped']}[/]")
    summary.add_row("Errors", f"[red]{stats['errors']}[/]" if stats["errors"] else "[dim]0[/]")
    summary.add_row("Spatial links added", f"[cyan]{stats['spatial_added']}[/]")
    summary.add_row("Subject links added", f"[cyan]{stats['subject_added']}[/]")
    console.print(summary)

    console.print()
    console.print(Panel(
        f"[green]✓[/] Update complete!\n\n"
        f"Modified [cyan]{stats['modified']}[/] items with "
        f"[cyan]{stats['spatial_added']}[/] spatial and [cyan]{stats['subject_added']}[/] subject links",
        title="Step 5 Complete",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
