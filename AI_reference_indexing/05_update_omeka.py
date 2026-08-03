#!/usr/bin/env python3
"""
Update Omeka S items with reconciled subject and spatial resource links.

Reads the reconciled CSV from step 3 and optionally merges newly created item
mappings from step 4. For each item, deduplicates against existing links and
PATCHes via OmekaClient.

Usage:
    python 05_update_omeka.py --dry-run
    python 05_update_omeka.py
    python 05_update_omeka.py --new-subject output/newly_created_items_subject_20260307.csv
    python 05_update_omeka.py --new-spatial output/newly_created_items_spatial_20260307.csv

Writes are gated: --dry-run reports without PATCHing, the pre-write payloads are
dumped to output/ first, and a live run asks before the first write.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence

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
    DCTERMS_SPATIAL_PROPERTY_ID,
    DCTERMS_SUBJECT_PROPERTY_ID,
)
from common.console_utils import standard_progress  # noqa: E402
from common.omeka_link_updater import (  # noqa: E402
    ResourceLinkSpec,
    parse_resource_ids,
    update_item_resource_links,
)
from common.write_guard import WriteGuard, add_write_guard_args  # noqa: E402
from common.log_redaction import install_credential_redaction

# Credentials ride in Omeka query strings and provider headers; keep them
# out of anything urllib3 or an SDK decides to log.
install_credential_redaction()

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


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


parse_id_list = parse_resource_ids


def read_reconciled_rows(csv_path: str) -> List[Dict[str, str]]:
    """Read and validate a reconciled export."""
    csv.field_size_limit(10 * 1024 * 1024)  # sys.maxsize overflows C long on Windows
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV is empty")
        missing = [column for column in ("o:id",) if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")
        return list(reader)


def resolved_ids(
    row: Mapping[str, str],
    reconciled_column: str,
    source_column: str,
    new_mapping: Mapping[str, str],
) -> List[int]:
    """Combine reconciled IDs with IDs for newly created authority terms."""
    ids = parse_id_list(row.get(reconciled_column, ""))
    mapped_ids = (
        new_mapping.get(term.strip().lower(), "")
        for term in row.get(source_column, "").split("|")
    )
    ids.extend(parse_id_list("|".join(mapped_ids)))
    return ids


@dataclass(frozen=True)
class ItemUpdateResult:
    """Outcome of attempting to update one reconciled CSV row."""

    status: str
    spatial_added: int = 0
    subject_added: int = 0


def update_reconciled_item(
    client: OmekaClient,
    row: Mapping[str, str],
    *,
    new_spatial_map: Mapping[str, str],
    new_subject_map: Mapping[str, str],
    dry_run: bool = False,
    on_pre_write: Callable[[MutableMapping[str, Any]], None] | None = None,
) -> ItemUpdateResult:
    """Apply resource links from one row and return a countable outcome."""
    raw_item_id = row.get("o:id", "").strip()
    if not raw_item_id:
        return ItemUpdateResult("skipped")
    spatial_ids = resolved_ids(
        row, "Spatial AI Reconciled ID", "Spatial AI", new_spatial_map,
    )
    subject_ids = resolved_ids(
        row, "Subject AI Reconciled ID", "Subject AI", new_subject_map,
    )
    if not spatial_ids and not subject_ids:
        return ItemUpdateResult("skipped")

    result = update_item_resource_links(client, raw_item_id, [
        ResourceLinkSpec(
            "dcterms:spatial",
            DCTERMS_SPATIAL_PROPERTY_ID,
            spatial_ids,
            "Spatial Coverage",
        ),
        ResourceLinkSpec(
            "dcterms:subject",
            DCTERMS_SUBJECT_PROPERTY_ID,
            subject_ids,
            "Subject",
        ),
    ], dry_run=dry_run, on_pre_write=on_pre_write)
    if result.status == "unchanged":
        return ItemUpdateResult("skipped")
    if result.status not in {"updated", "would_update"}:
        return ItemUpdateResult("error")
    return ItemUpdateResult(
        "modified",
        result.added_by_term["dcterms:spatial"],
        result.added_by_term["dcterms:subject"],
    )


def update_reconciled_items(
    client: OmekaClient,
    rows: Sequence[Mapping[str, str]],
    *,
    new_spatial_map: Mapping[str, str],
    new_subject_map: Mapping[str, str],
    guard: WriteGuard | None = None,
) -> Counter:
    """Update all rows with progress reporting and aggregate their outcomes."""
    guard = guard or WriteGuard()
    stats = Counter(total=len(rows))
    pre_write: List[MutableMapping[str, Any]] = []
    console.rule("[bold cyan]Processing Items")
    with standard_progress(console) as progress:
        task = progress.add_task(
            "[cyan]Checking Omeka items...[/]" if guard.dry_run
            else "[cyan]Updating Omeka items...[/]",
            total=len(rows),
        )
        for row in rows:
            result = update_reconciled_item(
                client,
                row,
                new_spatial_map=new_spatial_map,
                new_subject_map=new_subject_map,
                dry_run=guard.dry_run,
                on_pre_write=pre_write.append,
            )
            stats[result.status] += 1
            stats["spatial_added"] += result.spatial_added
            stats["subject_added"] += result.subject_added
            progress.update(task, advance=1)

    backup_path = guard.dump_backup(pre_write, label="reference_links")
    if backup_path is not None:
        console.print(f"[dim]Pre-write payloads saved to {backup_path}[/]")
    return stats


def show_configuration(
    input_name: str,
    client: OmekaClient,
    new_subject_map: Mapping[str, str],
    new_spatial_map: Mapping[str, str],
) -> None:
    """Display the resolved input and authority-map configuration."""
    config_table = Table(title="Configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Input file", input_name)
    config_table.add_row("Omeka URL", client.base_url)
    if new_subject_map:
        config_table.add_row("New subject terms", str(len(new_subject_map)))
    if new_spatial_map:
        config_table.add_row("New spatial terms", str(len(new_spatial_map)))
    console.print(config_table)
    console.print()


def show_summary(stats: Mapping[str, int], *, dry_run: bool = False) -> None:
    """Display aggregate update counts."""
    console.print()
    summary = Table(
        title="Dry-run Summary" if dry_run else "Update Summary", box=box.ROUNDED
    )
    summary.add_column("Metric", style="dim")
    summary.add_column("Count", justify="right")
    summary.add_row("Total items processed", str(stats["total"]))
    summary.add_row(
        "Items that would change" if dry_run else "Items modified",
        f"[green]{stats['modified']}[/]",
    )
    summary.add_row("Items skipped (no changes)", f"[dim]{stats['skipped']}[/]")
    errors = stats["error"]
    summary.add_row("Errors", f"[red]{errors}[/]" if errors else "[dim]0[/]")
    summary.add_row("Spatial links added", f"[cyan]{stats['spatial_added']}[/]")
    summary.add_row("Subject links added", f"[cyan]{stats['subject_added']}[/]")
    console.print(summary)

    console.print()
    if dry_run:
        console.print(Panel(
            f"[cyan]Dry run — nothing was written.[/]\n\n"
            f"A live run would modify [cyan]{stats['modified']}[/] items with "
            f"[cyan]{stats['spatial_added']}[/] spatial and "
            f"[cyan]{stats['subject_added']}[/] subject links",
            title="Step 5 Dry Run",
            border_style="cyan",
        ))
        return
    console.print(Panel(
        f"[green]✓[/] Update complete!\n\n"
        f"Modified [cyan]{stats['modified']}[/] items with "
        f"[cyan]{stats['spatial_added']}[/] spatial and "
        f"[cyan]{stats['subject_added']}[/] subject links",
        title="Step 5 Complete",
        border_style="green",
    ))


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description="Update Omeka items with reconciled metadata links")
    parser.add_argument("--new-subject", default=None, help="CSV with newly created subject items (term,o:id)")
    parser.add_argument("--new-spatial", default=None, help="CSV with newly created spatial items (term,o:id)")
    add_write_guard_args(parser, default_backup_dir=Path(OUTPUT_DIR))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    guard = WriteGuard.from_args(args, default_backup_dir=Path(OUTPUT_DIR))

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
        return 1

    input_path = os.path.join(OUTPUT_DIR, latest)

    # Load newly created mappings (optional)
    new_subject_map = load_newly_created_mapping(args.new_subject) if args.new_subject else {}
    new_spatial_map = load_newly_created_mapping(args.new_spatial) if args.new_spatial else {}
    show_configuration(latest, client, new_subject_map, new_spatial_map)

    try:
        rows = read_reconciled_rows(input_path)
    except (OSError, ValueError) as exc:
        console.print(f"[red]✗[/] {exc}")
        return 1

    if not guard.confirm(
        console,
        action="Append dcterms:spatial and dcterms:subject links to references",
        base_url=client.base_url,
        item_count=len(rows),
        details=[f"Source:        {latest}"],
    ):
        return 1

    stats = update_reconciled_items(
        client,
        rows,
        new_spatial_map=new_spatial_map,
        new_subject_map=new_subject_map,
        guard=guard,
    )
    show_summary(stats, dry_run=guard.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
