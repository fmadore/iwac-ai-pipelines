import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Initialize rich console
console = Console()

# Shared Omeka client
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.omeka_client import OmekaClient
from common.console_utils import standard_progress
from common.iwac_config import (
    DCTERMS_SPATIAL_PROPERTY_ID,
    DCTERMS_SUBJECT_PROPERTY_ID,
)
from common.omeka_link_updater import (
    ResourceLinkSpec,
    parse_resource_ids,
    update_item_resource_links,
)
from common.write_guard import WriteGuard, add_write_guard_args


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def update_item_fields(
    client: OmekaClient,
    item_id: str,
    spatial_ids_str: str | None,
    subject_ids_str: str | None,
    *,
    dry_run: bool = False,
    on_pre_write: Callable[[MutableMapping[str, Any]], None] | None = None,
) -> dict:
    """
    Updates an Omeka item with new spatial and subject links.
    Preserves existing data and avoids adding duplicate links.

    Returns:
        dict with keys: 'modified', 'spatial_added', 'subject_added', 'error'
    """
    update = update_item_resource_links(client, item_id, [
        ResourceLinkSpec(
            "dcterms:spatial",
            DCTERMS_SPATIAL_PROPERTY_ID,
            parse_resource_ids(spatial_ids_str),
            "Spatial Coverage",
        ),
        ResourceLinkSpec(
            "dcterms:subject",
            DCTERMS_SUBJECT_PROPERTY_ID,
            parse_resource_ids(subject_ids_str),
            "Subject",
        ),
    ], dry_run=dry_run, on_pre_write=on_pre_write)
    return {
        "modified": update.status in {"updated", "would_update"},
        "spatial_added": update.added_by_term.get("dcterms:spatial", 0),
        "subject_added": update.added_by_term.get("dcterms:subject", 0),
        "error": update.status in {"failed", "invalid_id", "not_found"},
    }


def find_latest_reconciled_csv(output_dir: Path = OUTPUT_DIR) -> Path | None:
    """Return the newest main reconciliation CSV, excluding diagnostic files."""
    if not output_dir.is_dir():
        return None
    candidates = [
        path for path in output_dir.glob("*_reconciled.csv")
        if "_ambiguous_authorities" not in path.name
        and "_unreconciled" not in path.name
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def read_reconciled_rows(input_path: Path) -> list[dict[str, str]]:
    """Read the reconciliation CSV after validating its required columns."""
    with input_path.open("r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        if not reader.fieldnames:
            raise ValueError("CSV file is empty or header is missing")
        required = ["o:id", "Spatial AI Reconciled ID", "Subject AI Reconciled ID"]
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        return list(reader)


def update_rows(
    client: OmekaClient,
    rows: Sequence[Mapping[str, str]],
    *,
    guard: WriteGuard | None = None,
) -> Counter:
    """Update all valid rows and aggregate durable Omeka outcomes."""
    guard = guard or WriteGuard()
    stats = Counter(total=len(rows))
    pre_write: list[MutableMapping[str, Any]] = []
    console.rule("[bold cyan]Processing Items")
    with standard_progress(console) as progress:
        task = progress.add_task(
            "[cyan]Checking Omeka items...[/]" if guard.dry_run
            else "[cyan]Updating Omeka items...[/]",
            total=len(rows),
        )
        for row in rows:
            item_id = row.get("o:id")
            if not item_id:
                stats["skipped"] += 1
                progress.update(task, advance=1)
                continue
            result = update_item_fields(
                client,
                item_id,
                row.get("Spatial AI Reconciled ID"),
                row.get("Subject AI Reconciled ID"),
                dry_run=guard.dry_run,
                on_pre_write=pre_write.append,
            )
            status = "errors" if result["error"] else (
                "modified" if result["modified"] else "skipped"
            )
            stats[status] += 1
            if result["modified"]:
                stats["spatial_added"] += result["spatial_added"]
                stats["subject_added"] += result["subject_added"]
            progress.update(task, advance=1)

    backup_path = guard.dump_backup(pre_write, label="ner_links")
    if backup_path is not None:
        console.print(f"[dim]Pre-write payloads saved to {backup_path}[/]")
    return stats


def show_configuration(input_path: Path, client: OmekaClient) -> None:
    config_table = Table(title="Configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Input file", input_path.name)
    config_table.add_row("Omeka URL", client.base_url)
    console.print(config_table)
    console.print()


def show_summary(stats: Mapping[str, int], *, dry_run: bool = False) -> None:
    console.print()
    summary_table = Table(
        title="Dry-run Summary" if dry_run else "Update Summary", box=box.ROUNDED
    )
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Count", justify="right")
    summary_table.add_row("Total items processed", str(stats["total"]))
    summary_table.add_row(
        "Items that would change" if dry_run else "Items modified",
        f"[green]{stats['modified']}[/]",
    )
    summary_table.add_row("Items skipped (no changes)", f"[dim]{stats['skipped']}[/]")
    summary_table.add_row(
        "Errors",
        f"[red]{stats['errors']}[/]" if stats["errors"] else "[dim]0[/]",
    )
    prefix = "Spatial links to add" if dry_run else "Spatial links added"
    summary_table.add_row(prefix, f"[cyan]{stats['spatial_added']}[/]")
    summary_table.add_row(
        "Subject links to add" if dry_run else "Subject links added",
        f"[cyan]{stats['subject_added']}[/]",
    )
    console.print(summary_table)

    console.print()
    if dry_run:
        console.print(Panel(
            f"[cyan]Dry run — nothing was written.[/]\n\n"
            f"A live run would modify [cyan]{stats['modified']}[/] items with "
            f"[cyan]{stats['spatial_added']}[/] spatial and "
            f"[cyan]{stats['subject_added']}[/] subject links",
            title="Dry Run Complete",
            border_style="cyan",
        ))
        return
    console.print(Panel(
        f"[green]{chr(10003)}[/] Update complete!\n\n"
        f"Modified [cyan]{stats['modified']}[/] items with "
        f"[cyan]{stats['spatial_added']}[/] spatial and "
        f"[cyan]{stats['subject_added']}[/] subject links",
        title="Process Complete",
        border_style="green",
    ))


def build_parser() -> argparse.ArgumentParser:
    """Parse argv so a stray flag is an error, never a silent live run."""
    parser = argparse.ArgumentParser(
        description=(
            "Write reconciled dcterms:spatial and dcterms:subject links from the "
            "latest AI_NER reconciliation CSV back into Omeka S."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Reconciliation CSV to apply (default: newest *_reconciled.csv in output/).",
    )
    add_write_guard_args(parser, default_backup_dir=OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    guard = WriteGuard.from_args(args, default_backup_dir=OUTPUT_DIR)

    # Display welcome banner
    console.print(Panel(
        "[bold]Omeka S Database Update[/bold]\n"
        "Updates items with reconciled spatial and subject entity links",
        title="Omeka Update Pipeline",
        border_style="cyan"
    ))

    # Initialize shared Omeka client
    try:
        client = OmekaClient.from_env()
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        return 1

    if not OUTPUT_DIR.is_dir():
        console.print(f"[red]Output directory not found: {OUTPUT_DIR}[/]")
        return 1

    input_csv_path = args.input or find_latest_reconciled_csv()
    if input_csv_path is None:
        console.print(f"[red]No '*_reconciled.csv' files found in {OUTPUT_DIR}[/]")
        return 1
    if not input_csv_path.is_file():
        console.print(f"[red]Input file not found: {input_csv_path}[/]")
        return 1

    show_configuration(input_csv_path, client)

    try:
        rows_to_process = read_reconciled_rows(input_csv_path)
    except (OSError, ValueError) as e:
        console.print(f"[red]{e}[/]")
        return 1

    if not guard.confirm(
        console,
        action="Append dcterms:spatial and dcterms:subject links",
        base_url=client.base_url,
        item_count=len(rows_to_process),
        details=[f"Source:        {input_csv_path.name}"],
    ):
        return 1

    try:
        stats = update_rows(client, rows_to_process, guard=guard)
    except (OSError, ValueError) as e:
        console.print(f"[red]{e}[/]")
        return 1
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/]")
        console.print_exception()
        return 1

    show_summary(stats, dry_run=guard.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
