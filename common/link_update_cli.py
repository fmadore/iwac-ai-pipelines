"""The write step shared by NER and reference indexing.

Both pipelines end the same way: a ``*_reconciled.csv`` whose rows carry
``Spatial AI Reconciled ID`` / ``Subject AI Reconciled ID`` columns is applied
to Omeka as ``dcterms:spatial`` / ``dcterms:subject`` resource links, each new
link annotated with the model that proposed it. One implementation here; the
two ``03``/``05`` scripts are thin entry points that name their folder and
banner.

Reads are batched: every item the CSV names is fetched in pages of 100 via
``id[]`` before the PATCH loop, instead of one GET per item.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence

from rich.console import Console
from rich.panel import Panel

from common.console_utils import count_table, key_value_table, standard_progress
from common.iwac_config import (
    AI_MODEL_ITEMS,
    DCTERMS_SPATIAL_PROPERTY_ID,
    DCTERMS_SUBJECT_PROPERTY_ID,
    IWAC_NER_MODEL_PROPERTY_ID,
    model_annotation_value,
    select_model_key,
)
from common.omeka_client import OmekaClient
from common.omeka_link_updater import (
    NER_MODEL_LABEL,
    NER_MODEL_TERM,
    ResourceLinkSpec,
    parse_resource_ids,
    provenance_model_key,
    update_item_resource_links,
)
from common.write_guard import WriteGuard, add_write_guard_args

SPATIAL_RECONCILED_COLUMN = "Spatial AI Reconciled ID"
SUBJECT_RECONCILED_COLUMN = "Subject AI Reconciled ID"
SPATIAL_SOURCE_COLUMN = "Spatial AI"
SUBJECT_SOURCE_COLUMN = "Subject AI"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def find_latest_reconciled_csv(output_dir: Path) -> Optional[Path]:
    """Return the newest main reconciliation CSV, excluding diagnostic files."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None
    candidates = [
        path for path in output_dir.glob("*_reconciled.csv")
        if "_ambiguous_authorities" not in path.name
        and "_unreconciled" not in path.name
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def read_reconciled_rows(csv_path: Path | str) -> List[Dict[str, str]]:
    """Read a reconciliation export after validating that it names items."""
    csv.field_size_limit(10 * 1024 * 1024)  # sys.maxsize overflows C long on Windows
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV file is empty or header is missing")
        missing = [column for column in ("o:id",) if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")
        return list(reader)


def load_newly_created_mapping(csv_path: Optional[str]) -> Dict[str, str]:
    """``term -> id`` from a ``newly_created_items_*.csv`` written by step 04."""
    mapping: Dict[str, str] = {}
    if not csv_path or not Path(csv_path).exists():
        return mapping
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            term = (row.get("term") or "").strip()
            oid = (row.get("o:id") or "").strip()
            if term and oid:
                mapping[term.lower()] = oid
    return mapping


def resolved_ids(
    row: Mapping[str, str],
    reconciled_column: str,
    source_column: str,
    new_mapping: Mapping[str, str],
) -> List[int]:
    """Reconciled IDs plus the IDs of terms created since reconciliation."""
    ids = parse_resource_ids(row.get(reconciled_column, ""))
    mapped = (
        new_mapping.get(term.strip().lower(), "")
        for term in (row.get(source_column) or "").split("|")
    )
    ids.extend(parse_resource_ids("|".join(mapped)))
    return ids


# ---------------------------------------------------------------------------
# One item
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ItemUpdateResult:
    """Outcome of applying one reconciled CSV row."""

    status: str  # modified | skipped | error
    spatial_added: int = 0
    subject_added: int = 0


def update_reconciled_item(
    client: OmekaClient,
    row: Mapping[str, str],
    *,
    new_spatial_map: Mapping[str, str] = (),
    new_subject_map: Mapping[str, str] = (),
    annotation: Optional[Mapping[str, Any]] = None,
    dry_run: bool = False,
    on_pre_write: Optional[Callable[[MutableMapping[str, Any]], None]] = None,
    item_data: Optional[Dict[str, Any]] = None,
) -> ItemUpdateResult:
    """Apply one row's links; every link added carries the model annotation."""
    raw_item_id = (row.get("o:id") or "").strip()
    if not raw_item_id:
        return ItemUpdateResult("skipped")
    spatial_ids = resolved_ids(row, SPATIAL_RECONCILED_COLUMN, SPATIAL_SOURCE_COLUMN, dict(new_spatial_map))
    subject_ids = resolved_ids(row, SUBJECT_RECONCILED_COLUMN, SUBJECT_SOURCE_COLUMN, dict(new_subject_map))
    if not spatial_ids and not subject_ids:
        return ItemUpdateResult("skipped")

    annotation_term = NER_MODEL_TERM if annotation else None
    result = update_item_resource_links(client, raw_item_id, [
        ResourceLinkSpec(
            "dcterms:spatial", DCTERMS_SPATIAL_PROPERTY_ID, spatial_ids, "Spatial Coverage",
            annotation_term=annotation_term, annotation_value=annotation,
        ),
        ResourceLinkSpec(
            "dcterms:subject", DCTERMS_SUBJECT_PROPERTY_ID, subject_ids, "Subject",
            annotation_term=annotation_term, annotation_value=annotation,
        ),
    ], dry_run=dry_run, on_pre_write=on_pre_write, item_data=item_data)
    if result.status == "unchanged":
        return ItemUpdateResult("skipped")
    if result.status not in {"updated", "would_update"}:
        return ItemUpdateResult("error")
    return ItemUpdateResult(
        "modified",
        result.added_by_term.get("dcterms:spatial", 0),
        result.added_by_term.get("dcterms:subject", 0),
    )


# ---------------------------------------------------------------------------
# The batch
# ---------------------------------------------------------------------------

def _prefetch(client: OmekaClient, rows: Sequence[Mapping[str, str]]) -> Dict[int, Dict[str, Any]]:
    """Fetch every named item in pages of ``id[]``; empty when unsupported."""
    ids = []
    for row in rows:
        try:
            ids.append(int((row.get("o:id") or "").strip()))
        except ValueError:
            continue
    if not ids:
        return {}
    fetched = client.get_items_by_ids(ids)
    return fetched if isinstance(fetched, dict) else {}


def update_reconciled_items(
    client: OmekaClient,
    rows: Sequence[Mapping[str, str]],
    *,
    guard: Optional[WriteGuard] = None,
    annotation: Optional[Mapping[str, Any]] = None,
    new_spatial_map: Mapping[str, str] = (),
    new_subject_map: Mapping[str, str] = (),
    backup_label: str = "links",
    console: Optional[Console] = None,
) -> Counter:
    """Apply every row, dump the pre-write payloads, and count the outcomes.

    Counter keys: ``total``, ``modified``, ``skipped``, ``errors``,
    ``spatial_added``, ``subject_added``.
    """
    console = console or Console()
    guard = guard or WriteGuard()
    stats: Counter = Counter(total=len(rows))
    pre_write: List[MutableMapping[str, Any]] = []

    console.rule("[bold cyan]Processing Items")
    prefetched = _prefetch(client, rows)
    with standard_progress(console) as progress:
        task = progress.add_task(
            "[cyan]Checking Omeka items...[/]" if guard.dry_run else "[cyan]Updating Omeka items...[/]",
            total=len(rows),
        )
        for row in rows:
            cached = None
            try:
                cached = prefetched.get(int((row.get("o:id") or "").strip()))
            except ValueError:
                pass
            result = update_reconciled_item(
                client, row,
                new_spatial_map=new_spatial_map, new_subject_map=new_subject_map,
                annotation=annotation, dry_run=guard.dry_run,
                on_pre_write=pre_write.append,
                item_data=cached if isinstance(cached, dict) else None,
            )
            stats["errors" if result.status == "error" else result.status] += 1
            stats["spatial_added"] += result.spatial_added
            stats["subject_added"] += result.subject_added
            progress.update(task, advance=1)

    backup_path = guard.dump_backup(pre_write, label=backup_label)
    if backup_path is not None:
        console.print(f"[dim]Pre-write payloads saved to {backup_path}[/]")
    return stats


def show_summary(console: Console, stats: Mapping[str, int], *, dry_run: bool) -> None:
    console.print()
    console.print(count_table([
        ("Total items processed", stats["total"]),
        ("Items that would change" if dry_run else "Items modified", stats["modified"]),
        ("Items skipped (no changes)", stats["skipped"]),
        ("Errors", stats["errors"]),
        ("Spatial links to add" if dry_run else "Spatial links added", stats["spatial_added"]),
        ("Subject links to add" if dry_run else "Subject links added", stats["subject_added"]),
    ], title="Dry-run Summary" if dry_run else "Update Summary"))
    console.print()
    verb = "would modify" if dry_run else "Modified"
    console.print(Panel(
        f"{'[cyan]Dry run — nothing was written.[/]' if dry_run else '[green]✓[/] Update complete!'}\n\n"
        f"{'A live run ' if dry_run else ''}{verb} [cyan]{stats['modified']}[/] items with "
        f"[cyan]{stats['spatial_added']}[/] spatial and [cyan]{stats['subject_added']}[/] subject links",
        title="Dry Run Complete" if dry_run else "Process Complete",
        border_style="cyan" if dry_run else "green",
    ))


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------

def add_link_update_args(parser: argparse.ArgumentParser, *, output_dir: Path) -> argparse.ArgumentParser:
    """The flags every link write step exposes."""
    parser.add_argument(
        "--input", type=Path,
        help="Reconciliation CSV to apply (default: newest *_reconciled.csv in output/).",
    )
    parser.add_argument(
        "--model", choices=list(AI_MODEL_ITEMS),
        help=(
            "AI model that proposed the keywords, recorded as an iwac:nerModel "
            "annotation on every link added. Default: the model named in the "
            "checkpoint the AI step left beside its CSV; prompts when there is none."
        ),
    )
    parser.add_argument("--new-subject", default=None, help="CSV of subject items created by step 04 (term,o:id)")
    parser.add_argument("--new-spatial", default=None, help="CSV of spatial items created by step 04 (term,o:id)")
    add_write_guard_args(parser, default_backup_dir=output_dir)
    return parser


def run_link_update(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    banner: str,
    backup_label: str,
    console: Optional[Console] = None,
) -> int:
    """Everything after argument parsing, shared by both entry points."""
    console = console or Console()
    guard = WriteGuard.from_args(args, default_backup_dir=output_dir)
    console.print(Panel(banner, title="Omeka Update", border_style="cyan"))

    try:
        client = OmekaClient.from_env()
    except ValueError as exc:
        console.print(f"[red]✗[/] {exc}")
        return 1

    input_path = args.input or find_latest_reconciled_csv(output_dir)
    if input_path is None:
        console.print(f"[red]✗[/] No '*_reconciled.csv' found in {output_dir}")
        return 1
    if not Path(input_path).is_file():
        console.print(f"[red]✗[/] Input file not found: {input_path}")
        return 1

    # Which model proposed these keywords? Recorded as an iwac:nerModel
    # annotation on every link added, so the archive can tell AI-assigned
    # subjects from hand-catalogued ones.
    model_key = args.model or provenance_model_key(input_path) or select_model_key()
    if model_key is None:
        return 1
    annotation = model_annotation_value(client.base_url, model_key, IWAC_NER_MODEL_PROPERTY_ID, NER_MODEL_LABEL)
    new_subject_map = load_newly_created_mapping(args.new_subject)
    new_spatial_map = load_newly_created_mapping(args.new_spatial)

    try:
        rows = read_reconciled_rows(input_path)
    except (OSError, ValueError) as exc:
        console.print(f"[red]✗[/] {exc}")
        return 1

    model = AI_MODEL_ITEMS[model_key]
    console.print(key_value_table([
        ("Input file", Path(input_path).name),
        ("Omeka URL", client.base_url),
        ("Provenance", f"{NER_MODEL_TERM} -> {model['display_title']} (item {model['item_id']})"),
        ("New subject terms", len(new_subject_map) or None),
        ("New spatial terms", len(new_spatial_map) or None),
    ]))
    console.print()

    if not guard.confirm(
        console,
        action="Append dcterms:spatial and dcterms:subject links",
        base_url=client.base_url,
        item_count=len(rows),
        details=[
            f"Source:        {Path(input_path).name}",
            f"Annotation:    {NER_MODEL_TERM} -> {annotation['display_title']}",
        ],
    ):
        return 1

    try:
        stats = update_reconciled_items(
            client, rows, guard=guard, annotation=annotation,
            new_spatial_map=new_spatial_map, new_subject_map=new_subject_map,
            backup_label=backup_label, console=console,
        )
    except (OSError, ValueError) as exc:
        console.print(f"[red]✗[/] {exc}")
        return 1

    show_summary(console, stats, dry_run=guard.dry_run)
    return 0 if stats["errors"] == 0 else 1
