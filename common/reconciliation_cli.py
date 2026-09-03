"""The reconciliation step shared by NER and reference indexing.

Both pipelines hand a CSV with ``Spatial AI`` and ``Subject AI`` columns to
the same three-stage run: spatial terms against the location authorities,
subject terms against the subject and topic authorities built together (so a
term that maps to different items in each set is reported as ambiguous rather
than silently resolved), then fuzzy candidates for whatever did not match.
The matching itself lives in ``common/reconciliation.py``; this module is the
run around it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional, Sequence

from rich.console import Console
from rich.panel import Panel

from common.console_utils import key_value_table
from common.iwac_config import (
    SPATIAL_AUTHORITY_ITEM_SETS,
    SUBJECT_AUTHORITY_ITEM_SETS,
    TOPIC_AUTHORITY_ITEM_SETS,
)
from common.omeka_client import OmekaClient
from common.reconciliation import (
    DEFAULT_MAX_CANDIDATES,
    MULTI_WORD_MIN_SIMILARITY,
    build_authority_dict,
    create_potential_reconciliation_csv,
    display_authority_stats,
    display_reconciliation_stats,
    reconcile_column_values,
    write_ambiguous_terms_to_file,
)

SPATIAL_COLUMN = "Spatial AI"
SUBJECT_COLUMN = "Subject AI"
SPATIAL_OUTPUT_COLUMN = "Spatial AI Reconciled ID"
SUBJECT_OUTPUT_COLUMN = "Subject AI Reconciled ID"
TAG_SPATIAL = "spatial"

#: Suffixes the run itself writes, which must never be picked up as input.
DERIVED_MARKERS = ("_reconciled", "_unreconciled", "_ambiguous_authorities", "_potential_reconciliation")


def find_input_csv(output_dir: Path, *, accept: Callable[[str], bool] = lambda name: True) -> Optional[Path]:
    """The newest CSV in *output_dir* that is an AI output, not a derived file."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None
    candidates = [
        path for path in output_dir.glob("*.csv")
        if not any(marker in path.name for marker in DERIVED_MARKERS) and accept(path.name)
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def run_reconciliation(
    client: OmekaClient,
    input_path: Path,
    *,
    subject_tag: str,
    console: Optional[Console] = None,
) -> Path:
    """Reconcile one CSV and write the derived files beside it.

    *subject_tag* names the subject-side files (``_unreconciled_<tag>.csv`` …);
    NER uses ``subject_and_topic``, reference indexing ``subject``, because
    their step-4 tooling looks for those names.

    Returns the path of the main ``*_reconciled.csv``.
    """
    console = console or Console()
    input_path = Path(input_path)
    base = str(input_path.with_suffix(""))
    reconciled_path = Path(f"{base}_reconciled{input_path.suffix}")

    console.print(key_value_table([
        ("Input file", input_path.name),
        ("Output file", reconciled_path.name),
        ("Spatial item sets", ", ".join(SPATIAL_AUTHORITY_ITEM_SETS)),
        ("Subject item sets", ", ".join(SUBJECT_AUTHORITY_ITEM_SETS)),
        ("Topic item sets", ", ".join(TOPIC_AUTHORITY_ITEM_SETS)),
    ]))
    console.print()

    # --- Spatial ---------------------------------------------------------
    console.rule("[bold cyan]Step 1: Spatial Reconciliation")
    spatial_dict, spatial_ambiguous, spatial_metadata = build_authority_dict(
        client, SPATIAL_AUTHORITY_ITEM_SETS, "SPATIAL"
    )
    display_authority_stats(spatial_dict, spatial_ambiguous, "Spatial/Location")
    write_ambiguous_terms_to_file(spatial_ambiguous, f"{base}_ambiguous_authorities_{TAG_SPATIAL}.csv")
    after_spatial, matched, total, unreconciled = reconcile_column_values(
        input_csv_path=str(input_path),
        output_reconciled_csv_path=str(reconciled_path),
        authority_dict=spatial_dict,
        source_column_name=SPATIAL_COLUMN,
        target_column_name=SPATIAL_OUTPUT_COLUMN,
        initial_csv_base_for_unreconciled=base,
        output_file_tag=TAG_SPATIAL,
        ambiguous_authority_dict=spatial_ambiguous,
    )
    display_reconciliation_stats(matched, total, unreconciled, "Spatial")
    console.print()

    # --- Subject + topic, built in ONE call so ambiguity spans both sets ----
    console.rule("[bold cyan]Step 2: Subject & Topic Reconciliation")
    combined_dict, combined_ambiguous, combined_metadata = build_authority_dict(
        client, SUBJECT_AUTHORITY_ITEM_SETS + TOPIC_AUTHORITY_ITEM_SETS, "SUBJECT+TOPIC"
    )
    display_authority_stats(combined_dict, combined_ambiguous, "Subject + Topic")
    write_ambiguous_terms_to_file(combined_ambiguous, f"{base}_ambiguous_authorities_{subject_tag}.csv")
    _, matched, total, unreconciled = reconcile_column_values(
        input_csv_path=after_spatial,
        output_reconciled_csv_path=str(reconciled_path),
        authority_dict=combined_dict,
        source_column_name=SUBJECT_COLUMN,
        target_column_name=SUBJECT_OUTPUT_COLUMN,
        initial_csv_base_for_unreconciled=base,
        output_file_tag=subject_tag,
        ambiguous_authority_dict=combined_ambiguous,
    )
    display_reconciliation_stats(matched, total, unreconciled, "Subject & Topic")
    console.print()

    # --- Fuzzy candidates for what is left ---------------------------------
    console.rule("[bold cyan]Step 3: Generate Potential Matches")
    for tag, metadata in ((TAG_SPATIAL, spatial_metadata), (subject_tag, combined_metadata)):
        unreconciled_path = Path(f"{base}_unreconciled_{tag}.csv")
        if not unreconciled_path.exists():
            console.print(f"[dim]No {tag} unreconciled values to process[/]")
            continue
        console.print(f"\n[dim]Processing {tag} unreconciled values...[/]")
        create_potential_reconciliation_csv(
            unreconciled_csv_path=str(unreconciled_path),
            authority_metadata=metadata,
            output_csv_path=f"{base}_potential_reconciliation_{tag}.csv",
            min_similarity=MULTI_WORD_MIN_SIMILARITY,
            max_candidates_per_value=DEFAULT_MAX_CANDIDATES,
        )

    console.print()
    console.print(Panel(
        f"[green]✓[/] Reconciliation complete!\n\nMain output: [cyan]{reconciled_path.name}[/]",
        title="Reconciliation Complete",
        border_style="green",
    ))
    return reconciled_path


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--input", type=Path,
        help="AI output CSV to reconcile (default: the newest one in output/).",
    )
    return parser


def main(
    argv: Optional[Sequence[str]],
    *,
    output_dir: Path,
    subject_tag: str,
    banner: str,
    description: str,
    accept: Callable[[str], bool] = lambda name: True,
) -> int:
    """Argument parsing, input discovery and the run, for both entry points."""
    console = Console()
    args = build_parser(description).parse_args(argv)
    console.print(Panel(banner, title="Reconcile Keywords", border_style="cyan"))

    try:
        client = OmekaClient.from_env()
    except ValueError as exc:
        console.print(f"[red]✗[/] {exc}")
        return 1

    input_path = args.input or find_input_csv(output_dir, accept=accept)
    if input_path is None or not Path(input_path).is_file():
        console.print(f"[red]✗[/] No suitable CSV found in {output_dir}")
        return 1

    run_reconciliation(client, Path(input_path), subject_tag=subject_tag, console=console)
    return 0
