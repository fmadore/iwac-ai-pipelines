"""The reconciliation step shared by NER and reference indexing.

Both pipelines hand a CSV with ``Spatial AI`` and ``Subject AI`` columns to
the same three-stage run: spatial terms against the location authorities,
subject terms against the subject and topic authorities built together (so a
term that maps to different items in each set is reported as ambiguous rather
than silently resolved), then fuzzy candidates for whatever did not match.
The matching itself lives in ``common/reconciliation.py``; this module is the
run around it.

NER runs with ``extended_matching``: word-order keys on both columns, the
plural fold on subjects other than persons (names are proper nouns: "Lago", a
village in Zondoma, must not become "Lagos", nor "Prophètes" the Prophet),
subject terms that name a place
linked as ``dcterms:spatial``, and titles qualified by the run's country
("Ministère de la Santé (Burkina Faso)") also matched bare. The country comes
from the NER checkpoint beside the input. Reference indexing does not, so its matching is
unchanged until those rules have been checked on its own output.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from rich.console import Console
from rich.panel import Panel

from common.checkpoint import read_checkpoint_context
from common.console_utils import key_value_table
from common.iwac_config import (
    PERSON_AUTHORITY_ITEM_SET,
    SPATIAL_AUTHORITY_ITEM_SETS,
    SUBJECT_AUTHORITY_ITEM_SETS,
    TOPIC_AUTHORITY_ITEM_SETS,
)
from common.omeka_client import OmekaClient
from common.reconciliation import (
    DEFAULT_MAX_CANDIDATES,
    MULTI_WORD_MIN_SIMILARITY,
    Redirect,
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


@dataclass
class AuthorityDicts:
    """The two lookups a NER run reconciles against, with their metadata."""

    spatial: Dict[str, str]
    spatial_ambiguous: Dict[str, List[str]]
    spatial_metadata: Dict[str, Dict]
    subject: Dict[str, str]
    subject_ambiguous: Dict[str, List[str]]
    subject_metadata: Dict[str, Dict]


def build_dictionaries(
    client: OmekaClient, *, extended_matching: bool = False, country: Optional[str] = None,
) -> AuthorityDicts:
    """Spatial, then subject + topic in ONE call so ambiguity spans both sets.

    Shared by the reconciliation run and ``AI_NER/02b``, so the context check
    sees exactly the matches reconciliation made.
    """
    spatial, spatial_ambiguous, spatial_metadata = build_authority_dict(
        client, SPATIAL_AUTHORITY_ITEM_SETS, "SPATIAL", token_order=extended_matching,
        country=country,
    )
    subject_sets = SUBJECT_AUTHORITY_ITEM_SETS + TOPIC_AUTHORITY_ITEM_SETS
    subject, subject_ambiguous, subject_metadata = build_authority_dict(
        client, subject_sets, "SUBJECT+TOPIC",
        token_order=extended_matching,
        plural_fold_sets=(
            [s for s in subject_sets if s != PERSON_AUTHORITY_ITEM_SET]
            if extended_matching else ()
        ),
        country=country,
    )
    return AuthorityDicts(spatial, spatial_ambiguous, spatial_metadata,
                          subject, subject_ambiguous, subject_metadata)


def checkpoint_country(input_path: Path) -> Optional[str]:
    """The country a NER run was scoped to, from the checkpoint beside it."""
    context = read_checkpoint_context(Path(input_path))
    return context.get("country") or context.get("spatial_filter") or None


def run_reconciliation(
    client: OmekaClient,
    input_path: Path,
    *,
    subject_tag: str,
    console: Optional[Console] = None,
    extended_matching: bool = False,
    country: Optional[str] = None,
) -> Path:
    """Reconcile one CSV and write the derived files beside it.

    *subject_tag* names the subject-side files (``_unreconciled_<tag>.csv`` …);
    NER uses ``subject_and_topic``, reference indexing ``subject``, because
    their step-4 tooling looks for those names. *extended_matching* is
    described in the module docstring.

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
        ("Country qualifier", country or "none"),
    ]))
    console.print()

    dicts = build_dictionaries(client, extended_matching=extended_matching, country=country)
    spatial_dict, spatial_ambiguous, spatial_metadata = dicts.spatial, dicts.spatial_ambiguous, dicts.spatial_metadata

    # --- Spatial ---------------------------------------------------------
    console.rule("[bold cyan]Step 1: Spatial Reconciliation")
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
    combined_dict, combined_ambiguous, combined_metadata = dicts.subject, dicts.subject_ambiguous, dicts.subject_metadata
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
        redirect=(
            Redirect(spatial_dict, spatial_ambiguous, SPATIAL_OUTPUT_COLUMN)
            if extended_matching else None
        ),
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
    extended_matching: bool = False,
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

    country = checkpoint_country(Path(input_path)) if extended_matching else None
    try:
        run_reconciliation(
            client, Path(input_path), subject_tag=subject_tag, console=console,
            extended_matching=extended_matching, country=country,
        )
    except RuntimeError as exc:
        console.print(f"[red]✗[/] {exc}")
        return 1
    return 0
