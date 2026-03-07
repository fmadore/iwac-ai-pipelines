#!/usr/bin/env python3
"""
Reconcile AI-generated keywords against IWAC authority records.

Reads the latest items_enriched_*.csv and reconciles Subject AI and Spatial AI
columns against authority item sets, producing:
  - *_reconciled.csv           — enriched CSV with reconciled IDs added
  - *_unreconciled_spatial.csv — unmatched spatial terms (for review / creation)
  - *_unreconciled_subject.csv — unmatched subject terms (for review / creation)
  - *_potential_reconciliation_*.csv — fuzzy-match suggestions

Usage:
    python 03_reconcile_metadata.py
"""
from __future__ import annotations

import os
import sys
from typing import Dict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from common.omeka_client import OmekaClient  # noqa: E402
from common.reconciliation import (  # noqa: E402
    build_authority_dict,
    reconcile_column_values,
    create_potential_reconciliation_csv,
    write_ambiguous_terms_to_file,
    display_authority_stats,
    display_reconciliation_stats,
    MULTI_WORD_MIN_SIMILARITY,
    DEFAULT_MAX_CANDIDATES,
)

# Authority item set IDs (same as NER pipeline)
SPATIAL_AUTHORITY_ITEM_SETS = ["268"]
SUBJECT_AUTHORITY_ITEM_SETS = ["854", "2", "266"]
TOPIC_AUTHORITY_ITEM_SETS = ["1"]

# Column names
SPATIAL_COLUMN = "Spatial AI"
SUBJECT_COLUMN = "Subject AI"
SPATIAL_OUTPUT_COLUMN = "Spatial AI Reconciled ID"
SUBJECT_OUTPUT_COLUMN = "Subject AI Reconciled ID"

TAG_SPATIAL = "spatial"
TAG_SUBJECT = "subject"

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def find_latest_enriched_csv() -> str | None:
    """Find the most recent items_enriched_*.csv in the output directory."""
    try:
        candidates = [
            f for f in os.listdir(OUTPUT_DIR)
            if f.startswith("items_enriched_") and f.endswith(".csv")
            and "_reconciled" not in f
            and "_unreconciled" not in f
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)))
    except OSError:
        return None


def main():
    console.print(Panel(
        "[bold]Reference Indexing — Step 3[/bold]\n"
        "Reconcile AI keywords against IWAC authority records",
        title="Reconcile Keywords",
        border_style="cyan",
    ))

    client = OmekaClient.from_env()

    # Find input
    latest = find_latest_enriched_csv()
    if not latest:
        console.print("[red]✗[/] No items_enriched_*.csv found in output/")
        return

    input_path = os.path.join(OUTPUT_DIR, latest)
    base, ext = os.path.splitext(input_path)
    reconciled_path = f"{base}_reconciled{ext}"

    config_table = Table(title="Configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Input file", latest)
    config_table.add_row("Output file", os.path.basename(reconciled_path))
    config_table.add_row("Spatial item sets", ", ".join(SPATIAL_AUTHORITY_ITEM_SETS))
    config_table.add_row("Subject item sets", ", ".join(SUBJECT_AUTHORITY_ITEM_SETS + TOPIC_AUTHORITY_ITEM_SETS))
    console.print(config_table)
    console.print()

    # --- Spatial reconciliation ---
    console.rule("[bold cyan]Step 1: Spatial Reconciliation")

    spatial_dict, spatial_ambiguous, spatial_metadata = build_authority_dict(
        client, SPATIAL_AUTHORITY_ITEM_SETS, "SPATIAL"
    )
    display_authority_stats(spatial_dict, spatial_ambiguous, "Spatial")

    ambig_spatial_path = f"{base}_ambiguous_authorities_{TAG_SPATIAL}.csv"
    write_ambiguous_terms_to_file(spatial_ambiguous, ambig_spatial_path)

    output_after_spatial, s_matched, s_total, s_unrec = reconcile_column_values(
        input_csv_path=input_path,
        output_reconciled_csv_path=reconciled_path,
        authority_dict=spatial_dict,
        source_column_name=SPATIAL_COLUMN,
        target_column_name=SPATIAL_OUTPUT_COLUMN,
        initial_csv_base_for_unreconciled=base,
        output_file_tag=TAG_SPATIAL,
        ambiguous_authority_dict=spatial_ambiguous,
    )
    display_reconciliation_stats(s_matched, s_total, s_unrec, "Spatial")
    console.print()

    # --- Subject reconciliation (subject + topic authority sets combined) ---
    console.rule("[bold cyan]Step 2: Subject Reconciliation")

    console.print("[dim]Building subject authorities...[/]")
    subject_dict, subject_ambiguous, subject_metadata = build_authority_dict(
        client, SUBJECT_AUTHORITY_ITEM_SETS, "SUBJECT"
    )
    display_authority_stats(subject_dict, subject_ambiguous, "Subject")

    console.print("\n[dim]Building topic authorities...[/]")
    topic_dict, topic_ambiguous, topic_metadata = build_authority_dict(
        client, TOPIC_AUTHORITY_ITEM_SETS, "TOPIC"
    )
    display_authority_stats(topic_dict, topic_ambiguous, "Topic")

    combined_dict = {**subject_dict, **topic_dict}
    combined_ambiguous = {**subject_ambiguous, **topic_ambiguous}
    combined_metadata = {**subject_metadata, **topic_metadata}

    console.print(f"\n[dim]Combined: {len(combined_dict)} terms, {len(combined_ambiguous)} ambiguous[/]")

    ambig_subject_path = f"{base}_ambiguous_authorities_{TAG_SUBJECT}.csv"
    write_ambiguous_terms_to_file(combined_ambiguous, ambig_subject_path)

    _, subj_matched, subj_total, subj_unrec = reconcile_column_values(
        input_csv_path=output_after_spatial,
        output_reconciled_csv_path=reconciled_path,
        authority_dict=combined_dict,
        source_column_name=SUBJECT_COLUMN,
        target_column_name=SUBJECT_OUTPUT_COLUMN,
        initial_csv_base_for_unreconciled=base,
        output_file_tag=TAG_SUBJECT,
        ambiguous_authority_dict=combined_ambiguous,
    )
    display_reconciliation_stats(subj_matched, subj_total, subj_unrec, "Subject")
    console.print()

    # --- Potential matches ---
    console.rule("[bold cyan]Step 3: Generate Potential Matches")

    spatial_unrec_path = f"{base}_unreconciled_{TAG_SPATIAL}.csv"
    if os.path.exists(spatial_unrec_path):
        console.print("\n[dim]Processing spatial unreconciled values...[/]")
        create_potential_reconciliation_csv(
            unreconciled_csv_path=spatial_unrec_path,
            authority_metadata=spatial_metadata,
            output_csv_path=f"{base}_potential_reconciliation_{TAG_SPATIAL}.csv",
            min_similarity=MULTI_WORD_MIN_SIMILARITY,
            max_candidates_per_value=DEFAULT_MAX_CANDIDATES,
        )

    subject_unrec_path = f"{base}_unreconciled_{TAG_SUBJECT}.csv"
    if os.path.exists(subject_unrec_path):
        console.print("\n[dim]Processing subject unreconciled values...[/]")
        create_potential_reconciliation_csv(
            unreconciled_csv_path=subject_unrec_path,
            authority_metadata=combined_metadata,
            output_csv_path=f"{base}_potential_reconciliation_{TAG_SUBJECT}.csv",
            min_similarity=MULTI_WORD_MIN_SIMILARITY,
            max_candidates_per_value=DEFAULT_MAX_CANDIDATES,
        )

    console.print()
    console.print(Panel(
        f"[green]✓[/] Reconciliation complete!\n\n"
        f"Main output: [cyan]{os.path.basename(reconciled_path)}[/]",
        title="Step 3 Complete",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
