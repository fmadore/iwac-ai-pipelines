#!/usr/bin/env python3
"""
Omeka S Named Entity Recognition (NER) Reconciliation Script

This script reconciles AI-generated named entities (spatial locations, subjects, topics) 
with existing authority records in an Omeka S digital library system. It performs fuzzy 
matching to identify exact matches and generates potential reconciliation candidates 
for unmatched values.

Key Features:
- Reconciles spatial entities against location authorities
- Reconciles subject/topic entities against controlled vocabularies
- Identifies ambiguous terms that map to multiple authorities
- Generates potential reconciliation candidates using fuzzy matching
- Handles name reversals and spelling variants
- Creates detailed CSV reports for manual review

Prerequisites:
- .env file with OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL
- Input CSV file with AI-generated NER results
- Omeka S instance with authority item sets configured

CONFIGURATION:
=============

1. Environment Variables (.env file):
   OMEKA_BASE_URL=https://your-omeka-instance.com/api
   OMEKA_KEY_IDENTITY=your_key_identity
   OMEKA_KEY_CREDENTIAL=your_key_credential

2. Omeka S Item Set Configuration:
   Configure SPATIAL_AUTHORITY_ITEM_SETS, SUBJECT_AUTHORITY_ITEM_SETS, 
   and TOPIC_AUTHORITY_ITEM_SETS constants to match your setup.

3. Input CSV Format:
   Required columns:
   - 'Spatial AI': Pipe-separated spatial entities (e.g., "Paris|London|Berlin")
   - 'Subject AI': Pipe-separated subject/topic entities (e.g., "Religion|Politics")

4. Output Directory:
   Place input CSV files in ./output/ directory relative to script location.
   The script processes the most recently created CSV file that doesn't contain
   '_reconciled', '_unreconciled', or '_ambiguous_authorities' in the filename.

USAGE:
======
python 02_NER_reconciliation_Omeka.py

The script will automatically:
1. Find the latest input CSV file
2. Build authority dictionaries from Omeka S
3. Perform reconciliation 
4. Generate output files with reconciled IDs and potential matches

Author: [Your name]
Date: August 2025
"""

import os
import sys
from typing import Dict, List, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Initialize rich console
console = Console()

# Shared Omeka client
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.omeka_client import OmekaClient
from common.reconciliation import (
    build_authority_dict,
    reconcile_column_values,
    create_potential_reconciliation_csv,
    write_ambiguous_terms_to_file,
    display_authority_stats,
    display_reconciliation_stats,
    MULTI_WORD_MIN_SIMILARITY,
    DEFAULT_MAX_CANDIDATES,
    AMBIGUOUS_MARKER,
)

# Pipeline-specific item set IDs
SPATIAL_AUTHORITY_ITEM_SETS = ["268"]
SUBJECT_AUTHORITY_ITEM_SETS = ["854", "2", "266"]
TOPIC_AUTHORITY_ITEM_SETS = ["1"]

# Column names in input CSV
SPATIAL_COLUMN = "Spatial AI"
SUBJECT_COLUMN = "Subject AI"
SPATIAL_OUTPUT_COLUMN = "Spatial AI Reconciled ID"
SUBJECT_OUTPUT_COLUMN = "Subject AI Reconciled ID"

# File naming tags
TAG_SPATIAL = "spatial"
TAG_SUBJECT_AND_TOPIC = "subject_and_topic"


def find_input_csv(output_dir: str) -> str | None:
    """Find the most recent unprocessed CSV file in the output directory."""
    try:
        csv_files = [
            f for f in os.listdir(output_dir) 
            if f.endswith('.csv') 
            and '_reconciled' not in f 
            and '_unreconciled' not in f 
            and '_ambiguous_authorities' not in f
            and '_potential_reconciliation' not in f
        ]
        if not csv_files:
            return None
        return max(csv_files, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
    except OSError:
        return None


def run_spatial_reconciliation(
    client: OmekaClient,
    initial_csv_path: str,
    initial_csv_base: str,
    final_reconciled_csv_path: str
) -> Tuple[str, Dict[str, Dict[str, str]]]:
    """Run spatial entity reconciliation and return the output path and metadata."""
    console.rule("[bold cyan]Step 1: Spatial Reconciliation")

    # Build spatial authority dictionary
    spatial_authority_dict, spatial_ambiguous_dict, spatial_metadata = build_authority_dict(
        client, SPATIAL_AUTHORITY_ITEM_SETS, "SPATIAL"
    )
    display_authority_stats(spatial_authority_dict, spatial_ambiguous_dict, "Spatial/Location")
    
    # Write ambiguous spatial terms
    ambiguous_spatial_path = f"{initial_csv_base}_ambiguous_authorities_{TAG_SPATIAL}.csv"
    write_ambiguous_terms_to_file(spatial_ambiguous_dict, ambiguous_spatial_path)
    
    # Perform spatial reconciliation
    output_path, matched, total, unreconciled = reconcile_column_values(
        input_csv_path=initial_csv_path,
        output_reconciled_csv_path=final_reconciled_csv_path,
        authority_dict=spatial_authority_dict,
        source_column_name=SPATIAL_COLUMN,
        target_column_name=SPATIAL_OUTPUT_COLUMN,
        initial_csv_base_for_unreconciled=initial_csv_base,
        output_file_tag=TAG_SPATIAL,
        ambiguous_authority_dict=spatial_ambiguous_dict
    )
    
    display_reconciliation_stats(matched, total, unreconciled, "Spatial")
    return output_path, spatial_metadata


def run_subject_topic_reconciliation(
    client: OmekaClient,
    current_input_path: str,
    initial_csv_base: str,
    final_reconciled_csv_path: str
) -> Tuple[str, Dict[str, Dict[str, str]]]:
    """Run combined subject and topic entity reconciliation."""
    console.rule("[bold cyan]Step 2: Subject & Topic Reconciliation")

    # Build subject authority dictionary
    console.print("\n[dim]Building subject authorities...[/]")
    subject_authority_dict, subject_ambiguous_dict, subject_metadata = build_authority_dict(
        client, SUBJECT_AUTHORITY_ITEM_SETS, "SUBJECT"
    )
    display_authority_stats(subject_authority_dict, subject_ambiguous_dict, "Subject")

    # Build topic authority dictionary
    console.print("\n[dim]Building topic authorities...[/]")
    topic_authority_dict, topic_ambiguous_dict, topic_metadata = build_authority_dict(
        client, TOPIC_AUTHORITY_ITEM_SETS, "TOPIC"
    )
    display_authority_stats(topic_authority_dict, topic_ambiguous_dict, "Topic")
    
    # Combine dictionaries
    combined_authority_dict = {**subject_authority_dict, **topic_authority_dict}
    combined_ambiguous_dict = {**subject_ambiguous_dict, **topic_ambiguous_dict}
    combined_metadata = {**subject_metadata, **topic_metadata}
    
    console.print(f"\n[dim]Combined: {len(combined_authority_dict)} terms, {len(combined_ambiguous_dict)} ambiguous[/]")
    
    # Write combined ambiguous terms
    ambiguous_subject_path = f"{initial_csv_base}_ambiguous_authorities_{TAG_SUBJECT_AND_TOPIC}.csv"
    write_ambiguous_terms_to_file(combined_ambiguous_dict, ambiguous_subject_path)
    
    # Perform subject/topic reconciliation
    output_path, matched, total, unreconciled = reconcile_column_values(
        input_csv_path=current_input_path,
        output_reconciled_csv_path=final_reconciled_csv_path,
        authority_dict=combined_authority_dict,
        source_column_name=SUBJECT_COLUMN,
        target_column_name=SUBJECT_OUTPUT_COLUMN,
        initial_csv_base_for_unreconciled=initial_csv_base,
        output_file_tag=TAG_SUBJECT_AND_TOPIC,
        ambiguous_authority_dict=combined_ambiguous_dict
    )
    
    display_reconciliation_stats(matched, total, unreconciled, "Subject & Topic")
    return output_path, combined_metadata


def generate_potential_matches(
    initial_csv_base: str,
    spatial_metadata: Dict[str, Dict[str, str]],
    combined_metadata: Dict[str, Dict[str, str]]
):
    """Generate potential reconciliation candidates for unreconciled values."""
    console.rule("[bold cyan]Step 3: Generate Potential Matches")
    
    # Spatial potential matches
    spatial_unreconciled_path = f"{initial_csv_base}_unreconciled_{TAG_SPATIAL}.csv"
    spatial_potential_path = f"{initial_csv_base}_potential_reconciliation_{TAG_SPATIAL}.csv"
    if os.path.exists(spatial_unreconciled_path):
        console.print("\n[dim]Processing spatial unreconciled values...[/]")
        create_potential_reconciliation_csv(
            unreconciled_csv_path=spatial_unreconciled_path,
            authority_metadata=spatial_metadata,
            output_csv_path=spatial_potential_path,
            min_similarity=MULTI_WORD_MIN_SIMILARITY,
            max_candidates_per_value=DEFAULT_MAX_CANDIDATES
        )
    else:
        console.print("[dim]No spatial unreconciled values to process[/]")
    
    # Subject/Topic potential matches
    subject_topic_unreconciled_path = f"{initial_csv_base}_unreconciled_{TAG_SUBJECT_AND_TOPIC}.csv"
    subject_topic_potential_path = f"{initial_csv_base}_potential_reconciliation_{TAG_SUBJECT_AND_TOPIC}.csv"
    if os.path.exists(subject_topic_unreconciled_path):
        console.print("\n[dim]Processing subject/topic unreconciled values...[/]")
        create_potential_reconciliation_csv(
            unreconciled_csv_path=subject_topic_unreconciled_path,
            authority_metadata=combined_metadata,
            output_csv_path=subject_topic_potential_path,
            min_similarity=MULTI_WORD_MIN_SIMILARITY,
            max_candidates_per_value=DEFAULT_MAX_CANDIDATES
        )
    else:
        console.print("[dim]No subject/topic unreconciled values to process[/]")

def main():
    """
    Main reconciliation workflow for Omeka S Named Entity Recognition results.
    
    This function orchestrates the complete reconciliation process:
    1. Finds the latest unprocessed CSV file in the output directory
    2. Builds authority dictionaries from configured Omeka item sets
    3. Reconciles spatial entities against location authorities
    4. Reconciles subject/topic entities against controlled vocabularies  
    5. Generates potential reconciliation candidates for unmatched values
    
    Input Requirements:
        - CSV file with columns defined by SPATIAL_COLUMN and SUBJECT_COLUMN constants
        - .env file with Omeka S API credentials
        - Configured item sets via SPATIAL_AUTHORITY_ITEM_SETS, SUBJECT_AUTHORITY_ITEM_SETS,
          and TOPIC_AUTHORITY_ITEM_SETS constants
    
    Output Files:
        - *_reconciled.csv: Main file with reconciled entity IDs
        - *_unreconciled_spatial.csv: Unmatched spatial entities
        - *_unreconciled_subject_and_topic.csv: Unmatched subject/topic entities
        - *_ambiguous_authorities_*.csv: Terms matching multiple authorities
        - *_potential_reconciliation_*.csv: Suggested matches for manual review
    """
    try:
        # Display welcome banner
        console.print(Panel(
            "[bold]NER Reconciliation Pipeline[/bold]\n"
            "Reconciles AI-generated entities with Omeka S authority records",
            title="🔗 Omeka S NER Reconciliation",
            border_style="cyan"
        ))

        # Initialize shared Omeka client
        client = OmekaClient.from_env()

        # Setup and file discovery
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "output")

        latest_csv_filename = find_input_csv(output_dir)
        if not latest_csv_filename:
            console.print("[red]✗[/] No suitable CSV files found in output directory.")
            return
            
        initial_csv_path = os.path.join(output_dir, latest_csv_filename)
        initial_csv_base, initial_csv_ext = os.path.splitext(initial_csv_path)
        final_reconciled_csv_path = f"{initial_csv_base}_reconciled{initial_csv_ext}"

        # Display configuration
        config_table = Table(title="📁 File Configuration", box=box.ROUNDED)
        config_table.add_column("Setting", style="dim")
        config_table.add_column("Value", style="green")
        config_table.add_row("Input file", latest_csv_filename)
        config_table.add_row("Output file", os.path.basename(final_reconciled_csv_path))
        config_table.add_row("Spatial item sets", ", ".join(SPATIAL_AUTHORITY_ITEM_SETS))
        config_table.add_row("Subject item sets", ", ".join(SUBJECT_AUTHORITY_ITEM_SETS))
        config_table.add_row("Topic item sets", ", ".join(TOPIC_AUTHORITY_ITEM_SETS))
        console.print(config_table)
        console.print()

        # Step 1: Spatial Reconciliation
        reconciled_after_spatial_path, spatial_metadata = run_spatial_reconciliation(
            client, initial_csv_path, initial_csv_base, final_reconciled_csv_path
        )
        console.print()

        # Step 2: Subject and Topic Reconciliation
        final_reconciled_path, combined_metadata = run_subject_topic_reconciliation(
            client, reconciled_after_spatial_path, initial_csv_base, final_reconciled_csv_path
        )
        console.print()

        # Step 3: Generate Potential Reconciliation Candidates
        generate_potential_matches(initial_csv_base, spatial_metadata, combined_metadata)
        
        # Final summary
        console.print()
        console.print(Panel(
            f"[green]✓[/] Reconciliation complete!\n\n"
            f"Main output: [cyan]{os.path.basename(final_reconciled_path)}[/]",
            title="✨ Process Complete",
            border_style="green"
        ))
        
    except ValueError as val_err:
        console.print(f"[red]✗[/] Configuration error: {val_err}")
    except FileNotFoundError as fnf_err:
        console.print(f"[red]✗[/] File not found: {fnf_err}")
    except Exception as e:
        console.print(f"[red]✗[/] Unexpected error: {e}")
        console.print_exception()

if __name__ == "__main__":
    main()
