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

import csv
import os
import sys
import re
import unicodedata
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich import box

# Initialize rich console
console = Console()

# Shared Omeka client
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.omeka_client import OmekaClient

"""------------------------------------------------------------------------------
CONFIGURATION CONSTANTS (Tunable)
Adjust thresholds & behaviour here instead of in-line magic numbers.
------------------------------------------------------------------------------"""
# Modify these item set IDs to match your Omeka S configuration
SPATIAL_AUTHORITY_ITEM_SETS = ["268"]  # Location/place authorities
SUBJECT_AUTHORITY_ITEM_SETS = ["854", "2", "266"]  # Subject heading authorities  
TOPIC_AUTHORITY_ITEM_SETS = ["1"]  # Topic/theme authorities

# Base minimum similarity for proposing suggestions (used as lower bound)
BASE_MIN_SIMILARITY = 0.80
# For multi‑word values we usually want higher confidence
MULTI_WORD_MIN_SIMILARITY = 0.88
# Strong match threshold (SequenceMatcher or normalized) for auto keeping
STRONG_MATCH_THRESHOLD = 0.92
# When using token Jaccard overlap, require at least this ratio (after stopword filtering)
MIN_TOKEN_OVERLAP = 0.5
# Maximum number of candidate suggestions per unreconciled value (can be overridden)
DEFAULT_MAX_CANDIDATES = 1

# Generic / overly broad terms for which we DO NOT generate fuzzy suggestions unless exact
GENERIC_TERMS = {
    'islam','religion','communauté','communaute','musulmans','musulmane','musulman',
    'fête','fete','ville','region','pays','organisation','organization','dialogue',
    'fraternité','fraternite','cohésion','cohesion','sacrifice','transport','formation',
    'financement','développement','developpement','paix','projet','association'
}

# Stopwords (French + generic organisational terms) removed before token overlap tests
STOPWORDS = {
    'de','du','des','la','le','les','et','au','aux','d','l','en','pour','par','sur','avec',
    'a','the','of','et','dans','au','aux','un','une','pour','vers','islamique','islamic',
    'association','association','centre','centre','organisation','organization','fondation',
    'groupe','union','reseau','réseau','societe','société','gouvernement','parti','club'
}

# Column names in input CSV
SPATIAL_COLUMN = "Spatial AI"
SUBJECT_COLUMN = "Subject AI"
SPATIAL_OUTPUT_COLUMN = "Spatial AI Reconciled ID"
SUBJECT_OUTPUT_COLUMN = "Subject AI Reconciled ID"

# Match type constants
MATCH_TYPE_PRIMARY = "primary_title"
MATCH_TYPE_ALTERNATIVE = "alternative"

# Ambiguous marker suffix
AMBIGUOUS_MARKER = "(Ambiguous)"

# File naming tags
TAG_SPATIAL = "spatial"
TAG_SUBJECT_AND_TOPIC = "subject_and_topic"

# API pagination
API_PER_PAGE = 100


def _strip_diacritics(text: str) -> str:
    """
    Remove accents/diacritics while preserving base characters.
    
    Example: 'café' → 'cafe', 'naïve' → 'naive'
    
    Args:
        text: Input text with potential diacritics
        
    Returns:
        Text with diacritics removed
    """
    return ''.join(ch for ch in unicodedata.normalize('NFD', text) if unicodedata.category(ch) != 'Mn')

def normalize_location_name(name: str) -> str:
    """
    Normalize location names for better matching by removing variations in 
    case, spacing, punctuation, and accents.

    Processing steps:
      1. Trim whitespace
      2. Convert to lowercase
      3. Standardize apostrophes/quotes/dashes to common forms
      4. Remove diacritics/accents
      5. Collapse multiple whitespace to single spaces
      6. Remove all spaces and dashes for compact comparison key
    
    Args:
        name: Original location name
        
    Returns:
        Normalized compact string for matching
        
    Example:
        "Saint-André-de-Cubzac" → "saintandredecubzac"
        "N'Djamena" → "ndjamena"
    """
    if not name:
        return ''
    text = name.strip().lower()
    # Standardize common apostrophes / quotes / dashes
    replacements = {
        '’': "'",
        '´': "'",
        '‘': "'",
        '‐': '-',  # hyphen variants
        '‑': '-',
        '‒': '-',
        '–': '-',
        '—': '-',
        '―': '-',
    }
    text = ''.join(replacements.get(c, c) for c in text)
    # Collapse multiple whitespace
    text = ' '.join(text.split())
    # Remove diacritics AFTER standardization
    text_no_diacritics = _strip_diacritics(text)
    # Build compact key (remove spaces & dashes)
    compact = text_no_diacritics.replace('-', '').replace(' ', '')
    return compact

def build_authority_dict(client: OmekaClient, item_set_ids: List[str], authority_type: str = "authority") -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, Dict[str, str]]]:
    """Build dictionary of authority terms from specified item sets and identify ambiguous terms.

    Args:
        client: OmekaClient instance
        item_set_ids: List of Omeka item set IDs to fetch authorities from
        authority_type: Label for progress display (e.g., "SPATIAL", "SUBJECT")

    Returns:
        - authority_dict: mapping from normalized names to item IDs
        - ambiguous_terms_dict: mapping from names to multiple item IDs
        - authority_metadata: mapping from item IDs to their metadata (original titles, alternatives)
    """

    potential_lookups: List[tuple[str, str]] = []
    authority_metadata: Dict[str, Dict[str, str]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]Building {authority_type} authority dictionary...", total=None)

        for item_set_id in item_set_ids:
            progress.update(task, description=f"[cyan]Fetching from item set {item_set_id}...")

            try:
                all_items = client.get_items(int(item_set_id))
            except Exception as e:
                console.print(f"[red]✗[/] Error fetching item set {item_set_id}: {e}")
                continue

            if not all_items:
                continue

            for item in all_items:
                    item_id = str(item['o:id'])
                    
                    metadata = {
                        'primary_title': '',
                        'alternatives': []
                    }
                    
                    titles_to_process = []
                    if 'dcterms:title' in item:
                        for title_obj in item['dcterms:title']:
                            title = title_obj['@value'].strip()
                            if title:
                                titles_to_process.append(title)
                                if not metadata['primary_title']:
                                    metadata['primary_title'] = title
                    
                    if 'dcterms:alternative' in item:
                        for alt in item['dcterms:alternative']:
                            alt_value = alt['@value'].strip()
                            if alt_value:
                                titles_to_process.append(alt_value)
                                metadata['alternatives'].append(alt_value)
                    
                    authority_metadata[item_id] = metadata

                    for title_text in titles_to_process:
                        lower_variant = title_text.lower()
                        normalized_title = normalize_location_name(title_text)
                        diacritic_stripped_spaced = _strip_diacritics(lower_variant)
                        for variant in {lower_variant, normalized_title, diacritic_stripped_spaced}:
                            if variant:
                                potential_lookups.append((variant, item_id))

    # Process lookups to identify authorities and ambiguities
    name_to_ids_map = defaultdict(set)
    for name, item_id_val in potential_lookups:
        name_to_ids_map[name].add(item_id_val)
            
    authority_dict_final = {}
    ambiguous_terms_dict = {}
    
    for name, ids_set in name_to_ids_map.items():
        if len(ids_set) == 1:
            authority_dict_final[name] = list(ids_set)[0]
        else:
            ambiguous_terms_dict[name] = sorted(list(ids_set))
    
    return authority_dict_final, ambiguous_terms_dict, authority_metadata

def calculate_similarity(s1: str, s2: str) -> float:
    """Return a conservative similarity score (0..1) between two strings.

    The function now emphasises token overlap & strict thresholds to avoid
    spurious high scores from SequenceMatcher alone.
    """
    s1_lower, s2_lower = s1.lower().strip(), s2.lower().strip()
    if not s1_lower or not s2_lower:
        return 0.0

    # Exact match
    if s1_lower == s2_lower:
        return 1.0

    # Normalized equality (diacritics / punctuation removed)
    s1_norm = normalize_location_name(s1)
    s2_norm = normalize_location_name(s2)
    if s1_norm and s1_norm == s2_norm:
        return STRONG_MATCH_THRESHOLD  # treat as strong but allow > with exact

    # Token based analysis (retain originals for ratio but use cleaned tokens)
    token_pattern = re.compile(r"\b\w+\b", re.UNICODE)
    s1_tokens = [t for t in token_pattern.findall(_strip_diacritics(s1_lower)) if t not in STOPWORDS]
    s2_tokens = [t for t in token_pattern.findall(_strip_diacritics(s2_lower)) if t not in STOPWORDS]

    s1_token_set = set(s1_tokens)
    s2_token_set = set(s2_tokens)
    common_tokens = s1_token_set & s2_token_set
    all_tokens = s1_token_set | s2_token_set

    # If both multi-word after stopword removal, require minimal overlap for any score > 0
    if len(all_tokens) >= 2:
        if not common_tokens:
            return 0.0  # no token overlap => reject
        token_overlap_ratio = len(common_tokens) / len(all_tokens)
    else:
        token_overlap_ratio = 1.0 if common_tokens else 0.0

    # Compute a conservative character similarity
    char_similarity = SequenceMatcher(None, s1_lower, s2_lower).ratio()
    norm_similarity = SequenceMatcher(None, s1_norm, s2_norm).ratio() if (s1_norm and s2_norm) else 0.0
    max_char_sim = max(char_similarity, norm_similarity)

    # Penalise if token overlap is low
    if len(all_tokens) >= 2 and token_overlap_ratio < MIN_TOKEN_OVERLAP:
        return 0.0

    # Short strings: require extremely high similarity
    if len(s1_lower) <= 4 or len(s2_lower) <= 4:
        return max_char_sim if max_char_sim >= 0.95 else 0.0

    # Reversed tokens (e.g., "john smith" vs "smith john")
    if s1_tokens and s2_tokens and s1_token_set == s2_token_set and s1_tokens[::-1] == s2_tokens:
        return max(STRONG_MATCH_THRESHOLD, max_char_sim)

    # Weighted combination: emphasise token overlap first, then char similarity
    if max_char_sim < 0.85:
        return 0.0

    # Blend with token overlap (gives slight boost when both high)
    blended = (max_char_sim * 0.7) + (token_overlap_ratio * 0.3)
    return blended

def find_potential_matches(unreconciled_value: str, 
                          authority_metadata: Dict[str, Dict[str, str]], 
                          min_similarity: float = BASE_MIN_SIMILARITY,
                          max_candidates: int = DEFAULT_MAX_CANDIDATES) -> List[Tuple[str, str, float, str]]:
    """
    Find potential reconciliation matches for an unreconciled value using fuzzy matching.
    
    This function searches through authority metadata to find similar names that could
    represent the same entity. It applies conservative filtering to avoid false positives.
    
    Args:
        unreconciled_value: The value that couldn't be automatically reconciled
        authority_metadata: Dictionary mapping item IDs to their titles and alternatives
        min_similarity: Minimum similarity threshold (0.0-1.0)
        max_candidates: Maximum number of candidates to return
    
    Returns:
        List of tuples containing:
        - item_id: Omeka item ID of potential match
        - matched_name: The specific name variant that matched
        - similarity_score: Confidence score (0.0-1.0)
        - match_type: 'primary_title' or 'alternative'
        
    Special handling:
        - Generic terms (islam, religion, etc.) require 95%+ similarity
        - Multiple candidates are filtered to 85%+ similarity
        - Results sorted by similarity score (highest first)
    """
    candidates = []

    value_clean = unreconciled_value.lower().strip()
    # Block suggestions for generic / broad terms unless an almost exact canonical exists
    if value_clean in GENERIC_TERMS:
        min_similarity = max(min_similarity, 0.97)

    # Adaptive threshold: multi-word => higher baseline
    if len(value_clean.split()) >= 2:
        min_similarity = max(min_similarity, MULTI_WORD_MIN_SIMILARITY)

    for item_id, metadata in authority_metadata.items():
        primary = metadata.get('primary_title')
        if primary:
            sim_primary = calculate_similarity(unreconciled_value, primary)
            if sim_primary >= min_similarity:
                candidates.append((item_id, primary, sim_primary, MATCH_TYPE_PRIMARY))
        for alt in metadata.get('alternatives', [])[:50]:  # safety slice
            sim_alt = calculate_similarity(unreconciled_value, alt)
            if sim_alt >= min_similarity:
                candidates.append((item_id, alt, sim_alt, MATCH_TYPE_ALTERNATIVE))

    # De-duplicate by item_id keeping best variant
    best_by_item = {}
    for item_id, name, score, kind in candidates:
        current = best_by_item.get(item_id)
        if (current is None) or (score > current[2]):
            best_by_item[item_id] = (item_id, name, score, kind)

    deduped = list(best_by_item.values())
    deduped.sort(key=lambda x: x[2], reverse=True)

    # If everything is below strong threshold and term is generic, drop all
    if value_clean in GENERIC_TERMS and all(c[2] < STRONG_MATCH_THRESHOLD for c in deduped):
        return []

    # Final pruning: enforce a narrow score band around top result to avoid noise
    if deduped:
        top_score = deduped[0][2]
        band = 0.02 if top_score >= STRONG_MATCH_THRESHOLD else 0.03
        deduped = [c for c in deduped if (top_score - c[2]) <= band]

    return deduped[:max_candidates]

def create_potential_reconciliation_csv(unreconciled_csv_path: str,
                                      authority_metadata: Dict[str, Dict[str, str]],
                                      output_csv_path: str,
                                      min_similarity: float = BASE_MIN_SIMILARITY,
                                      max_candidates_per_value: int = DEFAULT_MAX_CANDIDATES):
    """
    Create a CSV file with potential reconciliation candidates for unreconciled values.
    
    This function processes unreconciled entities and finds similar authority records
    that could represent the same concepts. The output helps librarians identify
    spelling variants or alternative forms that should be added to authority files.
    
    Args:
        unreconciled_csv_path: Path to CSV containing unreconciled values and counts
        authority_metadata: Dictionary of authority metadata (from build_authority_dict)
        output_csv_path: Where to save the potential matches CSV
        min_similarity: Minimum similarity for suggestions (0.0-1.0)
        max_candidates_per_value: Maximum suggestions per unreconciled value
    
    Output CSV columns:
        - Unreconciled Value: Original unmatched value
        - Count: How many times it appeared in source data
        - Potential Match: Authority name that might match
        - Item ID: Omeka item ID of potential authority
        - Similarity Score: Confidence level (0.000-1.000)
        - Match Type: 'primary_title' or 'alternative'
        - Primary Title: Main title of the authority record
        - All Alternatives: Complete list of alternative titles
        
    Example output row:
        Hamdallaye,8,Hamdalaye,329,0.947,primary_title,Hamdalaye,
    """
    if not os.path.exists(unreconciled_csv_path):
        console.print(f"[yellow]⚠[/] Unreconciled CSV file not found: {unreconciled_csv_path}")
        return
    
    potential_matches = []
    
    try:
        # Count rows first for progress bar
        with open(unreconciled_csv_path, 'r', encoding='utf-8') as csvfile:
            row_count = sum(1 for _ in csv.DictReader(csvfile))
        
        with open(unreconciled_csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Finding potential matches...", total=row_count)
                
                for row in reader:
                    unreconciled_value = row['Unreconciled Value']
                    count = row['Count']
                    
                    # Skip ambiguous values
                    if AMBIGUOUS_MARKER in unreconciled_value:
                        progress.update(task, advance=1)
                        continue
                    
                    matches = find_potential_matches(
                        unreconciled_value,
                        authority_metadata,
                        min_similarity=min_similarity,
                        max_candidates=max_candidates_per_value
                    )
                    
                    for item_id, matched_name, similarity, match_type in matches:
                        potential_matches.append({
                            'Unreconciled Value': unreconciled_value,
                            'Count': count,
                            'Potential Match': matched_name,
                            'Item ID': item_id,
                            'Similarity Score': f"{similarity:.3f}",
                            'Match Type': match_type,
                            'Primary Title': authority_metadata[item_id]['primary_title'],
                            'All Alternatives': ' | '.join(authority_metadata[item_id]['alternatives']) if authority_metadata[item_id]['alternatives'] else ''
                        })
                    
                    progress.update(task, advance=1)
        
        # Write potential matches to CSV
        if potential_matches:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'Unreconciled Value', 'Count', 'Potential Match', 'Item ID', 
                    'Similarity Score', 'Match Type', 'Primary Title', 'All Alternatives'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(potential_matches)
            
            console.print(f"[green]✓[/] Potential reconciliation candidates written to: {os.path.basename(output_csv_path)}")
            console.print(f"  Found {len(potential_matches)} potential matches")
        else:
            console.print(f"[dim]No potential matches found above similarity threshold {min_similarity}[/]")
            
    except Exception as e:
        console.print(f"[red]✗[/] Error creating potential reconciliation CSV: {e}")

def reconcile_column_values(input_csv_path: str, 
                            output_reconciled_csv_path: str, 
                            authority_dict: Dict[str, str], 
                            source_column_name: str, 
                            target_column_name: str, 
                            initial_csv_base_for_unreconciled: str, 
                            output_file_tag: str,
                            ambiguous_authority_dict: Dict[str, List[str]]) -> Tuple[str, int, int, int]:
    """Reconcile values in a specific CSV column, writing to a designated output CSV. 
       Unreconciled values are saved separately, named based on the initial input file and a tag.
       Ambiguous values are not reconciled and are logged.
       
    Returns:
        Tuple of (output_path, matched_count, total_values, unreconciled_count)
    """
    unreconciled_path = f"{initial_csv_base_for_unreconciled}_unreconciled_{output_file_tag}.csv"
    
    rows_processed = []
    unreconciled_counts = Counter()
    matched_count = 0
    total_values_in_source_col = 0
    rows_with_source_col = 0
    ambiguous_warnings = []
    
    try:
        # Count rows first for progress bar
        with open(input_csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames:
                console.print(f"[red]✗[/] CSV file is empty or header is missing: {input_csv_path}")
                return output_reconciled_csv_path, 0, 0, 0
            row_count = sum(1 for _ in csvfile)
        
        with open(input_csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            original_fieldnames = list(reader.fieldnames)
            output_fieldnames = list(original_fieldnames)
            if target_column_name not in output_fieldnames:
                output_fieldnames.append(target_column_name)

            if source_column_name not in original_fieldnames:
                console.print(f"[yellow]⚠[/] Source column '{source_column_name}' not found. Skipping reconciliation.")
                for row in reader:
                    processed_row = row.copy()
                    if target_column_name not in processed_row:
                        processed_row[target_column_name] = ""
                    rows_processed.append(processed_row)
            else:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task(f"[cyan]Reconciling {source_column_name}...", total=row_count)
                    
                    for row in reader:
                        processed_row = row.copy()
                        reconciled_values_str = processed_row.get(target_column_name, "")
                        source_value = processed_row.get(source_column_name)

                        if source_value: 
                            rows_with_source_col += 1
                            individual_values = source_value.split('|')
                            total_values_in_source_col += len(individual_values)
                            reconciled_ids = []

                            for value in individual_values:
                                value_stripped = value.strip()
                                if not value_stripped: 
                                    continue 
                                
                                value_lower = value_stripped.lower()
                                value_normalized = normalize_location_name(value_stripped)
                                value_diacritic_free = _strip_diacritics(value_lower)
                                
                                # Check if the term is ambiguous first
                                if (value_lower in ambiguous_authority_dict or
                                    value_normalized in ambiguous_authority_dict or
                                    value_diacritic_free in ambiguous_authority_dict):
                                    ambiguous_warnings.append(value_stripped)
                                    unreconciled_counts[f"{value_stripped} {AMBIGUOUS_MARKER}"] += 1
                                    continue

                                found_match = False
                                if value_lower in authority_dict:
                                    reconciled_id = authority_dict[value_lower]
                                    reconciled_ids.append(reconciled_id) 
                                    matched_count += 1
                                    found_match = True
                                elif value_normalized in authority_dict:
                                    reconciled_id = authority_dict[value_normalized]
                                    reconciled_ids.append(reconciled_id) 
                                    matched_count += 1
                                    found_match = True
                                elif value_diacritic_free in authority_dict:
                                    reconciled_id = authority_dict[value_diacritic_free]
                                    reconciled_ids.append(reconciled_id)
                                    matched_count += 1
                                    found_match = True
                                    
                                if not found_match:
                                    unreconciled_counts[value_stripped] += 1
                                    
                            reconciled_values_str = '|'.join(reconciled_ids)
                        
                        processed_row[target_column_name] = reconciled_values_str
                        rows_processed.append(processed_row)
                        progress.update(task, advance=1)
    
    except FileNotFoundError:
        console.print(f"[red]✗[/] Input CSV file not found: {input_csv_path}")
        return output_reconciled_csv_path, 0, 0, 0
    except Exception as e:
        console.print(f"[red]✗[/] Error reading CSV: {e}")
        raise 

    # Write reconciled data
    try:
        with open(output_reconciled_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=output_fieldnames)
            writer.writeheader()
            writer.writerows(rows_processed)
    except Exception as e:
        console.print(f"[red]✗[/] Error writing reconciled CSV: {e}")
        raise

    # Write unreconciled values
    if unreconciled_counts:
        try:
            with open(unreconciled_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Unreconciled Value', 'Count'])
                sorted_unreconciled = sorted(unreconciled_counts.items(), key=lambda item: item[1], reverse=True)
                for value, count in sorted_unreconciled:
                    writer.writerow([value, count])
        except Exception as e:
            console.print(f"[red]✗[/] Error writing unreconciled CSV: {e}")

    # Log ambiguous term warnings (summarized)
    if ambiguous_warnings:
        unique_ambiguous = set(ambiguous_warnings)
        console.print(f"[yellow]⚠[/] {len(unique_ambiguous)} ambiguous terms skipped (see unreconciled file)")
    
    return output_reconciled_csv_path, matched_count, total_values_in_source_col, len(unreconciled_counts)

def write_ambiguous_terms_to_file(ambiguous_dict: Dict[str, List[str]], output_path: str):
    """Writes ambiguous terms and their associated item IDs to a CSV file."""
    if not ambiguous_dict:
        return

    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile_obj:
            writer = csv.writer(csvfile_obj)
            writer.writerow(['Ambiguous Term', 'Item IDs'])
            for term, ids in sorted(ambiguous_dict.items()):
                writer.writerow([term, '|'.join(ids)])
        console.print(f"[green]✓[/] Ambiguous terms written to: {os.path.basename(output_path)}")
    except Exception as e:
        console.print(f"[red]✗[/] Error writing ambiguous terms CSV: {e}")


def display_authority_stats(authority_dict: Dict, ambiguous_dict: Dict, authority_type: str):
    """Display statistics for an authority dictionary in a formatted table."""
    stats_table = Table(box=box.ROUNDED, show_header=False)
    stats_table.add_column("Metric", style="dim")
    stats_table.add_column("Value", style="cyan")
    stats_table.add_row("Authority type", authority_type)
    stats_table.add_row("Unique terms", str(len(authority_dict)))
    stats_table.add_row("Ambiguous terms", str(len(ambiguous_dict)))
    console.print(stats_table)


def display_reconciliation_stats(matched: int, total: int, unreconciled: int, column_name: str):
    """Display reconciliation statistics in a formatted table."""
    match_rate = (matched / total * 100) if total > 0 else 0
    
    stats_table = Table(title=f"📊 {column_name} Reconciliation Results", box=box.ROUNDED)
    stats_table.add_column("Metric", style="dim")
    stats_table.add_column("Value", justify="right")
    stats_table.add_row("Total values processed", str(total))
    stats_table.add_row("Matched with authorities", f"[green]{matched}[/]")
    stats_table.add_row("Unreconciled unique values", f"[yellow]{unreconciled}[/]")
    stats_table.add_row("Match rate", f"[cyan]{match_rate:.1f}%[/]")
    console.print(stats_table)


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
