"""
Shared reconciliation utilities for matching AI-generated terms against
Omeka S authority records.

Extracted from AI_NER/02_NER_reconciliation_Omeka.py so that any pipeline
(NER, metadata enrichment, etc.) can reuse the same fuzzy-matching logic.

Usage:
    from common.reconciliation import (
        build_authority_dict,
        reconcile_column_values,
        create_potential_reconciliation_csv,
        display_authority_stats,
        display_reconciliation_stats,
        write_ambiguous_terms_to_file,
    )
"""

import csv
import os
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich import box

from common.omeka_client import OmekaClient

console = Console()

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

BASE_MIN_SIMILARITY = 0.80
MULTI_WORD_MIN_SIMILARITY = 0.88
STRONG_MATCH_THRESHOLD = 0.92
MIN_TOKEN_OVERLAP = 0.5
DEFAULT_MAX_CANDIDATES = 1

GENERIC_TERMS = {
    'islam', 'religion', 'communauté', 'communaute', 'musulmans', 'musulmane',
    'musulman', 'fête', 'fete', 'ville', 'region', 'pays', 'organisation',
    'organization', 'dialogue', 'fraternité', 'fraternite', 'cohésion',
    'cohesion', 'sacrifice', 'transport', 'formation', 'financement',
    'développement', 'developpement', 'paix', 'projet', 'association',
}

STOPWORDS = {
    'de', 'du', 'des', 'la', 'le', 'les', 'et', 'au', 'aux', 'd', 'l',
    'en', 'pour', 'par', 'sur', 'avec', 'a', 'the', 'of', 'et', 'dans',
    'au', 'aux', 'un', 'une', 'pour', 'vers', 'islamique', 'islamic',
    'association', 'centre', 'organisation', 'organization', 'fondation',
    'groupe', 'union', 'reseau', 'réseau', 'societe', 'société',
    'gouvernement', 'parti', 'club',
}

MATCH_TYPE_PRIMARY = "primary_title"
MATCH_TYPE_ALTERNATIVE = "alternative"
AMBIGUOUS_MARKER = "(Ambiguous)"

# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------


def _strip_diacritics(text: str) -> str:
    """Remove accents/diacritics while preserving base characters."""
    return ''.join(
        ch for ch in unicodedata.normalize('NFD', text)
        if unicodedata.category(ch) != 'Mn'
    )


def normalize_location_name(name: str) -> str:
    """Normalize a name to a compact comparison key (lowercase, no accents,
    no spaces/dashes)."""
    if not name:
        return ''
    text = name.strip().lower()
    replacements = {
        '\u2018': "'", '\u00b4': "'", '\u2019': "'",
        '\u2010': '-', '\u2011': '-', '\u2012': '-',
        '\u2013': '-', '\u2014': '-', '\u2015': '-',
    }
    text = ''.join(replacements.get(c, c) for c in text)
    text = ' '.join(text.split())
    text_no_diacritics = _strip_diacritics(text)
    return text_no_diacritics.replace('-', '').replace(' ', '')


# ---------------------------------------------------------------------------
# Authority dictionary builder
# ---------------------------------------------------------------------------


def build_authority_dict(
    client: OmekaClient,
    item_set_ids: List[str],
    authority_type: str = "authority",
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, Dict]]:
    """Build a lookup dict of authority terms from specified item sets.

    Returns:
        (authority_dict, ambiguous_terms_dict, authority_metadata)
    """
    potential_lookups: List[tuple] = []
    authority_metadata: Dict[str, Dict] = {}

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Building {authority_type} authority dictionary...", total=None,
        )

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
                metadata = {'primary_title': '', 'alternatives': []}
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

    # Resolve ambiguities
    name_to_ids_map: Dict[str, set] = defaultdict(set)
    for name, item_id_val in potential_lookups:
        name_to_ids_map[name].add(item_id_val)

    authority_dict_final: Dict[str, str] = {}
    ambiguous_terms_dict: Dict[str, List[str]] = {}

    for name, ids_set in name_to_ids_map.items():
        if len(ids_set) == 1:
            authority_dict_final[name] = list(ids_set)[0]
        else:
            ambiguous_terms_dict[name] = sorted(list(ids_set))

    return authority_dict_final, ambiguous_terms_dict, authority_metadata


# ---------------------------------------------------------------------------
# Similarity / fuzzy matching
# ---------------------------------------------------------------------------


def calculate_similarity(s1: str, s2: str) -> float:
    """Return a conservative similarity score (0..1) between two strings."""
    s1_lower, s2_lower = s1.lower().strip(), s2.lower().strip()
    if not s1_lower or not s2_lower:
        return 0.0
    if s1_lower == s2_lower:
        return 1.0

    s1_norm = normalize_location_name(s1)
    s2_norm = normalize_location_name(s2)
    if s1_norm and s1_norm == s2_norm:
        return STRONG_MATCH_THRESHOLD

    token_pattern = re.compile(r"\b\w+\b", re.UNICODE)
    s1_tokens = [t for t in token_pattern.findall(_strip_diacritics(s1_lower)) if t not in STOPWORDS]
    s2_tokens = [t for t in token_pattern.findall(_strip_diacritics(s2_lower)) if t not in STOPWORDS]
    s1_token_set, s2_token_set = set(s1_tokens), set(s2_tokens)
    common_tokens = s1_token_set & s2_token_set
    all_tokens = s1_token_set | s2_token_set

    if len(all_tokens) >= 2:
        if not common_tokens:
            return 0.0
        token_overlap_ratio = len(common_tokens) / len(all_tokens)
    else:
        token_overlap_ratio = 1.0 if common_tokens else 0.0

    char_similarity = SequenceMatcher(None, s1_lower, s2_lower).ratio()
    norm_similarity = SequenceMatcher(None, s1_norm, s2_norm).ratio() if (s1_norm and s2_norm) else 0.0
    max_char_sim = max(char_similarity, norm_similarity)

    if len(all_tokens) >= 2 and token_overlap_ratio < MIN_TOKEN_OVERLAP:
        return 0.0
    if len(s1_lower) <= 4 or len(s2_lower) <= 4:
        return max_char_sim if max_char_sim >= 0.95 else 0.0
    if s1_tokens and s2_tokens and s1_token_set == s2_token_set and s1_tokens[::-1] == s2_tokens:
        return max(STRONG_MATCH_THRESHOLD, max_char_sim)
    if max_char_sim < 0.85:
        return 0.0

    return (max_char_sim * 0.7) + (token_overlap_ratio * 0.3)


def find_potential_matches(
    unreconciled_value: str,
    authority_metadata: Dict[str, Dict],
    min_similarity: float = BASE_MIN_SIMILARITY,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> List[Tuple[str, str, float, str]]:
    """Find fuzzy-match candidates for an unreconciled value.

    Returns list of (item_id, matched_name, score, match_type).
    """
    candidates = []
    value_clean = unreconciled_value.lower().strip()

    if value_clean in GENERIC_TERMS:
        min_similarity = max(min_similarity, 0.97)
    if len(value_clean.split()) >= 2:
        min_similarity = max(min_similarity, MULTI_WORD_MIN_SIMILARITY)

    for item_id, metadata in authority_metadata.items():
        primary = metadata.get('primary_title')
        if primary:
            sim = calculate_similarity(unreconciled_value, primary)
            if sim >= min_similarity:
                candidates.append((item_id, primary, sim, MATCH_TYPE_PRIMARY))
        for alt in metadata.get('alternatives', [])[:50]:
            sim = calculate_similarity(unreconciled_value, alt)
            if sim >= min_similarity:
                candidates.append((item_id, alt, sim, MATCH_TYPE_ALTERNATIVE))

    # De-duplicate by item_id keeping best variant
    best_by_item: Dict[str, tuple] = {}
    for item_id, name, score, kind in candidates:
        current = best_by_item.get(item_id)
        if current is None or score > current[2]:
            best_by_item[item_id] = (item_id, name, score, kind)

    deduped = sorted(best_by_item.values(), key=lambda x: x[2], reverse=True)

    if value_clean in GENERIC_TERMS and all(c[2] < STRONG_MATCH_THRESHOLD for c in deduped):
        return []

    if deduped:
        top_score = deduped[0][2]
        band = 0.02 if top_score >= STRONG_MATCH_THRESHOLD else 0.03
        deduped = [c for c in deduped if (top_score - c[2]) <= band]

    return deduped[:max_candidates]


# ---------------------------------------------------------------------------
# CSV reconciliation
# ---------------------------------------------------------------------------


def create_potential_reconciliation_csv(
    unreconciled_csv_path: str,
    authority_metadata: Dict[str, Dict],
    output_csv_path: str,
    min_similarity: float = BASE_MIN_SIMILARITY,
    max_candidates_per_value: int = DEFAULT_MAX_CANDIDATES,
):
    """Create CSV with fuzzy-match suggestions for unreconciled values."""
    if not os.path.exists(unreconciled_csv_path):
        console.print(f"[yellow]⚠[/] Unreconciled CSV not found: {unreconciled_csv_path}")
        return

    potential_matches = []

    try:
        with open(unreconciled_csv_path, 'r', encoding='utf-8') as f:
            row_count = sum(1 for _ in csv.DictReader(f))

        with open(unreconciled_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                BarColumn(), TaskProgressColumn(), TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("[cyan]Finding potential matches...", total=row_count)

                for row in reader:
                    value = row['Unreconciled Value']
                    count = row['Count']

                    if AMBIGUOUS_MARKER in value:
                        progress.update(task, advance=1)
                        continue

                    matches = find_potential_matches(
                        value, authority_metadata,
                        min_similarity=min_similarity,
                        max_candidates=max_candidates_per_value,
                    )
                    for item_id, matched_name, similarity, match_type in matches:
                        potential_matches.append({
                            'Unreconciled Value': value,
                            'Count': count,
                            'Potential Match': matched_name,
                            'Item ID': item_id,
                            'Similarity Score': f"{similarity:.3f}",
                            'Match Type': match_type,
                            'Primary Title': authority_metadata[item_id]['primary_title'],
                            'All Alternatives': ' | '.join(authority_metadata[item_id]['alternatives']) or '',
                        })
                    progress.update(task, advance=1)

        if potential_matches:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'Unreconciled Value', 'Count', 'Potential Match', 'Item ID',
                    'Similarity Score', 'Match Type', 'Primary Title', 'All Alternatives',
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(potential_matches)
            console.print(f"[green]✓[/] Potential reconciliation candidates written to: {os.path.basename(output_csv_path)}")
            console.print(f"  Found {len(potential_matches)} potential matches")
        else:
            console.print(f"[dim]No potential matches found above similarity threshold {min_similarity}[/]")

    except Exception as e:
        console.print(f"[red]✗[/] Error creating potential reconciliation CSV: {e}")


def reconcile_column_values(
    input_csv_path: str,
    output_reconciled_csv_path: str,
    authority_dict: Dict[str, str],
    source_column_name: str,
    target_column_name: str,
    initial_csv_base_for_unreconciled: str,
    output_file_tag: str,
    ambiguous_authority_dict: Dict[str, List[str]],
) -> Tuple[str, int, int, int]:
    """Reconcile pipe-separated values in *source_column_name* against
    *authority_dict*, writing reconciled IDs to *target_column_name*.

    Returns (output_path, matched_count, total_values, unreconciled_count).
    """
    unreconciled_path = f"{initial_csv_base_for_unreconciled}_unreconciled_{output_file_tag}.csv"

    rows_processed = []
    unreconciled_counts: Counter = Counter()
    matched_count = 0
    total_values = 0
    ambiguous_warnings = []

    try:
        with open(input_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                console.print(f"[red]✗[/] CSV file is empty or header is missing: {input_csv_path}")
                return output_reconciled_csv_path, 0, 0, 0
            row_count = sum(1 for _ in f)

        with open(input_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            original_fieldnames = list(reader.fieldnames)
            output_fieldnames = list(original_fieldnames)
            if target_column_name not in output_fieldnames:
                output_fieldnames.append(target_column_name)

            if source_column_name not in original_fieldnames:
                console.print(f"[yellow]⚠[/] Source column '{source_column_name}' not found. Skipping reconciliation.")
                for row in reader:
                    processed = row.copy()
                    if target_column_name not in processed:
                        processed[target_column_name] = ""
                    rows_processed.append(processed)
            else:
                with Progress(
                    SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                    BarColumn(), TaskProgressColumn(), TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(f"[cyan]Reconciling {source_column_name}...", total=row_count)

                    for row in reader:
                        processed = row.copy()
                        reconciled_str = processed.get(target_column_name, "")
                        source_value = processed.get(source_column_name)

                        if source_value:
                            individual = source_value.split('|')
                            total_values += len(individual)
                            reconciled_ids = []

                            for value in individual:
                                v = value.strip()
                                if not v:
                                    continue
                                v_lower = v.lower()
                                v_norm = normalize_location_name(v)
                                v_diacfree = _strip_diacritics(v_lower)

                                if (v_lower in ambiguous_authority_dict
                                        or v_norm in ambiguous_authority_dict
                                        or v_diacfree in ambiguous_authority_dict):
                                    ambiguous_warnings.append(v)
                                    unreconciled_counts[f"{v} {AMBIGUOUS_MARKER}"] += 1
                                    continue

                                found = False
                                for variant in (v_lower, v_norm, v_diacfree):
                                    if variant in authority_dict:
                                        reconciled_ids.append(authority_dict[variant])
                                        matched_count += 1
                                        found = True
                                        break
                                if not found:
                                    unreconciled_counts[v] += 1

                            reconciled_str = '|'.join(reconciled_ids)

                        processed[target_column_name] = reconciled_str
                        rows_processed.append(processed)
                        progress.update(task, advance=1)

    except FileNotFoundError:
        console.print(f"[red]✗[/] Input CSV file not found: {input_csv_path}")
        return output_reconciled_csv_path, 0, 0, 0
    except Exception as e:
        console.print(f"[red]✗[/] Error reading CSV: {e}")
        raise

    # Write reconciled CSV
    with open(output_reconciled_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(rows_processed)

    # Write unreconciled values
    if unreconciled_counts:
        with open(unreconciled_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Unreconciled Value', 'Count'])
            for value, count in sorted(unreconciled_counts.items(), key=lambda x: x[1], reverse=True):
                writer.writerow([value, count])

    if ambiguous_warnings:
        unique = set(ambiguous_warnings)
        console.print(f"[yellow]⚠[/] {len(unique)} ambiguous terms skipped (see unreconciled file)")

    return output_reconciled_csv_path, matched_count, total_values, len(unreconciled_counts)


# ---------------------------------------------------------------------------
# Ambiguous terms output
# ---------------------------------------------------------------------------


def write_ambiguous_terms_to_file(ambiguous_dict: Dict[str, List[str]], output_path: str):
    """Write ambiguous terms and their associated item IDs to a CSV file."""
    if not ambiguous_dict:
        return
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Ambiguous Term', 'Item IDs'])
            for term, ids in sorted(ambiguous_dict.items()):
                writer.writerow([term, '|'.join(ids)])
        console.print(f"[green]✓[/] Ambiguous terms written to: {os.path.basename(output_path)}")
    except Exception as e:
        console.print(f"[red]✗[/] Error writing ambiguous terms CSV: {e}")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def display_authority_stats(authority_dict: Dict, ambiguous_dict: Dict, authority_type: str):
    """Display statistics for an authority dictionary."""
    stats_table = Table(box=box.ROUNDED, show_header=False)
    stats_table.add_column("Metric", style="dim")
    stats_table.add_column("Value", style="cyan")
    stats_table.add_row("Authority type", authority_type)
    stats_table.add_row("Unique terms", str(len(authority_dict)))
    stats_table.add_row("Ambiguous terms", str(len(ambiguous_dict)))
    console.print(stats_table)


def display_reconciliation_stats(matched: int, total: int, unreconciled: int, column_name: str):
    """Display reconciliation statistics."""
    match_rate = (matched / total * 100) if total > 0 else 0
    stats_table = Table(title=f"📊 {column_name} Reconciliation Results", box=box.ROUNDED)
    stats_table.add_column("Metric", style="dim")
    stats_table.add_column("Value", justify="right")
    stats_table.add_row("Total values processed", str(total))
    stats_table.add_row("Matched with authorities", f"[green]{matched}[/]")
    stats_table.add_row("Unreconciled unique values", f"[yellow]{unreconciled}[/]")
    stats_table.add_row("Match rate", f"[cyan]{match_rate:.1f}%[/]")
    console.print(stats_table)
