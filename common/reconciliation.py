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
import sys
import unicodedata
from dataclasses import dataclass, field

# Increase CSV field size limit to handle large bibo:content fields
csv.field_size_limit(10 * 1024 * 1024)
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table
from rich import box

from common.omeka_client import OmekaClient
from common.console_utils import standard_progress

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

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
    'en', 'pour', 'par', 'sur', 'avec', 'a', 'the', 'of', 'dans',
    'un', 'une', 'vers', 'islamique', 'islamic',
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


def _authority_titles(item: Dict) -> Tuple[Dict, List[str]]:
    """Extract literal primary/alternative titles from one authority item."""
    metadata = {'primary_title': '', 'alternatives': []}
    titles: List[str] = []
    for title_obj in item.get('dcterms:title', []):
        title = str(title_obj.get('@value') or '').strip()
        if title:
            titles.append(title)
            if not metadata['primary_title']:
                metadata['primary_title'] = title
    for alternative in item.get('dcterms:alternative', []):
        value = str(alternative.get('@value') or '').strip()
        if value:
            titles.append(value)
            metadata['alternatives'].append(value)
    return metadata, titles


def _authority_variants(title: str) -> set[str]:
    lower = title.lower()
    return {
        variant for variant in (
            lower,
            normalize_location_name(title),
            _strip_diacritics(lower),
        ) if variant
    }


def _resolve_authority_lookups(
    potential_lookups: List[tuple],
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    name_to_ids: Dict[str, set] = defaultdict(set)
    for name, item_id in potential_lookups:
        name_to_ids[name].add(item_id)
    authority: Dict[str, str] = {}
    ambiguous: Dict[str, List[str]] = {}
    for name, item_ids in name_to_ids.items():
        if len(item_ids) == 1:
            authority[name] = next(iter(item_ids))
        else:
            ambiguous[name] = sorted(item_ids)
    return authority, ambiguous


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

    with standard_progress(console) as progress:
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
                metadata, titles_to_process = _authority_titles(item)
                authority_metadata[item_id] = metadata
                for title_text in titles_to_process:
                    potential_lookups.extend(
                        (variant, item_id) for variant in _authority_variants(title_text)
                    )

    authority, ambiguous = _resolve_authority_lookups(potential_lookups)
    return authority, ambiguous, authority_metadata


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


def _similarity_threshold(value: str, base: float) -> float:
    threshold = max(base, 0.97) if value in GENERIC_TERMS else base
    return max(threshold, MULTI_WORD_MIN_SIMILARITY) if len(value.split()) >= 2 else threshold


def _candidate_matches(
    value: str, authority_metadata: Dict[str, Dict], threshold: float
) -> List[Tuple[str, str, float, str]]:
    candidates: List[Tuple[str, str, float, str]] = []
    for item_id, metadata in authority_metadata.items():
        variants = [(metadata.get('primary_title'), MATCH_TYPE_PRIMARY)]
        variants.extend(
            (alternative, MATCH_TYPE_ALTERNATIVE)
            for alternative in metadata.get('alternatives', [])[:50]
        )
        for name, match_type in variants:
            if not name:
                continue
            score = calculate_similarity(value, name)
            if score >= threshold:
                candidates.append((item_id, name, score, match_type))
    return candidates


def _best_candidates_by_item(
    candidates: List[Tuple[str, str, float, str]],
) -> List[Tuple[str, str, float, str]]:
    best: Dict[str, Tuple[str, str, float, str]] = {}
    for candidate in candidates:
        item_id = candidate[0]
        if item_id not in best or candidate[2] > best[item_id][2]:
            best[item_id] = candidate
    return sorted(best.values(), key=lambda candidate: candidate[2], reverse=True)


def find_potential_matches(
    unreconciled_value: str,
    authority_metadata: Dict[str, Dict],
    min_similarity: float = BASE_MIN_SIMILARITY,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> List[Tuple[str, str, float, str]]:
    """Find fuzzy-match candidates for an unreconciled value.

    Returns list of (item_id, matched_name, score, match_type).
    """
    value_clean = unreconciled_value.lower().strip()
    threshold = _similarity_threshold(value_clean, min_similarity)
    deduped = _best_candidates_by_item(
        _candidate_matches(unreconciled_value, authority_metadata, threshold)
    )

    if value_clean in GENERIC_TERMS and all(c[2] < STRONG_MATCH_THRESHOLD for c in deduped):
        return []

    if deduped:
        top_score = deduped[0][2]
        band = 0.02 if top_score >= STRONG_MATCH_THRESHOLD else 0.03
        deduped = [candidate for candidate in deduped if top_score - candidate[2] <= band]

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

            with standard_progress(console) as progress:
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
        raise


@dataclass
class ReconciliationState:
    matched_count: int = 0
    total_values: int = 0
    unreconciled_counts: Counter = field(default_factory=Counter)
    ambiguous_terms: set[str] = field(default_factory=set)


def _value_variants(value: str) -> Tuple[str, str, str]:
    lower = value.lower()
    return lower, normalize_location_name(value), _strip_diacritics(lower)


def _reconcile_value(
    value: str,
    authority_dict: Dict[str, str],
    ambiguous_authority_dict: Dict[str, List[str]],
    state: ReconciliationState,
) -> Optional[str]:
    variants = _value_variants(value)
    if any(variant in ambiguous_authority_dict for variant in variants):
        state.ambiguous_terms.add(value)
        state.unreconciled_counts[f"{value} {AMBIGUOUS_MARKER}"] += 1
        return None
    for variant in variants:
        if variant in authority_dict:
            state.matched_count += 1
            return authority_dict[variant]
    state.unreconciled_counts[value] += 1
    return None


def _reconcile_row(
    row: Dict[str, str],
    source_column_name: str,
    target_column_name: str,
    authority_dict: Dict[str, str],
    ambiguous_authority_dict: Dict[str, List[str]],
    state: ReconciliationState,
) -> Dict[str, str]:
    processed = row.copy()
    source = processed.get(source_column_name)
    if not source:
        processed.setdefault(target_column_name, "")
        return processed
    values = source.split('|')
    state.total_values += len(values)
    reconciled = [
        item_id
        for raw_value in values
        if (value := raw_value.strip())
        if (item_id := _reconcile_value(
            value, authority_dict, ambiguous_authority_dict, state
        )) is not None
    ]
    processed[target_column_name] = '|'.join(reconciled)
    return processed


def _read_csv_rows(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, 'r', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV file is empty or header is missing: {path}")
        return list(reader.fieldnames), list(reader)


def _write_unreconciled(path: str, counts: Counter) -> None:
    if not counts:
        return
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Unreconciled Value', 'Count'])
        writer.writerows(sorted(counts.items(), key=lambda item: item[1], reverse=True))


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
    """Reconcile pipe-separated values and write IDs plus an exception report."""
    unreconciled_path = (
        f"{initial_csv_base_for_unreconciled}_unreconciled_{output_file_tag}.csv"
    )
    try:
        original_fieldnames, rows = _read_csv_rows(input_csv_path)
    except FileNotFoundError:
        console.print(f"[red]✗[/] Input CSV file not found: {input_csv_path}")
        return output_reconciled_csv_path, 0, 0, 0
    except ValueError as exc:
        console.print(f"[red]✗[/] {exc}")
        return output_reconciled_csv_path, 0, 0, 0

    output_fieldnames = list(original_fieldnames)
    if target_column_name not in output_fieldnames:
        output_fieldnames.append(target_column_name)

    state = ReconciliationState()
    if source_column_name not in original_fieldnames:
        console.print(
            f"[yellow]⚠[/] Source column '{source_column_name}' not found. "
            "Skipping reconciliation."
        )
        processed_rows = [
            {**row, target_column_name: row.get(target_column_name, "")} for row in rows
        ]
    else:
        processed_rows = []
        with standard_progress(console) as progress:
            task = progress.add_task(
                f"[cyan]Reconciling {source_column_name}...", total=len(rows)
            )
            for row in rows:
                processed_rows.append(_reconcile_row(
                    row,
                    source_column_name,
                    target_column_name,
                    authority_dict,
                    ambiguous_authority_dict,
                    state,
                ))
                progress.update(task, advance=1)

    with open(output_reconciled_csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(processed_rows)
    _write_unreconciled(unreconciled_path, state.unreconciled_counts)

    if state.ambiguous_terms:
        console.print(
            f"[yellow]⚠[/] {len(state.ambiguous_terms)} ambiguous terms skipped "
            "(see unreconciled file)"
        )
    return (
        output_reconciled_csv_path,
        state.matched_count,
        state.total_values,
        len(state.unreconciled_counts),
    )


 # ---------------------------------------------------------------------------
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
        raise


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
