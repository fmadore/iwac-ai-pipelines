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
from dataclasses import dataclass, field

# Increase CSV field size limit to handle large bibo:content fields
csv.field_size_limit(10 * 1024 * 1024)
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Collection, Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table
from rich import box

from common.omeka_client import OmekaClient
from common.console_utils import standard_progress


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

#: The looser keys live in the same dict as the exact ones under their own
#: prefix, and are only consulted once the exact keys have missed: a looser
#: key can neither shadow an exact match nor make it ambiguous.
ORDER_KEY_PREFIX = "[order] "
FOLD_KEY_PREFIX = "[fold] "
#: Single-letter tokens that are elided articles, not initials.
_ELISIONS = {'d', 'l'}

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


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def token_order_key(text: str) -> str:
    """The words of *text* in sorted order, so that "Kassoum Coulibaly" and
    "Coulibaly Kassoum" share a key. Only ever compared for equality.

    Needs two words of two letters or more, and a title carrying initials gets
    no key at all: "A. B." and "B. A." are two different signatures.
    """
    tokens = _WORD_RE.findall(_strip_diacritics(text.lower()))
    if any(len(token) == 1 and token not in _ELISIONS for token in tokens):
        return ''
    if sum(len(token) > 1 for token in tokens) < 2:
        return ''
    return ORDER_KEY_PREFIX + ' '.join(sorted(tokens))


def plural_fold_key(text: str) -> str:
    """The words of *text* with a trailing s or x dropped from any word of
    five letters or more: "Cercle d'Étude, de Recherche…" meets "Cercle
    d'Études, de Recherches…". Punctuation is ignored.

    Accents are kept, unlike the exact keys: dropping them as well would give
    "Jeunes" and "Jeûne" the same key.
    """
    tokens = _WORD_RE.findall(unicodedata.normalize('NFC', text.lower()))
    if not tokens:
        return ''
    return FOLD_KEY_PREFIX + ' '.join(
        token[:-1] if len(token) > 4 and token[-1] in 'sx' else token
        for token in tokens
    )


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


def _authority_variants(
    title: str, token_order: bool = False, plural_fold: bool = False,
) -> set[str]:
    lower = title.lower()
    return {
        variant for variant in (
            lower,
            normalize_location_name(title),
            _strip_diacritics(lower),
            token_order_key(title) if token_order else '',
            plural_fold_key(title) if plural_fold else '',
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
    *,
    token_order: bool = False,
    plural_fold_sets: Collection[str] = (),
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, Dict]]:
    """Build a lookup dict of authority terms from specified item sets.

    *token_order* adds the key of :func:`token_order_key` to every title, and
    titles in *plural_fold_sets* also get that of :func:`plural_fold_key`;
    reconciliation tries them only after every exact key has missed, in that
    order. Leave persons out of the fold: a plural never names one person, and
    "Prophètes" would otherwise reach Mahomet through his alias "Prophète".

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
            # A set that cannot be fetched must stop the run: reconciling
            # against the rest would miss cross-set ambiguity and link a term
            # to the wrong item, or leave it "unreconciled" and invite a
            # duplicate authority.
            try:
                all_items = client.get_items(int(item_set_id))
            except Exception as e:
                raise RuntimeError(
                    f"Could not fetch {authority_type} item set {item_set_id}: {e}"
                ) from e
            if not all_items:
                continue

            plural_fold = item_set_id in plural_fold_sets
            for item in all_items:
                item_id = str(item['o:id'])
                metadata, titles_to_process = _authority_titles(item)
                authority_metadata[item_id] = metadata
                for title_text in titles_to_process:
                    potential_lookups.extend(
                        (variant, item_id)
                        for variant in _authority_variants(title_text, token_order, plural_fold)
                    )

    authority, ambiguous = _resolve_authority_lookups(potential_lookups)
    return authority, ambiguous, authority_metadata


# ---------------------------------------------------------------------------
# Similarity / fuzzy matching
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


@lru_cache(maxsize=65536)
def _prepared(text: str) -> Tuple[str, str, Tuple[str, ...]]:
    """(lowercase, compact key, content tokens) for one string, computed once.

    A fuzzy pass compares every unreconciled value against every authority
    title; without this each authority's forms were re-derived per value.
    """
    lower = text.lower().strip()
    norm = normalize_location_name(text)
    tokens = tuple(t for t in _TOKEN_RE.findall(_strip_diacritics(lower)) if t not in STOPWORDS)
    return lower, norm, tokens


def _ratio_at_least(a: str, b: str, floor: float) -> float:
    """``SequenceMatcher.ratio()`` unless its cheap upper bound rules it out."""
    matcher = SequenceMatcher(None, a, b)
    if matcher.real_quick_ratio() < floor or matcher.quick_ratio() < floor:
        return 0.0
    return matcher.ratio()


def calculate_similarity(s1: str, s2: str) -> float:
    """Return a conservative similarity score (0..1) between two strings."""
    s1_lower, s1_norm, s1_tokens = _prepared(s1)
    s2_lower, s2_norm, s2_tokens = _prepared(s2)
    if not s1_lower or not s2_lower:
        return 0.0
    if s1_lower == s2_lower:
        return 1.0
    if s1_norm and s1_norm == s2_norm:
        return STRONG_MATCH_THRESHOLD

    s1_token_set, s2_token_set = set(s1_tokens), set(s2_tokens)
    common_tokens = s1_token_set & s2_token_set
    all_tokens = s1_token_set | s2_token_set

    if len(all_tokens) >= 2:
        if not common_tokens:
            return 0.0
        token_overlap_ratio = len(common_tokens) / len(all_tokens)
    else:
        token_overlap_ratio = 1.0 if common_tokens else 0.0

    if len(all_tokens) >= 2 and token_overlap_ratio < MIN_TOKEN_OVERLAP:
        return 0.0

    # A reversed name ("Bamba Amadou" / "Amadou Bamba") is a strong match
    # whatever the character similarity, so it is decided before the ratio
    # gate below, which would otherwise discard it.
    reversed_name = bool(s1_tokens and s2_tokens and s1_token_set == s2_token_set
                         and tuple(reversed(s1_tokens)) == s2_tokens)
    short = len(s1_lower) <= 4 or len(s2_lower) <= 4
    floor = 0.0 if reversed_name else (0.95 if short else 0.85)
    char_similarity = _ratio_at_least(s1_lower, s2_lower, floor)
    norm_similarity = _ratio_at_least(s1_norm, s2_norm, floor) if (s1_norm and s2_norm) else 0.0
    max_char_sim = max(char_similarity, norm_similarity)

    if short:
        return max_char_sim if max_char_sim >= 0.95 else 0.0
    if reversed_name:
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
    redirected_count: int = 0
    total_values: int = 0
    unreconciled_counts: Counter = field(default_factory=Counter)
    ambiguous_terms: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class Redirect:
    """A second authority dict for values the first one does not know, whose
    matches are written to another column. NER uses it to send a subject term
    that names a place to ``dcterms:spatial``, where the catalogue links every
    place, rather than leave it unreconciled."""

    authority_dict: Dict[str, str]
    ambiguous_dict: Dict[str, List[str]]
    target_column: str


def _value_variants(value: str) -> Tuple[str, str, str]:
    lower = value.lower()
    return lower, normalize_location_name(value), _strip_diacritics(lower)


def _value_key_tiers(value: str) -> Tuple[Tuple[str, ...], ...]:
    """Exact keys first, then word order, then plural fold. A dict built
    without a looser key simply never contains it."""
    return _value_variants(value), (token_order_key(value),), (plural_fold_key(value),)


def _lookup(
    value: str,
    authority_dict: Dict[str, str],
    ambiguous_authority_dict: Dict[str, List[str]],
) -> Tuple[Optional[str], bool]:
    """(item id, ambiguous) from the first key tier that knows *value*."""
    for tier in _value_key_tiers(value):
        keys = [key for key in tier if key]
        if any(key in ambiguous_authority_dict for key in keys):
            return None, True
        for key in keys:
            if key in authority_dict:
                return authority_dict[key], False
    return None, False


def _reconcile_value(
    value: str,
    authority_dict: Dict[str, str],
    ambiguous_authority_dict: Dict[str, List[str]],
    state: ReconciliationState,
    redirect: Optional[Redirect] = None,
) -> Tuple[Optional[str], bool]:
    """(item id, redirected) for one value, counting it in *state*."""
    item_id, ambiguous = _lookup(value, authority_dict, ambiguous_authority_dict)
    if ambiguous:
        state.ambiguous_terms.add(value)
        state.unreconciled_counts[f"{value} {AMBIGUOUS_MARKER}"] += 1
        return None, False
    if item_id is not None:
        state.matched_count += 1
        return item_id, False
    if redirect is not None:
        item_id, ambiguous = _lookup(value, redirect.authority_dict, redirect.ambiguous_dict)
        if item_id is not None and not ambiguous:
            state.matched_count += 1
            state.redirected_count += 1
            return item_id, True
    state.unreconciled_counts[value] += 1
    return None, False


def _reconcile_row(
    row: Dict[str, str],
    source_column_name: str,
    target_column_name: str,
    authority_dict: Dict[str, str],
    ambiguous_authority_dict: Dict[str, List[str]],
    state: ReconciliationState,
    redirect: Optional[Redirect] = None,
) -> Dict[str, str]:
    processed = row.copy()
    if redirect is not None:
        processed.setdefault(redirect.target_column, "")
    source = processed.get(source_column_name)
    if not source:
        processed.setdefault(target_column_name, "")
        return processed
    values = source.split('|')
    state.total_values += len(values)
    reconciled: List[str] = []
    redirected: List[str] = []
    for raw_value in values:
        if not (value := raw_value.strip()):
            continue
        item_id, was_redirected = _reconcile_value(
            value, authority_dict, ambiguous_authority_dict, state, redirect
        )
        if item_id is not None:
            (redirected if was_redirected else reconciled).append(item_id)
    processed[target_column_name] = '|'.join(reconciled)
    if redirected:
        existing = [i for i in processed[redirect.target_column].split('|') if i]
        processed[redirect.target_column] = '|'.join(
            existing + [i for i in dict.fromkeys(redirected) if i not in existing]
        )
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
    redirect: Optional[Redirect] = None,
) -> Tuple[str, int, int, int]:
    """Reconcile pipe-separated values and write IDs plus an exception report.

    With *redirect*, a value unknown to *authority_dict* (and not ambiguous
    there) that matches the redirect dict is appended, deduplicated, to
    ``redirect.target_column`` instead. It counts as matched.
    """
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
    if redirect is not None and redirect.target_column not in output_fieldnames:
        output_fieldnames.append(redirect.target_column)

    state = ReconciliationState()
    if source_column_name not in original_fieldnames:
        console.print(
            f"[yellow]⚠[/] Source column '{source_column_name}' not found. "
            "Skipping reconciliation."
        )
        processed_rows = [
            {**row, target_column_name: row.get(target_column_name, "")} for row in rows
        ]
        if redirect is not None:
            for row in processed_rows:
                row.setdefault(redirect.target_column, "")
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
                    redirect,
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
    if state.redirected_count:
        console.print(
            f"[dim]{state.redirected_count} of the matched values written to "
            f"{redirect.target_column}[/]"
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
