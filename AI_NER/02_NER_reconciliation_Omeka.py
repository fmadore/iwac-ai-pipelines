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
   - Item Set 268: Spatial/location authorities
   - Item Set 854, 2, 266: Subject authorities
   - Item Set 1: Topic authorities
   
   (Modify the item_set_ids in main() function to match your setup)

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

from typing import Dict, List, Tuple # Added Tuple
from dotenv import load_dotenv
from tqdm import tqdm
from collections import Counter, defaultdict # Added defaultdict
import csv # Added csv
import logging # Ensure logging, os, requests are imported
import os
import requests
import unicodedata
from difflib import SequenceMatcher
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

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

def make_api_request(endpoint: str, params: Dict = None) -> Dict:
    """
    Make authenticated API request to Omeka S.
    
    Args:
        endpoint: API endpoint (e.g., 'items', 'item_sets')
        params: Additional query parameters
        
    Returns:
        JSON response from the API
        
    Raises:
        HTTPError: If the API request fails
    """
    base_url = os.getenv('OMEKA_BASE_URL')
    url = f"{base_url}/{endpoint}"
    
    params = params or {}
    params.update({
        'key_identity': os.getenv('OMEKA_KEY_IDENTITY'),
        'key_credential': os.getenv('OMEKA_KEY_CREDENTIAL')
    })
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

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

def build_authority_dict(item_set_ids: List[str]) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, Dict[str, str]]]: # Signature changed
    """Build dictionary of authority terms from specified item sets and identify ambiguous terms.
    
    Returns:
        - authority_dict: mapping from normalized names to item IDs
        - ambiguous_terms_dict: mapping from names to multiple item IDs
        - authority_metadata: mapping from item IDs to their metadata (original titles, alternatives)
    """
    
    potential_lookups: List[tuple[str, str]] = [] # Stores (lookup_key, item_id)
    authority_metadata: Dict[str, Dict[str, str]] = {} # Stores metadata for each item_id

    logger.info(f"Starting to build authority dictionary for item sets: {item_set_ids}")
    for item_set_id in item_set_ids:
        page = 1
        logger.info(f"Fetching items from item set ID: {item_set_id}")
        
        while True:
            try:
                items = make_api_request('items', {
                    'item_set_id': item_set_id,
                    'page': page,
                    'per_page': 100  # Assuming 100 is the per_page limit
                })
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error {e.response.status_code} for item_set_id {item_set_id}, page {page}: {e.response.text}")
                break 
            
            if not items:
                break
            
            for item in items:
                item_id = str(item['o:id'])
                
                # Store metadata for this item
                metadata = {
                    'primary_title': '',
                    'alternatives': []
                }
                
                titles_to_process = []
                if 'dcterms:title' in item:
                    for title_obj in item['dcterms:title']:
                        title = title_obj['@value'].strip()
                        if title: # Ensure title is not empty
                            titles_to_process.append(title)
                            if not metadata['primary_title']:
                                metadata['primary_title'] = title
                
                if 'dcterms:alternative' in item:
                    for alt in item['dcterms:alternative']:
                        alt_value = alt['@value'].strip()
                        if alt_value: # Ensure alt_value is not empty
                            titles_to_process.append(alt_value)
                            metadata['alternatives'].append(alt_value)
                
                # Store metadata
                authority_metadata[item_id] = metadata

                for title_text in titles_to_process:
                    lower_variant = title_text.lower()
                    normalized_title = normalize_location_name(title_text)
                    # Also add diacritic-stripped spaced form to broaden matches
                    diacritic_stripped_spaced = _strip_diacritics(lower_variant)
                    for variant in {lower_variant, normalized_title, diacritic_stripped_spaced}:
                        if variant:
                            potential_lookups.append((variant, item_id))
            
            if len(items) < 100:
                break
            page += 1
        logger.info(f"Finished fetching items for item set ID: {item_set_id}")

    logger.info(f"Processing {len(potential_lookups)} potential lookups to identify authorities and ambiguities.")
    
    name_to_ids_map = defaultdict(set)
    for name, item_id_val in potential_lookups:
        name_to_ids_map[name].add(item_id_val)
            
    authority_dict_final = {}
    ambiguous_terms_dict = {}
    
    for name, ids_set in name_to_ids_map.items():
        if len(ids_set) == 1:
            authority_dict_final[name] = list(ids_set)[0]
        else:
            # This name maps to multiple item IDs, so it's ambiguous
            ambiguous_terms_dict[name] = sorted(list(ids_set)) # Store as sorted list of IDs
            
    logger.info(f"Authority dictionary built with {len(authority_dict_final)} unique terms.")
    logger.info(f"Identified {len(ambiguous_terms_dict)} ambiguous terms.")
    
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
                candidates.append((item_id, primary, sim_primary, 'primary_title'))
        for alt in metadata.get('alternatives', [])[:50]:  # safety slice
            sim_alt = calculate_similarity(unreconciled_value, alt)
            if sim_alt >= min_similarity:
                candidates.append((item_id, alt, sim_alt, 'alternative'))

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
        logger.warning(f"Unreconciled CSV file not found: {unreconciled_csv_path}")
        return
    
    potential_matches = []
    
    try:
        with open(unreconciled_csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in tqdm(reader, desc="Finding potential matches"):
                unreconciled_value = row['Unreconciled Value']
                count = row['Count']
                
                # Skip ambiguous values (they contain "(Ambiguous)" in the name)
                if "(Ambiguous)" in unreconciled_value:
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
            
            logger.info(f"Potential reconciliation candidates written to: {output_csv_path}")
            logger.info(f"Found {len(potential_matches)} potential matches for unreconciled values")
        else:
            logger.info(f"No potential matches found above similarity threshold {min_similarity}")
            
    except Exception as e:
        logger.error(f"Error creating potential reconciliation CSV: {e}", exc_info=True)

def reconcile_column_values(input_csv_path: str, 
                            output_reconciled_csv_path: str, 
                            authority_dict: Dict[str, str], 
                            source_column_name: str, 
                            target_column_name: str, 
                            initial_csv_base_for_unreconciled: str, 
                            output_file_tag: str,
                            ambiguous_authority_dict: Dict[str, List[str]]) -> str: # Added ambiguous_authority_dict
    """Reconcile values in a specific CSV column, writing to a designated output CSV. 
       Unreconciled values are saved separately, named based on the initial input file and a tag.
       Ambiguous values are not reconciled and are logged.
    """
    # Use the provided path for the main reconciled output
    # Unreconciled file path is based on the initial CSV base and the current tag
    unreconciled_path = f"{initial_csv_base_for_unreconciled}_unreconciled_{output_file_tag}.csv"
    
    rows_processed = []
    unreconciled_counts = Counter()
    matched_count = 0
    total_values_in_source_col = 0
    rows_with_source_col = 0
    
    try:
        with open(input_csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames:
                logger.error(f"CSV file is empty or header is missing: {input_csv_path}")
                # If input is invalid, we probably shouldn't write an output file.
                # Return the intended output path, but it won't be valid.
                return output_reconciled_csv_path 
                
            original_fieldnames = list(reader.fieldnames)
            output_fieldnames = list(original_fieldnames) # Start with existing fieldnames
            if target_column_name not in output_fieldnames:
                output_fieldnames.append(target_column_name)
            else:
                # If target column already exists, we will overwrite its contents for matched rows.
                logger.info(f"Target column '{target_column_name}' already exists. Its content will be updated.")

            if source_column_name not in original_fieldnames:
                logger.warning(f"Source column '{source_column_name}' not found in {input_csv_path}. "
                               f"Skipping reconciliation for this column. Target column '{target_column_name}' will be empty.")
                # Populate rows_processed with original rows, ensuring target_column_name exists
                for row in tqdm(reader, desc=f"Copying rows (source '{source_column_name}' missing)"):
                    processed_row = row.copy()
                    if target_column_name not in processed_row:
                         processed_row[target_column_name] = "" # Ensure column exists
                    rows_processed.append(processed_row)
            else:
                for row in tqdm(reader, desc=f"Reconciling {source_column_name}"):
                    processed_row = row.copy()
                    reconciled_values_str = processed_row.get(target_column_name, "") # Preserve existing if not overwritten
                    source_value = processed_row.get(source_column_name)

                    if source_value: 
                        rows_with_source_col += 1
                        individual_values = source_value.split('|')
                        total_values_in_source_col += len(individual_values)
                        reconciled_ids = []
                        # unreconciled_ids = [] # This line was present in the original but seems unused, consider removing if truly not needed.

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
                                logger.warning(f"Term '{value_stripped}' is ambiguous and will not be reconciled. Found in row: {row}")
                                unreconciled_counts[f"{value_stripped} (Ambiguous)"] += 1
                                # Do not add to reconciled_ids, effectively skipping it for reconciliation
                                continue # Move to the next value in individual_values

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
    
    except FileNotFoundError:
        logger.error(f"Input CSV file not found: {input_csv_path}")
        return output_reconciled_csv_path # Return intended path, but it might not be created
    except Exception as e:
        logger.error(f"Error reading CSV {input_csv_path}: {e}", exc_info=True)
        raise 

    # Write reconciled data to the designated output CSV path
    try:
        with open(output_reconciled_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=output_fieldnames)
            writer.writeheader()
            writer.writerows(rows_processed)
    except Exception as e:
        logger.error(f"Error writing reconciled CSV {output_reconciled_csv_path}: {e}", exc_info=True)
        raise

    # Write unreconciled values
    if unreconciled_counts: # Only write if there are unreconciled items
        try:
            with open(unreconciled_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Unreconciled Value', 'Count'])
                sorted_unreconciled = sorted(unreconciled_counts.items(), key=lambda item: item[1], reverse=True)
                for value, count in sorted_unreconciled:
                    writer.writerow([value, count])
                logger.info(f"Unreconciled values for {output_file_tag} written to: {unreconciled_path}")
        except Exception as e:
            logger.error(f"Error writing unreconciled CSV {unreconciled_path}: {e}", exc_info=True)
    else:
        logger.info(f"No unreconciled values to write for {output_file_tag}.")

    # Log statistics
    match_rate = (matched_count / total_values_in_source_col * 100) if total_values_in_source_col > 0 else 0
    logger.info(f"Reconciliation for column '{source_column_name}' (tag: {output_file_tag}) complete:")
    logger.info(f"Input file processed: {input_csv_path}")
    logger.info(f"Rows with data in '{source_column_name}': {rows_with_source_col}")
    logger.info(f"Total values processed in '{source_column_name}': {total_values_in_source_col}")
    logger.info(f"Values matched with authorities: {matched_count}")
    logger.info(f"Unreconciled unique values: {len(unreconciled_counts)}")
    logger.info(f"Match rate: {match_rate:.2f}%")
    logger.info(f"Main reconciled data updated in: {output_reconciled_csv_path}")
    
    return output_reconciled_csv_path

def write_ambiguous_terms_to_file(ambiguous_dict: Dict[str, List[str]], output_path: str):
    """Writes ambiguous terms and their associated item IDs to a CSV file."""
    if not ambiguous_dict:
        logger.info(f"No ambiguous terms to write to {output_path}.")
        return

    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile_obj: # Renamed csvfile to csvfile_obj
            writer = csv.writer(csvfile_obj)
            writer.writerow(['Ambiguous Term', 'Item IDs'])
            # Sort by term for consistent output
            for term, ids in sorted(ambiguous_dict.items()):
                writer.writerow([term, '|'.join(ids)]) # Join IDs with a pipe
        logger.info(f"Ambiguous terms written to: {output_path}")
    except Exception as e:
        logger.error(f"Error writing ambiguous terms CSV {output_path}: {e}", exc_info=True)

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
        - CSV file with columns 'Spatial AI' and 'Subject AI' containing pipe-separated values
        - .env file with Omeka S API credentials
        - Configured item sets in Omeka S:
          * Item set 268: Spatial/location authorities
          * Item sets 854, 2, 266: Subject authorities  
          * Item set 1: Topic authorities
    
    Output Files:
        - *_reconciled.csv: Main file with reconciled entity IDs
        - *_unreconciled_spatial.csv: Unmatched spatial entities
        - *_unreconciled_subject_and_topic.csv: Unmatched subject/topic entities
        - *_ambiguous_authorities_*.csv: Terms matching multiple authorities
        - *_potential_reconciliation_*.csv: Suggested matches for manual review
    
    File Naming Convention:
        All output files use the base name of the input CSV with descriptive suffixes.
        Example: input_file.csv → input_file_reconciled.csv, input_file_unreconciled_spatial.csv
    
    Error Handling:
        - HTTP errors from Omeka API are logged and continue processing
        - File not found errors are logged 
        - Unexpected errors include full traceback for debugging
    """
    try:
        # STEP 1: Setup and file discovery
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "output")

        # Find the most recent unprocessed CSV file
        csv_files = [
            f for f in os.listdir(output_dir) 
            if f.endswith('.csv') and '_reconciled' not in f and '_unreconciled' not in f and '_ambiguous_authorities' not in f
        ]
        if not csv_files:
            logger.error("No suitable CSV files found in output directory for initial processing.")
            return
            
        latest_csv_filename = max(csv_files, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
        initial_csv_path = os.path.join(output_dir, latest_csv_filename)
        initial_csv_base, initial_csv_ext = os.path.splitext(initial_csv_path)
        
        # Define the single output path for all reconciled data
        final_reconciled_csv_path = f"{initial_csv_base}_reconciled{initial_csv_ext}"

        logger.info(f"Starting reconciliation process with initial file: {initial_csv_path}")
        logger.info(f"Final reconciled data will be written to: {final_reconciled_csv_path}")

        current_input_path = initial_csv_path

        # STEP 2: Spatial Entity Reconciliation
        logger.info("--- Starting Spatial Reconciliation ---")
        logger.info(f"Building SPATIAL authority dictionary from item set(s): {SPATIAL_AUTHORITY_ITEM_SETS}...")
        spatial_authority_dict, spatial_ambiguous_dict, spatial_metadata = build_authority_dict(SPATIAL_AUTHORITY_ITEM_SETS)
        logger.info(f"SPATIAL authority dictionary built with {len(spatial_authority_dict)} terms. Found {len(spatial_ambiguous_dict)} ambiguous terms.")
        
        # Write ambiguous spatial terms for manual review
        ambiguous_spatial_tag = "spatial"
        ambiguous_spatial_path = f"{initial_csv_base}_ambiguous_authorities_{ambiguous_spatial_tag}.csv"
        write_ambiguous_terms_to_file(spatial_ambiguous_dict, ambiguous_spatial_path)
        
        # Perform spatial reconciliation
        logger.info(f"Processing SPATIAL reconciliation for column '{SPATIAL_COLUMN}'. Input: {current_input_path}, Output: {final_reconciled_csv_path}")
        reconciled_after_spatial_path = reconcile_column_values(
            input_csv_path=current_input_path,
            output_reconciled_csv_path=final_reconciled_csv_path, 
            authority_dict=spatial_authority_dict,
            source_column_name=SPATIAL_COLUMN,
            target_column_name=SPATIAL_OUTPUT_COLUMN,
            initial_csv_base_for_unreconciled=initial_csv_base,
            output_file_tag="spatial",
            ambiguous_authority_dict=spatial_ambiguous_dict
        )
        logger.info(f"Spatial reconciliation step complete. Main data at: {reconciled_after_spatial_path}")
        current_input_path = reconciled_after_spatial_path  # Chain reconciliation steps

        # STEP 3: Subject and Topic Entity Reconciliation
        logger.info("--- Starting Combined Subject and Topic Reconciliation ---")
        
        # Build subject authority dictionary
        logger.info(f"Building SUBJECT authority dictionary from item set(s): {SUBJECT_AUTHORITY_ITEM_SETS}...")
        subject_authority_dict, subject_ambiguous_dict, subject_metadata = build_authority_dict(SUBJECT_AUTHORITY_ITEM_SETS)
        logger.info(f"SUBJECT authority dictionary built with {len(subject_authority_dict)} terms. Found {len(subject_ambiguous_dict)} ambiguous terms.")

        # Build topic authority dictionary
        logger.info(f"Building TOPIC authority dictionary from item set(s): {TOPIC_AUTHORITY_ITEM_SETS}...")
        topic_authority_dict, topic_ambiguous_dict, topic_metadata = build_authority_dict(TOPIC_AUTHORITY_ITEM_SETS)
        logger.info(f"TOPIC authority dictionary built with {len(topic_authority_dict)} terms. Found {len(topic_ambiguous_dict)} ambiguous terms.")

        # Combine authority dictionaries for unified reconciliation
        combined_authority_dict = {**subject_authority_dict, **topic_authority_dict}
        combined_ambiguous_dict = {**subject_ambiguous_dict, **topic_ambiguous_dict}
        logger.info(f"Combined authority dictionary has {len(combined_authority_dict)} unique terms.")
        logger.info(f"Combined ambiguous terms: {len(combined_ambiguous_dict)} terms.")

        # Write combined ambiguous terms for manual review
        ambiguous_subject_tag = "subject_and_topic"
        ambiguous_subject_path = f"{initial_csv_base}_ambiguous_authorities_{ambiguous_subject_tag}.csv"
        write_ambiguous_terms_to_file(combined_ambiguous_dict, ambiguous_subject_path)

        # Perform subject/topic reconciliation
        logger.info(f"Processing combined SUBJECT+TOPIC reconciliation for column '{SUBJECT_COLUMN}'. Input: {current_input_path}, Output: {final_reconciled_csv_path}")
        final_reconciled_path = reconcile_column_values(
            input_csv_path=current_input_path, 
            output_reconciled_csv_path=final_reconciled_csv_path, 
            authority_dict=combined_authority_dict,
            source_column_name=SUBJECT_COLUMN,
            target_column_name=SUBJECT_OUTPUT_COLUMN,
            initial_csv_base_for_unreconciled=initial_csv_base,
            output_file_tag="subject_and_topic",
            ambiguous_authority_dict=combined_ambiguous_dict
        )
        logger.info(f"Combined subject and topic reconciliation complete. Final reconciled data at: {final_reconciled_path}")
        
        # STEP 4: Generate Potential Reconciliation Candidates for Manual Review
        logger.info("--- Generating Potential Reconciliation Candidates ---")
        
        # Create potential matches for spatial unreconciled values
        spatial_unreconciled_path = f"{initial_csv_base}_unreconciled_spatial.csv"
        spatial_potential_path = f"{initial_csv_base}_potential_reconciliation_spatial.csv"
        if os.path.exists(spatial_unreconciled_path):
            logger.info(f"Creating potential reconciliation candidates for spatial values...")
            create_potential_reconciliation_csv(
                unreconciled_csv_path=spatial_unreconciled_path,
                authority_metadata=spatial_metadata,
                output_csv_path=spatial_potential_path,
                min_similarity=MULTI_WORD_MIN_SIMILARITY,
                max_candidates_per_value=DEFAULT_MAX_CANDIDATES
            )
        else:
            logger.info(f"No spatial unreconciled file found at: {spatial_unreconciled_path}")
        
        # Create potential matches for subject and topic unreconciled values
        subject_topic_unreconciled_path = f"{initial_csv_base}_unreconciled_subject_and_topic.csv"
        subject_topic_potential_path = f"{initial_csv_base}_potential_reconciliation_subject_and_topic.csv"
        if os.path.exists(subject_topic_unreconciled_path):
            logger.info(f"Creating potential reconciliation candidates for subject and topic values...")
            # Combine metadata for both subject and topic authorities
            combined_metadata = {**subject_metadata, **topic_metadata}
            create_potential_reconciliation_csv(
                unreconciled_csv_path=subject_topic_unreconciled_path,
                authority_metadata=combined_metadata,
                output_csv_path=subject_topic_potential_path,
                min_similarity=MULTI_WORD_MIN_SIMILARITY,
                max_candidates_per_value=DEFAULT_MAX_CANDIDATES
            )
        else:
            logger.info(f"No subject/topic unreconciled file found at: {subject_topic_unreconciled_path}")
        
        logger.info(f"--- Potential Reconciliation Generation Complete ---")
        logger.info(f"--- Reconciliation Process Finished ---")
        
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"An HTTP error occurred during API request: {http_err} - Response: {http_err.response.text if http_err.response else 'No response'}")
        # Depending on the severity, you might want to raise or handle differently
    except FileNotFoundError as fnf_err:
        logger.error(f"A required file was not found: {fnf_err}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}", exc_info=True) # Add exc_info for traceback
        # raise # Optionally re-raise if you want the script to terminate with an error status

if __name__ == "__main__":
    main()
