from typing import Dict, List, Tuple # Added Tuple
from dotenv import load_dotenv
from tqdm import tqdm
from collections import Counter, defaultdict # Added defaultdict
import csv # Added csv
import logging # Ensure logging, os, requests are imported
import os
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def make_api_request(endpoint: str, params: Dict = None) -> Dict:
    """Make API request to Omeka S."""
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

def normalize_location_name(name: str) -> str:
    """Normalize location names for better matching."""
    # Convert to lowercase
    normalized = name.lower().strip()
    
    # Normalize various forms of apostrophes and dashes
    normalized = normalized.replace("'", "'").replace("'", "'")
    normalized = normalized.replace("-", "").replace(" ", "")
    
    return normalized

def build_authority_dict(item_set_ids: List[str]) -> Tuple[Dict[str, str], Dict[str, List[str]]]: # Signature changed
    """Build dictionary of authority terms from specified item sets and identify ambiguous terms."""
    
    potential_lookups: List[tuple[str, str]] = [] # Stores (lookup_key, item_id)

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
                
                titles_to_process = []
                if 'dcterms:title' in item:
                    for title_obj in item['dcterms:title']:
                        title = title_obj['@value'].strip()
                        if title: # Ensure title is not empty
                            titles_to_process.append(title)
                
                if 'dcterms:alternative' in item:
                    for alt in item['dcterms:alternative']:
                        alt_value = alt['@value'].strip()
                        if alt_value: # Ensure alt_value is not empty
                            titles_to_process.append(alt_value)

                for title_text in titles_to_process:
                    potential_lookups.append((title_text.lower(), item_id))
                    normalized_title = normalize_location_name(title_text)
                    if normalized_title: # Ensure normalized_title is not empty
                        potential_lookups.append((normalized_title, item_id))
            
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
    
    return authority_dict_final, ambiguous_terms_dict

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
                            
                            # Check if the term is ambiguous first
                            if value_lower in ambiguous_authority_dict or value_normalized in ambiguous_authority_dict:
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
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "output")

        csv_files = [
            f for f in os.listdir(output_dir) 
            if f.endswith('.csv') and '_reconciled' not in f and '_unreconciled' not in f
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

        # 1. Spatial Reconciliation
        logger.info("--- Starting Spatial Reconciliation ---")
        spatial_authority_item_sets = ["268"]
        logger.info(f"Building SPATIAL authority dictionary from item set(s): {spatial_authority_item_sets}...")
        spatial_authority_dict, spatial_ambiguous_dict = build_authority_dict(spatial_authority_item_sets) # Capture both dicts
        logger.info(f"SPATIAL authority dictionary built with {len(spatial_authority_dict)} terms. Found {len(spatial_ambiguous_dict)} ambiguous terms.")
        
        # Write ambiguous spatial terms
        ambiguous_spatial_tag = "spatial"
        ambiguous_spatial_path = f"{initial_csv_base}_ambiguous_authorities_{ambiguous_spatial_tag}.csv"
        write_ambiguous_terms_to_file(spatial_ambiguous_dict, ambiguous_spatial_path)
        
        logger.info(f"Processing SPATIAL reconciliation for column 'Spatial AI'. Input: {current_input_path}, Output: {final_reconciled_csv_path}")
        reconciled_after_spatial_path = reconcile_column_values(
            input_csv_path=current_input_path,
            output_reconciled_csv_path=final_reconciled_csv_path, 
            authority_dict=spatial_authority_dict,
            source_column_name="Spatial AI",
            target_column_name="Spatial AI Reconciled ID",
            initial_csv_base_for_unreconciled=initial_csv_base, # Base for unreconciled file name
            output_file_tag="spatial",
            ambiguous_authority_dict=spatial_ambiguous_dict # Pass ambiguous dict
        )
        logger.info(f"Spatial reconciliation step complete. Main data at: {reconciled_after_spatial_path}")
        current_input_path = reconciled_after_spatial_path # Output of this step is input for the next

        # 2. Subject Reconciliation
        logger.info("--- Starting Subject Reconciliation ---")
        subject_authority_item_sets = ["854", "2", "266"]
        logger.info(f"Building SUBJECT authority dictionary from item set(s): {subject_authority_item_sets}...")
        subject_authority_dict, subject_ambiguous_dict = build_authority_dict(subject_authority_item_sets) # Capture both dicts
        logger.info(f"SUBJECT authority dictionary built with {len(subject_authority_dict)} terms. Found {len(subject_ambiguous_dict)} ambiguous terms.")

        # Write ambiguous subject terms
        ambiguous_subject_tag = "subject"
        ambiguous_subject_path = f"{initial_csv_base}_ambiguous_authorities_{ambiguous_subject_tag}.csv"
        write_ambiguous_terms_to_file(subject_ambiguous_dict, ambiguous_subject_path)

        logger.info(f"Processing SUBJECT reconciliation for column 'Subject AI'. Input: {current_input_path}, Output: {final_reconciled_csv_path}")
        reconciled_after_subject_path = reconcile_column_values(
            input_csv_path=current_input_path, 
            output_reconciled_csv_path=final_reconciled_csv_path, 
            authority_dict=subject_authority_dict,
            source_column_name="Subject AI",
            target_column_name="Subject AI Reconciled ID",
            initial_csv_base_for_unreconciled=initial_csv_base, # Base for unreconciled file name
            output_file_tag="subject",
            ambiguous_authority_dict=subject_ambiguous_dict # Pass ambiguous dict
        )
        logger.info(f"Subject reconciliation step complete. Main data at: {reconciled_after_subject_path}")
        current_input_path = reconciled_after_subject_path # Output of this step is input for the next

        # 3. Topic Reconciliation
        logger.info("--- Starting Topic Reconciliation ---")
        topic_authority_item_sets = ["1"]
        logger.info(f"Building TOPIC authority dictionary from item set(s): {topic_authority_item_sets}...")
        topic_authority_dict, topic_ambiguous_dict = build_authority_dict(topic_authority_item_sets) # Capture both dicts
        logger.info(f"TOPIC authority dictionary built with {len(topic_authority_dict)} terms. Found {len(topic_ambiguous_dict)} ambiguous terms.")

        # Write ambiguous topic terms
        ambiguous_topic_tag = "topic"
        ambiguous_topic_path = f"{initial_csv_base}_ambiguous_authorities_{ambiguous_topic_tag}.csv"
        write_ambiguous_terms_to_file(topic_ambiguous_dict, ambiguous_topic_path)

        logger.info(f"Processing TOPIC reconciliation for column 'Subject AI'. Input: {current_input_path}, Output: {final_reconciled_csv_path}")
        final_reconciled_path = reconcile_column_values(
            input_csv_path=current_input_path, 
            output_reconciled_csv_path=final_reconciled_csv_path, 
            authority_dict=topic_authority_dict,
            source_column_name="Subject AI",
            target_column_name="Topic AI Reconciled ID",
            initial_csv_base_for_unreconciled=initial_csv_base, # Base for unreconciled file name
            output_file_tag="topic",
            ambiguous_authority_dict=topic_ambiguous_dict # Pass ambiguous dict
        )
        logger.info(f"Topic reconciliation step complete. Final reconciled data at: {final_reconciled_path}")
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
