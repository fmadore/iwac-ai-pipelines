import csv
import os
import requests
import logging
from dotenv import load_dotenv
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

OMEKA_BASE_URL = os.getenv('OMEKA_BASE_URL')
OMEKA_KEY_IDENTITY = os.getenv('OMEKA_KEY_IDENTITY')
OMEKA_KEY_CREDENTIAL = os.getenv('OMEKA_KEY_CREDENTIAL')

def get_full_item_data(item_id: str) -> dict | None:
    """Fetch the full data for a specific Omeka item."""
    if not OMEKA_BASE_URL:
        logger.error("OMEKA_BASE_URL is not set.")
        return None
    
    url = f"{OMEKA_BASE_URL}/items/{item_id}"
    params = {
        'key_identity': OMEKA_KEY_IDENTITY,
        'key_credential': OMEKA_KEY_CREDENTIAL
    }
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error fetching item {item_id}: {http_err} - Response: {http_err.response.text if http_err.response else 'No response'}")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request error fetching item {item_id}: {req_err}")
    return None

def update_item_fields(item_id: str, spatial_ids_str: str | None, subject_ids_str: str | None):
    """
    Updates an Omeka item with new spatial and subject links.
    Preserves existing data and avoids adding duplicate links.
    """
    logger.info(f"Processing item ID: {item_id}")
    item_data = get_full_item_data(item_id)

    if not item_data:
        logger.error(f"Failed to fetch data for item {item_id}. Skipping update.")
        return

    modified = False

    # --- Handle Spatial IDs (dcterms:spatial, property_id: 40) ---
    if spatial_ids_str:
        if 'dcterms:spatial' not in item_data or not isinstance(item_data['dcterms:spatial'], list):
            item_data['dcterms:spatial'] = []
        
        existing_spatial_resource_ids = set()
        for entry in item_data['dcterms:spatial']:
            if isinstance(entry, dict) and entry.get('type') == 'resource:item' and 'value_resource_id' in entry:
                try:
                    existing_spatial_resource_ids.add(int(entry['value_resource_id']))
                except ValueError:
                    logger.warning(f"Found non-integer value_resource_id in existing spatial data for item {item_id}: {entry.get('value_resource_id')}")


        new_spatial_ids_to_add = [s_id.strip() for s_id in spatial_ids_str.split('|') if s_id.strip()]
        for s_id_val_str in new_spatial_ids_to_add:
            try:
                s_id_int = int(s_id_val_str)
                if s_id_int not in existing_spatial_resource_ids:
                    item_data['dcterms:spatial'].append({
                        "type": "resource:item",
                        "property_id": 40,
                        "property_label": "Spatial Coverage",
                        "is_public": True,
                        "value_resource_id": s_id_int,
                        "value_resource_name": "items"
                    })
                    modified = True
                    logger.info(f"Added dcterms:spatial ID {s_id_int} to item {item_id}")
                    existing_spatial_resource_ids.add(s_id_int) # Add to set to prevent re-adding if listed multiple times in input
            except ValueError:
                logger.warning(f"Invalid spatial ID '{s_id_val_str}' for item {item_id}. Skipping this ID.")

    # --- Handle Subject IDs (dcterms:subject, property_id: 3) ---
    if subject_ids_str:
        if 'dcterms:subject' not in item_data or not isinstance(item_data['dcterms:subject'], list):
            item_data['dcterms:subject'] = []

        existing_subject_resource_ids = set()
        for entry in item_data['dcterms:subject']:
            if isinstance(entry, dict) and entry.get('type') == 'resource:item' and 'value_resource_id' in entry:
                try:
                    existing_subject_resource_ids.add(int(entry['value_resource_id']))
                except ValueError:
                    logger.warning(f"Found non-integer value_resource_id in existing subject data for item {item_id}: {entry.get('value_resource_id')}")


        new_subject_ids_to_add = [s_id.strip() for s_id in subject_ids_str.split('|') if s_id.strip()]
        for s_id_val_str in new_subject_ids_to_add:
            try:
                s_id_int = int(s_id_val_str)
                if s_id_int not in existing_subject_resource_ids:
                    item_data['dcterms:subject'].append({
                        "type": "resource:item",
                        "property_id": 3,
                        "property_label": "Subject",
                        "is_public": True,
                        "value_resource_id": s_id_int,
                        "value_resource_name": "items"
                    })
                    modified = True
                    logger.info(f"Added dcterms:subject ID {s_id_int} to item {item_id}")
                    existing_subject_resource_ids.add(s_id_int) # Add to set
            except ValueError:
                logger.warning(f"Invalid subject ID '{s_id_val_str}' for item {item_id}. Skipping this ID.")

    if modified:
        if not OMEKA_BASE_URL:
            logger.error("OMEKA_BASE_URL is not set. Cannot send update.")
            return
            
        url = f"{OMEKA_BASE_URL}/items/{item_id}"
        params = {
            'key_identity': OMEKA_KEY_IDENTITY,
            'key_credential': OMEKA_KEY_CREDENTIAL
        }
        headers = {'Content-Type': 'application/json'}
        
        try:
            response = requests.patch(url, json=item_data, params=params, headers=headers)
            response.raise_for_status()
            logger.info(f"Successfully updated item {item_id}.")
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error updating item {item_id}: {http_err} - Response: {http_err.response.text if http_err.response else 'No response'}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request error updating item {item_id}: {req_err}")
    else:
        logger.info(f"No changes to apply for item {item_id}.")

def main():
    if not all([OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL]):
        logger.error("Error: Missing one or more Omeka API environment variables (OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL).")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")

    if not os.path.isdir(output_dir):
        logger.error(f"Output directory not found: {output_dir}")
        return

    # Find the latest reconciled CSV file directly
    reconciled_csv_files = [
        f for f in os.listdir(output_dir)
        if f.endswith('_reconciled.csv')
    ]

    if not reconciled_csv_files:
        logger.error(f"No '*_reconciled.csv' files found in {output_dir}.")
        return

    latest_reconciled_csv_filename = max(reconciled_csv_files, key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
    input_csv_path = os.path.join(output_dir, latest_reconciled_csv_filename)

    if not os.path.exists(input_csv_path):
        logger.error(f"Reconciled CSV file not found: {input_csv_path}") # Should not happen if logic above is correct
        return

    logger.info(f"Starting Omeka update process using reconciled file: {input_csv_path}")

    try:
        with open(input_csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames:
                logger.error(f"CSV file is empty or header is missing: {input_csv_path}")
                return

            required_columns = ['o:id', 'Spatial AI Reconciled ID', 'Subject AI Reconciled ID']
            missing_cols = [col for col in required_columns if col not in reader.fieldnames]
            if missing_cols:
                logger.error(f"CSV file {input_csv_path} is missing required columns: {', '.join(missing_cols)}")
                return
            
            rows_to_process = list(reader) # Read all rows to use tqdm effectively with file closing

        for row in tqdm(rows_to_process, desc="Updating Omeka items"):
            item_id = row.get('o:id')
            spatial_ids = row.get('Spatial AI Reconciled ID')
            subject_ids = row.get('Subject AI Reconciled ID')

            if not item_id:
                logger.warning(f"Skipping row due to missing 'o:id': {row}")
                continue
            
            update_item_fields(item_id, spatial_ids, subject_ids)
            
    except FileNotFoundError:
        logger.error(f"Input CSV file not found during processing: {input_csv_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}", exc_info=True)

    logger.info("--- Omeka Update Process Finished ---")

if __name__ == "__main__":
    main()
