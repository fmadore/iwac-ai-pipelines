"""
This script updates Omeka S database items with French summaries generated by Gemini AI.
It reads summary files from the Summaries_FR_TXT directory and updates the corresponding
Omeka S items by adding or updating the bibo:shortDescription field.

The script:
1. Reads all .txt files from the Summaries_FR_TXT directory
2. For each summary file, extracts the item ID from the filename
3. Fetches the current item data from Omeka S to preserve existing fields
4. Updates or adds the bibo:shortDescription field with the generated summary
5. Sends the updated data back to Omeka S via PATCH request

Requirements:
- Environment variables: OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL
- Summary files in Summaries_FR_TXT directory (generated by 02_gemini_generate_summaries.py)
- Omeka S API access with appropriate permissions
"""

import os
import requests
from tqdm import tqdm
import logging

# Load environment variables from .env if present
from dotenv import load_dotenv
load_dotenv()

# Configure logging to track script execution and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API details from environment variables
base_url = os.environ.get('OMEKA_BASE_URL')  # Base URL for the Omeka S API
key_identity = os.environ.get('OMEKA_KEY_IDENTITY')  # API key identity
key_credential = os.environ.get('OMEKA_KEY_CREDENTIAL')  # API key credential

# Omeka S property configuration
BIBO_SHORT_DESCRIPTION_PROPERTY_ID = 116  # Property ID for bibo:shortDescription in Omeka S

# Omeka S property configuration
BIBO_SHORT_DESCRIPTION_PROPERTY_ID = 116  # Property ID for bibo:shortDescription in Omeka S


def get_full_item_data(item_id):
    """
    Retrieve complete item data from Omeka S including all properties.
    
    This function fetches the full item record from Omeka S, which is necessary
    to preserve all existing data when updating the item with new summary information.
    
    Args:
        item_id (str): The unique identifier of the Omeka S item
        
    Returns:
        dict: Complete item data from Omeka S API, or None if item not found
        
    Raises:
        requests.RequestException: If there's an error communicating with the API
    """
    url = f"{base_url}/items/{item_id}"
    params = {
        'key_identity': key_identity,
        'key_credential': key_credential
    }
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Error fetching item {item_id}: {e}")
        return None

def update_item_summary(item_id, new_summary):
    """
    Update an Omeka S item with a new French summary in the bibo:shortDescription field.
    
    This function preserves all existing item data while adding or updating the summary.
    It follows the Omeka S PATCH pattern to ensure data integrity.
    
    Args:
        item_id (str): The unique identifier of the Omeka S item to update
        new_summary (str): The French summary text to add to the item
        
    Returns:
        bool: True if update was successful, False otherwise
        
    Process:
        1. Fetch current item data to preserve existing fields
        2. Check if bibo:shortDescription already exists
        3. Update existing description or add new one
        4. Send PATCH request with complete item data
    """
    item_data = get_full_item_data(item_id)
    if not item_data:
        logging.warning(f"No data found for item {item_id}. Skipping update.")
        return False

    # Initialize item_data as an empty dictionary if it's None to prevent errors
    if item_data is None:
        item_data = {}

    # Ensure 'bibo:shortDescription' key exists as a list
    if 'bibo:shortDescription' not in item_data:
        item_data['bibo:shortDescription'] = []

    # Check if bibo:shortDescription already exists and update it
    summary_found = False
    for desc in item_data['bibo:shortDescription']:
        # Update the existing description if it matches the property ID
        if desc.get('property_id') == BIBO_SHORT_DESCRIPTION_PROPERTY_ID:
            desc['@value'] = new_summary
            desc['type'] = 'literal'  # Ensure type is set
            desc['property_label'] = 'shortDescription'  # Ensure label is correct
            summary_found = True
            logging.info(f"Updated existing summary for item {item_id}")
            break

    # If no existing description with the correct property_id was found, add a new one
    if not summary_found:
        item_data['bibo:shortDescription'].append({
            "type": "literal",
            "property_id": BIBO_SHORT_DESCRIPTION_PROPERTY_ID,
            "property_label": "shortDescription",
            "is_public": True,
            "@value": new_summary
        })
        logging.info(f"Added new summary for item {item_id}")

    # Send PATCH request to update the item data
    return _send_item_update(item_id, item_data)


def _send_item_update(item_id, item_data):
    """
    Send the updated item data to Omeka S via PATCH request.
    
    This is a helper function that handles the actual API communication
    to update the item in Omeka S.
    
    Args:
        item_id (str): The unique identifier of the Omeka S item
        item_data (dict): Complete item data with updates
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    url = f"{base_url}/items/{item_id}"
    params = {
        'key_identity': key_identity,
        'key_credential': key_credential
    }
    headers = {'Content-Type': 'application/json'}

    try:
        # Send the full item_data object to preserve all existing fields
        response = requests.patch(url, json=item_data, params=params, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        logging.info(f"Successfully updated item {item_id} with new summary")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to update item {item_id}: {e}")
        # Log the response body for debugging if available
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response body: {e.response.text}")
        return False


def main():
    """
    Main execution function for updating Omeka S items with French summaries.
    
    This function orchestrates the entire update process:
    1. Validates environment variables
    2. Locates the summary files directory
    3. Processes each summary file to update corresponding Omeka S items
    4. Provides progress tracking and error reporting
    
    The function expects summary files to be named with the pattern: {item_id}.txt
    where item_id corresponds to the Omeka S item identifier.
    
    Environment Variables Required:
        - OMEKA_BASE_URL: Base URL for the Omeka S API
        - OMEKA_KEY_IDENTITY: API key identity for authentication
        - OMEKA_KEY_CREDENTIAL: API key credential for authentication
    
    Returns:
        None: Exits with status code 0 on success, 1 on error
    """
    # Validate required environment variables
    if not all([base_url, key_identity, key_credential]):
        logging.error("Missing required environment variables. Please set OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, and OMEKA_KEY_CREDENTIAL.")
        return 1

    # Locate the summary files directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_folder = os.path.join(script_dir, "Summaries_FR_TXT")

    if not os.path.exists(summary_folder):
        logging.error(f"Summary folder not found: {summary_folder}")
        logging.error("Please run 02_gemini_generate_summaries.py first to generate summary files.")
        return 1

    # Get all .txt files in the summary folder
    txt_files = [f for f in os.listdir(summary_folder) if f.endswith('.txt')]

    if not txt_files:
        logging.warning(f"No .txt files found in {summary_folder}")
        return 0

    logging.info(f"Found {len(txt_files)} summary files to process")

    # Process each summary file
    success_count = 0
    error_count = 0
    
    for txt_file in tqdm(txt_files, desc="Updating Omeka S items with summaries"):
        # Extract item ID from filename (remove .txt extension)
        item_id = os.path.splitext(txt_file)[0]
        txt_path = os.path.join(summary_folder, txt_file)

        try:
            # Read the summary content
            with open(txt_path, 'r', encoding='utf-8') as f:
                summary_text = f.read().strip()
            
            # Only proceed if summary_text is not empty
            if summary_text:
                if update_item_summary(item_id, summary_text):
                    success_count += 1
                else:
                    error_count += 1
            else:
                logging.warning(f"Skipping item {item_id}: Summary file is empty")
                
        except FileNotFoundError:
            logging.error(f"Summary file not found: {txt_path}")
            error_count += 1
        except Exception as e:
            logging.error(f"Error processing file {txt_path}: {e}")
            error_count += 1

    # Final summary of operations
    logging.info(f"Update process completed:")
    logging.info(f"  - Successfully updated: {success_count} items")
    logging.info(f"  - Errors encountered: {error_count} items")
    logging.info(f"  - Total processed: {len(txt_files)} files")
    
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
