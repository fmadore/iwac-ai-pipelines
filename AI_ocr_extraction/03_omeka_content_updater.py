"""
Database Update Script for OCR Results

This script processes OCR text files and updates corresponding Omeka S items
with the extracted text content. It preserves all existing metadata while
adding or updating the bibo:content property with the OCR results.

Usage:
    python 03_update_database.py

Requirements:
    - Environment variables: OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL
    - OCR_Results directory with .txt files named after item IDs
"""

import os
import requests
from tqdm import tqdm

# Load environment variables from .env if present
from dotenv import load_dotenv
load_dotenv()


# Load API credentials from environment variables (now loaded from .env if present)
base_url = os.environ.get('OMEKA_BASE_URL')
key_identity = os.environ.get('OMEKA_KEY_IDENTITY')
key_credential = os.environ.get('OMEKA_KEY_CREDENTIAL')

def get_full_item_data(item_id):
    """
    Retrieve complete item data from Omeka S API.
    
    Args:
        item_id (str): The ID of the item to retrieve
        
    Returns:
        dict or None: Complete item data including all properties, or None if failed
    """
    url = f"{base_url}/items/{item_id}"
    params = {
        'key_identity': key_identity,
        'key_credential': key_credential
    }
    headers = {'Content-Type': 'application/json'}
    
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        return response.json()
    return None

def update_item_with_new_content(item_id, new_content):
    """
    Update an Omeka S item with new bibo:content while preserving all other metadata.
    
    This function retrieves the current item data, updates or adds the bibo:content
    property with the new OCR text, and sends the updated data back to the API.
    
    Args:
        item_id (str): The ID of the item to update
        new_content (str): The new OCR text content to add/update
    """
    # Retrieve current item data to preserve existing properties
    item_data = get_full_item_data(item_id)
    if not item_data:
        print(f"No data found for item {item_id}. Skipping update.")
        return

    # Ensure the 'value' key exists for storing property values
    if 'value' not in item_data:
        item_data['value'] = []

    # Search for existing bibo:content property and update it
    bibo_content_found = False
    for value in item_data['value']:
        # Property ID 91 corresponds to bibo:content (this may vary by installation)
        if value.get('property_id') == 91:
            value['@value'] = new_content  # Update existing content
            bibo_content_found = True
            break

    # If bibo:content doesn't exist, create a new property entry
    if not bibo_content_found:
        item_data['value'].append({
            "type": "literal",
            "property_id": 91,  # bibo:content property ID
            "property_label": "content",
            "is_public": True,
            "@value": new_content
        })

    # Send updated item data back to the API
    url = f"{base_url}/items/{item_id}"
    params = {
        'key_identity': key_identity,
        'key_credential': key_credential
    }
    headers = {'Content-Type': 'application/json'}
    
    response = requests.patch(url, json=item_data, params=params, headers=headers)
    if response.status_code == 200:
        print(f"Successfully updated item {item_id} with new content")
    else:
        print(f"Failed to update item {item_id}: {response.text}")

def main():
    """
    Main function to process OCR results and update Omeka S database.
    
    Reads all text files from the OCR_Results directory and updates the
    corresponding Omeka S items with the extracted text content.
    """
    # Verify that all required environment variables are set
    if not all([base_url, key_identity, key_credential]):
        print("Error: Missing required environment variables. Please set OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, and OMEKA_KEY_CREDENTIAL.")
        return

    # Set up directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ocr_folder = os.path.join(script_dir, "OCR_Results")

    # Verify that the OCR results directory exists
    if not os.path.exists(ocr_folder):
        print(f"Error: OCR_Results folder not found: {ocr_folder}")
        return

    # Find all text files in the OCR results directory
    txt_files = [f for f in os.listdir(ocr_folder) if f.endswith('.txt')]
    
    if not txt_files:
        print("No .txt files found in OCR_Results directory.")
        return
    
    print(f"Found {len(txt_files)} text files to process.")

    # Process each text file and update corresponding Omeka S item
    for txt_file in tqdm(txt_files, desc="Updating Omeka S items"):
        # Extract item ID from filename (assumes filename format: {item_id}.txt)
        item_id = os.path.splitext(txt_file)[0]
        txt_path = os.path.join(ocr_folder, txt_file)

        # Read the OCR text content
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Update the Omeka S item with the OCR content
            update_item_with_new_content(item_id, content)
            
        except Exception as e:
            print(f"Error processing file {txt_file}: {e}")
            continue

    print("Database update process completed.")

if __name__ == "__main__":
    main()
