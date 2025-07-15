"""
This script updates Omeka S items with corrected OCR text content.
It reads corrected text files and updates the corresponding items in the Omeka S database
while preserving existing metadata and handling the bibo:content property appropriately.
"""

import os
import requests
from dotenv import load_dotenv
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Get the directory of the current script and set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate to parent directory for .env file access
parent_dir = os.path.dirname(script_dir)

# Load environment variables from the parent directory's .env file
load_dotenv(os.path.join(parent_dir, '.env'))

# Load API configuration from environment variables
base_url = os.getenv('OMEKA_BASE_URL')  # Base URL for Omeka S API
key_identity = os.getenv('OMEKA_KEY_IDENTITY')  # API key identity
key_credential = os.getenv('OMEKA_KEY_CREDENTIAL')  # API key credential

# Directory containing the corrected text files (relative to script location)
txt_directory = os.path.join(script_dir, 'Corrected_TXT')

# Configure retry strategy
retry_strategy = Retry(
    total=5,  # number of retries
    backoff_factor=1,  # wait 1, 2, 4, 8, 16 seconds between retries
    status_forcelist=[429, 500, 502, 503, 504]  # HTTP status codes to retry on
)

# Create a session with the retry strategy
session = requests.Session()
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

def get_full_item_data(item_id):
    """
    Retrieve complete item data from Omeka S API including all properties.
    
    Args:
        item_id (str): ID of the Omeka S item to retrieve
        
    Returns:
        dict: Complete item data if successful, None if request fails
    """
    url = f"{base_url}/items/{item_id}"
    params = {
        'key_identity': key_identity,
        'key_credential': key_credential
    }
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = session.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving item {item_id}: {str(e)}")
        return None

def update_item_with_new_content(item_id, new_content):
    """
    Update an Omeka S item with new OCR content while preserving existing metadata.
    
    Args:
        item_id (str): ID of the Omeka S item to update
        new_content (str): New OCR content to add to the item
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            # Retrieve current item data
            item_data = get_full_item_data(item_id)
            if not item_data:
                print(f"No data found for item {item_id}. Skipping update.")
                return False

            # Initialize value array if it doesn't exist
            if 'value' not in item_data:
                item_data['value'] = []

            # Update or create bibo:content property (property_id: 91)
            bibo_content_found = False
            for value in item_data['value']:
                if value.get('property_id') == 91:  # 91 is the property_id for bibo:content
                    if '@value' in value:
                        value['@value'] = new_content  # Replace existing content
                    else:
                        value['@value'] = new_content  # Create new content
                    bibo_content_found = True
                    break

            # Create new bibo:content property if it doesn't exist
            if not bibo_content_found:
                item_data['value'].append({
                    "type": "literal",
                    "property_id": 91,
                    "property_label": "content",
                    "is_public": True,
                    "@value": new_content
                })

            # Send update request to Omeka S API
            url = f"{base_url}/items/{item_id}"
            params = {
                'key_identity': key_identity,
                'key_credential': key_credential
            }
            headers = {'Content-Type': 'application/json'}
            
            response = session.patch(url, json=item_data, params=params, headers=headers)
            response.raise_for_status()
            
            print(f"Successfully updated item {item_id} with new content")
            return True
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed for item {item_id}: {str(e)}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed to update item {item_id} after {max_retries} attempts: {str(e)}")
                return False

# Main execution block
# Get total number of text files for progress tracking
total_files = len([f for f in os.listdir(txt_directory) if f.endswith('.txt')])
processed_files = 0
successful_updates = 0
failed_updates = []

print(f"Starting to process {total_files} text files...")

# Iterate through all text files and update corresponding items
for filename in os.listdir(txt_directory):
    if filename.endswith('.txt'):
        processed_files += 1
        progress = (processed_files / total_files) * 100
        print(f"\nProcessing file {processed_files}/{total_files} ({progress:.1f}%): {filename}")
        
        item_id = os.path.splitext(filename)[0]  # Extract item ID from filename
        file_path = os.path.join(txt_directory, filename)
        
        try:
            # Read and update content
            with open(file_path, 'r', encoding='utf-8') as file:
                new_content = file.read().strip()
                if update_item_with_new_content(item_id, new_content):
                    successful_updates += 1
                else:
                    failed_updates.append(filename)
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            failed_updates.append(filename)

print(f"\nProcess completed!")
print(f"Successfully processed: {successful_updates}/{total_files} files")
if failed_updates:
    print(f"Failed updates ({len(failed_updates)}):")
    for failed_file in failed_updates:
        print(f"- {failed_file}")
