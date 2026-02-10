"""
This script updates Omeka S items with corrected OCR text content.
It reads corrected text files and updates the corresponding items in the Omeka S database
while preserving existing metadata and handling the bibo:content property appropriately.
"""

import os
import sys

# Get the directory of the current script and set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# Shared Omeka client
sys.path.insert(0, os.path.join(script_dir, '..'))
from common.omeka_client import OmekaClient

# Directory containing the corrected text files (relative to script location)
txt_directory = os.path.join(script_dir, 'Corrected_TXT')

# Initialize shared client
client = OmekaClient.from_env()


def update_item_with_new_content(item_id, new_content):
    """
    Update an Omeka S item with new OCR content while preserving existing metadata.

    Args:
        item_id (str): ID of the Omeka S item to update
        new_content (str): New OCR content to add to the item

    Returns:
        bool: True if update was successful, False otherwise
    """
    item_data = client.get_item(int(item_id))
    if not item_data:
        print(f"No data found for item {item_id}. Skipping update.")
        return False

    if 'value' not in item_data:
        item_data['value'] = []

    bibo_content_found = False
    for value in item_data['value']:
        if value.get('property_id') == 91:
            value['@value'] = new_content
            bibo_content_found = True
            break

    if not bibo_content_found:
        item_data['value'].append({
            "type": "literal",
            "property_id": 91,
            "property_label": "content",
            "is_public": True,
            "@value": new_content
        })

    if client.update_item(int(item_id), item_data):
        print(f"Successfully updated item {item_id} with new content")
        return True
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
