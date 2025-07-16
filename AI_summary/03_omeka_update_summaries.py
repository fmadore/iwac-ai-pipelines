import os
import requests
from tqdm import tqdm

# API details from environment variables
base_url = os.environ.get('OMEKA_BASE_URL')
key_identity = os.environ.get('OMEKA_KEY_IDENTITY')
key_credential = os.environ.get('OMEKA_KEY_CREDENTIAL')

# Function to get the current item data along with all its properties
def get_full_item_data(item_id):
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

# Function to update the item with a new bibo:shortDescription while preserving other data
def update_item_summary(item_id, new_summary):
    item_data = get_full_item_data(item_id)
    if not item_data:
        print(f"No data found for item {item_id}. Skipping update.")
        return

    # Ensure the top-level structure includes common keys expected by Omeka S PATCH
    # Example: ensure @context, o:id etc., are present if needed, based on typical PATCH requirements.
    # This might require fetching the item with more details or ensuring the initial GET includes them.

    # Initialize item_data as an empty dictionary if it's None to prevent errors
    if item_data is None:
        item_data = {}

    # Ensure 'bibo:shortDescription' key exists as a list
    if 'bibo:shortDescription' not in item_data:
        item_data['bibo:shortDescription'] = []

    # Check if bibo:shortDescription already exists
    summary_found = False
    for desc in item_data['bibo:shortDescription']:
        # Update the existing description if it matches the property ID
        if desc.get('property_id') == 116:
            desc['@value'] = new_summary
            desc['type'] = 'literal' # Ensure type is set
            desc['property_label'] = 'shortDescription' # Ensure label is correct
            summary_found = True
            break

    # If no existing description with the correct property_id was found, add a new one
    if not summary_found:
        item_data['bibo:shortDescription'].append({
            "type": "literal",
            "property_id": 116,             # Use the correct property ID
            "property_label": "shortDescription", # Use the correct property label
            "is_public": True,
            "@value": new_summary
        })

    # --- Reverted Payload Logic ---
    # Send the entire modified item_data object to preserve other fields,
    # matching the pattern in OCR_AI/3. Update_omeka_with_ocr.py

    # Send a PATCH request to update the item data
    url = f"{base_url}/items/{item_id}"
    params = {
        'key_identity': key_identity,
        'key_credential': key_credential
    }
    headers = {'Content-Type': 'application/json'}

    # Debugging: Print the payload being sent (the full item_data)
    # import json
    # print(f"Updating item {item_id} with payload: {json.dumps(item_data, indent=2)}")

    try:
        # Send the full item_data object
        response = requests.patch(url, json=item_data, params=params, headers=headers)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        print(f"Successfully updated item {item_id} with new summary")
    except requests.exceptions.RequestException as e:
        print(f"Failed to update item {item_id}: {e}")
        # Optionally log the response body for more details
        if e.response is not None:
            print(f"Response body: {e.response.text}")


def main():
    # Check if required environment variables are set
    if not all([base_url, key_identity, key_credential]):
        print("Error: Missing required environment variables. Please set OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, and OMEKA_KEY_CREDENTIAL.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Changed input folder name
    summary_folder = os.path.join(script_dir, "Summaries_FR_TXT")

    if not os.path.exists(summary_folder):
        print(f"Error: Summary folder not found: {summary_folder}")
        return

    # Get all .txt files in the Summary folder
    txt_files = [f for f in os.listdir(summary_folder) if f.endswith('.txt')]

    for txt_file in tqdm(txt_files, desc="Updating Omeka S item summaries"):
        item_id = os.path.splitext(txt_file)[0]  # Assuming the filename is the item ID
        txt_path = os.path.join(summary_folder, txt_file)

        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                summary_text = f.read().strip() # Read and strip whitespace
            # Only proceed if summary_text is not empty
            if summary_text:
                update_item_summary(item_id, summary_text) # Use the renamed function
            else:
                print(f"Skipping item {item_id}: Summary file is empty.")
        except FileNotFoundError:
            print(f"Error: Summary file not found: {txt_path}")
        except Exception as e:
            print(f"Error processing file {txt_path}: {e}")


if __name__ == "__main__":
    main()
