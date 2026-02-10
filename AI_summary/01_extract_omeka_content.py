"""
This script extracts OCR text content from items in an Omeka S database and saves them as individual text files.

WORKFLOW:
1. Connects to Omeka S API using environment credentials
2. Prompts user for item set ID(s) to process
3. Fetches all items from specified item sets using pagination
4. Extracts OCR text from the 'bibo:content' field of each item
5. Saves extracted content as individual .txt files in the TXT/ directory
6. Provides detailed progress tracking and error reporting

FEATURES:
- Concurrent processing using thread pools for improved performance
- Automatic retry mechanisms for API failures
- Progress bars for user feedback
- Comprehensive error handling and logging
- Skips items with no OCR content
- Preserves item IDs as filenames for downstream processing

REQUIREMENTS:
- Environment variables: OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL
- Items in Omeka S must have OCR text in the 'bibo:content' field
- Network access to the Omeka S API endpoint

OUTPUT:
- Text files named {item_id}.txt in the TXT/ directory
- Each file contains the OCR text content from the corresponding Omeka S item
"""

import os
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Directory configuration - output directory is relative to script location for portability
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TXT")

# Performance configuration
MAX_WORKERS = 5  # Maximum number of concurrent threads for processing items

# Configure logging to track script execution, errors, and progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Shared Omeka client
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.omeka_client import OmekaClient


def extract_and_save_content(item, output_dir):
    """
    Extract OCR text from an Omeka S item and save it to a text file.

    Args:
        item (dict): Item data dictionary from Omeka S API containing OCR text
        output_dir (str): Directory where text files will be saved

    Returns:
        tuple: (item_id, success_status, skipped_status)
    """
    item_id = item["o:id"]

    # Extract OCR text from the bibo:content field
    extracted_text = item.get("bibo:content", [])
    # Filter out empty content and extract @value fields
    content_values = [
        content["@value"]
        for content in extracted_text
        if "@value" in content and content["@value"].strip()
    ]

    # Skip items with no OCR content to avoid creating empty files
    if not content_values:
        logging.info(f"Skipping item {item_id}: No content in bibo:content")
        return item_id, False, True  # item_id, success=False, skipped=True

    # Join multiple content blocks with newlines
    content_text = "\n".join(content_values)
    file_name = os.path.join(output_dir, f"{item_id}.txt")

    try:
        # Save OCR content to text file with UTF-8 encoding
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(content_text)
        return item_id, True, False  # item_id, success=True, skipped=False
    except IOError as e:
        logging.error(f"Error writing file for item {item_id}: {e}")
        return item_id, False, False  # item_id, success=False, skipped=False


def process_items(items, output_dir):
    """
    Process multiple items concurrently using a thread pool.

    Args:
        items (list): List of items to process
        output_dir (str): Directory where text files will be saved

    Returns:
        tuple: (success_count, skipped_count)
    """
    success_count = 0
    skipped_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(extract_and_save_content, item, output_dir): item for item in items}
        for future in tqdm(as_completed(future_to_item), total=len(items), desc="Processing items"):
            item_id, success, skipped = future.result()
            if success:
                success_count += 1
            elif skipped:
                skipped_count += 1
    return success_count, skipped_count


def main():
    """
    Main execution function that orchestrates the OCR text extraction process.

    Prompts for item set ID, initializes the client, and processes all items
    while providing progress feedback through logging.
    """
    item_set_ids_str = input("Enter the item set ID(s), separated by comma or space: ")
    # Split by comma or space and remove any empty strings that might result from multiple spaces
    item_set_ids = [int(id_str.strip()) for id_str in item_set_ids_str.replace(',', ' ').split() if id_str.strip()]

    if not item_set_ids:
        logging.info("No item set IDs provided. Exiting.")
        return

    # Create the OUTPUT_DIR in the same folder as the script
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize shared Omeka client
    client = OmekaClient.from_env()

    total_items_fetched = 0
    total_success_count = 0
    total_skipped_count = 0

    for item_set_id in item_set_ids:
        logging.info(f"Processing item set ID: {item_set_id}")
        logging.info("Starting item fetch...")
        items = client.get_items(item_set_id)
        logging.info(f"Fetched {len(items)} items for item set {item_set_id}.")
        total_items_fetched += len(items)

        if items:
            logging.info("Extracting and saving content...")
            success_count, skipped_count = process_items(items, OUTPUT_DIR)
            total_success_count += success_count
            total_skipped_count += skipped_count
            failed_count = len(items) - success_count - skipped_count
            logging.info(f"Content saved for {success_count} out of {len(items)} items for item set {item_set_id}.")
            if skipped_count > 0:
                logging.info(f"Skipped {skipped_count} items with no content.")
            if failed_count > 0:
                logging.info(f"Failed to process {failed_count} items.")
        else:
            logging.info(f"No items found for item set {item_set_id}.")

    logging.info(f"Finished processing all item sets.")
    logging.info(f"Total items fetched: {total_items_fetched}")
    logging.info(f"Total content saved for {total_success_count} out of {total_items_fetched} items.")
    if total_skipped_count > 0:
        logging.info(f"Total items skipped (no content): {total_skipped_count}")
    failed_items = total_items_fetched - total_success_count - total_skipped_count
    if failed_items > 0:
        logging.info(f"Total items failed: {failed_items}")


if __name__ == "__main__":
    main()
