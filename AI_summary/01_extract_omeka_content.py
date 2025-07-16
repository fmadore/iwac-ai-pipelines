"""
This script retrieves OCR text from items in an Omeka S database and saves them as individual text files.
It handles pagination, concurrent processing, and includes error handling and logging functionality.
"""

import requests
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration constants
BASE_URL = os.getenv('OMEKA_BASE_URL')  # Base URL for the Omeka S API
KEY_IDENTITY = os.getenv('OMEKA_KEY_IDENTITY')  # API key identity
KEY_CREDENTIAL = os.getenv('OMEKA_KEY_CREDENTIAL')  # API key credential

# Output directory is set relative to the script's location for portability
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TXT")
ITEMS_PER_PAGE = 100  # Number of items to fetch per API request
MAX_WORKERS = 5  # Maximum number of concurrent threads for processing

# Configure logging to track script execution and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ItemFetcher:
    """
    A class to handle fetching and processing items from the Omeka S API.
    
    This class manages API connections, item retrieval, and content extraction
    with built-in retry mechanisms and concurrent processing capabilities.
    """

    def __init__(self, base_url, output_dir):
        """
        Initialize the ItemFetcher with API endpoint and output location.
        
        Args:
            base_url (str): The base URL of the Omeka S API
            output_dir (str): Directory where extracted text files will be saved
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.session = self.create_session()
        # Add authentication parameters
        self.auth_params = {
            'key_identity': KEY_IDENTITY,
            'key_credential': KEY_CREDENTIAL
        }

    @staticmethod
    def create_session():
        """
        Create a requests session with retry mechanism for robust API communication.
        
        Returns:
            requests.Session: Configured session object with retry capabilities
        """
        session = requests.Session()
        # Configure automatic retries for failed requests
        retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    def fetch_items(self, item_set_id):
        """
        Fetch all items from a specific item set using pagination.
        
        Args:
            item_set_id (int): ID of the Omeka S item set to fetch
            
        Returns:
            list: List of items retrieved from the API
        """
        endpoint = f"{self.base_url}/items"
        params = {
            **self.auth_params,  # Include authentication parameters
            "item_set_id": item_set_id,
            "per_page": ITEMS_PER_PAGE,
            "page": 1
        }
        items = []

        with tqdm(desc="Fetching items", unit="page") as pbar:
            while True:
                try:
                    response = self.session.get(endpoint, params=params)
                    response.raise_for_status()
                    page_items = response.json()
                    if not page_items:
                        break
                    items.extend(page_items)
                    params["page"] += 1
                    pbar.update(1)
                except requests.RequestException as e:
                    logging.error(f"Error fetching page {params['page']}: {e}")
                    break

        return items

    def extract_and_save_content(self, item):
        """
        Extract OCR text from an item and save it to a file.
        Skips items that don't have any content in bibo:content.
        
        Args:
            item (dict): Item data containing OCR text
            
        Returns:
            tuple: (item_id, success_status, skipped_status)
        """
        item_id = item["o:id"]
        
        extracted_text = item.get("bibo:content", [])
        content_values = [content["@value"] for content in extracted_text if "@value" in content and content["@value"].strip()]
        
        # Skip if no content is found
        if not content_values:
            logging.info(f"Skipping item {item_id}: No content in bibo:content")
            return item_id, False, True  # item_id, success, skipped
        
        content_text = "\n".join(content_values)
        file_name = os.path.join(self.output_dir, f"{item_id}.txt")

        try:
            with open(file_name, "w", encoding="utf-8") as file:
                file.write(content_text)
            return item_id, True, False  # item_id, success, not skipped
        except IOError as e:
            logging.error(f"Error writing file for item {item_id}: {e}")
            return item_id, False, False  # item_id, not success, not skipped

    def process_items(self, items):
        """
        Process multiple items concurrently using a thread pool.
        
        Args:
            items (list): List of items to process
            
        Returns:
            tuple: (success_count, skipped_count)
        """
        success_count = 0
        skipped_count = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_item = {executor.submit(self.extract_and_save_content, item): item for item in items}
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
    
    Prompts for item set ID, initializes the fetcher, and processes all items
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
    fetcher = ItemFetcher(BASE_URL, OUTPUT_DIR)

    total_items_fetched = 0
    total_success_count = 0
    total_skipped_count = 0

    for item_set_id in item_set_ids:
        logging.info(f"Processing item set ID: {item_set_id}")
        logging.info("Starting item fetch...")
        items = fetcher.fetch_items(item_set_id)
        logging.info(f"Fetched {len(items)} items for item set {item_set_id}.")
        total_items_fetched += len(items)

        if items:
            logging.info("Extracting and saving content...")
            success_count, skipped_count = fetcher.process_items(items)
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
