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

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Directory configuration - output directory is relative to script location for portability
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TXT")

# Performance configuration
MAX_WORKERS = 5  # Maximum number of concurrent threads for processing items

#: bibo:Article on the IWAC instance. Articles span 58 item sets and 39 belong to
#: none, so selecting them by item set both misses items and is unusable by hand —
#: --resource-class is how you address the whole class.
ARTICLE_RESOURCE_CLASS_ID = 36

#: The summarization prompt is written in French and assumes French input. The
#: collection also holds Ewé (32), Kabiyè (11) and Dendi (2) articles plus a few
#: with no language value; a French-prompted model returns confident, unusable
#: output for those rather than failing visibly — the same reason the sentiment
#: pipeline restricts itself to these two. Widen with --language at your own risk.
DEFAULT_LANGUAGES = ("Français", "Anglais")

# Configure logging to track script execution, errors, and progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Shared Omeka client
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.omeka_client import OmekaClient
from common.log_redaction import install_credential_redaction

# Credentials ride in Omeka query strings and provider headers; keep them
# out of anything urllib3 or an SDK decides to log.
install_credential_redaction()


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


def fetch_by_resource_class(client: OmekaClient, resource_class_id: int) -> List[Dict[str, Any]]:
    """Fetch every item of a resource class, paginating through the API.

    ``OmekaClient.get_items`` is keyed on an item set, which cannot express
    "the whole class". This goes through the client's own ``get_resource`` rather
    than raw ``requests`` so authentication, timeouts and retries still apply.
    """
    items: List[Dict[str, Any]] = []
    page = 1
    while True:
        batch = client.get_resource(
            f"{client.base_url}/items"
            f"?resource_class_id={resource_class_id}&per_page=100&page={page}"
        )
        if not batch:
            break
        items.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return items


def item_languages(item: Dict[str, Any]) -> List[str]:
    """Read ``dcterms:language`` labels off an item.

    The value is a LINK to an authority item, not an ISO code, so the label is
    in ``display_title``.
    """
    return [
        (value.get("display_title") or value.get("@value") or "").strip()
        for value in item.get("dcterms:language", [])
    ]


def filter_by_language(
    items: List[Dict[str, Any]], allowed: Optional[List[str]]
) -> tuple[List[Dict[str, Any]], int]:
    """Keep items carrying at least one allowed language. Returns (kept, dropped).

    Items with no language value at all are dropped too: an untagged item is one
    whose language nobody has established, not one that is presumed French.
    """
    if not allowed:
        return items, 0
    allowed_set = {name.casefold() for name in allowed}
    kept = [
        item for item in items
        if any(lang.casefold() in allowed_set for lang in item_languages(item))
    ]
    return kept, len(items) - len(kept)


def main():
    """Fetch items, filter them to the summarizable ones, and save their OCR text."""
    parser = argparse.ArgumentParser(
        description="Extract OCR text from Omeka S items into TXT/ for summarization."
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--item-set", type=int, nargs="+", metavar="ID",
        help="Item set ID(s) to process. Prompts interactively when no scope is given.",
    )
    scope.add_argument(
        "--resource-class", type=int, nargs="?", const=ARTICLE_RESOURCE_CLASS_ID,
        metavar="ID",
        help=f"Process a whole resource class (default {ARTICLE_RESOURCE_CLASS_ID}, "
             f"bibo:Article). Use this for the full corpus — articles span 58 item "
             f"sets and 39 belong to none.",
    )
    parser.add_argument(
        "--language", nargs="*", default=list(DEFAULT_LANGUAGES), metavar="NAME",
        help="dcterms:language labels to keep (default: %(default)s). "
             "Pass --language with no value to disable filtering.",
    )
    size = parser.add_mutually_exclusive_group()
    size.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Keep only the first N items after filtering.",
    )
    size.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Keep N items spread evenly across the whole selection. Prefer this "
             "for a pilot: items come back in ID order, so the first N are one "
             "newspaper's consecutive issues rather than a cross-section.",
    )
    args = parser.parse_args()

    client = OmekaClient.from_env()

    if args.resource_class is not None:
        logging.info(f"Fetching every item of resource class {args.resource_class}...")
        items = fetch_by_resource_class(client, args.resource_class)
        logging.info(f"Fetched {len(items)} items.")
    else:
        item_set_ids = args.item_set
        if not item_set_ids:
            raw = input("Enter the item set ID(s), separated by comma or space: ")
            item_set_ids = [
                int(part) for part in raw.replace(",", " ").split() if part.strip()
            ]
        if not item_set_ids:
            logging.info("No item set IDs provided. Exiting.")
            return
        items = []
        for item_set_id in item_set_ids:
            logging.info(f"Fetching item set {item_set_id}...")
            batch = client.get_items(item_set_id)
            logging.info(f"  {len(batch)} items.")
            items.extend(batch)

    if not items:
        logging.info("No items found. Exiting.")
        return

    items, dropped = filter_by_language(items, args.language)
    if dropped:
        logging.info(
            f"Skipped {dropped} item(s) outside {args.language} — the prompt is "
            f"French and cannot reliably summarize other languages."
        )
    if args.limit is not None and len(items) > args.limit:
        logging.info(f"Limiting to the first {args.limit} of {len(items)} items.")
        items = items[: args.limit]
    elif args.sample is not None and len(items) > args.sample:
        step = len(items) / args.sample
        logging.info(
            f"Sampling {args.sample} of {len(items)} items, every ~{step:.0f}th."
        )
        items = [items[int(i * step)] for i in range(args.sample)]

    if not items:
        logging.info("Nothing left to process after filtering. Exiting.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(f"Extracting content from {len(items)} items...")
    success_count, skipped_count = process_items(items, OUTPUT_DIR)
    failed_count = len(items) - success_count - skipped_count

    logging.info(f"Total content saved for {success_count} out of {len(items)} items.")
    if skipped_count:
        logging.info(f"Total items skipped (no content): {skipped_count}")
    if failed_count:
        logging.info(f"Total items failed: {failed_count}")


if __name__ == "__main__":
    main()
