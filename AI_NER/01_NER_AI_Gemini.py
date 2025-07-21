"""
Named Entity Recognition (NER) using Google's Gemini AI for Omeka S Collections

This script processes items from an Omeka S collection, extracts named entities (persons, organizations, 
and locations) from text content using Google's Gemini API, and saves the results to a CSV file.

Features:
- Asynchronous and synchronous processing modes
- Configurable batch size for optimized performance
- Rate limiting to avoid API throttling
- Proper error handling and retry mechanisms
- Command-line argument support
- Detailed progress tracking and statistics
- Modular prompt system using external markdown file

Requirements:
- Python 3.7+
- Dependencies listed in requirements.txt
- Environment variables for API credentials
- ner_system_prompt.md file in the same directory

Usage:
    python NER_AI_Gemini.py --item-set-id 123 --async --batch-size 20 --timeout 30
"""
import csv
import asyncio
from tqdm import tqdm
from dotenv import load_dotenv
import os
import requests
import time
from requests.exceptions import RequestException
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import TypedDict, List, Optional, Dict, Any
from ratelimit import limits, sleep_and_retry
import json
from datetime import datetime
import os.path
import argparse
from functools import partial
from google import genai
from google.genai import types
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Schema definition for entity extraction results
class EntityList(TypedDict):
    """
    TypedDict for storing named entities extracted from text.
    
    Attributes:
        persons (List[str]): List of person names
        organizations (List[str]): List of organization names
        locations (List[str]): List of location names
        subjects (List[str]): List of thematic subjects
    """
    persons: List[str]
    organizations: List[str]
    locations: List[str]
    subjects: List[str]

# Pydantic model for structured output schema
class EntityListPydantic(BaseModel):
    """
    Pydantic model for storing named entities, used for response schema.
    
    Attributes:
        persons (List[str]): List of person names
        organizations (List[str]): List of organization names
        locations (List[str]): List of location names
        subjects (List[str]): List of thematic subjects
    """
    persons: List[str]
    organizations: List[str]
    locations: List[str]
    subjects: List[str]

# Constants
CALLS_PER_MINUTE = 60  # API rate limit
ONE_MINUTE = 60  # seconds
BATCH_SIZE = 10  # Default number of items to process in a batch
DEFAULT_TIMEOUT = 30  # Default request timeout in seconds
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"  # Current Gemini model name

class Config(TypedDict):
    """
    Configuration settings for the application.
    
    Attributes:
        omeka_base_url (str): Base URL for the Omeka S instance
        omeka_key_identity (str): Omeka S API key identity
        omeka_key_credential (str): Omeka S API key credential
        gemini_api_key (str): Google Gemini API key
        batch_size (int): Number of items to process in a batch
        max_retries (int): Maximum number of retries for failed requests
        timeout (int): Request timeout in seconds
        temperature (float): Temperature for the Gemini API request
    """
    omeka_base_url: str
    omeka_key_identity: str
    omeka_key_credential: str
    gemini_api_key: str
    batch_size: int
    max_retries: int
    timeout: int
    temperature: float

def load_config(batch_size: int = BATCH_SIZE, max_retries: int = 3, timeout: int = DEFAULT_TIMEOUT, temperature: float = 0.2) -> Config:
    """
    Load configuration from environment variables.
    
    Args:
        batch_size: Number of items to process in a batch
        max_retries: Maximum number of retries for failed requests
        timeout: Request timeout in seconds
        temperature: Temperature for the Gemini API request
        
    Returns:
        Config: Application configuration dictionary
        
    Raises:
        ValueError: If any required environment variable is missing
    """
    required_vars = {
        'OMEKA_BASE_URL': 'omeka_base_url',
        'OMEKA_KEY_IDENTITY': 'omeka_key_identity',
        'OMEKA_KEY_CREDENTIAL': 'omeka_key_credential',
        'GEMINI_API_KEY': 'gemini_api_key'
    }
    
    config = {}
    for env_var, config_key in required_vars.items():
        value = os.getenv(env_var)
        if not value:
            raise ValueError(f"Missing required environment variable: {env_var}")
        config[config_key] = value
    
    config['batch_size'] = batch_size
    config['max_retries'] = max_retries
    config['timeout'] = timeout
    config['temperature'] = temperature
    
    return Config(**config)

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=ONE_MINUTE)
def make_api_request(endpoint: str, params: Optional[Dict[str, Any]] = None, 
                    max_retries: int = 3, retry_delay: int = 5, timeout: int = DEFAULT_TIMEOUT) -> Optional[Dict]:
    """
    Make a rate-limited API request to Omeka S.
    
    Args:
        endpoint: API endpoint to request
        params: Query parameters to include in the request
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        timeout: Request timeout in seconds
        
    Returns:
        Optional[Dict]: JSON response data or None if the request failed
    """
    base_url = os.getenv('OMEKA_BASE_URL')
    url = f"{base_url}/{endpoint}"
    params = params or {}
    params.update({
        'key_identity': os.getenv('OMEKA_KEY_IDENTITY'),
        'key_credential': os.getenv('OMEKA_KEY_CREDENTIAL')
    })
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"Error making request (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Max retries reached. Skipping this request.")
                return None

def get_items_from_set(item_set_id, max_retries: int = 3, timeout: int = DEFAULT_TIMEOUT):
    """
    Get all items from an Omeka S item set with pagination support.
    
    Args:
        item_set_id: ID of the item set to retrieve
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
        
    Returns:
        List[Dict]: List of item dictionaries from the item set
    """
    items = []
    page = 1
    
    # Retrieve all pages of results (100 items per page)
    while True:
        response = make_api_request('items', 
                                   {'item_set_id': item_set_id, 'page': page, 'per_page': 100},
                                   max_retries=max_retries,
                                   timeout=timeout)
        if response is None:
            break
        items.extend(response)
        # If we received fewer items than requested, we've reached the last page
        if len(response) < 100:
            break
        page += 1
    return items

def get_value(item, property_name):
    """
    Extract a value from an item property, with language preference for French.
    
    Args:
        item: Omeka S item dictionary
        property_name: Name of the property to extract
        
    Returns:
        str: Property value or empty string if not found
    """
    if property_name == 'o:id':
        return str(item.get('o:id', ''))
    
    values = item.get(property_name, [])
    if not values:
        return ''
    
    # First try to get French value, then fall back to any value
    value = next((v['@value'] for v in values if v.get('@language') == 'fr'), None)
    if value is None:
        value = next((v['@value'] for v in values), '')
    return value

def clean_entity(entity: str) -> str:
    """
    Clean and normalize entity strings with proper name capitalization.
    
    Handles special cases like name prefixes (de, van, etc.) and apostrophes.
    
    Args:
        entity: Raw entity string to clean
        
    Returns:
        str: Cleaned and normalized entity string
    """
    if not entity:
        return ""
        
    # Split on common name separators
    parts = entity.strip().split()
    
    # Capitalize each part of the name, handling special cases
    cleaned_parts = []
    for part in parts:
        # Handle prefixes like 'el', 'van', 'de', 'von', 'der', 'den', 'hadj', 'ben', 'ibn', 'et', 'du', 'des', 'le', 'la', 'les', 'l', 'd'}:
        if part.lower() in {'el', 'van', 'de', 'von', 'der', 'den', 'hadj', 'ben', 'ibn', 'et', 'du', 'des', 'le', 'la', 'les', 'l', 'd'}:
            cleaned_parts.append(part.lower())
        # Handle special case for names with apostrophes
        elif "'" in part or "’" in part:
            # Handle both straight and curly apostrophes
            parts_with_apostrophe = part.replace("'", " ' ").replace("'", " ' ").split()
            cleaned_subparts = []
            for subpart in parts_with_apostrophe:
                if subpart == "'":
                    cleaned_subparts.append("'")
                else:
                    cleaned_subparts.append(subpart.capitalize())
            cleaned_parts.append("".join(cleaned_subparts))
        else:
            # Handle French characters properly
            cleaned_part = part.capitalize()
            # Special case for French characters at the start
            if cleaned_part.startswith("L"):
                cleaned_part = "L" + cleaned_part[1:].capitalize()
            cleaned_parts.append(cleaned_part)
    
    return ' '.join(cleaned_parts)

def clean_apostrophes(text: str) -> str:
    """
    Replace curly apostrophes with straight apostrophes in a string.
    
    Args:
        text: The input string.
        
    Returns:
        str: The string with curly apostrophes replaced by straight apostrophes.
    """
    if not text:
        return ""
    return text.replace("’", "'")

def deduplicate_entities(entities: List[str]) -> List[str]:
    """
    Remove duplicates from entity list while preserving order and proper capitalization.
    
    Args:
        entities: List of entity strings, potentially with duplicates
        
    Returns:
        List[str]: Deduplicated list of properly capitalized entities
    """
    # Create a case-insensitive set for deduplication while preserving the first occurrence
    seen = set()
    cleaned_entities = []
    
    for entity in filter(None, entities):
        cleaned = clean_entity(entity)
        if cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            cleaned_entities.append(cleaned)
    
    return cleaned_entities

def validate_item_set_id(item_set_id: str) -> bool:
    """
    Validate that the item set ID is a positive integer.
    
    Args:
        item_set_id: Item set ID to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        return bool(int(item_set_id) > 0)
    except ValueError:
        return False

def load_ner_prompt() -> str:
    """
    Load the NER system prompt from the external markdown file.
    
    Returns:
        str: The system prompt template with {text_content} placeholder
        
    Raises:
        FileNotFoundError: If the prompt file is not found
        IOError: If there's an error reading the file
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(script_dir, 'ner_system_prompt.md')
    
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"NER prompt file not found: {prompt_file}")
        raise
    except IOError as e:
        logger.error(f"Error reading NER prompt file: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def perform_ner(api_key: str, text: str, timeout: int = DEFAULT_TIMEOUT, temperature: float = 0.2) -> EntityList:
    """
    Extract named entities from text using Google's Gemini AI API.
    
    Uses the official Google Generative AI Python client library with structured output.
    
    Args:
        api_key: Google Gemini API key
        text: Text content to analyze
        timeout: Request timeout in seconds
        temperature: Temperature for the Gemini API request
        
    Returns:
        EntityList: Dictionary containing persons, organizations, and locations lists
        
    Raises:
        Exception: If the API request fails or response parsing fails
    """
    # Return empty results for empty text
    if not text.strip():
        return EntityList(persons=[], organizations=[], locations=[], subjects=[])

    # Load prompt template from external file and format with text content
    try:
        prompt_template = load_ner_prompt()
        prompt = prompt_template.replace('{text_content}', text)
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Failed to load NER prompt: {e}")
        raise Exception(f"Could not load NER prompt: {str(e)}")

    try:
        # Initialize the Gemini client
        client = genai.Client(api_key=api_key)
        
        # Create the content object
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]
        
        # Configure generation parameters
        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=0.9,
            top_k=40,
            max_output_tokens=65536,
            response_mime_type="application/json",
            response_schema=EntityListPydantic  # Use Pydantic model for schema
        )
        
        # Call the model
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=generate_content_config
        )
        
        # Parse the response
        try:
            # Extract structured data from response
            result = json.loads(response.text)
            
            # Handle both list and single object responses
            if isinstance(result, list):
                # Combine all entities from the list
                entities = {
                    "persons": [],
                    "organizations": [],
                    "locations": [],
                    "subjects": []
                }
                
                for item in result:
                    if isinstance(item, dict):
                        # Handle different key formats (lowercase, uppercase, etc.)
                        for key, value in item.items():
                            key = key.lower()
                            if key == "person" or key == "persons":
                                entities["persons"].extend(value if isinstance(value, list) else [value])
                            elif key == "organization" or key == "organizations":
                                entities["organizations"].extend(value if isinstance(value, list) else [value])
                            elif key == "location" or key == "locations":
                                entities["locations"].extend(value if isinstance(value, list) else [value])
                            elif key == "subject" or key == "subjects":
                                entities["subjects"].extend(value if isinstance(value, list) else [value])
            else:
                # Handle single object response
                entities = {
                    "persons": result.get("persons", []),
                    "organizations": result.get("organizations", []),
                    "locations": result.get("locations", []),
                    "subjects": result.get("subjects", [])
                }
            
            # Return deduplicated entities in standard format
            return EntityList(
                persons=deduplicate_entities(entities["persons"]),
                organizations=deduplicate_entities(entities["organizations"]),
                locations=deduplicate_entities(entities["locations"]),
                subjects=deduplicate_entities(entities["subjects"])
            )
            
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            logger.error(f"Error parsing Gemini response: {e}")
            logger.error(f"Response content: {response}")
            raise Exception("Failed to parse Gemini response")
    
    except Exception as e:
        logger.error(f"Error in Gemini API call: {e}")
        raise Exception(f"Gemini API error: {str(e)}")

class ProcessingStats:
    """
    Tracks statistics for the processing job with thread-safe updates.
    
    Attributes:
        total_items: Total number of items to process
        processed_items: Number of items processed so far
        successful_items: Number of items that failed processing
        failed_items: Number of items that failed processing
        empty_content_items: Number of items with empty content
        start_time: Time when processing started
        lock: Lock for thread-safe updates in async mode
    """
    def __init__(self, total_items: int):
        """Initialize statistics tracker."""
        self.total_items = total_items
        self.processed_items = 0
        self.successful_items = 0
        self.failed_items = 0
        self.empty_content_items = 0
        self.start_time = datetime.now()
        # Create lock for async mode if running in an event loop
        try:
            asyncio.get_running_loop()
            self.lock = asyncio.Lock()
        except RuntimeError:
            self.lock = None
    
    async def update_async(self, success=False, failed=False, empty=False):
        """
        Thread-safe async update of stats.
        
        Args:
            success: Whether item was processed successfully
            failed: Whether item processing failed
            empty: Whether item had empty content
        """
        if self.lock:
            async with self.lock:
                self._update(success, failed, empty)
        else:
            self._update(success, failed, empty)
    
    def update(self, success=False, failed=False, empty=False):
        """
        Thread-safe update of stats for synchronous operation.
        
        Args:
            success: Whether item was processed successfully
            failed: Whether item processing failed
            empty: Whether item had empty content
        """
        self._update(success, failed, empty)
            
    def _update(self, success=False, failed=False, empty=False):
        """
        Internal update method used by both sync and async interfaces.
        
        Args:
            success: Whether item was processed successfully
            failed: Whether item processing failed
            empty: Whether item had empty content
        """
        self.processed_items += 1
        if success:
            self.successful_items += 1
        if failed:
            self.failed_items += 1
        if empty:
            self.empty_content_items += 1

def get_item_set_spatial_coverage(item_set_id: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """
    Get the spatial coverage from the item set metadata.
    
    This value is used to filter out matching locations from the results.
    
    Args:
        item_set_id: ID of the item set
        timeout: Request timeout in seconds
        
    Returns:
        str: Spatial coverage value or empty string if not found
    """
    response = make_api_request(f'item_sets/{item_set_id}', timeout=timeout)
    if response and 'dcterms:spatial' in response:
        spatial_values = response.get('dcterms:spatial', [])
        for value in spatial_values:
            if value.get('@value'):
                return value['@value'].strip()
    return ""

async def process_item_async(item: Dict, api_key: str, writer: csv.DictWriter, 
                      stats: ProcessingStats, spatial_filter: str = None,
                      timeout: int = DEFAULT_TIMEOUT, temperature: float = 0.2) -> None:
    """
    Process a single item asynchronously.
    
    Args:
        item: Omeka S item dictionary
        api_key: Google Gemini API key
        writer: CSV writer for output
        stats: Statistics tracker
        spatial_filter: Location name to filter out from results
        timeout: Request timeout in seconds
        temperature: Temperature for the Gemini API request
    """
    try:
        row = {
            'o:id': get_value(item, 'o:id'),
            'Title': get_value(item, 'dcterms:title'),
            'bibo:content': get_value(item, 'bibo:content')
        }
        
        if not row['bibo:content'].strip():
            await stats.update_async(empty=True)
            return
        
        # Run NER in a thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        entities = await loop.run_in_executor(
            None, 
            partial(perform_ner, api_key, row['bibo:content'], timeout, temperature)
        )
        
        # Combine persons, organizations, and subjects for Subject AI
        subjects_all = entities['persons'] + entities['organizations'] + entities['subjects']
        # Filter out the spatial_filter value from locations if it exists
        locations = entities['locations']
        if spatial_filter:
            locations = [loc for loc in locations if spatial_filter.lower() not in loc.lower()]
        row['Subject AI'] = clean_apostrophes('|'.join(subjects_all) if subjects_all else '')
        row['Spatial AI'] = clean_apostrophes('|'.join(locations) if locations else '')
        
        # Write to CSV - needs to be synchronized
        writer.writerow(row)
        await stats.update_async(success=True)
            
    except Exception as e:
        item_id = item.get('o:id', 'unknown')
        logger.error(f"Error processing item {item_id}: {e}")
        await stats.update_async(failed=True)

def process_items_batch(items: List[Dict], api_key: str, writer: csv.DictWriter, 
                       stats: ProcessingStats, pbar: tqdm, spatial_filter: str = None,
                       timeout: int = DEFAULT_TIMEOUT, temperature: float = 0.2) -> None:
    """
    Process a batch of items synchronously with progress tracking.
    
    Args:
        items: List of Omeka S item dictionaries to process
        api_key: Google Gemini API key
        writer: CSV writer for output
        stats: Statistics tracker
        pbar: Progress bar
        spatial_filter: Location name to filter out from results
        timeout: Request timeout in seconds
        temperature: Temperature for the Gemini API request
    """
    # Process each item in the batch
    for item in items:
        try:
            # Extract basic item information
            row = {
                'o:id': get_value(item, 'o:id'),
                'Title': get_value(item, 'dcterms:title'),
                'bibo:content': get_value(item, 'bibo:content')
            }
            
            # Skip items with empty content
            if not row['bibo:content'].strip():
                stats.update(empty=True)
                pbar.set_postfix(
                    successful=stats.successful_items,
                    failed=stats.failed_items,
                    empty=stats.empty_content_items
                )
                pbar.update(1)
                continue
            
            # Extract named entities
            entities = perform_ner(api_key, row['bibo:content'], timeout, temperature)
            
            # Combine persons, organizations, and subjects for Subject AI
            subjects_all = entities['persons'] + entities['organizations'] + entities['subjects']
            # Filter out the spatial_filter value from locations if it exists
            locations = entities['locations']
            if spatial_filter:
                locations = [loc for loc in locations if spatial_filter.lower() not in loc.lower()]
            
            # Format results as pipe-separated lists
            row['Subject AI'] = clean_apostrophes('|'.join(subjects_all) if subjects_all else '')
            row['Spatial AI'] = clean_apostrophes('|'.join(locations) if locations else '')
            
            # Write to CSV
            writer.writerow(row)
            stats.update(success=True)
            
        except Exception as e:
            item_id = item.get('o:id', 'unknown')
            logger.error(f"Error processing item {item_id}: {e}")
            stats.update(failed=True)
        
        # Update progress bar with current statistics
        elapsed_time = (datetime.now() - stats.start_time).total_seconds()
        items_per_second = stats.processed_items / elapsed_time if elapsed_time > 0 else 0
        
        pbar.set_postfix(
            successful=stats.successful_items,
            failed=stats.failed_items,
            empty=stats.empty_content_items,
            speed=f"{items_per_second:.2f} items/s"
        )
        pbar.update(1)

async def process_items_async(items: List[Dict], api_key: str, output_csv: str, 
                             stats: ProcessingStats, pbar: tqdm, 
                             spatial_filter: str = None,
                             batch_size: int = BATCH_SIZE,
                             timeout: int = DEFAULT_TIMEOUT,
                             temperature: float = 0.2) -> None:
    """
    Process items asynchronously with controlled concurrency for better performance.
    
    Args:
        items: List of Omeka S item dictionaries to process
        api_key: Google Gemini API key
        output_csv: Path to output CSV file
        stats: Statistics tracker
        pbar: Progress bar
        spatial_filter: Location name to filter out from results
        batch_size: Number of concurrent tasks to run
        timeout: Request timeout in seconds
        temperature: Temperature for the Gemini API request
    """
    # Define CSV structure
    fieldnames = ['o:id', 'Title', 'bibo:content', 'Subject AI', 'Spatial AI']
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process in batches for controlled concurrency
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            tasks = []
            
            # Create a task for each item in the batch
            for item in batch:
                tasks.append(process_item_async(
                    item, api_key, writer, stats, spatial_filter, timeout, temperature
                ))
            
            # Wait for all tasks in the batch to complete
            await asyncio.gather(*tasks)
            
            # Update progress bar
            pbar.n = stats.processed_items
            
            elapsed_time = (datetime.now() - stats.start_time).total_seconds()
            items_per_second = stats.processed_items / elapsed_time if elapsed_time > 0 else 0
            
            pbar.set_postfix(
                successful=stats.successful_items,
                failed=stats.failed_items,
                empty=stats.empty_content_items,
                speed=f"{items_per_second:.2f} items/s"
            )
            pbar.refresh()

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Extract named entities from Omeka S items using Gemini API")
    parser.add_argument("--item-set-id", type=str, help="Item set ID to process")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size for processing (default: {BATCH_SIZE})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"Timeout for API requests in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for API requests (default: 3)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for Gemini API (default: 0.2)")
    parser.add_argument("--async", action="store_true", help="Use asynchronous processing for better performance")
    parser.add_argument("--output-dir", type=str, help="Output directory for results (default: script directory)")
    return parser.parse_args()

async def async_main(args) -> None:
    """
    Main asynchronous execution flow.
    
    Args:
        args: Command line arguments
    """
    try:
        # Load configuration
        config = load_config(
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            timeout=args.timeout,
            temperature=args.temperature
        )
        
        # Get and validate item set ID
        item_set_id = args.item_set_id or input("Enter the item set ID: ")
        if not validate_item_set_id(item_set_id):
            logger.error("Invalid item set ID")
            return
        
        # Get spatial coverage from item set
        spatial_filter = get_item_set_spatial_coverage(item_set_id, timeout=config['timeout'])
        if spatial_filter:
            logger.info(f"Found spatial coverage filter: {spatial_filter}")
        
        # Get items from the set
        logger.info(f"Fetching items from set {item_set_id}...")
        items = get_items_from_set(
            item_set_id, 
            max_retries=config['max_retries'],
            timeout=config['timeout']
        )
        
        if not items:
            logger.warning("No items found in the set.")
            return
        
        # Create output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, "output")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize statistics
        stats = ProcessingStats(total_items=len(items))
        
        # Prepare output file
        output_csv = os.path.join(output_dir, f'item_set_{item_set_id}_processed.csv')
        
        logger.info(f"Starting processing of {stats.total_items} items...")
        
        # Process items with progress bar
        with tqdm(total=stats.total_items, 
                desc="Processing items", 
                unit="items") as pbar:
            
            await process_items_async(
                items, 
                config['gemini_api_key'], 
                output_csv,
                stats, 
                pbar, 
                spatial_filter,
                config['batch_size'],
                config['timeout'],
                config['temperature']
            )
        
        # Calculate final statistics
        elapsed_time = (datetime.now() - stats.start_time).total_seconds()
        avg_speed = stats.processed_items / elapsed_time if elapsed_time > 0 else 0
        success_rate = (stats.successful_items / stats.total_items) * 100 if stats.total_items > 0 else 0
        
        # Print final report
        logger.info("\nProcessing Summary:")
        logger.info(f"Total items processed: {stats.total_items}")
        logger.info(f"Successfully processed: {stats.successful_items} ({success_rate:.1f}%)")
        logger.info(f"Failed items: {stats.failed_items}")
        logger.info(f"Items with empty content: {stats.empty_content_items}")
        logger.info(f"Total processing time: {elapsed_time:.1f} seconds")
        logger.info(f"Average processing speed: {avg_speed:.2f} items/second")
        logger.info(f"Results written to: {output_csv}")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

def main() -> None:
    """
    Main entry point for the script.
    
    Parses arguments and dispatches to either synchronous or asynchronous processing.
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Use asynchronous version if requested
        if getattr(args, 'async', False):
            asyncio.run(async_main(args))
            return
            
        # Continue with synchronous version if async not requested
        # Load configuration
        config = load_config(
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            timeout=args.timeout,
            temperature=args.temperature
        )
        
        # Get and validate item set ID
        item_set_id = args.item_set_id or input("Enter the item set ID: ")
        if not validate_item_set_id(item_set_id):
            logger.error("Invalid item set ID")
            return
        
        # Get spatial coverage from item set
        spatial_filter = get_item_set_spatial_coverage(item_set_id, timeout=config['timeout'])
        if spatial_filter:
            logger.info(f"Found spatial coverage filter: {spatial_filter}")
        
        # Get items from the set
        logger.info(f"Fetching items from set {item_set_id}...")
        items = get_items_from_set(
            item_set_id, 
            max_retries=config['max_retries'],
            timeout=config['timeout']
        )
        
        if not items:
            logger.warning("No items found in the set.")
            return
        
        # Create output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, "output")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize statistics
        stats = ProcessingStats(total_items=len(items))
        
        # Prepare output file
        output_csv = os.path.join(output_dir, f'item_set_{item_set_id}_processed.csv')
        fieldnames = ['o:id', 'Title', 'bibo:content', 'Subject AI', 'Spatial AI']
        
        logger.info(f"Starting processing of {stats.total_items} items...")
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Process items in batches with progress bar
            with tqdm(total=stats.total_items, 
                     desc="Processing items", 
                     unit="items") as pbar:
                
                for i in range(0, len(items), config['batch_size']):
                    batch = items[i:i + config['batch_size']]
                    process_items_batch(
                        batch, 
                        config['gemini_api_key'], 
                        writer, 
                        stats, 
                        pbar, 
                        spatial_filter,
                        config['timeout'],
                        config['temperature']
                    )
        
        # Calculate final statistics
        elapsed_time = (datetime.now() - stats.start_time).total_seconds()
        avg_speed = stats.processed_items / elapsed_time if elapsed_time > 0 else 0
        success_rate = (stats.successful_items / stats.total_items) * 100 if stats.total_items > 0 else 0
        
        # Print final report
        logger.info("\nProcessing Summary:")
        logger.info(f"Total items processed: {stats.total_items}")
        logger.info(f"Successfully processed: {stats.successful_items} ({success_rate:.1f}%)")
        logger.info(f"Failed items: {stats.failed_items}")
        logger.info(f"Items with empty content: {stats.empty_content_items}")
        logger.info(f"Total processing time: {elapsed_time:.1f} seconds")
        logger.info(f"Average processing speed: {avg_speed:.2f} items/second")
        logger.info(f"Results written to: {output_csv}")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
