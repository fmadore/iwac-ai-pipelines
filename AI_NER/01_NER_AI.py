"""
Unified Named Entity Recognition (NER) script for Omeka S Collections using either:
  (1) OpenAI Responses API (ChatGPT)  OR
  (2) Google Gemini (google-genai client)

User can choose provider interactively (prompt 1 or 2) or via --model openai|gemini.

OpenAI call preserves EXACT required settings block (only dynamic part is the input list):

	from openai import OpenAI
	client = OpenAI()

	response = client.responses.create(
	  model="gpt-5-mini",
	  input=[],
	  text={
		"format": {
		  "type": "text"
		},
		"verbosity": "low"
	  },
	  reasoning={
		"effort": "low"
	  },
	  tools=[],
	  store=True
	)

Environment variables (set depending on model used):
  Common (Omeka):
    OMEKA_BASE_URL
    OMEKA_KEY_IDENTITY
    OMEKA_KEY_CREDENTIAL
  OpenAI:
    OPENAI_API_KEY
  Gemini:
    GEMINI_API_KEY

Output CSV columns: o:id, Title, bibo:content, Subject AI, Spatial AI

Usage examples:
    python 01_NER_AI.py --item-set-id 123         (interactive model select)
    python 01_NER_AI.py --item-set-id 123 --model openai --async
    python 01_NER_AI.py --item-set-id 123 --model gemini --batch-size 8

Notes:
  * Gemini implementation mirrors OpenAI logic (JSON instruction + extraction) for parity.
  * If provider-specific errors occur, they are logged and item marked failed.

"""
from __future__ import annotations

import os
import csv
import re
import json
import argparse
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, TypedDict, Callable
from functools import partial

import requests
from dotenv import load_dotenv
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits, sleep_and_retry

# Providers (imports guarded lazily to allow running with only one provider env ready)
try:
    from openai import OpenAI  # EXACT import required for OpenAI path
except Exception:  # pragma: no cover - optional dependency path
    OpenAI = None  # type: ignore

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover
    genai = None  # type: ignore
    genai_types = None  # type: ignore

# ---------------------------------------------------------------------------
# Logging & Environment
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# ---------------------------------------------------------------------------
# Constants & Types
# ---------------------------------------------------------------------------
CALLS_PER_MINUTE = 60
ONE_MINUTE = 60
BATCH_SIZE = 10
DEFAULT_TIMEOUT = 30
OPENAI_MODEL = "gpt-5-mini"
GEMINI_MODEL = "gemini-2.5-flash"

PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"

NER_JSON_INSTRUCTION = (
    "Return ONLY a compact JSON object with keys: persons, organizations, locations, subjects. "
    "Each value must be an array of distinct strings (no duplicates). If none, use []."
)

class EntityList(TypedDict):
    persons: List[str]
    organizations: List[str]
    locations: List[str]
    subjects: List[str]

class Config(TypedDict):
    omeka_base_url: str
    omeka_key_identity: str
    omeka_key_credential: str
    batch_size: int
    max_retries: int
    timeout: int
    temperature: float
    provider: str

LOWER_PREFIXES = {"el", "van", "de", "von", "der", "den", "hadj", "ben", "ibn", "et", "du", "des", "le", "la", "les", "l", "d"}

# ---------------------------------------------------------------------------
# Config & Prompt
# ---------------------------------------------------------------------------

def load_config(provider: str, batch_size: int = BATCH_SIZE, max_retries: int = 3, timeout: int = DEFAULT_TIMEOUT,
                temperature: float = 0.2) -> Config:
    required = {
        'OMEKA_BASE_URL': 'omeka_base_url',
        'OMEKA_KEY_IDENTITY': 'omeka_key_identity',
        'OMEKA_KEY_CREDENTIAL': 'omeka_key_credential',
    }
    cfg: Dict[str, Any] = {}
    missing = []
    for env_var, key in required.items():
        val = os.getenv(env_var)
        if not val:
            missing.append(env_var)
        else:
            cfg[key] = val
    # Provider specific check
    if provider == PROVIDER_OPENAI and not os.getenv('OPENAI_API_KEY'):
        missing.append('OPENAI_API_KEY')
    if provider == PROVIDER_GEMINI and not os.getenv('GEMINI_API_KEY'):
        missing.append('GEMINI_API_KEY')
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    cfg.update({
        'batch_size': batch_size,
        'max_retries': max_retries,
        'timeout': timeout,
        'temperature': temperature,
        'provider': provider,
    })
    return Config(**cfg)  # type: ignore

def load_ner_prompt() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, 'ner_system_prompt.md')
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# ---------------------------------------------------------------------------
# Omeka Helpers
# ---------------------------------------------------------------------------
@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=ONE_MINUTE)
def make_api_request(endpoint: str, params: Optional[Dict[str, Any]] = None, *,
                     max_retries: int = 3, timeout: int = DEFAULT_TIMEOUT) -> Optional[Any]:
    base_url = os.getenv('OMEKA_BASE_URL')
    if not base_url:
        raise ValueError("OMEKA_BASE_URL not set")
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    params = params.copy() if params else {}
    params.update({
        'key_identity': os.getenv('OMEKA_KEY_IDENTITY'),
        'key_credential': os.getenv('OMEKA_KEY_CREDENTIAL')
    })
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            logger.warning(f"Request {url} failed with status {resp.status_code} attempt {attempt}/{max_retries}")
        except requests.RequestException as e:
            logger.warning(f"Request exception attempt {attempt}/{max_retries}: {e}")
        if attempt == max_retries:
            logger.error(f"Max retries reached for {url}")
            return None
    return None

def get_items_from_set(item_set_id: str, *, max_retries: int, timeout: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    page = 1
    while True:
        data = make_api_request('items', {
            'item_set_id': item_set_id,
            'page': page,
            'per_page': 100
        }, max_retries=max_retries, timeout=timeout)
        if data is None or not data:
            break
        items.extend(data)
        if len(data) < 100:
            break
        page += 1
    return items

def get_items_from_multiple_sets(item_set_ids: List[str], *, max_retries: int, timeout: int) -> List[Dict[str, Any]]:
    """Get items from multiple item sets and combine them into a single list."""
    all_items: List[Dict[str, Any]] = []
    for item_set_id in item_set_ids:
        logger.info(f"Fetching items from set {item_set_id}...")
        items = get_items_from_set(item_set_id, max_retries=max_retries, timeout=timeout)
        logger.info(f"Found {len(items)} items in set {item_set_id}")
        all_items.extend(items)
    return all_items

# ---------------------------------------------------------------------------
# Data Utilities
# ---------------------------------------------------------------------------

def get_value(item: Dict[str, Any], prop: str) -> str:
    if prop == 'o:id':
        return str(item.get('o:id', ''))
    values = item.get(prop, [])
    if not values:
        return ''
    fr_val = next((v.get('@value', '') for v in values if v.get('@language') == 'fr'), None)
    return fr_val if fr_val is not None else values[0].get('@value', '')

def clean_entity(entity: str) -> str:
    if not entity:
        return ''
    entity = entity.strip()
    parts = entity.split()
    cleaned: List[str] = []
    for part in parts:
        if part.lower() in LOWER_PREFIXES:
            cleaned.append(part.lower())
        elif "'" in part or "’" in part:
            segs = re.split(r"['’]", part)
            if len(segs) == 2:
                cleaned.append(f"{segs[0].capitalize()}'{segs[1].capitalize()}")
            else:
                cleaned.append(part.capitalize())
        else:
            cleaned.append(part.capitalize())
    return ' '.join(cleaned)

def clean_apostrophes(text: str) -> str:
    return text.replace("’", "'") if text else ''

def deduplicate_entities(entities: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for e in entities:
        if not e:
            continue
        c = clean_entity(e)
        key = c.lower()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out

def validate_item_set_id(item_set_id: str) -> bool:
    try:
        return int(item_set_id) > 0
    except (TypeError, ValueError):
        return False

def parse_item_set_ids(input_str: str) -> List[str]:
    """Parse comma-separated item set IDs and validate them."""
    # Split by comma and clean whitespace
    ids = [id_str.strip() for id_str in input_str.split(',') if id_str.strip()]
    
    # Validate each ID
    valid_ids = []
    for id_str in ids:
        if validate_item_set_id(id_str):
            valid_ids.append(id_str)
        else:
            logger.warning(f"Invalid item set ID: '{id_str}' - skipping")
    
    return valid_ids

# ---------------------------------------------------------------------------
# JSON Extraction Helper
# ---------------------------------------------------------------------------

def extract_json_block(text: str) -> str:
    text = text.strip()
    if text.startswith('{') and text.endswith('}'):
        return text
    match = re.search(r'(\{(?:.|\n)*?\})', text)
    if match:
        return match.group(1)
    return '{"persons": [], "organizations": [], "locations": [], "subjects": []}'

# ---------------------------------------------------------------------------
# Provider Implementations
# ---------------------------------------------------------------------------

# OpenAI client (lazy init)
# Use a loose type for the lazy-initialized OpenAI client to avoid issues if package missing at import time
_openai_client: Any = None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def perform_ner_openai(text_content: str) -> EntityList:
    if not text_content.strip():
        return EntityList(persons=[], organizations=[], locations=[], subjects=[])
    global _openai_client
    if _openai_client is None:
        if OpenAI is None:
            raise RuntimeError("openai package not available")
        _openai_client = OpenAI()  # EXACT per spec
    prompt_template = load_ner_prompt()
    system_prompt = prompt_template
    user_prompt = f"{NER_JSON_INSTRUCTION}\n\nTEXT TO ANALYZE:\n{text_content}\n"
    response = _openai_client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        text={
            "format": {
                "type": "text"
            },
            "verbosity": "low"
        },
        reasoning={
            "effort": "low"
        },
        tools=[],
        store=True
    )
    raw_output = getattr(response, 'output_text', None)
    if not raw_output:
        segments = []
        for seg in getattr(response, 'output', []) or []:
            if isinstance(seg, dict):
                txt = seg.get('content') or seg.get('text') or ''
                if isinstance(txt, list):
                    txt = ''.join(str(t) for t in txt)
                segments.append(str(txt))
        raw_output = '\n'.join(filter(None, segments))
    if not raw_output:
        return EntityList(persons=[], organizations=[], locations=[], subjects=[])
    json_text = extract_json_block(raw_output)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        logger.warning("OpenAI JSON parse failure")
        return EntityList(persons=[], organizations=[], locations=[], subjects=[])
    return EntityList(
        persons=deduplicate_entities(data.get('persons', []) if isinstance(data.get('persons'), list) else []),
        organizations=deduplicate_entities(data.get('organizations', []) if isinstance(data.get('organizations'), list) else []),
        locations=deduplicate_entities(data.get('locations', []) if isinstance(data.get('locations'), list) else []),
        subjects=deduplicate_entities(data.get('subjects', []) if isinstance(data.get('subjects'), list) else []),
    )

_gemini_client = None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def perform_ner_gemini(text_content: str, temperature: float = 0.2) -> EntityList:
    if not text_content.strip():
        return EntityList(persons=[], organizations=[], locations=[], subjects=[])
    if genai is None:
        raise RuntimeError("google-genai package not available")
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        _gemini_client = genai.Client(api_key=api_key)
    prompt_template = load_ner_prompt()
    system_prompt = prompt_template
    user_prompt = f"{NER_JSON_INSTRUCTION}\n\nTEXT TO ANALYZE:\n{text_content}\n"
    # Combine prompts (Gemini generate_content doesn't use role separation the same way)
    full_prompt = system_prompt + "\n\n" + user_prompt + "\nReturn ONLY the JSON object."
    # Build generation config (omit response_mime_type for wider compatibility)
    gen_config = None
    if genai_types is not None:
        try:
            gen_config = genai_types.GenerateContentConfig(temperature=temperature)
        except Exception:
            gen_config = None
    try:
        response = _gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
            config=gen_config
        )
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return EntityList(persons=[], organizations=[], locations=[], subjects=[])
    raw_output = getattr(response, 'text', None)
    if not raw_output:
        return EntityList(persons=[], organizations=[], locations=[], subjects=[])
    json_text = extract_json_block(raw_output)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        logger.warning("Gemini JSON parse failure")
        return EntityList(persons=[], organizations=[], locations=[], subjects=[])
    return EntityList(
        persons=deduplicate_entities(data.get('persons', []) if isinstance(data.get('persons'), list) else []),
        organizations=deduplicate_entities(data.get('organizations', []) if isinstance(data.get('organizations'), list) else []),
        locations=deduplicate_entities(data.get('locations', []) if isinstance(data.get('locations'), list) else []),
        subjects=deduplicate_entities(data.get('subjects', []) if isinstance(data.get('subjects'), list) else []),
    )

# Provider dispatcher
def get_ner_callable(provider: str, temperature: float) -> Callable[[str], EntityList]:
    if provider == PROVIDER_OPENAI:
        return perform_ner_openai
    if provider == PROVIDER_GEMINI:
        return partial(perform_ner_gemini, temperature=temperature)  # type: ignore
    raise ValueError(f"Unsupported provider: {provider}")

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
class ProcessingStats:
    def __init__(self, total_items: int):
        self.total_items = total_items
        self.processed_items = 0
        self.successful_items = 0
        self.failed_items = 0
        self.empty_content_items = 0
        self.start_time = datetime.now()
        try:
            import asyncio as _a
            self.lock = _a.Lock()
        except Exception:
            self.lock = None
    def _update(self, success=False, failed=False, empty=False):
        self.processed_items += 1
        if success:
            self.successful_items += 1
        if failed:
            self.failed_items += 1
        if empty:
            self.empty_content_items += 1
    def update(self, **flags):
        self._update(**flags)
    async def update_async(self, **flags):
        if self.lock:
            async with self.lock:
                self._update(**flags)
        else:
            self._update(**flags)

# ---------------------------------------------------------------------------
# Spatial Coverage
# ---------------------------------------------------------------------------

def get_item_set_spatial_coverage(item_set_id: str, *, timeout: int) -> str:
    data = make_api_request(f'item_sets/{item_set_id}', timeout=timeout)
    if not data:
        return ''
    values = data.get('dcterms:spatial', [])
    for v in values:
        val = v.get('@value') if isinstance(v, dict) else None
        if val:
            return val.strip()
    return ''

def get_combined_spatial_coverage(item_set_ids: List[str], *, timeout: int) -> Optional[str]:
    """Get spatial coverage from multiple item sets. Returns the first non-empty one found."""
    for item_set_id in item_set_ids:
        coverage = get_item_set_spatial_coverage(item_set_id, timeout=timeout)
        if coverage:
            logger.info(f"Using spatial coverage from set {item_set_id}: {coverage}")
            return coverage
    return None

# ---------------------------------------------------------------------------
# Processing Functions
# ---------------------------------------------------------------------------
async def process_item_async(item: Dict[str, Any], writer: csv.DictWriter, stats: ProcessingStats,
                             spatial_filter: Optional[str], timeout: int, ner_fn: Callable[[str], EntityList], pbar: tqdm) -> None:
    try:
        row = {
            'o:id': get_value(item, 'o:id'),
            'Title': get_value(item, 'dcterms:title'),
            'bibo:content': get_value(item, 'bibo:content')
        }
        content = row['bibo:content'].strip()
        if not content:
            await stats.update_async(empty=True)
            writer.writerow({**row, 'Subject AI': '', 'Spatial AI': ''})
            return
        loop = asyncio.get_running_loop()
        entities: EntityList = await loop.run_in_executor(None, partial(ner_fn, content))
        subjects_all = entities['persons'] + entities['organizations'] + entities['subjects']
        locations = [l for l in entities['locations'] if not spatial_filter or l.lower() != spatial_filter.lower()]
        row['Subject AI'] = clean_apostrophes('|'.join(subjects_all))
        row['Spatial AI'] = clean_apostrophes('|'.join(locations))
        writer.writerow(row)
        await stats.update_async(success=True)
    except Exception as e:
        logger.error(f"Error processing item {item.get('o:id')}: {e}")
        await stats.update_async(failed=True)
    finally:
        elapsed = (datetime.now() - stats.start_time).total_seconds()
        speed = stats.processed_items / elapsed if elapsed else 0.0
        pbar.set_postfix(success=stats.successful_items, failed=stats.failed_items,
                         empty=stats.empty_content_items, speed=f"{speed:.2f} it/s")
        pbar.update(1)

def process_items_batch(items: List[Dict[str, Any]], writer: csv.DictWriter, stats: ProcessingStats,
                        spatial_filter: Optional[str], timeout: int, ner_fn: Callable[[str], EntityList], pbar: tqdm) -> None:
    for item in items:
        try:
            row = {
                'o:id': get_value(item, 'o:id'),
                'Title': get_value(item, 'dcterms:title'),
                'bibo:content': get_value(item, 'bibo:content')
            }
            content = row['bibo:content'].strip()
            if not content:
                stats.update(empty=True)
                writer.writerow({**row, 'Subject AI': '', 'Spatial AI': ''})
            else:
                entities = ner_fn(content)
                subjects_all = entities['persons'] + entities['organizations'] + entities['subjects']
                locations = [l for l in entities['locations'] if not spatial_filter or l.lower() != spatial_filter.lower()]
                row['Subject AI'] = clean_apostrophes('|'.join(subjects_all))
                row['Spatial AI'] = clean_apostrophes('|'.join(locations))
                writer.writerow(row)
                stats.update(success=True)
        except Exception as e:
            logger.error(f"Error processing item {item.get('o:id')}: {e}")
            stats.update(failed=True)
        elapsed = (datetime.now() - stats.start_time).total_seconds()
        speed = stats.processed_items / elapsed if elapsed else 0.0
        pbar.set_postfix(success=stats.successful_items, failed=stats.failed_items,
                         empty=stats.empty_content_items, speed=f"{speed:.2f} it/s")
        pbar.update(1)

async def process_items_async(items: List[Dict[str, Any]], output_csv: str, stats: ProcessingStats,
                              spatial_filter: Optional[str], batch_size: int, timeout: int,
                              ner_fn: Callable[[str], EntityList], pbar: tqdm) -> None:
    fieldnames = ['o:id', 'Title', 'bibo:content', 'Subject AI', 'Spatial AI']
    semaphore = asyncio.Semaphore(batch_size)
    async def worker(item: Dict[str, Any], writer: csv.DictWriter):
        async with semaphore:
            await process_item_async(item, writer, stats, spatial_filter, timeout, ner_fn, pbar)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        tasks = [worker(item, writer) for item in items]
        for chunk_start in range(0, len(tasks), batch_size):
            chunk = tasks[chunk_start:chunk_start + batch_size]
            await asyncio.gather(*chunk)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(stats: ProcessingStats, output_csv: str) -> None:
    elapsed = (datetime.now() - stats.start_time).total_seconds()
    speed = stats.processed_items / elapsed if elapsed else 0
    success_rate = (stats.successful_items / stats.total_items * 100) if stats.total_items else 0
    logger.info("\nProcessing Summary:")
    logger.info(f"Total: {stats.total_items}")
    logger.info(f"Success: {stats.successful_items} ({success_rate:.1f}%)")
    logger.info(f"Failed: {stats.failed_items}")
    logger.info(f"Empty: {stats.empty_content_items}")
    logger.info(f"Elapsed: {elapsed:.1f}s Speed: {speed:.2f} items/s")
    logger.info(f"Output: {output_csv}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Unified NER extraction for Omeka S (OpenAI or Gemini)")
    parser.add_argument("--item-set-id", type=str, help="Item set ID(s) to process (comma-separated)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default {BATCH_SIZE})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"Timeout seconds (default {DEFAULT_TIMEOUT})")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for Omeka requests")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature (Gemini only; OpenAI fixed settings)")
    parser.add_argument("--async", action="store_true", help="Use async processing")
    parser.add_argument("--output-dir", type=str, help="Directory for output CSV")
    parser.add_argument("--model", choices=[PROVIDER_OPENAI, PROVIDER_GEMINI], help="Provider: openai or gemini (optional)")
    return parser.parse_args()

def interactive_select_provider() -> str:
    while True:
        choice = input("Select AI model: 1) OpenAI ChatGPT  2) Google Gemini  > ").strip()
        if choice == '1':
            return PROVIDER_OPENAI
        if choice == '2':
            return PROVIDER_GEMINI
        print("Invalid choice. Enter 1 or 2.")

async def async_main(args) -> None:
    provider = args.model or interactive_select_provider()
    config = load_config(provider=provider, batch_size=args.batch_size, max_retries=args.max_retries,
                         timeout=args.timeout, temperature=args.temperature)
    item_set_input = args.item_set_id or input("Enter item set ID(s) (comma-separated): ")
    item_set_ids = parse_item_set_ids(item_set_input)
    if not item_set_ids:
        raise ValueError("No valid item set IDs provided")
    
    logger.info(f"Processing item sets: {', '.join(item_set_ids)}")
    spatial_filter = get_combined_spatial_coverage(item_set_ids, timeout=config['timeout'])
    if spatial_filter:
        logger.info(f"Spatial coverage filter: {spatial_filter}")
    
    items = get_items_from_multiple_sets(item_set_ids, max_retries=config['max_retries'], timeout=config['timeout'])
    if not items:
        logger.warning("No items found.")
        return
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(script_dir, 'output')
    output_dir = args.output_dir or default_output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename for multiple sets
    if len(item_set_ids) == 1:
        output_csv = os.path.join(output_dir, f'item_set_{item_set_ids[0]}_processed_{provider}.csv')
    else:
        sets_str = '_'.join(item_set_ids)
        output_csv = os.path.join(output_dir, f'item_sets_{sets_str}_processed_{provider}.csv')
    
    stats = ProcessingStats(total_items=len(items))
    logger.info(f"Processing {stats.total_items} items (async) using {provider}...")
    ner_fn = get_ner_callable(provider, config['temperature'])
    with tqdm(total=stats.total_items, desc="Processing items", unit="item") as pbar:
        await process_items_async(items, output_csv, stats, spatial_filter, config['batch_size'],
                                  config['timeout'], ner_fn, pbar)
    summarize(stats, output_csv)

def main() -> None:
    args = parse_arguments()
    provider = args.model or interactive_select_provider()
    if getattr(args, 'async', False):
        asyncio.run(async_main(args))
        return
    config = load_config(provider=provider, batch_size=args.batch_size, max_retries=args.max_retries,
                         timeout=args.timeout, temperature=args.temperature)
    item_set_input = args.item_set_id or input("Enter item set ID(s) (comma-separated): ")
    item_set_ids = parse_item_set_ids(item_set_input)
    if not item_set_ids:
        raise ValueError("No valid item set IDs provided")
    
    logger.info(f"Processing item sets: {', '.join(item_set_ids)}")
    spatial_filter = get_combined_spatial_coverage(item_set_ids, timeout=config['timeout'])
    if spatial_filter:
        logger.info(f"Spatial coverage filter: {spatial_filter}")
    
    items = get_items_from_multiple_sets(item_set_ids, max_retries=config['max_retries'], timeout=config['timeout'])
    if not items:
        logger.warning("No items found.")
        return
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(script_dir, 'output')
    output_dir = args.output_dir or default_output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename for multiple sets
    if len(item_set_ids) == 1:
        output_csv = os.path.join(output_dir, f'item_set_{item_set_ids[0]}_processed_{provider}.csv')
    else:
        sets_str = '_'.join(item_set_ids)
        output_csv = os.path.join(output_dir, f'item_sets_{sets_str}_processed_{provider}.csv')
    
    stats = ProcessingStats(total_items=len(items))
    fieldnames = ['o:id', 'Title', 'bibo:content', 'Subject AI', 'Spatial AI']
    logger.info(f"Processing {stats.total_items} items (sync) using {provider}...")
    ner_fn = get_ner_callable(provider, config['temperature'])
    with open(output_csv, 'w', newline='', encoding='utf-8') as f, \
            tqdm(total=stats.total_items, desc="Processing items", unit="item") as pbar:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        process_items_batch(items, writer, stats, spatial_filter, config['timeout'], ner_fn, pbar)
    summarize(stats, output_csv)

if __name__ == '__main__':
    main()
