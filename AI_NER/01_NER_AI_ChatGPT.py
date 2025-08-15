"""
Named Entity Recognition (NER) using OpenAI ChatGPT (Responses API) for Omeka S Collections

This script mirrors the (Gemini) implementation but uses OpenAI's Responses API
with the EXACT required settings block (do not modify):

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

The script dynamically fills the input list but preserves all other settings exactly.

Features:
- Sync and async processing modes
- Batch control with progress bar
- Rate limiting, retries, and statistics
- Loads prompt from `ner_system_prompt.md` (same prompt as Gemini version)
- Outputs CSV with columns: o:id, Title, bibo:content, Subject AI, Spatial AI

Prerequisites (environment variables):
  OMEKA_BASE_URL
  OMEKA_KEY_IDENTITY
  OMEKA_KEY_CREDENTIAL
  OPENAI_API_KEY

Usage examples:
  python 01_NER_AI_ChatGPT.py --item-set-id 123
  python 01_NER_AI_ChatGPT.py --item-set-id 123 --async --batch-size 8

Note: Model output is requested in JSON; fallback regex extraction is applied if
surrounding text appears.
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
from typing import List, Dict, Any, Optional, TypedDict
from functools import partial

import requests
from dotenv import load_dotenv
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits, sleep_and_retry

from openai import OpenAI  # Required EXACT import

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
OPENAI_MODEL = "gpt-5-mini"  # From required settings


class EntityList(TypedDict):
	persons: List[str]
	organizations: List[str]
	locations: List[str]
	subjects: List[str]


class Config(TypedDict):
	omeka_base_url: str
	omeka_key_identity: str
	omeka_key_credential: str
	openai_api_key: str
	batch_size: int
	max_retries: int
	timeout: int
	temperature: float


def load_config(batch_size: int = BATCH_SIZE, max_retries: int = 3, timeout: int = DEFAULT_TIMEOUT,
				temperature: float = 0.2) -> Config:
	required = {
		'OMEKA_BASE_URL': 'omeka_base_url',
		'OMEKA_KEY_IDENTITY': 'omeka_key_identity',
		'OMEKA_KEY_CREDENTIAL': 'omeka_key_credential',
		'OPENAI_API_KEY': 'openai_api_key'
	}
	cfg: Dict[str, Any] = {}
	missing = []
	for env_var, key in required.items():
		val = os.getenv(env_var)
		if not val:
			missing.append(env_var)
		else:
			cfg[key] = val
	if missing:
		raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
	cfg['batch_size'] = batch_size
	cfg['max_retries'] = max_retries
	cfg['timeout'] = timeout
	cfg['temperature'] = temperature  # Temperature included for parity; OpenAI call stays fixed per spec
	return Config(**cfg)  # type: ignore


# ---------------------------------------------------------------------------
# HTTP Helpers (Omeka)
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
		if data is None:
			break
		if not data:
			break
		items.extend(data)
		if len(data) < 100:
			break
		page += 1
	return items


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


LOWER_PREFIXES = {"el", "van", "de", "von", "der", "den", "hadj", "ben", "ibn", "et", "du", "des", "le", "la", "les", "l", "d"}


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


def load_ner_prompt() -> str:
	script_dir = os.path.dirname(os.path.abspath(__file__))
	path = os.path.join(script_dir, 'ner_system_prompt.md')
	with open(path, 'r', encoding='utf-8') as f:
		return f.read()


# ---------------------------------------------------------------------------
# OpenAI NER
# ---------------------------------------------------------------------------
client = OpenAI()  # EXACT per required settings


NER_JSON_INSTRUCTION = (
	"Return ONLY a compact JSON object with keys: persons, organizations, locations, subjects. "
	"Each value must be an array of distinct strings (no duplicates). If none, use []."
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def perform_ner(text_content: str) -> EntityList:
	if not text_content.strip():
		return EntityList(persons=[], organizations=[], locations=[], subjects=[])
	prompt_template = load_ner_prompt()
	# Combine: original prompt + JSON instruction + inserted text
	system_prompt = prompt_template
	user_prompt = (
		f"{NER_JSON_INSTRUCTION}\n\nTEXT TO ANALYZE:\n{text_content}\n"
	)

	# REQUIRED SETTINGS BLOCK (structure preserved; only input list filled dynamically):
	response = client.responses.create(
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
		# Fallback: build from segments
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
		logger.warning("Failed to parse JSON from model output; returning empty result")
		return EntityList(persons=[], organizations=[], locations=[], subjects=[])

	persons = deduplicate_entities(data.get('persons', [])) if isinstance(data.get('persons'), list) else []
	orgs = deduplicate_entities(data.get('organizations', [])) if isinstance(data.get('organizations'), list) else []
	locs = deduplicate_entities(data.get('locations', [])) if isinstance(data.get('locations'), list) else []
	subjects = deduplicate_entities(data.get('subjects', [])) if isinstance(data.get('subjects'), list) else []
	return EntityList(persons=persons, organizations=orgs, locations=locs, subjects=subjects)


def extract_json_block(text: str) -> str:
	# Try strict JSON first
	text = text.strip()
	if text.startswith('{') and text.endswith('}'):
		return text
	match = re.search(r'(\{(?:.|\n)*\})', text)
	if match:
		return match.group(1)
	return '{"persons": [], "organizations": [], "locations": [], "subjects": []}'


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
# Spatial Coverage / Filtering
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


# ---------------------------------------------------------------------------
# Processing (Async + Sync)
# ---------------------------------------------------------------------------
async def process_item_async(item: Dict[str, Any], writer: csv.DictWriter, stats: ProcessingStats,
							 spatial_filter: Optional[str], timeout: int, temperature: float) -> None:
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
		entities: EntityList = await loop.run_in_executor(None, partial(perform_ner, content))
		subjects_all = entities['persons'] + entities['organizations'] + entities['subjects']
		locations = [l for l in entities['locations'] if not spatial_filter or l.lower() != spatial_filter.lower()]
		row['Subject AI'] = clean_apostrophes('|'.join(subjects_all))
		row['Spatial AI'] = clean_apostrophes('|'.join(locations))
		writer.writerow(row)
		await stats.update_async(success=True)
	except Exception as e:
		logger.error(f"Error processing item {item.get('o:id')}: {e}")
		await stats.update_async(failed=True)


def process_items_batch(items: List[Dict[str, Any]], writer: csv.DictWriter, stats: ProcessingStats,
						spatial_filter: Optional[str], timeout: int, temperature: float, pbar: tqdm) -> None:
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
				entities = perform_ner(content)
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
							  spatial_filter: Optional[str], batch_size: int, timeout: int, temperature: float,
							  pbar: tqdm) -> None:
	fieldnames = ['o:id', 'Title', 'bibo:content', 'Subject AI', 'Spatial AI']
	semaphore = asyncio.Semaphore(batch_size)
	loop = asyncio.get_running_loop()
	# To keep CSV thread-safe, write sequentially after each task's result is ready
	async def worker(item: Dict[str, Any], writer: csv.DictWriter):
		async with semaphore:
			await process_item_async(item, writer, stats, spatial_filter, timeout, temperature)
			elapsed = (datetime.now() - stats.start_time).total_seconds()
			speed = stats.processed_items / elapsed if elapsed else 0.0
			pbar.set_postfix(success=stats.successful_items, failed=stats.failed_items,
							 empty=stats.empty_content_items, speed=f"{speed:.2f} it/s")
			pbar.update(1)
	# Open file once
	with open(output_csv, 'w', newline='', encoding='utf-8') as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		tasks = [worker(item, writer) for item in items]
		# Run tasks in controlled concurrency
		for chunk_start in range(0, len(tasks), batch_size):
			chunk = tasks[chunk_start:chunk_start + batch_size]
			await asyncio.gather(*chunk)


# ---------------------------------------------------------------------------
# CLI & Main
# ---------------------------------------------------------------------------
def parse_arguments():
	parser = argparse.ArgumentParser(description="Extract named entities from Omeka S items using OpenAI ChatGPT")
	parser.add_argument("--item-set-id", type=str, help="Item set ID to process")
	parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default {BATCH_SIZE})")
	parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"Timeout seconds (default {DEFAULT_TIMEOUT})")
	parser.add_argument("--max-retries", type=int, default=3, help="Max retries for Omeka requests")
	parser.add_argument("--temperature", type=float, default=0.2, help="(Ignored for fixed OpenAI settings; kept for parity)")
	parser.add_argument("--async", action="store_true", help="Use async processing")
	parser.add_argument("--output-dir", type=str, help="Directory for output CSV")
	return parser.parse_args()


async def async_main(args) -> None:
	config = load_config(batch_size=args.batch_size, max_retries=args.max_retries,
						 timeout=args.timeout, temperature=args.temperature)
	item_set_id = args.item_set_id or input("Enter the item set ID: ")
	if not validate_item_set_id(item_set_id):
		raise ValueError("Invalid item set ID")
	spatial_filter = get_item_set_spatial_coverage(item_set_id, timeout=config['timeout'])
	if spatial_filter:
		logger.info(f"Spatial coverage filter: {spatial_filter}")
	logger.info(f"Fetching items from set {item_set_id}...")
	items = get_items_from_set(item_set_id, max_retries=config['max_retries'], timeout=config['timeout'])
	if not items:
		logger.warning("No items found.")
		return
	# Default to shared output directory (mirrors Gemini script behavior)
	script_dir = os.path.dirname(os.path.abspath(__file__))
	default_output_dir = os.path.join(script_dir, 'output')
	output_dir = args.output_dir or default_output_dir
	os.makedirs(output_dir, exist_ok=True)
	output_csv = os.path.join(output_dir, f'item_set_{item_set_id}_processed_chatgpt.csv')
	stats = ProcessingStats(total_items=len(items))
	logger.info(f"Processing {stats.total_items} items (async)...")
	with tqdm(total=stats.total_items, desc="Processing items", unit="item") as pbar:
		await process_items_async(items, output_csv, stats, spatial_filter, config['batch_size'],
								  config['timeout'], config['temperature'], pbar)
	summarize(stats, output_csv)


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


def main() -> None:
	args = parse_arguments()
	if getattr(args, 'async', False):
		asyncio.run(async_main(args))
		return
	# Sync path
	config = load_config(batch_size=args.batch_size, max_retries=args.max_retries,
						 timeout=args.timeout, temperature=args.temperature)
	item_set_id = args.item_set_id or input("Enter the item set ID: ")
	if not validate_item_set_id(item_set_id):
		raise ValueError("Invalid item set ID")
	spatial_filter = get_item_set_spatial_coverage(item_set_id, timeout=config['timeout'])
	if spatial_filter:
		logger.info(f"Spatial coverage filter: {spatial_filter}")
	logger.info(f"Fetching items from set {item_set_id}...")
	items = get_items_from_set(item_set_id, max_retries=config['max_retries'], timeout=config['timeout'])
	if not items:
		logger.warning("No items found.")
		return
	script_dir = os.path.dirname(os.path.abspath(__file__))
	default_output_dir = os.path.join(script_dir, 'output')
	output_dir = args.output_dir or default_output_dir
	os.makedirs(output_dir, exist_ok=True)
	output_csv = os.path.join(output_dir, f'item_set_{item_set_id}_processed_chatgpt.csv')
	stats = ProcessingStats(total_items=len(items))
	fieldnames = ['o:id', 'Title', 'bibo:content', 'Subject AI', 'Spatial AI']
	logger.info(f"Processing {stats.total_items} items (sync)...")
	with open(output_csv, 'w', newline='', encoding='utf-8') as f, \
			tqdm(total=stats.total_items, desc="Processing items", unit="item") as pbar:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		process_items_batch(items, writer, stats, spatial_filter, config['timeout'], config['temperature'], pbar)
	summarize(stats, output_csv)


if __name__ == '__main__':
	main()
