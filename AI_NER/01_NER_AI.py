"""
Named Entity Recognition (NER) script for Omeka S metadata extraction.

Supported models:
  - gpt-5.6-luna: Fast, cost-effective OpenAI model (GPT-5.6 high-volume tier)
  - gemini-3.7-flash: Fast, cost-effective Gemini model
  - gemma-4: Google Gemma 4 31B — open-weights flagship, via Gemini API
  - mistral-large: Mistral Large 3 flagship model
  - ministral-14b: Ministral 3 14B cost-effective model

Configuration:
  Uses medium reasoning effort and verbosity for accurate metadata extraction.
  All models are optimized for named entity recognition tasks.

Environment variables:
  Common (Omeka):
    OMEKA_BASE_URL
    OMEKA_KEY_IDENTITY
    OMEKA_KEY_CREDENTIAL
  OpenAI:
    OPENAI_API_KEY
  Gemini / Gemma:
    GEMINI_API_KEY
  Mistral:
    MISTRAL_API_KEY

Output CSV columns: o:id, Title, bibo:content, Subject AI, Spatial AI

Usage examples:
    python 01_NER_AI.py --item-set-id 123
    python 01_NER_AI.py --item-set-id 123 --model gpt-5.6-luna
    python 01_NER_AI.py --item-set-id 123 --model gemini-3.7-flash --async
    python 01_NER_AI.py --item-set-id 123 --model gemma-4
    python 01_NER_AI.py --item-set-id 123 --model mistral-large
    python 01_NER_AI.py --item-set-id 123 --model ministral-14b
    python 01_NER_AI.py --item-set-id 123,456,789 --batch-size 8

"""
from __future__ import annotations

import os
import sys
import csv
import re
import argparse
import asyncio
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Iterator, TextIO
from functools import partial

from pydantic import BaseModel, Field

# Rich console for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, MofNCompleteColumn
from rich import box

console = Console()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.omeka_client import OmekaClient  # noqa: E402
from common.retry import retry_with_backoff  # noqa: E402
from common.checkpoint import (  # noqa: E402
    CheckpointMismatch,
    JsonCheckpoint,
    load_csv_ids,
    sha256_text,
)
from common.llm_provider import (  # noqa: E402
    DEFAULT_TEXT_MODEL_KEY,
    LEGACY_CLI_MODEL_KEYS,
    TEXT_EXTENDED_MODELS,
    BaseLLMClient,
    ModelOption,
    LLMConfig,
    UsageTotals,
    build_llm_client,
    get_model_option,
    summary_from_option,
    PROVIDER_GEMINI,
    PROVIDER_OPENAI,
    PROVIDER_MISTRAL,
    PROVIDER_OPENROUTER,
    PROVIDER_SELFHOSTED,
)
from common.log_redaction import install_credential_redaction

# ---------------------------------------------------------------------------
# Logging & Environment
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Credentials ride in Omeka query strings and provider headers; keep them
# out of anything urllib3 or an SDK decides to log.
install_credential_redaction()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Types
# ---------------------------------------------------------------------------
BATCH_SIZE = 10
ALLOWED_MODEL_KEYS = TEXT_EXTENDED_MODELS
LEGACY_MODEL_KEYS = LEGACY_CLI_MODEL_KEYS
CSV_FIELDNAMES = ['o:id', 'Title', 'bibo:content', 'Subject AI', 'Spatial AI']

class NERResult(BaseModel):
    """Pydantic model for NER structured output.

    This schema is used for native structured output support in both OpenAI and Gemini APIs,
    guaranteeing valid JSON responses that match this exact structure.
    """
    persons: List[str] = Field(default_factory=list, description="List of person names extracted from text")
    organizations: List[str] = Field(default_factory=list, description="List of organization names extracted from text")
    locations: List[str] = Field(default_factory=list, description="List of location/place names extracted from text")
    subjects: List[str] = Field(default_factory=list, description="List of subject/topic keywords extracted from text")

@dataclass
class Config:
    """Configuration for NER processing."""
    model_option: ModelOption
    llm_config: LLMConfig
    batch_size: int = BATCH_SIZE

LOWER_PREFIXES = {"el", "van", "de", "von", "der", "den", "hadj", "ben", "ibn", "et", "du", "des", "le", "la", "les", "l", "d"}

# ---------------------------------------------------------------------------
# Config & Prompt
# ---------------------------------------------------------------------------

def load_config(model_option: ModelOption, batch_size: int = BATCH_SIZE) -> Config:
    missing = []
    if model_option.provider == PROVIDER_OPENAI and not os.getenv('OPENAI_API_KEY'):
        missing.append('OPENAI_API_KEY')
    if model_option.provider == PROVIDER_GEMINI:
        if not (os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')):
            missing.append('GEMINI_API_KEY or GOOGLE_APPLICATION_CREDENTIALS')
    if model_option.provider == PROVIDER_MISTRAL and not os.getenv('MISTRAL_API_KEY'):
        missing.append('MISTRAL_API_KEY')
    if model_option.provider == PROVIDER_OPENROUTER and not os.getenv('OPENROUTER_API_KEY'):
        missing.append('OPENROUTER_API_KEY')
    # A self-hosted endpoint needs its address, not a key: the key is optional
    # and defaults to vLLM's "EMPTY" when the server runs without --api-key.
    if model_option.provider == PROVIDER_SELFHOSTED and not os.getenv('SELFHOSTED_LLM_BASE_URL'):
        missing.append('SELFHOSTED_LLM_BASE_URL')
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    # NER-specific LLM configuration for accurate metadata extraction
    llm_config = LLMConfig(
        reasoning_effort="medium",      # OpenAI/OpenRouter: balanced where supported
        text_verbosity="medium",        # OpenAI: detailed entity context
        thinking_level="minimal",       # Gemini 3: minimal thinking sufficient for NER extraction
        # No temperature: each model's vendor-recommended value comes from
        # MODEL_REGISTRY. Consistency comes from the schema and the prompt.
    )

    return Config(
        batch_size=batch_size,
        model_option=model_option,
        llm_config=llm_config,
    )

_NER_PROMPT_CACHE: Optional[str] = None

def get_ner_system_prompt() -> str:
    """Load the NER system prompt lazily (cached after the first read).

    Loading at import time crashed the script before --help could run
    whenever the prompt file was missing.
    """
    global _NER_PROMPT_CACHE
    if _NER_PROMPT_CACHE is None:
        path = os.path.join(SCRIPT_DIR, 'ner_system_prompt.md')
        with open(path, 'r', encoding='utf-8') as f:
            _NER_PROMPT_CACHE = f.read()
    return _NER_PROMPT_CACHE

def get_items_from_multiple_sets(client: OmekaClient, item_set_ids: List[str]) -> List[Dict[str, Any]]:
    """Get items from multiple item sets and combine them into a single list."""
    all_items: List[Dict[str, Any]] = []
    for item_set_id in item_set_ids:
        with console.status(f"[cyan]Fetching items from set {item_set_id}...", spinner="dots"):
            items = client.get_items(int(item_set_id))
        console.print(f"  [green]✓[/] Set {item_set_id}: [bold]{len(items)}[/] items")
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
# Provider-agnostic NER execution
# ---------------------------------------------------------------------------

@retry_with_backoff(max_retries=3, base_delay=2.0)
def perform_ner(llm_client: BaseLLMClient, text_content: str) -> NERResult:
    """Perform Named Entity Recognition using structured output.

    Uses native structured output support from both OpenAI and Gemini APIs
    to guarantee valid JSON responses matching the NERResult schema.
    """
    if not text_content.strip():
        return NERResult(persons=[], organizations=[], locations=[], subjects=[])

    user_prompt = f"TEXT TO ANALYZE:\n{text_content}\n"

    # Use structured output - guaranteed to return valid NERResult.
    # Failures propagate (after retry_with_backoff retries) so callers count
    # the item as failed — returning an empty result here would be
    # indistinguishable from "no entities found" and the item could never be
    # retried.
    result = llm_client.generate_structured(
        get_ner_system_prompt(),
        user_prompt,
        NERResult
    )
    # Apply deduplication and cleaning to the structured result
    return NERResult(
        persons=deduplicate_entities(result.persons),
        organizations=deduplicate_entities(result.organizations),
        locations=deduplicate_entities(result.locations),
        subjects=deduplicate_entities(result.subjects),
    )

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
        self.lock = asyncio.Lock()
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
        async with self.lock:
            self._update(**flags)

# ---------------------------------------------------------------------------
# Spatial Coverage
# ---------------------------------------------------------------------------

def get_item_set_spatial_coverage(client: OmekaClient, item_set_id: str) -> str:
    data = client.get_item_set(int(item_set_id))
    if not data:
        return ''
    values = data.get('dcterms:spatial', [])
    for v in values:
        val = v.get('@value') if isinstance(v, dict) else None
        if val:
            return val.strip()
    return ''

def get_combined_spatial_coverage(client: OmekaClient, item_set_ids: List[str]) -> Optional[str]:
    """Get spatial coverage from multiple item sets. Returns the first non-empty one found."""
    for item_set_id in item_set_ids:
        coverage = get_item_set_spatial_coverage(client, item_set_id)
        if coverage:
            logger.info(f"Using spatial coverage from set {item_set_id}: {coverage}")
            return coverage
    return None

# ---------------------------------------------------------------------------
# Processing Functions
# ---------------------------------------------------------------------------

def process_single_item(item: Dict[str, Any], writer: csv.DictWriter,
                        spatial_filter: Optional[str],
                        ner_fn: Callable[[str], NERResult]) -> str:
    """Process one item: build the CSV row, run NER, and write the row.

    Shared by the sync and async drivers. Returns 'success', 'failed', or
    'empty'; stats updates are left to the caller (the async driver must
    take a lock via ``stats.update_async``).
    """
    try:
        row = {
            'o:id': get_value(item, 'o:id'),
            'Title': get_value(item, 'dcterms:title'),
            'bibo:content': get_value(item, 'bibo:content')
        }
        content = row['bibo:content'].strip()
        if not content:
            writer.writerow({**row, 'Subject AI': '', 'Spatial AI': ''})
            return 'empty'
        entities = ner_fn(content)
        subjects_all = entities.persons + entities.organizations + entities.subjects
        locations = [loc for loc in entities.locations if not spatial_filter or loc.lower() != spatial_filter.lower()]
        row['Subject AI'] = clean_apostrophes('|'.join(subjects_all))
        row['Spatial AI'] = clean_apostrophes('|'.join(locations))
        writer.writerow(row)
        return 'success'
    except Exception as e:
        logger.error(f"Error processing item {item.get('o:id')}: {e}")
        return 'failed'

def _progress_description(stats: ProcessingStats) -> str:
    return (f"[cyan]NER extraction[/] [green]✓{stats.successful_items}[/] "
            f"[red]✗{stats.failed_items}[/] [dim]○{stats.empty_content_items}[/]")

class _DurableWriter:
    """Serialize rows and flush each one so interruption loses no completed item."""

    def __init__(self, writer: csv.DictWriter, handle: TextIO):
        self._writer = writer
        self._handle = handle
        self._lock = threading.Lock()

    def writerow(self, row: Dict[str, Any]) -> None:
        with self._lock:
            self._writer.writerow(row)
            self._handle.flush()


@contextmanager
def durable_csv_writer(output_csv: str, *, resume: bool) -> Iterator[_DurableWriter]:
    path = Path(output_csv)
    mode = "a" if resume and path.exists() else "w"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        if mode == "w" or path.stat().st_size == 0:
            writer.writeheader()
            handle.flush()
        yield _DurableWriter(writer, handle)

async def process_item_async(item: Dict[str, Any], writer, stats: ProcessingStats,
                             spatial_filter: Optional[str], ner_fn: Callable[[str], NERResult],
                             progress: Progress, task_id) -> None:
    loop = asyncio.get_running_loop()
    outcome = await loop.run_in_executor(
        None, partial(process_single_item, item, writer, spatial_filter, ner_fn)
    )
    await stats.update_async(**{outcome: True})
    progress.update(task_id, advance=1, description=_progress_description(stats))

def process_items_batch(items: List[Dict[str, Any]], writer: csv.DictWriter, stats: ProcessingStats,
                        spatial_filter: Optional[str], ner_fn: Callable[[str], NERResult],
                        progress: Progress, task_id) -> None:
    for item in items:
        outcome = process_single_item(item, writer, spatial_filter, ner_fn)
        stats.update(**{outcome: True})
        progress.update(task_id, advance=1, description=_progress_description(stats))

async def process_items_async(items: List[Dict[str, Any]], output_csv: str, stats: ProcessingStats,
                              spatial_filter: Optional[str], batch_size: int,
                              ner_fn: Callable[[str], NERResult], progress: Progress,
                              task_id, *, resume: bool = False) -> None:
    semaphore = asyncio.Semaphore(batch_size)
    async def worker(item: Dict[str, Any], writer):
        async with semaphore:
            await process_item_async(item, writer, stats, spatial_filter, ner_fn, progress, task_id)
    with durable_csv_writer(output_csv, resume=resume) as writer:
        tasks = [worker(item, writer) for item in items]
        for chunk_start in range(0, len(tasks), batch_size):
            chunk = tasks[chunk_start:chunk_start + batch_size]
            await asyncio.gather(*chunk)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(stats: ProcessingStats, output_csv: str, usage: Optional[UsageTotals] = None) -> None:
    elapsed = (datetime.now() - stats.start_time).total_seconds()
    speed = stats.processed_items / elapsed if elapsed else 0
    success_rate = (stats.successful_items / stats.total_items * 100) if stats.total_items else 0

    # Create summary table
    console.print()
    summary_table = Table(title="🏁 Processing Complete", box=box.ROUNDED, title_style="bold green")
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", justify="right")

    summary_table.add_row("[green]✓ Successful[/]", f"{stats.successful_items} ({success_rate:.1f}%)")
    if stats.failed_items > 0:
        summary_table.add_row("[red]✗ Failed[/]", str(stats.failed_items))
    if stats.empty_content_items > 0:
        summary_table.add_row("[dim]○ Empty content[/]", str(stats.empty_content_items))
    summary_table.add_row("[cyan]Total[/]", str(stats.total_items))
    summary_table.add_row("", "")
    summary_table.add_row("⏱️ Duration", f"{elapsed:.1f}s")
    summary_table.add_row("⚡ Speed", f"{speed:.2f} items/s")
    summary_table.add_row("📁 Output", output_csv)
    if usage is not None and usage.requests:
        summary_table.add_row("💸 Model usage", usage.summary())

    console.print(summary_table)

    logger.info(f"Processing complete: {stats.successful_items}/{stats.total_items} successful")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="NER extraction for Omeka S metadata (OpenAI, Gemini, or Mistral)")
    parser.add_argument("--item-set-id", type=str, help="Item set ID(s) to process (comma-separated)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default {BATCH_SIZE})")
    parser.add_argument("--async", action="store_true", help="Use async processing")
    parser.add_argument("--output-dir", type=str, help="Directory for output CSV")
    parser.add_argument(
        "--force", action="store_true",
        help="Replace an output whose model, prompt, or source scope differs",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=ALLOWED_MODEL_KEYS + LEGACY_MODEL_KEYS,
        default=DEFAULT_TEXT_MODEL_KEY,
        help=f"Text model (default: {DEFAULT_TEXT_MODEL_KEY})",
    )
    return parser.parse_args()

def _collect_item_sets(args) -> List[str]:
    item_set_input = args.item_set_id or input("Enter item set ID(s) (comma-separated): ")
    item_set_ids = parse_item_set_ids(item_set_input)
    if not item_set_ids:
        raise ValueError("No valid item set IDs provided")
    return item_set_ids

def _build_output_path(item_set_ids: List[str], output_dir: str, model_key: str) -> str:
    slug = model_key.replace('-', '_')
    if len(item_set_ids) == 1:
        return os.path.join(output_dir, f"item_set_{item_set_ids[0]}_processed_{slug}.csv")
    sets_str = '_'.join(item_set_ids)
    return os.path.join(output_dir, f"item_sets_{sets_str}_processed_{slug}.csv")

def _build_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False
    )

@dataclass
class RunSetup:
    """Everything the sync/async drivers need after shared startup."""
    config: Config
    llm_client: BaseLLMClient
    items: List[Dict[str, Any]]
    spatial_filter: Optional[str]
    output_csv: str
    resume: bool = False
    resumed_items: int = 0


def _deduplicate_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Preserve API order while removing records repeated across item sets."""
    unique: Dict[str, Dict[str, Any]] = {}
    for item in items:
        item_id = str(item.get("o:id", "")).strip()
        if item_id and item_id not in unique:
            unique[item_id] = item
    return list(unique.values())


def _prepare_checkpointed_output(
    output_csv: str,
    *,
    context: Dict[str, Any],
    items: List[Dict[str, Any]],
    force: bool,
) -> tuple[List[Dict[str, Any]], bool, int]:
    output_path = Path(output_csv)
    checkpoint_path = output_path.with_suffix(output_path.suffix + ".checkpoint.json")
    if output_path.exists() and not checkpoint_path.exists() and not force:
        raise CheckpointMismatch(
            f"Existing NER CSV has no provenance checkpoint: {output_path}. "
            "Use --force to replace it."
        )
    JsonCheckpoint.open(checkpoint_path, context, reset=force)
    processed_ids = set() if force else load_csv_ids(output_path, "o:id")
    pending = [
        item for item in items
        if str(item.get("o:id", "")).strip() not in processed_ids
    ]
    resume = output_path.exists() and not force
    return pending, resume, len(processed_ids)

def prepare_run(args, mode_label: str) -> Optional[RunSetup]:
    """Shared startup for both drivers: banner, model selection, config
    table, client setup, item collection, spatial filter, and output path.

    Returns None when there is nothing to process.
    """
    intro_text = (
        "[bold cyan]Named Entity Recognition Pipeline[/]\n\n"
        "[dim]Extract persons, organizations, locations, and subjects from Omeka S items[/]"
    )
    console.print(Panel(intro_text, title="🔍 NER Extraction", border_style="cyan", padding=(1, 2)))

    model_option = get_model_option(args.model, allowed_keys=ALLOWED_MODEL_KEYS)
    config = load_config(model_option=model_option, batch_size=args.batch_size)

    # Initialize shared Omeka client
    client = OmekaClient.from_env()

    # Display configuration table
    config_table = Table(title="🤖 Configuration", box=box.ROUNDED, show_header=True, header_style="bold cyan")
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Model", summary_from_option(model_option))
    config_table.add_row("Mode", mode_label)
    config_table.add_row("Batch Size", str(config.batch_size))
    if model_option.provider == PROVIDER_OPENAI:
        config_table.add_row("Reasoning Effort", config.llm_config.reasoning_effort or "default")
    elif model_option.provider == PROVIDER_GEMINI:
        config_table.add_row("Thinking Level", config.llm_config.thinking_level or "default")
    elif model_option.provider in (PROVIDER_OPENROUTER, PROVIDER_SELFHOSTED):
        # Show the effective value after the provider adapter clamps the
        # pipeline-wide request to this model's declared levels.
        requested = config.llm_config.reasoning_effort
        effective = (
            requested
            if requested in model_option.supported_reasoning_efforts
            else model_option.default_reasoning_effort
        )
        config_table.add_row("Reasoning Effort", effective or "off")
    console.print(config_table)
    console.print()

    logger.info(f"Using AI model: {summary_from_option(model_option)}")
    llm_client = build_llm_client(model_option, config=config.llm_config)
    item_set_ids = _collect_item_sets(args)

    console.print(f"[cyan]📂 Item sets:[/] {', '.join(item_set_ids)}")

    spatial_filter = get_combined_spatial_coverage(client, item_set_ids)
    if spatial_filter:
        console.print(f"[cyan]🌍 Spatial filter:[/] {spatial_filter}")

    items = _deduplicate_items(get_items_from_multiple_sets(client, item_set_ids))
    if not items:
        console.print("[yellow]⚠[/] No items found.")
        return None

    console.print(f"\n[bold]Total items to process:[/] {len(items)}\n")

    output_dir = args.output_dir or os.path.join(SCRIPT_DIR, 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_csv = _build_output_path(item_set_ids, output_dir, model_option.key)
    items, resume, resumed_items = _prepare_checkpointed_output(
        output_csv,
        context={
            "pipeline": "ner-csv-v2",
            "model_key": model_option.key,
            "model_id": model_option.model,
            "prompt_sha256": sha256_text(get_ner_system_prompt()),
            "item_set_ids": item_set_ids,
            "spatial_filter": spatial_filter,
            "fieldnames": CSV_FIELDNAMES,
        },
        items=items,
        force=args.force,
    )
    if resumed_items:
        console.print(
            f"[green]✓[/] Resuming: [bold]{resumed_items}[/] completed item(s), "
            f"[bold]{len(items)}[/] remaining"
        )

    return RunSetup(
        config=config,
        llm_client=llm_client,
        items=items,
        spatial_filter=spatial_filter,
        output_csv=output_csv,
        resume=resume,
        resumed_items=resumed_items,
    )

async def async_main(args) -> None:
    setup = prepare_run(args, mode_label="[yellow]Async[/]")
    if setup is None:
        return
    if not setup.items:
        console.print("[green]✓[/] Output is already complete for this model and prompt.")
        return

    stats = ProcessingStats(total_items=len(setup.items))
    ner_fn = partial(perform_ner, setup.llm_client)

    with _build_progress() as progress:
        task_id = progress.add_task("[cyan]NER extraction[/]", total=stats.total_items)
        await process_items_async(setup.items, setup.output_csv, stats, setup.spatial_filter,
                                  setup.config.batch_size, ner_fn, progress, task_id,
                                  resume=setup.resume)

    summarize(stats, setup.output_csv, setup.llm_client.usage)

def main() -> None:
    args = parse_arguments()
    try:
        if getattr(args, 'async', False):
            asyncio.run(async_main(args))
            return

        setup = prepare_run(args, mode_label="Sync")
        if setup is None:
            return
        if not setup.items:
            console.print("[green]✓[/] Output is already complete for this model and prompt.")
            return

        stats = ProcessingStats(total_items=len(setup.items))
        ner_fn = partial(perform_ner, setup.llm_client)

        with durable_csv_writer(setup.output_csv, resume=setup.resume) as writer:
            with _build_progress() as progress:
                task_id = progress.add_task("[cyan]NER extraction[/]", total=stats.total_items)
                process_items_batch(
                    setup.items, writer, stats, setup.spatial_filter, ner_fn, progress, task_id
                )

        summarize(stats, setup.output_csv, setup.llm_client.usage)
    except (CheckpointMismatch, ValueError) as exc:
        console.print(f"[red]✗[/] {exc}")


if __name__ == '__main__':
    main()
