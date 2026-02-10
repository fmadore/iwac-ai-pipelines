"""
Named Entity Recognition (NER) script for Omeka S metadata extraction.

Supported models:
  - gpt-5-mini: Fast, cost-effective OpenAI model
  - gemini-flash: Fast, cost-effective Gemini model
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
  Gemini:
    GEMINI_API_KEY
  Mistral:
    MISTRAL_API_KEY

Output CSV columns: o:id, Title, bibo:content, Subject AI, Spatial AI

Usage examples:
    python 01_NER_AI.py --item-set-id 123
    python 01_NER_AI.py --item-set-id 123 --model gpt-5-mini
    python 01_NER_AI.py --item-set-id 123 --model gemini-flash --async
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
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from functools import partial

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field

# Rich console for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, MofNCompleteColumn
from rich import box

console = Console()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from common.omeka_client import OmekaClient  # noqa: E402
from common.llm_provider import (  # noqa: E402
    BaseLLMClient,
    ModelOption,
    LLMConfig,
    build_llm_client,
    get_model_option,
    summary_from_option,
    PROVIDER_GEMINI,
    PROVIDER_OPENAI,
    PROVIDER_MISTRAL,
)

# ---------------------------------------------------------------------------
# Logging & Environment
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# ---------------------------------------------------------------------------
# Constants & Types
# ---------------------------------------------------------------------------
BATCH_SIZE = 10

class NERResult(BaseModel):
    """Pydantic model for NER structured output.
    
    This schema is used for native structured output support in both OpenAI and Gemini APIs,
    guaranteeing valid JSON responses that match this exact structure.
    """
    persons: List[str] = Field(default_factory=list, description="List of person names extracted from text")
    organizations: List[str] = Field(default_factory=list, description="List of organization names extracted from text")
    locations: List[str] = Field(default_factory=list, description="List of location/place names extracted from text")
    subjects: List[str] = Field(default_factory=list, description="List of subject/topic keywords extracted from text")

class Config(BaseModel):
    """Configuration for NER processing."""
    model_config = {"arbitrary_types_allowed": True}

    batch_size: int = BATCH_SIZE
    model_option: Any = None  # ModelOption type
    llm_config: Any = None    # LLMConfig type

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
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    # NER-specific LLM configuration for accurate metadata extraction
    llm_config = LLMConfig(
        reasoning_effort="medium",      # OpenAI: balanced reasoning (cost-effective)
        text_verbosity="medium",        # OpenAI: detailed entity context
        thinking_budget=500,            # Gemini: moderate thinking (cost-effective)
        temperature=0.2                 # Gemini: consistent, low-variance results
    )

    return Config(
        batch_size=batch_size,
        model_option=model_option,
        llm_config=llm_config,
    )

def load_ner_prompt() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, 'ner_system_prompt.md')
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

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

NER_SYSTEM_PROMPT = load_ner_prompt()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def perform_ner(llm_client: BaseLLMClient, text_content: str) -> NERResult:
    """Perform Named Entity Recognition using structured output.
    
    Uses native structured output support from both OpenAI and Gemini APIs
    to guarantee valid JSON responses matching the NERResult schema.
    """
    if not text_content.strip():
        return NERResult(persons=[], organizations=[], locations=[], subjects=[])
    
    user_prompt = f"TEXT TO ANALYZE:\n{text_content}\n"
    
    try:
        # Use structured output - guaranteed to return valid NERResult
        result = llm_client.generate_structured(
            NER_SYSTEM_PROMPT,
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
    except Exception as e:
        logger.warning(f"NER structured output failed: {e}")
        return NERResult(persons=[], organizations=[], locations=[], subjects=[])

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
async def process_item_async(item: Dict[str, Any], writer: csv.DictWriter, stats: ProcessingStats,
                             spatial_filter: Optional[str], ner_fn: Callable[[str], NERResult],
                             progress: Progress, task_id) -> None:
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
        entities: NERResult = await loop.run_in_executor(None, partial(ner_fn, content))
        subjects_all = entities.persons + entities.organizations + entities.subjects
        locations = [l for l in entities.locations if not spatial_filter or l.lower() != spatial_filter.lower()]
        row['Subject AI'] = clean_apostrophes('|'.join(subjects_all))
        row['Spatial AI'] = clean_apostrophes('|'.join(locations))
        writer.writerow(row)
        await stats.update_async(success=True)
    except Exception as e:
        logger.error(f"Error processing item {item.get('o:id')}: {e}")
        await stats.update_async(failed=True)
    finally:
        progress.update(task_id, advance=1,
                       description=f"[cyan]NER extraction[/] [green]✓{stats.successful_items}[/] [red]✗{stats.failed_items}[/] [dim]○{stats.empty_content_items}[/]")

def process_items_batch(items: List[Dict[str, Any]], writer: csv.DictWriter, stats: ProcessingStats,
                        spatial_filter: Optional[str], ner_fn: Callable[[str], NERResult],
                        progress: Progress, task_id) -> None:
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
                subjects_all = entities.persons + entities.organizations + entities.subjects
                locations = [l for l in entities.locations if not spatial_filter or l.lower() != spatial_filter.lower()]
                row['Subject AI'] = clean_apostrophes('|'.join(subjects_all))
                row['Spatial AI'] = clean_apostrophes('|'.join(locations))
                writer.writerow(row)
                stats.update(success=True)
        except Exception as e:
            logger.error(f"Error processing item {item.get('o:id')}: {e}")
            stats.update(failed=True)
        progress.update(task_id, advance=1,
                       description=f"[cyan]NER extraction[/] [green]✓{stats.successful_items}[/] [red]✗{stats.failed_items}[/] [dim]○{stats.empty_content_items}[/]")

async def process_items_async(items: List[Dict[str, Any]], output_csv: str, stats: ProcessingStats,
                              spatial_filter: Optional[str], batch_size: int,
                              ner_fn: Callable[[str], NERResult], progress: Progress, task_id) -> None:
    fieldnames = ['o:id', 'Title', 'bibo:content', 'Subject AI', 'Spatial AI']
    semaphore = asyncio.Semaphore(batch_size)
    async def worker(item: Dict[str, Any], writer: csv.DictWriter):
        async with semaphore:
            await process_item_async(item, writer, stats, spatial_filter, ner_fn, progress, task_id)
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
        "--model",
        type=str,
        choices=["gpt-5-mini", "gemini-flash", "mistral-large", "ministral-14b"],
        help="Model: 'gpt-5-mini', 'gemini-flash', 'mistral-large', or 'ministral-14b'. Defaults to interactive prompt."
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

async def async_main(args) -> None:
    # Display welcome banner
    intro_text = (
        "[bold cyan]Named Entity Recognition Pipeline[/]\n\n"
        "[dim]Extract persons, organizations, locations, and subjects from Omeka S items[/]"
    )
    console.print(Panel(intro_text, title="🔍 NER Extraction", border_style="cyan", padding=(1, 2)))
    
    model_option = get_model_option(args.model, allowed_keys=["gpt-5-mini", "gemini-flash", "mistral-large", "ministral-14b"])
    config = load_config(model_option=model_option, batch_size=args.batch_size)

    # Initialize shared Omeka client
    client = OmekaClient.from_env()

    # Display configuration table
    config_table = Table(title="🤖 Configuration", box=box.ROUNDED, show_header=True, header_style="bold cyan")
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Model", summary_from_option(model_option))
    config_table.add_row("Mode", "[yellow]Async[/]")
    config_table.add_row("Batch Size", str(config.batch_size))
    config_table.add_row("Reasoning Effort", config.llm_config.reasoning_effort or "default")
    config_table.add_row("Thinking Budget", str(config.llm_config.thinking_budget) if config.llm_config.thinking_budget else "default")
    console.print(config_table)
    console.print()

    logger.info(f"Using AI model: {summary_from_option(model_option)}")
    llm_client = build_llm_client(model_option, config=config.llm_config)
    item_set_ids = _collect_item_sets(args)

    console.print(f"[cyan]📂 Item sets:[/] {', '.join(item_set_ids)}")

    spatial_filter = get_combined_spatial_coverage(client, item_set_ids)
    if spatial_filter:
        console.print(f"[cyan]🌍 Spatial filter:[/] {spatial_filter}")

    items = get_items_from_multiple_sets(client, item_set_ids)
    if not items:
        console.print("[yellow]⚠[/] No items found.")
        return
    
    console.print(f"\n[bold]Total items to process:[/] {len(items)}\n")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(script_dir, 'output')
    output_dir = args.output_dir or default_output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_csv = _build_output_path(item_set_ids, output_dir, model_option.key)
    stats = ProcessingStats(total_items=len(items))
    ner_fn = partial(perform_ner, llm_client)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False
    ) as progress:
        task_id = progress.add_task("[cyan]NER extraction[/]", total=stats.total_items)
        await process_items_async(items, output_csv, stats, spatial_filter, config.batch_size,
                                  ner_fn, progress, task_id)
    
    summarize(stats, output_csv)

def main() -> None:
    args = parse_arguments()
    if getattr(args, 'async', False):
        asyncio.run(async_main(args))
        return
    
    # Display welcome banner
    intro_text = (
        "[bold cyan]Named Entity Recognition Pipeline[/]\n\n"
        "[dim]Extract persons, organizations, locations, and subjects from Omeka S items[/]"
    )
    console.print(Panel(intro_text, title="🔍 NER Extraction", border_style="cyan", padding=(1, 2)))
    
    model_option = get_model_option(args.model, allowed_keys=["gpt-5-mini", "gemini-flash", "mistral-large", "ministral-14b"])
    config = load_config(model_option=model_option, batch_size=args.batch_size)

    # Initialize shared Omeka client
    client = OmekaClient.from_env()

    # Display configuration table
    config_table = Table(title="🤖 Configuration", box=box.ROUNDED, show_header=True, header_style="bold cyan")
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Model", summary_from_option(model_option))
    config_table.add_row("Mode", "Sync")
    config_table.add_row("Batch Size", str(config.batch_size))
    config_table.add_row("Reasoning Effort", config.llm_config.reasoning_effort or "default")
    config_table.add_row("Thinking Budget", str(config.llm_config.thinking_budget) if config.llm_config.thinking_budget else "default")
    console.print(config_table)
    console.print()

    logger.info(f"Using AI model: {summary_from_option(model_option)}")
    llm_client = build_llm_client(model_option, config=config.llm_config)
    item_set_ids = _collect_item_sets(args)

    console.print(f"[cyan]📂 Item sets:[/] {', '.join(item_set_ids)}")

    spatial_filter = get_combined_spatial_coverage(client, item_set_ids)
    if spatial_filter:
        console.print(f"[cyan]🌍 Spatial filter:[/] {spatial_filter}")

    items = get_items_from_multiple_sets(client, item_set_ids)
    if not items:
        console.print("[yellow]⚠[/] No items found.")
        return
    
    console.print(f"\n[bold]Total items to process:[/] {len(items)}\n")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(script_dir, 'output')
    output_dir = args.output_dir or default_output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_csv = _build_output_path(item_set_ids, output_dir, model_option.key)
    stats = ProcessingStats(total_items=len(items))
    fieldnames = ['o:id', 'Title', 'bibo:content', 'Subject AI', 'Spatial AI']
    ner_fn = partial(perform_ner, llm_client)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False
        ) as progress:
            task_id = progress.add_task("[cyan]NER extraction[/]", total=stats.total_items)
            process_items_batch(items, writer, stats, spatial_filter, ner_fn, progress, task_id)
    
    summarize(stats, output_csv)

if __name__ == '__main__':
    main()
