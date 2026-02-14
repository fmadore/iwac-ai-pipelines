#!/usr/bin/env python3
"""
01_sentiment_analysis.py
========================

AI Sentiment Analysis Pipeline for IWAC Omeka S Items.

This script:
1. Fetches items from Omeka S API based on item set ID(s).
2. Extracts bibo:content text from each item.
3. Runs sentiment analysis using all 3 AI models concurrently (Gemini, ChatGPT, Mistral).
4. Caches results to JSON files.
5. Updates Omeka S items with sentiment analysis results via PATCH.

The analysis evaluates:
- Centralité (Centrality): How central Islam/Muslims are to the article
- Subjectivité (Subjectivity): Score from 1-5 measuring objectivity
- Polarité (Polarity): Sentiment towards Islam/Muslims

Usage
-----
    python AI_sentiment_analysis/01_sentiment_analysis.py --item-set-id 123
    python AI_sentiment_analysis/01_sentiment_analysis.py --item-set-id 123,456 --skip-update

Environment Variables
---------------------
OMEKA_BASE_URL        Base URL for Omeka S API
OMEKA_KEY_IDENTITY    API key identity
OMEKA_KEY_CREDENTIAL  API key credential
GEMINI_API_KEY        Google Gemini API key
OPENAI_API_KEY        OpenAI API key
MISTRAL_API_KEY       Mistral API key
"""
import os
import sys
import json
import time
import argparse
import logging
import concurrent.futures
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal

from pydantic import BaseModel, Field

# Rich for beautiful console output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.prompt import Prompt
from rich import box

# Shared Omeka client and LLM provider
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.omeka_client import OmekaClient
from common.llm_provider import build_llm_client, get_model_option, LLMConfig, BaseLLMClient

# Global Rich console
console = Console()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model registry keys for the 3 concurrent models
MODEL_KEYS = {
    "gemini": "gemini-flash",
    "chatgpt": "gpt-5-mini",
    "mistral": "ministral-14b",
}

# Cache file names
CACHE_DIR_NAME = "cache"
CACHE_FILE_NAME = "sentiment_cache.json"

# Omeka S Property Mappings (IWAC ontology)
# These are the property keys used in Omeka S API requests
PROPERTY_MAPPINGS = {
    "gemini": {
        "centralite": "iwac:geminiCentralite",
        "centralite_justification": "iwac:geminiCentraliteJustification",
        "polarite": "iwac:geminiPolarite",
        "polarite_justification": "iwac:geminiPolariteJustification",
        "subjectivite_score": "iwac:geminiSubjectiviteScore",
        "subjectivite_justification": "iwac:geminiSubjectiviteJustification",
    },
    "chatgpt": {
        "centralite": "iwac:chatgptCentralite",
        "centralite_justification": "iwac:chatgptCentraliteJustification",
        "polarite": "iwac:chatgptPolarite",
        "polarite_justification": "iwac:chatgptPolariteJustification",
        "subjectivite_score": "iwac:chatgptSubjectiviteScore",
        "subjectivite_justification": "iwac:chatgptSubjectiviteJustification",
    },
    "mistral": {
        "centralite": "iwac:mistralCentralite",
        "centralite_justification": "iwac:mistralCentraliteJustification",
        "polarite": "iwac:mistralPolarite",
        "polarite_justification": "iwac:mistralPolariteJustification",
        "subjectivite_score": "iwac:mistralSubjectiviteScore",
        "subjectivite_justification": "iwac:mistralSubjectiviteJustification",
    },
}

# Controlled Vocabulary Item IDs for linking
CENTRALITE_ITEM_IDS = {
    "Très central": 78048,
    "Central": 78049,
    "Secondaire": 78050,
    "Marginal": 78051,
    "Non abordé": 78052,
}

POLARITE_ITEM_IDS = {
    "Très positif": 78031,
    "Positif": 78038,
    "Neutre": 78039,
    "Négatif": 78040,
    "Très négatif": 78041,
    "Non applicable": 78042,
}

SUBJECTIVITE_ITEM_IDS = {
    1: 78043,  # Très objectif
    2: 78044,  # Plutôt objectif
    3: 78045,  # Mixte
    4: 78046,  # Plutôt subjectif
    5: 78047,  # Très subjectif
}

# Property IDs for IWAC Omeka S instance
# These are stable within the instance and used for API updates
PROPERTY_IDS: Dict[str, int] = {
    # Gemini properties (319-324)
    "iwac:geminiCentralite": 319,
    "iwac:geminiCentraliteJustification": 320,
    "iwac:geminiPolarite": 321,
    "iwac:geminiPolariteJustification": 322,
    "iwac:geminiSubjectiviteScore": 323,
    "iwac:geminiSubjectiviteJustification": 324,
    # ChatGPT properties (325-330)
    "iwac:chatgptCentralite": 325,
    "iwac:chatgptCentraliteJustification": 326,
    "iwac:chatgptPolarite": 327,
    "iwac:chatgptPolariteJustification": 328,
    "iwac:chatgptSubjectiviteScore": 329,
    "iwac:chatgptSubjectiviteJustification": 330,
    # Mistral properties (331-336)
    "iwac:mistralCentralite": 331,
    "iwac:mistralCentraliteJustification": 332,
    "iwac:mistralPolarite": 333,
    "iwac:mistralPolariteJustification": 334,
    "iwac:mistralSubjectiviteScore": 335,
    "iwac:mistralSubjectiviteJustification": 336,
}


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def configure_logging() -> logging.Logger:
    """Configure logging with Rich for elegant display."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
    )
    return logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class SentimentAnalysisOutput(BaseModel):
    """Schema for sentiment analysis output - used for structured outputs with AI APIs."""
    centralite_islam_musulmans: Literal[
        "Très central", "Central", "Secondaire", "Marginal", "Non abordé"
    ] = Field(description="Importance accordée aux thèmes liés à l'islam et aux musulmans dans l'article")
    centralite_justification: str = Field(description="Courte justification en 1 phrase sur la centralité de l'islam/des musulmans")
    subjectivite_score: Optional[int] = Field(
        default=None,
        description="Score de subjectivité de 1 à 5 (entier), ou null si le sujet n'est pas abordé",
    )
    subjectivite_justification: str = Field(description="Justification en 1-2 phrases pour le score de subjectivité")
    polarite: Literal[
        "Très positif", "Positif", "Neutre", "Négatif", "Très négatif", "Non applicable"
    ] = Field(description="Sentiment général exprimé dans l'article envers l'islam et/ou les musulmans")
    polarite_justification: str = Field(description="Justification en 1-2 phrases pour la polarité")


# ============================================================================
# PROMPT LOADING
# ============================================================================

def load_system_prompt() -> str:
    """Load the sentiment analysis prompt from the markdown file."""
    try:
        script_dir = Path(__file__).resolve().parent
        prompt_path = script_dir / "sentiment_prompt.md"
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        console.print(f"[red]Error loading prompt file:[/] {e}")
        return ""


def create_user_prompt(article_text: str) -> str:
    """Create the user prompt with the article text to analyze."""
    return f"Texte à analyser:\n---\n{article_text}\n---"


# ============================================================================
# OMEKA S API FUNCTIONS
# ============================================================================


def get_item_content(item: Dict[str, Any]) -> str:
    """Extract bibo:content text from an Omeka item."""
    content_values = item.get('bibo:content', [])
    if not content_values:
        return ""

    # Prefer French language if available
    for val in content_values:
        if isinstance(val, dict) and val.get('@language') == 'fr':
            return val.get('@value', '')

    # Fall back to first value
    if content_values and isinstance(content_values[0], dict):
        return content_values[0].get('@value', '')

    return ""


def has_existing_sentiment(item: Dict[str, Any], available_models: List[str]) -> bool:
    """
    Check if an item already has sentiment analysis values in Omeka.

    Returns True if ALL available models have centralite values set,
    meaning the item has already been fully analyzed.
    """
    for model in available_models:
        props = PROPERTY_MAPPINGS.get(model)
        if not props:
            continue

        # Check if centralite property exists and has a value
        centralite_key = props["centralite"]
        centralite_values = item.get(centralite_key, [])

        if not centralite_values:
            # This model doesn't have results yet
            return False

    # All available models have sentiment values
    return True


def update_item_sentiment(
    client: OmekaClient,
    item_id: int,
    sentiment_data: Dict[str, Dict[str, Any]],
    logger: logging.Logger
) -> bool:
    """
    Update an Omeka S item with sentiment analysis results from all models.

    Args:
        client: OmekaClient instance
        item_id: Omeka item ID
        sentiment_data: Dictionary with model names as keys and sentiment results as values
        logger: Logger instance

    Returns:
        True if update successful, False otherwise
    """
    item_data = client.get_item(item_id)
    if not item_data:
        return False

    modified = False

    for model_name, results in sentiment_data.items():
        if results.get("analysis_error"):
            logger.warning(f"Skipping {model_name} for item {item_id} due to analysis error")
            continue

        props = PROPERTY_MAPPINGS.get(model_name)
        if not props:
            continue

        # Helper to build property value with property_id if available
        def build_property_value(prop_key: str, value_type: str, value: Any, is_resource: bool = False, language: str = None) -> Dict[str, Any]:
            prop_value = {
                "type": value_type,
                "property_label": prop_key.split(":")[-1],
                "is_public": True,
            }
            # Add property_id if we have it
            prop_id = PROPERTY_IDS.get(prop_key)
            if prop_id:
                prop_value["property_id"] = prop_id

            if is_resource:
                prop_value["value_resource_id"] = value
                prop_value["value_resource_name"] = "items"
            else:
                prop_value["@value"] = value
                if language:
                    prop_value["@language"] = language

            return prop_value

        # --- Centralité (resource:item link) ---
        centralite_value = results.get("centralite_islam_musulmans")
        centralite_item_id = CENTRALITE_ITEM_IDS.get(centralite_value)
        if centralite_item_id:
            prop_key = props["centralite"]
            item_data[prop_key] = [build_property_value(prop_key, "resource:item", centralite_item_id, is_resource=True)]
            modified = True

        # --- Centralité Justification (literal) ---
        centralite_just = results.get("centralite_justification", "")
        if centralite_just:
            prop_key = props["centralite_justification"]
            item_data[prop_key] = [build_property_value(prop_key, "literal", centralite_just, language="fr")]
            modified = True

        # --- Polarité (resource:item link) ---
        polarite_value = results.get("polarite")
        polarite_item_id = POLARITE_ITEM_IDS.get(polarite_value)
        if polarite_item_id:
            prop_key = props["polarite"]
            item_data[prop_key] = [build_property_value(prop_key, "resource:item", polarite_item_id, is_resource=True)]
            modified = True

        # --- Polarité Justification (literal) ---
        polarite_just = results.get("polarite_justification", "")
        if polarite_just:
            prop_key = props["polarite_justification"]
            item_data[prop_key] = [build_property_value(prop_key, "literal", polarite_just, language="fr")]
            modified = True

        # --- Subjectivité Score (resource:item link) ---
        subj_score = results.get("subjectivite_score")
        subj_item_id = SUBJECTIVITE_ITEM_IDS.get(subj_score)
        if subj_item_id:
            prop_key = props["subjectivite_score"]
            item_data[prop_key] = [build_property_value(prop_key, "resource:item", subj_item_id, is_resource=True)]
            modified = True

        # --- Subjectivité Justification (literal) ---
        subj_just = results.get("subjectivite_justification", "")
        if subj_just:
            prop_key = props["subjectivite_justification"]
            item_data[prop_key] = [build_property_value(prop_key, "literal", subj_just, language="fr")]
            modified = True

    if not modified:
        return True  # Nothing to update

    return client.update_item(item_id, item_data)


# ============================================================================
# AI ANALYSIS FUNCTIONS
# ============================================================================

def analyze_with_model(
    llm_client: BaseLLMClient,
    text: str,
    system_prompt: str,
    model_label: str,
    logger: logging.Logger,
    max_retries: int = 3
) -> Dict[str, Any]:
    """Analyze text using any LLM provider via the shared provider."""
    default_error = {
        "centralite_islam_musulmans": "ERREUR_ANALYSE",
        "centralite_justification": f"Erreur lors de l'analyse {model_label}.",
        "subjectivite_score": None,
        "subjectivite_justification": f"Erreur lors de l'analyse {model_label}.",
        "polarite": "ERREUR_ANALYSE",
        "polarite_justification": f"Erreur lors de l'analyse {model_label}.",
        "analysis_error": "Unknown error"
    }

    if not text or not text.strip():
        return {
            **default_error,
            "centralite_islam_musulmans": "Non abordé",
            "centralite_justification": "Texte non fourni ou vide.",
            "subjectivite_justification": "Non applicable - texte vide.",
            "polarite": "Non applicable",
            "polarite_justification": "Non applicable - texte vide.",
            "analysis_error": "Empty text"
        }

    user_prompt = create_user_prompt(text)

    for attempt in range(max_retries):
        try:
            result = llm_client.generate_structured(
                system_prompt, user_prompt, SentimentAnalysisOutput
            )
            return {**result.model_dump(), "analysis_error": None}
        except Exception as e:
            logger.debug(f"{model_label} attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return {**default_error, "analysis_error": "Max retries exceeded"}


# Display labels for each model key
MODEL_LABELS = {
    "gemini": "Gemini",
    "chatgpt": "ChatGPT",
    "mistral": "Mistral",
}


def analyze_with_all_models(
    text: str,
    llm_clients: Dict[str, BaseLLMClient],
    system_prompt: str,
    logger: logging.Logger
) -> Dict[str, Dict[str, Any]]:
    """
    Run sentiment analysis with all available models concurrently.

    Returns:
        Dictionary with model names as keys and results as values
    """
    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(llm_clients)) as executor:
        futures = {}

        for model_name, llm_client in llm_clients.items():
            label = MODEL_LABELS.get(model_name, model_name.capitalize())
            futures[model_name] = executor.submit(
                analyze_with_model, llm_client, text, system_prompt, label, logger
            )

        for model_name, future in futures.items():
            try:
                results[model_name] = future.result(timeout=120)
            except concurrent.futures.TimeoutError:
                results[model_name] = {
                    "analysis_error": "Timeout",
                    "centralite_islam_musulmans": "ERREUR_ANALYSE",
                    "centralite_justification": "Timeout lors de l'analyse.",
                    "subjectivite_score": None,
                    "subjectivite_justification": "Timeout lors de l'analyse.",
                    "polarite": "ERREUR_ANALYSE",
                    "polarite_justification": "Timeout lors de l'analyse.",
                }
            except Exception as e:
                results[model_name] = {
                    "analysis_error": str(e),
                    "centralite_islam_musulmans": "ERREUR_ANALYSE",
                    "centralite_justification": f"Erreur: {e}",
                    "subjectivite_score": None,
                    "subjectivite_justification": f"Erreur: {e}",
                    "polarite": "ERREUR_ANALYSE",
                    "polarite_justification": f"Erreur: {e}",
                }

    return results


# ============================================================================
# CACHE FUNCTIONS
# ============================================================================

def load_cache(cache_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Load cache from JSON file."""
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"Loaded cache with {len(data)} entries")
                return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load cache: {e}")
    return {}


def save_cache(cache_path: Path, cache_data: Dict[str, Any], logger: logging.Logger) -> None:
    """Save cache to JSON file."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Cache saved with {len(cache_data)} entries")
    except IOError as e:
        logger.error(f"Failed to save cache: {e}")


def is_cache_valid(cached_entry: Dict[str, Any], available_models: List[str]) -> bool:
    """Check if a cached entry has valid results for available models."""
    for model in available_models:
        if model not in cached_entry:
            return False
        model_data = cached_entry[model]
        if model_data.get("analysis_error"):
            return False
        if model_data.get("centralite_islam_musulmans") == "ERREUR_ANALYSE":
            return False
    return True


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    logger = configure_logging()

    # Display welcome banner
    console.print(Panel.fit(
        "[bold cyan]AI Sentiment Analysis Pipeline[/bold cyan]\n"
        "[dim]Gemini + ChatGPT + Mistral for IWAC Omeka S[/dim]",
        border_style="cyan"
    ))

    script_dir = Path(__file__).resolve().parent

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run AI sentiment analysis on Omeka S items"
    )
    parser.add_argument(
        "--item-set-id",
        type=str,
        required=False,
        help="Item set ID(s) to process (comma-separated for multiple)"
    )
    parser.add_argument(
        "--skip-update",
        action="store_true",
        help="Skip updating Omeka S (only run analysis and cache)"
    )
    parser.add_argument(
        "--force-reanalyze",
        action="store_true",
        help="Force re-analysis even if cached results exist"
    )

    args = parser.parse_args()

    # Get item set ID(s)
    item_set_ids_str = args.item_set_id
    if not item_set_ids_str:
        item_set_ids_str = Prompt.ask("[cyan]Enter item set ID(s)[/cyan] (comma-separated)")

    try:
        item_set_ids = [int(x.strip()) for x in item_set_ids_str.split(",")]
    except ValueError:
        console.print("[red]✗[/] Invalid item set ID format")
        return

    # Initialize Omeka client
    try:
        omeka_client = OmekaClient.from_env()
    except ValueError as e:
        console.print(f"[red]✗[/] {e}")
        return

    # Build LLM clients for each model
    llm_clients: Dict[str, BaseLLMClient] = {}
    config = LLMConfig(temperature=0.2)
    for model_name, model_key in MODEL_KEYS.items():
        try:
            option = get_model_option(model_key)
            llm_clients[model_name] = build_llm_client(option, config=config)
        except (RuntimeError, ValueError):
            logger.debug(f"{model_name} not available (SDK or API key missing)")

    available_models = list(llm_clients.keys())

    if not available_models:
        console.print("[red]✗[/] No AI models available")
        console.print("[dim]Required: At least one of GEMINI_API_KEY, OPENAI_API_KEY, MISTRAL_API_KEY[/]")
        return

    # Display configuration
    config_table = Table(title="Configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Item Set IDs", ", ".join(map(str, item_set_ids)))
    config_table.add_row("Omeka URL", omeka_client.base_url)
    config_table.add_row("Available Models", ", ".join(available_models))
    config_table.add_row("Skip Update", "Yes" if args.skip_update else "No")
    config_table.add_row("Force Re-analyze", "Yes" if args.force_reanalyze else "No")
    console.print(config_table)
    console.print()

    # Load system prompt
    system_prompt = load_system_prompt()
    if not system_prompt:
        console.print("[red]✗[/] Failed to load sentiment prompt")
        return

    # Load cache
    cache_dir = script_dir / CACHE_DIR_NAME
    cache_path = cache_dir / CACHE_FILE_NAME
    cache_data = load_cache(cache_path, logger)

    # Fetch items from Omeka
    console.print("[cyan]Fetching items from Omeka S...[/]")
    items = []
    for item_set_id in item_set_ids:
        logger.info(f"Fetching items from item set {item_set_id}...")
        items.extend(omeka_client.get_items(item_set_id))

    if not items:
        console.print("[yellow]![/] No items found in the specified item set(s)")
        return

    console.print(f"[green]✓[/] Found [bold]{len(items)}[/] items")

    # Filter items with content and check for existing sentiment values
    items_with_content = []
    skipped_existing = 0

    for item in items:
        item_id = item.get("o:id")
        content = get_item_content(item)

        if not content or not content.strip():
            continue

        # Skip items that already have sentiment values (unless force-reanalyze)
        if not args.force_reanalyze and has_existing_sentiment(item, available_models):
            skipped_existing += 1
            continue

        items_with_content.append({
            "id": item_id,
            "content": content,
            "title": item.get("o:title", f"Item {item_id}")
        })

    console.print(f"[green]✓[/] {len(items_with_content)} items need sentiment analysis")
    if skipped_existing > 0:
        console.print(f"[dim]  (Skipped {skipped_existing} items with existing sentiment values)[/]")

    if not items_with_content:
        console.print("[yellow]![/] No items to process (all have existing values or no content)")
        return

    # Process items
    stats = {
        "total": len(items_with_content),
        "analyzed": 0,
        "cached": 0,
        "updated": 0,
        "errors": 0,
        "skipped_existing": skipped_existing
    }

    console.rule("[bold cyan]Processing Items")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            "[cyan]Analyzing sentiment...",
            total=len(items_with_content)
        )

        for item_info in items_with_content:
            item_id = item_info["id"]
            content = item_info["content"]
            cache_key = str(item_id)

            # Check cache
            if cache_key in cache_data and not args.force_reanalyze:
                if is_cache_valid(cache_data[cache_key], available_models):
                    stats["cached"] += 1
                    sentiment_results = cache_data[cache_key]
                else:
                    # Re-analyze if cache has errors
                    sentiment_results = analyze_with_all_models(
                        content, llm_clients, system_prompt, logger
                    )
                    cache_data[cache_key] = sentiment_results
                    stats["analyzed"] += 1
                    save_cache(cache_path, cache_data, logger)
            else:
                # Analyze with all models
                sentiment_results = analyze_with_all_models(
                    content, llm_clients, system_prompt, logger
                )
                cache_data[cache_key] = sentiment_results
                stats["analyzed"] += 1
                save_cache(cache_path, cache_data, logger)

            # Update Omeka if not skipped
            if not args.skip_update:
                success = update_item_sentiment(
                    omeka_client,
                    item_id,
                    sentiment_results,
                    logger
                )
                if success:
                    stats["updated"] += 1
                else:
                    stats["errors"] += 1

            progress.update(task, advance=1)

    # Save final cache
    save_cache(cache_path, cache_data, logger)

    # Display summary
    console.print()
    summary_table = Table(title="Summary", box=box.ROUNDED)
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Count", justify="right")
    if stats["skipped_existing"] > 0:
        summary_table.add_row("Skipped (existing)", f"[dim]{stats['skipped_existing']}[/]")
    summary_table.add_row("Processed", str(stats["total"]))
    summary_table.add_row("Newly analyzed", f"[cyan]{stats['analyzed']}[/]")
    summary_table.add_row("From cache", f"[dim]{stats['cached']}[/]")
    if not args.skip_update:
        summary_table.add_row("Updated in Omeka", f"[green]{stats['updated']}[/]")
        summary_table.add_row("Update errors", f"[red]{stats['errors']}[/]" if stats['errors'] > 0 else "[dim]0[/]")
    console.print(summary_table)

    console.print()
    console.print(Panel.fit(
        f"[bold green]✓ Processing complete![/bold green]\n\n"
        f"Analyzed [cyan]{stats['analyzed']}[/] items with [cyan]{len(available_models)}[/] models\n"
        f"Cache saved to [dim]{cache_path}[/]",
        title="Done",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
