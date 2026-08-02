#!/usr/bin/env python3
"""
01_sentiment_analysis.py
========================

AI Sentiment Analysis Pipeline for IWAC Omeka S Items — generation 2.

Annotates items with the five-model panel defined in ``sentiment_core.PANEL``
and writes the results to Omeka, each model into its own six properties named
for that model.

The analysis evaluates:

- Centralité   how central Islam/Muslims are to the article
- Subjectivité 1-5, how opinionated the writing is
- Polarité     sentiment towards Islam/Muslims

What changed from generation 1
------------------------------
Generation 1 wrote to vendor-keyed properties (``iwac:gemini*``), so the stored
corpus could not say which model produced it — recovering that took a dig
through git history. This run writes to properties named for the *model*, which
is what fixes it. Generation-1 values are never read, written or deleted here;
the two generations sit side by side.

An ``iwac:sentimentModel`` value annotation was written alongside until
2026-07-31 and has been dropped. Omeka S does not index value annotations, so
it was unsearchable — a query for ``iwac:sentimentModel = 79613`` returned 0
items while 498 carried exactly that annotation — making it a second, unusable
copy of what the property name already says. Its removal is also the answer to
"why not one ``iwac:polarite`` with a value per model": that design puts the
model in the annotation, i.e. in the layer no query can reach.

Throughput
----------
Items are annotated by a pool of ``--concurrency`` workers. This is the whole
difference between a feasible run and an infeasible one: the work is almost
entirely waiting on someone else's API, so a serial loop spends the corpus's
worth of latency doing nothing. Measured 2026-07-31, median call latency ran
1.1 s (Gemini 3.5 Flash-Lite) to 104 s (Qwen3.5 via OpenRouter) — and every
provider in the panel served 5 concurrent structured requests without one
rejection, so the parallelism is free.

Concurrency multiplies with the per-item model fan-out. Running the panel one
member at a time — the normal mode — keeps in-flight requests equal to
``--concurrency``; running all five multiplies it by five.

Resuming
--------
A run over the full corpus is long, so it is built to be interrupted:

- Items already carrying values for every panel member are skipped without an
  API call beyond the listing.
- Results are cached per (item, model) as they are produced, so a resume asks
  each model only for what it has not already answered.
- Only successful results are cached; errors are retried on the next run.

So the safe response to any failure is to run the same command again.

One model at a time
-------------------
``--models`` restricts a run to part of the panel, and this is a first-class
mode rather than a degraded one. Each member writes to its own six properties,
so running them one after another builds the same result as running all five at
once — with a much smaller blast radius per run, and a clean read on one model's
cost and failure rate before committing to the next.

A scoped run reads and writes only the models named. Values already on the item
from another member are neither re-requested nor rewritten.

Usage
-----
    python AI_sentiment_analysis/01_sentiment_analysis.py --item-set-id 123
    python AI_sentiment_analysis/01_sentiment_analysis.py --resource-class-id 36
    python AI_sentiment_analysis/01_sentiment_analysis.py --item-set-id 123 --dry-run
    python AI_sentiment_analysis/01_sentiment_analysis.py --resource-class-id 36 --limit 50
    python AI_sentiment_analysis/01_sentiment_analysis.py --resource-class-id 36 \
        --models qwen3_5_122b_a10b --concurrency 24

Environment Variables
---------------------
OMEKA_BASE_URL / OMEKA_KEY_IDENTITY / OMEKA_KEY_CREDENTIAL   Omeka S API
GEMINI_API_KEY, OPENAI_API_KEY, MISTRAL_API_KEY              first-party models
OPENROUTER_API_KEY                                           Qwen + DeepSeek
"""
import sys
import argparse
import concurrent.futures as futures
import logging
import threading
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.logging import RichHandler
from rich import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.omeka_client import OmekaClient
from common.llm_provider import build_llm_client, get_model_option, LLMConfig, BaseLLMClient
from common.console_utils import standard_progress
from common.iwac_config import resolve_property_ids

# Schema, prompt, panel and analysis calls are shared with the pilot so the two
# runs stay comparable (see sentiment_core's module docstring).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.rate_limiter import QuotaExhaustedError
from sentiment_core import (  # noqa: E402
    ANALYSABLE_LANGUAGES,
    CENTRALITE_ITEM_IDS,
    MODEL_MAX_ATTEMPTS,
    MODEL_RETRY_DELAYS,
    PROMPT_FILENAME,
    PANEL,
    PANEL_REASONING_EFFECTIVE,
    POLARITE_ITEM_IDS,
    RESOURCE_FIELDS,
    RESULT_FIELD_SUFFIXES,
    SUBJECTIVITE_ITEM_IDS,
    SentimentAnalysisOutput,  # noqa: F401  (re-exported: external callers import it from here)
    PanelMember,
    analyze_with_all_models,
    get_item_content,
    get_item_language,
    is_valid_result,
    load_system_prompt,
    prompt_fingerprint,
    panel_reasoning,
    request_timeout_for_budget,
)
from sentiment_cache import SentimentCache  # noqa: E402

console = Console()

# ============================================================================
# CONFIGURATION
# ============================================================================

CACHE_DIR_NAME = "cache"
CACHE_FILE_NAME = "sentiment_v2.jsonl"

#: Items per API page when listing. Omeka caps this at 100.
PER_PAGE = 100

#: Resource-class id for newspaper articles, the usual target.
ARTICLE_CLASS_ID = 36

#: Seconds one item costs a single worker, for the up-front estimate only.
#: Divided by ``--concurrency`` to get wall clock.
#:
#: Measured 2026-07-31 across the panel on real articles (2.1-3.7k chars):
#: median call latency ran 1.1 s (Gemini 3.5 Flash-Lite, low) to 5.8 s
#: (GPT-5.6 Luna / Mistral Small 4 at their middle setting). Six is the
#: pessimistic end of the first-party range, so the estimate does not flatter a
#: run about to take all night. The progress bar's own ETA supersedes it within
#: a minute, which is the number to actually trust.
SECONDS_PER_ITEM_SERIAL = 6

#: Items annotated in parallel.
#:
#: The run used to be strictly serial, which made it hostage to whichever
#: provider was slowest — a model answering in 30 s meant 12,305 articles took
#: four days *of latency*, almost none of it work. Every provider in the panel
#: was verified to serve 5 concurrent structured requests without a single
#: rejection (2026-07-31), so the serial loop was leaving roughly a 5x speedup
#: on the table for nothing.
#:
#: Note this multiplies with the per-item model fan-out: running the whole
#: panel at concurrency 6 puts 30 requests in flight. Running one member at a
#: time — the normal mode now — makes the two numbers the same.
DEFAULT_CONCURRENCY = 6
DEFAULT_MODEL_TIMEOUT_SECONDS = 120.0
MIN_MODEL_TIMEOUT_SECONDS = MODEL_MAX_ATTEMPTS + sum(MODEL_RETRY_DELAYS) + 5


def configure_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging with Rich for elegant display."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    return logging.getLogger(__name__)


# ============================================================================
# LISTING
# ============================================================================

def _list_params(
    item_set_id: Optional[int] = None,
    resource_class_id: Optional[int] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if item_set_id is not None:
        params["item_set_id"] = item_set_id
    if resource_class_id is not None:
        params["resource_class_id"] = resource_class_id
    return params


def count_items(client: OmekaClient, params: Dict[str, Any]) -> int:
    """Total matching items, from Omeka's own count header.

    Cheaper and more honest than paging to the end to find out, and it is what
    makes an accurate progress bar possible without holding the corpus in
    memory.
    """
    response = client.session.get(
        f"{client.base_url}/items",
        params={**client._auth_params(), **params, "per_page": 1, "page": 1},
        timeout=client.timeout,
    )
    response.raise_for_status()
    return int(response.headers.get("Omeka-S-Total-Results", 0))


def iter_items(client: OmekaClient, params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield items a page at a time.

    Streaming rather than collecting: the full article corpus is 12,356 items
    whose ``bibo:content`` is the complete OCR text, which is a lot to hold in
    memory for no reason when each item is used once and discarded.
    """
    page = 1
    while True:
        url = (
            f"{client.base_url}/items?per_page={PER_PAGE}&page={page}"
            + "".join(f"&{key}={value}" for key, value in params.items())
        )
        batch = client.get_resource(url)
        if not isinstance(batch, list) or not batch:
            return
        yield from batch
        if len(batch) < PER_PAGE:
            return
        page += 1


# ============================================================================
# OMEKA WRITE
# ============================================================================

def models_already_written(item: Dict[str, Any], members: List[PanelMember]) -> List[str]:
    """Panel members whose values are already on this item.

    Probing one property per model is enough: the six are written in a single
    PATCH, so centralité present means the set is present.
    """
    return [m.key for m in members if item.get(m.probe_term)]


def build_property_values(
    member: PanelMember,
    result: Dict[str, Any],
    property_ids: Dict[str, int],
) -> Dict[str, List[Dict[str, Any]]]:
    """Build the six Omeka values for one model's answer.

    Fields whose value did not map to a controlled-vocabulary item, or whose
    justification came back empty, are omitted rather than written blank — a
    missing property is recoverable, a wrong one is not.

    No ``iwac:sentimentModel`` value annotation is written (dropped 2026-07-31).
    The property name already identifies the model, and the annotation bought
    nothing back: Omeka S does not index value annotations, so they are
    unsearchable and unfilterable — verified live, a query for
    ``iwac:sentimentModel = 79613`` returned 0 items while 498 items carried
    exactly that annotation, and ``/api/value_annotations`` is a 500. It was
    therefore a second copy of the property name, written six times per model
    per item, that no query could ever reach.

    This is also why the panel keeps thirty model-keyed properties instead of
    six multi-valued ones. The tidier design — one ``iwac:polarite`` holding a
    value per model — puts the only thing distinguishing those values in the
    unsearchable layer, so "polarité = Négatif according to DeepSeek" stops
    being answerable by query at all.
    """
    resource_ids = {
        "centralite_islam_musulmans": CENTRALITE_ITEM_IDS,
        "polarite": POLARITE_ITEM_IDS,
        "subjectivite_score": SUBJECTIVITE_ITEM_IDS,
    }

    values: Dict[str, List[Dict[str, Any]]] = {}
    for field in RESULT_FIELD_SUFFIXES:
        term = member.term(field)
        raw = result.get(field)

        value: Dict[str, Any] = {
            "type": "resource:item" if field in RESOURCE_FIELDS else "literal",
            "property_id": property_ids[term],
            "property_label": term.split(":")[-1],
            "is_public": True,
        }

        if field in RESOURCE_FIELDS:
            linked_id = resource_ids[field].get(raw)
            if not linked_id:
                continue
            value["value_resource_id"] = linked_id
            value["value_resource_name"] = "items"
        else:
            if not raw or not str(raw).strip():
                continue
            value["@value"] = str(raw).strip()
            value["@language"] = "fr"

        values[term] = [value]

    return values


def update_item_sentiment(
    client: OmekaClient,
    item_id: int,
    results: Dict[str, Dict[str, Any]],
    property_ids: Dict[str, int],
    *,
    dry_run: bool = False,
) -> str:
    """Write every model's answer for one item in a single PATCH.

    The item is re-fetched and sent back whole. Omeka treats an item's
    properties as one block, so anything missing from the payload is deleted —
    trimming the payload to "just the sentiment fields" would erase the
    article's own metadata.

    Returns one of ``updated`` / ``would_update`` / ``unchanged`` /
    ``not_found`` / ``failed``.
    """
    item_data = client.get_item(int(item_id))
    if not item_data:
        return "not_found"

    modified = False
    for model_key, result in results.items():
        member = PANEL.get(model_key)
        if member is None or not is_valid_result(result):
            continue
        for term, values in build_property_values(
            member, result, property_ids
        ).items():
            if item_data.get(term) != values:
                item_data[term] = values
                modified = True

    if not modified:
        return "unchanged"
    if dry_run:
        return "would_update"
    return "updated" if client.update_item(int(item_id), item_data) else "failed"


# ============================================================================
# SETUP
# ============================================================================

def build_clients(
    selected: List[str],
    model_timeout: float = DEFAULT_MODEL_TIMEOUT_SECONDS,
) -> Tuple[Dict[str, BaseLLMClient], Dict[str, str], Dict[str, str], List[Tuple[str, str]]]:
    """One client per selected panel member; report the ones we cannot reach.

    Reasoning depth is standardised as closely as each API allows; temperature is not.
    Temperature stays vendor-owned — the recommended values differ and both
    Google and Alibaba document a lowered temperature as a cause of looping.
    Each client takes its own from MODEL_REGISTRY.
    """
    clients: Dict[str, BaseLLMClient] = {}
    labels: Dict[str, str] = {}
    model_ids: Dict[str, str] = {}
    skipped: List[Tuple[str, str]] = []
    request_timeout = request_timeout_for_budget(model_timeout)

    for key in selected:
        member = PANEL[key]
        try:
            option = get_model_option(member.registry_key)
            clients[key] = build_llm_client(
                option,
                config=LLMConfig(
                    **panel_reasoning(key),
                    request_timeout_seconds=request_timeout,
                    # ``analyze_with_model`` owns the three-attempt budget.
                    sdk_max_retries=0,
                ),
            )
            labels[key] = member.label
            model_ids[key] = option.model
        except (RuntimeError, ValueError) as exc:
            skipped.append((member.label, str(exc)))
    return clients, labels, model_ids, skipped


def bounded_completions(
    pool: futures.ThreadPoolExecutor,
    fn: Callable[[Any], Any],
    jobs: Iterable[Any],
    max_pending: int,
) -> Iterator[futures.Future]:
    """Submit *jobs* to *pool*, keeping at most *max_pending* in flight.

    ``ThreadPoolExecutor.map`` would drain the whole generator up front. Here
    that means materialising 12,305 items — each carrying its complete OCR text
    — to gain nothing, since the pool can only work on a handful at a time.
    Bounding the queue keeps the producer lazy and the run's memory flat, and it
    keeps the listing request that produces item N+1 from happening hours before
    anything looks at it.

    Yields futures as they finish, so the caller can advance a progress bar
    against real completions rather than submissions.
    """
    pending: set = set()
    for job in jobs:
        pending.add(pool.submit(fn, job))
        if len(pending) >= max_pending:
            done, pending = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
            yield from done
    while pending:
        done, pending = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
        yield from done


def format_duration(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes = remainder // 60
    if hours >= 24:
        return f"~{hours // 24}d {hours % 24}h"
    return f"~{hours}h {minutes}m" if hours else f"~{minutes}m"


def _initial_stats() -> Dict[str, int]:
    return {
        "seen": 0,
        "no_content": 0,
        "language_skipped": 0,
        "language_unknown": 0,
        "already_done": 0,
        "analyzed": 0,
        "from_cache": 0,
        "model_errors": 0,
        "updated": 0,
        "would_update": 0,
        "unchanged": 0,
        "not_found": 0,
        "failed": 0,
    }


@dataclass
class SentimentRunner:
    """Stateful, thread-safe execution of one configured sentiment run."""

    args: argparse.Namespace
    listing_client: OmekaClient
    clients: Dict[str, BaseLLMClient]
    labels: Dict[str, str]
    model_ids: Dict[str, str]
    members: List[PanelMember]
    property_ids: Dict[str, int]
    system_prompt: str
    prompt_id: str
    expected_provenance: Dict[str, Dict[str, str]]
    sources: List[Dict[str, Any]]
    cache: SentimentCache
    logger: logging.Logger
    stats: Dict[str, int] = dataclass_field(default_factory=_initial_stats)
    skipped_languages: Dict[str, int] = dataclass_field(default_factory=dict)
    halted: Dict[str, str] = dataclass_field(default_factory=dict)
    stop: threading.Event = dataclass_field(default_factory=threading.Event)
    _stats_lock: threading.Lock = dataclass_field(default_factory=threading.Lock)
    _thread_local: threading.local = dataclass_field(default_factory=threading.local)

    def bump(self, key: str, by: int = 1) -> None:
        with self._stats_lock:
            self.stats[key] = self.stats.get(key, 0) + by

    def worker_client(self) -> OmekaClient:
        """Return one Omeka client per worker; requests.Session is not thread-safe."""
        existing = getattr(self._thread_local, "omeka", None)
        if existing is None:
            existing = OmekaClient.from_env()
            self._thread_local.omeka = existing
        return existing

    def jobs(self) -> Iterator[Tuple[Any, str, List[str]]]:
        """Yield eligible item text and the panel members it still needs."""
        produced = 0
        for params in self.sources:
            for item in iter_items(self.listing_client, params):
                if self.stop.is_set() or (
                    self.args.limit and produced >= self.args.limit
                ):
                    return

                self.bump("seen")
                item_id = item.get("o:id")
                language = get_item_language(item)
                if language is None:
                    self.bump("language_unknown")
                    continue
                if language not in ANALYSABLE_LANGUAGES:
                    self.bump("language_skipped")
                    with self._stats_lock:
                        self.skipped_languages[language] = (
                            self.skipped_languages.get(language, 0) + 1
                        )
                    continue

                content = get_item_content(item)
                if not content.strip():
                    self.bump("no_content")
                    continue

                written = (
                    []
                    if (self.args.force_reanalyze or self.args.rewrite)
                    else models_already_written(item, self.members)
                )
                if len(written) == len(self.members):
                    self.bump("already_done")
                    continue

                pending = (
                    list(self.clients)
                    if self.args.force_reanalyze
                    else [
                        key
                        for key in self.clients
                        if key not in written
                        and not self.cache.has(
                            item_id,
                            key,
                            **self.expected_provenance[key],
                        )
                    ]
                )
                yield item_id, content, pending
                produced += 1

    def annotate(self, job: Tuple[Any, str, List[str]]) -> None:
        """Analyze and optionally write one eligible Omeka item."""
        item_id, content, pending = job
        if self.stop.is_set():
            return

        if pending:
            try:
                fresh = analyze_with_all_models(
                    content,
                    {key: self.clients[key] for key in pending},
                    self.system_prompt,
                    self.logger,
                    self.labels,
                    timeout=self.args.model_timeout,
                )
            except QuotaExhaustedError as exc:
                self.halted.setdefault("reason", str(exc))
                self.stop.set()
                return
            for key, result in fresh.items():
                if is_valid_result(result):
                    self.cache.put(
                        item_id,
                        key,
                        result,
                        model_id=self.model_ids[key],
                        reasoning=PANEL_REASONING_EFFECTIVE[key],
                        prompt=self.prompt_id,
                    )
                else:
                    self.bump("model_errors")
                    if self.args.verbose:
                        console.print(
                            f"  [red]✗[/] {self.labels[key]} on item {item_id}: "
                            f"{result.get('analysis_error')}"
                        )
            self.bump("analyzed")
        else:
            self.bump("from_cache")

        if self.args.skip_update:
            return

        results = {
            key: result
            for key, result in self.cache.results_for(
                item_id, expected=self.expected_provenance
            ).items()
            if key in self.clients
        }
        if not results:
            return
        status = update_item_sentiment(
            self.worker_client(),
            item_id,
            results,
            self.property_ids,
            dry_run=self.args.dry_run,
        )
        self.bump(status)
        if status == "failed":
            console.print(f"  [red]✗[/] PATCH failed for item {item_id} (see log)")

    def run(self, progress_total: int) -> None:
        """Consume the lazy job stream with a bounded worker queue."""
        console.rule("[bold cyan]Processing")
        with self.cache, standard_progress(console, show_eta=True) as progress:
            task = progress.add_task("[cyan]Annotating...", total=progress_total)
            with futures.ThreadPoolExecutor(max_workers=self.args.concurrency) as pool:
                for future in bounded_completions(
                    pool,
                    self.annotate,
                    self.jobs(),
                    self.args.concurrency * 2,
                ):
                    future.result()
                    progress.update(task, advance=1)


def report_run(runner: SentimentRunner, cache_path: Path) -> int:
    """Render the final run summary and return the process exit code."""
    stats = runner.stats
    args = runner.args
    summary = Table(title="Summary", box=box.ROUNDED)
    summary.add_column("Metric", style="dim")
    summary.add_column("Count", justify="right")
    summary.add_row("Items listed", f"{stats['seen']:,}")
    summary.add_row("Skipped — already annotated", f"[dim]{stats['already_done']:,}[/]")
    summary.add_row("Skipped — no content", f"[dim]{stats['no_content']:,}[/]")
    language_detail = (
        f" ({', '.join(f'{key} {value}' for key, value in sorted(runner.skipped_languages.items()))})"
        if runner.skipped_languages else ""
    )
    summary.add_row(
        "Skipped — other language",
        f"[dim]{stats['language_skipped']:,}{language_detail}[/]",
    )
    summary.add_row("Skipped — language untagged", f"[dim]{stats['language_unknown']:,}[/]")
    summary.add_row("Newly analysed", f"[cyan]{stats['analyzed']:,}[/]")
    summary.add_row("Served from cache", f"[dim]{stats['from_cache']:,}[/]")
    summary.add_row(
        "Model call failures",
        f"[red]{stats['model_errors']:,}[/]" if stats["model_errors"] else "[dim]0[/]",
    )
    if not args.skip_update:
        label = "Would update" if args.dry_run else "Updated in Omeka"
        key = "would_update" if args.dry_run else "updated"
        summary.add_row(label, f"[green]{stats[key]:,}[/]")
        summary.add_row("Already up to date", f"[dim]{stats['unchanged']:,}[/]")
        summary.add_row("Not found", f"[dim]{stats['not_found']:,}[/]")
        summary.add_row(
            "PATCH failures",
            f"[red]{stats['failed']:,}[/]" if stats["failed"] else "[dim]0[/]",
        )
    console.print()
    console.print(summary)

    if runner.halted:
        console.print()
        console.print(Panel(
            f"[bold red]Stopped: the provider account is out of quota or "
            f"credit.[/bold red]\n\n{runner.halted['reason']}\n\n"
            f"This is not a model failure and retrying will not help — top up "
            f"the account, then re-run the same command. Everything already "
            f"answered is in the cache and will not be requested again:\n"
            f"[dim]{cache_path}[/]",
            title="Halted", border_style="red",
        ))
        return 2

    incomplete = stats["model_errors"] + stats["failed"]
    console.print()
    console.print(Panel.fit(
        (f"[bold yellow]Finished with {incomplete:,} failure(s)[/bold yellow]\n\n"
         if incomplete else "[bold green]✓ Complete[/bold green]\n\n")
        + f"Cache: [dim]{cache_path}[/]\n"
        + ("Re-run the same command to retry the failures — cached successes "
           "are not re-requested." if incomplete else
           "Re-running is a no-op: annotated items are skipped."),
        title="Done", border_style="yellow" if incomplete else "green",
    ))
    return 1 if incomplete else 0


@dataclass(frozen=True)
class PreparedSentimentRun:
    runner: SentimentRunner
    cache_path: Path
    total_items: int


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the generation-2 sentiment panel on Omeka S items."
    )
    source = parser.add_argument_group("what to annotate")
    source.add_argument("--item-set-id", type=str, help="Item set ID(s), comma-separated")
    source.add_argument(
        "--resource-class-id",
        type=int,
        nargs="?",
        const=ARTICLE_CLASS_ID,
        help=f"Annotate a whole resource class (default {ARTICLE_CLASS_ID}, newspaper articles)",
    )
    source.add_argument(
        "--limit", type=int, default=None,
        help="Stop after N items needing work (for a trial run)",
    )
    source.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated subset of the panel to run "
             f"(default: all). Choices: {', '.join(PANEL)}",
    )

    behaviour = parser.add_argument_group("behaviour")
    behaviour.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Items annotated in parallel (default {DEFAULT_CONCURRENCY}; "
             "1 = serial behaviour)",
    )
    behaviour.add_argument(
        "--model-timeout", type=float, default=DEFAULT_MODEL_TIMEOUT_SECONDS,
        help="Maximum seconds for one model across all retry attempts "
             f"(default {DEFAULT_MODEL_TIMEOUT_SECONDS:g})",
    )
    behaviour.add_argument("--dry-run", action="store_true", help="Analyse and cache, but PATCH nothing")
    behaviour.add_argument(
        "--skip-update", action="store_true",
        help="Analyse and cache only; do not touch Omeka at all",
    )
    behaviour.add_argument(
        "--force-reanalyze", action="store_true",
        help="Ignore the cache AND the already-annotated guard",
    )
    behaviour.add_argument(
        "--rewrite", action="store_true",
        help="Re-PATCH items that already carry values, reusing answers whose "
             "cached provenance still matches",
    )
    behaviour.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    behaviour.add_argument(
        "--verbose", action="store_true", help="Log per-model failures as they happen"
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> List[int]:
    """Validate CLI invariants and return normalized item-set IDs."""
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.model_timeout < MIN_MODEL_TIMEOUT_SECONDS:
        raise ValueError(
            f"--model-timeout must be at least {MIN_MODEL_TIMEOUT_SECONDS:g} seconds"
        )
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if not args.item_set_id and args.resource_class_id is None:
        raise ValueError("Give --item-set-id or --resource-class-id")
    try:
        return (
            [int(value.strip()) for value in args.item_set_id.split(",")]
            if args.item_set_id else []
        )
    except ValueError as exc:
        raise ValueError("Invalid item set ID format") from exc


def selected_model_keys(raw_models: Optional[str]) -> List[str]:
    selected = (
        [value.strip() for value in raw_models.split(",") if value.strip()]
        if raw_models else list(PANEL)
    )
    unknown = [key for key in selected if key not in PANEL]
    if unknown:
        raise ValueError(
            f"Unknown model(s): {', '.join(unknown)}. Available: {', '.join(PANEL)}"
        )
    return selected


def available_clients(
    selected: List[str], model_timeout: float
) -> Tuple[Dict[str, BaseLLMClient], Dict[str, str], Dict[str, str]]:
    clients, labels, model_ids, skipped = build_clients(selected, model_timeout)
    if skipped:
        console.print()
        for label, reason in skipped:
            console.print(f"[yellow]![/] Skipping [bold]{label}[/] — {reason}")
        console.print("[dim]  Qwen and DeepSeek both need OPENROUTER_API_KEY.[/]")
    if not clients:
        raise RuntimeError("No models available — nothing to run")
    return clients, labels, model_ids


def panel_property_ids(
    client: OmekaClient,
    members: List[PanelMember],
    *,
    skip_update: bool,
) -> Dict[str, int]:
    if skip_update:
        return {}
    wanted_terms = [term for member in members for term in member.terms]
    try:
        return resolve_property_ids(client, wanted_terms)
    except KeyError as exc:
        raise RuntimeError(f"Vocabulary not ready: {exc.args[0]}") from exc


def show_configuration(
    args: argparse.Namespace,
    client: OmekaClient,
    item_set_ids: List[int],
    clients: Dict[str, BaseLLMClient],
    labels: Dict[str, str],
    model_ids: Dict[str, str],
    prompt_id: str,
    total_items: int,
) -> None:
    table = Table(title="Configuration", box=box.ROUNDED)
    table.add_column("Setting", style="dim")
    table.add_column("Value", style="green")
    table.add_row("Omeka", client.base_url)
    table.add_row(
        "Target",
        f"item sets {', '.join(map(str, item_set_ids))}" if item_set_ids
        else f"resource class {args.resource_class_id}",
    )
    table.add_row("Items in scope", f"{total_items:,}")
    table.add_row(
        "Languages", ", ".join(sorted(ANALYSABLE_LANGUAGES)) + " [dim](others skipped)[/]"
    )
    table.add_row(
        "Panel",
        "\n".join(
            f"{labels[key]}  [dim]{model_ids[key]}  →  "
            f"iwac:{PANEL[key].property_prefix}*[/]"
            for key in clients
        ),
    )
    table.add_row(
        "Reasoning",
        ", ".join(
            f"{labels[key]}={PANEL_REASONING_EFFECTIVE[key].split(' ')[0]}"
            for key in clients
        ),
    )
    table.add_row(
        "Model timeout",
        f"{args.model_timeout:g}s total / "
        f"{request_timeout_for_budget(args.model_timeout):.1f}s per attempt",
    )
    table.add_row("Prompt", f"{PROMPT_FILENAME} [dim]#{prompt_id}[/]")
    table.add_row(
        "Writes to Omeka",
        "[bold]no — analysis only[/]" if args.skip_update
        else "[bold]no — dry run[/]" if args.dry_run
        else "[bold yellow]YES[/]",
    )
    concurrency = f"{args.concurrency} items in parallel"
    if len(clients) > 1:
        concurrency += (
            f" [dim](x{len(clients)} models = "
            f"{args.concurrency * len(clients)} requests in flight)[/]"
        )
    table.add_row("Concurrency", concurrency)
    if args.limit:
        table.add_row("Limit", str(args.limit))
    if args.force_reanalyze:
        table.add_row("Force re-analyze", "[yellow]YES — ignores cache & guard[/]")
    console.print(table)


def load_run_cache(
    expected_provenance: Dict[str, Dict[str, str]], logger: logging.Logger
) -> Tuple[SentimentCache, Path]:
    cache_path = Path(__file__).resolve().parent / CACHE_DIR_NAME / CACHE_FILE_NAME
    cache = SentimentCache(path=cache_path, logger=logger)
    report = cache.load()
    if report.records:
        matching = cache.count_matching(expected_provenance)
        console.print(
            f"\n[green]✓[/] Cache: [bold]{report.records:,}[/] results across "
            f"[bold]{report.items:,}[/] items; [bold]{matching:,}[/] match this "
            "run's model, reasoning, and prompt and can be reused"
        )
    if report.skipped_malformed:
        console.print(
            f"[yellow]![/] {report.skipped_malformed} unreadable cache line(s) "
            "skipped (expected after an interrupted run)"
        )
    return cache, cache_path


def prepare_run(
    args: argparse.Namespace,
    logger: logging.Logger,
    item_set_ids: List[int],
) -> PreparedSentimentRun:
    client = OmekaClient.from_env()
    clients, labels, model_ids = available_clients(
        selected_model_keys(args.models), args.model_timeout
    )
    members = [PANEL[key] for key in clients]
    property_ids = panel_property_ids(client, members, skip_update=args.skip_update)
    system_prompt = load_system_prompt()
    prompt_id = prompt_fingerprint(system_prompt)
    expected_provenance = {
        key: {
            "model_id": model_ids[key],
            "reasoning": PANEL_REASONING_EFFECTIVE[key],
            "prompt": prompt_id,
        }
        for key in clients
    }
    sources = (
        [_list_params(item_set_id=item_set_id) for item_set_id in item_set_ids]
        if item_set_ids else [_list_params(resource_class_id=args.resource_class_id)]
    )
    total_items = sum(count_items(client, params) for params in sources)
    show_configuration(
        args, client, item_set_ids, clients, labels, model_ids, prompt_id, total_items
    )
    cache, cache_path = load_run_cache(expected_provenance, logger)
    runner = SentimentRunner(
        args=args,
        listing_client=client,
        clients=clients,
        labels=labels,
        model_ids=model_ids,
        members=members,
        property_ids=property_ids,
        system_prompt=system_prompt,
        prompt_id=prompt_id,
        expected_provenance=expected_provenance,
        sources=sources,
        cache=cache,
        logger=logger,
    )
    return PreparedSentimentRun(runner=runner, cache_path=cache_path, total_items=total_items)


def confirm_run(prepared: PreparedSentimentRun) -> bool:
    args = prepared.runner.args
    if args.skip_update or args.dry_run or args.yes:
        return True
    console.print(Panel(
        f"About to PATCH up to [bold]{prepared.total_items:,}[/] items on "
        f"[cyan]{prepared.runner.listing_client.base_url}[/].\n\n"
        f"Writes {len(prepared.runner.members) * 6} properties per item, all "
        f"under iwac:<model>* — generation-1 values are not read or modified.\n"
        f"Estimated runtime: [bold]"
        f"{format_duration(prepared.total_items * SECONDS_PER_ITEM_SERIAL / args.concurrency)}[/] "
        f"at concurrency {args.concurrency} (interrupt and re-run to resume).",
        title="Confirm", border_style="yellow",
    ))
    return console.input("\n[bold]Proceed? [y/N]:[/] ").strip().lower() in ("y", "yes")


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    console.print(Panel.fit(
        "[bold cyan]AI Sentiment Analysis Pipeline[/bold cyan]\n"
        f"[dim]{len(PANEL)}-model panel with model-keyed properties[/dim]",
        border_style="cyan",
    ))

    args = build_argument_parser().parse_args()
    logger = configure_logging(args.verbose)
    try:
        item_set_ids = validate_arguments(args)
        prepared = prepare_run(args, logger, item_set_ids)
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]✗[/] {exc}")
        return 2

    if not confirm_run(prepared):
        console.print("[yellow]Aborted — no changes made.[/]")
        return 1

    prepared.runner.run(args.limit or prepared.total_items)
    return report_run(prepared.runner, prepared.cache_path)


if __name__ == "__main__":
    sys.exit(main())
