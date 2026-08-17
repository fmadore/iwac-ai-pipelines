#!/usr/bin/env python3
"""
02_pilot_new_panel.py
=====================

Pilot a candidate sentiment panel on a sample of already-annotated articles.

Reads from Omeka, writes nothing to it. The point is to find out whether a new
set of models earns a place — and 24 new Omeka properties across 12,000+ items
— before any of that is created. Results land in a local JSON file that
``03_pilot_report.py`` turns into agreement and self-consistency tables.

The sample is drawn from articles the live panel has already annotated, so a
candidate is measured on the same population production runs against.

Because the schema, prompt and call path come from ``sentiment_core``, the
pilot annotations are produced exactly the way production produces them; the
only variable is the model.

Usage
-----
    python AI_sentiment_analysis/02_pilot_new_panel.py
    python AI_sentiment_analysis/02_pilot_new_panel.py --sample-size 200 --seed 42
    python AI_sentiment_analysis/02_pilot_new_panel.py --repeats 3 --sample-size 50
    python AI_sentiment_analysis/02_pilot_new_panel.py --models gemini_3_6_flash,mistral_small_2603

``--repeats N`` re-annotates each article N times with the same models. Use it
to measure self-consistency: DeepSeek V4 runs at the vendor-recommended
temperature 1.0 and Qwen3.5 at 0.7, so a low agreement score for them is
ambiguous between "disagrees with the panel" and "disagrees with itself". The
2026-07-29 pilot measured DeepSeek at 0.52 polarite self-consistency, so this
is not hypothetical.

Environment Variables
---------------------
OMEKA_BASE_URL / OMEKA_KEY_IDENTITY / OMEKA_KEY_CREDENTIAL   Omeka S API
GEMINI_API_KEY, OPENAI_API_KEY, MISTRAL_API_KEY              first-party models
OPENROUTER_API_KEY                                           Qwen + DeepSeek
SELFHOSTED_LLM_BASE_URL / SELFHOSTED_LLM_API_KEY             self-hosted candidates

A model whose credentials are missing is reported as skipped and the pilot runs
without it, so no tunnel is needed to trial the hosted members.
"""
import sys
import json
import random
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.logging import RichHandler
from rich import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.omeka_client import OmekaClient
from common.llm_provider import build_llm_client, get_model_option, LLMConfig, BaseLLMClient
from common.checkpoint import atomic_write_text
from common.console_utils import standard_progress
from common.log_redaction import install_credential_redaction

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sentiment_core import (  # noqa: E402
    ANALYSABLE_LANGUAGES,
    PANEL,
    PILOT_CANDIDATES,
    PANEL_REASONING_EFFECTIVE,
    analyze_with_all_models,
    get_item_content,
    get_item_language,
    load_system_prompt,
    prompt_fingerprint,
    panel_reasoning,
)

console = Console()

# ============================================================================
# CONFIGURATION
# ============================================================================

#: Omeka resource class for newspaper articles.
ARTICLE_CLASS_ID = 36

#: An "is already annotated" probe, so the sample is guaranteed comparable.
#:
#: This was ``iwac:geminiCentralite`` (id 319) until the generation-1 properties
#: were deleted on 2026-08-07 — after which it matched nothing and the pilot
#: sampled an empty corpus. It now names a live generation-2 property and is
#: resolved through the API rather than hardcoded: property IDs are assigned at
#: vocabulary-import time, so a literal is a claim this file cannot check.
PROBE_PROPERTY_TERM = "iwac:gpt56LunaCentralite"

#: The panel under test: everything production writes, plus whatever is
#: currently under evaluation. Both halves, their reasoning depth and their
#: property prefixes come from ``sentiment_core`` — the same definitions
#: production runs against. When this file carried its own copy, "the pilot"
#: and "what shipped" were two different things that merely looked alike.
#:
#: Running the live members alongside the candidates is what makes the report
#: readable: agreement is only interesting against the annotators already in
#: use, so ``03_pilot_report.py`` needs both in one payload. Candidates stay
#: unwritable throughout — ``01`` iterates ``PANEL`` alone.
V2_PANEL = {**PANEL, **PILOT_CANDIDATES}

OUTPUT_DIR_NAME = "cache/pilot"
DEFAULT_SAMPLE_SIZE = 200
DEFAULT_SEED = 42
PER_PAGE = 100
ITEMS_PER_SAMPLED_PAGE = 10


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    # Credentials ride in Omeka query strings and provider headers; keep them
    # out of anything urllib3 or an SDK decides to log.
    install_credential_redaction()
    return logging.getLogger(__name__)


# ============================================================================
# SAMPLING
# ============================================================================

def resolve_probe_property(client: OmekaClient) -> int:
    """Property ID of :data:`PROBE_PROPERTY_TERM`, or abort.

    Failing loudly matters: a probe that resolves to nothing does not error, it
    silently matches zero items, and the pilot reports an empty sample as though
    the corpus were unannotated.
    """
    pid = client.get_property_id(PROBE_PROPERTY_TERM)
    if pid is None:
        raise SystemExit(
            f"{PROBE_PROPERTY_TERM} is not in the Omeka vocabulary — the pilot "
            f"cannot find already-annotated articles to sample."
        )
    return pid


def _items_page(
    client: OmekaClient, page: int, probe_property_id: int, per_page: int = PER_PAGE
) -> List[Dict[str, Any]]:
    """One page of articles that already carry a live panel annotation."""
    url = (
        f"{client.base_url}/items"
        f"?resource_class_id={ARTICLE_CLASS_ID}"
        f"&property%5B0%5D%5Bproperty%5D={probe_property_id}"
        f"&property%5B0%5D%5Btype%5D=ex"
        f"&per_page={per_page}&page={page}"
    )
    result = client.get_resource(url)
    return result if isinstance(result, list) else []


def _last_page(client: OmekaClient, probe_property_id: int) -> int:
    """Binary-search the final page rather than paging the whole corpus."""
    lo, hi = 1, 2
    while _items_page(client, hi, probe_property_id, per_page=1):
        lo, hi = hi, hi * 2
        if hi > 1_000_000:  # pathological guard
            break
    # invariant: page `lo` has data, page `hi` does not
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _items_page(client, mid, probe_property_id, per_page=1):
            lo = mid
        else:
            hi = mid
    return lo


def sample_articles(
    client: OmekaClient,
    sample_size: int,
    seed: int,
    console: Console,
) -> List[Dict[str, Any]]:
    """Draw a sample of annotated articles spread across the corpus.

    This is a two-stage cluster sample, not a simple random sample: random
    pages, then random items within each page. Pages are id-ordered, so the
    spread tracks ingest order. Cheap (a few dozen requests) and good enough to
    compare panels; it is not a basis for population-level estimates.
    """
    probe_property_id = resolve_probe_property(client)
    with console.status("[bold green]Locating corpus bounds...", spinner="dots"):
        total_items = _last_page(client, probe_property_id)
    max_page = max(1, -(-total_items // PER_PAGE))  # ceil
    console.print(
        f"[green]✓[/] ~{total_items:,} annotated articles across {max_page} pages"
    )

    rng = random.Random(seed)
    n_pages = max(1, -(-sample_size // ITEMS_PER_SAMPLED_PAGE))
    pages = sorted(rng.sample(range(1, max_page + 1), min(n_pages, max_page)))

    sampled: List[Dict[str, Any]] = []
    with standard_progress(console) as progress:
        task = progress.add_task("[cyan]Sampling articles...", total=len(pages))
        for page in pages:
            # Same language gate as production (01): a pilot drawn from a
            # different population than the run it validates measures the wrong
            # thing.
            items = [
                it for it in _items_page(client, page, probe_property_id)
                if get_item_content(it).strip()
                and get_item_language(it) in ANALYSABLE_LANGUAGES
            ]
            if items:
                take = min(ITEMS_PER_SAMPLED_PAGE, len(items))
                sampled.extend(rng.sample(items, take))
            progress.update(task, advance=1)

    rng.shuffle(sampled)
    return sampled[:sample_size]


# A pilot used to read each article's generation-1 annotations off the item and
# score the candidate against them. Those properties were deleted from Omeka on
# 2026-08-07; generation 1 now lives only on the Hugging Face full mirror. A
# pilot compares candidates against each other and, through the sample, against
# the live panel.


# ============================================================================
# MAIN
# ============================================================================

def load_partial(
    path: Path, prompt_id: str, logger: logging.Logger
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Read repeats already recorded for this run: item_id -> run index -> result.

    Records written under a *different* prompt are ignored rather than reused.
    Resuming across a prompt edit would silently mix two instruments in one
    pilot, which is precisely what the pilot exists to rule out.

    A torn final line is skipped: it is the expected shape of an interrupted
    run, which is the case this file is for.
    """
    done: Dict[str, Dict[int, Dict[str, Any]]] = {}
    if not path.exists():
        return done

    skipped_prompt = malformed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if record.get("prompt") != prompt_id:
            skipped_prompt += 1
            continue
        item_id, run_index = str(record.get("item_id")), record.get("run")
        if item_id and isinstance(run_index, int) and isinstance(record.get("result"), dict):
            done.setdefault(item_id, {})[run_index] = record["result"]
        else:
            malformed += 1

    if skipped_prompt:
        logger.warning(
            f"Ignored {skipped_prompt} cached repeat(s) from a different prompt."
        )
    if malformed:
        logger.warning(f"Skipped {malformed} unreadable line(s) in {path.name}.")
    return done


@dataclass(frozen=True)
class PilotModels:
    """Reachable pilot clients and the metadata needed in its manifest."""

    clients: Dict[str, BaseLLMClient]
    labels: Dict[str, str]
    model_ids: Dict[str, str]
    skipped: List[tuple[str, str, str]]


def build_clients(selected: List[str], logger: logging.Logger) -> PilotModels:
    """Build one client per requested model; report the ones we cannot reach."""
    clients: Dict[str, BaseLLMClient] = {}
    labels: Dict[str, str] = {}
    model_ids: Dict[str, str] = {}
    skipped: List[tuple] = []

    # Reasoning depth is standardised as closely as each API allows; temperature is
    # not. Temperature stays vendor-owned — the recommended values differ
    # (1.0 DeepSeek, 0.7 Qwen, 0.3 Mistral Small 4, unset Gemini) and both
    # Google and Alibaba document a lowered temperature as a cause of looping.
    # Each client takes its own from MODEL_REGISTRY.
    for prefix in selected:
        member = V2_PANEL[prefix]
        try:
            option = get_model_option(member.registry_key)
            clients[prefix] = build_llm_client(
                option, config=LLMConfig(**panel_reasoning(prefix))
            )
            labels[prefix] = member.label
            model_ids[prefix] = option.model
        except (RuntimeError, ValueError) as exc:
            skipped.append((prefix, member.label, str(exc)))
    return PilotModels(clients, labels, model_ids, skipped)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the pilot CLI."""
    parser = argparse.ArgumentParser(
        description="Pilot a candidate sentiment panel on a sample of annotated articles. "
                    "Never writes to Omeka."
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help=f"Articles to annotate (default: {DEFAULT_SAMPLE_SIZE})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"RNG seed for a reproducible sample (default: {DEFAULT_SEED})")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Annotate each article N times to measure self-consistency (default: 1)")
    parser.add_argument("--models", type=str, default=None,
                        help=f"Comma-separated subset of: {','.join(V2_PANEL)}")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: cache/pilot/pilot_<timestamp>.json)")
    return parser


def selected_models(raw_models: Optional[str]) -> List[str]:
    """Validate an optional comma-separated subset of panel member keys."""
    selected = list(V2_PANEL) if raw_models is None else [
        model.strip() for model in raw_models.split(",") if model.strip()
    ]
    unknown = [model for model in selected if model not in V2_PANEL]
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")
    if not selected:
        raise ValueError("No models selected")
    return selected


def show_skipped_models(skipped: List[tuple[str, str, str]]) -> None:
    """Explain unavailable providers while allowing a partial pilot."""
    if not skipped:
        return
    console.print()
    for _prefix, label, reason in skipped:
        console.print(f"[yellow]![/] Skipping [bold]{label}[/] — {reason}")
    console.print(
        "[dim]  Qwen and DeepSeek both need OPENROUTER_API_KEY; a self-hosted "
        "candidate needs SELFHOSTED_LLM_BASE_URL, i.e. a running server and an "
        "open tunnel (see serving/README.md). The pilot continues with whatever "
        "is reachable.[/]"
    )


def show_configuration(
    args: argparse.Namespace,
    client: OmekaClient,
    models: PilotModels,
    system_prompt: str,
    prompt_id: str,
) -> None:
    """Display the complete read-only pilot configuration."""
    config_table = Table(title="Pilot configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Omeka URL", client.base_url)
    config_table.add_row("Sample size", str(args.sample_size))
    config_table.add_row("Seed", str(args.seed))
    config_table.add_row("Repeats", str(args.repeats))
    config_table.add_row(
        "Models",
        "\n".join(
            f"{models.labels[prefix]}  [dim]{models.model_ids[prefix]}[/]"
            for prefix in models.clients
        ),
    )
    config_table.add_row(
        "Prompt", f"#{prompt_id} · {len(system_prompt):,} chars",
    )
    config_table.add_row("Writes to Omeka", "[bold green]none[/]")
    console.print(config_table)
    console.print()


def resolve_output_path(raw_output: Optional[str], timestamp: str) -> Path:
    """Resolve an explicit output or the timestamped default pilot path."""
    if raw_output:
        return Path(raw_output)
    return Path(__file__).resolve().parent / OUTPUT_DIR_NAME / f"pilot_{timestamp}.json"


def annotate_articles(
    articles: List[Dict[str, Any]],
    models: PilotModels,
    system_prompt: str,
    prompt_id: str,
    repeats: int,
    partial_path: Path,
    completed: Dict[str, Dict[int, Dict[str, Any]]],
    logger: logging.Logger,
) -> tuple[Dict[str, Any], int]:
    """Run or resume all article repeats, appending each completed call."""
    results: Dict[str, Any] = {}
    errors = 0
    total_calls = len(articles) * repeats
    console.rule("[bold cyan]Annotating")
    with open(partial_path, "a", encoding="utf-8") as partial, \
            standard_progress(console, show_eta=True) as progress:
        task = progress.add_task("[cyan]Running panel...", total=total_calls)
        for item in articles:
            item_id = str(item.get("o:id"))
            content = get_item_content(item)
            cached = completed.get(item_id, {})
            runs = []
            for run_index in range(repeats):
                run = cached.get(run_index)
                if run is None:
                    run = analyze_with_all_models(
                        content, models.clients, system_prompt, logger, models.labels
                    )
                    partial.write(json.dumps({
                        "prompt": prompt_id,
                        "item_id": item_id,
                        "run": run_index,
                        "result": run,
                    }, ensure_ascii=False) + "\n")
                    partial.flush()
                errors += sum(
                    1 for result in run.values() if result.get("analysis_error")
                )
                runs.append(run)
                progress.update(task, advance=1)
            results[item_id] = {
                "title": item.get("o:title"),
                "n_chars": len(content),
                "v2_runs": runs,
            }
    return results, errors


def build_payload(
    *,
    timestamp: str,
    articles: List[Dict[str, Any]],
    results: Dict[str, Any],
    models: PilotModels,
    seed: int,
    repeats: int,
    system_prompt: str,
    prompt_id: str,
) -> Dict[str, Any]:
    """Build a self-describing pilot artifact for later comparison."""
    return {
        "manifest": {
            "generated_utc": timestamp,
            "sample_size": len(articles),
            "seed": seed,
            "repeats": repeats,
            "sampling": "two-stage cluster: random pages, random items within page",
            "v2_models": {
                prefix: {
                    "label": models.labels[prefix],
                    "model_id": models.model_ids[prefix],
                }
                for prefix in models.clients
            },
            "v2_skipped": [
                {"prefix": prefix, "label": label, "reason": reason}
                for prefix, label, reason in models.skipped
            ],
            "v2_reasoning_requested": {
                prefix: panel_reasoning(prefix) for prefix in models.clients
            },
            "v2_reasoning_effective": {
                prefix: PANEL_REASONING_EFFECTIVE[prefix] for prefix in models.clients
            },
            "prompt_chars": len(system_prompt),
            "prompt_fingerprint": prompt_id,
        },
        "articles": results,
    }


def show_summary(
    results: Dict[str, Any],
    models: PilotModels,
    repeats: int,
    errors: int,
    out_path: Path,
) -> None:
    """Display pilot totals and the next report command."""
    summary = Table(title="Pilot summary", box=box.ROUNDED)
    summary.add_column("Metric", style="dim")
    summary.add_column("Value", justify="right")
    summary.add_row("Articles annotated", str(len(results)))
    summary.add_row("Models run", str(len(models.clients)))
    summary.add_row(
        "Models skipped",
        f"[yellow]{len(models.skipped)}[/]" if models.skipped else "[dim]0[/]",
    )
    summary.add_row("Total annotations", str(len(results) * repeats * len(models.clients)))
    summary.add_row("Annotation errors", f"[red]{errors}[/]" if errors else "[dim]0[/]")
    console.print()
    console.print(summary)
    console.print()
    console.print(Panel.fit(
        "[bold green]✓ Pilot complete[/bold green]\n\n"
        f"Results: [cyan]{out_path}[/]\n"
        f"[dim]Nothing was written to Omeka. Next: 03_pilot_report.py {out_path.name}[/]",
        title="Done",
        border_style="green",
    ))


def main() -> int:
    logger = configure_logging()
    console.print(Panel.fit(
        "[bold cyan]Sentiment Panel Pilot[/bold cyan]\n"
        "[dim]Candidate models vs the generation-1 annotations — "
        "reads Omeka, writes nothing to it[/dim]",
        border_style="cyan",
    ))
    args = build_argument_parser().parse_args()

    if args.sample_size <= 0 or args.repeats <= 0:
        console.print("[red]✗[/] --sample-size and --repeats must be positive")
        return 2

    try:
        selected = selected_models(args.models)
    except ValueError as exc:
        console.print(f"[red]✗[/] {exc}")
        console.print(f"[dim]Available: {', '.join(V2_PANEL)}[/]")
        return 2

    try:
        client = OmekaClient.from_env()
    except ValueError as exc:
        console.print(f"[red]✗[/] {exc}")
        return 2

    models = build_clients(selected, logger)
    show_skipped_models(models.skipped)
    if not models.clients:
        console.print("\n[red]✗[/] No models available — nothing to pilot.")
        return 1

    system_prompt = load_system_prompt()
    prompt_id = prompt_fingerprint(system_prompt)

    show_configuration(args, client, models, system_prompt, prompt_id)

    articles = sample_articles(client, args.sample_size, args.seed, console)
    if not articles:
        console.print("[red]✗[/] Sample came back empty.")
        return 1
    console.print(f"[green]✓[/] Sampled [bold]{len(articles)}[/] articles\n")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = resolve_output_path(args.output, timestamp)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Persist each repeat as it lands. A 50x3 pilot is hours of API calls, and
    # writing only at the end meant an interruption at minute 150 lost every
    # one of them — the same all-or-nothing failure the production cache was
    # rewritten to remove.
    partial_path = out_path.with_suffix(".partial.jsonl")
    done = load_partial(partial_path, prompt_id, logger)
    if done:
        console.print(
            f"[green]✓[/] Resuming: [bold]{sum(len(r) for r in done.values())}[/] "
            f"repeat(s) already recorded in [dim]{partial_path.name}[/]\n"
        )

    results, errors = annotate_articles(
        articles,
        models,
        system_prompt,
        prompt_id,
        args.repeats,
        partial_path,
        done,
        logger,
    )
    payload = build_payload(
        timestamp=timestamp,
        articles=articles,
        results=results,
        models=models,
        seed=args.seed,
        repeats=args.repeats,
        system_prompt=system_prompt,
        prompt_id=prompt_id,
    )
    atomic_write_text(out_path, json.dumps(payload, indent=2, ensure_ascii=False))
    # Scaffolding, removed only once the real output is safely on disk.
    partial_path.unlink(missing_ok=True)

    show_summary(results, models, args.repeats, errors, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
