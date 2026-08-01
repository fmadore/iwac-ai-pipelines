#!/usr/bin/env python3
"""
02_pilot_new_panel.py
=====================

Pilot a candidate sentiment panel on a sample of already-annotated articles.

Reads from Omeka, writes nothing to it. The point is to find out whether a new
set of models earns a place — and 30 new Omeka properties across 12,000+ items
— before any of that is created. Results land in a local JSON file alongside
the generation-1 annotations for the same articles, so
``03_pilot_report.py`` can compare them.

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
"""
import sys
import json
import random
import argparse
import logging
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
from common.console_utils import standard_progress

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sentiment_core import (  # noqa: E402
    ANALYSABLE_LANGUAGES,
    ITEM_ID_TO_SUBJECTIVITE,
    PANEL,
    PANEL_REASONING,
    PANEL_REASONING_EFFECTIVE,
    V1_PANEL,
    analyze_with_all_models,
    get_item_content,
    get_item_language,
    load_system_prompt,
    prompt_fingerprint,
)

console = Console()

# ============================================================================
# CONFIGURATION
# ============================================================================

#: Omeka resource class for newspaper articles.
ARTICLE_CLASS_ID = 36

#: iwac:geminiCentralite. Used only as an "has generation-1 annotations" probe
#: so the sample is guaranteed comparable.
V1_PROBE_PROPERTY_ID = 319

#: The candidate panel, its reasoning depth and the generation-1 property
#: prefixes all come from ``sentiment_core`` — the same definitions production
#: runs against. When this file carried its own copy, "the pilot" and "what
#: shipped" were two different things that merely looked alike.
V2_PANEL = PANEL

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
    return logging.getLogger(__name__)


# ============================================================================
# SAMPLING
# ============================================================================

def _items_page(client: OmekaClient, page: int, per_page: int = PER_PAGE) -> List[Dict[str, Any]]:
    """One page of articles that already carry generation-1 annotations."""
    url = (
        f"{client.base_url}/items"
        f"?resource_class_id={ARTICLE_CLASS_ID}"
        f"&property%5B0%5D%5Bproperty%5D={V1_PROBE_PROPERTY_ID}"
        f"&property%5B0%5D%5Btype%5D=ex"
        f"&per_page={per_page}&page={page}"
    )
    result = client.get_resource(url)
    return result if isinstance(result, list) else []


def _last_page(client: OmekaClient, console: Console) -> int:
    """Binary-search the final page rather than paging the whole corpus."""
    lo, hi = 1, 2
    while _items_page(client, hi, per_page=1):
        lo, hi = hi, hi * 2
        if hi > 1_000_000:  # pathological guard
            break
    # invariant: page `lo` has data, page `hi` does not
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _items_page(client, mid, per_page=1):
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
    with console.status("[bold green]Locating corpus bounds...", spinner="dots"):
        total_items = _last_page(client, console)
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
                it for it in _items_page(client, page)
                if get_item_content(it).strip()
                and get_item_language(it) in ANALYSABLE_LANGUAGES
            ]
            if items:
                take = min(ITEMS_PER_SAMPLED_PAGE, len(items))
                sampled.extend(rng.sample(items, take))
            progress.update(task, advance=1)

    rng.shuffle(sampled)
    return sampled[:sample_size]


# ============================================================================
# GENERATION-1 READBACK
# ============================================================================

def _resource_label(values: List[Dict[str, Any]]) -> Optional[str]:
    """Label of a resource:item value (centralité / polarité)."""
    if not values:
        return None
    return values[0].get("display_title")


def _resource_item_id(values: List[Dict[str, Any]]) -> Optional[int]:
    if not values:
        return None
    return values[0].get("value_resource_id")


def _literal(values: List[Dict[str, Any]]) -> Optional[str]:
    if not values:
        return None
    return values[0].get("@value")


def read_v1_annotations(item: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Read the generation-1 annotations already stored on an Omeka item."""
    out: Dict[str, Dict[str, Any]] = {}
    for prefix, prop in V1_PANEL.items():
        subj_item_id = _resource_item_id(item.get(f"{prop}SubjectiviteScore", []))
        out[prefix] = {
            "centralite_islam_musulmans": _resource_label(item.get(f"{prop}Centralite", [])),
            "centralite_justification": _literal(item.get(f"{prop}CentraliteJustification", [])),
            "polarite": _resource_label(item.get(f"{prop}Polarite", [])),
            "polarite_justification": _literal(item.get(f"{prop}PolariteJustification", [])),
            "subjectivite_score": ITEM_ID_TO_SUBJECTIVITE.get(subj_item_id),
            "subjectivite_justification": _literal(item.get(f"{prop}SubjectiviteJustification", [])),
        }
    return out


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


def build_clients(
    selected: List[str], logger: logging.Logger
) -> tuple[Dict[str, BaseLLMClient], Dict[str, str], List[tuple]]:
    """Build one client per requested model; report the ones we cannot reach."""
    clients: Dict[str, BaseLLMClient] = {}
    labels: Dict[str, str] = {}
    model_ids: Dict[str, str] = {}
    skipped: List[tuple] = []

    # Reasoning depth IS standardised (see PANEL_REASONING); temperature is
    # not. Temperature stays vendor-owned — the recommended values differ
    # (1.0 DeepSeek, 0.7 Qwen, 0.3 Mistral Small 4, unset Gemini) and both
    # Google and Alibaba document a lowered temperature as a cause of looping.
    # Each client takes its own from MODEL_REGISTRY.
    config = LLMConfig(**PANEL_REASONING)
    for prefix in selected:
        member = V2_PANEL[prefix]
        try:
            option = get_model_option(member.registry_key)
            clients[prefix] = build_llm_client(option, config=config)
            labels[prefix] = member.label
            model_ids[prefix] = option.model
        except (RuntimeError, ValueError) as exc:
            skipped.append((prefix, member.label, str(exc)))
    return clients, labels, (skipped, model_ids)


def main() -> int:
    logger = configure_logging()

    console.print(Panel.fit(
        "[bold cyan]Sentiment Panel Pilot[/bold cyan]\n"
        "[dim]Candidate models vs the generation-1 annotations — "
        "reads Omeka, writes nothing to it[/dim]",
        border_style="cyan",
    ))

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
    parser.add_argument("--without-examples", action="store_true",
                        help="Drop the prompt's worked examples, keeping the "
                             "definitions and boundary rules. The second arm of "
                             "the examples A/B: run both on the same --seed and "
                             "compare with 03_pilot_report.py")
    args = parser.parse_args()

    selected = list(V2_PANEL)
    if args.models:
        selected = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in selected if m not in V2_PANEL]
        if unknown:
            console.print(f"[red]✗[/] Unknown model(s): {', '.join(unknown)}")
            console.print(f"[dim]Available: {', '.join(V2_PANEL)}[/]")
            return 2

    try:
        client = OmekaClient.from_env()
    except ValueError as exc:
        console.print(f"[red]✗[/] {exc}")
        return 2

    clients, labels, (skipped, model_ids) = build_clients(selected, logger)

    if skipped:
        console.print()
        for _prefix, label, reason in skipped:
            console.print(f"[yellow]![/] Skipping [bold]{label}[/] — {reason}")
        console.print(
            "[dim]  Qwen and DeepSeek both need OPENROUTER_API_KEY; the pilot "
            "continues with whatever is reachable.[/]"
        )

    if not clients:
        console.print("\n[red]✗[/] No models available — nothing to pilot.")
        return 1

    system_prompt = load_system_prompt(include_examples=not args.without_examples)
    prompt_id = prompt_fingerprint(system_prompt)

    config_table = Table(title="Pilot configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Omeka URL", client.base_url)
    config_table.add_row("Sample size", str(args.sample_size))
    config_table.add_row("Seed", str(args.seed))
    config_table.add_row("Repeats", str(args.repeats))
    config_table.add_row("Models", "\n".join(f"{labels[p]}  [dim]{model_ids[p]}[/]" for p in clients))
    config_table.add_row(
        "Prompt",
        f"#{prompt_id} · {len(system_prompt):,} chars · "
        + ("with examples" if not args.without_examples
           else "[yellow]examples stripped[/]"),
    )
    config_table.add_row("Writes to Omeka", "[bold green]none[/]")
    console.print(config_table)
    console.print()

    articles = sample_articles(client, args.sample_size, args.seed, console)
    if not articles:
        console.print("[red]✗[/] Sample came back empty.")
        return 1
    console.print(f"[green]✓[/] Sampled [bold]{len(articles)}[/] articles\n")

    # Resolved before annotating, not after: the partial file lives beside the
    # output, so the output path has to exist before the first call is made.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = (
        Path(args.output) if args.output
        else Path(__file__).resolve().parent / OUTPUT_DIR_NAME / f"pilot_{timestamp}.json"
    )
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

    results: Dict[str, Any] = {}
    errors = 0

    console.rule("[bold cyan]Annotating")
    total_calls = len(articles) * args.repeats
    with open(partial_path, "a", encoding="utf-8") as partial, \
            standard_progress(console, show_eta=True) as progress:
        task = progress.add_task("[cyan]Running panel...", total=total_calls)
        for item in articles:
            item_id = str(item.get("o:id"))
            content = get_item_content(item)
            cached = done.get(item_id, {})
            runs = []
            for run_index in range(args.repeats):
                run = cached.get(run_index)
                if run is None:
                    run = analyze_with_all_models(
                        content, clients, system_prompt, logger, labels
                    )
                    partial.write(json.dumps({
                        "prompt": prompt_id,
                        "item_id": item_id,
                        "run": run_index,
                        "result": run,
                    }, ensure_ascii=False) + "\n")
                    partial.flush()
                errors += sum(1 for r in run.values() if r.get("analysis_error"))
                runs.append(run)
                progress.update(task, advance=1)

            results[item_id] = {
                "title": item.get("o:title"),
                "n_chars": len(content),
                "v1": read_v1_annotations(item),
                "v2_runs": runs,
            }

    payload = {
        "manifest": {
            "generated_utc": timestamp,
            "sample_size": len(articles),
            "seed": args.seed,
            "repeats": args.repeats,
            "sampling": "two-stage cluster: random pages, random items within page",
            "v2_models": {p: {"label": labels[p], "model_id": model_ids[p]} for p in clients},
            "v2_skipped": [{"prefix": p, "label": lbl, "reason": r} for p, lbl, r in skipped],
            # Requested depth, plus what each model actually accepted — they
            # differ for Mistral, and a run record that hid that would be wrong.
            "v2_reasoning_requested": dict(PANEL_REASONING),
            "v2_reasoning_effective": {
                p: PANEL_REASONING_EFFECTIVE[p] for p in clients
            },
            # Generation-1 models and the config they actually ran with,
            # recovered from commit 07fb007 (2026-01-27).
            "v1_models": {
                "gemini_3_flash_preview": "gemini-3-flash-preview",
                "gpt_5_mini": "gpt-5-mini",
                "ministral_14b_2512": "ministral-14b-2512",
            },
            "v1_run_config": {
                "gemini_3_flash_preview": "temperature=0.2; response_schema; no thinking_level",
                "gpt_5_mini": "response_format schema only; no temperature; no reasoning_effort",
                "ministral_14b_2512": "temperature=0.2; max_tokens=512; response_format schema",
            },
            "prompt_chars": len(system_prompt),
            # Prompt wording moves label distributions in ways a diff does not
            # predict, so a pilot that cannot name its prompt cannot be
            # compared with the run it was meant to validate.
            "prompt_fingerprint": prompt_fingerprint(system_prompt),
            "prompt_examples": not args.without_examples,
        },
        "articles": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    # Scaffolding, removed only once the real output is safely on disk.
    partial_path.unlink(missing_ok=True)

    summary = Table(title="Pilot summary", box=box.ROUNDED)
    summary.add_column("Metric", style="dim")
    summary.add_column("Value", justify="right")
    summary.add_row("Articles annotated", str(len(results)))
    summary.add_row("Models run", str(len(clients)))
    summary.add_row("Models skipped", f"[yellow]{len(skipped)}[/]" if skipped else "[dim]0[/]")
    summary.add_row("Total annotations", str(total_calls * len(clients)))
    summary.add_row("Annotation errors", f"[red]{errors}[/]" if errors else "[dim]0[/]")
    console.print()
    console.print(summary)

    console.print()
    console.print(Panel.fit(
        f"[bold green]✓ Pilot complete[/bold green]\n\n"
        f"Results: [cyan]{out_path}[/]\n"
        f"[dim]Nothing was written to Omeka. Next: 03_pilot_report.py {out_path.name}[/]",
        title="Done",
        border_style="green",
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
