#!/usr/bin/env python3
"""
probe_reasoning.py
==================

Measure whether a model's reasoning levels are real *through the route you are
actually using*.

Reads from Omeka, writes nothing anywhere. It exists because a model card is a
claim about weights, not about a deployment. Gemma 4 documents two thinking
levels and honours them through the Gemini API; reached through OpenRouter,
which fans a request across third-party backends that disagree about what an
effort means, ``medium`` and ``high`` came back indistinguishable in both
latency and reasoning length — one backend reporting ``reasoning_tokens: 1``
while emitting 3.7k characters of reasoning. The panel had standardised on
``medium`` and was, in fact, getting on/off.

So before a candidate is judged on agreement, this answers a cheaper question:
does asking for more deliberation produce more deliberation here? Three levels,
several calls each, on real articles with the real prompt and the real schema.
If the numbers do not separate, the level is decorative and any write-up should
say so.

Usage
-----
    python serving/probe_reasoning.py
    python serving/probe_reasoning.py --model qwen3.8-27b-openrouter
    python serving/probe_reasoning.py --item-ids 12345,12346 --repeats 5
    python serving/probe_reasoning.py --levels low,xhigh --articles 1

Environment Variables
---------------------
OMEKA_BASE_URL / OMEKA_KEY_IDENTITY / OMEKA_KEY_CREDENTIAL   Omeka S API
SELFHOSTED_LLM_BASE_URL / SELFHOSTED_LLM_API_KEY             self-hosted route
OPENROUTER_API_KEY                                           OpenRouter route
"""
import sys
import time
import json
import argparse
import logging
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler
from rich import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.omeka_client import OmekaClient
from common.llm_provider import (
    LLMConfig,
    _extract_json_payload,
    build_llm_client,
    get_model_option,
)
from common.console_utils import standard_progress
from common.log_redaction import install_credential_redaction

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "AI_sentiment_analysis"))
from sentiment_core import (  # noqa: E402
    ANALYSABLE_LANGUAGES,
    SentimentAnalysisOutput,
    create_user_prompt,
    get_item_content,
    get_item_language,
    load_system_prompt,
)

console = Console()

#: Newspaper articles — the population the sentiment panel annotates.
ARTICLE_CLASS_ID = 36

DEFAULT_MODEL = "qwen3.8-27b-selfhosted"
DEFAULT_ARTICLES = 2
DEFAULT_REPEATS = 3


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    install_credential_redaction()
    return logging.getLogger(__name__)


def fetch_articles(
    client: OmekaClient, item_ids: Optional[List[int]], count: int
) -> List[Dict[str, Any]]:
    """Real articles to probe with, either named explicitly or taken off the top.

    No sampling machinery here: this measures a server's behaviour, not the
    corpus, so which articles they are matters only in that they are genuine
    ones of representative length in a language the panel analyses.
    """
    if item_ids:
        items = [client.get_item(item_id) for item_id in item_ids]
    else:
        items = client.list_page(1, 50, resource_class_id=ARTICLE_CLASS_ID)

    usable = [
        item for item in items
        if get_item_content(item).strip()
        and get_item_language(item) in ANALYSABLE_LANGUAGES
    ]
    if not usable:
        raise SystemExit(
            "No usable articles found — none had content in an analysable "
            f"language ({', '.join(sorted(ANALYSABLE_LANGUAGES))})."
        )
    return usable[:count]


def probe_once(client, system_prompt: str, text: str, level: str) -> Dict[str, Any]:
    """One structured call at one reasoning level, with the meters read off it.

    Uses ``structured_response()`` rather than ``generate_structured()``: the
    request has to be the one production sends — same schema, same body, same
    headers — while the response object stays reachable, because ``usage`` and
    ``reasoning_content`` are exactly what is being measured and a validated
    Pydantic model has neither.
    """
    config = LLMConfig(reasoning_effort=level).merged_over(client.config)

    started = time.monotonic()
    response = client.structured_response(
        system_prompt, create_user_prompt(text), SentimentAnalysisOutput, config
    )
    elapsed = time.monotonic() - started

    message = response.choices[0].message
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)

    # Both spellings are checked because servers disagree. vLLM 0.27.1 returns
    # `reasoning` and omits `reasoning_content` entirely (measured 2026-08-16
    # against Qwen3.8-27B with --reasoning-parser qwen3); other builds and
    # OpenRouter use `reasoning_content`. Measure the text, not the reported
    # token count — vLLM leaves `completion_tokens_details` null, so
    # `reasoning_tokens` is simply unavailable on this route.
    reasoning = (
        getattr(message, "reasoning_content", None)
        or getattr(message, "reasoning", None)
        or ""
    )

    # Validate the answer as production would, so a level that breaks structured
    # output is caught here rather than mid-pilot. `fenced` records whether the
    # answer needed the recovery path at all: a server whose guided decoding is
    # working should return bare JSON, and a rising fence rate is a signal the
    # schema is not actually constraining generation.
    valid, error, fenced = True, None, False
    raw = message.content or ""
    try:
        payload = _extract_json_payload(raw)
        fenced = payload.strip() != raw.strip()
        SentimentAnalysisOutput.model_validate_json(payload)
    except Exception as exc:  # noqa: BLE001 — reporting, not handling
        valid, error = False, f"{type(exc).__name__}: {exc}"

    return {
        "seconds": elapsed,
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None),
        "reasoning_chars": len(reasoning),
        "valid": valid,
        "fenced": fenced,
        "error": error,
    }


def summarise(level: str, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    def median_of(field: str) -> Optional[float]:
        values = [r[field] for r in runs if r.get(field) is not None]
        return statistics.median(values) if values else None

    return {
        "level": level,
        "calls": len(runs),
        "median_seconds": median_of("seconds"),
        "median_completion_tokens": median_of("completion_tokens"),
        "median_reasoning_tokens": median_of("reasoning_tokens"),
        "median_reasoning_chars": median_of("reasoning_chars"),
        "invalid": sum(1 for r in runs if not r["valid"]),
        "fenced": sum(1 for r in runs if r.get("fenced")),
        "errors": sorted({r["error"] for r in runs if r["error"]}),
    }


def render(model_key: str, model_id: str, rows: List[Dict[str, Any]]) -> None:
    table = Table(
        title=f"Reasoning depth through this route — {model_key} ({model_id})",
        box=box.ROUNDED,
    )
    table.add_column("Level", style="cyan")
    table.add_column("Calls", justify="right", style="dim")
    table.add_column("Median s", justify="right")
    table.add_column("Completion tok", justify="right")
    table.add_column("Reasoning tok", justify="right")
    table.add_column("Reasoning chars", justify="right")
    table.add_column("Fenced", justify="right")
    table.add_column("Invalid", justify="right")

    def fmt(value: Optional[float], places: int = 0) -> str:
        if value is None:
            return "—"
        return f"{value:,.{places}f}"

    for row in rows:
        table.add_row(
            row["level"],
            str(row["calls"]),
            fmt(row["median_seconds"], 1),
            fmt(row["median_completion_tokens"]),
            fmt(row["median_reasoning_tokens"]),
            fmt(row["median_reasoning_chars"]),
            str(row["fenced"]) if row["fenced"] else "-",
            str(row["invalid"]) if row["invalid"] else "-",
        )
    console.print()
    console.print(table)

    measured = [r for r in rows if r["median_reasoning_chars"] is not None]
    if len(measured) >= 2:
        lengths = [r["median_reasoning_chars"] for r in measured]
        spread = max(lengths) - min(lengths)
        floor = max(1.0, min(lengths))
        if spread < 0.25 * floor:
            console.print(
                "\n[yellow]![/] The levels barely separate: median reasoning "
                f"length varies by {spread:,.0f} characters across "
                f"{len(measured)} levels. That is the Gemma pattern — the depth "
                "is on/off through this route, not graduated. Report the depth "
                "as requested, never as measured."
            )
        else:
            console.print(
                "\n[green]✓[/] The levels separate: median reasoning length "
                f"spans {min(lengths):,.0f}–{max(lengths):,.0f} characters. "
                "Asking for more deliberation produces more here."
            )

    for row in rows:
        for error in row["errors"]:
            console.print(f"[red]✗[/] {row['level']}: {error}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure whether a model's reasoning levels differ through "
                    "the route in use. Reads from Omeka; writes nothing."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Registry key to probe (default: {DEFAULT_MODEL})")
    parser.add_argument("--levels", default=None,
                        help="Comma-separated reasoning levels "
                             "(default: everything the model declares)")
    parser.add_argument("--articles", type=int, default=DEFAULT_ARTICLES,
                        help=f"Articles to probe with (default: {DEFAULT_ARTICLES})")
    parser.add_argument("--item-ids", default=None,
                        help="Comma-separated Omeka item ids, instead of sampling")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help=f"Calls per article per level (default: {DEFAULT_REPEATS})")
    parser.add_argument("--output", default=None,
                        help="Optional JSON path for the raw measurements")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    logger = configure_logging()

    option = get_model_option(args.model)
    levels = (
        [level.strip() for level in args.levels.split(",") if level.strip()]
        if args.levels
        else list(option.supported_reasoning_efforts)
    )
    if not levels:
        raise SystemExit(
            f"{option.key} declares no reasoning levels; there is nothing to probe."
        )

    item_ids = (
        [int(value) for value in args.item_ids.split(",") if value.strip()]
        if args.item_ids else None
    )

    omeka = OmekaClient.from_env()
    articles = fetch_articles(omeka, item_ids, args.articles)
    system_prompt = load_system_prompt()

    # sdk_max_retries=0: a probe measures one attempt. A silent SDK retry would
    # be indistinguishable from a slow model and would corrupt the latency
    # column, which is half the measurement.
    client = build_llm_client(option, config=LLMConfig(sdk_max_retries=0))

    console.print(
        f"[cyan]Probing[/] {option.label} at {', '.join(levels)} — "
        f"{len(articles)} article(s) × {args.repeats} repeat(s)"
    )

    raw: Dict[str, List[Dict[str, Any]]] = {level: [] for level in levels}
    total = len(levels) * len(articles) * args.repeats
    with standard_progress(console) as progress:
        task = progress.add_task("[cyan]Calling...", total=total)
        for level in levels:
            for article in articles:
                text = get_item_content(article)
                for _ in range(args.repeats):
                    try:
                        raw[level].append(probe_once(client, system_prompt, text, level))
                    except Exception as exc:  # noqa: BLE001 — a failed call is data
                        logger.warning("%s call failed: %s", level, exc)
                        raw[level].append({
                            "seconds": None, "completion_tokens": None,
                            "reasoning_tokens": None, "reasoning_chars": None,
                            "valid": False, "fenced": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                    progress.update(task, advance=1)

    rows = [summarise(level, raw[level]) for level in levels]
    render(option.key, option.model, rows)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "model_key": option.key,
                    "model_id": option.model,
                    "item_ids": [article.get("o:id") for article in articles],
                    "repeats": args.repeats,
                    "summary": rows,
                    "raw": raw,
                },
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        console.print(f"\n[green]✓[/] Measurements written to {path}")


if __name__ == "__main__":
    main()
