#!/usr/bin/env python3
"""
annotate_offline.py
===================

Annotate a prepared corpus with a self-hosted model, with no credentials and no
network access beyond the local server.

This is the batch counterpart to driving a self-hosted endpoint interactively.
The interactive route needs a tunnel held open for the whole run, which suits a
laptop that stays awake; a cluster job that starts at midnight does not. So the
work is split where the secrets are:

* **On your machine**, where the Omeka keys already live, sample the corpus and
  write a plain JSON file of article ids and text.
* **On the cluster**, this script reads that file, annotates it against
  ``localhost``, and writes JSONL. It never sees a key, never contacts Omeka,
  and never reaches the internet — so nothing has to be stored on shared
  university storage to make an unattended run possible.
* **Afterwards**, copy the JSONL back and merge it wherever it belongs.

Results are appended one line at a time and completed items are skipped on
restart, because the failure mode this is built for is the job hitting its
walltime mid-corpus. A partial run is data; a lost run is not.

Input JSON: ``{"system_prompt": str, "prompt_fingerprint": str,
"articles": [{"item_id": int, "language": str, "content": str}]}``

Output JSONL, one object per annotation: ``{"item_id", "prompt", "model",
"reasoning_effort", "seconds", "result": {...}}``

One object per *annotation*, not per article: a failed item is retried on the
next run and appends a second record, so an item id can appear more than once.
When merging, key on ``item_id`` and keep the last record without an
``analysis_error`` — earlier failures are kept on purpose, because the failure
rate is itself a finding about the model.

Usage
-----
    python serving/annotate_offline.py --input pilot_input.json --output out.jsonl
    python serving/annotate_offline.py --input in.json --output out.jsonl --limit 20

Environment Variables
---------------------
SELFHOSTED_LLM_BASE_URL   e.g. http://127.0.0.1:8000/v1 (required)
SELFHOSTED_LLM_API_KEY    only if the server was started with --api-key
"""
import sys
import json
import time
import argparse
import logging
import threading
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm_provider import LLMConfig, build_llm_client, get_model_option

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "AI_sentiment_analysis"))
from sentiment_core import (  # noqa: E402
    analyze_with_model,
    panel_reasoning,
    request_timeout_for_budget,
)

DEFAULT_MODEL = "qwen3.8-27b-selfhosted"
#: Articles in flight at once. A self-hosted server is not someone else's rate
#: limit — the ceiling is the GPU's own batch scheduler, and vLLM is built to
#: keep many sequences resident. Serial annotation leaves most of that idle: at
#: one article at a time a 12,300-article corpus would take weeks. Matches the
#: main pipeline's ``--concurrency`` default; raise it if the server reports
#: spare KV-cache capacity, lower it if requests start queueing.
DEFAULT_CONCURRENCY = 6
#: Panel member key whose reasoning depth this run adopts. The candidate is
#: annotated exactly as it would be in production rather than at the registry
#: default, so the output is comparable with a normal pilot.
PANEL_MEMBER_KEY = "qwen3_8_27b"


def configure_logging() -> logging.Logger:
    # Plain stdout: this runs unattended into a Slurm log, where rich's progress
    # rendering would arrive as a wall of escape codes.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger("annotate")


def load_done(path: Path, prompt_id: str, effort: Optional[str],
              logger: logging.Logger) -> set:
    """Item ids already annotated *successfully*, by this instrument.

    Four things are deliberately not counted as done:

    * a record written under a different prompt — resuming across a prompt edit
      would silently mix two instruments in one output file;
    * a record produced at a different reasoning depth, for exactly the same
      reason. Depth changes the answers, so ``medium`` and ``low`` results in
      one file are two instruments wearing one name;
    * a torn final line, the expected shape of a job killed at its walltime;
    * **a record carrying an ``analysis_error``.** Failures are not results, so
      re-running retries exactly those, which is the same rule the main pipeline
      follows for its cache. Treating them as done would freeze a transient
      timeout into a permanent hole in the corpus, and the hole would be
      invisible: the item id is present, just useless.
    """
    done, stale, torn, errored, other_depth = set(), 0, 0, 0, 0
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            torn += 1
            continue
        if record.get("prompt") != prompt_id:
            stale += 1
            continue
        if record.get("reasoning_effort") != effort:
            other_depth += 1
            continue
        if (record.get("result") or {}).get("analysis_error"):
            errored += 1
            continue
        if record.get("item_id") is not None:
            done.add(record["item_id"])
    if stale or torn or errored or other_depth:
        logger.info(
            "resume: %d from another prompt, %d at another depth, %d torn, "
            "%d failed — failed items will be retried",
            stale, other_depth, torn, errored,
        )
    return done


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate a prepared corpus against a self-hosted endpoint. "
                    "Reads and writes local files only; contacts no API but the "
                    "local server."
    )
    parser.add_argument("--input", required=True, help="Prepared corpus JSON")
    parser.add_argument("--output", required=True, help="JSONL to append to")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Registry key (default: {DEFAULT_MODEL})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N articles (default: all)")
    parser.add_argument("--reasoning-effort", default=None,
                        help="Override the panel's reasoning depth (e.g. low). "
                             "Annotations produced at a different depth are a "
                             "different instrument and are NOT comparable with "
                             "the panel — use a separate output file")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"Articles in flight at once (default: {DEFAULT_CONCURRENCY}). "
                             "A serial run wastes most of the GPU: vLLM batches "
                             "concurrent requests, so this is what makes a "
                             "full-corpus pass finish in days rather than weeks")
    parser.add_argument("--timeout-budget", type=float, default=600.0,
                        help="Seconds per article across all retries (default: 600)")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    logger = configure_logging()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    system_prompt = payload["system_prompt"]
    prompt_id = payload.get("prompt_fingerprint", "unknown")
    articles = payload["articles"]
    if args.limit:
        articles = articles[:args.limit]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    option = get_model_option(args.model)
    reasoning = panel_reasoning(PANEL_MEMBER_KEY)
    if args.reasoning_effort:
        reasoning = {**reasoning, "reasoning_effort": args.reasoning_effort}
        logger.warning(
            "reasoning depth overridden to %r (panel uses %r) — these "
            "annotations are NOT comparable with panel results, and records at "
            "another depth in this file will be re-annotated",
            args.reasoning_effort,
            panel_reasoning(PANEL_MEMBER_KEY).get("reasoning_effort"),
        )
    # Resume is keyed on the instrument, so it has to know the depth first.
    done = load_done(output, prompt_id, reasoning.get("reasoning_effort"), logger)

    # One attempt must fit inside the per-article budget with room for retries;
    # xhigh calls were measured past 300s on L40s, so this is not theoretical.
    client = build_llm_client(option, config=LLMConfig(
        **reasoning,
        request_timeout_seconds=request_timeout_for_budget(args.timeout_budget),
        sdk_max_retries=0,
    ))

    logger.info("model=%s (%s)", option.key, option.model)
    logger.info("reasoning=%s temperature=%s", reasoning, option.default_temperature)
    logger.info("prompt=%s articles=%d already done=%d",
                prompt_id, len(articles), len(done))

    pending = [a for a in articles if a["item_id"] not in done]
    logger.info("annotating %d article(s) at concurrency %d",
                len(pending), args.concurrency)

    started = time.monotonic()
    counters = {"written": 0, "failed": 0}
    # One writer lock, not one file per worker: the resume path reads a single
    # JSONL, and a line is only ever appended whole.
    write_lock = threading.Lock()

    def annotate(article: Dict[str, Any]) -> None:
        """Annotate ONE article, independently of every other.

        The independence is the load-bearing property here, not an
        implementation detail — concurrency is only sound because of it, and it
        is easy to destroy later without any test noticing:

        * every call sends ``[system_prompt, this article]`` and nothing else.
          There is no conversation history, no running context, no batching of
          several articles into one prompt;
        * nothing mutable is shared between workers. ``create_user_prompt`` is a
          pure function of the article text, ``analyze_with_model`` builds its
          messages and its retry state per call, and the client's config is
          read-only after construction (``_get_effective_config`` merges into a
          copy);
        * the lock below guards the output *file*, never a model call — writes
          are serialised so a line lands whole, and that is all it does.

        Reordering, resuming, or re-running a subset therefore cannot change any
        other article's answer. Do not add a shared cache keyed on anything but
        the item id, and do not "optimise" by putting several articles in one
        request: the annotations are research data whose independence is what
        makes per-item agreement statistics mean anything.
        """
        call_start = time.monotonic()
        result: Dict[str, Any] = analyze_with_model(
            client, article["content"], system_prompt, option.label, logger,
        )
        elapsed = time.monotonic() - call_start
        record = json.dumps({
            "item_id": article["item_id"],
            "prompt": prompt_id,
            "model": option.model,
            "model_key": option.key,
            "reasoning_effort": reasoning.get("reasoning_effort"),
            "language": article.get("language"),
            "seconds": round(elapsed, 2),
            "result": result,
        }, ensure_ascii=False)

        with write_lock:
            handle.write(record + "\n")
            handle.flush()  # a killed job must not lose the last few records
            counters["written"] += 1
            if result.get("analysis_error"):
                counters["failed"] += 1
            written = counters["written"]
            if written % 10 == 0 or written == len(pending):
                spent = time.monotonic() - started
                rate = spent / written
                logger.info(
                    "%d/%d done (%d failed) — %.1fs/article wall, ~%.0f min left",
                    written, len(pending), counters["failed"], rate,
                    (len(pending) - written) * rate / 60,
                )

    with output.open("a", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(annotate, article) for article in pending]
            for future in concurrent.futures.as_completed(futures):
                # analyze_with_model swallows transient errors into a result;
                # anything reaching here is a bug or a quota stop, and must be
                # visible rather than silently reducing the corpus.
                future.result()

    total_min = (time.monotonic() - started) / 60
    logger.info("finished: %d written, %d failed, %.1f min total (%.1f articles/hour)",
                counters["written"], counters["failed"], total_min,
                counters["written"] / max(total_min / 60, 1e-9))


if __name__ == "__main__":
    main()
