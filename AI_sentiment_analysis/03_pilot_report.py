#!/usr/bin/env python3
"""
03_pilot_report.py
==================

Read a pilot JSON from ``02_pilot_new_panel.py`` and answer the question the
pilot exists to answer: does a candidate model earn a slot?

Three things are reported, because a single agreement number cannot separate
them:

1. **Agreement with the generation-1 consensus** — does the candidate broadly
   reproduce the existing annotation, or is it an outlier?
2. **Agreement within the candidate panel** — pairwise Cohen's kappa. Two models
   that agree only because both are wrong look identical to two that are right,
   so this is read alongside (1), not instead of it.
3. **Self-consistency** (needs ``--repeats`` > 1 in the pilot) — how often a
   model gives the same answer to the same article twice. DeepSeek V4 runs at
   the vendor-recommended temperature 1.0 and Qwen3.7 at 0.7, so without this a
   low agreement score is ambiguous between "disagrees" and "is noisy".

Reads a local file and prints. Writes nothing anywhere.

Usage
-----
    python AI_sentiment_analysis/03_pilot_report.py
    python AI_sentiment_analysis/03_pilot_report.py cache/pilot/pilot_<ts>.json
"""
import sys
import json
import argparse
from pathlib import Path
from itertools import combinations
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sentiment_core import CENTRALITE_ORDER, POLARITE_ORDER  # noqa: E402

console = Console()

PILOT_DIR = Path(__file__).resolve().parent / "cache" / "pilot"

#: (dimension key, result field, label→ordinal map or None when already numeric)
DIMENSIONS: List[Tuple[str, str, Optional[Dict[str, int]]]] = [
    ("polarité", "polarite", POLARITE_ORDER),
    ("centralité", "centralite_islam_musulmans", CENTRALITE_ORDER),
    ("subjectivité", "subjectivite_score", None),
]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def cohen_kappa(a: List[Any], b: List[Any]) -> Optional[float]:
    """Unweighted Cohen's kappa over paired, non-null labels."""
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    n = len(pairs)
    observed = sum(1 for x, y in pairs if x == y) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    expected = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if expected >= 1.0:
        # Every rating identical on both sides: perfect but undefined kappa.
        return None
    return (observed - expected) / (1 - expected)


def exact_agreement(a: List[Any], b: List[Any]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if not pairs:
        return None
    return sum(1 for x, y in pairs if x == y) / len(pairs)


def majority(votes: List[Any]) -> Optional[Any]:
    """Strict-majority label among the models that voted (min 2 voters)."""
    votes = [v for v in votes if v is not None]
    if len(votes) < 2:
        return None
    label, count = Counter(votes).most_common(1)[0]
    return label if count > len(votes) / 2 else None


def fmt(value: Optional[float], width: int = 6) -> str:
    if value is None:
        return "[dim]—[/]".rjust(width + 10)
    colour = "green" if value >= 0.6 else "yellow" if value >= 0.4 else "red"
    return f"[{colour}]{value:.3f}[/]"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _value(result: Dict[str, Any], field: str) -> Optional[Any]:
    """One dimension's value, or None when the model errored on that article."""
    if not result or result.get("analysis_error"):
        return None
    value = result.get(field)
    if value in ("ERREUR_ANALYSE", ""):
        return None
    return value


def collect(payload: Dict[str, Any], field: str):
    """Per-model value lists for one dimension, aligned across articles.

    Returns ``(v1, v2, v1_consensus, v1_loo)`` where ``v1_loo[m]`` is the
    leave-one-out consensus for v1 model *m* — the majority of the OTHER v1
    models. A v1 model scored against the full v1 consensus is scored against
    something it is a voting member of (2 of 3), which inflates it structurally
    and makes it useless as a yardstick for candidates that had no vote. The
    leave-one-out figure is the honest comparison.
    """
    articles = payload["articles"]
    v1_models = list(payload["manifest"]["v1_models"])
    v2_models = list(payload["manifest"]["v2_models"])

    v1: Dict[str, List[Any]] = {m: [] for m in v1_models}
    v2: Dict[str, List[Any]] = {m: [] for m in v2_models}
    v1_consensus: List[Any] = []
    v1_loo: Dict[str, List[Any]] = {m: [] for m in v1_models}

    for article in articles.values():
        for m in v1_models:
            v1[m].append(_value(article["v1"].get(m, {}), field))
        first_run = article["v2_runs"][0]
        for m in v2_models:
            v2[m].append(_value(first_run.get(m, {}), field))
        row = {m: v1[m][-1] for m in v1_models}
        v1_consensus.append(majority(list(row.values())))
        for m in v1_models:
            v1_loo[m].append(majority([v for k, v in row.items() if k != m]))

    return v1, v2, v1_consensus, v1_loo


def self_consistency(payload: Dict[str, Any], field: str) -> Dict[str, Optional[float]]:
    """Share of articles where every repeat produced the same value."""
    v2_models = list(payload["manifest"]["v2_models"])
    out: Dict[str, Optional[float]] = {}
    for m in v2_models:
        stable = total = 0
        for article in payload["articles"].values():
            values = [_value(run.get(m, {}), field) for run in article["v2_runs"]]
            values = [v for v in values if v is not None]
            if len(values) < 2:
                continue
            total += 1
            stable += 1 if len(set(values)) == 1 else 0
        out[m] = (stable / total) if total else None
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report agreement for a sentiment panel pilot. Read-only."
    )
    parser.add_argument("pilot_file", nargs="?", default=None,
                        help="Pilot JSON (default: newest in cache/pilot/)")
    args = parser.parse_args()

    if args.pilot_file:
        path = Path(args.pilot_file)
        if not path.exists() and (PILOT_DIR / args.pilot_file).exists():
            path = PILOT_DIR / args.pilot_file
    else:
        candidates = sorted(PILOT_DIR.glob("pilot_*.json"))
        if not candidates:
            console.print(f"[red]✗[/] No pilot files in {PILOT_DIR}")
            return 2
        path = candidates[-1]

    if not path.exists():
        console.print(f"[red]✗[/] Not found: {path}")
        return 2

    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = payload["manifest"]
    n_articles = len(payload["articles"])
    repeats = manifest.get("repeats", 1)

    console.print(Panel.fit(
        "[bold cyan]Sentiment Panel Pilot — Report[/bold cyan]\n"
        f"[dim]{path.name} · {n_articles} articles · seed {manifest.get('seed')} · "
        f"{repeats} repeat(s)[/dim]",
        border_style="cyan",
    ))

    models_table = Table(title="Models", box=box.ROUNDED)
    models_table.add_column("Generation", style="dim")
    models_table.add_column("Prefix")
    models_table.add_column("Model id", style="green")
    for prefix, model_id in manifest["v1_models"].items():
        models_table.add_row("v1 (Jan–Feb 2026)", prefix, model_id)
    for prefix, info in manifest["v2_models"].items():
        models_table.add_row("v2 (candidate)", prefix, info["model_id"])
    for skip in manifest.get("v2_skipped", []):
        models_table.add_row("v2 (skipped)", skip["prefix"], f"[yellow]{skip['reason']}[/]")
    console.print(models_table)

    for dim_name, field, _order in DIMENSIONS:
        v1, v2, v1_consensus, v1_loo = collect(payload, field)
        console.print()
        console.rule(f"[bold]{dim_name}")

        # 1. candidate vs the v1 consensus, with a leave-one-out v1 baseline
        t1 = Table(title=f"{dim_name}: agreement with the generation-1 consensus", box=box.ROUNDED)
        t1.add_column("Model")
        t1.add_column("Exact", justify="right")
        t1.add_column("Kappa", justify="right")
        t1.add_column("n", justify="right", style="dim")
        for m, values in v2.items():
            n = sum(1 for x, y in zip(values, v1_consensus, strict=True) if x is not None and y is not None)
            t1.add_row(m, fmt(exact_agreement(values, v1_consensus)),
                       fmt(cohen_kappa(values, v1_consensus)), str(n))
        # v1 members scored leave-one-out: against the majority of the OTHER
        # two, so they face the same outsider's task as a candidate.
        for m, values in v1.items():
            target = v1_loo[m]
            n = sum(1 for x, y in zip(values, target, strict=True) if x is not None and y is not None)
            t1.add_row(f"[dim]{m} (v1, leave-one-out)[/]", fmt(exact_agreement(values, target)),
                       fmt(cohen_kappa(values, target)), str(n))
        console.print(t1)

        # 2. pairwise within the candidate panel
        if len(v2) > 1:
            t2 = Table(title=f"{dim_name}: pairwise within candidate panel", box=box.ROUNDED)
            t2.add_column("Pair")
            t2.add_column("Exact", justify="right")
            t2.add_column("Kappa", justify="right")
            for m1, m2 in combinations(v2, 2):
                t2.add_row(f"{m1} ↔ {m2}", fmt(exact_agreement(v2[m1], v2[m2])),
                           fmt(cohen_kappa(v2[m1], v2[m2])))
            console.print(t2)

        # 3. self-consistency
        if repeats > 1:
            sc = self_consistency(payload, field)
            t3 = Table(title=f"{dim_name}: self-consistency across {repeats} repeats", box=box.ROUNDED)
            t3.add_column("Model")
            t3.add_column("Identical every run", justify="right")
            for m, value in sc.items():
                t3.add_row(m, fmt(value))
            console.print(t3)

    console.print()
    if repeats < 2:
        console.print(Panel.fit(
            "[yellow]Self-consistency not measured.[/] The pilot ran with "
            "--repeats 1, so a low agreement score cannot be separated from "
            "sampling noise — which matters most for DeepSeek (temperature 1.0) "
            "and Qwen (0.7).\n"
            "[dim]Re-run: 02_pilot_new_panel.py --repeats 3 --sample-size 50[/]",
            border_style="yellow",
        ))
    if manifest.get("v2_skipped"):
        missing = ", ".join(s["label"] for s in manifest["v2_skipped"])
        console.print(Panel.fit(
            f"[yellow]Incomplete panel:[/] {missing} did not run.\n"
            "[dim]Set OPENROUTER_API_KEY in .env and re-run the pilot to include them.[/]",
            border_style="yellow",
        ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
