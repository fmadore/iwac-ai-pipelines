#!/usr/bin/env python3
"""
03_pilot_report.py
==================

Read a pilot JSON from ``02_pilot_new_panel.py`` and answer the question the
pilot exists to answer: does a candidate model earn a slot?

Three things are reported, because a single agreement number cannot separate
them:

1. **Agreement with the rest of the panel** — each model against the majority of
   the *others* (leave-one-out). Scoring a model against a consensus it votes in
   inflates it structurally, which is why the yardstick excludes it.
2. **Agreement within the panel** — pairwise Cohen's kappa. Two models that agree
   only because both are wrong look identical to two that are right, so this is
   read alongside (1), not instead of it.
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
from common.log_redaction import install_credential_redaction
from sentiment_core import (  # noqa: E402
    CENTRALITE_ORDER,
    POLARITE_ORDER,
    SUBJECTIVITE_ORDER,
)

# Credentials ride in Omeka query strings and provider headers; keep them
# out of anything urllib3 or an SDK decides to log.
install_credential_redaction()

console = Console()

PILOT_DIR = Path(__file__).resolve().parent / "cache" / "pilot"

#: (dimension key, result field, label→ordinal map or None when already numeric)
DIMENSIONS: List[Tuple[str, str, Optional[Dict[str, int]]]] = [
    ("polarité", "polarite", POLARITE_ORDER),
    ("centralité", "centralite_islam_musulmans", CENTRALITE_ORDER),
    # Ordinal since 2026-07-31: the models now return the label rather than the
    # 1-5 integer, and generation-1 links are read back as labels too, so the
    # two generations compare on one scale.
    ("subjectivité", "subjectivite_score", SUBJECTIVITE_ORDER),
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

    Returns ``(values, consensus, loo)`` where ``loo[m]`` is the leave-one-out
    consensus for model *m* — the majority of the OTHER models. Scoring a model
    against the full consensus scores it against something it is a voting member
    of, which inflates it structurally; the leave-one-out figure is the honest
    comparison, and is what the κ values in the README were computed with.

    Pilots run before 2026-08-07 also carry a ``v1`` block per article, read
    from the generation-1 Omeka properties that no longer exist. It is ignored:
    reporting it would mix a live measurement with a frozen one.
    """
    articles = payload["articles"]
    models = list(payload["manifest"]["v2_models"])

    values: Dict[str, List[Any]] = {m: [] for m in models}
    consensus: List[Any] = []
    loo: Dict[str, List[Any]] = {m: [] for m in models}

    for article in articles.values():
        first_run = article["v2_runs"][0]
        for m in models:
            values[m].append(_value(first_run.get(m, {}), field))
        row = {m: values[m][-1] for m in models}
        consensus.append(majority(list(row.values())))
        for m in models:
            loo[m].append(majority([v for k, v in row.items() if k != m]))

    return values, consensus, loo


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

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report agreement for a sentiment panel pilot. Read-only."
    )
    parser.add_argument("pilot_file", nargs="?", default=None,
                        help="Pilot JSON (default: newest in cache/pilot/)")
    return parser


def resolve_pilot_path(pilot_file: Optional[str]) -> Optional[Path]:
    """Resolve an explicit pilot path or select the newest default output."""
    if pilot_file:
        path = Path(pilot_file)
        if not path.exists() and (PILOT_DIR / pilot_file).exists():
            return PILOT_DIR / pilot_file
        return path
    candidates = sorted(PILOT_DIR.glob("pilot_*.json"))
    return candidates[-1] if candidates else None


def show_models(manifest: Dict[str, Any]) -> None:
    """Display models that ran and candidates skipped during the pilot."""
    models_table = Table(title="Models", box=box.ROUNDED)
    models_table.add_column("Generation", style="dim")
    models_table.add_column("Prefix")
    models_table.add_column("Model id", style="green")
    for prefix, info in manifest["v2_models"].items():
        models_table.add_row("v2 (candidate)", prefix, info["model_id"])
    for skip in manifest.get("v2_skipped", []):
        models_table.add_row("v2 (skipped)", skip["prefix"], f"[yellow]{skip['reason']}[/]")
    console.print(models_table)


def _paired_count(values: List[Any], target: List[Any]) -> int:
    return sum(
        1 for value, expected in zip(values, target, strict=True)
        if value is not None and expected is not None
    )


def consensus_table(
    dim_name: str,
    values: Dict[str, List[Any]],
    loo: Dict[str, List[Any]],
) -> Table:
    """Each model against the majority of the others."""
    table = Table(
        title=f"{dim_name}: agreement with the panel (leave-one-out)",
        box=box.ROUNDED,
    )
    table.add_column("Model")
    table.add_column("Exact", justify="right")
    table.add_column("Kappa", justify="right")
    table.add_column("n", justify="right", style="dim")
    for model, series in values.items():
        target = loo[model]
        table.add_row(
            model,
            fmt(exact_agreement(series, target)),
            fmt(cohen_kappa(series, target)),
            str(_paired_count(series, target)),
        )
    return table


def pairwise_table(dim_name: str, values: Dict[str, List[Any]]) -> Optional[Table]:
    """Build within-panel pairwise agreement metrics when a pair exists."""
    if len(values) < 2:
        return None
    table = Table(title=f"{dim_name}: pairwise within candidate panel", box=box.ROUNDED)
    table.add_column("Pair")
    table.add_column("Exact", justify="right")
    table.add_column("Kappa", justify="right")
    for first, second in combinations(values, 2):
        table.add_row(
            f"{first} ↔ {second}",
            fmt(exact_agreement(values[first], values[second])),
            fmt(cohen_kappa(values[first], values[second])),
        )
    return table


def consistency_table(
    payload: Dict[str, Any],
    dim_name: str,
    field: str,
    repeats: int,
) -> Optional[Table]:
    """Build repeat stability metrics when the pilot contains repeats."""
    if repeats < 2:
        return None
    table = Table(
        title=f"{dim_name}: self-consistency across {repeats} repeats",
        box=box.ROUNDED,
    )
    table.add_column("Model")
    table.add_column("Identical every run", justify="right")
    for model, value in self_consistency(payload, field).items():
        table.add_row(model, fmt(value))
    return table


def show_dimension(payload: Dict[str, Any], dim_name: str, field: str, repeats: int) -> None:
    """Render all agreement views for one sentiment dimension."""
    values, _consensus, loo = collect(payload, field)
    console.print()
    console.rule(f"[bold]{dim_name}")
    console.print(consensus_table(dim_name, values, loo))
    for table in (
        pairwise_table(dim_name, values),
        consistency_table(payload, dim_name, field, repeats),
    ):
        if table is not None:
            console.print(table)


def show_caveats(manifest: Dict[str, Any], repeats: int) -> None:
    """Explain missing repeat or provider evidence after the metrics."""
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
        missing = ", ".join(skip["label"] for skip in manifest["v2_skipped"])
        console.print(Panel.fit(
            f"[yellow]Incomplete panel:[/] {missing} did not run.\n"
            "[dim]Set OPENROUTER_API_KEY in .env and re-run the pilot to include them.[/]",
            border_style="yellow",
        ))


def main() -> int:
    args = build_argument_parser().parse_args()
    path = resolve_pilot_path(args.pilot_file)

    if path is None or not path.exists():
        location = path if path is not None else PILOT_DIR
        console.print(f"[red]✗[/] No pilot file found at {location}")
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

    show_models(manifest)

    for dim_name, field, _order in DIMENSIONS:
        show_dimension(payload, dim_name, field, repeats)

    show_caveats(manifest, repeats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
