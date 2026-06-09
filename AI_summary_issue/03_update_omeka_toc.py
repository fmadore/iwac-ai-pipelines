"""
Update Omeka S items with the AI-generated table of contents via the REST API.

Scans Magazine_Extractions/ for *_final_index.json files produced by
02_AI_generate_summaries_issue.py, prompts for the annotation model, and writes
`dcterms:tableOfContents` to each item — annotated with the model that produced it
(`iwac:summaryModel` value annotation).

SAFETY — preserves existing metadata:
    For each item we GET the full representation, modify ONLY the
    dcterms:tableOfContents property in place, then PATCH the whole object back.
    Every other property is therefore sent back unchanged. A defensive guard
    aborts the write for any item where a top-level property key would disappear.
    (Same pattern as AI_ocr_extraction/03_omeka_content_updater.py.)

Usage:
    python 03_update_omeka_toc.py            # prompts, then updates live
    python 03_update_omeka_toc.py --dry-run  # fetch + report only, writes nothing

Requirements:
    OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL in .env
"""

import argparse
import json
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn,
)
from rich import box

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from common.omeka_client import OmekaClient  # noqa: E402

console = Console()

# Property IDs in Omeka S (verified live against islam.zmo.de)
TABLE_OF_CONTENTS_PROPERTY_ID = 18   # dcterms:tableOfContents
SUMMARY_MODEL_PROPERTY_ID = 313      # iwac:summaryModel ("AI Model - Summary")

# Annotation models — Omeka authority items (class 244, "Notice d'autorité").
# display_title mirrors the actual Omeka item title.
MODEL_CHOICES = {
    "1": {"item_id": 78528, "display_title": "Claude Opus 4.6"},
    "2": {"item_id": 78536, "display_title": "Gemini 3.1 pro"},
    "3": {"item_id": 78630, "display_title": "Gemini 3.5 flash"},
    "4": {"item_id": 78631, "display_title": "Gemini 3.1 flash lite"},
}


def format_article_toc(article: dict) -> str:
    """Format a single article from a final_index.json into a TOC line."""
    title = article["titre"]
    pages = article["pages"]
    authors = article.get("auteurs")
    resume = article.get("resume", "")
    header = (
        f"p. {pages} : {title} ({', '.join(authors)})" if authors else f"p. {pages} : {title}"
    )
    return f"{header}\n{resume}"


def load_from_extractions(extractions_dir: Path) -> list:
    """Scan Magazine_Extractions/ for *_final_index.json files."""
    toc_entries = []
    if not extractions_dir.exists():
        return toc_entries
    for item_dir in sorted(extractions_dir.iterdir()):
        if not item_dir.is_dir():
            continue
        try:
            item_id = int(item_dir.name)
        except ValueError:
            continue
        index_files = list(item_dir.glob("*_final_index.json"))
        if not index_files:
            continue
        with open(index_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        articles = data.get("articles", [])
        if not articles:
            continue
        toc_text = "\n\n".join(format_article_toc(a) for a in articles)
        toc_entries.append({"item_id": item_id, "table_of_contents": toc_text})
    return toc_entries


def build_model_annotation(model: dict, base_url: str) -> dict:
    """Build the iwac:summaryModel resource value used to annotate the TOC value."""
    item_id = model["item_id"]
    return {
        "type": "resource:item",
        "property_id": SUMMARY_MODEL_PROPERTY_ID,
        "property_label": "AI Model - Summary",
        "is_public": True,
        "@id": f"{base_url}/items/{item_id}",
        "value_resource_id": item_id,
        "value_resource_name": "items",
        "url": None,
        "display_title": model["display_title"],
    }


def apply_toc(item_data: dict, toc_text: str, model_value: dict) -> None:
    """Set dcterms:tableOfContents on a fetched item *in place*.

    Replaces the first existing literal TOC value (so re-runs are idempotent) or
    appends a new one, and attaches the model as an iwac:summaryModel value
    annotation. No other property of `item_data` is touched.
    """
    key = "dcterms:tableOfContents"
    new_value = {
        "type": "literal",
        "property_id": TABLE_OF_CONTENTS_PROPERTY_ID,
        "property_label": "Table Of Contents",
        "is_public": True,
        "@value": toc_text,
        "@annotation": {"iwac:summaryModel": [dict(model_value)]},
    }
    values = item_data.get(key)
    if isinstance(values, list) and values:
        for i, v in enumerate(values):
            if isinstance(v, dict) and v.get("type") == "literal":
                values[i] = new_value
                return
        values.append(new_value)
    else:
        item_data[key] = [new_value]


def count_properties(keys) -> int:
    """Number of vocabulary-property keys (e.g. dcterms:*, bibo:*) on an item."""
    return sum(1 for k in keys if ":" in k and not k.startswith(("o:", "@")))


def main():
    parser = argparse.ArgumentParser(
        description="Update Omeka S items with the AI table of contents (preserves existing metadata)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch each item and report what would change, but write nothing.",
    )
    args = parser.parse_args()

    extractions_dir = Path(SCRIPT_DIR) / "Magazine_Extractions"
    toc_entries = load_from_extractions(extractions_dir)
    if not toc_entries:
        console.print("[yellow]No entries found in Magazine_Extractions/.[/]")
        sys.exit(0)

    try:
        client = OmekaClient.from_env()
    except ValueError as e:
        console.print(f"[red]✗[/] {e}")
        return

    # Model selection
    console.print("\n[bold]Select annotation model:[/]")
    for key, info in MODEL_CHOICES.items():
        console.print(f"  {key}. {info['display_title']} (item {info['item_id']})")
    choice = console.input("\nChoice [1]: ").strip() or "1"
    if choice not in MODEL_CHOICES:
        console.print(f"[red]Invalid choice: {choice}[/]")
        sys.exit(1)
    selected_model = MODEL_CHOICES[choice]
    model_value = build_model_annotation(selected_model, client.base_url)

    # Confirm before touching live data
    console.print(Panel(
        f"Items to update:  {len(toc_entries)}\n"
        f"Omeka:            {client.base_url}\n"
        f"Annotation model: {selected_model['display_title']} (item {selected_model['item_id']})\n"
        f"Property written: dcterms:tableOfContents (id {TABLE_OF_CONTENTS_PROPERTY_ID})\n"
        f"Mode:             {'DRY RUN — no writes' if args.dry_run else 'LIVE update'}",
        title="About to update Omeka",
        border_style="cyan" if args.dry_run else "yellow",
    ))
    if not args.dry_run:
        confirm = console.input("\n[bold]Proceed with updating these items? [y/N]:[/] ").strip().lower()
        if confirm not in ("y", "yes"):
            console.print("[yellow]Aborted — no changes made.[/]")
            return

    stats = {"updated": 0, "would_update": 0, "not_found": 0, "failed": 0}
    results = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing items...", total=len(toc_entries))
        for entry in toc_entries:
            item_id = entry["item_id"]
            item_data = client.get_item(item_id)
            if item_data is None:
                stats["not_found"] += 1
                results.append((item_id, "[yellow]not found / fetch failed[/]"))
                progress.update(task, advance=1)
                continue

            before = set(item_data.keys())
            had_toc = bool(item_data.get("dcterms:tableOfContents"))
            apply_toc(item_data, entry["table_of_contents"], model_value)

            # Defensive guard: never PATCH if an existing top-level key vanished.
            lost = before - set(item_data.keys())
            if lost:
                stats["failed"] += 1
                results.append((item_id, f"[red]ABORTED — would drop {sorted(lost)}[/]"))
                progress.update(task, advance=1)
                continue

            kept = count_properties(before)
            verb, verb_past = ("replace", "replaced") if had_toc else ("add", "added")
            if args.dry_run:
                stats["would_update"] += 1
                results.append((item_id, f"would {verb} TOC — {kept} existing properties kept"))
            elif client.update_item(item_id, item_data):
                stats["updated"] += 1
                results.append((item_id, f"[green]{verb_past} TOC[/] — {kept} properties kept"))
            else:
                stats["failed"] += 1
                results.append((item_id, "[red]FAILED (see log)[/]"))
            progress.update(task, advance=1)

    # Results
    table = Table(title="Result", box=box.ROUNDED)
    table.add_column("o:id", style="cyan", justify="right")
    table.add_column("Status")
    for item_id, status in results:
        table.add_row(str(item_id), status)
    console.print(table)

    headline = (
        f"Would update: {stats['would_update']}" if args.dry_run else f"Updated: {stats['updated']}"
    )
    console.print(Panel(
        f"{headline}   Not found: {stats['not_found']}   Failed/aborted: {stats['failed']}",
        title="Dry run complete" if args.dry_run else "Done",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
