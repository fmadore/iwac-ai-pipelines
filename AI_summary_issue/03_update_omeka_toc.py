"""
Generate a CSV file with table of contents for Omeka S CSV Import.

Scans Magazine_Extractions/ for *_final_index.json files produced by
02_AI_generate_summaries_issue.py, prompts for the annotation model,
and outputs a CSV ready for import.

Usage:
    python 03_update_omeka_toc.py
"""

import csv
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Available model annotations
MODEL_CHOICES = {
    "1": {
        "item_id": 78528,
        "display_title": "Claude Opus 4.6",
    },
    "2": {
        "item_id": 78536,
        "display_title": "Gemini 3.1 pro",
    },
}


def format_article_toc(article: dict) -> str:
    """Format a single article from a final_index.json into a TOC line."""
    title = article["titre"]
    pages = article["pages"]
    authors = article.get("auteurs")
    resume = article.get("resume", "")

    if authors:
        header = f"p. {pages} : {title} ({', '.join(authors)})"
    else:
        header = f"p. {pages} : {title}"

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

        with open(index_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)

        articles = data.get("articles", [])
        if not articles:
            continue

        toc_text = "\n\n".join(format_article_toc(a) for a in articles)
        toc_entries.append({"item_id": item_id, "table_of_contents": toc_text})

    return toc_entries


def main():
    script_dir = Path(__file__).parent
    extractions_dir = script_dir / "Magazine_Extractions"

    # Load TOC entries from Magazine_Extractions/
    toc_entries = load_from_extractions(extractions_dir)
    if not toc_entries:
        console.print("[yellow]No entries found in Magazine_Extractions/.[/]")
        sys.exit(0)

    # Model selection
    console.print("\n[bold]Select annotation model:[/]")
    for key, info in MODEL_CHOICES.items():
        console.print(f"  {key}. {info['display_title']}")
    choice = input("\nChoice [1]: ").strip() or "1"
    if choice not in MODEL_CHOICES:
        console.print(f"[red]Invalid choice: {choice}[/]")
        sys.exit(1)
    selected_model = MODEL_CHOICES[choice]

    # Write CSV
    output_path = script_dir / "toc_import.csv"
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "o:id",
            "dcterms:tableOfContents",
            "iwac:summaryModel [annotation]",
        ])
        for entry in toc_entries:
            writer.writerow([
                entry["item_id"],
                entry["table_of_contents"],
                selected_model["item_id"],
            ])

    # Display summary
    console.print(Panel(
        f"Source: {extractions_dir}\n"
        f"Items: {len(toc_entries)}\n"
        f"Model: {selected_model['display_title']}\n"
        f"Output: {output_path}",
        title="CSV Generated",
    ))

    # Preview
    preview = Table(title="Preview")
    preview.add_column("o:id", style="cyan")
    preview.add_column("dcterms:tableOfContents", max_width=60)
    preview.add_column("annotation", style="green")
    for entry in toc_entries:
        toc_preview = entry["table_of_contents"][:80] + "..."
        preview.add_row(
            str(entry["item_id"]),
            toc_preview,
            str(selected_model["item_id"]),
        )
    console.print(preview)


if __name__ == "__main__":
    main()
