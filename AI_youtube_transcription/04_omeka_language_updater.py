#!/usr/bin/env python3
"""
Correct ``dcterms:language`` on Omeka items from what step 02 actually heard.

Step 02 detects the spoken languages before transcribing, because the catalogue
record is not reliable enough to prompt from: all 46 YouTube items are catalogued
``Français`` and at least one is dominated by Mooré. That detection is evidence
about the record, and this step is what acts on it — the transcripts themselves
already tag each code-switch, so the information exists whether or not the record
reflects it.

``dcterms:language`` is a *link* to an authority item, not a literal, so this
writes resource links through ``common/omeka_link_updater.py``.

**It only ever appends.** A catalogued language that was not detected is reported,
never removed: detection samples 90 seconds of a recording, which is enough to
prove a language is present and not enough to prove one is absent. Removing
``Français`` from an item because two windows happened to be in Mooré would be
deleting a curator's judgement on the strength of a sample.

Languages marked ``occasional`` — an isolated phrase, a quotation, a line of
Qur'anic recitation — are skipped by default. Cataloguing Arabic for one
``bismillah`` would drown the field it is meant to describe.

Usage:
    python 04_omeka_language_updater.py --dry-run
    python 04_omeka_language_updater.py
    python 04_omeka_language_updater.py --include-occasional --yes

Requirements:
    - Environment variables: OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL
    - Transcriptions/_language_report.json from 02_AI_transcribe_youtube.py
"""

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Add repo root to path for shared imports, and this pipeline's own directory
# for the sibling format module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.console_utils import count_table, key_value_table, standard_progress
from common.iwac_config import (
    AUTHORITY_ITEM_SET_ID,
    DCTERMS_LANGUAGE_PROPERTY_ID,
    DCTERMS_TITLE_PROPERTY_ID,
    LANGUAGE_LABELS_BY_CODE,
)
from common.log_redaction import install_credential_redaction
from common.omeka_client import OmekaClient
from common.omeka_link_updater import ResourceLinkSpec, update_item_resource_links
from common.write_guard import WriteGuard, add_write_guard_args

from youtube_source import DetectedLanguage, parse_detected_languages

install_credential_redaction()

console = Console()
LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.resolve()
LANGUAGE_REPORT = SCRIPT_DIR / "Transcriptions" / "_language_report.json"
BACKUP_DIR = SCRIPT_DIR / "backups"

LANGUAGE_TERM = "dcterms:language"

#: Shares written by default. ``occasional`` is excluded — see the module
#: docstring; ``--include-occasional`` adds it.
DEFAULT_SHARES = ("dominant", "secondary")


@dataclass
class ItemLanguages:
    """One item's detected languages, resolved to authority items."""

    item_id: int
    identifier: str = ""
    catalogued: str = ""
    detected: List[DetectedLanguage] = field(default_factory=list)
    #: label -> Omeka item id, for the languages that have an authority record.
    resolved: Dict[str, int] = field(default_factory=dict)
    #: Labels with no authority record — reported, never created.
    unresolved: List[str] = field(default_factory=list)


def setup_logging(log_folder: Path) -> None:
    log_folder.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_folder / "language_update.log", mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    install_credential_redaction()


def read_report(path: Path, *, shares: Sequence[str]) -> List[ItemLanguages]:
    """Read step 02's language report, keeping the requested shares."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} is not a list of per-item records")

    items: List[ItemLanguages] = []
    for row in payload:
        # The report names the field ``detected``; ``parse_detected_languages``
        # reads the ``languages`` key of a model response. Passing the row
        # straight through silently finds nothing, which made this whole step a
        # no-op until a test caught it.
        detected = [
            language
            for language in parse_detected_languages({"languages": row.get("detected")})
            if language.share in shares
        ]
        if not detected:
            continue
        items.append(ItemLanguages(
            item_id=int(row["item_id"]),
            identifier=str(row.get("identifier") or ""),
            catalogued=str(row.get("catalogued") or ""),
            detected=detected,
        ))
    return items


def resolve_authority_items(
    client: OmekaClient, labels: Sequence[str]
) -> Dict[str, Optional[int]]:
    """Look up each French language label in the authority item set.

    Resolved by title rather than from a hardcoded table: the IDs are assigned
    per installation and the ones here are scattered (Français 8355, Ewé 66720,
    Kabyè 79081), so a table would be one more thing to keep true.
    """
    resolved: Dict[str, Optional[int]] = {}
    for label in sorted(set(labels)):
        hits = client.search_items_by_property(
            DCTERMS_TITLE_PROPERTY_ID, label, per_page=5,
            item_set_id=AUTHORITY_ITEM_SET_ID,
        )
        exact = [
            hit for hit in hits
            if str(hit.get("o:title") or "").strip().casefold() == label.casefold()
        ]
        resolved[label] = int(exact[0]["o:id"]) if exact else None
    return resolved


def attach_authority_items(
    items: Sequence[ItemLanguages], authority: Dict[str, Optional[int]]
) -> None:
    """Split each item's languages into resolved links and reportable gaps."""
    for item in items:
        for language in item.detected:
            label = LANGUAGE_LABELS_BY_CODE.get(language.code)
            if label is None:
                # Detection named a language this instance has no label for at
                # all. Reported under the model's own name so it can be added.
                item.unresolved.append(f"{language.name_en} ({language.bcp47})")
                continue
            item_id = authority.get(label)
            if item_id is None:
                item.unresolved.append(label)
            else:
                item.resolved[label] = item_id


def report_plan(items: Sequence[ItemLanguages]) -> None:
    """Show what would be linked, and what cannot be."""
    table = Table(title="🗣 Detected languages", box=box.ROUNDED)
    table.add_column("Item", style="cyan", justify="right")
    table.add_column("Catalogued", style="dim")
    table.add_column("To link", style="green")
    table.add_column("No authority record", style="yellow")
    for item in items:
        table.add_row(
            str(item.item_id),
            item.catalogued or "—",
            ", ".join(sorted(item.resolved)) or "—",
            ", ".join(sorted(set(item.unresolved))) or "",
        )
    console.print(table)

    missing = Counter(
        label for item in items for label in set(item.unresolved)
    )
    if missing:
        console.print(
            f"\n[yellow]⚠[/] {len(missing)} language(s) have no authority record and "
            "will not be linked:"
        )
        for label, count in missing.most_common():
            console.print(f"    [dim]{label} — {count} item(s)[/]")
        console.print(
            f"[dim]  Create the record in item set {AUTHORITY_ITEM_SET_ID} "
            "(class 244, template 3, dcterms:type → \"Notice d'autorité\") and re-run.[/]"
        )

    # A catalogued language that no window contained. Informational only: 90
    # seconds of sampling cannot prove a language is absent from an hour of video.
    unconfirmed = [
        item for item in items
        if item.catalogued and not any(
            label.casefold() == item.catalogued.strip().casefold() for label in item.resolved
        )
    ]
    if unconfirmed:
        console.print(
            f"\n[dim]{len(unconfirmed)} item(s) are catalogued in a language the "
            "samples did not contain. Nothing is removed — check those by hand:[/]"
        )
        for item in unconfirmed[:10]:
            console.print(
                f"    [dim]{item.item_id}: catalogued {item.catalogued}, "
                f"heard {', '.join(lang.name_en for lang in item.detected)}[/]"
            )


def apply_updates(
    client: OmekaClient, items: Sequence[ItemLanguages], *, guard: WriteGuard
) -> Counter:
    """Append the resolved language links, one PATCH per changed item."""
    stats: Counter = Counter(total=len(items))
    pre_write: List[MutableMapping[str, Any]] = []

    console.print()
    console.rule("[bold blue]Updating dcterms:language")
    console.print()

    with standard_progress(console) as progress:
        task = progress.add_task(
            "[cyan]Checking items...[/]" if guard.dry_run else "[cyan]Updating items...[/]",
            total=len(items),
        )
        for item in items:
            if not item.resolved:
                stats["skipped"] += 1
                progress.update(task, advance=1)
                continue
            result = update_item_resource_links(
                client, item.item_id,
                [ResourceLinkSpec(
                    term=LANGUAGE_TERM,
                    property_id=DCTERMS_LANGUAGE_PROPERTY_ID,
                    resource_ids=sorted(item.resolved.values()),
                    property_label="Language",
                )],
                dry_run=guard.dry_run,
                on_pre_write=pre_write.append,
            )
            stats[result.status] += 1
            stats["links_added"] += result.total_added
            if result.status == "failed":
                console.print(f"  [red]✗[/] PATCH failed for item {item.item_id} (see log)")
            elif result.status == "not_found":
                console.print(f"  [yellow]⚠[/] Item {item.item_id} not found — skipped")
            progress.update(task, advance=1)

    backup_path = guard.dump_backup(pre_write, label="youtube_languages")
    if backup_path is not None:
        console.print(f"[dim]Pre-write payloads saved to {backup_path}[/]")
    return stats


def show_summary(stats: Counter, *, dry_run: bool) -> None:
    console.print()
    console.rule("[bold blue]Summary")
    console.print()
    console.print(count_table([
        ("[green]Would update[/]" if dry_run else "[green]Updated[/]",
         stats["would_update"] if dry_run else stats["updated"]),
        ("[dim]Already correct[/]", stats["unchanged"]),
        ("[dim]Nothing linkable[/]", stats["skipped"] or None),
        ("[yellow]Not found[/]", stats["not_found"] or None),
        ("[red]Failed[/]", stats["failed"] or None),
        ("Language links added", stats["links_added"]),
        ("Total items", stats["total"]),
    ]))
    console.print(
        "\n[green]✓[/] Dry run completed — no changes were made."
        if dry_run else "\n[green]✓[/] Update process completed."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append detected spoken languages to dcterms:language on Omeka items.",
    )
    parser.add_argument(
        "--report", type=Path, default=LANGUAGE_REPORT,
        help=f"Language report from step 02 (default: {LANGUAGE_REPORT.name}).",
    )
    parser.add_argument(
        "--include-occasional", action="store_true",
        help="Also link languages heard only in isolated phrases or quotations. "
             "Off by default: one 'bismillah' does not make an item Arabic.",
    )
    parser.add_argument(
        "--item-id", type=int, action="append", default=[], dest="item_ids",
        help="Restrict to these items (repeatable).",
    )
    add_write_guard_args(parser, default_backup_dir=BACKUP_DIR)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    guard = WriteGuard.from_args(args, default_backup_dir=BACKUP_DIR)
    setup_logging(SCRIPT_DIR / "log")

    console.print(Panel(
        "Correct dcterms:language from the languages step 02 heard. Appends only — "
        "a catalogued language the samples missed is reported, never removed.",
        title="🗣 Omeka S Language Updater",
        border_style="cyan",
    ))

    if not args.report.exists():
        console.print(
            f"\n[red]✗[/] No language report at [cyan]{args.report}[/]\n"
            "[dim]Run 02_AI_transcribe_youtube.py first (with language detection on).[/]"
        )
        return 1

    shares = tuple(DEFAULT_SHARES) + (("occasional",) if args.include_occasional else ())

    try:
        client = OmekaClient.from_env()
    except ValueError as exc:
        console.print(f"[red]Configuration Error:[/] {exc}")
        return 1

    try:
        items = read_report(args.report, shares=shares)
    except (OSError, ValueError, KeyError) as exc:
        console.print(f"[red]✗[/] Could not read {args.report}: {exc}")
        return 1

    if args.item_ids:
        wanted = set(args.item_ids)
        items = [item for item in items if item.item_id in wanted]
    if not items:
        console.print("[yellow]No detected languages to write.[/]")
        return 0

    console.print()
    console.print(key_value_table([
        ("Report", str(args.report)),
        ("Items", str(len(items))),
        ("Shares written", ", ".join(shares)),
        ("Property", f"{LANGUAGE_TERM} (id {DCTERMS_LANGUAGE_PROPERTY_ID})"),
    ]))
    console.print()

    with console.status("[cyan]Resolving language authority items...[/]"):
        labels = [
            label for item in items for label in (
                LANGUAGE_LABELS_BY_CODE.get(language.code) for language in item.detected
            ) if label
        ]
        authority = resolve_authority_items(client, labels)
    attach_authority_items(items, authority)

    report_plan(items)

    linkable = [item for item in items if item.resolved]
    if not linkable:
        console.print("\n[yellow]Nothing to link — no detected language has an authority record.[/]")
        return 0

    if not guard.confirm(
        console,
        action=f"Append {LANGUAGE_TERM} links",
        base_url=client.base_url,
        item_count=len(linkable),
        details=[f"Source:        {args.report.name}"],
    ):
        return 1

    try:
        stats = apply_updates(client, items, guard=guard)
    except Exception as exc:
        console.print(f"[red]Unexpected error: {exc}[/]")
        console.print_exception()
        return 1

    show_summary(stats, dry_run=guard.dry_run)
    return 0 if not stats["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
