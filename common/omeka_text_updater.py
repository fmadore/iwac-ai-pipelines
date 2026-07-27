"""
Shared "write a block of text back to Omeka S" step.

Four pipelines ended each run by reading a directory of ``.txt`` files and
PATCHing one property per item: AI_summary (bibo:shortDescription),
AI_ocr_extraction and AI_ocr_correction (bibo:content), AI_audio_summary
(bibo:content, matched by dcterms:identifier). They had drifted into four
different behaviours for the same operation — only two detected "nothing
changed" before PATCHing, only two offered ``--dry-run``, only two asked for
confirmation, and two carried their own copy of the model-selection prompt.

This module owns the write half so every pipeline gets the safest behaviour:

- The full item is fetched and PATCHed back (never a trimmed payload — Omeka
  deletes any property missing from the request).
- ``@annotation`` is re-attached after ``upsert_property_value``, which builds
  a bare literal and would otherwise silently drop the model provenance.
- Unchanged items are skipped rather than re-PATCHed.
- ``--dry-run`` and an interactive confirmation gate are available to all.

Usage:
    from common.omeka_text_updater import PropertyTarget, TextUpdate, run_text_updates

    target = PropertyTarget(
        term="bibo:shortDescription",
        property_id=summary_property_id,
        property_label="shortDescription",
        annotation_term="iwac:summaryModel",
        annotation_value=model_value,
    )
    updates = updates_from_directory(Path("Summaries_FR_TXT"))
    stats = run_text_updates(client, updates, target, console=console, dry_run=args.dry_run)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rich.console import Console
from rich.panel import Panel

from common.console_utils import count_table, standard_progress
from common.omeka_client import OmekaClient

# Outcome buckets, in the order they are reported.
STATUSES = ("updated", "would_update", "unchanged", "empty", "not_found", "failed")


@dataclass(frozen=True)
class PropertyTarget:
    """Which property to write, and the provenance annotation to attach."""

    term: str
    property_id: int
    property_label: str = ""
    #: e.g. ``"iwac:summaryModel"``; omit for writes without model provenance.
    annotation_term: Optional[str] = None
    #: The ``resource:item`` value object built by ``iwac_config.model_annotation_value``.
    annotation_value: Optional[Dict[str, Any]] = None

    def describe(self) -> str:
        return f"{self.term} (id {self.property_id})"


@dataclass
class TextUpdate:
    """One pending write.

    ``item_id`` is ``None`` when the source could not be resolved to an item
    (e.g. an identifier with no match); such entries are counted as
    ``not_found`` rather than silently dropped.
    """

    label: str
    item_id: Optional[int]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def apply_text_value(item_data: Dict[str, Any], target: PropertyTarget, text: str) -> bool:
    """Set *target*'s literal on a fetched item *in place*, annotation included.

    ``OmekaClient.upsert_property_value`` rebuilds the value object from five
    keys when it appends, so any existing ``@annotation`` is lost. The
    annotation is therefore re-attached here, and a change to the annotation
    alone still counts as a change.

    Returns:
        True if *item_data* differs from what Omeka currently holds.
    """
    changed = OmekaClient.upsert_property_value(
        item_data,
        target.term,
        target.property_id,
        text,
        property_label=target.property_label or target.term.split(":")[-1],
    )

    if not target.annotation_term or target.annotation_value is None:
        return changed

    annotation = {target.annotation_term: [dict(target.annotation_value)]}
    for value in item_data.get(target.term, []) or []:
        if (
            isinstance(value, dict)
            and value.get("property_id") == target.property_id
            and value.get("type", "literal") == "literal"
            and value.get("@value") == text
        ):
            if value.get("@annotation") != annotation:
                value["@annotation"] = annotation
                changed = True
            break

    return changed


def update_item_text(
    client: OmekaClient,
    item_id: int,
    text: str,
    target: PropertyTarget,
    *,
    dry_run: bool = False,
) -> str:
    """Fetch, mutate and PATCH one item. Returns a status from :data:`STATUSES`."""
    item_data = client.get_item(int(item_id))
    if not item_data:
        return "not_found"

    if not apply_text_value(item_data, target, text):
        return "unchanged"

    if dry_run:
        return "would_update"

    return "updated" if client.update_item(int(item_id), item_data) else "failed"


def updates_from_directory(
    directory: Path,
    *,
    suffix: str = ".txt",
    strip: bool = True,
) -> List[TextUpdate]:
    """Build updates from ``<item_id><suffix>`` files in *directory*.

    Files whose stem is not numeric are skipped: the item ID comes from the
    filename, so a non-numeric stem means the file was not produced by the
    pipeline step that owns this directory.
    """
    updates: List[TextUpdate] = []
    for path in sorted(directory.glob(f"*{suffix}")):
        if not path.stem.isdigit():
            continue
        text = path.read_text(encoding="utf-8")
        updates.append(TextUpdate(label=path.name, item_id=int(path.stem),
                                  text=text.strip() if strip else text))
    return updates


def confirm_write(
    console: Console,
    updates: Sequence[TextUpdate],
    target: PropertyTarget,
    client: OmekaClient,
    *,
    dry_run: bool,
    extra_lines: Sequence[str] = (),
) -> bool:
    """Show what is about to be written and, in live mode, ask to proceed.

    Returns False when the operator declines. Bulk PATCH against a live archive
    is not something to start from a bare argv.
    """
    lines = [
        f"Items to update:  {len(updates)}",
        f"Omeka:            {client.base_url}",
        f"Property written: {target.describe()}",
    ]
    if target.annotation_term and target.annotation_value:
        lines.append(
            f"Annotation:       {target.annotation_term} -> "
            f"{target.annotation_value.get('display_title', '?')}"
        )
    lines.extend(extra_lines)
    lines.append(f"Mode:             {'DRY RUN — no writes' if dry_run else 'LIVE update'}")

    console.print(Panel(
        "\n".join(lines),
        title="About to update Omeka",
        border_style="cyan" if dry_run else "yellow",
    ))

    if dry_run:
        return True

    answer = console.input("\n[bold]Proceed with updating these items? [y/N]:[/] ").strip().lower()
    if answer not in ("y", "yes"):
        console.print("[yellow]Aborted — no changes made.[/]")
        return False
    return True


def run_text_updates(
    client: OmekaClient,
    updates: Sequence[TextUpdate],
    target: PropertyTarget,
    *,
    console: Optional[Console] = None,
    dry_run: bool = False,
    require_confirmation: bool = True,
    extra_confirm_lines: Sequence[str] = (),
    description: str = "Updating items...",
) -> Dict[str, int]:
    """Run the whole write step: confirm, PATCH each item, print a summary.

    Returns:
        A dict of :data:`STATUSES` counts. An empty dict means the operator
        declined at the confirmation prompt.
    """
    console = console or Console()
    stats: Dict[str, int] = {status: 0 for status in STATUSES}

    if not updates:
        console.print("[yellow]⚠[/] Nothing to update.")
        return stats

    if require_confirmation and not confirm_write(
        console, updates, target, client, dry_run=dry_run, extra_lines=extra_confirm_lines
    ):
        return {}

    console.print()
    console.rule("[bold blue]Updating Omeka S Items")
    console.print()

    with standard_progress(console) as progress:
        task = progress.add_task(f"[cyan]{description}", total=len(updates))

        for update in updates:
            try:
                if update.item_id is None:
                    stats["not_found"] += 1
                elif not update.text.strip():
                    console.print(f"  [yellow]⚠[/] {update.label} is empty — skipped")
                    stats["empty"] += 1
                else:
                    status = update_item_text(
                        client, update.item_id, update.text, target, dry_run=dry_run
                    )
                    if status == "failed":
                        console.print(f"  [red]✗[/] PATCH failed for item {update.item_id} (see log)")
                    elif status == "not_found":
                        console.print(f"  [yellow]⚠[/] Item {update.item_id} not found — skipped")
                    stats[status] += 1
            except Exception as exc:
                console.print(f"  [red]✗[/] Error processing {update.label}: {exc}")
                stats["failed"] += 1

            progress.update(task, advance=1)

    _print_summary(console, stats, len(updates), dry_run=dry_run)
    return stats


def _print_summary(console: Console, stats: Dict[str, int], total: int, *, dry_run: bool) -> None:
    console.print()
    console.rule("[bold blue]Summary")
    console.print()

    rows = []
    if dry_run:
        rows.append(("[green]Would Update[/]", f"[green]{stats['would_update']}[/]"))
    else:
        rows.append(("[green]Successfully Updated[/]", f"[green]{stats['updated']}[/]"))
    rows.append(("[dim]Already up to date[/]", f"[dim]{stats['unchanged']}[/]"))
    if stats["empty"]:
        rows.append(("[yellow]Empty (skipped)[/]", f"[yellow]{stats['empty']}[/]"))
    rows.append(("[yellow]Not Found (skipped)[/]", f"[yellow]{stats['not_found']}[/]"))
    rows.append(("[red]Failed[/]", f"[red]{stats['failed']}[/]"))
    rows.append(("Total", str(total)))
    console.print(count_table(rows))

    if dry_run:
        console.print("\n[green]✓[/] Dry run completed — no changes were made.")
    else:
        console.print("\n[green]✓[/] Update process completed.")
