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
- ``@annotation`` is attached to the value that was just written, so the model
  provenance survives both the append and the in-place rewrite path.
- Unchanged items are skipped rather than re-PATCHed.
- ``--dry-run`` and an interactive confirmation gate are available to all.
- Several values can be written to one item in ONE PATCH — see
  ``TextUpdate.extra_values``.

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

import copy
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from rich.console import Console

from common.console_utils import count_table, standard_progress
from common.omeka_client import OmekaClient
from common.write_guard import WriteGuard

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
    #: BCP-47 tag written as ``@language`` (e.g. ``"fr"``). When set, this write
    #: owns only the literal carrying that tag, so several languages can coexist
    #: on one property. ``None`` keeps the language-blind behaviour every
    #: pipeline had before: the first literal on the property is the target.
    language: Optional[str] = None
    #: Claim a pre-existing literal that carries no ``@language`` at all, tagging
    #: it on the way past. Set on the language that owns the legacy values —
    #: IWAC's ~12,300 French summaries predate the tag — so a bilingual run
    #: upgrades them instead of appending a second French value beside them.
    adopt_untagged: bool = False
    #: Visibility of the value written, as Omeka's per-value ``is_public`` flag.
    #:
    #: ``None`` — the default — means "do not decide": a value that already
    #: exists keeps whatever visibility a curator gave it, and a new one is
    #: created public, which is what every pipeline did before this field
    #: existed. Set it explicitly only where visibility is part of the
    #: pipeline's contract rather than a curatorial choice.
    #:
    #: ``AI_publication_extraction`` sets ``False``: its sources are
    #: copyrighted books, theses and journal articles, and a newly created
    #: ``bibo:content`` would otherwise default to public and publish a whole
    #: monograph. The flag is also what the Hugging Face export reads as
    #: ``OCR_is_public`` to decide whether to mask a row's full text, so it is
    #: load-bearing well beyond the archive's own UI.
    is_public: Optional[bool] = None

    def describe(self) -> str:
        suffix = f" @{self.language}" if self.language else ""
        return f"{self.term} (id {self.property_id}){suffix}"


@dataclass
class TextUpdate:
    """One pending write.

    ``item_id`` is ``None`` when the source could not be resolved to an item
    (e.g. an identifier with no match); such entries are counted as
    ``not_found`` rather than silently dropped.

    ``extra_values`` carries additional ``(target, text)`` pairs applied to the
    same item in the SAME PATCH. AI_summary uses it to write the French and
    English summaries as two language-tagged literals on one property: two
    PATCHes would double the round trips and leave a window where an item holds
    one language and not the other.
    """

    label: str
    item_id: Optional[int]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    extra_values: List[Tuple[PropertyTarget, str]] = field(default_factory=list)

    def writes(self, target: PropertyTarget) -> List[Tuple[PropertyTarget, str]]:
        """Every ``(target, text)`` pair this update applies, main value first."""
        return [(target, self.text), *self.extra_values]


def _own_literal(
    values: List[Any], target: PropertyTarget
) -> Optional[Dict[str, Any]]:
    """Return the value object *target* owns on an already-fetched property.

    Matching is by ``@language`` when the target declares one, so a French and
    an English summary on the same property do not overwrite each other. A
    language-blind target (``language=None``) keeps the historical rule — the
    first literal wins — which is what the OCR, correction and transcription
    updaters rely on.
    """
    untagged: Optional[Dict[str, Any]] = None
    for entry in values:
        if not isinstance(entry, dict):
            continue
        if entry.get("property_id") != target.property_id:
            continue
        if entry.get("type", "literal") != "literal":
            continue
        if target.language is None:
            return entry
        if entry.get("@language") == target.language:
            return entry
        if target.adopt_untagged and not entry.get("@language") and untagged is None:
            untagged = entry
    return untagged


def apply_text_value(item_data: Dict[str, Any], target: PropertyTarget, text: str) -> bool:
    """Set *target*'s literal on a fetched item *in place*, annotation included.

    This deliberately does not call ``OmekaClient.upsert_property_value``: that
    helper matches the first literal on a property regardless of ``@language``,
    so calling it once per language would make the second write clobber the
    first. It also rebuilds the value object from five keys when appending,
    dropping any ``@annotation`` — the provenance is therefore attached here, to
    the exact value just written, and a change to the annotation alone still
    counts as a change.

    Returns:
        True if *item_data* differs from what Omeka currently holds.
    """
    values = item_data.get(target.term)
    if not isinstance(values, list):
        values = item_data[target.term] = []

    changed = False
    value = _own_literal(values, target)

    if value is None:
        value = {
            "type": "literal",
            "property_id": target.property_id,
            "property_label": target.property_label or target.term.split(":")[-1],
            "is_public": True if target.is_public is None else target.is_public,
            "@value": text,
        }
        if target.language:
            value["@language"] = target.language
        values.append(value)
        changed = True
    else:
        if value.get("@value") != text:
            value["@value"] = text
            value["type"] = "literal"
            changed = True
        # Tags a legacy untagged literal claimed via ``adopt_untagged``.
        if target.language and value.get("@language") != target.language:
            value["@language"] = target.language
            changed = True
        # Only when the target states a visibility: a ``None`` target must not
        # republish a value a curator deliberately made private.
        if target.is_public is not None and value.get("is_public") != target.is_public:
            value["is_public"] = target.is_public
            changed = True

    if not target.annotation_term or target.annotation_value is None:
        return changed

    annotation = {target.annotation_term: [dict(target.annotation_value)]}
    if not _annotation_matches(value.get("@annotation"), annotation):
        value["@annotation"] = annotation
        changed = True

    return changed


def _annotation_matches(stored: Any, wanted: Dict[str, Any]) -> bool:
    """True if *stored* already carries every key/value in *wanted*.

    Deliberately not ``==``. Omeka echoes value objects back with keys it fills
    in itself that no client ever sends — a ``resource:item`` link comes back
    carrying ``"url": null`` — so an exact comparison can never match a value
    that was just written, every annotated item reports as changed, and the
    unchanged-skip that this module exists to provide silently never fires.
    That turned a resumed 12,305-item summary run into a full re-PATCH of the
    corpus: correct data, but every item written again.

    Compare only the keys we set, so a server-added key is not a difference.
    """
    if not isinstance(stored, dict):
        return False
    for term, values in wanted.items():
        found = stored.get(term)
        if not isinstance(found, list) or len(found) != len(values):
            return False
        for got, want in zip(found, values, strict=True):
            if not isinstance(got, dict):
                return False
            if any(got.get(key) != value for key, value in want.items()):
                return False
    return True


def apply_text_values(
    item_data: Dict[str, Any], writes: Sequence[Tuple[PropertyTarget, str]]
) -> bool:
    """Apply every ``(target, text)`` pair to one fetched item.

    Returns True if any of them changed it. Empty texts are skipped rather than
    written: a missing translation must not blank a value Omeka already holds.
    """
    changed = False
    for target, text in writes:
        if not text.strip():
            continue
        if apply_text_value(item_data, target, text):
            changed = True
    return changed


#: Called with each item's pre-write JSON, immediately before its PATCH.
BackupSink = Callable[[Dict[str, Any]], None]


@contextmanager
def open_backup(
    directory: Optional[Path],
    *,
    label: str,
    dry_run: bool = False,
    now: Optional[datetime] = None,
) -> Iterator[Optional[BackupSink]]:
    """Yield a sink that appends pre-write item payloads to a JSONL file.

    ``write_guard.WriteGuard.dump_backup`` buffers every payload and writes once
    at the end. That is right for a few hundred items and wrong for a corpus
    pass: it holds ~50 MB of OCR in memory for 12k articles, and a crash at item
    7,000 leaves no backup at all — precisely when one is needed. This writes and
    flushes each item *before* its PATCH, one JSON object per line, so an
    interrupted run still has every item it actually touched.

    Yields ``None`` when backups are off or this is a dry run (nothing changes,
    so there is nothing to roll back to), which callers pass straight through.
    """
    if directory is None or dry_run:
        yield None
        return

    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"_pre_write_{label}_{stamp}.jsonl"

    lock = threading.Lock()  # concurrent writers (the sentiment panel) must not interleave lines
    with path.open("w", encoding="utf-8") as handle:
        def sink(payload: Dict[str, Any]) -> None:
            line = json.dumps(payload, ensure_ascii=False) + "\n"
            with lock:
                handle.write(line)
                handle.flush()  # the PATCH follows immediately; an unflushed line is no backup

        sink.path = path  # type: ignore[attr-defined]
        yield sink


def update_item_text(
    client: OmekaClient,
    item_id: int,
    text: str,
    target: PropertyTarget,
    *,
    dry_run: bool = False,
    extra_values: Sequence[Tuple[PropertyTarget, str]] = (),
    backup: Optional[BackupSink] = None,
) -> str:
    """Fetch, mutate and PATCH one item. Returns a status from :data:`STATUSES`.

    Every value in *extra_values* lands in the same PATCH as *text*. When
    *backup* is given, the item's pre-write state is handed to it before the
    PATCH — and only for items that actually change, so the backup is a record
    of what was overwritten rather than of everything inspected.
    """
    item_data = client.get_item(int(item_id))
    if not item_data:
        return "not_found"

    # Copy before mutating, and only when it will be used: deep-copying a full
    # item means duplicating its OCR blob, which is most of the payload.
    original = copy.deepcopy(item_data) if backup is not None and not dry_run else None

    if not apply_text_values(item_data, [(target, text), *extra_values]):
        return "unchanged"

    if dry_run:
        return "would_update"

    if original is not None and backup is not None:
        backup(original)

    return "updated" if client.update_item(int(item_id), item_data) else "failed"


def texts_from_directory(
    directory: Path,
    *,
    suffix: str = ".txt",
    strip: bool = True,
) -> Dict[int, str]:
    """Map ``<item_id><suffix>`` files in *directory* to their text.

    Files whose stem is not numeric are skipped: the item ID comes from the
    filename, so a non-numeric stem means the file was not produced by the
    pipeline step that owns this directory.
    """
    texts: Dict[int, str] = {}
    for path in sorted(directory.glob(f"*{suffix}")):
        if not path.stem.isdigit():
            continue
        text = path.read_text(encoding="utf-8")
        texts[int(path.stem)] = text.strip() if strip else text
    return texts


def updates_from_directory(
    directory: Path,
    *,
    suffix: str = ".txt",
    strip: bool = True,
) -> List[TextUpdate]:
    """Build updates from ``<item_id><suffix>`` files in *directory*."""
    return [
        TextUpdate(label=f"{item_id}{suffix}", item_id=item_id, text=text)
        for item_id, text in texts_from_directory(
            directory, suffix=suffix, strip=strip
        ).items()
    ]


def describe_targets(updates: Sequence[TextUpdate], target: PropertyTarget) -> List[str]:
    """Panel lines naming every property and annotation a batch will write.

    Every distinct target across the batch, so a bilingual run names both
    values it is about to write rather than only the main one.
    """
    targets: List[PropertyTarget] = [target]
    for update in updates:
        for extra_target, _ in update.extra_values:
            if extra_target not in targets:
                targets.append(extra_target)
    lines = []
    for index, written in enumerate(targets):
        label = "Property:     " if index == 0 else " " * 14
        lines.append(f"{label} {written.describe()}")
    if target.annotation_term and target.annotation_value:
        lines.append(
            f"Annotation:    {target.annotation_term} -> "
            f"{target.annotation_value.get('display_title', '?')}"
        )
    return lines


def confirm_write(
    console: Console,
    updates: Sequence[TextUpdate],
    target: PropertyTarget,
    client: OmekaClient,
    *,
    dry_run: bool,
    extra_lines: Sequence[str] = (),
    guard: Optional[WriteGuard] = None,
) -> bool:
    """Show what is about to be written and, in live mode, ask to proceed.

    One gate for every write step: this is ``WriteGuard.confirm`` with the
    text-update details filled in, so the panel and the EOF-is-not-consent
    rule are the same whether a script writes literals or resource links.
    """
    guard = guard or WriteGuard(dry_run=dry_run)
    return guard.confirm(
        console,
        action=f"Update {len(updates)} item(s)",
        base_url=client.base_url,
        item_count=len(updates),
        details=[*describe_targets(updates, target), *extra_lines],
        title="About to update Omeka",
    )


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
    backup_dir: Optional[Path] = None,
    backup_label: str = "text_update",
    guard: Optional[WriteGuard] = None,
) -> Dict[str, int]:
    """Run the whole write step: confirm, PATCH each item, print a summary.

    Pass a :class:`WriteGuard` (built from ``add_write_guard_args``) and it
    supplies the dry-run flag, the confirmation policy and the backup folder;
    the older keyword arguments remain for callers that resolve those
    themselves. Every item's pre-write JSON is appended to a timestamped
    ``.jsonl`` in the backup folder before its PATCH — the only route back
    from a bulk overwrite.

    Returns:
        A dict of :data:`STATUSES` counts. An empty dict means the operator
        declined at the confirmation prompt.
    """
    console = console or Console()
    stats: Dict[str, int] = {status: 0 for status in STATUSES}
    if guard is not None:
        dry_run = guard.dry_run
        require_confirmation = not guard.assume_yes
        backup_dir = guard.backup_dir if guard.backup_enabled else None
    else:
        guard = WriteGuard(dry_run=dry_run, assume_yes=not require_confirmation, backup_dir=backup_dir,
                           backup_enabled=backup_dir is not None)

    if not updates:
        console.print("[yellow]⚠[/] Nothing to update.")
        return stats

    if require_confirmation and not confirm_write(
        console, updates, target, client, dry_run=dry_run,
        extra_lines=list(extra_confirm_lines), guard=guard,
    ):
        return {}

    console.print()
    console.rule("[bold blue]Updating Omeka S Items")
    console.print()

    with open_backup(backup_dir, label=backup_label, dry_run=dry_run) as backup:
        with standard_progress(console) as progress:
            task = progress.add_task(f"[cyan]{description}", total=len(updates))

            for update in updates:
                try:
                    if update.item_id is None:
                        stats["not_found"] += 1
                    elif not any(text.strip() for _, text in update.writes(target)):
                        console.print(f"  [yellow]⚠[/] {update.label} is empty — skipped")
                        stats["empty"] += 1
                    else:
                        status = update_item_text(
                            client, update.item_id, update.text, target,
                            dry_run=dry_run, extra_values=update.extra_values,
                            backup=backup,
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

        backup_path = getattr(backup, "path", None) if backup else None

    _print_summary(console, stats, len(updates), dry_run=dry_run)
    if backup_path:
        console.print(f"[dim]Pre-write backup: {backup_path}[/]")
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
