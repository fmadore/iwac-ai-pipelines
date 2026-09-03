"""Small, durable checkpoints for file-producing batch pipelines.

The checkpoint context identifies the run configuration (model, prompt, input
scope). Entries identify completed artifacts within that context. Every update
uses ``os.replace`` so interruption leaves either the old complete JSON file or
the new one, never a half-written manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping


CHECKPOINT_FORMAT_VERSION = 1


class CheckpointMismatch(ValueError):
    """Raised when an output belongs to a different model/prompt/input scope."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write *content* beside *path* and atomically replace the destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("w", encoding=encoding, newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass
class JsonCheckpoint:
    path: Path
    context: Dict[str, Any]
    entries: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def open(
        cls,
        path: Path,
        context: Mapping[str, Any],
        *,
        reset: bool = False,
    ) -> "JsonCheckpoint":
        path = Path(path)
        expected = dict(context)
        if path.exists() and not reset:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CheckpointMismatch(
                    f"Checkpoint is unreadable: {path}. Use --force to replace it."
                ) from exc
            if payload.get("version") != CHECKPOINT_FORMAT_VERSION:
                raise CheckpointMismatch(
                    f"Checkpoint version differs: {path}. Use --force to replace it."
                )
            if payload.get("context") != expected:
                raise CheckpointMismatch(
                    f"Output provenance differs from this run: {path}. "
                    "Use --force to replace the existing output."
                )
            entries = payload.get("entries", {})
            if not isinstance(entries, dict):
                raise CheckpointMismatch(
                    f"Checkpoint entries are invalid: {path}. Use --force to replace it."
                )
            return cls(path=path, context=expected, entries=dict(entries))

        checkpoint = cls(path=path, context=expected)
        checkpoint.save()
        return checkpoint

    def matches(self, key: str, fingerprint: str) -> bool:
        return self.entries.get(str(key)) == fingerprint

    def mark(self, key: str, fingerprint: str) -> None:
        self.entries[str(key)] = fingerprint
        self.save()

    def save(self) -> None:
        payload = {
            "version": CHECKPOINT_FORMAT_VERSION,
            "context": self.context,
            "entries": self.entries,
        }
        atomic_write_text(
            self.path,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )


def checkpoint_path_for(output_path: Path) -> Path:
    """The checkpoint that sits beside a pipeline output file.

    ``item_set_1_processed_x.csv`` -> ``item_set_1_processed_x.csv.checkpoint.json``.
    One naming rule, so a downstream step can find the provenance of the file
    it was handed without the upstream step having to pass it along.
    """
    output_path = Path(output_path)
    return output_path.with_suffix(output_path.suffix + ".checkpoint.json")


def read_checkpoint_context(output_path: Path) -> Dict[str, Any]:
    """Return the run context recorded beside *output_path*, or ``{}``.

    Tolerant on purpose: this is how a write step recovers which model
    produced the file it is about to upload, so a missing or unreadable
    checkpoint means "ask the operator", never "abort".
    """
    path = checkpoint_path_for(output_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    context = payload.get("context") if isinstance(payload, dict) else None
    return dict(context) if isinstance(context, dict) else {}


def load_csv_ids(path: Path, id_column: str) -> set[str]:
    """Return completed IDs from a CSV; a malformed/missing header is unsafe."""
    import csv

    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or id_column not in reader.fieldnames:
            raise CheckpointMismatch(
                f"Existing CSV has no {id_column!r} column: {path}. "
                "Use --force to replace it."
            )
        return {
            str(row[id_column]).strip()
            for row in reader
            if row.get(id_column) and str(row[id_column]).strip()
        }
