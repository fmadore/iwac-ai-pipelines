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
