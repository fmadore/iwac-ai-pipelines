"""Validated producer/consumer handoffs built on atomic JSON checkpoints.

A sidecar is committed last. Invalidating it before generation prevents old or
half-written output from being attributed to a new run after interruption.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from common.checkpoint import CheckpointMismatch, JsonCheckpoint, atomic_write_text, sha256_file

FORMAT = 1


def manifest_path(path: Path) -> Path:
    return Path(path).with_suffix(Path(path).suffix + ".artifact.json")


def invalidate_artifact(path: Path) -> None:
    atomic_write_text(manifest_path(path), json.dumps({"version": FORMAT, "complete": False}))


def commit_artifact(path: Path, *, context: Mapping[str, Any], source_sha256: str,
                    companions: Sequence[Path] = (), complete: bool = True,
                    details: Mapping[str, Any] | None = None) -> None:
    path = Path(path)
    payload = {
        "version": FORMAT, "complete": complete, "context": dict(context),
        "source_sha256": source_sha256, "sha256": sha256_file(path),
        "companions": {os.path.relpath(Path(p).resolve(), path.parent.resolve()): sha256_file(p)
                       for p in companions},
        "details": dict(details or {}),
    }
    atomic_write_text(manifest_path(path), json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def read_artifact(path: Path, *, allow_partial: bool = False) -> dict:
    path = Path(path)
    try:
        payload = json.loads(manifest_path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != FORMAT:
            raise ValueError("unsupported manifest")
        if not isinstance(payload.get("complete"), bool):
            raise ValueError("invalid completion status")
        if not payload["complete"] and not allow_partial:
            raise ValueError("incomplete output")
        if not isinstance(payload.get("context"), dict) or not payload.get("source_sha256"):
            raise ValueError("missing provenance")
        if payload.get("sha256") != sha256_file(path):
            raise ValueError("output hash differs")
        companions = payload.get("companions", {})
        if not isinstance(companions, dict):
            raise ValueError("invalid companion manifest")
        for relative, expected in companions.items():
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise ValueError("invalid companion entry")
            if sha256_file(path.parent / relative) != expected:
                raise ValueError("companion hash differs")
        return payload
    except (OSError, ValueError, TypeError) as exc:
        raise CheckpointMismatch(f"Invalid artifact {path}: {exc}. Regenerate or explicitly import legacy output.") from exc


def artifact_matches(path: Path, context: Mapping[str, Any], source_sha256: str) -> bool:
    try:
        payload = read_artifact(path)
    except CheckpointMismatch:
        return False
    return payload["context"] == dict(context) and payload["source_sha256"] == source_sha256


def validated_model(paths: Sequence[Path], *, requested: str | None = None,
                    legacy: bool = False) -> str:
    """Reject mixed models and forged/stale output before constructing a write."""
    if legacy:
        if not requested:
            raise CheckpointMismatch("--legacy-import requires an explicit --model")
        return requested
    models = {read_artifact(path)["context"].get("model_key") for path in paths}
    if len(models) != 1 or None in models or "" in models:
        raise CheckpointMismatch("Outputs must name one model; regenerate or separate the runs.")
    model = models.pop()
    if requested and requested != model:
        raise CheckpointMismatch(f"Requested model {requested} differs from recorded model {model}")
    return model


def checkpoint_artifacts(checkpoint: JsonCheckpoint) -> list[Path]:
    """Only completed artifacts with the current run's context may be uploaded."""
    paths = []
    for name, source_hash in checkpoint.entries.items():
        if Path(name).name != name:
            raise CheckpointMismatch("Checkpoint artifact names must be local filenames")
        path = checkpoint.path.parent / name
        artifact = read_artifact(path)
        if artifact["context"] != checkpoint.context or artifact["source_sha256"] != source_hash:
            raise CheckpointMismatch(f"Artifact {name} does not belong to this run")
        paths.append(path)
    return paths
