"""Tests for atomic, provenance-aware batch checkpoints."""

import json

import pytest

from common.checkpoint import (
    CheckpointMismatch,
    JsonCheckpoint,
    atomic_write_text,
    load_csv_ids,
    sha256_text,
)


def test_checkpoint_resumes_only_matching_context(tmp_path):
    path = tmp_path / "checkpoint.json"
    checkpoint = JsonCheckpoint.open(path, {"model": "dated-model", "prompt": "abc"})
    checkpoint.mark("one.txt", "source-a")

    resumed = JsonCheckpoint.open(path, {"model": "dated-model", "prompt": "abc"})
    assert resumed.matches("one.txt", "source-a")
    assert not resumed.matches("one.txt", "source-b")

    with pytest.raises(CheckpointMismatch, match="provenance differs"):
        JsonCheckpoint.open(path, {"model": "other-model", "prompt": "abc"})


def test_checkpoint_reset_replaces_incompatible_context(tmp_path):
    path = tmp_path / "checkpoint.json"
    first = JsonCheckpoint.open(path, {"model": "old"})
    first.mark("1", "done")

    reset = JsonCheckpoint.open(path, {"model": "new"}, reset=True)

    assert reset.entries == {}
    assert json.loads(path.read_text(encoding="utf-8"))["context"] == {"model": "new"}


def test_atomic_write_and_hash(tmp_path):
    path = tmp_path / "artifact.txt"
    atomic_write_text(path, "complete")
    assert path.read_text(encoding="utf-8") == "complete"
    assert sha256_text("complete") == sha256_text("complete")
    assert not list(tmp_path.glob("*.tmp"))


def test_load_csv_ids_requires_expected_header(tmp_path):
    csv_path = tmp_path / "output.csv"
    csv_path.write_text("o:id,Title\n12,A\n13,B\n", encoding="utf-8")
    assert load_csv_ids(csv_path, "o:id") == {"12", "13"}

    csv_path.write_text("Title\nA\n", encoding="utf-8")
    with pytest.raises(CheckpointMismatch, match="no 'o:id' column"):
        load_csv_ids(csv_path, "o:id")
