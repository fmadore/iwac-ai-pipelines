"""Regression tests for NER and text-summary resume semantics."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from common.checkpoint import CheckpointMismatch, JsonCheckpoint


_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


summary = _load("summary_generator", "AI_summary/02_AI_generate_summaries.py")
ner = _load("ner_generator", "AI_NER/01_NER_AI.py")


def test_summary_resume_reprocesses_only_changed_input(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("article a", encoding="utf-8")
    (input_dir / "b.txt").write_text("article b", encoding="utf-8")
    checkpoint = JsonCheckpoint.open(
        output_dir / ".summary_checkpoint.json", {"model": "pinned", "prompt": "one"}
    )
    client = Mock()
    client.generate.side_effect = lambda _system, user: f"summary:{user[-1]}"

    first = summary.process_txt_files(
        client, str(input_dir), str(output_dir), "system", checkpoint
    )
    second = summary.process_txt_files(
        client, str(input_dir), str(output_dir), "system", checkpoint
    )
    (input_dir / "b.txt").write_text("article b revised", encoding="utf-8")
    third = summary.process_txt_files(
        client, str(input_dir), str(output_dir), "system", checkpoint
    )

    assert first == (2, 0, 0)
    assert second == (0, 0, 2)
    assert third == (1, 0, 1)
    assert client.generate.call_count == 3


def test_ner_resume_uses_flushed_csv_ids(tmp_path):
    output = tmp_path / "ner.csv"
    items = [{"o:id": 1}, {"o:id": 2}]
    context = {"model": "dated", "prompt": "abc"}

    pending, resume, completed = ner._prepare_checkpointed_output(
        str(output), context=context, items=items, force=False
    )
    assert pending == items
    assert not resume
    assert completed == 0

    output.write_text("o:id,Title,bibo:content,Subject AI,Spatial AI\n1,A,T,,\n", encoding="utf-8")
    pending, resume, completed = ner._prepare_checkpointed_output(
        str(output), context=context, items=items, force=False
    )
    assert pending == [{"o:id": 2}]
    assert resume
    assert completed == 1

    with pytest.raises(CheckpointMismatch, match="provenance differs"):
        ner._prepare_checkpointed_output(
            str(output), context={"model": "other"}, items=items, force=False
        )


def test_ner_deduplicates_items_shared_by_multiple_sets():
    assert ner._deduplicate_items([
        {"o:id": 1, "title": "first"},
        {"o:id": 2},
        {"o:id": 1, "title": "duplicate"},
    ]) == [{"o:id": 1, "title": "first"}, {"o:id": 2}]
