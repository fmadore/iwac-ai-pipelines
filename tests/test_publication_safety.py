"""Failure injection across the publication workflows' producer/write boundaries."""
import importlib.util
import io
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from rich.console import Console

from common.artifacts import (
    artifact_matches, checkpoint_artifacts, commit_artifact, invalidate_artifact,
    read_artifact, validated_model,
)
from common.checkpoint import CheckpointMismatch, JsonCheckpoint
from common.gemini_page_processor import (
    GeminiPageProcessor, PagePolicy, TRUNCATION_MARKER, process_pdf_batch,
)
from common.gemini_utils import upload_and_wait_active
from common.link_update_cli import update_reconciled_items
from common.omeka_text_updater import PropertyTarget, apply_text_value
from common.outcomes import batch_exit_code
from common.rate_limiter import QuotaExhaustedError
from common.write_guard import WriteGuard

ROOT = Path(__file__).resolve().parent.parent
QUIET = Console(file=io.StringIO())


@pytest.mark.parametrize("field,value", [("companions", []), ("complete", "yes")])
def test_malformed_artifact_is_rejected(tmp_path, field, value):
    path = tmp_path / "output.txt"
    path.write_text("output", encoding="utf-8")
    commit_artifact(path, context={"model_key": "test"}, source_sha256="source")
    sidecar = path.with_suffix(".txt.artifact.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload[field] = value
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CheckpointMismatch):
        read_artifact(path)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


summary = load("publication_summary", "AI_summary/02_AI_generate_summaries.py")
ner = load("publication_ner", "AI_NER/01_NER_AI.py")
offline = load("publication_offline", "serving/annotate_offline.py")
merge = load("publication_merge", "serving/merge_shards.py")
authority = load("publication_authority", "AI_reference_indexing/04_create_index_items.py")
correction = load("publication_correction", "AI_ocr_correction/02_correct_ocr_text.py")


def test_backup_is_readable_before_each_patch_and_survives_interrupt(tmp_path):
    client = Mock()
    client.get_items_by_ids.return_value = {1: {"o:id": 1}, 2: {"o:id": 2}}

    def write(item_id, payload):
        lines = next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[-1]) == {"o:id": item_id}
        if item_id == 2:
            raise KeyboardInterrupt()
        return True

    client.update_item.side_effect = write
    with pytest.raises(KeyboardInterrupt):
        update_reconciled_items(client, [{"o:id": str(i), "Subject AI Reconciled ID": "42"} for i in (1, 2)],
                                guard=WriteGuard(backup_dir=tmp_path), console=QUIET)
    assert len(next(tmp_path.glob("*.jsonl")).read_text().splitlines()) == 2


def test_unwritable_backup_prevents_any_write(tmp_path):
    client = Mock()
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    with pytest.raises(OSError):
        update_reconciled_items(client, [{"o:id": "1", "Subject AI Reconciled ID": "42"}],
                                guard=WriteGuard(backup_dir=blocker), console=QUIET)
    client.update_item.assert_not_called()


def test_prefetch_is_bounded_to_a_hundred_rows():
    client = Mock()
    client.get_items_by_ids.side_effect = lambda ids: {i: {"o:id": i} for i in ids}
    client.update_item.return_value = True
    update_reconciled_items(client, [{"o:id": str(i), "Subject AI Reconciled ID": "42"} for i in range(205)],
                            guard=WriteGuard(dry_run=True), console=QUIET)
    assert [len(c.args[0]) for c in client.get_items_by_ids.call_args_list] == [100, 100, 5]


def test_annotation_update_keeps_curator_properties():
    item = {"bibo:content": [{"property_id": 91, "type": "literal", "@value": "old",
                             "@annotation": {"curator:note": [{"@value": "keep"}], "o:id": 23}}]}
    target = PropertyTarget("bibo:content", 91, annotation_term="iwac:ocrModel",
                            annotation_value={"value_resource_id": 2})
    apply_text_value(item, target, "new")
    annotation = item["bibo:content"][0]["@annotation"]
    assert annotation["curator:note"] == [{"@value": "keep"}]
    assert "o:id" not in annotation


@pytest.mark.parametrize("failure", [ConnectionError(), RuntimeError(), KeyboardInterrupt()])
def test_upload_cleanup_owns_handle_until_return(failure):
    files = Mock()
    files.upload.return_value = SimpleNamespace(name="file", state=SimpleNamespace(name="PROCESSING"))
    files.get.side_effect = failure
    with patch("common.gemini_utils.time.sleep"), pytest.raises(type(failure)):
        upload_and_wait_active(SimpleNamespace(files=files), b"pdf", mime_type="application/pdf")
    files.delete.assert_called_once_with(name="file")


def test_partial_and_truncated_ocr_preserve_previous_text_but_revoke_upload(tmp_path):
    source = tmp_path / "1.pdf"
    source.write_bytes(b"synthetic")
    output = tmp_path / "1.txt"
    output.write_text("previous complete text")
    commit_artifact(output, context={"model_key": "old"}, source_sha256="old")
    processor = GeminiPageProcessor(None, "model", None, PagePolicy("prompt"), console=QUIET)
    pages = Mock()
    pages.__len__ = Mock(return_value=2)
    pages.page_bytes.return_value = b"pdf"
    with patch("common.gemini_page_processor.PdfPageSource", return_value=pages), patch.object(
        processor, "process_page", side_effect=["cut" + TRUNCATION_MARKER, None]
    ):
        result = processor.process_pdf(source, output)
    assert not result.ok
    assert output.read_text() == "previous complete text"
    assert result.output_file.parent.name == "partial"
    with pytest.raises(CheckpointMismatch):
        read_artifact(output)
    with pytest.raises(CheckpointMismatch):
        read_artifact(result.output_file)


def test_quota_stop_counts_as_failed_batch(tmp_path):
    source = tmp_path / "1.pdf"
    source.write_bytes(b"synthetic")
    processor = GeminiPageProcessor(None, "model", None, PagePolicy("prompt"), console=QUIET)
    with patch.object(processor, "process_pdf", side_effect=QuotaExhaustedError()):
        batch = process_pdf_batch(processor, [source], tmp_path / "out")
    assert batch.failed == 1 and batch.quota_exhausted


def test_failed_force_summary_cannot_upload_old_files(tmp_path, monkeypatch):
    source, fr, en = [tmp_path / folder for folder in ("input", "fr", "en")]
    for folder in (source, fr, en):
        folder.mkdir()
    (source / "1.txt").write_text("source")
    for folder in (fr, en):
        (folder / "1.txt").write_text("old")
    checkpoint = JsonCheckpoint.open(fr / ".summary_checkpoint.json", {"model_key": "new"}, reset=True)
    monkeypatch.setattr(summary, "generate_summary", lambda *args: None)
    assert summary.process_txt_files(None, str(source), str(fr), str(en), "prompt", checkpoint) == (0, 1, 0)
    assert checkpoint_artifacts(checkpoint) == []
    with pytest.raises(CheckpointMismatch):
        validated_model([fr / "1.txt"], requested="new")


def test_companion_hash_rejects_interrupted_bilingual_pair(tmp_path):
    fr, en = tmp_path / "fr.txt", tmp_path / "en.txt"
    fr.write_text("fr")
    en.write_text("en")
    commit_artifact(fr, context={"model_key": "m"}, source_sha256="source", companions=[en])
    en.write_text("another run")
    assert not artifact_matches(fr, {"model_key": "m"}, "source")


def test_ner_changed_source_requires_explicit_regeneration(tmp_path):
    path = tmp_path / "ner.csv"
    items = [{"o:id": 1, "bibo:content": [{"@value": "before"}]}]
    ner._prepare_checkpointed_output(str(path), context={}, items=items, force=False)
    path.write_text("o:id\n1\n")
    items[0]["bibo:content"][0]["@value"] = "after"
    with pytest.raises(CheckpointMismatch):
        ner._prepare_checkpointed_output(str(path), context={}, items=items, force=False)


def record(**overrides):
    return {"item_id": 1, "model": "model", "prompt": "prompt", "reasoning_effort": "low",
            "source_sha256": "source", "result": {}, **overrides}


def test_offline_resume_requires_model_and_source(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text(json.dumps(record()) + "\n")
    kwargs = {"source_hashes": {1: "source"}, "model_id": "model"}
    assert offline.load_done(path, "prompt", "low", logging.getLogger(), **kwargs) == {1}
    assert not offline.load_done(path, "prompt", "low", logging.getLogger(), **{**kwargs, "model_id": "other"})
    assert not offline.load_done(path, "prompt", "low", logging.getLogger(), **{**kwargs, "source_hashes": {1: "changed"}})


@pytest.mark.parametrize("change", [{"model": "other"}, {"reasoning_effort": "high"},
                                    {"prompt": "other"}, {"source_sha256": "changed"}])
def test_merge_refuses_mixed_instruments_and_sources(change):
    with pytest.raises(ValueError):
        merge.resolve({1: [record(), record(**change)]})


def test_authority_creation_journals_before_post_and_recovers_uncertain_result(tmp_path, monkeypatch):
    monkeypatch.setattr(authority, "OUTPUT_DIR", str(tmp_path))
    client = Mock(base_url="https://example.test/api")
    client.search_items_by_property.return_value = []

    def create(payload):
        journal = json.loads(next(tmp_path.glob("authority_*.json")).read_text())
        assert "pending" in journal["entries"].values()
        raise KeyboardInterrupt()

    client.create_item.side_effect = create
    rows = [{"Unreconciled Value": "Authority"}]
    with pytest.raises(KeyboardInterrupt):
        authority.create_authority_items(client, rows, "subject", guard=WriteGuard())
    client.search_items_by_property.return_value = [{"o:id": 4}]
    created, errors = authority.create_authority_items(client, rows, "subject", guard=WriteGuard())
    assert created == [{"term": "Authority", "o:id": "4"}] and errors == 0
    assert client.create_item.call_count == 1
    assert len(list(tmp_path.glob("newly_created*.csv"))) == 2


def test_authority_uncertain_creation_is_not_reposted(tmp_path, monkeypatch):
    monkeypatch.setattr(authority, "OUTPUT_DIR", str(tmp_path))
    client = Mock(base_url="https://example.test/api")
    client.search_items_by_property.return_value = []
    client.create_item.return_value = None
    rows = [{"Unreconciled Value": "Authority"}]
    authority.create_authority_items(client, rows, "subject", guard=WriteGuard())
    with pytest.raises(ValueError, match="Unresolved"):
        authority.create_authority_items(client, rows, "subject", guard=WriteGuard())
    assert client.create_item.call_count == 1


def test_correction_reuses_success_and_stops_on_quota(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    (source / "1.txt").write_text("raw")
    client = Mock()
    client.generate.return_value = "corrected"
    assert correction.process_txt_files(client, source, output, "prompt") == (1, 0)
    assert correction.process_txt_files(client, source, output, "prompt") == (1, 0)
    assert client.generate.call_count == 1
    (source / "2.txt").write_text("raw")
    (source / "3.txt").write_text("raw")
    client.generate.side_effect = QuotaExhaustedError()
    assert correction.process_txt_files(client, source, output, "prompt") == (1, 2)
    assert client.generate.call_count == 2


@pytest.mark.parametrize("key", ["failed", "errors", "not_found", "empty", "incomplete", "cancelled"])
def test_non_success_outcomes_fail_exit(key):
    assert batch_exit_code({key: 1}) == 1
    assert batch_exit_code({"updated": 1}) == 0


def test_manifest_invalidation_revokes_previous_complete_output(tmp_path):
    path = tmp_path / "1.txt"
    path.write_text("original")
    commit_artifact(path, context={"model_key": "m"}, source_sha256="s")
    invalidate_artifact(path)
    with pytest.raises(CheckpointMismatch):
        validated_model([path])
