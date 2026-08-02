"""Tests for provider-independent magazine consolidation and batch setup."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

from rich.panel import Panel

import magazine_extraction
from common.llm_provider import get_model_option


SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "AI_summary_issue"
    / "02_Mistral_generate_summaries_issue.py"
)
SPEC = importlib.util.spec_from_file_location("mistral_magazine_pipeline", SCRIPT)
assert SPEC and SPEC.loader
mistral_pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mistral_pipeline
SPEC.loader.exec_module(mistral_pipeline)


def test_text_consolidator_uses_shared_structured_client(monkeypatch):
    expected = magazine_extraction.MagazineIndex(articles=[])
    client = Mock()
    client.generate_structured.return_value = expected
    build_client = Mock(return_value=client)
    monkeypatch.setattr(magazine_extraction, "build_llm_client", build_client)

    option = get_model_option("deepseek-v4-flash-0731")
    consolidate = magazine_extraction.build_text_consolidator(option)
    result = consolidate("system", '[{"page": 1}]')

    assert result is expected
    build_client.assert_called_once()
    assert build_client.call_args.args == (option,)
    assert build_client.call_args.kwargs["config"].reasoning_effort == "low"
    client.generate_structured.assert_called_once_with(
        "system",
        'Consolidez les articles extraits ci-dessous:\n\n[{"page": 1}]',
        magazine_extraction.MagazineIndex,
    )


def test_magazine_batch_requires_every_provider_key(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "configured")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    process_magazine = Mock()

    exit_code = magazine_extraction.run_magazine_batch(
        process_magazine,
        script_dir=tmp_path,
        intro_panel=Panel("test"),
        api_key_env=("GEMINI_API_KEY", "OPENROUTER_API_KEY"),
    )

    assert exit_code == 1
    process_magazine.assert_not_called()


def test_mistral_ocr_client_has_finite_timeout(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr(mistral_pipeline, "Mistral", constructor)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    mistral_pipeline._mistral_client_from_env()

    constructor.assert_called_once_with(
        api_key="test-key",
        timeout_ms=mistral_pipeline.MISTRAL_OCR_TIMEOUT_MS,
    )


def test_mistral_upload_cleanup_is_best_effort():
    client = Mock()
    client.files.delete.side_effect = RuntimeError("network failure")

    mistral_pipeline._delete_mistral_upload(client, "file-123")

    client.files.delete.assert_called_once_with(file_id="file-123")
