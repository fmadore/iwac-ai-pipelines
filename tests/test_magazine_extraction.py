"""Tests for provider-independent magazine consolidation and batch setup."""

from unittest.mock import Mock

from rich.panel import Panel

import magazine_extraction
from common.llm_provider import get_model_option


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
