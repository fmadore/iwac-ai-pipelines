"""Tests for the shared Gemini page-by-page PDF loop.

These pin the behaviours that differed between the OCR and HTR copies before
they were unified — each difference had lost something in one of the two.
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from common.gemini_page_processor import (
    TRUNCATION_MARKER,
    GeminiPageProcessor,
    PagePolicy,
    _join_pages,
)
from common import gemini_utils
from common.gemini_utils import INLINE_REQUEST_LIMIT_BYTES
from common.rate_limiter import QuotaExhaustedError


def quiet_console():
    return Console(file=io.StringIO(), force_terminal=False)


def make_response(text=None, finish_reason="FinishReason.STOP", has_parts=True):
    """Build a Gemini response stub with one candidate."""
    part = MagicMock(text=text, thought=False)
    candidate = MagicMock()
    candidate.finish_reason = finish_reason
    candidate.content = MagicMock(parts=[part] if has_parts else []) if has_parts else None
    response = MagicMock(candidates=[candidate])
    return response


def fake_page_source(page_count):
    """Stand in for PdfPageSource without touching a real PDF."""
    source = MagicMock()
    source.__len__.return_value = page_count
    source.page_bytes.return_value = b"%PDF"
    return source


def make_processor(response=None, policy=None, side_effect=None):
    client = MagicMock()
    if side_effect is not None:
        client.models.generate_content.side_effect = side_effect
    else:
        client.models.generate_content.return_value = response
    processor = GeminiPageProcessor(
        client,
        "gemini-flash-latest",
        MagicMock(),
        policy or PagePolicy(user_prompt="Transcribe."),
        console=quiet_console(),
    )
    return processor, client


# ---------------------------------------------------------------------------
# finish_reason handling
# ---------------------------------------------------------------------------

def test_normal_page_returns_text():
    processor, _ = make_processor(make_response("page text"))
    assert processor.process_page_inline(b"%PDF", 1) == "page text"


def test_max_tokens_salvages_partial_text():
    """HTR used to discard truncated pages; partial transcription is valuable."""
    processor, _ = make_processor(make_response("half a page", "FinishReason.MAX_TOKENS"))

    result = processor.process_page_inline(b"%PDF", 1)

    assert result == "half a page" + TRUNCATION_MARKER


def test_max_tokens_with_nothing_recoverable_skips_page():
    processor, _ = make_processor(make_response(None, "FinishReason.MAX_TOKENS"))
    assert processor.process_page_inline(b"%PDF", 1) is None


def test_recitation_without_hook_skips_page():
    processor, _ = make_processor(make_response(None, "FinishReason.RECITATION"))
    assert processor.process_page_inline(b"%PDF", 1) is None


def test_recitation_hook_can_recover_the_page():
    """HTR's reframed-prompt fallback survives as the on_blocked hook."""
    policy = PagePolicy(user_prompt="Transcribe.", on_blocked=lambda content, num: "recovered")
    processor, _ = make_processor(make_response(None, "FinishReason.RECITATION"), policy=policy)

    assert processor.process_page_inline(b"%PDF", 1) == "recovered"


# ---------------------------------------------------------------------------
# inline vs upload routing
# ---------------------------------------------------------------------------

def test_large_pages_skip_the_inline_path():
    """HTR gated on 20MB — above the API's own cap, so oversized pages 400'd."""
    processor, client = make_processor(make_response("text"))
    oversized = b"x" * (INLINE_REQUEST_LIMIT_BYTES + 1)

    with patch.object(processor, "process_page_inline") as inline, \
         patch.object(processor, "process_page_upload", return_value="uploaded") as upload:
        assert processor.process_page(oversized, 1) == "uploaded"

    inline.assert_not_called()
    upload.assert_called_once()


def test_small_pages_try_inline_first():
    processor, _ = make_processor(make_response("text"))

    with patch.object(processor, "process_page_inline", return_value="inline") as inline, \
         patch.object(processor, "process_page_upload") as upload:
        assert processor.process_page(b"%PDF", 1) == "inline"

    inline.assert_called_once()
    upload.assert_not_called()


def test_failed_inline_falls_back_to_upload():
    processor, _ = make_processor(make_response("text"))

    with patch.object(processor, "process_page_inline", return_value=None), \
         patch.object(processor, "process_page_upload", return_value="uploaded") as upload:
        assert processor.process_page(b"%PDF", 1) == "uploaded"

    upload.assert_called_once()


def test_media_resolution_is_set_on_the_part():
    """ULTRA_HIGH is only accepted per-Part, not in GenerateContentConfig."""
    policy = PagePolicy(user_prompt="Transcribe.", media_resolution="ULTRA_HIGH")
    processor, _ = make_processor(make_response("text"), policy=policy)

    part = processor._build_part(b"%PDF")

    assert part.media_resolution.level.value == "MEDIA_RESOLUTION_ULTRA_HIGH"


# ---------------------------------------------------------------------------
# Whole-PDF assembly
# ---------------------------------------------------------------------------

def test_pages_joined_with_markers_and_no_header_on_first():
    joined = _join_pages([(1, "first"), (3, "third")])
    assert joined == "first\n\n--- Page 3 ---\n\nthird"


def test_no_output_file_when_every_page_fails(tmp_path):
    """HTR wrote [ERROR: ...] placeholders, then judged success by file size.

    A file of nothing but error markers counted as processed — and an 03 step
    would have uploaded those markers to Omeka as page content.
    """
    processor, _ = make_processor(make_response(None, "FinishReason.RECITATION"))
    output = tmp_path / "doc.txt"

    with patch("common.gemini_page_processor.PdfPageSource", return_value=fake_page_source(2)):
        result = processor.process_pdf(tmp_path / "doc.pdf", output)

    assert not output.exists()
    assert result.ok is False
    assert result.failed_pages == [1, 2]


def test_partial_success_writes_only_good_pages(tmp_path):
    processor, _ = make_processor(
        None,
        side_effect=[make_response("good"), make_response(None, "FinishReason.RECITATION")],
    )
    output = tmp_path / "doc.txt"

    with patch("common.gemini_page_processor.PdfPageSource", return_value=fake_page_source(2)), \
         patch.object(processor, "process_page_upload", return_value=None):
        result = processor.process_pdf(tmp_path / "doc.pdf", output)

    assert output.read_text(encoding="utf-8") == "good"
    assert result.successful_pages == 1
    assert result.failed_pages == [2]
    assert result.ok is True


def test_quota_exhaustion_saves_partial_then_raises(tmp_path):
    processor, _ = make_processor(None)
    output = tmp_path / "doc.txt"

    def pages(page_bytes, page_num):
        if page_num == 1:
            return "first page"
        raise QuotaExhaustedError("daily quota")

    with patch("common.gemini_page_processor.PdfPageSource", return_value=fake_page_source(3)), \
         patch.object(processor, "process_page", side_effect=pages):
        with pytest.raises(QuotaExhaustedError):
            processor.process_pdf(tmp_path / "doc.pdf", output)

    # Partial results must survive the abort.
    assert output.read_text(encoding="utf-8") == "first page"


def test_direct_multimodal_client_has_finite_timeout(monkeypatch):
    constructor = MagicMock()
    monkeypatch.setattr(gemini_utils.genai, "Client", constructor)

    gemini_utils.build_gemini_client("test-key", timeout_seconds=45)

    kwargs = constructor.call_args.kwargs
    assert kwargs["api_key"] == "test-key"
    assert kwargs["http_options"].timeout == 45_000


def test_direct_multimodal_client_rejects_nonpositive_timeout():
    with pytest.raises(ValueError, match="positive"):
        gemini_utils.build_gemini_client("test-key", timeout_seconds=0)


def uploaded_file(state="ACTIVE", name="files/123"):
    uploaded = MagicMock()
    uploaded.name = name
    uploaded.state.name = state
    return uploaded


def test_upload_bytes_requires_mime_type():
    with pytest.raises(ValueError, match="mime_type"):
        gemini_utils.upload_and_wait_active(MagicMock(), b"data")


def test_active_upload_returns_without_polling():
    client = MagicMock()
    uploaded = uploaded_file()
    client.files.upload.return_value = uploaded

    assert gemini_utils.upload_and_wait_active(
        client, b"data", mime_type="audio/mpeg",
    ) is uploaded
    client.files.get.assert_not_called()


def test_failed_upload_is_deleted_before_error():
    client = MagicMock()
    uploaded = uploaded_file("FAILED")
    client.files.upload.return_value = uploaded

    with pytest.raises(RuntimeError, match="FAILED"):
        gemini_utils.upload_and_wait_active(
            client, b"data", mime_type="audio/mpeg",
        )

    client.files.delete.assert_called_once_with(name="files/123")


def test_processing_upload_honors_zero_wait_budget():
    client = MagicMock()
    uploaded = uploaded_file("PROCESSING")
    client.files.upload.return_value = uploaded

    with pytest.raises(TimeoutError, match="not ACTIVE"):
        gemini_utils.upload_and_wait_active(
            client, "recording.mp3", max_wait=0,
        )

    client.files.get.assert_not_called()
    client.files.delete.assert_called_once_with(name="files/123")
