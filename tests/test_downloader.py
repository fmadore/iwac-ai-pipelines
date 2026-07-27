"""Tests for common.downloader — the .part temp-file download semantics.

These moved here from test_pdf_downloader.py when the routine was extracted out
of PDFDownloader so the audio/video downloader could share it. The behaviour
they pin is the reason the module exists: these pipelines re-run against the
same output directory, and a half-finished transfer must never be left at the
final path where the next run would treat it as done.
"""

from unittest.mock import MagicMock, patch

import requests

import common.downloader as downloader
from common.downloader import stream_download


def make_response(chunks, headers=None, status=200):
    """Build a mock streaming requests response usable as a context manager."""
    resp = MagicMock()
    resp.iter_content.return_value = iter(chunks)
    resp.headers = headers or {}
    resp.status_code = status
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    else:
        resp.raise_for_status.return_value = None
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def test_streams_via_part_file_and_renames(tmp_path):
    target = tmp_path / "123.pdf"
    part = tmp_path / "123.pdf.part"
    seen = {}

    def iter_content(chunk_size):
        yield b"%PDF-"
        # Mid-stream, data goes to the .part file — never to the final path.
        seen["part_exists"] = part.exists()
        seen["target_exists"] = target.exists()
        yield b"body"

    resp = make_response([])
    resp.iter_content.side_effect = iter_content

    with patch.object(downloader.requests, "get", return_value=resp):
        result = stream_download("http://x/123.pdf", target)

    assert seen == {"part_exists": True, "target_exists": False}
    assert result == target
    assert target.read_bytes() == b"%PDF-body"
    assert not part.exists()


def test_truncated_download_removes_part_and_returns_none(tmp_path):
    target = tmp_path / "123.pdf"
    resp = make_response([b"%PDF-", b"trunc"], headers={"Content-Length": "999999"})

    with patch.object(downloader.requests, "get", return_value=resp):
        result = stream_download("http://x/123.pdf", target)

    assert result is None
    assert not target.exists()
    assert not (tmp_path / "123.pdf.part").exists()


def test_http_error_leaves_no_files(tmp_path):
    target = tmp_path / "123.pdf"
    resp = make_response([], status=404)

    with patch.object(downloader.requests, "get", return_value=resp):
        result = stream_download("http://x/123.pdf", target)

    assert result is None
    assert not target.exists()
    assert not (tmp_path / "123.pdf.part").exists()


def test_dropped_connection_leaves_no_partial_at_final_path(tmp_path):
    """A failed run must never leave a partial file at the final path."""
    target = tmp_path / "42.pdf"

    def explode(*args, **kwargs):
        raise requests.ConnectionError("connection dropped")

    with patch.object(downloader.requests, "get", side_effect=explode):
        assert stream_download("http://x/42.pdf", target) is None

    assert not target.exists()


def test_timeout_is_handled_and_cleaned_up(tmp_path):
    """Large media downloads time out; that must not leave a .part behind."""
    target = tmp_path / "big.mp4"

    def timeout(*args, **kwargs):
        raise requests.Timeout("too slow")

    with patch.object(downloader.requests, "get", side_effect=timeout):
        assert stream_download("http://x/big.mp4", target, timeout=300) is None

    assert not target.exists()
    assert not (tmp_path / "big.mp4.part").exists()


def test_pdf_downloader_delegates_here(tmp_path):
    """PDFDownloader.download_pdf is now a thin wrapper — keep it wired up."""
    from common.pdf_downloader import PDFDownloader

    target = tmp_path / "7.pdf"
    resp = make_response([b"%PDF-ok"])

    with patch.object(downloader.requests, "get", return_value=resp):
        assert PDFDownloader.download_pdf("http://x/7.pdf", target) == target

    assert target.read_bytes() == b"%PDF-ok"
