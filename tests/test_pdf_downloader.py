"""Tests for common.pdf_downloader against mocked Omeka client and HTTP layer."""

from unittest.mock import MagicMock, patch

import requests
from rich.console import Console

import common.pdf_downloader as pdf_downloader
from common.pdf_downloader import PDFDownloader, download_pdfs_from_item_set


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


def quiet_console():
    return Console(file=open("/dev/null", "w"), force_terminal=False)


# ---------------------------------------------------------------------------
# download_pdf: .part temp file semantics
# ---------------------------------------------------------------------------

def test_download_pdf_streams_via_part_file_and_renames(tmp_path):
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

    with patch.object(pdf_downloader.requests, "get", return_value=resp):
        result = PDFDownloader.download_pdf("http://x/123.pdf", target)

    assert seen == {"part_exists": True, "target_exists": False}
    assert result == target
    assert target.read_bytes() == b"%PDF-body"
    assert not part.exists()


def test_download_pdf_failure_removes_part_and_returns_none(tmp_path):
    target = tmp_path / "123.pdf"
    resp = make_response([b"%PDF-", b"trunc"], headers={"Content-Length": "999999"})

    with patch.object(pdf_downloader.requests, "get", return_value=resp):
        result = PDFDownloader.download_pdf("http://x/123.pdf", target)

    assert result is None
    assert not target.exists()
    assert not (tmp_path / "123.pdf.part").exists()


def test_download_pdf_http_error_leaves_no_files(tmp_path):
    target = tmp_path / "123.pdf"
    resp = make_response([], status=404)

    with patch.object(pdf_downloader.requests, "get", return_value=resp):
        result = PDFDownloader.download_pdf("http://x/123.pdf", target)

    assert result is None
    assert not target.exists()
    assert not (tmp_path / "123.pdf.part").exists()


def test_truncated_leftover_is_not_mistaken_for_complete_download(tmp_path):
    """A failed run must never leave a partial file at the final path."""
    target = tmp_path / "42.pdf"

    def explode(*args, **kwargs):
        raise requests.ConnectionError("connection dropped")

    with patch.object(pdf_downloader.requests, "get", side_effect=explode):
        assert PDFDownloader.download_pdf("http://x/42.pdf", target) is None

    assert not target.exists()


# ---------------------------------------------------------------------------
# create_valid_filename
# ---------------------------------------------------------------------------

def test_create_valid_filename_uses_item_id():
    assert PDFDownloader.create_valid_filename({"o:id": 4711}) == "4711.pdf"


# ---------------------------------------------------------------------------
# process_item
# ---------------------------------------------------------------------------

def test_process_item_handles_get_item_returning_none(tmp_path):
    client = MagicMock()
    client.get_item.return_value = None  # e.g. HTTP error swallowed by client
    downloader = PDFDownloader(client, tmp_path)

    assert downloader.process_item({"o:id": 1}) is None
    client.get_resource.assert_not_called()


def test_process_item_without_pdf_media_returns_none(tmp_path):
    client = MagicMock()
    client.get_item.return_value = {"o:id": 1, "o:media": []}
    downloader = PDFDownloader(client, tmp_path)

    assert downloader.process_item({"o:id": 1}) is None


def test_process_item_downloads_and_suffixes_multiple_pdfs(tmp_path):
    client = MagicMock()
    client.get_item.return_value = {
        "o:id": 7,
        "o:media": [{"@id": "http://x/media/1"}, {"@id": "http://x/media/2"}],
    }
    client.get_resource.side_effect = [
        {"o:source": "a.pdf", "o:original_url": "http://x/a.pdf"},
        {"o:source": "b.pdf", "o:original_url": "http://x/b.pdf"},
    ]
    downloader = PDFDownloader(client, tmp_path)

    with patch.object(PDFDownloader, "download_pdf", side_effect=lambda url, path: path):
        result = downloader.process_item({"o:id": 7})

    assert result is not None
    item_id, files = result
    assert item_id == 7
    assert files == f"{tmp_path / '7_1.pdf'}|{tmp_path / '7_2.pdf'}"


def test_process_item_all_downloads_failed_counts_as_failure(tmp_path):
    client = MagicMock()
    client.get_item.return_value = {"o:id": 7, "o:media": [{"@id": "http://x/media/1"}]}
    client.get_resource.return_value = {"o:source": "a.pdf", "o:original_url": "http://x/a.pdf"}
    downloader = PDFDownloader(client, tmp_path)

    with patch.object(PDFDownloader, "download_pdf", return_value=None):
        assert downloader.process_item({"o:id": 7}) is None


# ---------------------------------------------------------------------------
# download_pdfs_from_item_set
# ---------------------------------------------------------------------------

def make_set_client(items):
    client = MagicMock()
    client.base_url = "https://example.org/api"
    client.get_items.return_value = items
    return client


def test_item_set_loop_counts_successes_and_failures(tmp_path):
    items = [{"o:id": 1}, {"o:id": 2}, {"o:id": 3}]
    client = make_set_client(items)

    def fake_process_item(self, item):
        if item["o:id"] == 2:
            return None  # failed item
        return item["o:id"], f"{item['o:id']}.pdf"

    with patch.object(PDFDownloader, "process_item", fake_process_item):
        stats = download_pdfs_from_item_set(
            client, "99", tmp_path / "PDF", console=quiet_console()
        )

    assert stats == {"downloaded": 2, "failed": 1, "total_items": 3}
    assert (tmp_path / "PDF").is_dir()
    client.get_items.assert_called_once_with(99)


def test_item_set_loop_passes_resource_class_filter(tmp_path):
    client = make_set_client([])
    stats = download_pdfs_from_item_set(
        client, 5, tmp_path, resource_class_id=60, console=quiet_console()
    )
    assert stats == {"downloaded": 0, "failed": 0, "total_items": 0}
    client.get_items.assert_called_once_with(5, resource_class_id=60)


def test_item_set_loop_applies_type_backstop(tmp_path):
    items = [
        {"o:id": 1, "@type": ["o:Item", "bibo:Issue"]},
        {"o:id": 2, "@type": ["o:Item"]},  # must be dropped by the backstop
    ]
    client = make_set_client(items)

    processed = []

    def fake_process_item(self, item):
        processed.append(item["o:id"])
        return item["o:id"], "x.pdf"

    with patch.object(PDFDownloader, "process_item", fake_process_item):
        stats = download_pdfs_from_item_set(
            client, 5, tmp_path,
            resource_class_id=60,
            required_class_term="bibo:Issue",
            console=quiet_console(),
        )

    assert processed == [1]
    assert stats == {"downloaded": 1, "failed": 0, "total_items": 1}
