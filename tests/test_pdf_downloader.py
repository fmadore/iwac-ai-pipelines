"""Tests for common.pdf_downloader against a mocked Omeka client.

The ``.part`` download semantics moved to test_downloader.py when the routine
was extracted into common/downloader.py; what remains here is the Omeka-facing
half — media discovery, the item-set loop, and the resource-class filters.
"""

import io
from unittest.mock import MagicMock, patch

from rich.console import Console

from common.pdf_downloader import PDFDownloader, download_pdfs_from_item_set


def quiet_console():
    """Console that swallows output.

    An in-memory buffer rather than ``/dev/null``: the latter does not exist
    on Windows, and opening it leaked a file handle per test.
    """
    return Console(file=io.StringIO(), force_terminal=False)


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
