"""PdfPageSource parses a document once and serves each page as its own PDF."""

import io

import pytest
from pypdf import PdfReader, PdfWriter

from common import pdf_utils
from common.pdf_utils import PdfPageSource, get_pdf_page_count


@pytest.fixture
def three_pages(tmp_path):
    writer = PdfWriter()
    for width in (100, 200, 300):
        writer.add_blank_page(width=width, height=400)
    path = tmp_path / "issue.pdf"
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def test_page_count_and_len_agree(three_pages):
    assert get_pdf_page_count(three_pages) == 3
    assert len(PdfPageSource(three_pages)) == 3


def test_each_page_is_served_as_a_one_page_pdf_in_order(three_pages):
    source = PdfPageSource(three_pages)
    widths = []
    for index in range(len(source)):
        pages = PdfReader(io.BytesIO(source.page_bytes(index))).pages
        assert len(pages) == 1
        widths.append(int(pages[0].mediabox.width))
    assert widths == [100, 200, 300]


def test_the_document_is_parsed_once_however_many_pages_are_read(three_pages, monkeypatch):
    opened = []
    real_reader = pdf_utils.PdfReader

    def counting_reader(*args, **kwargs):
        opened.append(args)
        return real_reader(*args, **kwargs)

    monkeypatch.setattr(pdf_utils, "PdfReader", counting_reader)
    source = PdfPageSource(three_pages)
    for index in range(3):
        source.page_bytes(index)
    assert len(opened) == 1


def test_a_page_outside_the_document_raises(three_pages):
    with pytest.raises(IndexError):
        PdfPageSource(three_pages).page_bytes(3)


def test_an_unreadable_file_raises_instead_of_reporting_zero_pages(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    with pytest.raises(Exception):  # noqa: B017 — pypdf's error class varies by release
        PdfPageSource(broken)
    with pytest.raises(Exception):  # noqa: B017
        get_pdf_page_count(broken)
