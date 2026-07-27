"""
Shared PDF utilities for pipelines that process documents page by page.

Used by OCR extraction, HTR extraction, and magazine article extraction.

Prefer :class:`PdfPageSource` for page loops: :func:`extract_pdf_page` re-parses
the entire source document on every call, so extracting N pages that way costs N
full parses. On a 100-page magazine that dominates the local runtime.

Usage:
    from common.pdf_utils import PdfPageSource

    source = PdfPageSource(path)
    for index in range(len(source)):
        page_bytes = source.page_bytes(index)
"""

import io
import logging
from pathlib import Path

from pypdf import PdfReader, PdfWriter

LOGGER = logging.getLogger(__name__)


def _single_page_bytes(reader: PdfReader, page_number: int) -> bytes:
    """Serialize one page of an open reader as a minimal single-page PDF."""
    writer = PdfWriter()
    writer.add_page(reader.pages[page_number])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class PdfPageSource:
    """Serve single-page PDF bytes from a document parsed once.

    The per-page output is identical to :func:`extract_pdf_page`; the difference
    is that the source document is parsed a single time rather than once per
    page.
    """

    def __init__(self, pdf_path: Path) -> None:
        self.pdf_path = Path(pdf_path)
        try:
            self._reader = PdfReader(str(pdf_path))
        except Exception as exc:
            LOGGER.error("Error reading %s: %s", pdf_path, exc)
            raise

    def __len__(self) -> int:
        return len(self._reader.pages)

    def page_bytes(self, page_index: int) -> bytes:
        """Extract a single page as bytes (0-indexed)."""
        try:
            return _single_page_bytes(self._reader, page_index)
        except Exception as exc:
            LOGGER.error(
                "Error extracting page %d from %s: %s", page_index + 1, self.pdf_path, exc
            )
            raise


def get_pdf_page_count(pdf_path: Path) -> int:
    """Return the number of pages in a PDF file."""
    try:
        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception as exc:
        LOGGER.error("Error reading page count from %s: %s", pdf_path, exc)
        raise


def extract_pdf_page(pdf_path: Path, page_number: int) -> bytes:
    """Extract a single page from a PDF as bytes (0-indexed).

    Returns a minimal single-page PDF document. For more than one page from the
    same file, use :class:`PdfPageSource` instead — this function re-parses the
    whole document on every call.
    """
    try:
        return _single_page_bytes(PdfReader(str(pdf_path)), page_number)
    except Exception as exc:
        LOGGER.error(
            "Error extracting page %d from %s: %s",
            page_number + 1,
            pdf_path,
            exc,
        )
        raise
