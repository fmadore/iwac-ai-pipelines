"""
Shared PDF utilities for pipelines that process documents page by page.

Used by OCR extraction, HTR extraction, and magazine article extraction.

Usage:
    from common.pdf_utils import extract_pdf_page, get_pdf_page_count

    count = get_pdf_page_count(path)
    for i in range(count):
        page_bytes = extract_pdf_page(path, i)
"""

import io
import logging
from pathlib import Path

from PyPDF2 import PdfReader, PdfWriter

LOGGER = logging.getLogger(__name__)


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

    Returns a minimal single-page PDF document.
    """
    try:
        reader = PdfReader(str(pdf_path))
        writer = PdfWriter()
        writer.add_page(reader.pages[page_number])
        buf = io.BytesIO()
        writer.write(buf)
        buf.seek(0)
        return buf.getvalue()
    except Exception as exc:
        LOGGER.error(
            "Error extracting page %d from %s: %s",
            page_number + 1,
            pdf_path,
            exc,
        )
        raise
