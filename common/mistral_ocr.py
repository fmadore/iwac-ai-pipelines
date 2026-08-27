"""Shared Mistral Document AI (OCR) plumbing.

Two pipelines now send PDFs to Mistral's dedicated OCR endpoint — the newspaper
scans in ``AI_ocr_extraction`` and the scholarly publications in
``AI_publication_extraction`` — so the Markdown normalisation, the upload /
signed-URL / cleanup dance, and the oversized-file split live here rather than
being copied between them.

What the two pipelines do *not* share is what they keep. A newspaper page's
running head is furniture; a journal article's page foot is where the footnotes
are. That decision stays in the caller: this module hands back typed blocks and
offers :func:`classify_blocks` to label them, but never silently drops text.

The model id is **pinned**. ``mistral-ocr-latest`` currently resolves to 4.1,
but it is a rolling alias — and whatever ran is what ``03`` stamps into an
``iwac:ocrModel`` annotation. The same reasoning retired ``gemini-flash-latest``
as an annotation key; see ``iwac_config`` for that history.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from common.rate_limiter import RateLimiter, QuotaExhaustedError, is_mistral_quota_exhausted
from common.retry import retry_with_backoff

LOGGER = logging.getLogger(__name__)

#: Pinned OCR release. ``mistral-ocr-latest`` aliases to this today; the alias
#: is deliberately not used, because a run that cannot name its model cannot be
#: cited. Verified against ``client.models.list()`` on 2026-08-18.
MISTRAL_OCR_MODEL = "mistral-ocr-4-1"

#: Hard API limits. A file over the size cap is rejected outright, so
#: :class:`MistralOcrClient` splits it by page range rather than failing.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_PAGES_PER_DOCUMENT = 1000

#: Safety margin for the split: pypdf's per-part overhead is small but real, and
#: a part that comes back over the cap costs a whole round trip to discover.
SPLIT_TARGET_BYTES = 45 * 1024 * 1024

#: The structural labels OCR 4+ assigns to blocks. Declared so an unexpected
#: label from a future release is visible as such instead of silently joining
#: the body text.
BLOCK_LABELS = frozenset({
    "text", "title", "list", "table", "image", "equation", "caption",
    "code", "references", "aside_text", "header", "footer", "signature",
})

#: Labels whose content is prose the archive wants, in reading order.
BODY_LABELS = frozenset({
    "text", "title", "list", "table", "equation", "caption", "code", "signature",
})

#: Labels that carry scholarly apparatus — footnotes and bibliography. On this
#: corpus footnotes come back as ``footer``, not ``aside_text`` (measured on a
#: 33-page Cahiers du CERLESHS article: 0 ``aside_text``, 53 substantive
#: ``footer`` blocks totalling 7,388 characters), which is why ``footer`` needs
#: the page-furniture test below rather than a blanket drop.
APPARATUS_LABELS = frozenset({"references", "aside_text"})

# HTTP statuses worth retrying (transient rate limit / server errors).
TRANSIENT_STATUS = {429, 500, 502, 503, 504}


# --- Markdown -> plain text normalisation ---------------------------------
# Mistral OCR returns Markdown (headings, emphasis, tables, and image
# placeholders such as ``![img-0.jpeg](img-0.jpeg)``). For an archival
# full-text field consumed by search / NER / embeddings, that formatting is
# noise, so we strip the syntax while preserving the underlying text.

_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")          # ![alt](url)  -> removed
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")        # [text](url)  -> text
_BOLD_RE = re.compile(r"(\*\*|__)(.+?)\1")             # **t** / __t__ -> t
_ITALIC_RE = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])")  # *t* -> t
_CODE_RE = re.compile(r"`([^`]+)`")                    # `t`          -> t
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+")          # ## Heading   -> Heading
_HR_RE = re.compile(r"^\s*([-*_])\1{2,}\s*$")          # --- *** ___  -> removed
_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?")           # > quote      -> quote
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")

# HTML tables (``table_format="html"``) reach the plain-text path as markup.
_HTML_ROW_END_RE = re.compile(r"</(?:tr|thead|tbody|table)\s*>", re.I)
_HTML_CELL_END_RE = re.compile(r"</(?:td|th)\s*>", re.I)
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Mistral OCR sometimes renders footnote superscripts / math as inline LaTeX
# (e.g. ``$^{7}$``, ``XX$^{e}$``, ``$_{2}$``). Convert these to Unicode
# super/subscripts so they match the document's other footnote markers
# (¹ ² ³ …). Only ``$…$`` spans containing \, ^ or _ are treated as math, so
# literal text (and stray dollar signs) is left untouched.
_AUTOLINK_RE = re.compile(r"<(https?://[^>\s]+)>")          # <https://x> -> https://x
_INLINE_MATH_RE = re.compile(r"\$([^$\n]*[\\^_][^$\n]*)\$")
_SUP_MAP = str.maketrans("0123456789e+-()n", "⁰¹²³⁴⁵⁶⁷⁸⁹ᵉ⁺⁻⁽⁾ⁿ")
_SUB_MAP = str.maketrans("0123456789+-()", "₀₁₂₃₄₅₆₇₈₉₊₋₍₎")


def _delatex(match: "re.Match[str]") -> str:
    """Render an inline-LaTeX span (``$…$`` content) as readable plain text."""
    inner = match.group(1)
    inner = re.sub(r"\^\{([^}]*)\}", lambda m: m.group(1).translate(_SUP_MAP), inner)
    inner = re.sub(r"_\{([^}]*)\}", lambda m: m.group(1).translate(_SUB_MAP), inner)
    inner = re.sub(r"\\[a-zA-Z]+\s*", "", inner)            # drop \mathrm, \text, …
    return inner.replace("^", "").replace("_", "").replace("{", "").replace("}", "")


def _html_table_to_text(fragment: str) -> str:
    """Flatten an HTML table to tab-separated rows, one row per line."""
    text = _HTML_CELL_END_RE.sub("\t", fragment)
    text = _HTML_ROW_END_RE.sub("\n", text)
    text = _HTML_TAG_RE.sub("", text)
    lines = [re.sub(r"[ \t]+", "\t", ln).strip("\t ") for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def markdown_to_plain_text(md: str) -> str:
    """Convert Mistral OCR Markdown to clean plain text.

    Strips headings, emphasis, inline code, links, horizontal rules and image
    placeholders; flattens Markdown *and* HTML tables to tab-separated rows.
    The result matches the plain-text convention used by the Gemini OCR path,
    which the project standardises on for ``bibo:content``.
    """
    if not md:
        return ""

    if "<table" in md.lower():
        md = re.sub(
            r"<table.*?</table\s*>",
            lambda m: _html_table_to_text(m.group(0)),
            md,
            flags=re.I | re.S,
        )

    out_lines: List[str] = []
    in_code_fence = False
    for raw_line in md.splitlines():
        line = raw_line

        # Fenced code blocks (```): drop the fences, keep the inner text.
        if line.lstrip().startswith("```"):
            in_code_fence = not in_code_fence
            continue

        # Drop horizontal rules and Markdown table separator rows.
        if _HR_RE.match(line) or _TABLE_SEP_RE.match(line):
            continue

        # Strip heading and blockquote markers (keep the text).
        line = _HEADER_RE.sub("", line)
        line = _BLOCKQUOTE_RE.sub("", line)

        # Table content row -> tab-separated cells.
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            line = "\t".join(cells)

        out_lines.append(line)

    text = "\n".join(out_lines)

    # Inline elements (images first, so links don't eat the alt text of ![]()).
    text = _IMG_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _BOLD_RE.sub(r"\2", text)
    text = _ITALIC_RE.sub(r"\1", text)
    text = _CODE_RE.sub(r"\1", text)
    text = _AUTOLINK_RE.sub(r"\1", text)   # unwrap <https://…> autolinks
    text = _INLINE_MATH_RE.sub(_delatex, text)  # $^{7}$ -> ⁷, XX$^{e}$ -> XXᵉ

    # Normalise whitespace: trim trailing spaces, collapse 3+ blank lines.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _MULTI_BLANK_RE.sub("\n\n", text)

    return text.strip()


# --- Block classification --------------------------------------------------

#: A page number, with or without decoration: ``12``, ``- 12 -``, ``[iv]``.
_PAGE_NUMBER_RE = re.compile(
    r"^[\s\-–—\[\]()|.,·*]*[0-9ivxlcdmIVXLCDM]{1,7}[\s\-–—\[\]()|.,·*]*$"
)


def is_page_number(text: str) -> bool:
    """True when a header/footer holds nothing but a folio number."""
    stripped = (text or "").strip()
    return bool(stripped) and bool(_PAGE_NUMBER_RE.match(stripped))


#: A running head is a title or a byline; a footnote is a sentence or a
#: citation. Text longer than this is never treated as furniture — a long
#: string repeating across pages is likelier a block quote than a running head,
#: and keeping it costs a little noise where dropping it costs content.
MAX_RUNNING_HEAD_CHARS = 120


def _normalise_for_repeat(text: str) -> str:
    """Fold a running head to a comparable key (case and punctuation removed).

    Digits are deliberately **kept**. Folding them would let "Titre — 12" and
    "Titre — 13" match, which is the one thing this comparison cannot do — but
    it would equally collapse "Ibid., p. 12" and "Ibid., p. 45" onto a single
    key, and a thesis whose notes share that boilerplate would have them
    deleted as furniture. The two failure modes are not symmetrical: an
    unmatched running head survives into the text as a stray title line, while
    a matched footnote is gone. Folio numbers, the case this would have caught,
    are already excluded by :func:`is_page_number` before the comparison.
    """
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def find_running_furniture(
    pages: Sequence[Dict[str, Any]],
    *,
    min_share: float = 0.3,
    min_pages: int = 3,
    max_chars: int = MAX_RUNNING_HEAD_CHARS,
) -> set:
    """Identify header/footer text that repeats across the document.

    A running head is the same string on page after page; a footnote is unique
    to its page. So repetition — not position — is what separates furniture
    from apparatus.

    Only text shorter than *max_chars* is eligible, which is what keeps a
    numbered footnote out of this test even when digit-folding makes several of
    them look alike.

    Returns the set of normalised keys seen on at least *min_share* of pages
    (and at least *min_pages* pages, so a three-page extract is not pruned on
    the strength of two coincidences).
    """
    total = len(pages)
    if total < min_pages:
        return set()

    counts: Dict[str, int] = {}
    for page in pages:
        seen_here = set()
        for block in page.get("blocks") or []:
            if block.get("type") not in {"header", "footer"}:
                continue
            content = (block.get("content") or "").strip()
            if not content or len(content) > max_chars or is_page_number(content):
                continue
            seen_here.add(_normalise_for_repeat(content))
        for key in seen_here:
            counts[key] = counts.get(key, 0) + 1

    threshold = max(min_pages, int(total * min_share))
    return {key for key, n in counts.items() if n >= threshold and key}


@dataclass
class ClassifiedBlock:
    """One OCR block with the role this pipeline assigns it."""

    page_index: int
    type: str
    content: str
    #: ``body`` (prose), ``apparatus`` (footnotes / bibliography) or
    #: ``furniture`` (running heads, folio numbers) — dropped from the text but
    #: kept in the sidecar so the decision stays auditable.
    role: str
    bbox: Optional[Dict[str, int]] = None

    @property
    def is_kept(self) -> bool:
        return self.role != "furniture"


def classify_blocks(
    pages: Sequence[Dict[str, Any]],
    *,
    keep_first_page_furniture: bool = True,
) -> List[ClassifiedBlock]:
    """Label every block ``body`` / ``apparatus`` / ``furniture``.

    ``header`` and ``footer`` are the only ambiguous labels, and they are
    resolved by evidence rather than by position:

    * a folio number is furniture,
    * text repeating across the document is a running head — furniture,
    * anything else in a page foot is apparatus (a footnote), because on
      scholarly PDFs that is overwhelmingly what it is.

    On page 1 the head and foot carry front matter — byline, journal citation,
    the ``*`` note on the title — so *keep_first_page_furniture* keeps them.
    """
    running = find_running_furniture(pages)
    out: List[ClassifiedBlock] = []

    for page in pages:
        index = page.get("index", 0)
        for block in page.get("blocks") or []:
            btype = block.get("type") or "text"
            content = block.get("content") or ""
            bbox = None
            if "top_left_x" in block:
                bbox = {
                    "top_left_x": block.get("top_left_x"),
                    "top_left_y": block.get("top_left_y"),
                    "bottom_right_x": block.get("bottom_right_x"),
                    "bottom_right_y": block.get("bottom_right_y"),
                }

            if btype in APPARATUS_LABELS:
                role = "apparatus"
            elif btype in {"header", "footer"}:
                first_page = index == 0 and keep_first_page_furniture
                if is_page_number(content):
                    role = "furniture"
                elif _normalise_for_repeat(content) in running:
                    role = "body" if first_page else "furniture"
                elif btype == "footer":
                    role = "apparatus"
                else:
                    role = "body" if first_page else "furniture"
            elif btype == "image":
                # An image block's "content" is a placeholder, not text.
                role = "furniture"
            else:
                role = "body"

            out.append(ClassifiedBlock(index, btype, content, role, bbox))

    return out


def render_plain_text(
    blocks: Iterable[ClassifiedBlock], *, page_markers: bool = True
) -> str:
    """Join kept blocks into the plain text written to ``bibo:content``.

    Blocks arrive in reading order, so footnotes already trail the body of
    their own page and no reordering is needed. Page markers match the
    convention the Gemini OCR path established (``--- Page N ---``).
    """
    by_page: Dict[int, List[str]] = {}
    for block in blocks:
        if not block.is_kept:
            continue
        text = markdown_to_plain_text(block.content)
        if text:
            by_page.setdefault(block.page_index, []).append(text)

    if not by_page:
        return ""

    chunks: List[str] = []
    for position, index in enumerate(sorted(by_page), start=1):
        body = "\n\n".join(by_page[index])
        if position == 1 or not page_markers:
            chunks.append(body)
        else:
            chunks.append(f"\n\n--- Page {index + 1} ---\n\n{body}")
    return "".join(chunks).strip()


# --- Oversized documents ---------------------------------------------------


@dataclass
class PdfPart:
    """A page range of a source PDF, small enough to upload on its own."""

    data: bytes
    first_page_index: int
    page_count: int


def split_pdf_for_upload(
    pdf_path: Path, *, target_bytes: int = SPLIT_TARGET_BYTES
) -> List[PdfPart]:
    """Split a PDF into parts under the upload cap, preserving page indices.

    Returns a single whole-file part when the document already fits, so callers
    can treat every document the same way. Scanned theses in this collection run
    to 2.3 MB per page, so the parts can be a handful of pages each.
    """
    from pypdf import PdfReader, PdfWriter  # local import: pypdf is heavy

    size = pdf_path.stat().st_size
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    if size <= target_bytes and total_pages <= MAX_PAGES_PER_DOCUMENT:
        return [PdfPart(pdf_path.read_bytes(), 0, total_pages)]

    # Estimate how many pages fit, then verify each part and halve any that
    # still overshoot. Estimating first avoids re-serialising the document once
    # per page, which on a 276 MB scan is minutes of pure CPU.
    per_page = max(1, size // max(1, total_pages))
    step = max(1, min(int(target_bytes // per_page), MAX_PAGES_PER_DOCUMENT))

    def serialise(start: int, count: int) -> bytes:
        writer = PdfWriter()
        for offset in range(count):
            writer.add_page(reader.pages[start + offset])
        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()

    parts: List[PdfPart] = []
    start = 0
    while start < total_pages:
        count = min(step, total_pages - start)
        while True:
            data = serialise(start, count)
            if len(data) <= MAX_UPLOAD_BYTES or count == 1:
                break
            count = max(1, count // 2)
        if len(data) > MAX_UPLOAD_BYTES:
            LOGGER.warning(
                "%s page %d alone is %.1f MB — over the %d MB cap; sending anyway",
                pdf_path.name, start + 1, len(data) / 1e6, MAX_UPLOAD_BYTES // 1024 // 1024,
            )
        parts.append(PdfPart(data, start, count))
        start += count

    return parts


# --- The endpoint ----------------------------------------------------------


def status_code(error: Exception) -> Optional[int]:
    """Best-effort extraction of an HTTP status code from a Mistral SDK error.

    Mistral's ``SDKError`` exposes the response as ``raw_response`` (an
    httpx.Response), so the status lives at ``error.raw_response.status_code``
    rather than on a ``.code`` attribute.
    """
    resp = getattr(error, "raw_response", None)
    code = getattr(resp, "status_code", None)
    if code is None:
        code = getattr(error, "status_code", None)
    return code


@dataclass
class OcrResult:
    """One document's OCR output, already stitched across parts."""

    pages: List[Dict[str, Any]]
    model: str
    pages_processed: int = 0
    parts: int = 1
    warnings: List[str] = field(default_factory=list)


class MistralOcrClient:
    """Wraps ``client.ocr.process`` with upload, retry, quota and splitting.

    Every request asks for structural blocks; what the caller keeps is its own
    business (see :func:`classify_blocks`).
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = MISTRAL_OCR_MODEL,
        requests_per_minute: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        from mistralai.client import Mistral  # local import: keeps CI import-light

        self.client = Mistral(api_key=api_key)
        self.model = model
        self.logger = logger or LOGGER
        self.rate_limiter = RateLimiter(requests_per_minute, logger=self.logger)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        """Retry transient errors only; quota exhaustion is terminal."""
        from mistralai.client.errors import SDKError

        if is_mistral_quota_exhausted(error):
            return False
        if isinstance(error, SDKError):
            return status_code(error) in TRANSIENT_STATUS
        return True

    def _process_part(self, part: PdfPart, name: str) -> List[Dict[str, Any]]:
        """Upload one part and OCR it, returning raw page dicts.

        Raises:
            QuotaExhaustedError: the account quota is dead — the batch should
                stop rather than burn retries on every remaining document.
        """
        from mistralai.client.errors import SDKError

        uploaded_id: Optional[str] = None
        signed_url: Optional[str] = None

        @retry_with_backoff(max_retries=3, base_delay=5.0, is_retryable=self._is_retryable)
        def _run():
            nonlocal uploaded_id, signed_url
            if uploaded_id is None:
                uploaded = self.client.files.upload(
                    file={"file_name": name, "content": part.data}, purpose="ocr",
                )
                if not uploaded or not uploaded.id:
                    raise RuntimeError("File upload returned no id")
                uploaded_id = uploaded.id
                signed_url = self.client.files.get_signed_url(file_id=uploaded_id).url

            self.rate_limiter.wait()
            response = self.client.ocr.process(
                model=self.model,
                document={"type": "document_url", "document_url": signed_url},
                include_blocks=True,       # OCR 4+: paragraph-level structure
                table_format="html",       # tables survive as markup, not soup
                extract_header=True,
                extract_footer=True,
            )
            if not response.pages:
                raise RuntimeError("OCR response contained no pages")
            return response

        try:
            response = _run()
        except SDKError as exc:
            code = status_code(exc)
            if is_mistral_quota_exhausted(exc) or code == 429:
                raise QuotaExhaustedError(str(exc)) from exc
            raise
        finally:
            if uploaded_id is not None:
                try:
                    self.client.files.delete(file_id=uploaded_id)
                except Exception:  # best effort — never mask the real error
                    pass

        pages: List[Dict[str, Any]] = []
        for page in response.pages:
            data = page.model_dump() if hasattr(page, "model_dump") else dict(page)
            # Re-base the page index onto the source document, so a split file
            # still reports the page numbers a reader would see.
            data["index"] = part.first_page_index + int(data.get("index", 0))
            pages.append(data)
        return pages

    # -- public ------------------------------------------------------------

    def process_pdf(self, pdf_path: Path) -> OcrResult:
        """OCR a whole PDF, splitting it first when it exceeds the upload cap."""
        parts = split_pdf_for_upload(pdf_path)
        warnings: List[str] = []
        if len(parts) > 1:
            warnings.append(
                f"{pdf_path.stat().st_size / 1e6:.0f} MB exceeds the "
                f"{MAX_UPLOAD_BYTES // 1024 // 1024} MB upload cap — "
                f"sent as {len(parts)} parts"
            )
            self.logger.info("Split %s into %d parts", pdf_path.name, len(parts))

        all_pages: List[Dict[str, Any]] = []
        for position, part in enumerate(parts, start=1):
            suffix = f"_part{position}" if len(parts) > 1 else ""
            all_pages.extend(self._process_part(part, f"{pdf_path.stem}{suffix}.pdf"))

        all_pages.sort(key=lambda p: p.get("index", 0))
        return OcrResult(
            pages=all_pages,
            model=self.model,
            pages_processed=len(all_pages),
            parts=len(parts),
            warnings=warnings,
        )
