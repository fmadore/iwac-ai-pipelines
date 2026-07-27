"""
Shared page-by-page Gemini PDF processing loop.

``AI_ocr_extraction/02_gemini_ocr_processor.py`` and
``AI_htr_extraction/gemini_htr_processor.py`` were the same machine with
different prompts: split a PDF into single-page documents, send each page
inline (falling back to the Files API when it is too large), retry only
transient failures, interpret ``finish_reason``, join the surviving pages with
``--- Page N ---`` markers, and abort the whole batch on quota exhaustion.

Keeping two copies let them drift, and each drift lost something:

- HTR gated inline requests on ``page_size_mb < 20`` — above the API's own cap,
  so oversized pages were sent inline and rejected. Both now use
  ``INLINE_REQUEST_LIMIT_BYTES``.
- HTR wrote ``[ERROR: ...]`` placeholders into the output file and then judged
  success by ``st_size > 100``, so a file of nothing but error markers counted
  as processed — and would have been uploaded to Omeka as page content by an
  ``03`` step. Failures are now recorded in the result, never in the text.
- OCR salvaged partial text on ``MAX_TOKENS``; HTR discarded the page. Both now
  salvage.

Provider-specific behaviour stays with the pipeline through :class:`PagePolicy`.

Usage:
    policy = PagePolicy(
        user_prompt="Transcribe this page.",
        on_blocked=my_recitation_fallback,   # optional
    )
    processor = GeminiPageProcessor(client, model_name, config, policy, console=console)
    result = processor.process_pdf(pdf_path, output_dir / f"{pdf_path.stem}.txt")
"""

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from google.genai import types
from google.genai import errors as genai_errors
from rich.console import Console

from common.gemini_utils import (
    INLINE_REQUEST_LIMIT_BYTES,
    delete_uploaded_file,
    extract_text_from_response,
    upload_and_wait_active,
)
from common.pdf_utils import PdfPageSource
from common.rate_limiter import QuotaExhaustedError, RateLimiter, is_quota_exhausted

LOGGER = logging.getLogger(__name__)

# Transient Gemini API errors worth retrying with backoff. Anything else — in
# particular programming errors — must surface immediately instead of being
# retried (a NameError was once retried with backoff here for months).
RETRYABLE_API_CODES = (429, 500, 503)

TRUNCATION_MARKER = "\n\n[... TRANSCRIPTION TRUNCATED - OUTPUT EXCEEDED MAX TOKENS ...]"


@dataclass
class PagePolicy:
    """The provider-specific parts of a page request.

    Attributes:
        user_prompt: Sent after the document part, per Google's guidance that
            the document should come first.
        media_resolution: Optional per-Part resolution. ``ULTRA_HIGH`` is only
            accepted here, not in ``GenerateContentConfig`` — which is what
            archival scans and handwriting want.
        on_blocked: Called when the model returns RECITATION. Receives the page
            content (Part or uploaded File) and page number, and returns
            recovered text or None. Defaults to skipping the page.
        max_retries: Attempts for the Files API path (including the first).
        base_delay: Initial backoff in seconds; doubles per attempt.
    """

    user_prompt: str
    media_resolution: Optional[str] = None
    on_blocked: Optional[Callable[[object, int], Optional[str]]] = None
    max_retries: int = 3
    base_delay: float = 5.0


@dataclass
class PdfResult:
    """Outcome of one PDF."""

    pdf_path: Path
    total_pages: int
    successful_pages: int = 0
    failed_pages: List[int] = field(default_factory=list)
    quota_exhausted: bool = False
    output_file: Optional[Path] = None
    output_size: int = 0

    @property
    def success_rate(self) -> float:
        return (self.successful_pages / self.total_pages * 100) if self.total_pages else 0.0

    @property
    def ok(self) -> bool:
        """True when at least one page was transcribed and a file was written.

        Deliberately not a size heuristic: a stale file from a previous run, or
        one full of error markers, must not count as success.
        """
        return self.successful_pages > 0 and self.output_file is not None


class GeminiPageProcessor:
    """Runs a PDF through Gemini one page at a time."""

    def __init__(
        self,
        client,
        model_name: str,
        generation_config,
        policy: PagePolicy,
        *,
        rate_limiter: Optional[RateLimiter] = None,
        console: Optional[Console] = None,
        logger: Optional[logging.Logger] = None,
        verbose: bool = False,
    ):
        self.client = client
        self.model_name = model_name
        self.generation_config = generation_config
        self.policy = policy
        self.rate_limiter = rate_limiter or RateLimiter(None)
        self.console = console or Console()
        self.logger = logger or LOGGER
        self.verbose = verbose

    # ---- Request helpers ---------------------------------------------------

    def _build_part(self, page_bytes: bytes):
        """Wrap page bytes as a PDF Part, with optional per-Part resolution."""
        if self.policy.media_resolution:
            return types.Part.from_bytes(
                data=page_bytes,
                mime_type="application/pdf",
                media_resolution=f"MEDIA_RESOLUTION_{self.policy.media_resolution.upper()}",
            )
        return types.Part.from_bytes(data=page_bytes, mime_type="application/pdf")

    def _generate(self, page_content):
        self.rate_limiter.wait()
        return self.client.models.generate_content(
            model=self.model_name,
            contents=[page_content, self.policy.user_prompt],  # document first, then prompt
            config=self.generation_config,
        )

    def _extract(self, response, page_content, page_num: int) -> Optional[str]:
        """Validate a response and pull out the text.

        Returns None when the page should be skipped without retrying
        (RECITATION with no recovery, or MAX_TOKENS with nothing salvageable).

        Raises:
            RuntimeError: when the response is structurally invalid or empty,
                so the caller can retry or fall back.
        """
        if not response.candidates:
            raise RuntimeError("No candidates in Gemini response")

        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason
        reason = str(finish_reason)

        if reason == "FinishReason.RECITATION":
            if self.policy.on_blocked is not None:
                self.console.print(
                    f"  [yellow]⚠[/] Page {page_num}: copyright detection triggered, trying alternative..."
                )
                recovered = self.policy.on_blocked(page_content, page_num)
                if recovered:
                    return recovered
            self.console.print(
                f"  [yellow]⚠ Page {page_num} skipped[/] - content blocked (potential copyrighted material)"
            )
            self.logger.warning("Page %d: RECITATION — Gemini blocked output", page_num)
            return None

        if reason == "FinishReason.MAX_TOKENS":
            # Partial text is still valuable for a transcription; keep it and
            # mark the cut rather than discarding the page.
            partial = extract_text_from_response(response)
            if partial:
                self.console.print(
                    f"  [yellow]⚠ Page {page_num} truncated[/] - output exceeded max tokens (partial text saved)"
                )
                self.logger.warning(
                    "Page %d: MAX_TOKENS — truncated but %d chars recovered", page_num, len(partial)
                )
                return partial + TRUNCATION_MARKER
            self.console.print(f"  [red]✗ Page {page_num}[/] - MAX_TOKENS with no recoverable text")
            self.logger.error("Page %d: MAX_TOKENS but no text could be extracted", page_num)
            return None

        if not candidate.content or not candidate.content.parts:
            raise RuntimeError(f"No valid response. Finish reason: {finish_reason}")

        text = extract_text_from_response(response)
        if not text:
            raise RuntimeError("Empty text response from Gemini")
        return text

    # ---- Per-page paths ----------------------------------------------------

    def process_page_inline(self, page_bytes: bytes, page_num: int) -> Optional[str]:
        """Send the page as inline bytes. Returns text, or None on failure."""
        try:
            if self.verbose:
                self.console.print(f"  └─ [cyan]📄[/] Processing page {page_num} inline...")
            # Build the Part once: the RECITATION fallback re-sends this exact part.
            part = self._build_part(page_bytes)
            return self._extract(self._generate(part), part, page_num)
        except QuotaExhaustedError:
            raise
        except genai_errors.APIError as exc:
            if is_quota_exhausted(exc):
                raise QuotaExhaustedError(str(exc)) from exc
            self._log_error(exc, "inline PDF processing", page_num)
            return None
        except (TimeoutError, ConnectionError, RuntimeError) as exc:
            self._log_error(exc, "inline PDF processing", page_num)
            return None

    def process_page_upload(self, page_bytes: bytes, page_num: int) -> Optional[str]:
        """Send the page via the Files API, retrying only transient failures.

        The upload is deleted after every attempt, success or failure: uploads
        expire after 48h, but leaked multi-GB files waste quota meanwhile.
        """
        for attempt in range(self.policy.max_retries):
            uploaded = None
            retryable = False
            try:
                try:
                    if self.verbose:
                        self.console.print(f"  └─ [cyan]⬆[/]  Uploading page {page_num}...")
                    uploaded = upload_and_wait_active(
                        self.client, page_bytes,
                        mime_type="application/pdf", max_wait=60, poll_interval=1.0,
                    )
                    return self._extract(self._generate(uploaded), uploaded, page_num)
                finally:
                    if uploaded is not None:
                        delete_uploaded_file(self.client, uploaded)

            except QuotaExhaustedError:
                raise
            except genai_errors.APIError as exc:
                if is_quota_exhausted(exc):
                    raise QuotaExhaustedError(str(exc)) from exc
                retryable = getattr(exc, "code", 0) in RETRYABLE_API_CODES
                self._log_error(
                    exc, f"upload PDF processing (attempt {attempt + 1}/{self.policy.max_retries})", page_num
                )
            except (TimeoutError, ConnectionError) as exc:
                retryable = True
                self._log_error(
                    exc, f"upload PDF processing (attempt {attempt + 1}/{self.policy.max_retries})", page_num
                )
            except RuntimeError as exc:
                # Empty/blocked responses and upload failures: not worth retrying.
                self._log_error(exc, "upload PDF processing", page_num)
                return None

            if not retryable:
                return None
            if attempt < self.policy.max_retries - 1:
                delay = self.policy.base_delay * (2 ** attempt) + random.uniform(0, 2)
                self.console.print(f"    [dim]Retrying in {delay:.1f}s...[/]")
                time.sleep(delay)
            else:
                self.console.print(f"  [red]✗[/] Page {page_num}: max retries reached.")

        return None

    def process_page(self, page_bytes: bytes, page_num: int) -> Optional[str]:
        """Try inline, then the Files API. Returns text or None."""
        text = None
        if len(page_bytes) <= INLINE_REQUEST_LIMIT_BYTES:
            text = self.process_page_inline(page_bytes, page_num)
        if not text:
            text = self.process_page_upload(page_bytes, page_num)
        return text

    # ---- Whole PDF ---------------------------------------------------------

    def process_pdf(
        self,
        pdf_path: Path,
        output_file: Path,
        *,
        progress=None,
    ) -> PdfResult:
        """Process every page and write the surviving text to *output_file*.

        Pages are buffered and the file is written only if at least one page
        succeeded, so a failed run never leaves a misleading output file behind.

        Raises:
            QuotaExhaustedError: after saving whatever completed, so the caller
                can stop the batch.
        """
        # Parse the document once, not once per page.
        page_source = PdfPageSource(pdf_path)
        total_pages = len(page_source)
        result = PdfResult(pdf_path=pdf_path, total_pages=total_pages)
        pages: List[Tuple[int, str]] = []

        page_task = progress.add_task("[dim]  Pages", total=total_pages) if progress else None

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            try:
                text = self.process_page(page_source.page_bytes(page_idx), page_num)
                if text and text.strip():
                    pages.append((page_num, text))
                    result.successful_pages += 1
                else:
                    result.failed_pages.append(page_num)

            except QuotaExhaustedError:
                remaining = total_pages - page_idx
                self.console.print(
                    f"  [red]✗ Quota exhausted![/] Completed {result.successful_pages}/{total_pages} "
                    f"pages, {remaining} remaining — stopping early"
                )
                self.logger.error(
                    "Quota exhausted during %s at page %d. %d pages completed, %d remaining.",
                    pdf_path.name, page_num, result.successful_pages, remaining,
                )
                result.quota_exhausted = True
                break

            except Exception as exc:
                result.failed_pages.append(page_num)
                self.logger.error("Error processing page %d of %s: %s", page_num, pdf_path, exc)

            if progress and page_task is not None:
                progress.update(page_task, advance=1)

        if progress and page_task is not None:
            progress.remove_task(page_task)

        if pages:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(_join_pages(pages), encoding="utf-8")
            result.output_file = output_file
            result.output_size = output_file.stat().st_size

        self._report(result)

        if result.quota_exhausted:
            raise QuotaExhaustedError("Daily quota exhausted")

        return result

    # ---- Reporting ---------------------------------------------------------

    def _report(self, result: PdfResult) -> None:
        name = result.pdf_path.name
        if result.quota_exhausted and result.successful_pages:
            self.console.print(
                f"  [yellow]⚠[/] {result.successful_pages}/{result.total_pages} pages "
                f"(quota exhausted, partial results saved)"
            )
        elif result.quota_exhausted:
            self.console.print("  [red]✗[/] Quota exhausted before any pages completed - no output file created")
        elif not result.successful_pages:
            self.console.print(f"  [red]✗[/] All {result.total_pages} pages failed - no output file created")
        elif result.failed_pages:
            self.console.print(
                f"  [yellow]⚠[/] {result.successful_pages}/{result.total_pages} pages "
                f"([red]failed: {result.failed_pages}[/])"
            )
        else:
            self.console.print(
                f"  [green]✓[/] {result.successful_pages}/{result.total_pages} pages "
                f"({result.success_rate:.0f}%)"
            )
        if result.output_size:
            self.console.print(f"  [dim]Output size:[/] {result.output_size:,} bytes")

        self.logger.info(
            "PDF %s: %d/%d pages successful (%.1f%%)",
            name, result.successful_pages, result.total_pages, result.success_rate,
        )
        if result.failed_pages:
            self.logger.warning("PDF %s: failed pages: %s", name, result.failed_pages)

    def _log_error(self, error: Exception, context: str, page_num: Optional[int]) -> None:
        page_info = f" (page {page_num})" if page_num else ""
        if isinstance(error, genai_errors.APIError):
            code = getattr(error, "code", "unknown")
            message = getattr(error, "message", str(error))
            hint = _API_ERROR_HINTS.get(code, "Check the full error log for details.")
            self.console.print(f"  [red]✗ API error{page_info}[/] - code {code}: {str(message)[:80]}")
            self.logger.error(
                "Gemini API error during %s%s\n  HTTP code: %s\n  Message: %s\n  Suggestion: %s",
                context, page_info, code, message, hint,
            )
        else:
            self.console.print(
                f"  [red]✗ Error{page_info}[/] - {type(error).__name__}: {str(error)[:80]}"
            )
            self.logger.error(
                "Error during %s%s: %s", context, page_info, error, exc_info=True
            )


_API_ERROR_HINTS = {
    429: "Reduce request frequency, or pass --rpm to throttle proactively.",
    400: "Check request parameters, content format, or file size.",
    401: "Verify GEMINI_API_KEY is valid and has proper permissions.",
    403: "Verify GEMINI_API_KEY is valid and has proper permissions.",
    404: "Verify the model name or resource exists.",
    500: "Gemini API is experiencing issues. Retry later.",
    503: "Gemini API is experiencing issues. Retry later.",
}


def _join_pages(pages: List[Tuple[int, str]]) -> str:
    """Join page texts with markers; the first page carries no header."""
    parts: List[str] = []
    for index, (page_num, text) in enumerate(pages):
        if index == 0:
            parts.append(text)
        else:
            parts.append(f"\n\n--- Page {page_num} ---\n\n{text}")
    return "".join(parts)


@dataclass
class BatchResult:
    """Outcome of a directory of PDFs."""

    results: List[PdfResult] = field(default_factory=list)
    quota_exhausted: bool = False
    elapsed_seconds: float = 0.0

    @property
    def processed(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def failed(self) -> int:
        return len(self.results) - self.processed

    @property
    def total_size_mb(self) -> float:
        return sum(r.pdf_path.stat().st_size for r in self.results if r.ok) / (1024 * 1024)


def process_pdf_batch(
    processor: GeminiPageProcessor,
    pdf_files: List[Path],
    output_dir: Path,
    *,
    console: Optional[Console] = None,
    progress=None,
) -> BatchResult:
    """Run every PDF through *processor*, stopping the batch on quota exhaustion.

    A single failing PDF must not abort the run; an exhausted quota must.
    """
    console = console or processor.console
    output_dir.mkdir(parents=True, exist_ok=True)
    batch = BatchResult()
    started = time.monotonic()

    pdf_task = progress.add_task("[cyan]Processing PDFs...", total=len(pdf_files)) if progress else None

    for pdf_path in pdf_files:
        if progress and pdf_task is not None:
            progress.update(pdf_task, description=f"[cyan]Processing {pdf_path.name}...")
        console.print()
        console.rule(f"[bold]📄 {pdf_path.name}[/]")
        console.print(f"  [dim]Size:[/] {pdf_path.stat().st_size / (1024 * 1024):.2f} MB")

        try:
            batch.results.append(
                processor.process_pdf(pdf_path, output_dir / f"{pdf_path.stem}.txt", progress=progress)
            )
        except QuotaExhaustedError:
            batch.quota_exhausted = True
            processor.logger.error("Quota exhausted — aborting remaining PDFs.")
            break
        except Exception as exc:
            console.print(f"[red]✗[/] Failed to process {pdf_path.name}: {exc}")
            processor.logger.error("Failed to process %s: %s", pdf_path.name, exc, exc_info=True)
            batch.results.append(PdfResult(pdf_path=pdf_path, total_pages=0))

        if progress and pdf_task is not None:
            progress.update(pdf_task, advance=1)

    batch.elapsed_seconds = time.monotonic() - started
    return batch
