"""
Shared rate-limiting and quota-exhaustion utilities for Gemini API pipelines.

Usage:
    from common.rate_limiter import RateLimiter, QuotaExhaustedError, is_quota_exhausted

    limiter = RateLimiter(requests_per_minute=5)
    limiter.wait()  # call before each API request

    try:
        response = client.models.generate_content(...)
    except APIError as e:
        if is_quota_exhausted(e):
            raise QuotaExhaustedError(str(e))
"""

import logging
import threading
import time
from typing import Optional


class QuotaExhaustedError(Exception):
    """Raised when the daily or billing API quota is exhausted.

    Signals the pipeline to stop processing immediately — retrying is
    pointless until the quota resets.
    """


def is_quota_exhausted(error: Exception) -> bool:
    """Detect whether an API error indicates quota exhaustion (not a transient rate limit).

    Returns ``True`` when either:

    - The HTTP status is **402 Payment Required** — the account is out of
      money. Never transient, never worth a retry.
    - OR the status is 429 **and** the message indicates a *quota* limit
      (daily, billing) rather than a transient per-minute rate limit.

    Quota-exhaustion signatures (from Gemini error logs):
    - message contains "exceeded your current quota"
    - message contains "requests_per_model_per_day"

    Note: the ``status`` field alone is deliberately NOT used — Gemini returns
    ``RESOURCE_EXHAUSTED`` for *every* 429, including transient per-minute
    rate limits that should be retried, not treated as daily exhaustion.

    The 402 arm was added 2026-08-01 after an OpenRouter balance ran dry
    partway through a 12,305-article sentiment run. Every remaining call
    returned ``402 Insufficient credits``; because nothing recognised it as
    terminal, each was retried three times with backoff and the run continued
    for hours, producing 823 identical failures and no work. A 402 must stop
    the run on the first occurrence.
    """
    # Providers disagree on where the status lives: google-genai uses ``code``,
    # the OpenAI SDK (and therefore OpenRouter) uses ``status_code``.
    code = getattr(error, "code", None)
    status = getattr(error, "status_code", None)
    if 402 in (code, status):
        return True
    if code != 429 and status != 429:
        return False

    message = str(getattr(error, "message", "") or str(error)).lower()

    quota_indicators = [
        "exceeded your current quota",
        "requests_per_model_per_day",
        "per_day",
        "billing",
        "insufficient credits",   # OpenRouter, when it answers 429 rather than 402
    ]

    return any(indicator in message for indicator in quota_indicators)


def is_mistral_quota_exhausted(error: Exception) -> bool:
    """Detect quota exhaustion for Mistral SDK errors.

    Mistral exceptions carry ``status_code`` rather than ``code``, and their
    transient rate-limit messages contain generic wording like
    "rate limit exceeded" — so only unambiguous quota/billing indicators
    are treated as exhaustion.
    """
    status_code = getattr(error, "status_code", None) or getattr(error, "code", None)
    if status_code != 429:
        return False

    message = str(error).lower()
    quota_indicators = [
        "exceeded your current quota",
        "quota exceeded",
        "per_day",
        "billing",
        "insufficient credits",
    ]
    return any(indicator in message for indicator in quota_indicators)


class RateLimiter:
    """Proactive request throttler.

    When *requests_per_minute* is set, :meth:`wait` sleeps as needed so
    that successive API calls are spaced at least ``60 / rpm`` seconds
    apart.  When *requests_per_minute* is ``None`` (the default), no
    throttling is applied — suitable for paid tiers with generous limits.

    Args:
        requests_per_minute: Maximum requests per minute, or ``None`` to
            disable throttling.
        logger: Optional logger; falls back to the module logger.
    """

    def __init__(
        self,
        requests_per_minute: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.rpm = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._last_request_time: float = 0.0
        self._lock = threading.Lock()
        self._logger = logger or logging.getLogger(__name__)

    def wait(self) -> None:
        """Block until enough time has elapsed since the previous request.

        Thread-safe: concurrent callers are serialized so batched/async
        pipelines still respect the configured RPM.
        """
        if self.min_interval <= 0:
            return

        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                self._logger.info("Rate limiter: sleeping %.1fs (RPM=%d)", sleep_time, self.rpm)
                time.sleep(sleep_time)

            self._last_request_time = time.monotonic()
