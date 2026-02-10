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
import time
from typing import Optional


class QuotaExhaustedError(Exception):
    """Raised when the daily or billing API quota is exhausted.

    Signals the pipeline to stop processing immediately — retrying is
    pointless until the quota resets.
    """


def is_quota_exhausted(error: Exception) -> bool:
    """Detect whether an API error indicates quota exhaustion (not a transient rate limit).

    Returns ``True`` when all of the following hold:
    - The error has an HTTP status code of 429
    - AND the error message / status indicates a *quota* limit (daily, billing)
      rather than a transient per-minute rate limit

    Quota-exhaustion signatures (from Gemini error logs):
    - message contains "exceeded your current quota"
    - message contains "requests_per_model_per_day"
    - status field is "RESOURCE_EXHAUSTED"
    """
    code = getattr(error, "code", None)
    if code != 429:
        return False

    message = str(getattr(error, "message", "")).lower()
    status = str(getattr(error, "status", "")).upper()

    quota_indicators = [
        "exceeded your current quota",
        "requests_per_model_per_day",
        "per_day",
        "billing",
    ]

    if any(indicator in message for indicator in quota_indicators):
        return True

    if status == "RESOURCE_EXHAUSTED":
        return True

    return False


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
        self._logger = logger or logging.getLogger(__name__)

    def wait(self) -> None:
        """Block until enough time has elapsed since the previous request."""
        if self.min_interval <= 0:
            return

        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            self._logger.info("Rate limiter: sleeping %.1fs (RPM=%d)", sleep_time, self.rpm)
            time.sleep(sleep_time)

        self._last_request_time = time.monotonic()
