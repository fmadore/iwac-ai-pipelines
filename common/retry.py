"""
Shared retry decorator with exponential backoff.

Usage:
    from common.retry import retry_with_backoff

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def call_api():
        ...
"""

import functools
import logging
import random
import time
from typing import Callable, Optional, TypeVar

from common.rate_limiter import (
    TRANSIENT_RETRY_DELAY_CEILING_SECONDS,
    QuotaExhaustedError,
    retry_delay_seconds,
)

LOGGER = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    is_retryable: Optional[Callable[[BaseException], bool]] = None,
) -> Callable[[F], F]:
    """Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of attempts (including the first call).
            Must be at least 1.
        base_delay: Initial delay in seconds; doubles after each failure.
        exceptions: Tuple of exception types that trigger a retry.
        is_retryable: Optional predicate refining *exceptions*; when it
            returns ``False`` the exception is re-raised immediately
            (e.g. a 400 among retryable API errors).

    ``QuotaExhaustedError`` is never retried, regardless of the arguments.
    When the error states how long to wait ("Please retry in 23.3s", as Gemini
    throttles do), that wait is used if it is longer than the backoff: guessing
    short turns one throttle into three.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except QuotaExhaustedError:
                    raise  # never retry quota exhaustion
                except exceptions as exc:
                    if is_retryable is not None and not is_retryable(exc):
                        raise
                    last_exc = exc
                    if attempt < max_retries:
                        jittered_delay = delay + random.uniform(0, delay * 0.25)
                        stated = retry_delay_seconds(exc)
                        if stated is not None:
                            jittered_delay = max(
                                jittered_delay,
                                min(stated, TRANSIENT_RETRY_DELAY_CEILING_SECONDS) + random.uniform(0, 2),
                            )
                        LOGGER.warning(
                            "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                            func.__name__,
                            attempt,
                            max_retries,
                            exc,
                            jittered_delay,
                        )
                        time.sleep(jittered_delay)
                        delay *= 2
                    else:
                        LOGGER.error(
                            "%s failed after %d attempts: %s",
                            func.__name__,
                            max_retries,
                            exc,
                        )
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
