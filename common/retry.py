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
import time
from typing import Callable, TypeVar

LOGGER = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of attempts (including the first call).
        base_delay: Initial delay in seconds; doubles after each failure.
        exceptions: Tuple of exception types that trigger a retry.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        LOGGER.warning(
                            "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                            func.__name__,
                            attempt,
                            max_retries,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
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
