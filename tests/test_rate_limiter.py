"""Tests for common.rate_limiter quota detection and throttling."""

import time

from common.rate_limiter import (
    RateLimiter,
    is_mistral_quota_exhausted,
    is_quota_exhausted,
)


class FakeGeminiError(Exception):
    """Mimics google.genai errors: carries .code / .message / .status."""

    def __init__(self, code=None, message="", status=""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class FakeMistralError(Exception):
    def __init__(self, status_code=None, message=""):
        super().__init__(message)
        self.status_code = status_code


def test_daily_quota_is_exhaustion():
    err = FakeGeminiError(429, "You exceeded your current quota, please check your plan")
    assert is_quota_exhausted(err) is True


def test_per_day_limit_is_exhaustion():
    err = FakeGeminiError(429, "quota metric: requests_per_model_per_day")
    assert is_quota_exhausted(err) is True


def test_transient_429_is_not_exhaustion():
    # Gemini returns status=RESOURCE_EXHAUSTED for EVERY 429, including
    # per-minute rate limits that must be retried, not treated as fatal.
    err = FakeGeminiError(429, "Resource has been exhausted (e.g. check quota).", status="RESOURCE_EXHAUSTED")
    assert is_quota_exhausted(err) is False


def test_non_429_is_not_exhaustion():
    err = FakeGeminiError(500, "exceeded your current quota")
    assert is_quota_exhausted(err) is False


def test_mistral_rate_limit_is_not_exhaustion():
    # "rate limit exceeded" contains 'exceeded' but is transient
    err = FakeMistralError(429, "Requests rate limit exceeded")
    assert is_mistral_quota_exhausted(err) is False


def test_mistral_quota_is_exhaustion():
    err = FakeMistralError(429, "You have exceeded your current quota for the day")
    assert is_mistral_quota_exhausted(err) is True


def test_mistral_non_429_is_not_exhaustion():
    err = FakeMistralError(503, "billing")
    assert is_mistral_quota_exhausted(err) is False


def test_rate_limiter_disabled_by_default():
    limiter = RateLimiter(requests_per_minute=None)
    start = time.monotonic()
    for _ in range(100):
        limiter.wait()
    assert time.monotonic() - start < 0.5


def test_rate_limiter_spaces_requests():
    limiter = RateLimiter(requests_per_minute=600)  # 0.1s interval
    limiter.wait()
    start = time.monotonic()
    limiter.wait()
    assert time.monotonic() - start >= 0.09
