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


# ---------------------------------------------------------------------------
# A throttle is not an exhausted quota, however it is worded
# ---------------------------------------------------------------------------

#: Verbatim from the live API on 2026-08-27, transcribing a 20-minute segment
#: with gemini-3.5-transcribe. It carries every word the quota indicators match
#: on — "exceeded your current quota", "billing" — and is a per-minute throttle.
TOKENS_PER_MINUTE_THROTTLE = (
    "Error code: 429 - {'error': {'message': 'You exceeded your current quota, "
    "please check your plan and billing details. For more information on this "
    "error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. \n"
    "* Quota exceeded for metric: generativelanguage.googleapis.com/"
    "generate_content_paid_tier_input_token_count, limit: 10000, model: "
    "gemini-3.5-transcribe\nPlease retry in 23.339472825s.', "
    "'code': 'too_many_requests'}}"
)


class _Error(Exception):
    def __init__(self, message, status_code=None, code=None):
        self.message = message
        self.status_code = status_code
        self.code = code
        super().__init__(message)


def test_a_stated_delay_is_read_off_the_message():
    from common.rate_limiter import retry_delay_seconds

    assert retry_delay_seconds(_Error(TOKENS_PER_MINUTE_THROTTLE, 429)) == 23.339472825
    assert retry_delay_seconds(_Error("Please retry in 28s.", 429)) == 28.0
    assert retry_delay_seconds(_Error("no delay here", 429)) is None


def test_a_token_throttle_is_not_treated_as_exhaustion():
    """The failure this guards: a 10,000 input-tokens-per-minute cap stopped an
    82-minute transcription at its second segment, twice, reporting "API quota
    exhausted" while the server was asking for 23 seconds. Every word the
    indicators match on is present — the retry delay is what tells them apart.
    """
    from common.rate_limiter import is_quota_exhausted

    error = _Error(TOKENS_PER_MINUTE_THROTTLE, status_code=429)
    assert "exceeded your current quota" in error.message
    assert "billing" in error.message
    assert is_quota_exhausted(error) is False


def test_a_daily_quota_is_still_terminal():
    """Removing the prose match entirely would have broken this instead."""
    from common.rate_limiter import is_quota_exhausted

    assert is_quota_exhausted(_Error(
        "429 RESOURCE_EXHAUSTED: quota exceeded for "
        "generativelanguage.googleapis.com/requests_per_model_per_day",
        status_code=429,
    )) is True


def test_a_long_stated_delay_stops_the_run_rather_than_sleeping_through_it():
    """An hour is not a throttle to sit on: save what completed and report."""
    from common.rate_limiter import is_quota_exhausted

    assert is_quota_exhausted(_Error(
        "You exceeded your current quota, please check your plan and billing "
        "details. Please retry in 3600s.",
        status_code=429,
    )) is True


def test_payment_required_is_unaffected_by_the_delay_rule():
    """The 402 arm exists because 823 identical failures once ran for hours."""
    from common.rate_limiter import is_quota_exhausted

    assert is_quota_exhausted(_Error("Insufficient credits. Please retry in 5s.", 402)) is True
