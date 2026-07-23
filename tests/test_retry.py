"""Tests for common.retry.retry_with_backoff."""

import pytest

from common.rate_limiter import QuotaExhaustedError
from common.retry import retry_with_backoff


def test_retries_until_success(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    calls = {"n": 0}

    @retry_with_backoff(max_retries=3, base_delay=0.01)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("boom")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3


def test_raises_after_max_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)

    @retry_with_backoff(max_retries=2, base_delay=0.01)
    def always_fails():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        always_fails()


def test_quota_exhaustion_never_retried():
    calls = {"n": 0}

    @retry_with_backoff(max_retries=5, base_delay=0.01)
    def quota():
        calls["n"] += 1
        raise QuotaExhaustedError("daily quota")

    with pytest.raises(QuotaExhaustedError):
        quota()
    assert calls["n"] == 1


def test_zero_max_retries_rejected():
    with pytest.raises(ValueError):
        retry_with_backoff(max_retries=0)


def test_is_retryable_predicate_short_circuits(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    calls = {"n": 0}

    @retry_with_backoff(max_retries=5, base_delay=0.01, is_retryable=lambda exc: "retry me" in str(exc))
    def fatal():
        calls["n"] += 1
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        fatal()
    assert calls["n"] == 1


def test_only_listed_exceptions_are_retried(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    calls = {"n": 0}

    @retry_with_backoff(max_retries=3, base_delay=0.01, exceptions=(ConnectionError,))
    def type_error():
        calls["n"] += 1
        raise TypeError("programming bug")

    with pytest.raises(TypeError):
        type_error()
    assert calls["n"] == 1
