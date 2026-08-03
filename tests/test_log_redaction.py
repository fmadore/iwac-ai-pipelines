"""Credentials must not survive into log output or a log file.

The regression these guard is real: a 2026-08-03 sentiment run wrote live Omeka
API keys into a log file via three ``urllib3.connectionpool`` retry warnings.
"""

import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.log_redaction import (  # noqa: E402
    REDACTED,
    CredentialRedactingFilter,
    install_credential_redaction,
    redact,
    scrub_known_secrets,
)

IDENTITY = "OUosGcCsvSNTMoDq1tX1RfFj2us9kHDj"
CREDENTIAL = "wP9Y4H49oB5cCwOoOG4Eda7x6FNzb0Ia"
OMEKA_URL = (
    f"/api/items?per_page=100&page=19&resource_class_id=36"
    f"&key_identity={IDENTITY}&key_credential={CREDENTIAL}"
)


# ---------------------------------------------------------------------------
# redact() — single unwrapped message
# ---------------------------------------------------------------------------

def test_redact_removes_omeka_query_credentials():
    cleaned = redact(OMEKA_URL)
    assert IDENTITY not in cleaned
    assert CREDENTIAL not in cleaned
    assert f"key_identity={REDACTED}" in cleaned


def test_redact_keeps_diagnostic_context():
    """A scrubbed URL is only useful if you can still see what was retried."""
    cleaned = redact(OMEKA_URL)
    assert "page=19" in cleaned
    assert "resource_class_id=36" in cleaned


def test_redact_removes_bearer_and_bare_keys():
    assert "sk-proj-abcdefghijklmnop0123" not in redact(
        "Authorization: Bearer sk-proj-abcdefghijklmnop0123456789"
    )
    assert "AIzaSyA" not in redact("configured with AIzaSyA1234567890abcdefghij")


def test_redact_leaves_ordinary_text_alone():
    text = "Annotated 9,670 items; 0 failures at concurrency=6"
    assert redact(text) == text


# ---------------------------------------------------------------------------
# The logging filter — the path that actually prevents the leak
# ---------------------------------------------------------------------------

def _capture(emit) -> str:
    """Run ``emit`` against an isolated root logger and return what was written."""
    root = logging.getLogger()
    saved_handlers, saved_filters = root.handlers[:], root.filters[:]
    root.handlers, root.filters = [], []
    stream = io.StringIO()
    root.addHandler(logging.StreamHandler(stream))
    root.setLevel(logging.WARNING)
    try:
        install_credential_redaction()
        emit()
        return stream.getvalue()
    finally:
        root.handlers, root.filters = saved_handlers, saved_filters


def test_filter_scrubs_urllib3_style_lazy_args():
    """urllib3 passes the URL as a ``%s`` arg, not in the format string."""
    out = _capture(lambda: logging.getLogger("urllib3.connectionpool").warning(
        "Retrying (%r) after connection broken by %r: %s",
        "Retry(total=4)", "ConnectionResetError(10054)", OMEKA_URL,
    ))
    assert IDENTITY not in out
    assert CREDENTIAL not in out
    assert "Retrying" in out and "page=19" in out


def test_filter_survives_literal_percent_after_rewrite():
    """Rewriting msg must clear args, or the second interpolation raises."""
    out = _capture(lambda: logging.getLogger("x").warning(
        "%s progressed 50%% of the way", OMEKA_URL
    ))
    assert IDENTITY not in out
    assert "50%" in out


def test_install_is_idempotent():
    root = logging.getLogger()
    saved = root.filters[:]
    root.filters = []
    try:
        install_credential_redaction()
        install_credential_redaction()
        installed = [f for f in root.filters
                     if isinstance(f, CredentialRedactingFilter)]
        assert len(installed) == 1
    finally:
        root.filters = saved


# ---------------------------------------------------------------------------
# scrub_known_secrets() — files that are already on disk
# ---------------------------------------------------------------------------

def test_scrub_handles_credentials_wrapped_across_lines():
    """rich wraps long URLs, splitting ``key_identity=`` mid-parameter."""
    wrapped = (
        "                    /api/items?per_page=100&page=19&key_id\n"
        f"                    entity={IDENTITY[:20]}\n"
        f"                    {IDENTITY[20:]}&key_credential={CREDENTIAL}\n"
    )
    assert IDENTITY not in redact(wrapped), "pattern matching should not be trusted here"
    cleaned = scrub_known_secrets(wrapped, [IDENTITY, CREDENTIAL])
    assert IDENTITY not in cleaned.replace("\n", "").replace(" ", "")
    assert CREDENTIAL not in cleaned


def test_scrub_removes_orphaned_fragment_of_a_partial_scrub():
    """The exact failure mode hit on 2026-08-03: prefix gone, tail left behind."""
    half_scrubbed = f"key_credential={REDACTED}\n{CREDENTIAL[4:]}\n"
    cleaned = scrub_known_secrets(half_scrubbed, [CREDENTIAL])
    assert CREDENTIAL[4:] not in cleaned


def test_scrub_ignores_short_values_that_would_eat_prose():
    text = "the run used model gpt-5.6-luna at concurrency 6"
    assert scrub_known_secrets(text, ["6", "at", "run"]) == text
