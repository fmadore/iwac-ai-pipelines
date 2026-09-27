"""Credentials must not survive into log output or a log file.

The regression these guard is real: a 2026-08-03 sentiment run wrote live Omeka
API keys into a log file via three ``urllib3.connectionpool`` retry warnings.
"""

import contextlib
import io
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from common.log_redaction import (  # noqa: E402
    REDACTED,
    CredentialRedactingFilter,
    configure_logging,
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


# ---------------------------------------------------------------------------
# No entry point may skip the install
# ---------------------------------------------------------------------------

def _entry_points():
    repo_root = Path(__file__).resolve().parent.parent
    candidates = sorted(repo_root.glob("AI_*/*.py")) + sorted(
        repo_root.glob("NotebookLM/*.py")
    ) + sorted(repo_root.glob("serving/*.py"))
    for script in candidates:
        source = script.read_text(encoding="utf-8")
        if re.search(r'if __name__ == ["\']__main__["\']', source):
            yield script.relative_to(repo_root).as_posix(), source


_INSTALL_CALL = re.compile(r"^\s*(configure_logging\(|install_credential_redaction\(\))", re.M)


def test_every_entry_point_installs_redaction():
    """Every runnable script must install the filter.

    Not only the ones that log much: with no handler on the root logger a
    warning still reaches stderr through ``logging.lastResort``, so a script
    that never configures logging leaks just as readily. ``configure_logging()``
    installs it; the two PDF-downloader wrappers, which hand logging to a
    shared runner, call ``install_credential_redaction()`` directly.
    """
    offenders = [
        name for name, source in _entry_points() if not _INSTALL_CALL.search(source)
    ]
    assert offenders == [], f"entry points not installing redaction: {offenders}"


def test_redaction_call_never_precedes_its_import():
    """A call above its import is a NameError on startup, not a lint nit."""
    offenders = []
    for name, source in _entry_points():
        imported = source.find("from common.log_redaction import")
        called = _INSTALL_CALL.search(source)
        if imported < 0 or called is None or called.start() < imported:
            offenders.append(name)
    assert offenders == [], f"install called before it is imported: {offenders}"


def test_no_entry_point_configures_logging_by_hand():
    """One setup, so redaction and the quiet request loggers come with it —
    and never at import, where a test importing the script would open log
    files in the repository."""
    offenders = [
        name for name, source in _entry_points()
        if "logging.basicConfig(" in source
        or re.search(r"^(configure_logging|install_credential_redaction)\(", source, re.M)
        and "pdf_downloader" not in name
    ]
    assert offenders == [], f"entry points configuring logging by hand or at import: {offenders}"


# ---------------------------------------------------------------------------
# configure_logging()
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def bare_root():
    """An unconfigured root logger for the duration, then the real one back.

    Entered inside the test body: pytest attaches its capture handlers to the
    root when the call phase starts, and ``basicConfig`` does nothing while any
    handler is present — the behaviour that keeps a script run under test from
    opening its log file.
    """
    root = logging.getLogger()
    saved = (root.handlers[:], root.filters[:], root.level, logging.getLogger("httpx").level)
    root.handlers, root.filters = [], []
    try:
        yield root
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers, root.filters, level, httpx_level = saved
        root.setLevel(level)
        logging.getLogger("httpx").setLevel(httpx_level)


def test_configure_logging_redacts_what_it_writes(tmp_path):
    log_file = tmp_path / "log" / "run.log"
    with bare_root() as root:
        configure_logging(log_file=log_file, terminal=False)
        logging.getLogger("urllib3.connectionpool").warning("Retrying %s", OMEKA_URL)
        assert all(any(isinstance(f, CredentialRedactingFilter) for f in h.filters)
                   for h in root.handlers)

    written = log_file.read_text(encoding="utf-8")
    assert "Retrying" in written
    assert IDENTITY not in written and CREDENTIAL not in written


def test_configure_logging_quiets_per_request_lines_unless_debugging():
    with bare_root():
        configure_logging(stream=io.StringIO())
        assert logging.getLogger("httpx").getEffectiveLevel() == logging.WARNING


def test_configure_logging_keeps_request_lines_at_debug():
    with bare_root():
        logging.getLogger("httpx").setLevel(logging.NOTSET)
        configure_logging(logging.DEBUG, stream=io.StringIO())
        assert logging.getLogger("httpx").getEffectiveLevel() == logging.DEBUG


def test_configure_logging_needs_somewhere_to_write():
    with bare_root(), pytest.raises(ValueError):
        configure_logging(terminal=False)


def test_an_already_configured_process_gets_no_log_file(tmp_path):
    """Under pytest the root already has handlers; a script's main() must not
    open its log file in the repository then."""
    log_file = tmp_path / "log" / "run.log"
    with bare_root() as root:
        root.addHandler(logging.NullHandler())
        configure_logging(log_file=log_file, terminal=False)
    assert not log_file.parent.exists()
