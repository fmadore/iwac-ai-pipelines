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
    LOG_FORMAT,
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


def test_redact_removes_unprefixed_header_tokens():
    """Mistral and self-hosted keys have no ``sk-`` prefix for the bare-key rule."""
    token = "3f9a1c0b7e2d4a6f8b0c1d2e3f4a5b6c7d8e9f00a1b2c3d4"
    for text in (
        f"Authorization: Bearer {token}",
        f"{{'Authorization': 'Bearer {token}', 'Accept': 'application/json'}}",
        f'{{"x-api-key": "{token}"}}',
    ):
        cleaned = redact(text)
        assert token not in cleaned
        assert REDACTED in cleaned
    assert redact(f"Authorization: Bearer {token}") == f"Authorization: Bearer {REDACTED}"
    assert "'Accept': 'application/json'" in redact(
        f"{{'Authorization': 'Bearer {token}', 'Accept': 'application/json'}}"
    )


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
    saved_last_resort = logging.lastResort.filters[:]
    root.handlers, root.filters = [], []
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    try:
        install_credential_redaction()
        emit()
        return stream.getvalue()
    finally:
        root.handlers, root.filters = saved_handlers, saved_filters
        logging.lastResort.filters = saved_last_resort


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
    transports = ("httpx", "httpcore", "httpx2", "httpcore2")
    saved = (root.handlers[:], root.filters[:], root.level,
             {name: logging.getLogger(name).level for name in transports})
    saved_last_resort = logging.lastResort.filters[:]
    root.handlers, root.filters = [], []
    try:
        yield root
    finally:
        for handler in root.handlers:
            handler.close()
        logging.lastResort.filters = saved_last_resort
        root.handlers, root.filters, level, transport_levels = saved
        root.setLevel(level)
        for name, transport_level in transport_levels.items():
            logging.getLogger(name).setLevel(transport_level)


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
    """httpx under OpenAI and google-genai, httpx2 under mistralai 3.x."""
    with bare_root():
        configure_logging(stream=io.StringIO())
        for name in ("httpx", "httpcore.http11", "httpx2", "httpcore2.http11"):
            assert logging.getLogger(name).getEffectiveLevel() == logging.WARNING


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


# ---------------------------------------------------------------------------
# Logged exceptions — a requests error carries its URL into the traceback
# ---------------------------------------------------------------------------

def _leaking_error() -> ConnectionError:
    return ConnectionError(f"Max retries exceeded with url: {OMEKA_URL}")


def _log_exception(logger: logging.Logger) -> None:
    try:
        raise _leaking_error()
    except ConnectionError:
        logger.exception("Download failed")


def _assert_redacted_traceback(out: str) -> None:
    assert "Download failed" in out
    assert "Traceback" in out and "ConnectionError" in out
    assert f"key_credential={REDACTED}" in out
    assert IDENTITY not in out and CREDENTIAL not in out


def test_logged_traceback_is_redacted_by_a_plain_formatter():
    out = _capture(lambda: _log_exception(logging.getLogger("pipeline")))
    _assert_redacted_traceback(out)


def test_logged_traceback_is_redacted_by_the_rich_handler():
    """Rich renders ``exc_info`` itself, after the filters; the scrubbed text
    must still be shown in its place, not dropped."""
    from rich.console import Console

    buffer = io.StringIO()
    with bare_root():
        configure_logging(console=Console(file=buffer, width=1000, color_system=None))
        _log_exception(logging.getLogger("pipeline"))
    _assert_redacted_traceback(buffer.getvalue())


def test_last_resort_redacts_an_exception_argument(monkeypatch):
    """``LOGGER.error("...: %s", exc)`` with no handler configured: only the
    record factory and ``logging.lastResort`` see it."""
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)
    with bare_root():
        install_credential_redaction()
        logging.getLogger("pipeline").error("Fetch failed: %s", _leaking_error())
    printed = stderr.getvalue()
    assert "Fetch failed: Max retries exceeded" in printed
    assert IDENTITY not in printed and CREDENTIAL not in printed


def test_last_resort_redacts_a_logged_traceback(monkeypatch):
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)
    with bare_root():
        install_credential_redaction()
        logging.lastResort.filters = []  # the record factory alone must suffice
        _log_exception(logging.getLogger("pipeline"))
    _assert_redacted_traceback(stderr.getvalue())


def test_factory_scrubs_exception_args_and_keeps_harmless_ones(monkeypatch):
    import threading

    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    with bare_root():
        install_credential_redaction()
    harmless = ValueError("no such item")
    record = logging.getLogger("pipeline").makeRecord(
        "pipeline", logging.ERROR, __file__, 1, "%s / %s", (_leaking_error(), harmless), None
    )
    assert record.args[1] is harmless
    assert IDENTITY not in record.args[0] and CREDENTIAL not in record.args[0]
    mapped = logging.getLogger("pipeline").makeRecord(
        "pipeline", logging.ERROR, __file__, 1, "%(error)s", ({"error": _leaking_error()},), None
    )
    assert IDENTITY not in mapped.getMessage()


def test_filter_on_exc_info_is_idempotent():
    """Root logger and every root handler share one filter instance."""
    try:
        raise _leaking_error()
    except ConnectionError:
        # Built directly, so the record factory has not touched it.
        record = logging.LogRecord(
            "pipeline", logging.ERROR, __file__, 1, "Download failed", (), sys.exc_info()
        )
    log_filter = CredentialRedactingFilter()
    log_filter.filter(record)
    first = record.exc_text
    log_filter.filter(record)
    assert record.exc_text == first and record.exc_info is None
    out = logging.Formatter().format(record)
    assert out.count("Traceback") == 1
    _assert_redacted_traceback(out)


def test_filter_scrubs_a_traceback_already_formatted():
    """A handler without the filter may have cached the raw text in ``exc_text``."""
    try:
        raise _leaking_error()
    except ConnectionError:
        record = logging.LogRecord(
            "pipeline", logging.ERROR, __file__, 1, "Download failed", (), True
        )
        CredentialRedactingFilter().filter(record)  # exc_info=True: read sys.exc_info()
    _assert_redacted_traceback(logging.Formatter().format(record))

    record = logging.LogRecord("pipeline", logging.ERROR, __file__, 1, "Download failed", (), None)
    record.exc_text = f"Traceback ...\nConnectionError: {OMEKA_URL}"
    CredentialRedactingFilter().filter(record)
    assert CREDENTIAL not in logging.Formatter().format(record)


# ---------------------------------------------------------------------------
# Uncaught exceptions — they bypass logging entirely
# ---------------------------------------------------------------------------

def test_uncaught_exception_traceback_is_redacted():
    """A requests error escaping main() prints its URL through the excepthook.

    Run in a subprocess, because only a real interpreter exit goes through
    ``sys.excepthook``.
    """
    import subprocess

    script = (
        "import sys; sys.path.insert(0, sys.argv[1])\n"
        "from common.log_redaction import configure_logging\n"
        "configure_logging()\n"
        f"raise ConnectionError('Max retries exceeded with url: {OMEKA_URL}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve().parent.parent)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 1
    assert "ConnectionError" in result.stderr
    assert "Max retries exceeded" in result.stderr
    assert f"key_credential={REDACTED}" in result.stderr
    assert IDENTITY not in result.stderr and CREDENTIAL not in result.stderr


def test_uncaught_exception_in_a_worker_thread_is_redacted(monkeypatch):
    import threading

    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    install_credential_redaction()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)

    def fail():
        raise ConnectionError(f"Max retries exceeded with url: {OMEKA_URL}")

    worker = threading.Thread(target=fail, name="annotator-1")
    worker.start()
    worker.join()

    printed = stderr.getvalue()
    assert "Exception in thread annotator-1" in printed
    assert "ConnectionError" in printed
    assert IDENTITY not in printed and CREDENTIAL not in printed
