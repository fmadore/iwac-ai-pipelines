"""Keep API credentials out of log output.

Omeka S authenticates with ``key_identity`` / ``key_credential`` in the query
string, so every request URL carries live credentials. That is fine until
something logs the URL — and on 2026-08-03 something did: three
``urllib3.connectionpool`` retry warnings during a 12k-item sentiment run
rendered the full retry target, key and all, into a log file that was then
pasted around.

The leak is not in any pipeline's own logging. urllib3 warns with the URL it is
about to retry, requests/httpx do the same on some paths, and none of them know
which query parameters are secret. So the redaction belongs at the logging
layer, applied once to the root logger, rather than in a ``logger.warning()``
call that was never the problem.

Usage — call once, immediately after ``logging.basicConfig``::

    from common.log_redaction import install_credential_redaction
    install_credential_redaction()
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

__all__ = ["REDACTED", "redact", "scrub_known_secrets",
           "CredentialRedactingFilter", "install_credential_redaction"]

#: What a scrubbed value is replaced with. Deliberately visible: a log that
#: reads ``key_credential=<redacted>`` tells you the filter ran, whereas a
#: silently removed parameter looks like the URL was never built correctly.
REDACTED = "<redacted>"

#: Query parameters whose value is a credential. ``key_identity`` and
#: ``key_credential`` are Omeka's pair; the rest cover the provider SDKs that
#: also accept a key in the URL.
_SENSITIVE_PARAMS = (
    "key_identity",
    "key_credential",
    "api_key",
    "apikey",
    "access_token",
    "token",
)

#: Matches ``name=value`` up to the next separator. The value class stops at
#: ``&`` (next parameter), whitespace and quotes, so a URL embedded in a longer
#: log line is scrubbed without eating the rest of the message.
_QUERY_PARAM_RE = re.compile(
    r"(?i)\b(" + "|".join(_SENSITIVE_PARAMS) + r")=([^&\s'\"<>\]}]+)"
)

#: ``Authorization: Bearer sk-...`` and the ``Api-Key`` header variants.
_AUTH_HEADER_RE = re.compile(
    r"(?i)\b(authorization|x-api-key|api-key)(\s*[:=]\s*)(\S+)"
)

#: Bare provider keys, for the case where a key is logged with no surrounding
#: parameter name at all (an SDK echoing its own config, say).
_BARE_KEY_RE = re.compile(r"\b(sk-[A-Za-z0-9_\-]{16,}|AIza[A-Za-z0-9_\-]{20,})")


def redact(text: str) -> str:
    """Return ``text`` with any credential-looking substring replaced.

    Assumes ``text`` is a single unwrapped log message, which is what the
    filter below always has. It is **not** sufficient for scrubbing a log file
    that has already been rendered: rich wraps long lines, and a URL split
    across a line break turns ``key_identity=`` into ``key_id\\n  entity=``,
    which no amount of pattern matching on the parameter name will catch. Use
    :func:`scrub_known_secrets` for files.
    """
    text = _QUERY_PARAM_RE.sub(lambda m: f"{m.group(1)}={REDACTED}", text)
    text = _AUTH_HEADER_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
    return _BARE_KEY_RE.sub(REDACTED, text)


#: Shortest run of a secret's characters still worth removing. A 12-character
#: fragment of a 32-character key leaves far too little to guess.
MIN_FRAGMENT = 12


def scrub_known_secrets(
    text: str, secrets: Iterable[str], *, min_fragment: int = MIN_FRAGMENT
) -> str:
    """Remove secret values — and long fragments of them — from ``text``.

    For cleaning a log that is already on disk. Pattern matching fails there —
    see :func:`redact` — so this works from the values themselves, which the
    caller can read out of the environment. Whitespace (including a newline
    plus a renderer's indent) is allowed between any two characters, so a
    wrapped credential is still matched.

    Fragments are scrubbed as well as whole values, because the two ways a
    secret reaches a log file both truncate it. A renderer wraps it across
    lines, and an earlier partial scrub removes a prefix and orphans the tail —
    which is not a hypothetical: the first attempt at cleaning the 2026-08-03
    sentiment log replaced ``key_credential=wP9Y`` and left the remaining 28
    characters sitting on the next line, where a whole-value search no longer
    matched them. Longest fragments are removed first so the result is one
    ``<redacted>`` rather than a chain of them.
    """
    for secret in secrets:
        if not secret or len(secret) < min_fragment:
            continue  # too short to match safely; would eat ordinary prose
        length = len(secret)
        for size in range(length, min_fragment - 1, -1):
            for start in range(0, length - size + 1):
                fragment = secret[start:start + size]
                pattern = r"\s*".join(re.escape(char) for char in fragment)
                text = re.sub(pattern, REDACTED, text)
    return text


class CredentialRedactingFilter(logging.Filter):
    """Scrub credentials from a record before any handler formats it.

    Attached to the *root* logger so it covers third-party libraries, which is
    where the leak actually comes from. A filter on the root logger does not
    run for records emitted by child loggers, so ``install_credential_redaction``
    also attaches it to the handlers — handler filters see every record that
    reaches them regardless of which logger produced it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        cleaned = redact(message)
        if cleaned != message:
            # Collapse to a pre-formatted message. Formatting has already
            # happened above, so the args would otherwise be interpolated a
            # second time and raise on any literal '%' in the scrubbed text.
            record.msg = cleaned
            record.args = ()
        return True


def install_credential_redaction(
    logger_names: Optional[Iterable[str]] = None,
) -> CredentialRedactingFilter:
    """Attach the redacting filter to the root logger and its handlers.

    Returns the filter so a caller can attach it elsewhere. Safe to call more
    than once: an existing instance is reused rather than stacked, so repeated
    calls do not scrub the same record N times.
    """
    root = logging.getLogger()
    existing = next(
        (f for f in root.filters if isinstance(f, CredentialRedactingFilter)), None
    )
    log_filter = existing or CredentialRedactingFilter()

    if existing is None:
        root.addFilter(log_filter)

    # Handlers are where it matters: a record from urllib3.connectionpool
    # propagates up to the root handlers without ever consulting the root
    # logger's own filters.
    for handler in root.handlers:
        if not any(isinstance(f, CredentialRedactingFilter) for f in handler.filters):
            handler.addFilter(log_filter)

    for name in logger_names or ():
        named = logging.getLogger(name)
        if not any(isinstance(f, CredentialRedactingFilter) for f in named.filters):
            named.addFilter(log_filter)

    return log_filter
