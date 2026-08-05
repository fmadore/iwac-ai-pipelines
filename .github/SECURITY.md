# Security Policy

## Reporting a vulnerability

Report security issues privately through
[GitHub's private vulnerability reporting](https://github.com/fmadore/iwac-ai-pipelines/security/advisories/new)
rather than as a public issue. Expect a reply within two weeks.

## Scope

These pipelines hold two kinds of credential, both read from a local `.env`
that is git-ignored and never committed:

- **AI provider API keys** — Gemini, OpenAI, Mistral, OpenRouter.
- **Omeka S API keys** — `OMEKA_KEY_IDENTITY` and `OMEKA_KEY_CREDENTIAL`, which
  grant write access to a live archive.

Reports about credentials leaking into logs, terminal output, saved payload
dumps, or checkpoint files are in scope, as are ways to make a write script
PATCH the archive without passing through `common/write_guard.py`.

## What these pipelines send to third parties

Whole archival documents are sent to third-party model providers. OpenRouter
requests are pinned to `data_collection: "deny"` so they are routed only to
backends that do not retain data; the Gemini, OpenAI, and Mistral APIs are
governed by those vendors' own terms. Anyone reusing this code on their own
collection should confirm that arrangement suits their material before running
anything at scale.
