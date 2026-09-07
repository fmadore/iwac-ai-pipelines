# Offline workflow example

Run `python -X utf8 examples/offline_workflow.py --output-dir .cache/example`
from the repository root. It creates synthetic French text, a blank one-page
PDF and a one-second silent WAV, then commits and validates a synthetic text
artifact and previews a write against a fake archive. `preview.json` should
report one `would_update` and zero `updated` records.

The blank/silent inputs illustrate supported file containers, not OCR or speech
quality. There are no model calls, credentials or external writes. See the
[publication guide](../docs/PUBLICATION.md) for real-run validation requirements.
