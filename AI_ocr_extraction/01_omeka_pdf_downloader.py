"""
Download the PDFs of an Omeka S item set into ``PDF/`` for OCR extraction.

The download itself is ``common/pdf_downloader.py``; this file only names the
pipeline it serves. The magazine pipeline's ``01`` is the same entry point
restricted to ``bibo:Issue`` items.

Usage:
    python 01_omeka_pdf_downloader.py                      # asks for the item set id
    python 01_omeka_pdf_downloader.py --item-set-id 123
    python 01_omeka_pdf_downloader.py --item-set-id 123 --workers 4

Requirements:
    OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL in .env
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.pdf_downloader import run_cli  # noqa: E402
from common.log_redaction import install_credential_redaction  # noqa: E402

# Credentials ride in Omeka query strings; keep them out of anything urllib3
# decides to log. Installed here as well as in run_cli so the rule that every
# entry point installs it stays checkable.
install_credential_redaction()

PIPELINE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    sys.exit(run_cli(
        None,
        pipeline_dir=PIPELINE_DIR,
        description=(
            "[bold]Download PDF files from Omeka S digital collections[/]\n\n"
            "Retrieves every PDF media attachment of an item set and saves it "
            "locally for OCR processing."
        ),
    ))
