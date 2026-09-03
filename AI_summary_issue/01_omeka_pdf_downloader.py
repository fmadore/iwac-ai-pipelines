"""
Download the PDFs of an Omeka S item set into ``PDF/``, periodical issues only.

Only items whose resource class is **bibo:Issue** are fetched: the class filter
is applied server-side via ``resource_class_id``, with a defensive ``@type``
backstop. The download itself is ``common/pdf_downloader.py``, shared with
``AI_ocr_extraction/01``.

Usage:
    python 01_omeka_pdf_downloader.py                      # asks for the item set id
    python 01_omeka_pdf_downloader.py --item-set-id 123

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
BIBO_ISSUE_CLASS_ID = 60  # bibo:Issue, verified id

if __name__ == "__main__":
    sys.exit(run_cli(
        None,
        pipeline_dir=PIPELINE_DIR,
        description=(
            "[bold]Download magazine issues from Omeka S[/]\n\n"
            "Retrieves the PDF of every bibo:Issue item in an item set for "
            "article extraction."
        ),
        resource_class_id=BIBO_ISSUE_CLASS_ID,
        required_class_term="bibo:Issue",
    ))
