#!/usr/bin/env python3
"""
Reconcile AI-generated keywords against IWAC authority records.

Reads the newest ``items_enriched_*.csv`` (or ``--input``) and reconciles its
``Subject AI`` and ``Spatial AI`` columns, producing beside it:

  - ``*_reconciled.csv``                  — enriched CSV with reconciled ids added
  - ``*_unreconciled_spatial.csv``        — unmatched spatial terms (review / step 4)
  - ``*_unreconciled_subject.csv``        — unmatched subject terms (review / step 4)
  - ``*_ambiguous_authorities_*.csv``     — terms matching several authorities
  - ``*_potential_reconciliation_*.csv``  — fuzzy-match suggestions

The run is ``common/reconciliation_cli.py``, shared with
``AI_NER/02_NER_reconciliation_Omeka.py``.

Usage:
    python 03_reconcile_metadata.py
    python 03_reconcile_metadata.py --input output/items_enriched_78405_20260902.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.log_redaction import install_credential_redaction  # noqa: E402
from common.reconciliation_cli import main as run_main  # noqa: E402

# Credentials ride in Omeka query strings and provider headers; keep them
# out of anything urllib3 or an SDK decides to log.
install_credential_redaction()

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
TAG_SUBJECT = "subject"


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_main(
        argv,
        output_dir=OUTPUT_DIR,
        subject_tag=TAG_SUBJECT,
        banner="[bold]Reference Indexing — Step 3[/bold]\nReconcile AI keywords against IWAC authority records",
        description="Reconcile reference keywords against Omeka S authority records",
        accept=lambda name: name.startswith("items_enriched_"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
