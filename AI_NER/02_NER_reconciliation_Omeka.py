#!/usr/bin/env python3
"""
Reconcile AI-extracted entities against the IWAC authority records.

Reads the newest NER CSV in ``output/`` (or ``--input``), matches its
``Spatial AI`` terms against the location authorities and its ``Subject AI``
terms against the subject and topic authorities, and writes beside it:

  - ``*_reconciled.csv``                       — the input plus reconciled Omeka ids
  - ``*_unreconciled_spatial.csv``             — unmatched spatial terms
  - ``*_unreconciled_subject_and_topic.csv``   — unmatched subject/topic terms
  - ``*_ambiguous_authorities_*.csv``          — terms matching several authorities
  - ``*_potential_reconciliation_*.csv``       — fuzzy-match suggestions for review

The run is ``common/reconciliation_cli.py``, shared with
``AI_reference_indexing/03_reconcile_metadata.py``; the matching rules and their
thresholds are in ``common/reconciliation.py``.

Usage:
    python 02_NER_reconciliation_Omeka.py
    python 02_NER_reconciliation_Omeka.py --input output/item_set_123_processed_x.csv
"""

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
TAG_SUBJECT_AND_TOPIC = "subject_and_topic"


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_main(
        argv,
        output_dir=OUTPUT_DIR,
        subject_tag=TAG_SUBJECT_AND_TOPIC,
        banner="[bold]NER Reconciliation Pipeline[/bold]\nReconciles AI-generated entities with Omeka S authority records",
        description="Reconcile NER entities against Omeka S authority records",
    )


if __name__ == "__main__":
    raise SystemExit(main())
