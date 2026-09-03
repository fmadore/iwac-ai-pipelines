#!/usr/bin/env python3
"""
Update Omeka S reference items with reconciled subject and spatial links.

Reads the reconciled CSV from step 3 and optionally merges the items step 4
created. Every link added carries an ``iwac:nerModel`` annotation naming the
model that proposed the keyword. The implementation is
``common/link_update_cli.py``, shared with ``AI_NER/03_Omeka_update.py``.

Usage:
    python 05_update_omeka.py --dry-run
    python 05_update_omeka.py
    python 05_update_omeka.py --new-subject output/newly_created_items_subject_20260307.csv
    python 05_update_omeka.py --new-spatial output/newly_created_items_spatial_20260307.csv

Writes are gated: --dry-run reports without PATCHing, the pre-write payloads are
dumped to output/ first, and a live run asks before the first write.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.link_update_cli import (  # noqa: E402
    ItemUpdateResult,
    add_link_update_args,
    find_latest_reconciled_csv as _find_latest_reconciled_csv,
    load_newly_created_mapping,
    read_reconciled_rows,
    resolved_ids,
    run_link_update,
    update_reconciled_item,
    update_reconciled_items as _update_reconciled_items,
)
from common.log_redaction import install_credential_redaction  # noqa: E402
from common.omeka_client import OmekaClient  # noqa: E402
from common.write_guard import WriteGuard  # noqa: E402

# Credentials ride in Omeka query strings and provider headers; keep them
# out of anything urllib3 or an SDK decides to log.
install_credential_redaction()

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
BACKUP_LABEL = "reference_links"

__all__ = [
    "ItemUpdateResult", "find_latest_reconciled_csv", "load_newly_created_mapping",
    "read_reconciled_rows", "resolved_ids", "update_reconciled_item",
    "update_reconciled_items", "build_argument_parser", "main",
]


def find_latest_reconciled_csv(output_dir: Path = OUTPUT_DIR) -> Optional[Path]:
    return _find_latest_reconciled_csv(output_dir)


def update_reconciled_items(
    client: OmekaClient,
    rows: Sequence[Mapping[str, str]],
    *,
    new_spatial_map: Mapping[str, str] = (),
    new_subject_map: Mapping[str, str] = (),
    guard: Optional[WriteGuard] = None,
    annotation: Optional[Mapping[str, Any]] = None,
) -> Counter:
    """Update all rows with progress reporting and aggregate their outcomes."""
    return _update_reconciled_items(
        client, rows, guard=guard, annotation=annotation,
        new_spatial_map=new_spatial_map, new_subject_map=new_subject_map,
        backup_label=BACKUP_LABEL,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update Omeka items with reconciled metadata links")
    return add_link_update_args(parser, output_dir=OUTPUT_DIR)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    return run_link_update(
        args,
        output_dir=OUTPUT_DIR,
        banner="[bold]Reference Indexing — Step 5[/bold]\nUpdate Omeka S items with reconciled subject and spatial links",
        backup_label=BACKUP_LABEL,
    )


if __name__ == "__main__":
    raise SystemExit(main())
