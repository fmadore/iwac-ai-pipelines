"""
Write reconciled NER entities back to Omeka S as resource links.

Reads the newest ``*_reconciled.csv`` in ``output/`` (or ``--input``) and
appends ``dcterms:spatial`` / ``dcterms:subject`` links to each item, every new
link annotated with ``iwac:nerModel`` naming the model that extracted it. The
implementation is ``common/link_update_cli.py``, shared with
``AI_reference_indexing/05_update_omeka.py``.

Usage:
    python 03_Omeka_update.py --dry-run
    python 03_Omeka_update.py
    python 03_Omeka_update.py --model gemma-4-openrouter --yes

Writes are gated: ``--dry-run`` reports without PATCHing, the pre-write
payloads are dumped to ``output/`` first, and a live run asks before the first
write. On 2026-08-02 this script had no argument parser, so ``--help`` ran the
real update; ``tests/test_ner_omeka_update.py`` is the regression test.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.link_update_cli import (  # noqa: E402
    ItemUpdateResult,
    add_link_update_args,
    find_latest_reconciled_csv as _find_latest_reconciled_csv,
    read_reconciled_rows,
    run_link_update,
    update_reconciled_item,
    update_reconciled_items,
)
from common.log_redaction import install_credential_redaction  # noqa: E402
from common.omeka_client import OmekaClient  # noqa: E402
from common.write_guard import WriteGuard  # noqa: E402

# Credentials ride in Omeka query strings and provider headers; keep them
# out of anything urllib3 or an SDK decides to log.
install_credential_redaction()

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
BACKUP_LABEL = "ner_links"

__all__ = [
    "ItemUpdateResult", "find_latest_reconciled_csv", "read_reconciled_rows",
    "update_item_fields", "update_rows", "build_parser", "main",
]


def find_latest_reconciled_csv(output_dir: Path = OUTPUT_DIR) -> Optional[Path]:
    return _find_latest_reconciled_csv(output_dir)


def update_item_fields(
    client: OmekaClient,
    item_id: str,
    spatial_ids_str: Optional[str],
    subject_ids_str: Optional[str],
    *,
    annotation: Optional[Mapping[str, Any]] = None,
    dry_run: bool = False,
    on_pre_write: Optional[Callable[[MutableMapping[str, Any]], None]] = None,
) -> dict:
    """Apply one item's links; kept as the NER-shaped call the tests exercise."""
    result = update_reconciled_item(
        client,
        {
            "o:id": item_id,
            "Spatial AI Reconciled ID": spatial_ids_str or "",
            "Subject AI Reconciled ID": subject_ids_str or "",
        },
        annotation=annotation, dry_run=dry_run, on_pre_write=on_pre_write,
    )
    return {
        "modified": result.status == "modified",
        "spatial_added": result.spatial_added,
        "subject_added": result.subject_added,
        "error": result.status == "error",
    }


def update_rows(
    client: OmekaClient,
    rows: Sequence[Mapping[str, str]],
    *,
    guard: Optional[WriteGuard] = None,
    annotation: Optional[Mapping[str, Any]] = None,
) -> Counter:
    """Update all valid rows and aggregate durable Omeka outcomes."""
    return update_reconciled_items(
        client, rows, guard=guard, annotation=annotation, backup_label=BACKUP_LABEL,
    )


def build_parser() -> argparse.ArgumentParser:
    """Parse argv so a stray flag is an error, never a silent live run."""
    parser = argparse.ArgumentParser(
        description=(
            "Write reconciled dcterms:spatial and dcterms:subject links from the "
            "latest AI_NER reconciliation CSV back into Omeka S."
        ),
    )
    return add_link_update_args(parser, output_dir=OUTPUT_DIR)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run_link_update(
        args,
        output_dir=OUTPUT_DIR,
        banner="[bold]NER — Omeka S Update[/bold]\nAppend reconciled spatial and subject entity links",
        backup_label=BACKUP_LABEL,
    )


if __name__ == "__main__":
    raise SystemExit(main())
