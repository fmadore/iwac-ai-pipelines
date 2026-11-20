"""The curator's workbook for a run folder: every plan CSV as one sheet of ``review.xlsx``.

``02_build_plan.py`` and ``06_ner_plan.py`` call ``write_review_workbook()`` once
their plans are written, so every round ends with a file the curator can open
and filter. The CSVs stay the record the write scripts read: approvals and
edited descriptions go in them, and the workbook is regenerated from them.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

WORKBOOK_NAME = "review.xlsx"
SHEET_ORDER = ("plan_create", "plan_alternatives", "plan_sort", "plan_descriptions", "plan_roles",
               "plan_identifiers", "plan_merges", "duplicates_to_examine")
STATUS_FILL = {"ready": "E2F0D9", "flagged": "FFF2CC", "invalid": "F8CBAD", "rejected": "D9D9D9"}
MAX_WIDTH = 60


def plan_files(run_dir: Path) -> List[Path]:
    """Plan CSVs in the run folder, known plans first, in pipeline order."""
    found = {p.stem: p for p in run_dir.glob("*.csv")
             if p.stem.startswith("plan_") or p.stem == "duplicates_to_examine"}
    known = [found.pop(name) for name in SHEET_ORDER if name in found]
    return known + [found[name] for name in sorted(found)]


def write_review_workbook(run_dir: Path) -> Path:
    workbook = Workbook()
    workbook.remove(workbook.active)
    for path in plan_files(run_dir):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            continue
        sheet = workbook.create_sheet(path.stem.removeprefix("plan_")[:31])
        header, body = rows[0], rows[1:]
        sheet.append(header)
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        status_col = header.index("status") if "status" in header else None
        for row in body:
            sheet.append(row)
            cells = sheet[sheet.max_row]
            for cell in cells:
                if isinstance(cell.value, str) and cell.value.startswith("https://"):
                    cell.hyperlink = cell.value
                    cell.font = Font(color="0563C1", underline="single")
                cell.alignment = Alignment(vertical="top", wrap_text=True)
            if status_col is not None and row[status_col] in STATUS_FILL:
                fill = PatternFill("solid", fgColor=STATUS_FILL[row[status_col]])
                for cell in cells:
                    cell.fill = fill
        for index, name in enumerate(header, start=1):
            longest = max([len(name), *(len(r[index - 1]) for r in body if index <= len(r))])
            sheet.column_dimensions[get_column_letter(index)].width = min(max(longest + 2, 8), MAX_WIDTH)
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
    if not workbook.worksheets:
        workbook.create_sheet("plans")
    destination = run_dir / WORKBOOK_NAME
    workbook.save(destination)
    return destination
