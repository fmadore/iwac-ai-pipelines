#!/usr/bin/env python3
"""
Merge approved duplicate authorities: repoint every link, keep the names.

Reads ``plan_merges.csv`` and, for each row whose ``approve`` is ``yes``, folds
the ``loser_id`` record into ``survivor_id`` (swap the two columns to merge the
other way):

1. Every item that links to the loser, through any property, is found. Omeka has
   no "any property" reverse search, so each property is asked in turn, four at
   a time; a search that fails aborts that merge, because a half-finished scan
   reads exactly like "no more links".
2. Each linking item is PATCHed whole with the link pointed at the survivor, and
   no property may end up naming the survivor twice: a property that already
   links to the survivor keeps that link and drops the loser's, and a survivor
   link the item already held twice is reduced to one. Annotations on kept
   links are kept.
3. The loser's names become ``dcterms:alternative`` values on the survivor,
   unless it already carries them; dates in parentheses are not names.
4. Every PATCHed item is fetched again and checked: no link to the loser, no
   property naming the survivor twice, no other value lost. Then every
   property is searched again for links to the loser. Only a loser with none
   left and no failed check is reported **safe to delete**. Deleting it is the
   curator's act, never this script's; the log also lists any field the loser
   holds that the survivor lacks, to copy by hand before deleting.

Only items are scanned: a link from a media or an item set is not repointed.

Usage:
    python AI_authority_curation/04_merge.py --dry-run
    python AI_authority_curation/04_merge.py --run output/organisations_20261017T160000
"""
from __future__ import annotations

import argparse
import copy
import csv
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.console_utils import count_table, standard_progress  # noqa: E402
from common.log_redaction import configure_logging  # noqa: E402
from common.omeka_client import OmekaClient  # noqa: E402
from common.omeka_text_updater import open_backup  # noqa: E402
from common.write_guard import WriteGuard, add_write_guard_args  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curation_core import missing_alternatives, repeated_links, repoint_links  # noqa: E402

logger = logging.getLogger(__name__)
console = Console()
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
APPROVED = {"yes", "y", "oui", "o"}
MAX_WORKERS = 4
ATTEMPTS = 3


class ScanFailed(RuntimeError):
    pass


def newest_run(output_dir: Path) -> Optional[Path]:
    runs = sorted((p for p in output_dir.glob("*_*") if (p / "manifest.json").exists()),
                  key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def value_count(item: Dict[str, Any]) -> int:
    return sum(len(v) for k, v in item.items()
               if ":" in k and not k.startswith("o:") and isinstance(v, list))


def all_property_ids(client: OmekaClient) -> List[int]:
    ids: List[int] = []
    page = 1
    while True:
        result = client.get_resource(f"{client.base_url}/properties?per_page=100&page={page}")
        if not isinstance(result, list):
            raise ScanFailed(f"could not list properties (page {page})")
        if not result:
            return ids
        ids += [int(p["o:id"]) for p in result]
        page += 1


def linking_items(client: OmekaClient, target: int, property_ids: List[int]) -> Dict[int, Set[int]]:
    """``{item_id: {property ids}}`` for every item linking to *target*."""

    def search(pid: int) -> List[int]:
        for attempt in range(ATTEMPTS):
            try:
                return [int(i["o:id"]) for i in client.get_items(**{
                    "property[0][property]": pid, "property[0][type]": "res",
                    "property[0][text]": target})]
            except Exception as exc:  # noqa: BLE001 - any failure aborts the merge below
                if attempt == ATTEMPTS - 1:
                    raise ScanFailed(f"search on property {pid} failed: {exc}") from exc
                time.sleep(2 ** attempt)
        return []

    found: Dict[int, Set[int]] = {}
    with ThreadPoolExecutor(MAX_WORKERS) as executor:
        for pid, ids in zip(property_ids, executor.map(search, property_ids), strict=True):
            for item_id in ids:
                found.setdefault(item_id, set()).add(pid)
    return found


def alternative_value(name: str, property_id: int) -> Dict[str, Any]:
    return {"type": "literal", "property_id": property_id, "property_label": "Alternative Title",
            "is_public": True, "@value": name}


def merge_one(client: OmekaClient, survivor_id: int, loser_id: int, property_ids: List[int],
              alternative_pid: int, guard: WriteGuard, sink) -> Dict[str, Any]:
    survivor, loser = client.get_item(survivor_id), client.get_item(loser_id)
    if not survivor or not loser:
        return {"status": "not_found"}
    links = linking_items(client, loser_id, property_ids)
    links.pop(loser_id, None)
    names = missing_alternatives(survivor, loser)
    extra_fields = sorted(k for k in loser if ":" in k and not k.startswith("o:")
                          and loser.get(k) and not survivor.get(k))

    report = {"linking_items": len(links), "retargeted": 0, "dropped": 0,
              "alternatives_added": len(names),
              "loser_fields_survivor_lacks": " ".join(extra_fields), "check_failures": 0}
    # The survivor is visited even when it does not link to the loser: it
    # gains the loser's names.
    for item_id in sorted(set(links) | {survivor_id}):
        item = client.get_item(item_id)
        if not item:
            report["check_failures"] += 1
            continue
        original = copy.deepcopy(item)
        retargeted, dropped = repoint_links(item, loser_id, survivor_id, client.base_url)
        added = 0
        if item_id == survivor_id and names:
            item.setdefault("dcterms:alternative", []).extend(
                alternative_value(n, alternative_pid) for n in names)
            added = len(names)
        if not (retargeted or dropped or added):
            continue
        report["retargeted"] += retargeted
        report["dropped"] += dropped
        if guard.dry_run:
            continue
        if sink is not None:
            sink(original)
        if not client.update_item(item_id, item):
            report["check_failures"] += 1
            continue
        after = client.get_item(item_id) or {}
        still_linked = any(isinstance(v, dict) and v.get("value_resource_id") == loser_id
                           for k, vs in after.items() if isinstance(vs, list) for v in vs)
        twice = repeated_links(after, [survivor_id])
        if still_linked or twice or value_count(after) != value_count(original) - dropped + added:
            report["check_failures"] += 1

    if not guard.dry_run:
        # Every property again, not only those that held links: this answers
        # "may the loser be deleted?", and a partial rescan cannot.
        remaining = linking_items(client, loser_id, property_ids)
        remaining.pop(loser_id, None)
        report["links_left_on_loser"] = len(remaining)
        if remaining:
            report["check_failures"] += 1
        report["safe_to_delete"] = "yes" if not report["check_failures"] else "no"
    report["status"] = ("would_merge" if guard.dry_run else
                        "merged, verified" if not report["check_failures"] else "merged, CHECK FAILED")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, help="Run folder (default: newest)")
    add_write_guard_args(parser)
    args = parser.parse_args(argv)

    run_dir = args.run or newest_run(OUTPUT_DIR)
    plan = run_dir / "plan_merges.csv" if run_dir else None
    if plan is None or not plan.exists():
        console.print("[red]No plan_merges.csv found.[/]")
        return 1
    with plan.open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if (r.get("approve") or "").strip().lower() in APPROVED]
    for row in [r for r in rows if r["status"] == "invalid"]:
        console.print(f"[yellow]Skipping {row['survivor_id']}/{row['loser_id']}: approved but invalid[/]")
    rows = [r for r in rows if r["status"] != "invalid"]
    for row in [r for r in rows if r["status"] == "rejected"]:
        console.print(f"[yellow]{row['survivor_id']}/{row['loser_id']}: the reviewer judged these "
                      "two different entities; merging on your approval.[/]")
    losers = [int(r["loser_id"]) for r in rows]
    survivors = {int(r["survivor_id"]) for r in rows}
    clash = sorted(set(losers) & survivors) or sorted({x for x in losers if losers.count(x) > 1})
    if clash:
        console.print(f"[red]Records both kept and merged away, or merged twice: {clash}. "
                      "Chain merges one run at a time.[/]")
        return 1
    if not rows:
        console.print("Nothing approved to merge.")
        return 0

    guard = WriteGuard.from_args(args, default_backup_dir=run_dir / "backups")
    client = OmekaClient.from_env()
    if not guard.confirm(console, action="Merge duplicate authorities (repoint links, keep names)",
                         base_url=client.base_url, item_count=len(rows),
                         details=["Losers are left in place for you to delete by hand."],
                         title="About to merge in Omeka"):
        return 1
    property_ids = all_property_ids(client)
    alternative_pid = client.get_property_id("dcterms:alternative")

    results: List[Dict[str, Any]] = []
    with open_backup(guard.backup_dir if guard.backup_enabled else None,
                     label="authority_merge", dry_run=guard.dry_run) as sink, \
            standard_progress(console) as progress:
        task = progress.add_task("[cyan]Merging…", total=len(rows))
        for row in rows:
            survivor_id, loser_id = int(row["survivor_id"]), int(row["loser_id"])
            try:
                report = merge_one(client, survivor_id, loser_id, property_ids, alternative_pid, guard, sink)
            except ScanFailed as exc:
                report = {"status": f"aborted: {exc}"}
            results.append({"survivor_id": survivor_id, "loser_id": loser_id, **report})
            progress.advance(task)

    columns = ["survivor_id", "loser_id", "status", "safe_to_delete", "linking_items", "retargeted",
               "dropped", "alternatives_added", "links_left_on_loser", "check_failures",
               "loser_fields_survivor_lacks"]
    log_path = run_dir / f"merged_{'dryrun_' if guard.dry_run else ''}{datetime.now():%Y%m%dT%H%M%S}.csv"
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    tally: Dict[str, int] = {}
    for result in results:
        tally[result["status"]] = tally.get(result["status"], 0) + 1
    console.print(count_table(sorted(tally.items()), title=guard.mode_label))
    if not guard.dry_run:
        safe = [r["loser_id"] for r in results if r.get("safe_to_delete") == "yes"]
        unsafe = [r["loser_id"] for r in results if r.get("safe_to_delete") == "no"]
        if safe:
            console.print(f"[green]Safe to delete by hand (no item links to them, all properties "
                          f"rescanned): {safe}[/]")
        if unsafe:
            console.print(f"[red]Do NOT delete yet: links or checks still fail for {unsafe}. "
                          "See the log.[/]")
    console.print(f"Log: [cyan]{log_path}[/]")
    return 0 if all(str(r["status"]) in ("would_merge", "merged, verified") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
