"""The curation registry: the decisions the archive itself cannot show.

Omeka shows the records as they are. It cannot show a merge (the loser is
gone), links removed after a review, a name, match or record the curator
refused, or an act left to the curator (deleting a merged loser). Those go in
``decisions.csv``, one row per decision, and every curation pass reads it:

- ``02_build_plan.py`` and ``06_ner_plan.py`` mark a proposal the registry
  refuses as ``rejected`` instead of offering it again;
- ``03_apply.py``, ``04_merge.py`` and ``07_create.py`` refuse such rows even
  when an old plan still approves them, and record what they did or what the
  curator refused (``approve = no``);
- ``python AI_authority_curation/decision_log.py pending`` lists the acts left
  to the curator, ``about ID`` everything decided about a record, and
  ``check-deletes`` marks the deletions Omeka confirms as done.

Columns: ``date`` (ISO), ``action`` (one of ACTIONS), ``records`` (ids, space
separated; for a merge the survivor first), ``value`` (the name, QID or title
concerned, if any), ``status`` (done | rejected | pending), ``decided_by``,
``note``, ``source`` (run folder or changelog date). The CHANGELOG tells the
story; this file is what a script or an agent checks before acting.

Usage:
    python AI_authority_curation/decision_log.py pending
    python AI_authority_curation/decision_log.py about 23504
    python AI_authority_curation/decision_log.py add --action alternative --records 248 \
        --value "Fête nationale" --status rejected --note "generic; every country has one"
    python AI_authority_curation/decision_log.py check-deletes [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.log_redaction import configure_logging  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curation_core import normalize_name  # noqa: E402

REGISTRY = Path(__file__).with_name("decisions.csv")
COLUMNS = ["date", "action", "records", "value", "status", "decided_by", "note", "source"]
ACTIONS = ("merge", "split", "delete", "links", "alternative", "identifier", "create", "title", "hold")
STATUSES = ("done", "rejected", "pending")
REFUSED = {"no", "n", "non"}


@dataclass(frozen=True)
class Decision:
    date: str
    action: str
    records: Tuple[int, ...]
    value: str = ""
    status: str = "done"
    decided_by: str = "curator"
    note: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"unknown action {self.action!r}; expected one of {ACTIONS}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}; expected one of {STATUSES}")

    @property
    def key(self) -> Tuple[str, Tuple[int, ...], str, str]:
        return self.action, self.records, normalize_name(self.value), self.status

    def row(self) -> Dict[str, str]:
        data = asdict(self)
        data["records"] = " ".join(str(r) for r in self.records)
        return data


def parse_records(text: str) -> Tuple[int, ...]:
    return tuple(int(t) for t in (text or "").replace(",", " ").split())


def load(path: Path = REGISTRY) -> List[Decision]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [Decision(date=r["date"], action=r["action"], records=parse_records(r["records"]),
                         value=r.get("value") or "", status=r["status"], decided_by=r.get("decided_by") or "",
                         note=r.get("note") or "", source=r.get("source") or "")
                for r in csv.DictReader(handle)]


def append(decisions: Iterable[Decision], path: Path = REGISTRY) -> int:
    """Add decisions not already recorded (same action, records, value, status); return how many."""
    known = {d.key for d in load(path)}
    fresh = []
    for decision in decisions:
        if decision.key not in known:
            known.add(decision.key)
            fresh.append(decision)
    if not fresh:
        return 0
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerows(d.row() for d in fresh)
    return len(fresh)


def today() -> str:
    return date.today().isoformat()


# --- What the registry refuses -------------------------------------------------

def rejected_names(authority_id: int, names: Sequence[str], decisions: Sequence[Decision]) -> List[str]:
    """The names among ``names`` the curator refused as alternatives of that record."""
    refused = {normalize_name(d.value) for d in decisions
               if d.action == "alternative" and d.status == "rejected" and authority_id in d.records}
    return [n for n in names if normalize_name(n) in refused]


def merge_refusal(survivor: int, loser: int, decisions: Sequence[Decision]) -> Optional[str]:
    pair = {survivor, loser}
    for d in decisions:
        if d.action != "merge":
            continue
        if d.status == "rejected" and pair == set(d.records):
            return f"merge refused by the curator on {d.date}"
        if d.status == "done" and len(d.records) == 2 and d.records[1] in pair:
            return f"{d.records[1]} was already merged into {d.records[0]} on {d.date}"
    return None


def identifier_refusal(authority_id: int, qid: str, decisions: Sequence[Decision]) -> Optional[str]:
    """A refused match: the QID named, or every Wikidata match when no value is given."""
    for d in decisions:
        if (d.action == "identifier" and d.status == "rejected" and authority_id in d.records
                and (not d.value or d.value.strip().upper() == qid.strip().upper())):
            return f"Wikidata match refused by the curator on {d.date}"
    return None


def create_refusal(title: str, decisions: Sequence[Decision]) -> Optional[str]:
    folded = normalize_name(title)
    for d in decisions:
        if d.action == "create" and d.status == "rejected" and normalize_name(d.value) == folded:
            return f"record refused by the curator on {d.date}"
    return None


def awaiting_deletion(record_id: int, decisions: Sequence[Decision]) -> bool:
    return any(d.action == "delete" and d.status == "pending" and record_id in d.records for d in decisions)


def pending(decisions: Sequence[Decision]) -> List[Decision]:
    """Open decisions, minus a pending one a later row closed (same action, records and value)."""
    closed = {(d.action, d.records, normalize_name(d.value)) for d in decisions if d.status != "pending"}
    return [d for d in decisions
            if d.status == "pending" and (d.action, d.records, normalize_name(d.value)) not in closed]


def about(record_id: int, decisions: Sequence[Decision]) -> List[Decision]:
    return [d for d in decisions if record_id in d.records]


def refusal(plan: str, row: Dict[str, str], decisions: Sequence[Decision]) -> Optional[str]:
    """Why the registry refuses this plan row, or None.

    ``plan`` is the CSV's stem. On ``plan_alternatives`` the refused names are
    taken out of ``row["alternatives"]`` in place, and the row is refused only
    when none is left.
    """
    if plan == "plan_merges":
        return merge_refusal(int(row["survivor_id"]), int(row["loser_id"]), decisions)
    if plan == "plan_identifiers" and (row.get("qid") or "").strip():
        return identifier_refusal(int(row["authority_id"]), row["qid"], decisions)
    if plan == "plan_create":
        return create_refusal(row["title"], decisions)
    if plan == "plan_alternatives":
        names = [n.strip() for n in (row.get("alternatives") or "").split("|") if n.strip()]
        dropped = rejected_names(int(row["authority_id"]), names, decisions)
        if dropped:
            row["alternatives"] = " | ".join(n for n in names if n not in dropped)
            if not row["alternatives"]:
                return f"name(s) refused by the curator: {', '.join(dropped)}"
    return None


def screen_plan(plan: str, rows: List[Dict[str, Any]], decisions: Sequence[Decision]) -> int:
    """Mark the rows the registry refuses as ``rejected``, unapproved; return how many."""
    count = 0
    for row in rows:
        reason = refusal(plan, row, decisions)
        if reason:
            count += 1
            row["approve"], row["status"] = "", "rejected"
            field = "problems" if "problems" in row else "note"
            row[field] = "; ".join(filter(None, [row.get(field) or "", f"registry: {reason}"]))
    return count


def keep_unrefused(plan: str, rows: List[Dict[str, str]], decisions: Sequence[Decision],
                   console: Console) -> List[Dict[str, str]]:
    """For a write step: drop approved rows the registry refuses, saying why."""
    kept = []
    for row in rows:
        reason = refusal(plan, row, decisions)
        if reason:
            label = row.get("title") or row.get("authority_id") or f"{row.get('survivor_id')}/{row.get('loser_id')}"
            console.print(f"[yellow]Refusing {label}: {reason} (decisions.csv)[/]")
        else:
            kept.append(row)
    return kept


def refused_rows(path: Path) -> List[Dict[str, str]]:
    """Plan rows whose ``approve`` says no: the curator's explicit refusals."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [r for r in csv.DictReader(handle) if (r.get("approve") or "").strip().lower() in REFUSED]


def refusals_from_plan(path: Path, source: str) -> List[Decision]:
    """The ``approve = no`` rows of a plan, as rejected decisions."""
    plan, found = path.stem, []
    for row in refused_rows(path):
        note = (row.get("note") or row.get("reviewer_reason") or row.get("reason") or "")[:200]
        if plan == "plan_merges":
            found.append(Decision(today(), "merge", (int(row["survivor_id"]), int(row["loser_id"])),
                                  row.get("loser_title") or "", "rejected", "curator via plan", note, source))
        elif plan == "plan_identifiers" and (row.get("qid") or "").strip():
            found.append(Decision(today(), "identifier", (int(row["authority_id"]),), row["qid"].strip(),
                                  "rejected", "curator via plan", note, source))
        elif plan == "plan_create":
            found.append(Decision(today(), "create", (), row["title"].strip(), "rejected",
                                  "curator via plan", note, source))
        elif plan == "plan_alternatives":
            for name in (n.strip() for n in (row.get("alternatives") or "").split("|")):
                if name:
                    found.append(Decision(today(), "alternative", (int(row["authority_id"]),), name,
                                          "rejected", "curator via plan", note, source))
    return found


# --- Command line --------------------------------------------------------------

def show(console: Console, decisions: Sequence[Decision], title: str) -> None:
    if not decisions:
        console.print(f"{title}: none")
        return
    # One plain line per decision: an agent reads this output as often as a person.
    console.print(f"{title}: {len(decisions)}", markup=False)
    for d in decisions:
        records = " ".join(map(str, d.records)) or "-"
        value = f" {d.value!r}" if d.value else ""
        console.print(f"{d.date}  {d.action:<11} {records:<24} {d.status:<8}{value} — {d.note} [{d.source}]",
                      markup=False, highlight=False, soft_wrap=True)


def check_deletes(console: Console, dry_run: bool) -> int:
    from common.omeka_client import OmekaClient

    client = OmekaClient.from_env()
    gone = []
    for d in pending(load()):
        if d.action != "delete":
            continue
        # count_items raises on a failed request, where get_item would return
        # None: a network error must never read as a deletion.
        remaining = [r for r in d.records if client.count_items(id=r)]
        state = "deleted" if not remaining else f"still present: {remaining}"
        console.print(f"{' '.join(map(str, d.records))} ({d.value or d.note}): {state}")
        if not remaining:
            gone.append(Decision(date=today(), action="delete", records=d.records, value=d.value,
                                 status="done", decided_by="curator", note="deletion confirmed in Omeka",
                                 source="decision_log.py check-deletes"))
    if dry_run:
        console.print(f"DRY RUN: {len(gone)} deletion(s) would be marked done.")
        return 0
    console.print(f"{append(gone)} deletion(s) marked done.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pending", help="Acts left to the curator")
    about_cmd = sub.add_parser("about", help="Everything decided about a record")
    about_cmd.add_argument("record", type=int)
    add = sub.add_parser("add", help="Record a decision")
    add.add_argument("--action", choices=ACTIONS, required=True)
    add.add_argument("--records", required=True, help="Ids, space separated; survivor first for a merge")
    add.add_argument("--value", default="")
    add.add_argument("--status", choices=STATUSES, default="done")
    add.add_argument("--decided-by", default="curator")
    add.add_argument("--note", default="")
    add.add_argument("--source", default="")
    add.add_argument("--date", default=None, help="ISO date (default: today)")
    check = sub.add_parser("check-deletes", help="Mark pending deletions Omeka confirms as done")
    check.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    console = Console()
    if args.command == "pending":
        show(console, pending(load()), "Pending")
    elif args.command == "about":
        show(console, about(args.record, load()), f"Decisions about {args.record}")
    elif args.command == "add":
        decision = Decision(date=args.date or today(), action=args.action, records=parse_records(args.records),
                            value=args.value, status=args.status, decided_by=args.decided_by,
                            note=args.note, source=args.source)
        console.print("Recorded." if append([decision]) else "Already recorded.")
    elif args.command == "check-deletes":
        return check_deletes(console, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
