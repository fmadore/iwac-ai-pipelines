"""One gate in front of every bulk Omeka write.

Learned the hard way on 2026-08-02: ``AI_NER/03_Omeka_update.py`` had no
argument parsing at all, so a ``--help`` invocation was silently ignored and the
script ran its real update against the live archive, PATCHing 630 items before
it was killed. Ignoring argv is the dangerous part — a write entry point must
refuse arguments it does not understand rather than treat them as consent.

A script that PATCHes or creates Omeka resources therefore has to:

1. parse ``argv`` (so ``--help`` prints help and a typo is an error, not a run),
2. offer ``--dry-run``,
3. dump the pre-write payloads somewhere before the first write, and
4. ask before the first write unless ``--yes`` was passed deliberately.

Usage::

    parser = argparse.ArgumentParser(description="...")
    add_write_guard_args(parser)
    args = parser.parse_args()
    guard = WriteGuard.from_args(args, default_backup_dir=OUTPUT_DIR)

    if not guard.confirm(console, action="Link subjects and places",
                         base_url=client.base_url, item_count=len(rows)):
        return 1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from rich.console import Console
from rich.panel import Panel


BACKUP_FILENAME_TIME_FORMAT = "%Y%m%dT%H%M%SZ"


def add_write_guard_args(
    parser: argparse.ArgumentParser,
    *,
    default_backup_dir: Optional[Path] = None,
) -> argparse.ArgumentParser:
    """Add the flags every Omeka write entry point must expose."""
    group = parser.add_argument_group("write safety")
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without sending a single PATCH or POST.",
    )
    group.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation. For unattended runs only.",
    )
    group.add_argument(
        "--backup-dir",
        type=Path,
        default=default_backup_dir,
        help="Where pre-write payloads are dumped (the only route back).",
    )
    group.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not dump pre-write payloads. Not recommended.",
    )
    return parser


@dataclass(frozen=True)
class WriteGuard:
    """Dry-run state, the confirmation gate, and the pre-write backup."""

    dry_run: bool = False
    assume_yes: bool = False
    backup_dir: Optional[Path] = None
    backup_enabled: bool = True

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        *,
        default_backup_dir: Optional[Path] = None,
    ) -> "WriteGuard":
        backup_dir = getattr(args, "backup_dir", None) or default_backup_dir
        return cls(
            dry_run=bool(getattr(args, "dry_run", False)),
            assume_yes=bool(getattr(args, "yes", False)),
            backup_dir=Path(backup_dir) if backup_dir else None,
            backup_enabled=not bool(getattr(args, "no_backup", False)),
        )

    @property
    def mode_label(self) -> str:
        return "DRY RUN — no writes" if self.dry_run else "LIVE write"

    def confirm(
        self,
        console: Console,
        *,
        action: str,
        base_url: str,
        item_count: int,
        details: Sequence[str] = (),
        title: str = "About to write to Omeka",
    ) -> bool:
        """Show the blast radius and, in live mode, ask to proceed.

        Returns False when the operator declines. A closed or non-interactive
        stdin counts as declining: an unattended run must pass ``--yes`` on
        purpose rather than inherit consent from an EOF.
        """
        lines = [
            f"Action:        {action}",
            f"Omeka:         {base_url}",
            f"Resources:     {item_count}",
        ]
        lines.extend(details)
        if not self.dry_run:
            lines.append(
                f"Backup:        {self.backup_dir}" if self.backup_enabled and self.backup_dir
                else "Backup:        [red]disabled — no route back[/]"
            )
        lines.append(f"Mode:          {self.mode_label}")

        console.print(Panel(
            "\n".join(lines),
            title=title,
            border_style="cyan" if self.dry_run else "yellow",
        ))

        if self.dry_run or self.assume_yes:
            return True

        try:
            answer = console.input(f"\n[bold]Proceed with {item_count} live writes? [y/N]:[/] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]No answer on stdin — aborted, nothing written.[/]")
            return False
        if answer.strip().lower() not in ("y", "yes"):
            console.print("[yellow]Aborted — nothing written.[/]")
            return False
        return True

    def dump_backup(
        self,
        payloads: Iterable[Mapping[str, Any]],
        *,
        label: str,
        now: Optional[datetime] = None,
    ) -> Optional[Path]:
        """Write the pre-write payloads to a timestamped JSON file.

        Returns the path, or None when backups are off, the directory is unset,
        or nothing was captured. Dry runs never write a backup: nothing changes,
        so there is nothing to roll back to.
        """
        if self.dry_run or not self.backup_enabled or self.backup_dir is None:
            return None
        captured = [dict(payload) for payload in payloads]
        if not captured:
            return None
        stamp = (now or datetime.now(timezone.utc)).strftime(BACKUP_FILENAME_TIME_FORMAT)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        path = self.backup_dir / f"_pre_write_{label}_{stamp}.json"
        path.write_text(
            json.dumps(captured, indent=1, ensure_ascii=False),
            encoding="utf-8",
        )
        return path
