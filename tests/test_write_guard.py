"""The gate in front of every bulk Omeka write.

These tests exist because of a real incident: on 2026-08-02 a write script with
no argument parsing treated ``--help`` as a normal run and PATCHed 630 live
items. Argv handling and the confirmation gate are safety features here, not
ergonomics.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from rich.console import Console

from common.write_guard import WriteGuard, add_write_guard_args


def parse(argv):
    parser = argparse.ArgumentParser()
    add_write_guard_args(parser)
    return parser.parse_args(argv)


def test_help_exits_instead_of_running():
    parser = argparse.ArgumentParser()
    add_write_guard_args(parser)
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0


def test_unknown_flag_is_an_error_not_consent():
    parser = argparse.ArgumentParser()
    add_write_guard_args(parser)
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--not-a-real-flag"])
    assert excinfo.value.code != 0


def test_defaults_are_live_but_backed_up_and_interactive():
    guard = WriteGuard.from_args(parse([]))
    assert guard.dry_run is False
    assert guard.assume_yes is False
    assert guard.backup_enabled is True


def test_flags_are_read_from_argv(tmp_path):
    guard = WriteGuard.from_args(parse(["--dry-run", "--yes", "--backup-dir", str(tmp_path)]))
    assert guard.dry_run is True
    assert guard.assume_yes is True
    assert guard.backup_dir == tmp_path


class FakeConsole(Console):
    """Console whose input() answer is scripted, or raises EOFError."""

    def __init__(self, answer=None):
        super().__init__(record=True, force_terminal=False, width=100)
        self._answer = answer
        self.prompted = False

    def input(self, *args, **kwargs):  # type: ignore[override]
        self.prompted = True
        if self._answer is None:
            raise EOFError
        return self._answer


def confirm(guard, console):
    return guard.confirm(
        console, action="Append links", base_url="https://islam.zmo.de/api", item_count=630
    )


def test_live_run_requires_an_explicit_yes():
    console = FakeConsole("y")
    assert confirm(WriteGuard(), console) is True
    assert console.prompted is True


@pytest.mark.parametrize("answer", ["", "n", "no", "maybe", "Y es"])
def test_anything_but_yes_declines(answer):
    assert confirm(WriteGuard(), FakeConsole(answer)) is False


def test_closed_stdin_declines_rather_than_proceeding():
    """An EOF is not consent — unattended runs must pass --yes on purpose."""
    console = FakeConsole(None)
    assert confirm(WriteGuard(), console) is False


def test_dry_run_never_prompts():
    console = FakeConsole(None)
    assert confirm(WriteGuard(dry_run=True), console) is True
    assert console.prompted is False


def test_assume_yes_skips_the_prompt():
    console = FakeConsole(None)
    assert confirm(WriteGuard(assume_yes=True), console) is True
    assert console.prompted is False


def test_backup_writes_payloads_to_a_timestamped_file(tmp_path):
    guard = WriteGuard(backup_dir=tmp_path)
    stamp = datetime(2026, 8, 2, 12, 1, 29, tzinfo=timezone.utc)

    path = guard.dump_backup([{"o:id": 1}, {"o:id": 2}], label="ner_links", now=stamp)

    assert path == tmp_path / "_pre_write_ner_links_20260802T120129Z.json"
    assert json.loads(path.read_text(encoding="utf-8")) == [{"o:id": 1}, {"o:id": 2}]


def test_backup_is_skipped_when_nothing_would_change(tmp_path):
    assert WriteGuard(backup_dir=tmp_path).dump_backup([], label="ner_links") is None
    assert list(tmp_path.iterdir()) == []


def test_dry_run_writes_no_backup(tmp_path):
    guard = WriteGuard(dry_run=True, backup_dir=tmp_path)
    assert guard.dump_backup([{"o:id": 1}], label="ner_links") is None
    assert list(tmp_path.iterdir()) == []


def test_no_backup_flag_is_honoured(tmp_path):
    guard = WriteGuard.from_args(parse(["--no-backup", "--backup-dir", str(tmp_path)]))
    assert guard.dump_backup([{"o:id": 1}], label="ner_links") is None


def test_every_omeka_write_script_parses_argv():
    """No write entry point may ignore argv.

    A script that PATCHes or POSTs to Omeka without an argument parser treats an
    unrecognised flag as consent to run — which is how ``--help`` PATCHed 630
    live items on 2026-08-02.
    """
    repo_root = Path(__file__).resolve().parent.parent
    writes = ("client.update_item(", "client.create_item(", "update_item_resource_links(")

    offenders = []
    for script in sorted(repo_root.glob("AI_*/*.py")) + sorted(repo_root.glob("NotebookLM/*.py")):
        source = script.read_text(encoding="utf-8")
        if any(call in source for call in writes) and "argparse" not in source:
            offenders.append(script.relative_to(repo_root).as_posix())

    assert offenders == [], f"Omeka write scripts with no argument parser: {offenders}"
