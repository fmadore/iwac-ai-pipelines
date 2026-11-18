"""The release checker's heading anchors match the ones GitHub generates."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("check_publication", ROOT / "scripts" / "check_publication.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def test_anchors_follow_github_slug_rules():
    text = (
        "# Shared Utilities\n"
        "## Omeka S Client (`omeka_client.py`)\n"
        "### Quick Start\n"
        "### Quick Start\n"
        "## Language coverage — read this before choosing `02c_`\n"
        "```bash\n# a comment in a code block, not a heading\n```\n"
    )
    assert checker.heading_anchors(text) == {
        "shared-utilities",
        "omeka-s-client-omeka_clientpy",
        "quick-start",
        "quick-start-1",
        "language-coverage--read-this-before-choosing-02c_",
    }


def test_a_link_to_a_missing_heading_or_file_is_reported(tmp_path, monkeypatch, capsys):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.0.0"\n', encoding="utf-8")
    (tmp_path / "CITATION.cff").write_text(
        "cff-version: 1.2.0\ntitle: t\nauthors: []\nlicense: MIT\n"
        "repository-code: https://example.org\ndate-released: 2026-01-01\nversion: 1.0.0\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "# Title\n\n## Usage\n\n[ok](#usage) [gone](#removed) [guide](GUIDE.md#setup) "
        "[far](GUIDE.md#absent) [none](MISSING.md)\n",
        encoding="utf-8",
    )
    (tmp_path / "GUIDE.md").write_text("# Guide\n\n## Setup\n", encoding="utf-8")
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    monkeypatch.setattr(
        checker.subprocess, "check_output",
        lambda *args, **kwargs: "README.md\nGUIDE.md\npyproject.toml\n",
    )

    assert checker.main() == 1
    out = capsys.readouterr().out
    assert "#removed" in out and "#absent" in out and "MISSING.md" in out
    assert "#usage" not in out and "#setup" not in out


def test_the_repository_passes_its_own_check(capsys):
    assert checker.main() == 0
    assert '"errors": []' in capsys.readouterr().out
