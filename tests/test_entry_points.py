"""Every runnable script follows the conventions CLAUDE.md states for entry points.

Static checks only: nothing here imports or runs a pipeline script, so no
credentials, network or Omeka instance is involved.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIRS = ("AI_*", "NotebookLM", "serving", "scripts", "examples")
SCRIPTS = sorted(
    path
    for pattern in SCRIPT_DIRS
    for path in ROOT.glob(f"{pattern}/*.py")
)
ENTRY_POINTS = [path for path in SCRIPTS if '__name__ == "__main__"' in path.read_text(encoding="utf-8")
                or "__name__ == '__main__'" in path.read_text(encoding="utf-8")]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def test_the_scan_finds_the_pipelines():
    assert len(ENTRY_POINTS) > 40
    assert ROOT / "AI_NER" / "01_NER_AI.py" in ENTRY_POINTS


@pytest.mark.parametrize("script", ENTRY_POINTS, ids=rel)
def test_exit_status_reaches_the_shell(script):
    """`main()` returns an exit code, and the `__main__` guard hands it to the shell.

    A bare `main()` exits 0 whatever happened, so a failed batch looks like a
    finished one to a job scheduler or a `&&` chain.
    """
    tree = ast.parse(script.read_text(encoding="utf-8"))
    guard = next(
        node for node in tree.body
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)
    )
    body = ast.unparse(guard)
    assert re.search(r"sys\.exit\(|raise SystemExit\(", body), (
        f"{rel(script)}: the __main__ guard must be `sys.exit(main())`"
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=rel)
def test_repo_root_is_inserted_not_appended(script):
    """`append` lets a same-named module earlier on the path shadow `common`."""
    text = script.read_text(encoding="utf-8")
    assert not re.search(r"sys\.path\.append\(", text), rel(script)
    if re.search(r"^(from|import) common\b", text, re.M) and "sys.path" in text:
        assert "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))" in text, rel(script)


@pytest.mark.parametrize(
    "script", sorted(ROOT.glob("common/*.py")) + SCRIPTS, ids=rel,
)
def test_logging_is_configured_in_one_place(script):
    """`configure_logging()` owns the root logger; it is where redaction is installed."""
    if script.name == "log_redaction.py":
        return
    assert "logging.basicConfig" not in script.read_text(encoding="utf-8"), rel(script)


@pytest.mark.parametrize("script", SCRIPTS, ids=rel)
def test_sampling_parameters_are_never_set(script):
    """Temperature, top_p and top_k belong to MODEL_REGISTRY, never to a pipeline."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                assert keyword.arg not in {"temperature", "top_p", "top_k"}, (
                    f"{rel(script)}:{node.lineno} sets {keyword.arg}"
                )
