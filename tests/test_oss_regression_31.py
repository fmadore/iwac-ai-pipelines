import sys
from fnmatch import fnmatchcase
from pathlib import Path

from test_alto_correction import alto


def test_cli_processes_each_alto_file_once_with_case_insensitive_glob(
    tmp_path, monkeypatch, capsys,
):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    filenames = ["a.alto", "b.ALTO", "c.xml"]
    for name in reversed(filenames):
        (input_dir / name).touch()
    (input_dir / "notes.txt").touch()

    original_glob = Path.glob

    def case_insensitive_glob(path, pattern, **kwargs):
        if path == input_dir:
            return (
                entry for entry in path.iterdir()
                if fnmatchcase(entry.name.lower(), pattern.lower())
            )
        return original_glob(path, pattern, **kwargs)

    processed = []

    def record_file(_client, input_path, _output_path, _system_prompt, _max_lines):
        processed.append(input_path.name)
        return True, 0, 0, 0, "Success"

    monkeypatch.setattr(Path, "glob", case_insensitive_glob)
    monkeypatch.setattr(
        sys, "argv",
        ["02_correct_alto_xml.py", "--input-dir", str(input_dir),
         "--output-dir", str(tmp_path / "output")],
    )
    monkeypatch.setattr(alto, "configure_logging", lambda _level: None)
    monkeypatch.setattr(alto, "build_llm_client", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(alto, "get_system_prompt", lambda: "test prompt")
    monkeypatch.setattr(alto, "process_alto_file", record_file)

    assert alto.main() == 0
    assert processed == filenames
    assert "Found 3 ALTO XML files to process" in capsys.readouterr().out
