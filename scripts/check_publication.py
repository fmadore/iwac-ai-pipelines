"""Check local Markdown links and agreement between citation/package metadata."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent


def main():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    errors = []
    version = re.search(r"^version:\s*[\"']?([^\s\"']+)", citation, re.M)
    if not version or version[1] != metadata["version"]:
        errors.append("CITATION.cff version differs from pyproject.toml")
    for field in ("cff-version", "title", "authors", "license", "repository-code", "date-released"):
        if not re.search(rf"^{field}:", citation, re.M):
            errors.append(f"CITATION.cff is missing {field}")
    # Tracked docs plus new documentation, excluding ignored working data.
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                                    cwd=ROOT, text=True).splitlines()
    for name in names:
        path = ROOT / name
        if path.suffix != ".md":
            continue
        content = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
        for match in re.finditer(r"\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", content):
            target = unquote(match[1].strip("<>"))
            if re.match(r"(?:[\w+.-]+:|#|/)", target):
                continue
            target = target.split("#", 1)[0]
            if target and not (path.parent / target).exists():
                errors.append(f"{name}: missing link target {target}")
    print(json.dumps({"errors": errors}, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
