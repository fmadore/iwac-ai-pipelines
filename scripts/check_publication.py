"""Check local Markdown links, heading anchors and citation/package metadata agreement."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"```.*?```", re.S)


def heading_anchors(text):
    """The anchors GitHub generates for a Markdown document's headings."""
    anchors, seen = set(), {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*$", FENCE.sub("", text), re.M):
        heading = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(f"{slug}-{count}" if count else slug)
    return anchors


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
        content = FENCE.sub("", path.read_text(encoding="utf-8"))
        for match in re.finditer(r"\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", content):
            target = unquote(match[1].strip("<>"))
            if re.match(r"(?:[\w+.-]+:|/)", target):
                continue
            target, _, anchor = target.partition("#")
            linked = path.parent / target if target else path
            if not linked.exists():
                errors.append(f"{name}: missing link target {target}")
            elif anchor and linked.suffix == ".md":
                if anchor not in heading_anchors(linked.read_text(encoding="utf-8")):
                    errors.append(f"{name}: no heading for #{anchor} in {target or name}")
    print(json.dumps({"errors": errors}, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
