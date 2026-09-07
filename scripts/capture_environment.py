"""Record a tested environment; registry verification precedes writing constraints.

Run after installing and validating the release environment. This never upgrades
it: latest upstream versions are recorded alongside the versions actually tested.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import subprocess
from urllib.request import urlopen
from urllib.parse import quote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    packages = {dist.metadata["Name"]: dist.version for dist in distributions()
                if dist.metadata["Name"].lower() not in {"iwac-ai-pipelines", "pip", "setuptools", "wheel"}}

    def lookup(pair):
        name, installed = pair
        with urlopen(f"https://pypi.org/pypi/{quote(name)}/json", timeout=30) as response:
            payload = json.load(response)
        if installed not in payload["releases"]:
            raise ValueError(f"Installed {name}=={installed} not found on PyPI")
        return name, {"tested": installed, "latest_upstream": payload["info"]["version"],
                      "registry": f"https://pypi.org/pypi/{name}/json"}

    with ThreadPoolExecutor(max_workers=8) as executor:
        versions = dict(executor.map(lookup, sorted(packages.items())))
    root = Path(__file__).resolve().parent.parent
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()
    record = {"captured_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "platform": platform.system(), "architecture": platform.machine(),
              "commit": git("rev-parse", "HEAD"), "working_tree_dirty": bool(git("status", "--porcelain")),
              "purpose": "Local validation environment; not a claim about historical publication runs",
              "packages": versions}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "environment.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "constraints.txt").write_text(
        "# Tested versions; upstream verified in environment.json. Platform-specific validation snapshot.\n"
        + "\n".join(f"{name}=={version['tested']}" for name, version in versions.items()) + "\n", encoding="utf-8")
    print(f"Recorded {len(versions)} registry-verified packages in {args.output_dir}")


if __name__ == "__main__":
    main()
