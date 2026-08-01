"""Make the repo root importable so tests can `import common.*`.

Pipeline directories are added too, because their modules import each other by
bare name (``from sentiment_core import ...``) the way the scripts do — testing
them through a different import path would test a different thing.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

for pipeline in ("AI_sentiment_analysis", "AI_summary_issue"):
    path = str(REPO_ROOT / pipeline)
    if path not in sys.path:
        sys.path.insert(0, path)
