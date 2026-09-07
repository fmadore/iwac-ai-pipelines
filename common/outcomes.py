"""Exit status for batch jobs: an incomplete operation is not success."""
from typing import Mapping


def batch_exit_code(counts: Mapping[str, int], *, stopped: bool = False) -> int:
    """Missing or empty requested artifacts need attention, like failed writes."""
    return int(stopped or any(counts.get(key, 0) for key in (
        "failed", "errors", "not_found", "empty", "incomplete", "cancelled",
    )))
