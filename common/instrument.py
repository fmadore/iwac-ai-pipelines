"""Identity shared by offline annotation, shard merging and cache import."""
from typing import Any, Mapping


def record_identity(record: Mapping[str, Any]) -> dict[str, str]:
    return {"model_id": str(record.get("model") or ""),
            "reasoning": str(record.get("reasoning_effort") or ""),
            "prompt": str(record.get("prompt") or "")}


def identity_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    identity = record_identity(record)
    return identity["model_id"], identity["reasoning"], identity["prompt"]
