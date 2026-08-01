"""
sentiment_cache.py
==================

Resumable, append-only cache for a sentiment run.

The previous cache was a single JSON object rewritten in full after every item.
That is fine for a few hundred items and wrong for the 12,356-article corpus:
rewriting an n-entry file n times is quadratic, and the run spends more time
serialising than annotating well before the end. Worse, a crash *during*
``json.dump`` leaves a truncated file, so the one artefact that exists to make
a run resumable is also the one most likely to be destroyed by the failure it
is meant to survive.

This module fixes both, plus a third problem that only shows up with a panel:

- **Append-only JSONL.** One line per (item, model) result. Writing costs the
  same whether it is the first record or the sixty-thousandth.
- **A torn final line is survivable.** A process killed mid-write leaves one
  unparseable line; loading skips it and reports it rather than failing. Every
  earlier line is already complete and readable.
- **Granular to the model, not the item.** The old cache keyed on the item, so
  one model erroring meant re-running all of them on resume — with five models
  that is four wasted calls per retry. Here each (item, model) is cached
  independently and a resume asks only for what is actually missing.

Only successful results are cached. An errored call is deliberately *not*
written, so a resume retries it; a run that cached its own failures would
converge on a corpus of error placeholders.

Records also carry the exact model id, which the generation-1 cache did not —
that omission is why reconstructing which model produced the stored corpus
needed a dig through git history.
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO

#: Bumped when the record shape changes in a way that makes older lines
#: unreadable. Lines from a different version are ignored on load.
#:
#: 3 (2026-07-31): ``subjectivite_score`` became a label instead of a 1-5
#: integer. Version-2 records would not have crashed anything — an integer no
#: longer maps to a controlled-vocabulary item, so the field would simply be
#: dropped on write — and that is exactly the failure worth refusing: a run
#: that looked complete while silently omitting one field per item.
CACHE_FORMAT_VERSION = 3


@dataclass
class CacheLoadReport:
    """What loading the cache found. Worth printing: a run that silently drops
    half its cache and re-annotates 6,000 articles should be noisy about it."""

    records: int = 0
    items: int = 0
    skipped_malformed: int = 0
    skipped_version: int = 0
    path: Optional[Path] = None
    #: Byte offset of the first unparseable line, if any.
    torn_at: Optional[int] = None


@dataclass
class SentimentCache:
    """Per-(item, model) results on disk, appended as they are produced."""

    path: Path
    logger: Optional[logging.Logger] = None
    #: item_id -> model_key -> result payload
    _entries: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    _handle: Optional[TextIO] = None
    #: Serialises :meth:`put`. Items are annotated by a worker pool, so without
    #: this two threads can interleave inside one ``write()`` and produce a line
    #: that is not merely torn but *spliced* — half of one record followed by
    #: half of another. The torn-line tolerance in :meth:`load` handles a
    #: truncated tail, which is a crash; it cannot repair a splice, which would
    #: silently drop both results.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- reading ---------------------------------------------------------

    def load(self) -> CacheLoadReport:
        """Read the cache. Later records for the same (item, model) win.

        Last-wins matters for ``--force-reanalyze``: a re-annotation appends
        rather than rewriting history, so the file doubles as an audit trail of
        what was produced when.
        """
        report = CacheLoadReport(path=self.path)
        if not self.path.exists():
            return report

        offset = 0
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line_bytes = len(line.encode("utf-8"))
                stripped = line.strip()
                if not stripped:
                    offset += line_bytes
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    # Almost always the final line of a killed run.
                    report.skipped_malformed += 1
                    if report.torn_at is None:
                        report.torn_at = offset
                    offset += line_bytes
                    continue

                if record.get("v") != CACHE_FORMAT_VERSION:
                    report.skipped_version += 1
                    offset += line_bytes
                    continue

                item_id = str(record.get("item_id", ""))
                model_key = record.get("model")
                result = record.get("result")
                if not item_id or not model_key or not isinstance(result, dict):
                    report.skipped_malformed += 1
                    offset += line_bytes
                    continue

                self._entries.setdefault(item_id, {})[model_key] = result
                report.records += 1
                offset += line_bytes

        report.items = len(self._entries)
        if report.skipped_malformed and self.logger:
            self.logger.warning(
                f"Skipped {report.skipped_malformed} unreadable cache line(s) — "
                f"expected after an interrupted run."
            )
        return report

    def get(self, item_id: Any, model_key: str) -> Optional[Dict[str, Any]]:
        return self._entries.get(str(item_id), {}).get(model_key)

    def has(self, item_id: Any, model_key: str) -> bool:
        return model_key in self._entries.get(str(item_id), {})

    def results_for(self, item_id: Any) -> Dict[str, Dict[str, Any]]:
        """Every cached model result for one item."""
        return dict(self._entries.get(str(item_id), {}))

    def missing_models(self, item_id: Any, model_keys: Iterable[str]) -> List[str]:
        """Which of *model_keys* still need to be called for this item."""
        cached = self._entries.get(str(item_id), {})
        return [key for key in model_keys if key not in cached]

    # -- writing ---------------------------------------------------------

    def put(
        self,
        item_id: Any,
        model_key: str,
        result: Dict[str, Any],
        *,
        model_id: str = "",
        reasoning: str = "",
        prompt: str = "",
    ) -> None:
        """Append one result and make it immediately visible to :meth:`get`.

        ``model_id``, ``reasoning`` and ``prompt`` are provenance, not cache
        keys: they record what actually answered and what it was asked, so a
        cache file is still interpretable long after the code that produced it
        has moved on. ``prompt`` is the fingerprint from
        ``sentiment_core.prompt_fingerprint`` — prompt wording shifts label
        distributions unpredictably, so a value whose prompt is unknown cannot
        be compared with one whose prompt is known.
        """
        record = {
            "v": CACHE_FORMAT_VERSION,
            "item_id": str(item_id),
            "model": model_key,
            "model_id": model_id,
            "reasoning": reasoning,
            "prompt": prompt,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "result": result,
        }

        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            if self._handle is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._handle = open(self.path, "a", encoding="utf-8")

            self._handle.write(line)
            # Flush per record: the point of the cache is to survive the process
            # dying, and buffered lines do not.
            self._handle.flush()

            self._entries.setdefault(str(item_id), {})[model_key] = result

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None

    def __enter__(self) -> "SentimentCache":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
