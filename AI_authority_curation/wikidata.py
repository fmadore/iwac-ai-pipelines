"""Wikidata candidates for an authority record: search, then describe.

The reviewers never query Wikidata: this module finds the candidates, the
reviewer picks one or none from the batch, and ``02_build_plan.py`` refuses a
pick that was not offered. Answers are cached by name in the run output, so a
rerun asks Wikidata nothing it already asked.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

API = "https://www.wikidata.org/w/api.php"
ITEM_URL = "https://www.wikidata.org/wiki/{}"
#: Wikidata asks every client to identify itself.
USER_AGENT = "IWAC-authority-curation/1.0 (https://github.com/fmadore/iwac-ai-pipelines)"
LANGUAGES = ("fr", "en")


def qid_from(value: Dict[str, Any]) -> Optional[str]:
    """The QID of a ``dcterms:identifier`` value, if it is a Wikidata link."""
    url = str(value.get("@id") or value.get("@value") or "")
    if "wikidata.org/wiki/Q" not in url:
        return None
    return url.rsplit("/", 1)[-1]


def record_qids(record: Dict[str, Any]) -> List[str]:
    return [q for q in (qid_from(v) for v in record.get("dcterms:identifier", []) or []
                        if isinstance(v, dict)) if q]


class WikidataSearch:
    def __init__(self, cache_path: Path, *, limit: int = 4) -> None:
        self.cache_path = cache_path
        self.limit = limit
        self.cache: Dict[str, Any] = (json.loads(cache_path.read_text(encoding="utf-8"))
                                      if cache_path.exists() else {})
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.lock = threading.Lock()

    def _get(self, **params: Any) -> Dict[str, Any]:
        for attempt in range(3):
            response = self.session.get(API, params={**params, "format": "json"}, timeout=30)
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            return response.json()
        response.raise_for_status()
        return {}

    def _search(self, name: str) -> List[str]:
        ids: List[str] = []
        for language in LANGUAGES:
            found = self._get(action="wbsearchentities", search=name, language=language,
                              uselang=language, type="item", limit=self.limit)
            ids += [hit["id"] for hit in found.get("search", []) if hit["id"] not in ids]
        return ids[: self.limit * 2]

    def _describe(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not ids:
            return {}
        entities = self._get(action="wbgetentities", ids="|".join(ids),
                             props="labels|descriptions|claims", languages="|".join(LANGUAGES))
        entities = entities.get("entities", {})
        kinds = {claim["mainsnak"]["datavalue"]["value"]["id"]
                 for entity in entities.values()
                 for claim in entity.get("claims", {}).get("P31", [])[:3]
                 if claim.get("mainsnak", {}).get("datavalue")}
        kind_labels: Dict[str, str] = {}
        if kinds:
            found = self._get(action="wbgetentities", ids="|".join(sorted(kinds)[:50]),
                              props="labels", languages="fr|en")
            for qid, entity in found.get("entities", {}).items():
                labels = entity.get("labels", {})
                kind_labels[qid] = (labels.get("fr") or labels.get("en") or {}).get("value", qid)
        out: Dict[str, Dict[str, Any]] = {}
        for qid, entity in entities.items():
            labels, descriptions = entity.get("labels", {}), entity.get("descriptions", {})
            instance = [kind_labels.get(c["mainsnak"]["datavalue"]["value"]["id"], "")
                        for c in entity.get("claims", {}).get("P31", [])[:3]
                        if c.get("mainsnak", {}).get("datavalue")]
            out[qid] = {
                "qid": qid, "url": ITEM_URL.format(qid),
                "label": (labels.get("fr") or labels.get("en") or {}).get("value", ""),
                "description_fr": descriptions.get("fr", {}).get("value", ""),
                "description_en": descriptions.get("en", {}).get("value", ""),
                "instance_of": [k for k in instance if k],
            }
        return out

    def candidates(self, names: Iterable[str]) -> List[Dict[str, Any]]:
        """Wikidata items any of *names* finds, best-ranked first, deduplicated."""
        ids: List[str] = []
        for name in names:
            key = name.strip().lower()
            if not key:
                continue
            with self.lock:
                cached = self.cache.get(key)
            if cached is None:
                cached = self._search(name)
                with self.lock:
                    self.cache[key] = cached
            ids += [qid for qid in cached if qid not in ids]
        ids = ids[: self.limit * 2]
        described = self._describe(ids)
        return [described[q] for q in ids if q in described]

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
