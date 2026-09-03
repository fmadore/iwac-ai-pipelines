"""
Shared Omeka S API client for all pipelines.

Provides authenticated access to the Omeka S REST API with:
- Paginated item retrieval by item set, class, template, or any search filter
- Batch fetch by id, and a count from the ``Omeka-S-Total-Results`` header
- Single item fetch and update (PATCH)
- Retry-capable HTTP sessions with a finite timeout on every request
- Environment-based configuration

Usage:
    from common.omeka_client import OmekaClient

    client = OmekaClient.from_env()
    items = client.get_items(item_set_id=123)
    articles = client.get_items(resource_class_id=36)
    total = client.count_items(resource_class_id=36)
    item = client.get_item(456)
    client.update_item(456, item)

Omeka S authenticates with ``key_identity`` / ``key_credential`` as query
parameters — there is no header alternative — so every URL carries the
credential, and ``requests`` echoes the URL in its error messages. Entry
points call ``common.log_redaction.install_credential_redaction()`` so those
messages reach the logs masked.
"""

import os
import logging
import warnings
from typing import Any, Dict, Iterable, Iterator, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)

ITEMS_PER_PAGE = 100

# (connect, read) timeout applied to every request so a stalled connection
# can never hang a batch run indefinitely.
DEFAULT_TIMEOUT = (10, 120)

#: Header Omeka S sets on every list response: the count of matches across
#: all pages. Used for progress totals and to check a page walk was complete.
TOTAL_RESULTS_HEADER = "Omeka-S-Total-Results"
VERSION_HEADER = "Omeka-S-Version"


class OmekaRequestError(RuntimeError):
    """A search could not be answered — distinct from "no match"."""


class OmekaClient:
    """Lightweight client for the Omeka S REST API."""

    def __init__(
        self,
        base_url: str,
        key_identity: str,
        key_credential: str,
        timeout: tuple = DEFAULT_TIMEOUT,
    ):
        self.key_identity = key_identity
        self.key_credential = key_credential
        self.timeout = timeout

        # Normalize base URL: ensure it ends with /api
        base = base_url.rstrip("/")
        if base.endswith("/api"):
            base = base[:-4]
        self.base_url = f"{base}/api"

        self.session = self._create_session()
        #: Omeka S release the server reported on its last response, for run
        #: provenance. ``None`` until the first request.
        self.server_version: Optional[str] = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "OmekaClient":
        """Create a client from OMEKA_* environment variables.

        Loads ``.env`` automatically via python-dotenv.
        """
        load_dotenv()
        base_url = os.getenv("OMEKA_BASE_URL", "")
        key_identity = os.getenv("OMEKA_KEY_IDENTITY", "")
        key_credential = os.getenv("OMEKA_KEY_CREDENTIAL", "")
        if not all([base_url, key_identity, key_credential]):
            raise ValueError(
                "Missing required environment variables. Please set:\n"
                "  OMEKA_BASE_URL\n"
                "  OMEKA_KEY_IDENTITY\n"
                "  OMEKA_KEY_CREDENTIAL"
            )
        return cls(base_url, key_identity, key_credential)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_session() -> requests.Session:
        """Return a session with automatic retry on transient errors.

        GET and PATCH are retried (PATCH sends the full item representation,
        so replaying it is safe). POST is deliberately excluded: retrying a
        create whose first attempt actually succeeded would duplicate items.

        Omeka S core never rate-limits, so the retries are for an overloaded
        PHP host: ``backoff_factor=1`` waits 0, 2, 4, 8 and 16 s, long enough
        for a shared server to come back, and ``Retry-After`` is honoured when
        a proxy sends one.
        """
        session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset({"GET", "HEAD", "OPTIONS", "PATCH"}),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _auth_params(self) -> Dict[str, str]:
        return {
            "key_identity": self.key_identity,
            "key_credential": self.key_credential,
        }

    def _remember_version(self, response: Any) -> None:
        headers = getattr(response, "headers", None)
        try:
            version = headers.get(VERSION_HEADER)
        except AttributeError:
            return
        if isinstance(version, str) and version:
            self.server_version = version

    @staticmethod
    def _total_results(response: Any) -> Optional[int]:
        """The match count from the list header, or ``None`` when absent."""
        headers = getattr(response, "headers", None)
        try:
            value = headers.get(TOTAL_RESULTS_HEADER)
        except AttributeError:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            return None
        try:
            return int(value)
        except ValueError:
            return None

    def _list_page(self, params: Dict[str, Any]) -> Any:
        """One GET on ``/items`` with auth, timeout and version capture."""
        response = self.session.get(
            f"{self.base_url}/items",
            params={**self._auth_params(), **params},
            timeout=self.timeout,
        )
        response.raise_for_status()
        self._remember_version(response)
        return response

    # ------------------------------------------------------------------
    # JSON-LD value helpers
    #
    # Omeka S keys property values by vocabulary term (e.g. 'bibo:content');
    # writing anywhere else is silently dropped on PATCH. These helpers make
    # that class of bug impossible in pipeline scripts.
    # ------------------------------------------------------------------

    @staticmethod
    def upsert_property_value(
        item_data: Dict[str, Any],
        term: str,
        property_id: int,
        value: str,
        *,
        property_label: str = "",
        is_public: bool = True,
        language: Optional[str] = None,
    ) -> bool:
        """Set the literal value for *term* on an item's JSON-LD payload.

        .. deprecated:: 1.1.0
            Use :func:`common.omeka_text_updater.apply_text_value`. This helper
            rebuilds the value object from five keys when it appends, so a
            value annotation (``iwac:*Model``) is silently dropped, and it
            matches the *first* literal on the property whatever its
            ``@language``, so a second language clobbers the first. It is kept
            for a single untagged literal whose annotation the caller
            re-attaches itself.

        Replaces the first existing literal with the same *property_id*, or
        appends a new value object. Mutates *item_data* in place.

        Returns:
            True if *item_data* changed.
        """
        warnings.warn(
            "OmekaClient.upsert_property_value drops @annotation and ignores "
            "@language; use common.omeka_text_updater.apply_text_value",
            DeprecationWarning,
            stacklevel=2,
        )
        values = item_data.get(term)
        if not isinstance(values, list):
            values = item_data[term] = []

        for entry in values:
            if isinstance(entry, dict) and entry.get("property_id") == property_id \
                    and entry.get("type", "literal") == "literal":
                if entry.get("@value") == value:
                    return False
                entry["@value"] = value
                entry["type"] = "literal"
                return True

        new_value: Dict[str, Any] = {
            "type": "literal",
            "property_id": property_id,
            "property_label": property_label or term.split(":")[-1],
            "is_public": is_public,
            "@value": value,
        }
        if language:
            new_value["@language"] = language
        values.append(new_value)
        return True

    @staticmethod
    def append_resource_links(
        item_data: Dict[str, Any],
        term: str,
        property_id: int,
        resource_ids: List[int],
        *,
        property_label: str = "",
        is_public: bool = True,
    ) -> int:
        """Append ``resource:item`` links for IDs not already present on *term*.

        Mutates *item_data* in place and skips duplicates.

        Returns:
            Number of links added.
        """
        values = item_data.get(term)
        if not isinstance(values, list):
            values = item_data[term] = []

        existing_ids = set()
        for entry in values:
            if isinstance(entry, dict) and "value_resource_id" in entry:
                try:
                    existing_ids.add(int(entry["value_resource_id"]))
                except (TypeError, ValueError):
                    pass

        added = 0
        for resource_id in resource_ids:
            resource_id = int(resource_id)
            if resource_id in existing_ids:
                continue
            values.append({
                "type": "resource:item",
                "property_id": property_id,
                "property_label": property_label or term.split(":")[-1],
                "is_public": is_public,
                "value_resource_id": resource_id,
                "value_resource_name": "items",
            })
            existing_ids.add(resource_id)
            added += 1
        return added

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def iter_items(
        self,
        item_set_id: Optional[int] = None,
        per_page: int = ITEMS_PER_PAGE,
        **filters: Any,
    ) -> Iterator[Dict[str, Any]]:
        """Yield items matching the filters, one page at a time.

        Any ``/api/items`` search parameter is accepted by keyword:
        ``item_set_id``, ``resource_class_id``, ``resource_template_id``,
        ``modified_after``, ``property[0][property]``, ``sort_by`` … The walk
        follows ``Omeka-S-Total-Results`` when the server sends it, so it ends
        after exactly the announced number of items and warns if the count
        moved underneath it; without the header it stops on the first short
        page. Pages are sorted by id, which Omeka does by default, so the walk
        is deterministic.

        Streaming matters for the article corpus: 12,000 items whose
        ``bibo:content`` is the full OCR text is a lot to hold when each is
        used once.
        """
        params: Dict[str, Any] = {"per_page": per_page, "page": 1}
        if item_set_id is not None:
            params["item_set_id"] = item_set_id
        params.update({key: value for key, value in filters.items() if value is not None})

        seen = 0
        expected: Optional[int] = None
        while True:
            response = self._list_page(params)
            page_items = response.json()
            if expected is None:
                expected = self._total_results(response)
            if not isinstance(page_items, list) or not page_items:
                break
            yield from page_items
            seen += len(page_items)
            if expected is not None and seen >= expected:
                break
            if expected is None and len(page_items) < per_page:
                break
            params["page"] += 1

        if expected is not None and seen != expected:
            LOGGER.warning(
                "Item listing returned %d items but the server announced %d — "
                "the collection changed during the walk", seen, expected,
            )

    def get_items(
        self,
        item_set_id: Optional[int] = None,
        per_page: int = ITEMS_PER_PAGE,
        **filters: Any,
    ) -> List[Dict[str, Any]]:
        """Fetch every item matching the filters — see :meth:`iter_items`."""
        return list(self.iter_items(item_set_id, per_page, **filters))

    def list_page(self, page: int, per_page: int = ITEMS_PER_PAGE, **filters: Any) -> List[Dict[str, Any]]:
        """One page of a listing, for samplers that pick pages at random."""
        params = {key: value for key, value in filters.items() if value is not None}
        response = self._list_page({**params, "per_page": per_page, "page": page})
        result = response.json()
        return result if isinstance(result, list) else []

    def count_items(self, **filters: Any) -> int:
        """How many items match, from the count header — one request, no paging.

        This is what makes an accurate progress bar possible without first
        downloading the corpus to measure it.
        """
        params = {key: value for key, value in filters.items() if value is not None}
        response = self._list_page({**params, "per_page": 1, "page": 1})
        total = self._total_results(response)
        if total is None:
            raise OmekaRequestError(f"Server sent no {TOTAL_RESULTS_HEADER} header")
        return total

    def get_items_by_ids(
        self,
        item_ids: Iterable[int],
        per_page: int = ITEMS_PER_PAGE,
    ) -> Dict[int, Dict[str, Any]]:
        """Fetch many items in pages of ``id[]`` rather than one GET each.

        Returns ``{id: item}``; ids the server did not return are simply
        absent. A write step that pre-fetches this way trades ~99 % of its
        reads for a wider read-modify-write window, which is acceptable on a
        single-writer archive.
        """
        ids = [int(item_id) for item_id in item_ids]
        found: Dict[int, Dict[str, Any]] = {}
        for start in range(0, len(ids), per_page):
            chunk = ids[start:start + per_page]
            response = self._list_page({"id[]": chunk, "per_page": len(chunk), "page": 1})
            for item in response.json() or []:
                try:
                    found[int(item["o:id"])] = item
                except (KeyError, TypeError, ValueError):
                    continue
        return found

    def search_items_by_property(
        self,
        property_id: int,
        value: str,
        per_page: int = 1,
        **extra_params: Any,
    ) -> List[Dict[str, Any]]:
        """Search items whose *property_id* equals *value* (Omeka 'eq' query).

        Raises:
            OmekaRequestError: the search could not be answered. An empty list
                means "no match"; a transport failure is not that and must
                not be counted as one.
        """
        params: Dict[str, Any] = {
            "property[0][property]": property_id,
            "property[0][type]": "eq",
            "property[0][text]": value,
            "per_page": per_page,
            **extra_params,
        }
        try:
            response = self._list_page(params)
        except requests.RequestException as exc:
            LOGGER.error("Error searching items by property %s=%r: %s", property_id, value, exc)
            raise OmekaRequestError(f"search on property {property_id} failed: {exc}") from exc
        result = response.json()
        return result if isinstance(result, list) else []

    # ------------------------------------------------------------------
    # Single resources
    # ------------------------------------------------------------------

    def get_item(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single item by ID. Returns ``None`` on HTTP errors."""
        url = f"{self.base_url}/items/{item_id}"
        try:
            resp = self.session.get(url, params=self._auth_params(), timeout=self.timeout)
            resp.raise_for_status()
            self._remember_version(resp)
            return resp.json()
        except requests.RequestException as exc:
            LOGGER.error("Error fetching item %s: %s", item_id, exc)
            return None

    def update_item(self, item_id: int, data: Dict[str, Any]) -> bool:
        """PATCH an item. Returns ``True`` on success.

        The payload must be the whole item: Omeka treats every property as one
        block and deletes any value not sent back once at least one value is.
        """
        url = f"{self.base_url}/items/{item_id}"
        headers = {"Content-Type": "application/json"}
        try:
            resp = self.session.patch(
                url, json=data, params=self._auth_params(), headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            self._remember_version(resp)
            return True
        except requests.RequestException as exc:
            LOGGER.error("Failed to update item %s: %s", item_id, exc)
            response = getattr(exc, "response", None)
            if response is not None:
                LOGGER.error("Response body: %s", response.text)
            else:
                LOGGER.error("No response — the retry budget was exhausted or the connection failed")
            return False

    def create_item(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """POST a new item to Omeka S. Returns the created item dict or ``None``."""
        url = f"{self.base_url}/items"
        headers = {"Content-Type": "application/json"}
        try:
            resp = self.session.post(
                url, json=data, params=self._auth_params(), headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            self._remember_version(resp)
            return resp.json()
        except requests.RequestException as exc:
            LOGGER.error("Failed to create item: %s", exc)
            if hasattr(exc, "response") and exc.response is not None:
                LOGGER.error("Response body: %s", exc.response.text)
            return None

    def get_item_set(self, item_set_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single item set by ID. Returns ``None`` on HTTP errors."""
        url = f"{self.base_url}/item_sets/{item_set_id}"
        try:
            resp = self.session.get(url, params=self._auth_params(), timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            LOGGER.error("Error fetching item set %s: %s", item_set_id, exc)
            return None

    def get_property_id(self, term: str) -> Optional[int]:
        """Resolve a vocabulary term (e.g. ``dcterms:abstract``) to its property ID.

        Property IDs vary between Omeka S installations, so scripts should
        resolve them at runtime rather than hardcoding numbers.
        """
        url = f"{self.base_url}/properties"
        try:
            resp = self.session.get(
                url, params={**self._auth_params(), "term": term}, timeout=self.timeout
            )
            resp.raise_for_status()
            results = resp.json()
            if results:
                return int(results[0]["o:id"])
            LOGGER.error("No property found for term %s", term)
            return None
        except requests.RequestException as exc:
            LOGGER.error("Error resolving property term %s: %s", term, exc)
            return None

    def get_resource(self, url: str) -> Optional[Dict[str, Any]]:
        """GET any Omeka S resource URL (e.g. media @id)."""
        try:
            resp = self.session.get(url, params=self._auth_params(), timeout=self.timeout)
            resp.raise_for_status()
            self._remember_version(resp)
            return resp.json()
        except requests.RequestException as exc:
            LOGGER.error("Error fetching resource %s: %s", url, exc)
            return None
