"""
Shared Omeka S API client for all pipelines.

Provides authenticated access to the Omeka S REST API with:
- Paginated item retrieval
- Single item fetch and update (PATCH)
- Retry-capable HTTP sessions
- Environment-based configuration

Usage:
    from common.omeka_client import OmekaClient

    client = OmekaClient.from_env()
    items = client.get_items(item_set_id=123)
    item = client.get_item(456)
    client.update_item(456, item)
"""

import os
import logging
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)

ITEMS_PER_PAGE = 100


class OmekaClient:
    """Lightweight client for the Omeka S REST API."""

    def __init__(self, base_url: str, key_identity: str, key_credential: str):
        self.key_identity = key_identity
        self.key_credential = key_credential

        # Normalize base URL: ensure it ends with /api
        base = base_url.rstrip("/")
        if base.endswith("/api"):
            base = base[:-4]
        self.base_url = f"{base}/api"

        self.session = self._create_session()

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
        """Return a session with automatic retry on transient errors."""
        session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_items(
        self,
        item_set_id: int,
        per_page: int = ITEMS_PER_PAGE,
        **extra_params: Any,
    ) -> List[Dict[str, Any]]:
        """Fetch all items in an item set, handling pagination automatically."""
        url = f"{self.base_url}/items"
        params: Dict[str, Any] = {
            **self._auth_params(),
            "item_set_id": item_set_id,
            "per_page": per_page,
            "page": 1,
            **extra_params,
        }
        all_items: List[Dict[str, Any]] = []
        while True:
            resp = self.session.get(url, params=params)
            resp.raise_for_status()
            page_items = resp.json()
            if not page_items:
                break
            all_items.extend(page_items)
            if len(page_items) < per_page:
                break
            params["page"] += 1
        return all_items

    def get_item(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single item by ID. Returns ``None`` on HTTP errors."""
        url = f"{self.base_url}/items/{item_id}"
        try:
            resp = self.session.get(url, params=self._auth_params())
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            LOGGER.error("Error fetching item %s: %s", item_id, exc)
            return None

    def update_item(self, item_id: int, data: Dict[str, Any]) -> bool:
        """PATCH an item. Returns ``True`` on success."""
        url = f"{self.base_url}/items/{item_id}"
        headers = {"Content-Type": "application/json"}
        try:
            resp = self.session.patch(
                url, json=data, params=self._auth_params(), headers=headers
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            LOGGER.error("Failed to update item %s: %s", item_id, exc)
            if hasattr(exc, "response") and exc.response is not None:
                LOGGER.error("Response body: %s", exc.response.text)
            return False

    def get_item_set(self, item_set_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single item set by ID. Returns ``None`` on HTTP errors."""
        url = f"{self.base_url}/item_sets/{item_set_id}"
        try:
            resp = self.session.get(url, params=self._auth_params())
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            LOGGER.error("Error fetching item set %s: %s", item_set_id, exc)
            return None

    def get_resource(self, url: str) -> Optional[Dict[str, Any]]:
        """GET any Omeka S resource URL (e.g. media @id)."""
        try:
            resp = self.session.get(url, params=self._auth_params())
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            LOGGER.error("Error fetching resource %s: %s", url, exc)
            return None
