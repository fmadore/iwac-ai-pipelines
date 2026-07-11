"""Tests for common.omeka_client against a mocked requests session."""

from unittest.mock import MagicMock

import pytest
import requests

from common.omeka_client import OmekaClient


def make_client() -> OmekaClient:
    client = OmekaClient("https://example.org/api", "id", "cred")
    client.session = MagicMock()
    return client


def response_with(json_data, status=200):
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.status_code = status
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


def test_base_url_normalization():
    assert OmekaClient("https://example.org", "i", "c").base_url == "https://example.org/api"
    assert OmekaClient("https://example.org/api", "i", "c").base_url == "https://example.org/api"
    assert OmekaClient("https://example.org/api/", "i", "c").base_url == "https://example.org/api"


def test_get_items_paginates_until_short_page():
    client = make_client()
    page1 = [{"o:id": i} for i in range(100)]
    page2 = [{"o:id": 100}]
    client.session.get.side_effect = [response_with(page1), response_with(page2)]

    items = client.get_items(item_set_id=5)

    assert len(items) == 101
    assert client.session.get.call_count == 2


def test_every_request_carries_a_timeout():
    client = make_client()
    client.session.get.return_value = response_with([])
    client.get_items(item_set_id=5)
    _, kwargs = client.session.get.call_args
    assert kwargs.get("timeout") == client.timeout


def test_get_item_returns_none_on_http_error():
    client = make_client()
    client.session.get.return_value = response_with({}, status=404)
    assert client.get_item(42) is None


def test_update_item_returns_false_on_http_error():
    client = make_client()
    client.session.patch.return_value = response_with({}, status=500)
    assert client.update_item(42, {"o:id": 42}) is False


def test_update_item_success():
    client = make_client()
    client.session.patch.return_value = response_with({"o:id": 42})
    assert client.update_item(42, {"o:id": 42}) is True


def test_get_property_id_resolves_term():
    client = make_client()
    client.session.get.return_value = response_with([{"o:id": 19, "o:term": "dcterms:abstract"}])
    assert client.get_property_id("dcterms:abstract") == 19


def test_get_property_id_unknown_term():
    client = make_client()
    client.session.get.return_value = response_with([])
    assert client.get_property_id("dcterms:doesnotexist") is None


def test_from_env_requires_configuration(monkeypatch):
    for var in ("OMEKA_BASE_URL", "OMEKA_KEY_IDENTITY", "OMEKA_KEY_CREDENTIAL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("common.omeka_client.load_dotenv", lambda: None)
    with pytest.raises(ValueError):
        OmekaClient.from_env()
