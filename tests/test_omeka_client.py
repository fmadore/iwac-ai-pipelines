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


def test_upsert_property_value_appends_when_missing():
    item = {"o:id": 1}
    changed = OmekaClient.upsert_property_value(item, "bibo:content", 91, "text", property_label="content")
    assert changed is True
    assert item["bibo:content"] == [{
        "type": "literal",
        "property_id": 91,
        "property_label": "content",
        "is_public": True,
        "@value": "text",
    }]


def test_upsert_property_value_replaces_matching_literal():
    item = {"bibo:content": [{"type": "literal", "property_id": 91, "@value": "old"}]}
    assert OmekaClient.upsert_property_value(item, "bibo:content", 91, "new") is True
    assert item["bibo:content"][0]["@value"] == "new"
    assert len(item["bibo:content"]) == 1


def test_upsert_property_value_noop_when_identical():
    item = {"bibo:content": [{"type": "literal", "property_id": 91, "@value": "same"}]}
    assert OmekaClient.upsert_property_value(item, "bibo:content", 91, "same") is False


def test_upsert_property_value_leaves_other_properties_alone():
    linked = {"type": "resource:item", "property_id": 91, "value_resource_id": 5}
    item = {"bibo:content": [linked]}
    OmekaClient.upsert_property_value(item, "bibo:content", 91, "text")
    assert linked in item["bibo:content"]
    assert any(v.get("@value") == "text" for v in item["bibo:content"])


def test_append_resource_links_skips_duplicates():
    item = {"dcterms:spatial": [{"type": "resource:item", "property_id": 40, "value_resource_id": 7}]}
    added = OmekaClient.append_resource_links(item, "dcterms:spatial", 40, [7, 8])
    assert added == 1
    ids = [v["value_resource_id"] for v in item["dcterms:spatial"]]
    assert ids == [7, 8]


def test_append_resource_links_creates_term():
    item = {}
    added = OmekaClient.append_resource_links(item, "dcterms:subject", 3, [1, 2], property_label="Subject")
    assert added == 2
    assert all(v["type"] == "resource:item" for v in item["dcterms:subject"])


def test_search_items_by_property_builds_eq_query():
    client = make_client()
    client.session.get.return_value = response_with([{"o:id": 9}])
    items = client.search_items_by_property(10, "ABC-123")
    assert items == [{"o:id": 9}]
    _, kwargs = client.session.get.call_args
    params = kwargs["params"]
    assert params["property[0][property]"] == 10
    assert params["property[0][type]"] == "eq"
    assert params["property[0][text]"] == "ABC-123"


def test_search_items_by_property_returns_empty_on_error():
    client = make_client()
    client.session.get.return_value = response_with([], status=500)
    assert client.search_items_by_property(10, "x") == []
