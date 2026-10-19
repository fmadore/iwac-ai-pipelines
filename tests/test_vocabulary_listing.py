"""The IWAC property listing must stay complete past one API page."""

import re

import pytest

from common.iwac_config import (
    PROPERTY_PAGE_SIZE,
    list_vocabulary_properties,
    resolve_property_ids,
)


class FakeClient:
    """Serves ``count`` properties through Omeka's ``per_page``/``page`` paging."""

    base_url = "https://example.test/api"

    def __init__(self, count, fail_page=None):
        self.terms = [f"iwac:term{i}" for i in range(count)]
        self.fail_page = fail_page
        self.pages = []

    def get_resource(self, url):
        per_page = int(re.search(r"per_page=(\d+)", url).group(1))
        page = int(re.search(r"[?&]page=(\d+)", url).group(1))
        self.pages.append(page)
        if page == self.fail_page:
            return None
        start = (page - 1) * per_page
        return [
            {"o:term": term, "o:id": start + i}
            for i, term in enumerate(self.terms[start:start + per_page])
        ]


def test_listing_spans_several_pages():
    client = FakeClient(2 * PROPERTY_PAGE_SIZE + 30)
    listed = list_vocabulary_properties(client)
    assert len(listed) == 2 * PROPERTY_PAGE_SIZE + 30
    assert client.pages == [1, 2, 3]


def test_exact_multiple_ends_on_an_empty_page():
    client = FakeClient(PROPERTY_PAGE_SIZE)
    assert len(list_vocabulary_properties(client)) == PROPERTY_PAGE_SIZE
    assert client.pages == [1, 2]


def test_a_term_past_the_first_page_resolves():
    client = FakeClient(PROPERTY_PAGE_SIZE + 5)
    last = f"iwac:term{PROPERTY_PAGE_SIZE + 4}"
    assert resolve_property_ids(client, [last]) == {last: PROPERTY_PAGE_SIZE + 4}


def test_an_unreadable_page_fails_rather_than_truncating():
    client = FakeClient(PROPERTY_PAGE_SIZE + 5, fail_page=2)
    with pytest.raises(RuntimeError, match="page 2"):
        list_vocabulary_properties(client)
