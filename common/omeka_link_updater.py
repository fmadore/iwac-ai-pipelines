"""Shared, idempotent Omeka resource-link updates.

Several pipelines enrich the same item with ``resource:item`` values.  Keeping
the fetch/mutate/PATCH transaction here gives them identical deduplication and
failure accounting without coupling their CSV formats.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from common.omeka_client import OmekaClient


def parse_resource_ids(value: str | None) -> list[int]:
    """Parse pipe-separated Omeka IDs, ignoring blank and malformed tokens."""
    resource_ids: list[int] = []
    for token in (value or "").split("|"):
        try:
            resource_ids.append(int(token.strip()))
        except ValueError:
            continue
    return resource_ids


@dataclass(frozen=True)
class ResourceLinkSpec:
    """Resource links to append to one Omeka property."""

    term: str
    property_id: int
    resource_ids: Sequence[int]
    property_label: str = ""


@dataclass(frozen=True)
class ResourceLinkUpdate:
    """Result of one fetch/mutate/PATCH transaction."""

    status: str
    added_by_term: Mapping[str, int] = field(default_factory=dict)

    @property
    def total_added(self) -> int:
        return sum(self.added_by_term.values())


def update_item_resource_links(
    client: OmekaClient,
    item_id: str | int,
    links: Sequence[ResourceLinkSpec],
    *,
    dry_run: bool = False,
    on_pre_write: Callable[[MutableMapping[str, Any]], None] | None = None,
) -> ResourceLinkUpdate:
    """Append missing resource links and PATCH only when data changed.

    Status is one of ``updated``, ``would_update``, ``unchanged``,
    ``not_found``, ``failed``, or ``invalid_id``.  Added counts are reported
    after a successful PATCH and for ``would_update``, so a dry run reports the
    same totals the live run would produce.

    ``on_pre_write`` receives the untouched item exactly once per item that is
    about to change — this is the only pre-write state that exists, and the
    only route back after a bulk PATCH.
    """
    try:
        numeric_item_id = int(item_id)
    except (TypeError, ValueError):
        return ResourceLinkUpdate("invalid_id")

    item_data = client.get_item(numeric_item_id)
    if not item_data:
        return ResourceLinkUpdate("not_found")

    # Snapshot before mutating: append_resource_links edits item_data in place.
    original = deepcopy(item_data)

    added_by_term = {
        link.term: OmekaClient.append_resource_links(
            item_data,
            link.term,
            link.property_id,
            list(link.resource_ids),
            property_label=link.property_label,
        )
        for link in links
    }
    if not any(added_by_term.values()):
        return ResourceLinkUpdate("unchanged")
    if on_pre_write is not None:
        on_pre_write(original)
    if dry_run:
        return ResourceLinkUpdate("would_update", added_by_term)
    if not client.update_item(numeric_item_id, item_data):
        return ResourceLinkUpdate("failed")
    return ResourceLinkUpdate("updated", added_by_term)
