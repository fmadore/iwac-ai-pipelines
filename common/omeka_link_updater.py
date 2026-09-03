"""Shared, idempotent Omeka resource-link updates.

Several pipelines enrich the same item with ``resource:item`` values.  Keeping
the fetch/mutate/PATCH transaction here gives them identical deduplication,
provenance annotation and failure accounting without coupling their CSV
formats.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from common.checkpoint import read_checkpoint_context
from common.iwac_config import AI_MODEL_ITEMS
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
    """Resource links to append to one Omeka property.

    ``annotation_term`` / ``annotation_value`` name the value annotation to
    attach to every link this spec *adds* — typically ``iwac:nerModel`` and the
    ``resource:item`` object built by ``iwac_config.model_annotation_value``.
    Links already on the item are left untouched, annotation included: they
    may be hand-catalogued, and re-stamping them would claim otherwise.
    """

    term: str
    property_id: int
    resource_ids: Sequence[int]
    property_label: str = ""
    annotation_term: str | None = None
    annotation_value: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ResourceLinkUpdate:
    """Result of one fetch/mutate/PATCH transaction."""

    status: str
    added_by_term: Mapping[str, int] = field(default_factory=dict)

    @property
    def total_added(self) -> int:
        return sum(self.added_by_term.values())


def _linked_ids(values: Any) -> set[int]:
    """The ``value_resource_id`` set already present on one property."""
    linked: set[int] = set()
    if not isinstance(values, list):
        return linked
    for entry in values:
        if isinstance(entry, dict) and "value_resource_id" in entry:
            try:
                linked.add(int(entry["value_resource_id"]))
            except (TypeError, ValueError):
                continue
    return linked


def _annotate_new_links(
    item_data: Mapping[str, Any],
    link: ResourceLinkSpec,
    ids_before: set[int],
) -> None:
    """Attach the spec's annotation to the links appended by this transaction."""
    if not link.annotation_term or link.annotation_value is None:
        return
    for entry in item_data.get(link.term, []):
        if not isinstance(entry, dict):
            continue
        try:
            resource_id = int(entry.get("value_resource_id"))
        except (TypeError, ValueError):
            continue
        if resource_id in ids_before:
            continue
        entry["@annotation"] = {link.annotation_term: [dict(link.annotation_value)]}


def update_item_resource_links(
    client: OmekaClient,
    item_id: str | int,
    links: Sequence[ResourceLinkSpec],
    *,
    dry_run: bool = False,
    on_pre_write: Callable[[MutableMapping[str, Any]], None] | None = None,
    item_data: MutableMapping[str, Any] | None = None,
) -> ResourceLinkUpdate:
    """Append missing resource links and PATCH only when data changed.

    Pass *item_data* when the item was already fetched (a batch pre-fetch via
    ``get_items_by_ids``); otherwise it is fetched here.

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

    if item_data is None:
        item_data = client.get_item(numeric_item_id)
    if not item_data:
        return ResourceLinkUpdate("not_found")

    # Snapshot before mutating: append_resource_links edits item_data in place.
    original = deepcopy(item_data)

    added_by_term: dict[str, int] = {}
    for link in links:
        ids_before = _linked_ids(item_data.get(link.term))
        added_by_term[link.term] = OmekaClient.append_resource_links(
            item_data,
            link.term,
            link.property_id,
            list(link.resource_ids),
            property_label=link.property_label,
        )
        if added_by_term[link.term]:
            _annotate_new_links(item_data, link, ids_before)
    if not any(added_by_term.values()):
        return ResourceLinkUpdate("unchanged")
    if on_pre_write is not None:
        on_pre_write(original)
    if dry_run:
        return ResourceLinkUpdate("would_update", added_by_term)
    if not client.update_item(numeric_item_id, item_data):
        return ResourceLinkUpdate("failed")
    return ResourceLinkUpdate("updated", added_by_term)


NER_MODEL_TERM = "iwac:nerModel"
NER_MODEL_LABEL = "AI Model - NER"


def provenance_model_key(reconciled_csv: str | Path) -> str | None:
    """Recover which model produced the keywords in a ``*_reconciled.csv``.

    The reconciliation step derives its output name from the AI step's output
    (``<name>_reconciled.csv``), and the AI step leaves a checkpoint beside its
    own file naming the model that ran. Following that chain back lets the
    write step stamp the right provenance without asking — but only when the
    key is one an authority item exists for; otherwise the caller must ask.
    """
    reconciled_csv = Path(reconciled_csv)
    stem, suffix = reconciled_csv.stem, reconciled_csv.suffix
    if not stem.endswith("_reconciled"):
        return None
    source = reconciled_csv.with_name(stem[: -len("_reconciled")] + suffix)
    model_key = read_checkpoint_context(source).get("model_key")
    return model_key if model_key in AI_MODEL_ITEMS else None
