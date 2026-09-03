"""The shared resource-link transaction: dedup, provenance annotation, recovery."""

import json
from unittest.mock import MagicMock

from common.checkpoint import JsonCheckpoint, checkpoint_path_for, read_checkpoint_context
from common.iwac_config import (
    DCTERMS_SUBJECT_PROPERTY_ID,
    IWAC_NER_MODEL_PROPERTY_ID,
    model_annotation_value,
)
from common.omeka_link_updater import (
    NER_MODEL_TERM,
    ResourceLinkSpec,
    provenance_model_key,
    update_item_resource_links,
)


BASE_URL = "https://islam.zmo.de/api"


def _client(item):
    client = MagicMock()
    client.base_url = BASE_URL
    client.get_item.return_value = item
    client.update_item.return_value = True
    return client


def _annotation():
    return model_annotation_value(
        BASE_URL, "deepseek-v4-flash-0731", IWAC_NER_MODEL_PROPERTY_ID, "AI Model - NER"
    )


def _subject_spec(ids, annotation=None):
    return ResourceLinkSpec(
        "dcterms:subject",
        DCTERMS_SUBJECT_PROPERTY_ID,
        ids,
        "Subject",
        annotation_term=NER_MODEL_TERM if annotation else None,
        annotation_value=annotation,
    )


def test_new_links_carry_the_model_annotation_and_existing_ones_do_not():
    hand_catalogued = {"type": "resource:item", "value_resource_id": 10}
    client = _client({"o:id": 7, "dcterms:subject": [hand_catalogued]})
    annotation = _annotation()

    result = update_item_resource_links(client, 7, [_subject_spec([10, 20], annotation)])

    assert result.status == "updated"
    assert result.added_by_term == {"dcterms:subject": 1}
    patched = client.update_item.call_args.args[1]
    values = {v["value_resource_id"]: v for v in patched["dcterms:subject"]}
    assert "@annotation" not in values[10]
    assert values[20]["@annotation"] == {NER_MODEL_TERM: [annotation]}
    assert values[20]["@annotation"][NER_MODEL_TERM][0]["property_id"] == IWAC_NER_MODEL_PROPERTY_ID


def test_annotation_is_a_copy_so_one_value_object_serves_every_link():
    client = _client({"o:id": 7})
    annotation = _annotation()

    update_item_resource_links(client, 7, [_subject_spec([1, 2], annotation)])

    patched = client.update_item.call_args.args[1]
    stamped = [v["@annotation"][NER_MODEL_TERM][0] for v in patched["dcterms:subject"]]
    assert stamped == [annotation, annotation]
    assert all(s is not annotation for s in stamped)


def test_spec_without_annotation_writes_plain_links():
    client = _client({"o:id": 7})

    update_item_resource_links(client, 7, [_subject_spec([1])])

    patched = client.update_item.call_args.args[1]
    assert "@annotation" not in patched["dcterms:subject"][0]


def test_dry_run_reports_the_annotation_without_patching():
    client = _client({"o:id": 7})

    result = update_item_resource_links(
        client, 7, [_subject_spec([1], _annotation())], dry_run=True
    )

    assert result.status == "would_update"
    client.update_item.assert_not_called()


def test_provenance_model_key_follows_the_reconciled_name_back_to_the_checkpoint(tmp_path):
    source = tmp_path / "item_set_1_processed_deepseek_v4_flash_0731.csv"
    source.write_text("o:id\n1\n", encoding="utf-8")
    JsonCheckpoint.open(checkpoint_path_for(source), {"model_key": "deepseek-v4-flash-0731"})
    reconciled = tmp_path / "item_set_1_processed_deepseek_v4_flash_0731_reconciled.csv"
    reconciled.write_text("o:id\n1\n", encoding="utf-8")

    assert provenance_model_key(reconciled) == "deepseek-v4-flash-0731"


def test_provenance_model_key_refuses_a_model_with_no_authority_item(tmp_path):
    """A registry key without an annotation item means 'ask', never a wrong stamp."""
    source = tmp_path / "items_enriched_x.csv"
    source.write_text("o:id\n1\n", encoding="utf-8")
    JsonCheckpoint.open(checkpoint_path_for(source), {"model_key": "gemma-4"})

    assert provenance_model_key(tmp_path / "items_enriched_x_reconciled.csv") is None
    assert provenance_model_key(tmp_path / "unrelated.csv") is None
    assert provenance_model_key(tmp_path / "missing_reconciled.csv") is None


def test_read_checkpoint_context_tolerates_garbage(tmp_path):
    output = tmp_path / "out.csv"
    checkpoint_path_for(output).write_text("{not json", encoding="utf-8")
    assert read_checkpoint_context(output) == {}
    checkpoint_path_for(output).write_text(json.dumps({"context": "nope"}), encoding="utf-8")
    assert read_checkpoint_context(output) == {}
