"""A rerun must never overwrite a curator's correction with a stale local file.

AI_summary/01 never empties TXT/, so step 03 used to upload every summary still
on disk: a summary corrected in Omeka since its upload differed from the local
file, so change detection rewrote it. The upload ledger records what the step
wrote, and these tests pin each decision it makes.
"""

import io
from unittest.mock import MagicMock

from rich.console import Console

from common.omeka_text_updater import (
    PropertyTarget,
    TextUpdate,
    UploadLedger,
    run_text_updates,
)

BASE_URL = "https://archive.example/api"
FR = PropertyTarget(term="bibo:shortDescription", property_id=52, property_label="shortDescription",
                    language="fr", adopt_untagged=True)
EN = PropertyTarget(term="bibo:shortDescription", property_id=52, property_label="shortDescription",
                    language="en")


def summary(item_id, fr, en="An English summary."):
    update = TextUpdate(label=f"{item_id}.txt", item_id=item_id, text=fr)
    update.extra_values.append((EN, en))
    return update


def item(item_id, fr=None, en=None, untagged=None):
    values = []
    for text, language in ((fr, "fr"), (en, "en"), (untagged, None)):
        if text is not None:
            value = {"type": "literal", "property_id": 52, "@value": text}
            if language:
                value["@language"] = language
            values.append(value)
    return {"o:id": item_id, "bibo:shortDescription": values}


def run(client, updates, ledger, *, replace=False):
    def record(update, status):
        if status in ("updated", "unchanged"):
            ledger.record(update, FR)

    return run_text_updates(
        client, updates, FR,
        console=Console(file=io.StringIO(), force_terminal=False),
        require_confirmation=False,
        check=None if replace else (lambda update, data: ledger.held_back_reason(update, FR, data)),
        on_result=record,
    )


def client_for(data):
    client = MagicMock()
    client.base_url = BASE_URL
    client.get_item.return_value = data
    client.update_item.return_value = True
    return client


def test_an_item_with_no_summary_is_written_and_recorded(tmp_path):
    ledger = UploadLedger.load(tmp_path / "ledger.jsonl", BASE_URL)
    update = summary(7, "Un résumé.")

    stats = run(client_for(item(7)), [update], ledger)

    assert stats["updated"] == 1
    assert UploadLedger.load(tmp_path / "ledger.jsonl", BASE_URL).already_written(update, FR)


def test_a_summary_already_uploaded_is_not_even_fetched(tmp_path):
    """The step-03 filter: unchanged local text means nothing to do."""
    ledger = UploadLedger.load(tmp_path / "ledger.jsonl", BASE_URL)
    update = summary(7, "Un résumé.")
    ledger.record(update, FR)

    assert ledger.already_written(update, FR)
    assert not ledger.already_written(summary(7, "Un autre résumé."), FR)
    assert not ledger.already_written(summary(8, "Un résumé."), FR)


def test_a_curator_edit_made_after_upload_is_kept(tmp_path):
    ledger = UploadLedger.load(tmp_path / "ledger.jsonl", BASE_URL)
    ledger.record(summary(7, "Résumé de la machine."), FR)
    client = client_for(item(7, fr="Résumé corrigé par une archiviste.", en="An English summary."))

    stats = run(client, [summary(7, "Résumé régénéré.")], ledger)

    assert stats["held_back"] == 1 and stats["updated"] == 0
    client.update_item.assert_not_called()


def test_a_regenerated_summary_replaces_what_this_step_wrote(tmp_path):
    ledger = UploadLedger.load(tmp_path / "ledger.jsonl", BASE_URL)
    ledger.record(summary(7, "Résumé de la machine."), FR)
    client = client_for(item(7, fr="Résumé de la machine.", en="An English summary."))

    stats = run(client, [summary(7, "Résumé régénéré.")], ledger)

    assert stats["updated"] == 1
    written = client.update_item.call_args.args[1]["bibo:shortDescription"]
    assert {v["@language"]: v["@value"] for v in written}["fr"] == "Résumé régénéré."


def test_a_value_this_step_never_wrote_is_held_back(tmp_path):
    """The ~12,300 untagged summaries predate the ledger: not ours to overwrite."""
    ledger = UploadLedger.load(tmp_path / "ledger.jsonl", BASE_URL)
    client = client_for(item(7, untagged="Résumé de 2025."))

    stats = run(client, [summary(7, "Résumé régénéré.")], ledger)

    assert stats["held_back"] == 1
    client.update_item.assert_not_called()


def test_replace_existing_overwrites_and_then_records(tmp_path):
    ledger = UploadLedger.load(tmp_path / "ledger.jsonl", BASE_URL)
    update = summary(7, "Résumé régénéré.")
    client = client_for(item(7, untagged="Résumé de 2025."))

    stats = run(client, [update], ledger, replace=True)

    assert stats["updated"] == 1
    assert ledger.already_written(update, FR)


def test_a_summary_already_in_omeka_is_adopted_into_the_ledger(tmp_path):
    """Uploaded before the ledger existed and untouched since: recorded, not rewritten."""
    ledger = UploadLedger.load(tmp_path / "ledger.jsonl", BASE_URL)
    update = summary(7, "Un résumé.")
    client = client_for(item(7, fr="Un résumé.", en="An English summary."))

    stats = run(client, [update], ledger)

    assert stats["held_back"] == 0 and stats["unchanged"] == 1
    assert ledger.already_written(update, FR)


def test_the_ledger_belongs_to_one_omeka_instance(tmp_path):
    path = tmp_path / "ledger.jsonl"
    update = summary(7, "Un résumé.")
    UploadLedger.load(path, "https://staging.example/api").record(update, FR)

    assert not UploadLedger.load(path, BASE_URL).already_written(update, FR)


def test_a_torn_final_line_is_skipped(tmp_path):
    path = tmp_path / "ledger.jsonl"
    update = summary(7, "Un résumé.")
    UploadLedger.load(path, BASE_URL).record(update, FR)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"item_id": 8, "base_u')

    assert UploadLedger.load(path, BASE_URL).already_written(update, FR)


# --- AI_summary/03 end to end -------------------------------------------------

def _load(name, relative):
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _generate_summaries(pipeline_dir):
    """Run the real step-02 loop with a fake model, as the upload step will see it."""
    from common.checkpoint import JsonCheckpoint

    generator = _load("summary_generator_for_ledger", "AI_summary/02_AI_generate_summaries.py")
    source = pipeline_dir / "TXT"
    source.mkdir()
    for item_id in (1, 2):
        (source / f"{item_id}.txt").write_text(f"article {item_id}", encoding="utf-8")
    french, english = pipeline_dir / "Summaries_FR_TXT", pipeline_dir / "Summaries_EN_TXT"
    checkpoint = JsonCheckpoint.open(
        french / ".summary_checkpoint.json", {"model_key": "gpt-6-luna", "prompt": "p"}
    )
    model = MagicMock()
    model.generate_structured.side_effect = lambda _s, user, _schema: generator.BilingualSummary(
        summary_fr=f"Résumé {user[-1]}.", summary_en=f"Summary {user[-1]}."
    )
    generator.process_txt_files(model, str(source), str(french), str(english), "system", checkpoint)


class _Archive:
    """Item 1 has no summary; item 2 carries one a curator wrote."""

    base_url = BASE_URL

    def __init__(self):
        self.items = {1: item(1), 2: item(2, fr="Résumé rédigé par une archiviste.")}
        self.fetched, self.patched = [], []

    def get_property_id(self, term):
        return 52

    def get_item(self, item_id):
        self.fetched.append(item_id)
        return json_copy(self.items[item_id])

    def update_item(self, item_id, data):
        self.patched.append(item_id)
        self.items[item_id] = data
        return True


def json_copy(value):
    import json

    return json.loads(json.dumps(value))


def test_summary_upload_keeps_a_curator_edit_and_skips_what_it_uploaded(tmp_path, monkeypatch):
    import sys

    step03 = _load("summary_uploader_for_ledger", "AI_summary/03_omeka_update_summaries.py")
    _generate_summaries(tmp_path)
    archive = _Archive()
    monkeypatch.setattr(step03, "__file__", str(tmp_path / "03_omeka_update_summaries.py"))
    monkeypatch.setattr(step03.OmekaClient, "from_env", classmethod(lambda cls: archive))
    monkeypatch.setattr(step03, "console", Console(file=io.StringIO(), force_terminal=False))

    def upload(*flags):
        monkeypatch.setattr(sys, "argv", ["03", "--yes", "--no-backup", *flags])
        archive.fetched.clear()
        archive.patched.clear()
        return step03.main()

    assert upload("--dry-run") == 0
    assert archive.patched == []
    assert not (tmp_path / "Summaries_FR_TXT" / step03.LEDGER_NAME).exists()

    assert upload() == 0
    assert archive.patched == [1]
    assert "archiviste" in json_copy(archive.items[2])["bibo:shortDescription"][0]["@value"]

    assert upload() == 0
    assert archive.fetched == [2]          # item 1 is in the ledger: not even fetched
    assert archive.patched == []

    assert upload("--replace-existing") == 0
    assert archive.patched == [2]
    assert upload() == 0 and archive.fetched == []
