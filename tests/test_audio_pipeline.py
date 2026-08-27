"""Audio media discovery, Gemini transcription orchestration, and what step 03
writes back — including which model an ``iwac:transcriptionModel`` annotation is
allowed to name."""

import importlib.util
import io
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from segments import GENERATOR_FIELD, read_header, write_transcription


PIPELINE = Path(__file__).resolve().parent.parent / "AI_audio_summary"


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, PIPELINE / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


downloader_module = load_script("audio_media_downloader", "01_omeka_media_downloader.py")
transcription_module = load_script("gemini_audio_transcriber", "02_AI_transcribe_audio.py")


def bare_transcriber():
    worker = transcription_module.AudioTranscriber.__new__(
        transcription_module.AudioTranscriber
    )
    worker.client = MagicMock()
    worker.model = "gemini-flash-latest"
    worker.rate_limiter = MagicMock()
    worker.transcription_prompt = "Transcribe."
    worker.last_failure_reason = None
    return worker


def test_media_discovery_keeps_only_supported_original_urls(tmp_path):
    client = MagicMock()
    client.get_resource.side_effect = [
        {"o:source": "recording.mp3", "o:original_url": "https://x/recording.mp3"},
        {"o:source": "scan.pdf", "o:original_url": "https://x/scan.pdf"},
        {"o:source": "missing-url.wav"},
    ]
    downloader = downloader_module.MediaDownloader(client, tmp_path)

    urls = downloader._find_media_urls({
        "o:media": [{"@id": "m1"}, {"@id": "m2"}, {"@id": "m3"}],
    })

    assert [(url, media["o:source"]) for url, media in urls] == [
        ("https://x/recording.mp3", "recording.mp3"),
    ]


def test_existing_media_is_resumed_without_download(tmp_path):
    existing = tmp_path / "recording.mp3"
    existing.write_bytes(b"complete")
    client = MagicMock()
    client.get_item.return_value = {
        "o:id": 7,
        "o:media": [{"@id": "m1"}],
    }
    client.get_resource.return_value = {
        "o:source": "recording.mp3",
        "o:original_url": "https://x/recording.mp3",
    }
    downloader = downloader_module.MediaDownloader(client, tmp_path)

    with patch.object(downloader, "download_file") as download:
        result = downloader.process_item({"o:id": 7})

    assert result == ("7", [str(existing)])
    download.assert_not_called()


def test_large_media_uses_upload_transport():
    worker = bare_transcriber()
    path = MagicMock()
    path.stat.return_value.st_size = transcription_module.INLINE_REQUEST_LIMIT_BYTES + 1
    uploaded = MagicMock()
    worker._upload_via_files_api = MagicMock(return_value=uploaded)

    assert worker._prepare_media_part(path, "audio/mpeg") == (uploaded, uploaded)
    worker._upload_via_files_api.assert_called_once_with(path, "audio/mpeg")


def test_transcription_always_deletes_uploaded_file(tmp_path, monkeypatch):
    worker = bare_transcriber()
    audio_path = tmp_path / "long.mp3"
    audio_path.write_bytes(b"audio")
    uploaded = MagicMock(name="files/123")
    worker._prepare_media_part = MagicMock(return_value=(uploaded, uploaded))
    worker._transcribe_with_retries = MagicMock(return_value="transcript")
    monkeypatch.setattr(transcription_module, "get_mime_type", lambda path: "audio/mpeg")
    cleanup = MagicMock()
    monkeypatch.setattr(transcription_module, "delete_uploaded_file", cleanup)

    assert worker.transcribe_audio(audio_path) == "transcript"

    cleanup.assert_called_once_with(worker.client, uploaded)


def test_retryable_response_is_retried_then_succeeds(monkeypatch, tmp_path):
    worker = bare_transcriber()
    worker.client.models.generate_content.side_effect = [MagicMock(), MagicMock()]
    worker._response_to_text = MagicMock(side_effect=[
        transcription_module._RetryableResponse("RECITATION"),
        "transcript",
    ])
    monkeypatch.setattr(transcription_module.random, "uniform", lambda *_: 0)
    monkeypatch.setattr(transcription_module.time, "sleep", lambda *_: None)

    result = worker._transcribe_with_retries(
        MagicMock(), "prompt", tmp_path / "audio.mp3", 2,
    )

    assert result == "transcript"
    assert worker.client.models.generate_content.call_count == 2
    assert worker.rate_limiter.wait.call_count == 2


# ---------------------------------------------------------------------------
# Step 03: transcription provenance
#
# The audio pipeline can fill Transcriptions/ from two scripts and four models,
# and only one of those models has an Omeka authority item to point an
# iwac:transcriptionModel annotation at. Everything below is about that gap:
# what may be cited, what may not, and what the step must refuse to guess.
# ---------------------------------------------------------------------------

voxtral_module = load_script("voxtral_audio_transcriber", "02b_AI_transcribe_audio_voxtral.py")
updater = load_script("audio_transcription_updater", "03_omeka_transcription_updater.py")


def make_transcript(folder, stem, generator, body="Bismillah, wa alhamdulillah."):
    """Write a transcription file the way 02 and 02b write theirs."""
    return write_transcription(body, Path(f"{stem}.mp3"), folder, generator=generator)


def groups_for(folder):
    return updater.TranscriptionProcessor(folder).get_transcription_groups()


def resolve(counts, *, requested=None, skip=False, assume_yes=True):
    return updater.resolve_model_key(
        Counter(counts), requested=requested, skip=skip, assume_yes=assume_yes
    )


@pytest.fixture
def transcript(monkeypatch):
    """Capture what the step tells the operator, unwrapped and uncoloured."""
    stream = io.StringIO()
    monkeypatch.setattr(
        updater, "console", Console(file=stream, width=200, no_color=True, highlight=False)
    )
    return stream


# --- The entry point reads argv --------------------------------------------

def test_the_write_step_refuses_help_instead_of_running(monkeypatch):
    """A write entry point must parse argv, or it treats a typo as consent."""
    monkeypatch.setattr(sys, "argv", ["03_omeka_transcription_updater.py", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        updater.main()
    assert excinfo.value.code == 0


def test_asserting_a_model_and_skipping_the_annotation_is_an_error(monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        ["03", "--model", "gemini-3.7-flash", "--no-model-annotation"],
    )
    with pytest.raises(SystemExit) as excinfo:
        updater.main()
    assert excinfo.value.code == 2


def test_the_annotation_property_is_the_transcription_one():
    """Not iwac:ocrModel or iwac:summaryModel — 315 is the transcription one."""
    from common.iwac_config import IWAC_TRANSCRIPTION_MODEL_PROPERTY_ID

    assert IWAC_TRANSCRIPTION_MODEL_PROPERTY_ID == 315
    assert updater.TRANSCRIPTION_MODEL_TERM == "iwac:transcriptionModel"
    assert updater.CONTENT_TERM == "bibo:content"


# --- Which headers name something citable ----------------------------------

@pytest.mark.parametrize("generator,expected", [
    ("Google gemini-3.7-flash", "gemini-3.7-flash"),
    ("Google gemini-3.1-pro", "gemini-3.1-pro"),
    # Rolling aliases report their own version as "Gemini Pro Latest", so an
    # annotation through one asserts a release the run never confirmed.
    ("Google gemini-pro-latest", None),
    ("Google gemini-flash-lite-latest", None),
    # No authority item yet.
    ("Mistral voxtral-mini-2602", None),
    ("unknown model", None),
    ("", None),
])
def test_only_a_pinned_model_with_an_authority_item_can_be_named(generator, expected):
    assert updater.annotation_key_for(generator) == expected


def test_every_model_02_offers_is_citable_or_an_acknowledged_alias():
    """Adding a pinned model to 02 without an authority item breaks 03.

    03 refuses a folder whose header names something ``AI_MODEL_ITEMS`` does not
    hold, so a new pinned id here has to arrive with its Omeka item. A rolling
    alias is the deliberate exception and must stay absent from that registry.
    """
    from common.iwac_config import AI_MODEL_ITEMS

    assert transcription_module.DEFAULT_MODEL in transcription_module.ALLOWED_MODELS
    for model in transcription_module.ALLOWED_MODELS:
        if "latest" in model:
            assert model not in AI_MODEL_ITEMS, f"{model} is rolling and cannot be cited"
        else:
            assert model in AI_MODEL_ITEMS, f"{model} has no Omeka authority item"


def test_voxtral_still_has_no_authority_item():
    """Why --model is optional rather than required.

    The day one is created, ``02b``'s output becomes annotatable and this test
    is the thing that says so.
    """
    from common.iwac_config import AI_MODEL_ITEMS

    assert voxtral_module.MODEL == "voxtral-mini-2602"
    assert voxtral_module.MODEL not in AI_MODEL_ITEMS


# --- Reading the header back off disk ---------------------------------------

def test_the_generator_survives_a_write_read_round_trip(tmp_path):
    path = make_transcript(tmp_path, "khutba", "Google gemini-3.7-flash")
    header = read_header(path)
    assert header[GENERATOR_FIELD] == "Google gemini-3.7-flash"
    assert header["Transcription of"] == "khutba.mp3"


def test_transcript_text_is_never_read_as_a_provenance_claim(tmp_path):
    path = make_transcript(
        tmp_path, "khutba", "Google gemini-3.7-flash",
        body="Generated using: a model the speaker mentioned",
    )
    assert read_header(path)[GENERATOR_FIELD] == "Google gemini-3.7-flash"


def test_a_file_without_a_separator_has_no_header(tmp_path):
    path = tmp_path / "loose_transcription.txt"
    path.write_text("Generated using: not a header, just the first line\n", encoding="utf-8")
    assert read_header(path) == {}


def test_every_file_of_an_identifier_is_read_not_only_the_first(tmp_path):
    """One recording can be several media files, and so several transcripts.

    Counting one header per identifier would miss a group half-transcribed by
    02 and half by 02b — exactly the mix that misattributes silently.
    """
    make_transcript(tmp_path, "iwac-audio-0001-1", "Google gemini-3.7-flash")
    make_transcript(tmp_path, "iwac-audio-0001-2", "Mistral voxtral-mini-2602")

    groups = groups_for(tmp_path)

    assert list(groups) == ["iwac-audio-0001"]
    assert updater.count_generators(groups) == Counter({
        "Google gemini-3.7-flash": 1,
        "Mistral voxtral-mini-2602": 1,
    })


def test_a_headerless_transcript_counts_as_unrecorded(tmp_path):
    (tmp_path / "loose_transcription.txt").write_text("plain text", encoding="utf-8")
    counts = updater.count_generators(groups_for(tmp_path))
    assert counts == Counter({updater.UNRECORDED_GENERATOR: 1})


# --- And what of it reaches bibo:content ------------------------------------
#
# The header is read for the annotation and then left behind: bibo:content is
# the archive's full text, exported to Hugging Face as OCR and indexed for
# search, so a header inside it is indexed as though a speaker had said it.

def joined(folder, identifier):
    """Run 03's grouping and joining, as resolve_updates() would."""
    processor = updater.TranscriptionProcessor(folder)
    return processor.read_and_join_transcriptions(processor.get_transcription_groups()[identifier])


def test_the_uploaded_text_carries_no_provenance_header(tmp_path):
    make_transcript(
        tmp_path, "iwac-audio-0001-1", "Google gemini-3.7-flash",
        body="[00:00:01] Speaker 1: Bismillah.",
    )

    text = joined(tmp_path, "iwac-audio-0001")

    assert text == "[00:00:01] Speaker 1: Bismillah."
    assert GENERATOR_FIELD not in text
    assert "Transcription of" not in text
    assert "=" * 50 not in text


def test_voxtral_extra_header_fields_are_stripped_too(tmp_path):
    """02b adds Language/Diarization lines; they are header, not transcript."""
    write_transcription(
        "Speaker 1: Salaam.", Path("iwac-audio-0002-1.mp3"), tmp_path,
        generator="Mistral voxtral-mini-2602",
        extra_fields=[("Language", "Auto-detect"), ("Diarization", "ON")],
    )

    text = joined(tmp_path, "iwac-audio-0002")

    assert text == "Speaker 1: Salaam."
    assert "Diarization" not in text


def test_every_joined_segment_is_stripped_not_only_the_first(tmp_path):
    """One header per segment file, so stripping once leaves the rest inline."""
    for segment in (1, 2, 3):
        make_transcript(
            tmp_path, f"iwac-audio-0003-{segment}", "Google gemini-3.7-flash",
            body=f"Segment {segment} speech.",
        )

    text = joined(tmp_path, "iwac-audio-0003")

    assert GENERATOR_FIELD not in text
    assert "Transcription of" not in text
    for segment in (1, 2, 3):
        assert f"[Part {segment}]" in text
        assert f"Segment {segment} speech." in text


def test_a_file_without_a_separator_is_uploaded_whole(tmp_path):
    """The same refusal to guess that leaves it unattributed keeps its text.

    Dropping the first lines of a headerless file would silently truncate a
    transcript — the mirror of reading them as a provenance claim.
    """
    body = "Speaker 1: no header here.\nSpeaker 2: none at all."
    (tmp_path / "iwac-audio-0004-1_transcription.txt").write_text(
        body + "\n", encoding="utf-8",
    )

    assert joined(tmp_path, "iwac-audio-0004") == body


def test_the_header_the_annotation_was_read_from_stays_on_disk(tmp_path):
    """Stripped from the upload, not deleted: the file stays auditable."""
    path = make_transcript(tmp_path, "iwac-audio-0005-1", "Google gemini-3.7-flash")

    assert read_header(path)[GENERATOR_FIELD] == "Google gemini-3.7-flash"
    assert GENERATOR_FIELD in path.read_text(encoding="utf-8")
    assert GENERATOR_FIELD not in joined(tmp_path, "iwac-audio-0005")


# --- What gets annotated, and what stops the run ----------------------------

def test_a_pinned_header_annotates_itself_without_being_asked():
    assert resolve({"Google gemini-3.7-flash": 5}) == ("gemini-3.7-flash", True)


def test_voxtral_output_is_refused_rather_than_attributed_to_someone_else(transcript):
    assert resolve({"Mistral voxtral-mini-2602": 3}) == (None, False)
    assert "voxtral-mini-2602" in transcript.getvalue()
    assert "--no-model-annotation" in transcript.getvalue()


def test_a_rolling_alias_is_refused_for_the_same_reason():
    assert resolve({"Google gemini-pro-latest": 2}) == (None, False)


def test_an_unrecorded_generator_is_refused():
    assert resolve({updater.UNRECORDED_GENERATOR: 4}) == (None, False)


def test_the_annotation_can_be_skipped_on_purpose():
    """The only route for a model with no authority item — and it is a flag,
    so no provenance is something the operator states rather than omits."""
    assert resolve({"Mistral voxtral-mini-2602": 3}, skip=True) == (None, True)


def test_an_operator_may_name_the_release_behind_a_rolling_alias(transcript):
    """The reason --model exists: only a human knows what gemini-pro-latest
    resolved to on the day the run happened."""
    assert resolve({"Google gemini-pro-latest": 2}, requested="gemini-3.1-pro") == (
        "gemini-3.1-pro", True
    )
    assert "Gemini 3.1 Pro" in transcript.getvalue()


def test_a_disagreement_with_a_pinned_header_warns_but_obeys(transcript):
    """A warning, not a refusal: the header can be right and the operator can
    still have a reason. What must not happen is that nobody is told."""
    assert resolve({"Google gemini-3.7-flash": 2}, requested="gemini-3.5-flash-lite") == (
        "gemini-3.5-flash-lite", True
    )
    printed = transcript.getvalue()
    assert "gemini-3.7-flash" in printed
    assert "Gemini 3.5 Flash-Lite" in printed


def test_a_mixed_folder_is_refused_rather_than_half_misattributed(transcript):
    """One annotation covers the whole batch, so the folder has to be one model."""
    assert resolve({"Google gemini-3.7-flash": 4, "Mistral voxtral-mini-2602": 1}) == (
        None, False
    )
    printed = transcript.getvalue()
    assert "4 × Google gemini-3.7-flash" in printed
    assert "1 × Mistral voxtral-mini-2602" in printed


def test_a_mixed_folder_is_refused_even_when_a_model_was_asserted():
    """--model says which model, not that they were all the same one."""
    counts = {"Google gemini-3.7-flash": 4, "Google gemini-pro-latest": 1}
    assert resolve(counts, requested="gemini-3.7-flash") == (None, False)


def test_a_mixed_folder_can_still_be_uploaded_without_provenance():
    counts = {"Google gemini-3.7-flash": 4, "Mistral voxtral-mini-2602": 1}
    assert resolve(counts, skip=True) == (None, True)


# --- The interactive path ---------------------------------------------------

def test_an_interactive_run_offers_the_header_as_the_default(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "")
    assert resolve({"Google gemini-3.7-flash": 1}, assume_yes=False) == (
        "gemini-3.7-flash", True
    )


def test_an_interactive_run_can_override_the_header(monkeypatch):
    from common.iwac_config import AI_MODEL_ITEMS

    wanted = list(AI_MODEL_ITEMS).index("gemini-3.1-pro") + 1
    monkeypatch.setattr("builtins.input", lambda *_: str(wanted))
    assert resolve({"Google gemini-3.7-flash": 1}, assume_yes=False) == (
        "gemini-3.1-pro", True
    )


def test_an_invalid_choice_writes_nothing(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "999")
    assert resolve({"Google gemini-3.7-flash": 1}, assume_yes=False) == (None, False)


def test_a_closed_stdin_is_not_consent(monkeypatch):
    """No prompt was answered, so nothing may be written — matching the
    confirmation gate, which also refuses an EOF rather than proceeding."""
    def no_stdin(*_):
        raise EOFError

    monkeypatch.setattr("builtins.input", no_stdin)
    assert resolve({"Google gemini-3.7-flash": 1}, assume_yes=False) == (None, False)


# --- What reaches the item --------------------------------------------------

def test_no_model_means_no_annotation_term_on_the_target():
    target = updater.content_target(None)
    assert target.term == "bibo:content"
    assert target.annotation_term is None
    assert target.annotation_value is None


def test_the_annotation_lands_on_the_value_that_was_written():
    from common.iwac_config import (
        AI_MODEL_ITEMS,
        IWAC_TRANSCRIPTION_MODEL_PROPERTY_ID,
        model_annotation_value,
    )
    from common.omeka_text_updater import apply_text_value

    target = updater.content_target(model_annotation_value(
        "https://islam.zmo.de", "gemini-3.7-flash",
        IWAC_TRANSCRIPTION_MODEL_PROPERTY_ID, "AI Model - Transcription",
    ))
    item = {}

    assert apply_text_value(item, target, "[00:00:01] Speaker 1: Bismillah.")

    (written,) = item["bibo:content"]
    assert written["@value"] == "[00:00:01] Speaker 1: Bismillah."
    (annotation,) = written["@annotation"]["iwac:transcriptionModel"]
    assert annotation["property_id"] == IWAC_TRANSCRIPTION_MODEL_PROPERTY_ID
    assert annotation["value_resource_id"] == AI_MODEL_ITEMS["gemini-3.7-flash"]["item_id"]


def test_an_unannotated_write_still_writes_the_transcript():
    from common.omeka_text_updater import apply_text_value

    item = {}
    assert apply_text_value(item, updater.content_target(None), "transcript")

    (written,) = item["bibo:content"]
    assert written["@value"] == "transcript"
    assert "@annotation" not in written
