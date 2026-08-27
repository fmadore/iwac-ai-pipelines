"""Tests for the Gemini 3.5 Transcribe audio transcriber (``AI_audio_summary/02c``).

Every guard here corresponds to something the live API does *not* guard. The
model accepts unsupported locales without complaint, returns word offsets as
strings, and restarts both its clock and its speaker numbering in every request
— so a stitched multi-segment transcript is wrong in three ways unless the
client fixes them.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PIPELINE = Path(__file__).resolve().parent.parent / "AI_audio_summary"


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, PIPELINE / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


transcribe = load_script(
    "gemini_transcribe_audio", "02c_AI_transcribe_audio_gemini_transcribe.py"
)


def build(**kwargs):
    """Construct a transcriber with the network client stubbed out."""
    with patch.object(transcribe, "build_gemini_client", return_value=MagicMock()):
        return transcribe.GeminiTranscribeTranscriber(api_key="test-key", **kwargs)


def bare(**attrs):
    """A transcriber with only the attributes a parsing helper touches."""
    worker = transcribe.GeminiTranscribeTranscriber.__new__(
        transcribe.GeminiTranscribeTranscriber
    )
    worker.diarize = attrs.get("diarize", True)
    worker.timestamps = attrs.get("timestamps", True)
    worker.smart = attrs.get("smart", False)
    worker.timestamps_in_text = attrs.get("timestamps_in_text", False)
    worker.language_codes = attrs.get("language_codes", [])
    worker.segment_minutes = attrs.get("segment_minutes", 20)
    return worker


def word_annotation(text, speaker=None, start=None, end=None):
    annotation = MagicMock()
    annotation.type = "word_info"
    annotation.text = text
    annotation.speaker = speaker
    annotation.start_offset = start
    annotation.end_offset = end
    return annotation


def interaction_with(annotations, output_text=""):
    content = MagicMock()
    content.annotations = annotations
    step = MagicMock()
    step.content = [content]
    result = MagicMock()
    result.steps = [step]
    result.output_text = output_text
    return result


# ---------------------------------------------------------------------------
# Locale validation — the guard the server does not provide
# ---------------------------------------------------------------------------

def test_unsupported_locale_is_refused_because_the_api_accepts_it():
    """``mos-BF`` and ``dyu-CI`` were both accepted by the live API and answered
    normally. Nothing downstream can tell that transcript from a real one, so
    the refusal has to happen here or not at all.
    """
    for code in ("mos-BF", "dyu-CI", "ee-GH", "kbp-TG"):
        with pytest.raises(ValueError) as excinfo:
            transcribe.validate_locales([code])
        assert code in str(excinfo.value)


def test_supported_locales_pass_through_unchanged():
    assert transcribe.validate_locales(["fr-FR", "ha-NG"]) == ["fr-FR", "ha-NG"]


def test_hausa_and_french_are_supported():
    """The two locales that make this model worth using on IWAC material."""
    assert "ha-NG" in transcribe.SUPPORTED_LOCALES
    assert "fr-FR" in transcribe.SUPPORTED_LOCALES


def test_iwac_language_map_never_claims_an_unsupported_locale():
    """Every non-``None`` entry must name a locale the model really has.

    A wrong entry here is the same failure as an unvalidated ``--language``:
    it asserts a language the model cannot hear, and the output still looks
    like a transcript.
    """
    for code, locale in transcribe.IWAC_LANGUAGE_LOCALES.items():
        if locale is not None:
            assert locale in transcribe.SUPPORTED_LOCALES, (
                f"IWAC_LANGUAGE_LOCALES[{code!r}] claims {locale!r}, "
                f"which is not in SUPPORTED_LOCALES"
            )


def test_the_west_african_languages_are_recorded_as_uncovered():
    """Mooré, Dioula, Ewé, Kabyè and Dendi are catalogued in this collection and
    absent from the model's locale table. Recording that is what stops a curator
    reaching for ``--language`` and getting fluent nonsense back.
    """
    assert set(transcribe.unsupported_iwac_languages()) == {"mos", "dyu", "ee", "kbp", "ddn"}


def test_language_flag_auto_and_absent_both_mean_detect():
    assert transcribe.resolve_language_codes(None) == []
    assert transcribe.resolve_language_codes("auto") == []
    assert transcribe.resolve_language_codes("fr-FR,ha-NG") == ["fr-FR", "ha-NG"]


# ---------------------------------------------------------------------------
# Request configuration — the shape the API actually rejects
# ---------------------------------------------------------------------------

def test_smart_mode_with_timestamps_is_refused_before_the_request():
    """The live API answers ``400 Unknown parameter 'timestamp_granularities'
    at 'generation_config.transcription_config.mode'``. Failing at construction
    reports it against the flags the operator typed instead.
    """
    with pytest.raises(ValueError, match="smart mode"):
        build(smart=True, timestamps=True)
    with pytest.raises(ValueError, match="smart mode"):
        build(smart=True, diarize=True)


def test_verbatim_config_nests_timestamps_and_diarization_inside_mode():
    config = build(diarize=True, timestamps=True)._transcription_config()
    assert config["mode"]["type"] == "verbatim"
    assert config["mode"]["timestamp_granularities"] == ["word"]
    assert config["mode"]["diarization_mode"] == "speaker"


def test_smart_config_carries_no_annotation_options():
    config = build(smart=True, timestamps=False, diarize=False)._transcription_config()
    assert config["mode"] == {"type": "smart"}


def test_language_codes_are_omitted_when_auto_detecting():
    assert "language_codes" not in build(language_codes=[])._transcription_config()
    assert build(language_codes=["fr-FR"])._transcription_config()["language_codes"] == ["fr-FR"]


def test_no_temperature_or_thinking_level_is_sent():
    """Both are vendor territory, and this model rejects the ``minimal`` thinking
    level every other pipeline here asks for (``Allowed values are: low, high``).
    """
    config = build()._transcription_config()
    assert "temperature" not in config
    assert "thinking_level" not in config


# ---------------------------------------------------------------------------
# The cap, and the splitting it forces
# ---------------------------------------------------------------------------

def test_annotations_halve_the_request_cap():
    assert build(diarize=True, timestamps=True).cap_seconds == 30 * 60
    assert build(smart=True, timestamps=False, diarize=False).cap_seconds == 60 * 60


def test_segment_length_is_clamped_to_the_cap():
    """A 45-minute segment cannot be sent when the cap is 30 minutes; silently
    accepting it would fail after the upload rather than before it.
    """
    assert build(segment_minutes=45).segment_minutes == 30
    assert build(smart=True, timestamps=False, diarize=False, segment_minutes=45).segment_minutes == 45


def test_short_file_is_not_split():
    worker = build()
    with patch.object(transcribe, "probe_duration_seconds", return_value=600.0):
        paths, duration = worker._segment_paths(Path("short.mp3"))
    assert paths == [Path("short.mp3")]
    assert duration == 600.0


def test_file_over_the_cap_is_split():
    worker = build()
    with patch.object(transcribe, "probe_duration_seconds", return_value=3600.0), \
            patch.object(transcribe, "split_audio_file", return_value=[Path("a"), Path("b")]) as split:
        paths, duration = worker._segment_paths(Path("long.mp3"))
    assert paths == [Path("a"), Path("b")]
    assert duration == 3600.0
    assert split.call_args[0][2] == worker.segment_minutes


def test_unknown_duration_is_split_rather_than_gambled_with():
    """Without ffprobe the length is unknown; ``split_audio_file`` reads it and
    returns the original path when it fits, so splitting is the safe default.
    """
    worker = build()
    with patch.object(transcribe, "probe_duration_seconds", return_value=None), \
            patch.object(transcribe, "split_audio_file", return_value=[Path("x.mp3")]) as split:
        paths, duration = worker._segment_paths(Path("x.mp3"))
    assert paths == [Path("x.mp3")]
    assert duration is None
    assert split.called


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("0.100s", 0.1),
    ("2s", 2.0),
    ("12.750s", 12.75),
    (3.5, 3.5),
    (None, None),
    ("", None),
    ("2 seconds", None),
    ("abc", None),
])
def test_offsets_parse_from_the_string_form_the_api_returns(raw, expected):
    assert transcribe.parse_offset_seconds(raw) == expected


def test_word_offsets_are_shifted_to_absolute_positions():
    """Each request's clock restarts at zero. Segment 2 of a 20-minute split
    begins 1200 s into the recording, and its timings say so only if shifted.
    """
    result = interaction_with([
        word_annotation("bonjour", speaker="spk:0", start="1.500s", end="2s"),
    ])
    words = transcribe.GeminiTranscribeTranscriber._collect_words(result, 1200, 2)
    assert words[0]["start"] == 1201.5
    assert words[0]["end"] == 1202.0


def test_speakers_are_namespaced_per_segment():
    """``spk:0`` in segment 1 and ``spk:0`` in segment 2 are not evidence of the
    same person; merging them would invent a continuity the diarizer never claimed.
    """
    result = interaction_with([word_annotation("un", speaker="spk:0", start="0s", end="1s")])
    assert transcribe.GeminiTranscribeTranscriber._collect_words(result, 0, 2)[0]["speaker"] == "seg2-spk:0"
    # Unsplit files keep the model's own label.
    assert transcribe.GeminiTranscribeTranscriber._collect_words(result, 0, None)[0]["speaker"] == "spk:0"


def test_non_word_annotations_are_ignored():
    citation = MagicMock()
    citation.type = "url_citation"
    result = interaction_with([citation, word_annotation("mot", start="0s", end="1s")])
    words = transcribe.GeminiTranscribeTranscriber._collect_words(result, 0, None)
    assert [w["text"] for w in words] == ["mot"]


def test_turns_group_consecutive_words_by_speaker():
    words = [
        {"text": "bonjour", "speaker": "spk:0", "start": 0.1, "end": 0.8},
        {"text": "madame", "speaker": "spk:0", "start": 0.9, "end": 1.4},
        {"text": "oui", "speaker": "spk:1", "start": 1.6, "end": 2.0},
    ]
    text = bare()._format_text(interaction_with([], "bonjour madame oui"), words)
    assert text == "[Speaker spk:0]\nbonjour madame\n\n[Speaker spk:1]\noui"


def test_turn_clock_positions_are_opt_in():
    """The ``.txt`` becomes ``bibo:content``, the archive's indexed full text —
    timestamps in the body would be indexed as though someone had said them.
    """
    words = [{"text": "bonjour", "speaker": "spk:0", "start": 61.0, "end": 61.8}]
    assert "[00:01:01]" not in bare()._format_text(interaction_with([]), words)
    assert "[00:01:01]" in bare(timestamps_in_text=True)._format_text(interaction_with([]), words)


def test_text_falls_back_to_output_text_without_diarization():
    """``output_text`` is authoritative for wording and punctuation; it is only
    regrouped when speaker labels require it.
    """
    result = interaction_with([], "Bonjour, je m'appelle Fatoumata.")
    assert bare(diarize=False)._format_text(result, []) == "Bonjour, je m'appelle Fatoumata."
    # Smart mode returns no annotations at all.
    assert bare(diarize=True)._format_text(result, []) == "Bonjour, je m'appelle Fatoumata."


@pytest.mark.parametrize("seconds,expected", [
    (None, "--:--:--"), (0, "00:00:00"), (61.4, "00:01:01"), (3725, "01:02:05"),
])
def test_clock_formatting(seconds, expected):
    assert transcribe.format_clock(seconds) == expected


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def test_a_failed_segment_yields_nothing_rather_than_a_partial_transcript():
    """A transcript missing its middle third is invisible once it is a single
    Omeka value, so the whole file is withheld.
    """
    worker = build()
    with patch.object(worker, "_segment_paths", return_value=([Path("a"), Path("b")], 3600.0)), \
            patch.object(worker, "_create_interaction", side_effect=[interaction_with([], "first half"), None]), \
            patch.object(transcribe, "cleanup_temp_segments"):
        assert worker.transcribe_audio(Path("long.mp3")) is None


def test_multi_segment_output_carries_positional_headers():
    worker = build(diarize=False)
    results = [interaction_with([], "part one"), interaction_with([], "part two")]
    with patch.object(worker, "_segment_paths", return_value=([Path("a"), Path("b")], 2400.0)), \
            patch.object(worker, "_create_interaction", side_effect=results), \
            patch.object(transcribe, "cleanup_temp_segments"):
        text, payload = worker.transcribe_audio(Path("long.mp3"))

    assert "[Segment 1/2" in text and "[Segment 2/2" in text
    assert "part one" in text and "part two" in text
    assert payload["segments"] == 2
    assert payload["segment_minutes"] == worker.segment_minutes


def test_empty_transcription_is_not_saved():
    worker = build(diarize=False)
    with patch.object(worker, "_segment_paths", return_value=([Path("a")], 60.0)), \
            patch.object(worker, "_create_interaction", return_value=interaction_with([], "   ")), \
            patch.object(transcribe, "cleanup_temp_segments"):
        assert worker.transcribe_audio(Path("quiet.mp3")) is None


def test_segments_are_cleaned_up_even_when_a_segment_fails():
    worker = build()
    with patch.object(worker, "_segment_paths", return_value=([Path("a"), Path("b")], 3600.0)), \
            patch.object(worker, "_create_interaction", return_value=None), \
            patch.object(transcribe, "cleanup_temp_segments") as cleanup:
        worker.transcribe_audio(Path("long.mp3"))
    assert cleanup.called


def test_quota_exhaustion_stops_the_run_instead_of_retrying():
    """A daily quota is not a transient 429: retrying it burns the run instead
    of saving what completed.
    """
    from common.rate_limiter import QuotaExhaustedError

    worker = build()
    worker.rate_limiter = MagicMock()
    with patch.object(transcribe, "upload_and_wait_active", side_effect=RuntimeError("boom")), \
            patch.object(transcribe, "is_quota_exhausted", return_value=True), \
            patch.object(transcribe, "delete_uploaded_file"), \
            pytest.raises(QuotaExhaustedError):
        worker._create_interaction(Path("a.mp3"))


def test_uploads_are_deleted_even_when_the_request_fails():
    """Leaked multi-hour uploads waste Files API quota until they expire."""
    worker = build()
    worker.rate_limiter = MagicMock()
    uploaded = MagicMock()
    worker.client.interactions.create.side_effect = RuntimeError("nope")
    with patch.object(transcribe, "upload_and_wait_active", return_value=uploaded), \
            patch.object(transcribe, "is_quota_exhausted", return_value=False), \
            patch.object(transcribe, "delete_uploaded_file") as delete, \
            patch.object(transcribe.time, "sleep"):
        assert worker._create_interaction(Path("a.mp3"), max_retries=2) is None
    assert delete.call_count == 2


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def test_the_model_has_an_authority_item_to_be_annotated_with():
    """Step 03 stamps ``iwac:transcriptionModel`` by registry key; a key missing
    from ``AI_MODEL_ITEMS`` strands the operator at the write step.
    """
    from common.iwac_config import AI_MODEL_ITEMS

    assert transcribe.MODEL in AI_MODEL_ITEMS
    assert AI_MODEL_ITEMS[transcribe.MODEL]["item_id"] == 113077


def test_the_model_id_is_pinned_not_rolling():
    """A rolling alias reports its own version as a label, so a run through one
    cannot confirm which model produced the text.
    """
    assert "latest" not in transcribe.MODEL


def test_no_prompt_is_accepted_anywhere_in_the_cli():
    """The model answers ``400 Developer instruction is not enabled for this
    model``. Offering a ``--prompt`` would promise something it cannot do.
    """
    parser_args = vars(transcribe.parse_args([]))
    assert "prompt" not in parser_args
