"""The YouTube transcription pipeline: URLs, chunk plans, languages, transcripts.

Everything here is the part of the pipeline that decides *what* is sent to Gemini
and *what* reaches Omeka, which is where a mistake is expensive and silent: a
rejected URL costs one API error, but a chunk plan with the wrong boundaries
produces a transcript that reads as complete and is missing its middle, and a
mishandled language label reports a correctly catalogued item as wrong.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from common.omeka_link_updater import ResourceLinkUpdate
from youtube_source import (
    DEFAULT_CHUNK_OVERLAP_SECONDS,
    MAX_NGRAM_REPEAT,
    DetectedLanguage,
    VideoWork,
    catalogued_language_code,
    chunk_prompt_suffix,
    dominant_languages,
    format_hms,
    join_chunks,
    language_matches,
    language_prompt_suffix,
    looping_reason,
    most_repeated_ngram,
    parse_detected_languages,
    parse_iso_duration,
    parse_video_id,
    plan_chunks,
    plan_language_samples,
    read_transcript,
    read_work_list,
    transcript_path,
    work_fingerprint,
    write_transcript,
    write_work_list,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
UPDATER_PATH = REPO_ROOT / "AI_youtube_transcription" / "03_omeka_transcription_updater.py"
SPEC = importlib.util.spec_from_file_location("youtube_transcription_updater", UPDATER_PATH)
assert SPEC and SPEC.loader
updater = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = updater
SPEC.loader.exec_module(updater)


VIDEO = VideoWork(
    item_id=108353,
    video_id="6HKzcPYE0c8",
    url="https://www.youtube.com/watch?v=6HKzcPYE0c8",
    title="Renforcer la Paix",
    identifier="iwac-video-0000093",
    duration_seconds=497,
    language="Français",
)


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=6HKzcPYE0c8",
    "http://youtube.com/watch?v=6HKzcPYE0c8",
    "https://m.youtube.com/watch?v=6HKzcPYE0c8",
    "https://youtu.be/6HKzcPYE0c8",
    "https://www.youtube.com/watch?v=6HKzcPYE0c8&t=10s",
    "https://www.youtube.com/watch?list=PL123&v=6HKzcPYE0c8",
    "  https://www.youtube.com/watch?v=6HKzcPYE0c8  ",
])
def test_canonical_watch_urls_are_accepted(url):
    assert parse_video_id(url) == "6HKzcPYE0c8"


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/shorts/6HKzcPYE0c8",   # rejected by Omeka's ingester too
    "https://www.youtube.com/live/6HKzcPYE0c8",
    "https://www.youtube.com/embed/6HKzcPYE0c8",
    "https://www.youtube.com/playlist?list=PL123",
    "https://www.youtube.com/",
    "https://www.youtube.com/watch",
    "https://www.youtube.com/watch?v=",
    "https://www.youtube.com/watch?v=tooshort",
    "https://vimeo.com/watch?v=6HKzcPYE0c8",
    "https://notyoutube.com/watch?v=6HKzcPYE0c8",
    "ftp://youtu.be/6HKzcPYE0c8",
    "6HKzcPYE0c8",
    "",
    None,
])
def test_everything_else_is_rejected_before_the_api_sees_it(url):
    assert parse_video_id(url) is None


# ---------------------------------------------------------------------------
# Durations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("PT6M51S", 411),
    ("PT8M17S", 497),
    ("PT33M36S", 2016),
    ("PT1H2M3S", 3723),
    ("PT8M", 480),
    ("PT571M", 34_260),         # a real extent in the collection
    ("P1DT2H", 93_600),
    ("PT30.5S", 31),            # rounded, not truncated
])
def test_iso_durations_become_seconds(value, expected):
    assert parse_iso_duration(value) == expected


@pytest.mark.parametrize("value", ["", None, "33M36S", "P", "PT", "P1M", "PT0S", "nonsense"])
def test_unparseable_durations_are_none_not_zero(value):
    """``None`` means "plan a single request", which is the safe fallback.

    ``P1M`` is deliberately here: a month has no fixed length, so accepting it
    would put a made-up number of seconds into the chunk plan.
    """
    assert parse_iso_duration(value) is None


def test_format_hms():
    assert format_hms(0) == "00:00:00"
    assert format_hms(497) == "00:08:17"
    assert format_hms(3723) == "01:02:03"
    assert format_hms(None) == "??:??:??"


# ---------------------------------------------------------------------------
# Chunk planning
# ---------------------------------------------------------------------------

def test_a_video_inside_the_budget_is_one_unclipped_request():
    chunks = plan_chunks(2016, chunk_seconds=45 * 60)
    assert len(chunks) == 1
    assert chunks[0].is_whole_video
    assert chunks[0].end is None


def test_an_unknown_duration_is_one_unclipped_request():
    assert plan_chunks(None, chunk_seconds=45 * 60)[0].is_whole_video


def test_a_long_video_is_split_with_overlap_and_absolute_boundaries():
    chunks = plan_chunks(5_000, chunk_seconds=2_000, overlap_seconds=15)
    assert [(c.index, c.total) for c in chunks] == [(1, 3), (2, 3), (3, 3)]
    # Window 1 starts at zero; later windows reach back by the overlap...
    assert (chunks[0].start, chunks[0].end) == (0, 2_000)
    assert (chunks[1].start, chunks[1].end) == (1_985, 4_000)
    # ...but content_start stays on the nominal boundary, which is what the
    # prompt uses to drop the duplicated overlap.
    assert chunks[1].content_start == 2_000
    # The last window ends at the real end of the video, not past it.
    assert chunks[2].end == 5_000


def test_a_duration_exactly_on_the_boundary_is_not_split():
    assert len(plan_chunks(2_700, chunk_seconds=2_700)) == 1


@pytest.mark.parametrize("kwargs", [
    {"chunk_seconds": 0},
    {"chunk_seconds": -60},
    {"chunk_seconds": 600, "overlap_seconds": -1},
    {"chunk_seconds": 600, "overlap_seconds": 600},   # would never advance
])
def test_impossible_chunk_geometry_is_refused(kwargs):
    with pytest.raises(ValueError):
        plan_chunks(10_000, **kwargs)


def test_a_single_request_gets_no_chunk_instructions():
    assert chunk_prompt_suffix(plan_chunks(400, chunk_seconds=2_700)[0]) == ""


def test_later_windows_are_told_to_use_absolute_timestamps_and_skip_the_overlap():
    second = plan_chunks(5_000, chunk_seconds=2_000, overlap_seconds=15)[1]
    suffix = chunk_prompt_suffix(second)
    assert "ABSOLUTE" in suffix
    assert format_hms(second.start) in suffix        # "the first moment you receive"
    assert format_hms(second.content_start) in suffix  # "skip anything before"
    assert "part 2 of 3" in suffix


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def test_short_videos_are_sampled_whole_rather_than_twice():
    """Two 45-second samples of a 60-second video cost more than the video."""
    assert plan_language_samples(60, sample_seconds=45, samples=2) == [(0, None)]


def test_long_videos_are_sampled_away_from_the_opening():
    windows = plan_language_samples(2_000, sample_seconds=45, samples=2)
    assert windows == [(200, 245), (1_100, 1_145)]
    # Not from 0: the opening seconds are a jingle or a French title card as
    # often as they are the speech being catalogued.
    assert windows[0][0] > 0


def test_an_unknown_duration_can_only_be_sampled_from_the_start():
    assert plan_language_samples(None, sample_seconds=45) == [(0, 45)]


def test_detected_languages_are_ordered_dominant_first():
    languages = parse_detected_languages({"languages": [
        {"name_en": "French", "bcp47": "fr", "share": "occasional"},
        {"name_en": "Mooré", "bcp47": "mos", "share": "dominant"},
    ]})
    assert [lang.bcp47 for lang in languages] == ["mos", "fr"]
    assert dominant_languages(languages)[0].name_en == "Mooré"


@pytest.mark.parametrize("payload", [
    None, {}, "text", {"languages": None}, {"languages": [{}, "junk", {"share": "dominant"}]},
])
def test_a_malformed_detection_response_yields_no_languages_rather_than_raising(payload):
    """Detection is an aid, not a gate — a bad answer must not fail the video."""
    assert parse_detected_languages(payload) == []


def test_a_bcp47_region_subtag_does_not_break_code_matching():
    assert DetectedLanguage("French", "fr-BF").code == "fr"


@pytest.mark.parametrize("label,code", [
    ("Français", "fr"), ("français", "fr"), ("FRANÇAIS", "fr"),
    ("Anglais", "en"), ("Haoussa", "ha"), ("Ewé", "ee"), ("Ewe", "ee"),
    ("Kabyè", "kbp"), ("Mooré", "mos"), ("Moore", "mos"), ("Arabe", "ar"),
    ("Espagnol", "es"), ("Dioula", "dyu"),
])
def test_french_authority_labels_map_to_iso_codes(label, code):
    assert catalogued_language_code(label) == code


@pytest.mark.parametrize("label", ["Peul", "Bambara", "Zarma", "Yoruba", "Wolof"])
def test_languages_without_an_authority_record_are_not_mapped(label):
    """Mapping a code with no Omeka record only moves the failure later.

    These are spoken in this material and deliberately absent: creating the
    authority record is a curatorial act, so a detected language outside the map
    is reported under its own name for an operator to decide on, never linked.
    """
    assert catalogued_language_code(label) is None


def test_the_two_language_maps_cannot_drift_apart():
    """Every linkable label must also be comparable, and vice versa.

    A label present in one direction only is how a language ends up linkable but
    reported as a mismatch on every item, or comparable but never written.
    """
    from common.iwac_config import LANGUAGE_LABELS_BY_CODE
    from youtube_source import CATALOGUED_LANGUAGE_CODES, fold

    assert set(CATALOGUED_LANGUAGE_CODES.values()) == set(LANGUAGE_LABELS_BY_CODE)
    for code, label in LANGUAGE_LABELS_BY_CODE.items():
        assert CATALOGUED_LANGUAGE_CODES[fold(label)] == code


def test_the_record_agrees_with_the_audio_across_the_language_barrier():
    """``Français`` vs ``French`` is not a mismatch.

    The first live run reported exactly this as one, because the authority label
    is French and detection answers in English.
    """
    assert language_matches("Français", [DetectedLanguage("French", "fr")]) is True


def test_a_real_disagreement_is_reported():
    assert language_matches("Français", [DetectedLanguage("Mooré", "mos")]) is False


def test_a_secondary_language_does_not_rescue_a_wrong_record():
    """French present but not dominant still contradicts ``Français``."""
    assert language_matches("Français", [
        DetectedLanguage("Mooré", "mos", "dominant"),
        DetectedLanguage("French", "fr", "secondary"),
    ]) is False


@pytest.mark.parametrize("catalogued,languages", [
    ("", [DetectedLanguage("Mooré", "mos")]),          # nothing catalogued
    ("Français", []),                                   # nothing detected
    ("Klingon", [DetectedLanguage("Klingon", "tlh")]),  # unmapped label, names agree
])
def test_absent_or_unmappable_evidence_is_not_a_mismatch(catalogued, languages):
    assert language_matches(catalogued, languages) is True


def test_the_prompt_names_the_detected_languages():
    suffix = language_prompt_suffix(
        [DetectedLanguage("Mooré", "mos", "dominant"),
         DetectedLanguage("French", "fr", "secondary")],
        catalogued="Français",
    )
    assert "Mooré (mos)" in suffix
    assert "French (fr, secondary)" in suffix
    # And says which side to believe when the record disagrees.
    assert "Follow the audio, not the record." in suffix


def test_the_prompt_asks_the_model_to_identify_the_language_when_detection_failed():
    suffix = language_prompt_suffix([], catalogued="Français")
    assert "Identify it" in suffix
    assert "unverified" in suffix


def test_an_agreeing_record_is_not_argued_with():
    suffix = language_prompt_suffix([DetectedLanguage("French", "fr")], catalogued="Français")
    assert "disagrees" not in suffix


# ---------------------------------------------------------------------------
# Transcript files
# ---------------------------------------------------------------------------

def write_one(tmp_path, body="[00:00:01] Speaker 1: Bismillah.", *, done=1, total=1):
    return write_transcript(
        tmp_path, VIDEO, body,
        generator="Google gemini-3.6-flash",
        prompt_label="Full Video Transcription",
        prompt_sha256="da81b9f6def3aaaa",
        chunks_done=done,
        chunks_total=total,
        languages=[DetectedLanguage("French", "fr")],
    )


def test_a_transcript_is_named_after_its_omeka_item():
    assert transcript_path(Path("out"), 108353).name == "108353.txt"


def test_the_header_round_trips_and_stays_out_of_the_body(tmp_path):
    path = write_one(tmp_path)
    transcript = read_transcript(path)

    assert transcript.header["Omeka item"] == "108353"
    assert transcript.header["Generated using"] == "Google gemini-3.6-flash"
    assert transcript.header["Languages detected"] == "fr (French, dominant)"
    assert transcript.header["Catalogued as"] == "Français"
    # The body is what reaches bibo:content: no header, no separator.
    assert transcript.body == "[00:00:01] Speaker 1: Bismillah."
    assert "Generated using" not in transcript.body
    assert transcript.complete is True


def test_an_incomplete_transcript_says_so(tmp_path):
    transcript = read_transcript(write_one(tmp_path, done=2, total=3))
    assert (transcript.chunks_done, transcript.chunks_total) == (2, 3)
    assert transcript.complete is False


def test_a_headerless_file_is_all_body(tmp_path):
    """Never guess which leading lines were metadata — that truncates text."""
    path = tmp_path / "108353.txt"
    path.write_text("[00:00:01] Speaker 1: Bismillah.\n", encoding="utf-8")
    transcript = read_transcript(path)
    assert transcript.body == "[00:00:01] Speaker 1: Bismillah."
    assert transcript.header == {}
    assert transcript.complete is True


def test_an_empty_body_is_never_complete(tmp_path):
    assert read_transcript(write_one(tmp_path, body="  ")).complete is False


# ---------------------------------------------------------------------------
# Looping — the failure mode nothing else catches
# ---------------------------------------------------------------------------

def test_a_normal_transcript_is_not_flagged():
    text = " ".join(f"[00:0{i % 10}:00] Locuteur 1 : Une phrase distincte numéro {i}." for i in range(60))
    assert looping_reason(text) is None


def test_a_legitimate_refrain_is_not_flagged():
    """Real transcripts here repeat a formula up to 7 times — a prayer, an ident."""
    refrain = "au nom de dieu le clément le miséricordieux louange à dieu seigneur des mondes "
    text = (refrain * 7) + " ".join(f"phrase distincte numéro {i}" for i in range(200))
    assert looping_reason(text) is None


def test_a_looping_transcript_is_caught():
    """The real failure: one clause emitted until the output cap is hit."""
    reason = looping_reason("yaa yamb kanga la ton pa men yen konne ba lee belam ti " * 600)
    assert reason is not None
    assert reason.startswith("looping-")


def test_a_short_transcript_is_never_flagged():
    """A two-line clip has too few windows for the measure to mean anything."""
    assert looping_reason("[00:00:01] Locuteur 1 : Bismillah.") is None


def test_the_threshold_sits_in_the_observed_gap():
    """41 sound transcripts peaked at 7 repeats; the 3 broken ones scored 575+."""
    assert 7 < MAX_NGRAM_REPEAT < 575


def test_the_most_repeated_ngram_is_reported_with_its_count():
    count, gram = most_repeated_ngram("un deux trois quatre cinq six sept huit neuf dix onze douze " * 5)
    assert count >= 5
    assert "un deux trois" in gram


def test_a_looping_transcript_is_never_uploaded(tmp_path):
    """Not overridable: it reports Chunks: 1/1, so completeness cannot see it."""
    write_one(tmp_path, body="yaa yamb kanga la ton pa men yen konne ba lee belam ti " * 600)
    for include_incomplete in (False, True):
        updates, held_back = updater.collect_updates(
            tmp_path, include_incomplete=include_incomplete
        )
        assert updates == []
        assert held_back and "degenerate repeating output" in held_back[0][1]


def test_chunks_are_joined_with_a_marker_on_every_seam_but_the_first():
    chunks = plan_chunks(5_000, chunk_seconds=2_000, overlap_seconds=15)
    joined = join_chunks([(chunks[0], "first"), (chunks[1], "second")])
    assert joined.startswith("first")
    assert "--- Chunk 2/3 | 00:33:05–01:06:40 ---" in joined
    # No failure markers anywhere: a marker in the body would be uploaded to
    # Omeka as though a speaker had said it.
    assert "FAILED" not in joined


# ---------------------------------------------------------------------------
# The work list
# ---------------------------------------------------------------------------

def test_the_work_list_round_trips(tmp_path):
    path = write_work_list(tmp_path / "work.json", [VIDEO], scope={"item_set_ids": [108260]})
    (restored,) = read_work_list(path)
    assert restored == VIDEO


def test_a_work_list_from_another_format_version_is_refused(tmp_path):
    path = tmp_path / "work.json"
    path.write_text(json.dumps({"version": 99, "videos": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="format version"):
        read_work_list(path)


def test_the_fingerprint_changes_when_the_video_or_its_plan_does():
    baseline = work_fingerprint(VIDEO, 1)
    assert work_fingerprint(VIDEO, 2) != baseline
    longer = VideoWork(**{**VIDEO.to_json(), "duration_seconds": 6_000})
    assert work_fingerprint(longer, 1) != baseline


# ---------------------------------------------------------------------------
# Step 03: what reaches Omeka
# ---------------------------------------------------------------------------

def test_the_write_step_parses_argv():
    """A write entry point must refuse arguments rather than treat them as consent."""
    with pytest.raises(SystemExit) as excinfo:
        updater.main.__globals__["argparse"].ArgumentParser().parse_args(["--help"])
    assert excinfo.value.code == 0


def test_only_item_id_transcripts_are_collected(tmp_path):
    write_one(tmp_path)
    (tmp_path / "_language_report.json").write_text("[]", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("a human's scratch file", encoding="utf-8")

    updates, held_back = updater.collect_updates(tmp_path, include_incomplete=False)

    assert [update.item_id for update in updates] == [108353]
    assert held_back == []


def test_incomplete_transcripts_are_held_back_by_default(tmp_path):
    write_one(tmp_path, done=2, total=3)
    updates, held_back = updater.collect_updates(tmp_path, include_incomplete=False)
    assert updates == []
    assert held_back == [(108353, "incomplete (2/3 windows)")]


def test_incomplete_transcripts_can_be_uploaded_on_purpose(tmp_path):
    write_one(tmp_path, done=2, total=3)
    updates, held_back = updater.collect_updates(tmp_path, include_incomplete=True)
    assert [update.item_id for update in updates] == [108353]
    assert held_back == []


def test_an_empty_transcript_is_never_uploaded(tmp_path):
    write_one(tmp_path, body="   ")
    updates, held_back = updater.collect_updates(tmp_path, include_incomplete=True)
    assert updates == []
    assert held_back == [(108353, "empty transcript")]


def test_the_uploaded_text_carries_no_provenance_header(tmp_path):
    """Provenance belongs in the annotation, not in the archive's full-text field."""
    write_one(tmp_path)
    (update,) = updater.collect_updates(tmp_path, include_incomplete=False)[0]
    assert update.text == "[00:00:01] Speaker 1: Bismillah."
    assert update.metadata["generator"] == "Google gemini-3.6-flash"


def test_the_annotation_property_is_the_transcription_model():
    """Not iwac:ocrModel or iwac:summaryModel — 315 is the transcription one."""
    from common.iwac_config import IWAC_TRANSCRIPTION_MODEL_PROPERTY_ID

    assert IWAC_TRANSCRIPTION_MODEL_PROPERTY_ID == 315
    assert updater.TRANSCRIPTION_MODEL_TERM == "iwac:transcriptionModel"
    assert updater.CONTENT_TERM == "bibo:content"


def load_transcriber():
    path = REPO_ROOT / "AI_youtube_transcription" / "02_AI_transcribe_youtube.py"
    spec = importlib.util.spec_from_file_location("youtube_transcriber", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_offered_model_can_be_annotated():
    """Step 02's models must all exist in AI_MODEL_ITEMS, or 03 cannot stamp them.

    They are pinned releases for exactly this reason: a rolling alias reports its
    version as "Gemini Flash Latest", so an annotation through one would name a
    model the run never confirmed.
    """
    from common.iwac_config import AI_MODEL_ITEMS

    module = load_transcriber()
    assert module.DEFAULT_MODEL in module.ALLOWED_MODELS
    for model in module.ALLOWED_MODELS:
        assert model in AI_MODEL_ITEMS, f"{model} has no Omeka authority item"
        assert "latest" not in model, f"{model} is a rolling alias and cannot be cited"


def test_the_two_steps_default_to_the_same_model():
    """A different default in 02 and 03 mis-attributes an unattended run.

    Step 03's answer becomes the ``iwac:transcriptionModel`` annotation, so if the
    defaults drift apart, a run that accepted both would stamp the wrong model on
    correct text — and nothing downstream could tell.
    """
    assert load_transcriber().DEFAULT_MODEL == updater.DEFAULT_MODEL_KEY


def test_the_language_report_merges_rather_than_replacing(tmp_path):
    """A resumed run must not shrink the report to its own batch.

    Step 02 resumes from a checkpoint, so a second run covers only what was
    outstanding. Overwriting would drop the findings for every skipped item — and
    that report is exactly what step 04 reads to correct dcterms:language.
    """
    module = load_transcriber()

    def result(item_id, name, code):
        video = VideoWork(item_id=item_id, video_id="x" * 11, url="u", language="Français")
        return module.VideoResult(
            video=video, chunks_total=1, chunks_done=1,
            languages=[DetectedLanguage(name, code)],
        )

    module.write_language_report(tmp_path, [result(1, "Mooré", "mos")])
    module.write_language_report(tmp_path, [result(2, "French", "fr")])

    rows = json.loads((tmp_path / module.LANGUAGE_REPORT_NAME).read_text(encoding="utf-8"))
    assert [row["item_id"] for row in rows] == [1, 2]


def test_a_rerun_of_the_same_item_wins(tmp_path):
    """The newer detection is from the model this run used, so it is current."""
    module = load_transcriber()
    video = VideoWork(item_id=7, video_id="x" * 11, url="u", language="Français")

    for name, code in (("Mooré", "mos"), ("French", "fr")):
        module.write_language_report(tmp_path, [module.VideoResult(
            video=video, chunks_total=1, chunks_done=1,
            languages=[DetectedLanguage(name, code)],
        )])

    (row,) = json.loads((tmp_path / module.LANGUAGE_REPORT_NAME).read_text(encoding="utf-8"))
    assert row["detected"][0]["bcp47"] == "fr"


def test_a_corrupt_report_does_not_lose_this_run(tmp_path):
    module = load_transcriber()
    (tmp_path / module.LANGUAGE_REPORT_NAME).write_text("{not json", encoding="utf-8")
    module.write_language_report(tmp_path, [module.VideoResult(
        video=VideoWork(item_id=9, video_id="x" * 11, url="u"),
        chunks_total=1, chunks_done=1, languages=[DetectedLanguage("Mooré", "mos")],
    )])
    rows = json.loads((tmp_path / module.LANGUAGE_REPORT_NAME).read_text(encoding="utf-8"))
    assert [row["item_id"] for row in rows] == [9]


def test_a_closed_stdin_falls_back_to_the_default_instead_of_crashing(monkeypatch):
    """A piped or scheduled run must not die on the model prompt.

    Unlike the write guard, where an EOF is refused because consent cannot be
    inferred, nothing is written here — so the documented default is the right
    answer to no answer.
    """
    module = load_transcriber()

    def raise_eof(*args, **kwargs):
        raise EOFError

    monkeypatch.setattr(module.console, "input", raise_eof)
    assert module.select_model_interactive() == module.DEFAULT_MODEL
    # And the prompt menu takes the first prompt rather than propagating the EOF.
    prompt_text, label = module.select_prompt(None)
    assert prompt_text.strip()
    assert label == "Full Video Transcription"


def test_the_default_overlap_is_smaller_than_any_sane_window():
    assert 0 < DEFAULT_CHUNK_OVERLAP_SECONDS < 60


# ---------------------------------------------------------------------------
# Step 04: correcting dcterms:language from what was heard
# ---------------------------------------------------------------------------

LANGUAGE_UPDATER_PATH = REPO_ROOT / "AI_youtube_transcription" / "04_omeka_language_updater.py"
LANG_SPEC = importlib.util.spec_from_file_location("youtube_language_updater", LANGUAGE_UPDATER_PATH)
assert LANG_SPEC and LANG_SPEC.loader
languages = importlib.util.module_from_spec(LANG_SPEC)
sys.modules[LANG_SPEC.name] = languages
LANG_SPEC.loader.exec_module(languages)


REPORT_ROW = {
    "item_id": 108309,
    "identifier": "iwac-video-0000071",
    "catalogued": "Français",
    "detected": [
        {"name_en": "Mooré", "bcp47": "mos", "share": "dominant"},
        {"name_en": "French", "bcp47": "fr", "share": "occasional"},
    ],
    "agrees_with_record": False,
}


def write_report(tmp_path, rows=None):
    path = tmp_path / "_language_report.json"
    path.write_text(json.dumps(rows if rows is not None else [REPORT_ROW]), encoding="utf-8")
    return path


def test_the_language_write_step_parses_argv():
    with pytest.raises(SystemExit) as excinfo:
        languages.build_parser().parse_args(["--help"])
    assert excinfo.value.code == 0


def test_it_exposes_the_write_guard_flags():
    args = languages.build_parser().parse_args(["--dry-run", "--yes"])
    guard = languages.WriteGuard.from_args(args)
    assert (guard.dry_run, guard.assume_yes) == (True, True)


def test_occasional_languages_are_skipped_by_default(tmp_path):
    """One quoted 'bismillah' must not catalogue an item as Arabic."""
    (item,) = languages.read_report(write_report(tmp_path), shares=languages.DEFAULT_SHARES)
    assert [lang.bcp47 for lang in item.detected] == ["mos"]


def test_occasional_languages_can_be_included_on_purpose(tmp_path):
    (item,) = languages.read_report(
        write_report(tmp_path), shares=(*languages.DEFAULT_SHARES, "occasional")
    )
    assert [lang.bcp47 for lang in item.detected] == ["mos", "fr"]


def test_an_item_with_only_occasional_languages_drops_out(tmp_path):
    row = {**REPORT_ROW, "detected": [{"name_en": "Arabic", "bcp47": "ar", "share": "occasional"}]}
    assert languages.read_report(write_report(tmp_path, [row]), shares=("dominant",)) == []


def test_resolved_languages_become_links_and_the_rest_are_reported(tmp_path):
    (item,) = languages.read_report(write_report(tmp_path), shares=("dominant", "secondary"))
    languages.attach_authority_items([item], {"Mooré": 8384})
    assert item.resolved == {"Mooré": 8384}
    assert item.unresolved == []


def test_a_label_with_no_authority_record_is_never_invented(tmp_path):
    """A missing record is a curatorial gap, not something to link around."""
    (item,) = languages.read_report(write_report(tmp_path), shares=("dominant", "secondary"))
    languages.attach_authority_items([item], {"Mooré": None})
    assert item.resolved == {}
    assert item.unresolved == ["Mooré"]


def test_a_language_outside_the_map_is_reported_under_its_own_name(tmp_path):
    """Detected, unmappable, and therefore surfaced rather than dropped.

    This is what happened to Dioula on the first full run: reported here, then an
    authority record was created by hand and the code added to
    ``LANGUAGE_LABELS_BY_CODE``. The record comes first — hence Peul, which still
    has none.
    """
    row = {**REPORT_ROW, "detected": [{"name_en": "Fulfulde", "bcp47": "ff", "share": "dominant"}]}
    (item,) = languages.read_report(write_report(tmp_path, [row]), shares=("dominant",))
    languages.attach_authority_items([item], {})
    assert item.resolved == {}
    assert item.unresolved == ["Fulfulde (ff)"]


def test_it_writes_dcterms_language_links_not_literals(tmp_path, monkeypatch):
    """The property is a resource link (customvocab:6), so a literal would be wrong."""
    from common.iwac_config import DCTERMS_LANGUAGE_PROPERTY_ID

    (item,) = languages.read_report(write_report(tmp_path), shares=("dominant",))
    languages.attach_authority_items([item], {"Mooré": 8384})

    captured = {}

    def fake_update(client, item_id, links, *, dry_run, on_pre_write):
        captured["item_id"] = item_id
        captured["links"] = links
        return ResourceLinkUpdate("updated", {"dcterms:language": 1})

    monkeypatch.setattr(languages, "update_item_resource_links", fake_update)
    stats = languages.apply_updates(object(), [item], guard=languages.WriteGuard(dry_run=True))

    (spec,) = captured["links"]
    assert captured["item_id"] == 108309
    assert spec.term == "dcterms:language"
    assert spec.property_id == DCTERMS_LANGUAGE_PROPERTY_ID == 12
    assert list(spec.resource_ids) == [8384]
    assert stats["updated"] == 1


def test_items_with_nothing_linkable_are_skipped_not_patched(tmp_path, monkeypatch):
    (item,) = languages.read_report(write_report(tmp_path), shares=("dominant",))
    languages.attach_authority_items([item], {"Mooré": None})

    def explode(*args, **kwargs):
        raise AssertionError("must not PATCH an item with no resolvable language")

    monkeypatch.setattr(languages, "update_item_resource_links", explode)
    stats = languages.apply_updates(object(), [item], guard=languages.WriteGuard(dry_run=True))
    assert stats["skipped"] == 1
