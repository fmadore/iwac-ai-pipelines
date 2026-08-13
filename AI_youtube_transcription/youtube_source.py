"""Everything this pipeline needs to know about a YouTube video and its transcript.

Three things live here so they cannot drift between the three numbered steps:

* **URL handling** — Omeka stores the watch URL as a ``fabio:hasURL`` value, but
  Gemini only accepts the two canonical forms (``youtube.com/watch?v=<id>`` and
  ``youtu.be/<id>``). Everything else — ``/shorts/``, ``/live/``, ``/embed/``,
  ``m.youtube.com``, a playlist page — is rejected here rather than sent to the
  API to fail with a bare ``400 INVALID_ARGUMENT``.

* **The chunk plan** — a request's video payload costs ~93–103 tokens per second
  of runtime, measured across four videos of this corpus: 32 tokens of audio plus
  ~61–71 of frames at the default 1 fps. That is the documented *low*
  media-resolution rate, not the ~300/s default one, so Gemini already serves
  these YouTube videos at the cheap resolution and raising it would only cost
  more. A 1M context window therefore holds roughly 2.7 hours. Videos longer than
  the chunk budget are re-requested at different ``VideoMetadata`` offsets instead
  of being downloaded and split, which is the whole reason this pipeline needs no
  ``ffmpeg`` and no local disk.

* **Language detection** — ``dcterms:language`` is catalogued per item, and on
  this material it is not reliable enough to prompt from: the first video tested
  is cataloged ``Français`` and is actually dominated by Mooré. So the spoken
  languages are detected from two short sampled windows before transcription and
  named in the transcription prompt, rather than assumed from the record or left
  for the model to guess mid-transcript.

* **The transcript file format** — the header this pipeline writes and the body
  ``03`` uploads. Unlike ``AI_audio_summary``, the header is *not* part of what
  reaches Omeka: ``bibo:content`` is the archive's full-text field, exported to
  Hugging Face as ``OCR``, and "Generated using: ..." in it would pollute both
  the export and every search index built on it. Provenance rides an
  ``iwac:transcriptionModel`` value annotation instead.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

# Add repo root to path for shared imports
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.checkpoint import sha256_text  # noqa: E402  (path set above)
from common.iwac_config import LANGUAGE_LABELS_BY_CODE  # noqa: E402

#: A YouTube video id: 11 characters of the URL-safe base64 alphabet.
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

#: ``PT1H2M3S`` and friends, the xsd:duration Omeka stores in ``dcterms:extent``.
_ISO_DURATION_RE = re.compile(
    r"^P(?!$)(?:(\d+)D)?(?:T(?!$)(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?$"
)

_HEADER_SEPARATOR = "=" * 50

#: Default per-request window. Well inside the ~2.7 h a 1M context holds at the
#: measured token rate, and comfortably above the 33.6-minute longest video in
#: the collection today — so the corpus as it stands is transcribed in one
#: request per item and this bound only starts to matter for future uploads.
DEFAULT_CHUNK_MINUTES = 45

#: Seconds of the previous window re-sent at the start of the next one. Without
#: it an utterance straddling a boundary is lost by both requests; with it the
#: overlap is transcribed twice, which the prompt then resolves by absolute
#: timestamp (see :func:`chunk_prompt_suffix`).
DEFAULT_CHUNK_OVERLAP_SECONDS = 15

#: Runtime sampled per window when detecting the spoken languages, and how many
#: windows. Two 45-second samples cost ~9k tokens — under 5% of a 33-minute
#: transcription request — and one sample is not enough: a cold open, a jingle or
#: a French studio introduction over Mooré content would each answer for the
#: whole recording.
DEFAULT_LANGUAGE_SAMPLE_SECONDS = 45
DEFAULT_LANGUAGE_SAMPLES = 2

#: Frames per second sent to the model. 1.0 is the API default. Lowering it
#: scales only the frame half of the token cost (~71/s), leaving audio (~32/s)
#: untouched: 0.5 gives ~67 tok/s and 0.2 ~46 tok/s, measured. Worth it for a
#: static talking head, but these are news segments whose lower-thirds name the
#: speakers — drop frames and the transcript loses those names.
DEFAULT_FPS = 1.0


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------

def parse_video_id(value: Optional[str]) -> Optional[str]:
    """Extract the video id from a YouTube watch URL, or return ``None``.

    Accepts the two forms Omeka's own ``youtube`` media ingester accepts —
    ``https://www.youtube.com/watch?v=<id>`` (with any extra query parameters,
    such as the ``&t=`` a copied link carries) and ``https://youtu.be/<id>``.
    Everything else is rejected, including the ``/shorts/``, ``/live/`` and
    ``/embed/`` paths, so a URL this pipeline accepts is one the ingester would
    have accepted too.
    """
    if not value:
        return None
    parsed = urlparse(value.strip())
    if parsed.scheme not in ("http", "https"):
        return None

    host = (parsed.hostname or "").lower().removeprefix("www.")
    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/", 1)[0]
    elif host in ("youtube.com", "m.youtube.com", "music.youtube.com"):
        # Only the canonical /watch path: /shorts/, /live/ and /embed/ are
        # rejected upstream by Omeka and by the Gemini video fetcher alike.
        if parsed.path.rstrip("/") != "/watch":
            return None
        candidate = (parse_qs(parsed.query).get("v") or [""])[0]
    else:
        return None

    return candidate if _VIDEO_ID_RE.match(candidate) else None


def canonical_watch_url(video_id: str) -> str:
    """The single URL form this pipeline sends to Gemini."""
    return f"https://www.youtube.com/watch?v={video_id}"


# ---------------------------------------------------------------------------
# Durations
# ---------------------------------------------------------------------------

def parse_iso_duration(value: Optional[str]) -> Optional[int]:
    """Parse an ``xsd:duration`` (``PT33M36S``) into whole seconds.

    Returns ``None`` for anything unparseable — including the month/year
    designators, which have no fixed length and never describe a recording.
    A missing duration is not an error: it only means the chunk plan falls back
    to a single request, which is correct for every video short enough to fit.
    """
    if not value:
        return None
    match = _ISO_DURATION_RE.match(value.strip())
    if not match:
        return None
    days, hours, minutes, seconds = match.groups()
    total = (
        int(days or 0) * 86_400
        + int(hours or 0) * 3_600
        + int(minutes or 0) * 60
        + float(seconds or 0)
    )
    # Rounded UP, not to nearest: this number becomes the last window's
    # ``end_offset``, so understating it by a second cuts a second of speech off
    # the end of the transcript, while overstating it costs nothing — Gemini
    # stops at the real end of the video.
    return math.ceil(total) or None


def format_hms(seconds: Optional[float]) -> str:
    """Format seconds as ``HH:MM:SS`` (``??:??:??`` when unknown)."""
    if seconds is None:
        return "??:??:??"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# The work list (written by 01, consumed by 02)
# ---------------------------------------------------------------------------

@dataclass
class VideoWork:
    """One Omeka item resolved to a fetchable YouTube URL."""

    item_id: int
    video_id: str
    url: str
    title: str = ""
    identifier: str = ""
    duration_seconds: Optional[int] = None
    language: str = ""
    has_content: bool = False

    def to_json(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "video_id": self.video_id,
            "url": self.url,
            "title": self.title,
            "identifier": self.identifier,
            "duration_seconds": self.duration_seconds,
            "language": self.language,
            "has_content": self.has_content,
        }

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "VideoWork":
        return cls(
            item_id=int(payload["item_id"]),
            video_id=str(payload["video_id"]),
            url=str(payload["url"]),
            title=str(payload.get("title") or ""),
            identifier=str(payload.get("identifier") or ""),
            duration_seconds=payload.get("duration_seconds"),
            language=str(payload.get("language") or ""),
            has_content=bool(payload.get("has_content", False)),
        )


WORK_LIST_VERSION = 1


def write_work_list(path: Path, videos: List[VideoWork], *, scope: Dict[str, Any]) -> Path:
    """Write the work list ``02`` reads, recording how it was collected."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": WORK_LIST_VERSION,
        "scope": scope,
        "videos": [video.to_json() for video in videos],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def read_work_list(path: Path) -> List[VideoWork]:
    """Read a work list written by :func:`write_work_list`.

    Raises:
        ValueError: when the file was written by a different format version, so
            a stale list is never silently transcribed as if it were current.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("version")
    if version != WORK_LIST_VERSION:
        raise ValueError(
            f"Work list {path} has format version {version!r}, expected "
            f"{WORK_LIST_VERSION}. Re-run 01_omeka_youtube_fetcher.py."
        )
    return [VideoWork.from_json(entry) for entry in payload.get("videos", [])]


# ---------------------------------------------------------------------------
# Chunk planning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    """One request window over a video.

    ``start`` includes the overlap re-sent from the previous window;
    ``content_start`` is where this chunk's *own* content begins, and is what
    the prompt is told to skip past. ``end`` is ``None`` only for a single-chunk
    plan, meaning "no offsets — send the whole video".
    """

    index: int
    total: int
    start: int
    end: Optional[int]
    content_start: int

    @property
    def is_whole_video(self) -> bool:
        return self.total == 1 and self.start == 0 and self.end is None

    def label(self) -> str:
        span = f"{format_hms(self.start)}–{format_hms(self.end)}" if self.end is not None else "full"
        return f"Chunk {self.index}/{self.total} | {span}"


def plan_chunks(
    duration_seconds: Optional[int],
    *,
    chunk_seconds: int,
    overlap_seconds: int = DEFAULT_CHUNK_OVERLAP_SECONDS,
) -> List[Chunk]:
    """Split a video's runtime into request windows.

    A video that fits the budget — or whose duration Omeka does not record —
    becomes one offset-free request. Anything longer is cut into fixed windows,
    each re-sending *overlap_seconds* of the one before it.
    """
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must not be negative")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be smaller than chunk_seconds")

    if not duration_seconds or duration_seconds <= chunk_seconds:
        return [Chunk(index=1, total=1, start=0, end=None, content_start=0)]

    total = math.ceil(duration_seconds / chunk_seconds)
    chunks: List[Chunk] = []
    for position in range(total):
        content_start = position * chunk_seconds
        chunks.append(Chunk(
            index=position + 1,
            total=total,
            start=max(0, content_start - (overlap_seconds if position else 0)),
            end=min((position + 1) * chunk_seconds, duration_seconds),
            content_start=content_start,
        ))
    return chunks


def chunk_prompt_suffix(chunk: Chunk) -> str:
    """Instructions appended to the prompt for one window of a longer video.

    Two things have to be said, and neither is optional. Timestamps have to be
    absolute, or every chunk after the first restarts at ``[00:00:00]`` and the
    stitched transcript claims the whole recording happened in its first window.
    And the overlap has to be dropped by the model, or the seam repeats a
    sentence or two — which is the failure the overlap exists to avoid, moved
    rather than fixed.
    """
    if chunk.is_whole_video:
        return ""
    lines = [
        "",
        "## This request covers one part of a longer recording",
        f"- You are given the window {format_hms(chunk.start)}–{format_hms(chunk.end)} "
        f"of a recording that runs longer (part {chunk.index} of {chunk.total}).",
        "- Timestamps must be ABSOLUTE positions in the full recording, not offsets "
        f"within this window: the first moment you receive is {format_hms(chunk.start)}, "
        "not 00:00:00.",
    ]
    if chunk.content_start > chunk.start:
        lines.append(
            f"- The previous part already covered everything up to "
            f"{format_hms(chunk.content_start)}. Skip any utterance that begins before "
            f"that timestamp; start with the first one that begins at or after it."
        )
    lines.append(
        "- Do not summarise or announce the window; output only the transcript, "
        "continuing as if mid-document."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

#: Response schema for the detection pass. Kept to three flat fields: the point
#: is a machine-readable answer, and a schema the model can satisfy in a few
#: hundred tokens is one it will not truncate.
LANGUAGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "languages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name_en": {
                        "type": "string",
                        "description": "English name of the language, e.g. 'Mooré'.",
                    },
                    "bcp47": {
                        "type": "string",
                        "description": "BCP-47 / ISO 639 code, e.g. 'mos', 'fr', 'ar'.",
                    },
                    "share": {
                        "type": "string",
                        "enum": ["dominant", "secondary", "occasional"],
                    },
                },
                "required": ["name_en", "bcp47", "share"],
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["languages"],
}

LANGUAGE_DETECTION_PROMPT = """
Identify every language actually SPOKEN in these samples, which are windows taken
from one recording.

- Judge by the speech, not by on-screen text: West African broadcast material is
  routinely captioned in French while the speakers use a local language.
- Report a language only if you hear it spoken. Do not list a language you merely
  expect from the setting, the topic, or the captions.
- Mark exactly one language `dominant` when one clearly carries most of the
  speech; use `secondary` for a language present throughout in less of it, and
  `occasional` for isolated phrases, quotations or Qur'anic recitation.
- Give the BCP-47 or ISO 639 code (`fr`, `mos`, `dyu`, `ar`, `ha`, `en`, ...).
- Use `notes` only for something a transcriber must know: heavy code-switching,
  overlapping speech, unintelligible audio, or a language you could not name.

Output only the JSON object.
""".strip()


@dataclass(frozen=True)
class DetectedLanguage:
    """One language the detection pass heard."""

    name_en: str
    bcp47: str
    share: str = "dominant"

    def describe(self) -> str:
        return f"{self.bcp47} ({self.name_en}, {self.share})"

    @property
    def code(self) -> str:
        """The primary subtag, lowercased (``fr-FR`` → ``fr``)."""
        return self.bcp47.split("-", 1)[0].strip().lower()


#: ``dcterms:language`` is a link to an authority item whose title is a FRENCH
#: language name — there is no ISO code anywhere in the record. Detection answers
#: with English names and ISO codes, so comparing the two directly reports
#: "Français" and "French" as a disagreement, which is how the first live run
#: flagged a correctly catalogued item.
#:
#: Derived from ``iwac_config.LANGUAGE_LABELS_BY_CODE`` rather than written out
#: again: the two directions of one mapping drifting apart is how a label ends up
#: linkable but not comparable. That map holds exactly the languages with an
#: Omeka authority record, which is also exactly the set that can appear in
#: ``dcterms:language``.
CATALOGUED_LANGUAGE_CODES: Dict[str, str] = {}  # populated below, keyed on fold()


def fold(text: str) -> str:
    """Lowercase and strip diacritics, so ``Mooré`` matches ``moore``."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(
        char for char in decomposed if not unicodedata.combining(char)
    ).strip().lower()


CATALOGUED_LANGUAGE_CODES.update(
    {fold(label): code for code, label in LANGUAGE_LABELS_BY_CODE.items()}
)


def catalogued_language_code(label: str) -> Optional[str]:
    """Map a ``dcterms:language`` authority label to an ISO code, if known."""
    return CATALOGUED_LANGUAGE_CODES.get(fold(label))


def dominant_languages(languages: List[DetectedLanguage]) -> List[DetectedLanguage]:
    """The detected languages marked dominant, or the first one as a fallback."""
    return [lang for lang in languages if lang.share == "dominant"] or languages[:1]


def language_matches(catalogued: str, languages: List[DetectedLanguage]) -> bool:
    """True when the catalogued language is among the dominant detected ones.

    An unrecorded catalogued language, or one this module cannot map to a code,
    counts as a match: the point is to surface records that contradict the audio,
    and a label nobody has mapped yet is not evidence of a contradiction.
    """
    if not catalogued.strip() or not languages:
        return True
    dominant = dominant_languages(languages)
    code = catalogued_language_code(catalogued)
    if code and any(lang.code == code for lang in dominant):
        return True
    if code and any(lang.code for lang in dominant):
        # Both sides carry a code and they disagree — a real mismatch.
        return False
    folded = fold(catalogued)
    return any(
        folded in fold(lang.name_en) or fold(lang.name_en) in folded for lang in dominant
    )


def parse_detected_languages(payload: Any) -> List[DetectedLanguage]:
    """Read the detection response into ordered languages, dominant first.

    Tolerates a missing or malformed entry rather than failing the video: a
    transcription with no detected language is degraded, not useless — the prompt
    simply falls back to asking the model to work it out itself.
    """
    if not isinstance(payload, dict):
        return []
    order = {"dominant": 0, "secondary": 1, "occasional": 2}
    found: List[DetectedLanguage] = []
    for entry in payload.get("languages") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name_en") or "").strip()
        code = str(entry.get("bcp47") or "").strip()
        if not (name or code):
            continue
        share = str(entry.get("share") or "dominant").strip().lower()
        found.append(DetectedLanguage(
            name_en=name or code,
            bcp47=code or "und",
            share=share if share in order else "dominant",
        ))
    return sorted(found, key=lambda lang: order.get(lang.share, 3))


def plan_language_samples(
    duration_seconds: Optional[int],
    *,
    sample_seconds: int = DEFAULT_LANGUAGE_SAMPLE_SECONDS,
    samples: int = DEFAULT_LANGUAGE_SAMPLES,
) -> List[tuple[int, Optional[int]]]:
    """Pick the windows to sample for language detection.

    Anchored at fractions of the runtime rather than at the start, because the
    opening seconds of these videos are a channel jingle or a French voice-over
    title card as often as they are the actual speech.
    """
    if sample_seconds <= 0:
        raise ValueError("sample_seconds must be positive")
    if samples <= 0:
        raise ValueError("samples must be positive")

    if not duration_seconds:
        # No catalogued duration: one window from the start is all we can place.
        return [(0, sample_seconds)]
    if duration_seconds <= sample_seconds * samples:
        # Sampling would cost as much as sending the whole thing — so send it.
        return [(0, None)]

    anchors = (0.10, 0.55, 0.85, 0.30, 0.70)[:samples]
    windows: List[tuple[int, Optional[int]]] = []
    for anchor in anchors:
        start = int(duration_seconds * anchor)
        windows.append((start, min(start + sample_seconds, duration_seconds)))
    return windows


def language_prompt_suffix(
    languages: List[DetectedLanguage],
    *,
    catalogued: str = "",
) -> str:
    """Tell the transcription prompt which languages to expect.

    Naming the languages is worth a paragraph: an unprompted model transcribing
    Mooré tends to render it as approximate French, which reads as a clean
    transcript and is not one.
    """
    if not languages:
        lines = [
            "",
            "## Spoken language",
            "- The language of this recording has not been established. Identify it "
            "yourself from the speech and transcribe in it — never substitute a "
            "language you find easier to render.",
        ]
        if catalogued:
            lines.append(
                f"- The catalogue record says {catalogued}, which is unverified and "
                "may describe only part of the speech."
            )
        return "\n".join(lines)

    dominant = dominant_languages(languages)
    others = [lang for lang in languages if lang not in dominant]

    lines = [
        "",
        "## Spoken languages (detected from samples of this recording)",
        "- Dominant: " + ", ".join(f"{lang.name_en} ({lang.bcp47})" for lang in dominant),
    ]
    if others:
        lines.append(
            "- Also present: "
            + ", ".join(f"{lang.name_en} ({lang.bcp47}, {lang.share})" for lang in others)
        )
    lines.append(
        "- These were detected from short samples, so treat them as what to expect, "
        "not as a limit: transcribe whatever is actually spoken."
    )
    if catalogued and not language_matches(catalogued, languages):
        lines.append(
            f"- The catalogue record says {catalogued}, which disagrees with the "
            "detected dominant language. Follow the audio, not the record."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Transcript files
# ---------------------------------------------------------------------------

#: Header fields, in the order written. ``Chunks`` records ``done/total`` so an
#: incomplete transcript is visible on disk and can be refused by ``03``.
HEADER_TITLE = "Transcription of"
HEADER_ITEM = "Omeka item"
HEADER_IDENTIFIER = "Identifier"
HEADER_SOURCE = "Source"
HEADER_DURATION = "Duration"
HEADER_GENERATOR = "Generated using"
HEADER_PROMPT = "Prompt"
HEADER_LANGUAGES = "Languages detected"
HEADER_CATALOGUED = "Catalogued as"
HEADER_CHUNKS = "Chunks"


@dataclass
class Transcript:
    """A transcript file read back from disk."""

    path: Path
    header: Dict[str, str] = field(default_factory=dict)
    body: str = ""

    @property
    def chunks_done(self) -> Optional[int]:
        return self._chunk_counts()[0]

    @property
    def chunks_total(self) -> Optional[int]:
        return self._chunk_counts()[1]

    @property
    def complete(self) -> bool:
        """True when every planned chunk produced text.

        A file with no ``Chunks`` header predates the field or was hand-edited;
        treat it as complete rather than unusable, but the body still has to
        carry something.
        """
        done, total = self._chunk_counts()
        if done is None or total is None:
            return bool(self.body.strip())
        return done == total and bool(self.body.strip())

    def _chunk_counts(self) -> tuple[Optional[int], Optional[int]]:
        raw = self.header.get(HEADER_CHUNKS, "")
        match = re.match(r"^\s*(\d+)\s*/\s*(\d+)", raw)
        if not match:
            return None, None
        return int(match.group(1)), int(match.group(2))


def transcript_path(output_dir: Path, item_id: int) -> Path:
    """Transcripts are named after the Omeka item they belong to.

    ``03`` therefore needs no identifier lookup: unlike the audio pipeline,
    whose files are named after a ``dcterms:identifier`` and cost one search
    request per file to resolve, the item id is already in the filename.
    """
    return Path(output_dir) / f"{item_id}.txt"


def write_transcript(
    output_dir: Path,
    video: VideoWork,
    body: str,
    *,
    generator: str,
    prompt_label: str,
    prompt_sha256: str,
    chunks_done: int,
    chunks_total: int,
    languages: Optional[List[DetectedLanguage]] = None,
) -> Path:
    """Write one transcript with its provenance header."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = transcript_path(output_dir, video.item_id)

    lines = [
        f"{HEADER_TITLE}: {video.title or video.url}",
        f"{HEADER_ITEM}: {video.item_id}",
        f"{HEADER_IDENTIFIER}: {video.identifier}",
        f"{HEADER_SOURCE}: {video.url}",
        f"{HEADER_DURATION}: {format_hms(video.duration_seconds)}",
        f"{HEADER_GENERATOR}: {generator}",
        f"{HEADER_PROMPT}: {prompt_label} (#{prompt_sha256[:12]})",
        f"{HEADER_LANGUAGES}: "
        + ("; ".join(lang.describe() for lang in languages) if languages else "not detected"),
        f"{HEADER_CATALOGUED}: {video.language or 'unrecorded'}",
        f"{HEADER_CHUNKS}: {chunks_done}/{chunks_total}",
        _HEADER_SEPARATOR,
        "",
    ]
    path.write_text("\n".join(lines) + "\n" + body.strip() + "\n", encoding="utf-8")
    return path


def read_transcript(path: Path) -> Transcript:
    """Read a transcript, separating its header from the body ``03`` uploads."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    header: Dict[str, str] = {}

    if _HEADER_SEPARATOR in text:
        head, _, body = text.partition(_HEADER_SEPARATOR)
        for line in head.splitlines():
            name, sep, value = line.partition(":")
            if sep and name.strip():
                header[name.strip()] = value.strip()
    else:
        # No separator: the whole file is the transcript. Never guess which
        # leading lines were metadata — a wrong guess silently truncates text.
        body = text

    return Transcript(path=path, header=header, body=body.strip())


# ---------------------------------------------------------------------------
# Looping detection
# ---------------------------------------------------------------------------

#: Window size, in words, used to detect a repeating transcript.
LOOP_NGRAM_WORDS = 12

#: How many times one window may repeat before the transcript is judged to be
#: looping. Calibrated on the first full-corpus run (44 videos, Flash-Lite): the
#: 41 sound transcripts peaked at 7 repeats — a refrain, a prayer formula, a
#: repeated station ident — while the three broken ones scored 575, 1,266 and
#: 2,744. Nothing lands between 7 and 575, so 20 is well clear of both.
MAX_NGRAM_REPEAT = 20

#: A looping run also burns the whole output budget, so it arrives truncated. That
#: is a symptom, not the diagnosis: a truncated transcript can be legitimate.
LOOP_MIN_WORDS = LOOP_NGRAM_WORDS * 3


def most_repeated_ngram(text: str, *, size: int = LOOP_NGRAM_WORDS) -> tuple[int, str]:
    """Return ``(count, ngram)`` for the most repeated word window in *text*."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < size * 2:
        return 0, ""
    counts: Dict[str, int] = {}
    for index in range(len(words) - size + 1):
        gram = " ".join(words[index:index + size])
        counts[gram] = counts.get(gram, 0) + 1
    gram, count = max(counts.items(), key=lambda pair: pair[1])
    return count, gram


def looping_reason(text: str) -> Optional[str]:
    """Return a short reason when *text* is a degenerate repeating transcript.

    This is the failure mode that matters most here and the one nothing else
    catches. A model that cannot render a language — Mooré, on this corpus — does
    not fail visibly: it emits one plausible clause over and over until it hits the
    output cap, producing a file that is the right shape, marked complete, and
    entirely worthless. Three of the first 44 videos did exactly that.

    Length and window-completeness checks cannot see it, so without this the text
    goes into ``bibo:content`` as archive full text, is exported to Hugging Face as
    ``OCR``, and is indexed for search.
    """
    if len(text.split()) < LOOP_MIN_WORDS:
        return None
    count, _ = most_repeated_ngram(text)
    if count >= MAX_NGRAM_REPEAT:
        return f"looping-{count}x"
    return None


def join_chunks(chunks: List[tuple[Chunk, str]]) -> str:
    """Join the transcribed windows, marking every seam but the first.

    Failed chunks are simply absent — a marker in the body would be uploaded to
    Omeka as though it were content, which is the mistake
    ``common/gemini_page_processor`` was refactored to stop making. The
    ``Chunks: done/total`` header is where an incomplete run is recorded.
    """
    parts: List[str] = []
    for position, (chunk, text) in enumerate(chunks):
        if position == 0:
            parts.append(text.strip())
        else:
            parts.append(f"\n\n--- {chunk.label()} ---\n\n{text.strip()}")
    return "".join(parts)


def work_fingerprint(video: VideoWork, chunks_total: int) -> str:
    """Identify a completed transcript for the checkpoint.

    Covers the inputs that change what the transcript *should* contain. Model
    and prompt are deliberately excluded — they live in the checkpoint context,
    which invalidates the whole run rather than one entry.
    """
    return sha256_text(
        f"{video.url}|{video.duration_seconds or 0}|{chunks_total}"
    )
