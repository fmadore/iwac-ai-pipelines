#!/usr/bin/env python3
"""
Transcribe YouTube-hosted videos with Gemini, straight from their URLs.

Gemini accepts a YouTube watch URL as a ``file_data`` part: Google's servers
fetch and decode the video, so this pipeline never touches the stream. There is
no download, no ``ffmpeg`` re-encode, no Files API upload and no local media on
disk — which is what makes it thinner than ``AI_audio_summary`` rather than a
copy of it.

Two requests are made per video:

1. **Language detection** — two short sampled windows, answered against a small
   JSON schema. ``dcterms:language`` is not trustworthy enough to prompt from on
   this material (the first video tested is catalogued ``Français`` and is
   actually dominated by Mooré), and an unprompted model transcribing Mooré tends
   to render it as approximate French — which reads as a clean transcript and is
   not one. Costs ~9k tokens, under 5% of a 33-minute transcription.

2. **Transcription** — one request per video, or one per ``VideoMetadata`` window
   when the runtime exceeds the per-request budget.

Video payload costs ~103 tokens per second of runtime at the default 1 fps
(32 audio + ~71 frames, measured on this corpus). That is the documented *low*
media-resolution rate, so Gemini already serves these videos at the cheap
resolution; ``--fps`` is the lever that lowers it further, at the cost of the
on-screen lower-thirds that name the speakers.

Usage:
    python 02_AI_transcribe_youtube.py
    python 02_AI_transcribe_youtube.py --model gemini-3.5-flash-lite --prompt 1
    python 02_AI_transcribe_youtube.py --rpm 5 --fps 0.5
    python 02_AI_transcribe_youtube.py --item-id 108263 --force

Requirements:
    - GEMINI_API_KEY in .env or the environment
    - A work list from 01_omeka_youtube_fetcher.py
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google.genai import types
from google.genai import errors as genai_errors

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Add repo root to path for shared imports, and this pipeline's own directory
# for the sibling format module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.checkpoint import CheckpointMismatch, JsonCheckpoint, sha256_text
from common.console_utils import count_table, key_value_table, standard_progress
from common.gemini_utils import (
    build_gemini_client,
    build_generation_config,
    extract_text_from_response,
)
from common.log_redaction import install_credential_redaction
from common.prompt_loader import discover_prompts, load_prompt_md
from common.rate_limiter import QuotaExhaustedError, RateLimiter, is_quota_exhausted

from youtube_source import (
    Chunk,
    DEFAULT_CHUNK_MINUTES,
    DEFAULT_CHUNK_OVERLAP_SECONDS,
    DEFAULT_FPS,
    DEFAULT_LANGUAGE_SAMPLES,
    DEFAULT_LANGUAGE_SAMPLE_SECONDS,
    DetectedLanguage,
    LANGUAGE_DETECTION_PROMPT,
    LANGUAGE_SCHEMA,
    VideoWork,
    chunk_prompt_suffix,
    dominant_languages,
    format_hms,
    join_chunks,
    language_matches,
    language_prompt_suffix,
    looping_reason,
    parse_detected_languages,
    plan_chunks,
    plan_language_samples,
    read_work_list,
    transcript_path,
    work_fingerprint,
    write_transcript,
)

load_dotenv()
install_credential_redaction()

console = Console()
LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.resolve()
WORK_LIST_PATH = SCRIPT_DIR / "work" / "youtube_videos.json"
OUTPUT_DIR = SCRIPT_DIR / "Transcriptions"
CHECKPOINT_NAME = ".youtube_transcription_checkpoint.json"
LANGUAGE_REPORT_NAME = "_language_report.json"

#: Both models are *pinned* releases, not the ``gemini-flash-latest`` rolling
#: aliases the video pipeline uses, because step 03 stamps an
#: ``iwac:transcriptionModel`` annotation. A rolling alias reports its own
#: version as the string "Gemini Flash Latest", so a run through it cannot
#: confirm which model produced the text — and an annotation that names a model
#: the run cannot confirm is provenance in name only. See the retired-key notes
#: in ``common/iwac_config.py``.
ALLOWED_MODELS = ("gemini-3.5-flash-lite", "gemini-3.7-flash")

#: Flash-Lite by default: cost scales with runtime here, the corpus is 9.3 h of
#: video today with more channels to come, and it is catalogued as overwhelmingly
#: French. On French speech the two models are close.
#:
#: They are NOT close on the local languages. Run through this pipeline on the
#: same 81-second Mooré video, Flash-Lite produced 173 words that collapse into
#: hyphenated syllable fragments halfway through, while 3.6 Flash produced 310
#: with plausible romanisation and honest ``[inaudible]`` markers. Naming the
#: language in the prompt is what makes Flash-Lite attempt Mooré at all — without
#: it, it renders the speech as French — but attempting is not managing. Re-run
#: whatever the language report flags as non-French with
#: ``--model gemini-3.7-flash``.
#:
#: That second slot moved from 3.6 to 3.7 Flash on 2026-08-14. The Mooré
#: comparison above was measured on 3.6 and has not been repeated on 3.7, so
#: read it as "the Flash tier handles local languages that Flash-Lite does not",
#: which is the claim it is used for here — not as a 3.7 benchmark. The 10
#: transcripts already annotated with item 79611 keep naming 3.6, correctly.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

#: Transient Gemini API errors worth retrying with backoff.
RETRYABLE_API_CODES = (429, 500, 503)

#: Attempts allowed when the model returns a degenerate repeating transcript.
#: Deliberately far below ``max_retries``: see the comment in ``transcribe_chunk``.
LOOP_MAX_ATTEMPTS = 2

DEFAULT_PROMPT = """
Transcribe the spoken content of this video verbatim in the language actually
spoken. Insert a [hh:mm:ss] timestamp at the start of every speaker turn. Do not
summarise, do not translate, and do not omit passages.
""".strip()


class RetryableResponse(Exception):
    """A response came back empty or blocked but a fresh sample may succeed.

    The message is a short, parenthesis-free reason code (``RECITATION``,
    ``MAX_TOKENS``, ``no-candidates``) suitable for the run report.
    """


class VideoUnavailable(Exception):
    """Gemini could not fetch the video — terminal for this item.

    A YouTube URL that is private, unlisted, deleted, region-blocked or simply
    wrong comes back as ``400 INVALID_ARGUMENT`` with no detail (verified against
    the live API). Retrying cannot help, and it must not be counted as a quota
    problem: it means this item needs the downloaded-media path instead, which is
    the one case this pipeline cannot serve.
    """


def finish_reason_code(finish_reason: Any) -> str:
    """Return the bare finish-reason name (e.g. ``MAX_TOKENS``)."""
    if finish_reason is None:
        return "UNKNOWN"
    name = getattr(finish_reason, "name", None)
    if name:
        return name
    text = str(finish_reason)
    return text.rsplit(".", 1)[-1] if "." in text else text


@dataclass
class VideoResult:
    """Outcome of one video."""

    video: VideoWork
    chunks_total: int = 0
    chunks_done: int = 0
    languages: List[DetectedLanguage] = field(default_factory=list)
    output_file: Optional[Path] = None
    status: str = "failed"          # transcribed | partial | skipped | unavailable | failed
    reason: str = ""

    @property
    def complete(self) -> bool:
        return self.chunks_total > 0 and self.chunks_done == self.chunks_total


class YouTubeTranscriber:
    """Sends YouTube URLs to Gemini and writes transcripts."""

    def __init__(
        self,
        *,
        model: str,
        transcription_prompt: str,
        prompt_label: str,
        api_key: Optional[str] = None,
        requests_per_minute: Optional[int] = None,
        chunk_seconds: int = DEFAULT_CHUNK_MINUTES * 60,
        overlap_seconds: int = DEFAULT_CHUNK_OVERLAP_SECONDS,
        fps: float = DEFAULT_FPS,
        detect_language: bool = True,
        forced_language: str = "",
        max_retries: int = 4,
    ):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file or environment variables")

        self.model = model
        self.transcription_prompt = transcription_prompt
        self.prompt_label = prompt_label
        self.prompt_sha256 = sha256_text(transcription_prompt)
        self.client = build_gemini_client(api_key)
        self.rate_limiter = RateLimiter(requests_per_minute, logger=LOGGER)
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.fps = fps
        self.detect_language = detect_language
        self.forced_language = forced_language
        self.max_retries = max_retries

    # ---- Request building ---------------------------------------------------

    def _video_part(self, url: str, *, start: Optional[int] = None, end: Optional[int] = None):
        """Build the ``file_data`` part for a (possibly clipped) window.

        ``video_metadata`` is only meaningful on an ``inline_data`` or
        ``file_data`` part, which is exactly what a YouTube URL is. Offsets are
        the ``"<n>s"`` strings the API documents.
        """
        metadata: Dict[str, Any] = {}
        if start is not None and (start > 0 or end is not None):
            metadata["start_offset"] = f"{int(start)}s"
        if end is not None:
            metadata["end_offset"] = f"{int(end)}s"
        if self.fps and self.fps != DEFAULT_FPS:
            metadata["fps"] = self.fps
        return types.Part(
            file_data=types.FileData(file_uri=url),
            video_metadata=types.VideoMetadata(**metadata) if metadata else None,
        )

    def _generate(self, contents: List[Any], *, response_schema=None, max_output_tokens: int):
        """One Gemini call, with the shared rate limiter and error translation."""
        self.rate_limiter.wait()
        try:
            return self.client.models.generate_content(
                model=self.model,
                contents=contents,
                # No temperature: it is vendor-owned, and on a 40-minute
                # transcription a lowered one is what makes a model loop on one
                # paragraph for the rest of the recording.
                config=build_generation_config(
                    self.model,
                    max_output_tokens=max_output_tokens,
                    response_schema=response_schema,
                ),
            )
        except genai_errors.APIError as exc:
            if is_quota_exhausted(exc):
                raise QuotaExhaustedError(str(exc)) from exc
            if getattr(exc, "code", None) == 400:
                raise VideoUnavailable(
                    "Gemini could not fetch the video (private, unlisted, removed, "
                    "region-blocked, or not a valid video id)"
                ) from exc
            raise

    # ---- Language detection -------------------------------------------------

    def detect_languages(self, video: VideoWork) -> List[DetectedLanguage]:
        """Identify the spoken languages from short sampled windows."""
        windows = plan_language_samples(
            video.duration_seconds,
            sample_seconds=DEFAULT_LANGUAGE_SAMPLE_SECONDS,
            samples=DEFAULT_LANGUAGE_SAMPLES,
        )
        # Several windows of the SAME video in one request. The docs' "one video
        # per request" guidance is about mixing different videos; sampling one
        # recording twice is what stops a jingle or a French title card from
        # answering for the whole thing, and stays far inside the 10-video cap.
        contents: List[Any] = [
            self._video_part(video.url, start=start, end=end) for start, end in windows
        ]
        contents.append(types.Part(text=LANGUAGE_DETECTION_PROMPT))

        response = self._generate(contents, response_schema=LANGUAGE_SCHEMA, max_output_tokens=2048)
        raw = extract_text_from_response(response)
        if not raw:
            raise RetryableResponse(finish_reason_code(
                response.candidates[0].finish_reason if response.candidates else None
            ))
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RetryableResponse("bad-json") from exc
        return parse_detected_languages(payload)

    def _detect_with_retries(self, video: VideoWork) -> List[DetectedLanguage]:
        """Detect languages, tolerating failure.

        Detection is an aid, not a gate: if it cannot be established, the
        transcription prompt says so and asks the model to work it out, which is
        strictly better than skipping the video.
        """
        for attempt in range(2):
            try:
                return self.detect_languages(video)
            except (QuotaExhaustedError, VideoUnavailable):
                raise
            except Exception as exc:
                LOGGER.warning("Language detection failed for item %s: %s", video.item_id, exc)
                if attempt == 0:
                    time.sleep(3)
        console.print("  [yellow]⚠[/] Language detection failed — transcribing without it")
        return []

    # ---- Transcription ------------------------------------------------------

    def _prompt_for(self, chunk: Chunk, languages: List[DetectedLanguage], catalogued: str) -> str:
        return "".join([
            self.transcription_prompt,
            "\n",
            language_prompt_suffix(languages, catalogued=catalogued),
            "\n",
            chunk_prompt_suffix(chunk),
        ]).rstrip() + "\n"

    def transcribe_chunk(
        self,
        video: VideoWork,
        chunk: Chunk,
        languages: List[DetectedLanguage],
    ) -> str:
        """Transcribe one window, retrying transient failures.

        Raises:
            QuotaExhaustedError: daily quota gone — the caller stops the batch.
            VideoUnavailable: the video cannot be fetched at all.
            RetryableResponse: every attempt came back empty or blocked.
        """
        part = self._video_part(
            video.url,
            start=None if chunk.is_whole_video else chunk.start,
            end=chunk.end if not chunk.is_whole_video else None,
        )
        prompt = self._prompt_for(chunk, languages, video.language)
        last_reason = "error"
        loop_attempts = 0

        for attempt in range(self.max_retries):
            try:
                # Media first, then the prompt, per Google's guidance.
                response = self._generate(
                    [part, types.Part(text=prompt)], max_output_tokens=65_536
                )
                return self._response_to_text(response)
            except (QuotaExhaustedError, VideoUnavailable):
                raise
            except genai_errors.APIError as exc:
                code = getattr(exc, "code", None)
                last_reason = f"API-{code}"
                if code not in RETRYABLE_API_CODES:
                    raise RetryableResponse(last_reason) from exc
            except RetryableResponse as exc:
                last_reason = str(exc)
                if last_reason.startswith("looping-"):
                    # A loop gets a smaller budget than a transient error. It runs
                    # to the output cap every time, so each retry costs a full
                    # 65k-token generation, and it signals a model that cannot
                    # render this language rather than a bad draw — spending four
                    # attempts to confirm that is just paying to fail slower.
                    loop_attempts += 1
                    if loop_attempts >= LOOP_MAX_ATTEMPTS:
                        console.print(
                            f"  [red]✗[/] {chunk.label()} looped "
                            f"{loop_attempts}× — giving up on this window"
                        )
                        raise RetryableResponse(last_reason) from exc
            if attempt < self.max_retries - 1:
                delay = 2 ** (attempt + 1) + random.uniform(0, 2)
                console.print(
                    f"  [yellow]⏳[/] {chunk.label()} failed ([yellow]{last_reason}[/]) — "
                    f"retrying in {delay:.1f}s (attempt {attempt + 2}/{self.max_retries})"
                )
                time.sleep(delay)

        raise RetryableResponse(last_reason)

    def _response_to_text(self, response) -> str:
        """Pull the transcript out of a response, or signal a retry.

        ``MAX_TOKENS`` keeps whatever was generated: a truncated window is still
        most of a window, and discarding it loses text that was paid for. The
        truncation is marked inline, because a reader has to be able to tell a
        cut from a silence.
        """
        feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None) if feedback else None
        if block_reason:
            raise RetryableResponse(f"prompt-blocked-{finish_reason_code(block_reason)}")
        if not response.candidates:
            raise RetryableResponse("no-candidates")

        code = finish_reason_code(response.candidates[0].finish_reason)
        text = extract_text_from_response(response)

        # Checked before MAX_TOKENS, because a looping run is *why* the output was
        # truncated: keeping its "partial text" would keep 35,000 words of the same
        # clause. Retried rather than failed outright — the loop is a sampling
        # accident often enough to be worth one more draw.
        if text:
            loop = looping_reason(text)
            if loop:
                console.print(
                    f"  [yellow]⚠[/] Degenerate repeating output ([yellow]{loop}[/]) — discarded"
                )
                raise RetryableResponse(loop)

        if code == "MAX_TOKENS":
            if text:
                console.print("  [yellow]⚠[/] Output truncated (MAX_TOKENS) — partial text kept")
                return text + "\n\n[... TRANSCRIPTION TRUNCATED — OUTPUT EXCEEDED MAX TOKENS ...]"
            raise RetryableResponse("MAX_TOKENS")
        if text:
            return text
        raise RetryableResponse(code if code not in ("STOP", "UNKNOWN") else "empty")

    # ---- Per-video ----------------------------------------------------------

    def transcribe_video(self, video: VideoWork, output_dir: Path) -> VideoResult:
        """Detect languages, transcribe every window, write one transcript."""
        chunks = plan_chunks(
            video.duration_seconds,
            chunk_seconds=self.chunk_seconds,
            overlap_seconds=self.overlap_seconds,
        )
        result = VideoResult(video=video, chunks_total=len(chunks))

        console.print(
            f"[cyan]🎬[/] Item [bold]{video.item_id}[/] · "
            f"{format_hms(video.duration_seconds)} · "
            f"{len(chunks)} request{'s' if len(chunks) > 1 else ''} · {video.url}"
        )

        try:
            if self.forced_language:
                result.languages = [DetectedLanguage(
                    name_en=self.forced_language, bcp47=self.forced_language, share="dominant"
                )]
            elif self.detect_language:
                result.languages = self._detect_with_retries(video)
                if result.languages:
                    console.print(
                        "  [green]🗣[/] "
                        + "; ".join(lang.describe() for lang in result.languages)
                    )
                    self._warn_language_mismatch(video, result.languages)
        except VideoUnavailable as exc:
            result.status, result.reason = "unavailable", str(exc)
            console.print(f"  [red]✗[/] {exc}")
            return result

        transcribed: List[Tuple[Chunk, str]] = []
        for chunk in chunks:
            try:
                text = self.transcribe_chunk(video, chunk, result.languages)
            except VideoUnavailable as exc:
                result.status, result.reason = "unavailable", str(exc)
                console.print(f"  [red]✗[/] {exc}")
                return result
            except RetryableResponse as exc:
                result.reason = str(exc)
                console.print(f"  [red]✗[/] {chunk.label()} gave up ([yellow]{exc}[/])")
                continue
            except QuotaExhaustedError:
                # Save what completed before letting the batch stop: the point of
                # partial results is that a quota reset resumes rather than restarts.
                self._save(result, transcribed, output_dir)
                raise
            if text.strip():
                transcribed.append((chunk, text))
                result.chunks_done += 1
            else:
                result.reason = result.reason or "empty"

        self._save(result, transcribed, output_dir)
        if not transcribed:
            result.status = "failed"
        elif result.complete:
            result.status = "transcribed"
        else:
            result.status = "partial"
        return result

    def _warn_language_mismatch(self, video: VideoWork, languages: List[DetectedLanguage]) -> None:
        """Flag a detected language that contradicts the catalogue record."""
        if language_matches(video.language, languages):
            return
        dominant = dominant_languages(languages)
        console.print(
            f"  [yellow]⚠[/] Catalogued as [cyan]{video.language}[/] but the dominant "
            f"spoken language is [cyan]{dominant[0].name_en}[/] — dcterms:language "
            "may need correcting on this item"
        )

    def _save(
        self,
        result: VideoResult,
        transcribed: List[Tuple[Chunk, str]],
        output_dir: Path,
    ) -> None:
        """Write the transcript, if anything was transcribed at all.

        Nothing is written when every window failed: an output file of headers
        and no text would be indistinguishable from a real transcript to step 03.
        """
        if not transcribed:
            return
        result.output_file = write_transcript(
            output_dir,
            result.video,
            join_chunks(transcribed),
            generator=f"Google {self.model}",
            prompt_label=self.prompt_label,
            prompt_sha256=self.prompt_sha256,
            chunks_done=result.chunks_done,
            chunks_total=result.chunks_total,
            languages=result.languages,
        )
        words = len(join_chunks(transcribed).split())
        console.print(
            f"  [green]✓[/] {result.chunks_done}/{result.chunks_total} window(s), "
            f"~{words:,} words → [cyan]{result.output_file.name}[/]"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe YouTube-hosted Omeka videos with Gemini (no download).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=list(ALLOWED_MODELS), default=None,
        help=f"Gemini model (default: interactive, then {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--prompt", type=int, default=None,
        help="Prompt number from prompts/ (default: interactive selection).",
    )
    parser.add_argument(
        "--work-list", type=Path, default=WORK_LIST_PATH,
        help="Work list written by 01_omeka_youtube_fetcher.py.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"Where transcripts are written (default: {OUTPUT_DIR.name}/).",
    )
    parser.add_argument(
        "--item-id", type=int, action="append", default=[], dest="item_ids",
        help="Transcribe only these items from the work list (repeatable).",
    )
    parser.add_argument(
        "--chunk-minutes", type=int, default=DEFAULT_CHUNK_MINUTES,
        help=f"Per-request window for long videos (default: {DEFAULT_CHUNK_MINUTES}). "
             "At ~103 tokens/second a 1M context holds about 160 minutes, so this is "
             "a margin, not the limit.",
    )
    parser.add_argument(
        "--chunk-overlap-seconds", type=int, default=DEFAULT_CHUNK_OVERLAP_SECONDS,
        help=f"Seconds re-sent at each window boundary (default: "
             f"{DEFAULT_CHUNK_OVERLAP_SECONDS}), so no utterance falls in the gap.",
    )
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_FPS,
        help=f"Frames per second sent to the model (default: {DEFAULT_FPS}, the API "
             "default). 0.5 costs ~67 tok/s and 0.2 ~46 tok/s instead of ~103, but "
             "drops the on-screen text that names speakers.",
    )
    parser.add_argument(
        "--no-detect-language", action="store_true",
        help="Skip the language-detection pass. The prompt then asks the model to "
             "identify the language itself.",
    )
    parser.add_argument(
        "--language", default="",
        help="Force the spoken language instead of detecting it (e.g. 'French').",
    )
    parser.add_argument(
        "--rpm", type=int, default=None,
        help="Throttle to this many requests per minute (default: no throttling).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-transcribe videos already recorded in the checkpoint, and replace "
             "a checkpoint written with a different model or prompt.",
    )
    return parser.parse_args()


def ask(question: str, *, fallback_note: str) -> str:
    """Prompt for a choice, treating a closed stdin as "take the default".

    Deliberately unlike the write guard's confirmation, which refuses on EOF
    because consent cannot be inferred. Nothing is written here: choosing a model
    or a prompt has a documented default, so a piped or scheduled run should
    proceed with it rather than die on a traceback.
    """
    try:
        return console.input(question).strip()
    except (EOFError, KeyboardInterrupt):
        console.print(f"\n[dim]No answer on stdin — {fallback_note}.[/]")
        return ""


def select_model_interactive() -> str:
    table = Table(title="🤖 Available Models", box=box.ROUNDED)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Model", style="green")
    table.add_column("Notes", style="dim")
    table.add_row("1", "gemini-3.5-flash-lite", "Default. Cheapest and fastest")
    table.add_row("2", "gemini-3.7-flash", "Better on non-French speech")
    console.print(table)
    choice = ask(
        f"\n[bold]Select a model (1-2) or press Enter for {DEFAULT_MODEL}:[/] ",
        fallback_note=f"using {DEFAULT_MODEL}",
    )
    return "gemini-3.7-flash" if choice == "2" else DEFAULT_MODEL


def select_prompt(number: Optional[int]) -> Tuple[str, str]:
    """Resolve the transcription prompt to ``(text, label)``.

    Uses ``discover_prompts`` rather than ``select_prompt_interactive`` because
    the label is written into the transcript header, and the shared helper
    returns only the text and the number.
    """
    options = discover_prompts(SCRIPT_DIR / "prompts")
    numbered = [option for option in options if option.number > 0]
    if not numbered:
        console.print("[yellow]⚠[/] No prompt files found — using the built-in default.")
        return DEFAULT_PROMPT, "built-in default"

    if number is None:
        table = Table(title="📝 Available Transcription Prompts", box=box.ROUNDED)
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Description", style="green")
        for option in numbered:
            table.add_row(str(option.number), option.description)
        console.print()
        console.print(table)
        answer = ask(
            f"\n[bold]Select a prompt (1-{len(numbered)}) or press Enter for "
            f"{numbered[0].number}:[/] ",
            fallback_note=f"using prompt {numbered[0].number}",
        )
        number = int(answer) if answer.isdigit() else numbered[0].number

    selected = next((option for option in numbered if option.number == number), None)
    if selected is None:
        raise ValueError(
            f"No prompt numbered {number} in {SCRIPT_DIR / 'prompts'} "
            f"(available: {', '.join(str(option.number) for option in numbered)})"
        )
    console.print(f"[green]✓[/] Prompt: [cyan]{selected.description}[/]")
    return load_prompt_md(selected.path), selected.description


def open_checkpoint(
    output_dir: Path, transcriber: YouTubeTranscriber, *, force: bool
) -> JsonCheckpoint:
    """Open the resume checkpoint, guarding provenance.

    The context pins model, prompt and chunk geometry: resuming a run under a
    different prompt would leave one corpus transcribed two different ways with
    nothing on the outside recording which item got which.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / CHECKPOINT_NAME
    existing = [
        item for item in output_dir.glob("*.txt") if item.stem.isdigit()
    ]
    if existing and not path.exists() and not force:
        raise CheckpointMismatch(
            f"Existing transcripts have no provenance checkpoint: {output_dir}. "
            "Use --force to replace them."
        )
    return JsonCheckpoint.open(
        path,
        {
            "pipeline": "youtube-transcription-v1",
            "model": transcriber.model,
            "prompt_sha256": transcriber.prompt_sha256,
            "chunk_seconds": transcriber.chunk_seconds,
            "overlap_seconds": transcriber.overlap_seconds,
            "fps": transcriber.fps,
            "language_detection": bool(transcriber.detect_language and not transcriber.forced_language),
            "forced_language": transcriber.forced_language,
        },
        reset=force,
    )


def select_pending(
    videos: List[VideoWork],
    checkpoint: JsonCheckpoint,
    transcriber: YouTubeTranscriber,
    output_dir: Path,
) -> Tuple[List[VideoWork], int]:
    """Split the work list into what still needs doing and what is done.

    A checkpoint entry only counts when the transcript is still on disk: a
    deleted file has to be regenerated, not assumed.
    """
    pending: List[VideoWork] = []
    done = 0
    for video in videos:
        chunks = len(plan_chunks(
            video.duration_seconds,
            chunk_seconds=transcriber.chunk_seconds,
            overlap_seconds=transcriber.overlap_seconds,
        ))
        if (
            checkpoint.matches(str(video.item_id), work_fingerprint(video, chunks))
            and transcript_path(output_dir, video.item_id).exists()
        ):
            done += 1
            continue
        pending.append(video)
    return pending, done


def write_language_report(output_dir: Path, results: List[VideoResult]) -> Optional[Path]:
    """Record what was heard against what was catalogued.

    Written as its own file because it is the run's one finding about the archive
    rather than about the transcripts: a ``dcterms:language`` that disagrees with
    the audio is a metadata correction this pipeline can see and cannot make.

    Merged into whatever is already there, keyed on item id. This step resumes —
    a checkpointed run transcribes only what is outstanding — so overwriting would
    shrink the report to the last batch and silently drop the findings for every
    item skipped, which is exactly the input step 04 reads.
    """
    rows = [
        {
            "item_id": result.video.item_id,
            "identifier": result.video.identifier,
            "catalogued": result.video.language,
            "detected": [
                {"name_en": lang.name_en, "bcp47": lang.bcp47, "share": lang.share}
                for lang in result.languages
            ],
            "agrees_with_record": language_matches(result.video.language, result.languages),
        }
        for result in results if result.languages
    ]
    if not rows:
        return None

    path = output_dir / LANGUAGE_REPORT_NAME
    merged: Dict[int, Dict[str, Any]] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            for row in existing if isinstance(existing, list) else []:
                if isinstance(row, dict) and "item_id" in row:
                    merged[int(row["item_id"])] = row
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            # A corrupt report must not lose this run's findings; start clean and say so.
            console.print(f"  [yellow]⚠[/] Could not read the existing report ({exc}) — replacing it")
            merged = {}
    # This run's detections win for the items it covered: they were made by the
    # model this run used, and the report is read as current.
    merged.update({row["item_id"]: row for row in rows})

    path.write_text(
        json.dumps(
            [merged[key] for key in sorted(merged)], ensure_ascii=False, indent=2
        ) + "\n",
        encoding="utf-8",
    )
    return path


def print_summary(results: List[VideoResult], output_dir: Path, *, quota_stopped: bool) -> None:
    counts: Dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1

    console.print()
    console.rule("[bold]Transcription Summary", style="cyan")
    console.print(count_table([
        ("[green]Transcribed[/]", counts.get("transcribed", 0)),
        ("[yellow]Partial (some windows failed)[/]", counts.get("partial") or None),
        ("[red]Video unavailable[/]", counts.get("unavailable") or None),
        ("[red]Failed[/]", counts.get("failed") or None),
        ("Output folder", str(output_dir)),
    ], title="📊 Results"))

    problems = [result for result in results if result.status in ("partial", "failed", "unavailable")]
    if problems:
        table = Table(title="Needs attention", box=box.ROUNDED)
        table.add_column("Item", style="cyan", justify="right")
        table.add_column("Status", style="yellow")
        table.add_column("Windows", justify="right", style="dim")
        table.add_column("Reason", style="dim")
        for result in problems:
            table.add_row(
                str(result.video.item_id),
                result.status,
                f"{result.chunks_done}/{result.chunks_total}",
                result.reason or "—",
            )
        console.print(table)

    if counts.get("unavailable"):
        console.print(
            "\n[yellow]⚠[/] An unavailable video is private, unlisted, removed or "
            "region-blocked. Gemini can only read public videos, so those items need "
            "a deposited media file and the AI_audio_summary path instead."
        )
    if quota_stopped:
        console.print(
            "\n[red bold]Daily quota exhausted — stopped early.[/] Everything "
            "transcribed so far is saved; re-run after the reset to continue."
        )
    else:
        mismatches = [
            result for result in results
            if result.languages and not language_matches(result.video.language, result.languages)
        ]
        if mismatches:
            console.print(
                f"\n[yellow]⚠[/] {len(mismatches)} item(s) are catalogued in a language "
                "other than the one dominant in the audio — see "
                f"[cyan]{LANGUAGE_REPORT_NAME}[/]"
            )


def main() -> int:
    args = parse_args()

    console.print(Panel(
        "Transcribe YouTube-hosted videos with Gemini straight from their URLs — "
        "no download, no ffmpeg, no Files API upload.",
        title="🎬 YouTube Transcription using Google Gemini",
        border_style="cyan",
    ))

    try:
        if args.chunk_minutes <= 0:
            raise ValueError("--chunk-minutes must be positive")
        if not 0 < args.fps <= 24:
            raise ValueError("--fps must be in (0, 24]")

        if not args.work_list.exists():
            console.print(
                f"\n[red]✗[/] No work list at [cyan]{args.work_list}[/].\n"
                "[dim]Run 01_omeka_youtube_fetcher.py first.[/]"
            )
            return 1
        videos = read_work_list(args.work_list)
        if args.item_ids:
            wanted = set(args.item_ids)
            videos = [video for video in videos if video.item_id in wanted]
        if not videos:
            console.print("[yellow]No videos in the work list match the requested items.[/]")
            return 0

        model = args.model or select_model_interactive()
        console.print(f"[green]✓[/] Model: [cyan]{model}[/]")
        prompt_text, prompt_label = select_prompt(args.prompt)

        transcriber = YouTubeTranscriber(
            model=model,
            transcription_prompt=prompt_text,
            prompt_label=prompt_label,
            requests_per_minute=args.rpm,
            chunk_seconds=args.chunk_minutes * 60,
            overlap_seconds=args.chunk_overlap_seconds,
            fps=args.fps,
            detect_language=not args.no_detect_language,
            forced_language=args.language.strip(),
        )

        checkpoint = open_checkpoint(args.output_dir, transcriber, force=args.force)
        pending, already_done = select_pending(videos, checkpoint, transcriber, args.output_dir)

        total_seconds = sum(video.duration_seconds or 0 for video in pending)
        console.print()
        console.print(key_value_table([
            ("Videos in work list", str(len(videos))),
            ("Already transcribed", str(already_done) if already_done else None),
            ("To transcribe", str(len(pending))),
            ("Total runtime", f"{total_seconds / 3600:.2f} h" if total_seconds else None),
            ("Est. input tokens", f"~{total_seconds * 103 / 1_000:,.0f}k" if total_seconds else None),
            ("Language detection", "off" if args.no_detect_language else
                                   f"forced: {args.language}" if args.language else "on"),
            ("Window", f"{args.chunk_minutes} min (+{args.chunk_overlap_seconds}s overlap)"),
            ("Frame rate", f"{args.fps} fps"),
            ("Rate limit", f"{args.rpm} RPM" if args.rpm else "none"),
            ("Output folder", str(args.output_dir)),
        ]))
        console.print()

        if not pending:
            console.print("[green]✓[/] Every video in the work list is already transcribed.")
            return 0

        results: List[VideoResult] = []
        quota_stopped = False

        with standard_progress(console, show_eta=True) as progress:
            task = progress.add_task("[cyan]Transcribing...", total=len(pending))
            for video in pending:
                try:
                    result = transcriber.transcribe_video(video, args.output_dir)
                    results.append(result)
                    if result.complete and result.output_file:
                        checkpoint.mark(
                            str(video.item_id),
                            work_fingerprint(video, result.chunks_total),
                        )
                except QuotaExhaustedError as exc:
                    LOGGER.error("Quota exhausted: %s", exc)
                    quota_stopped = True
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    console.print(f"  [red]✗[/] Item {video.item_id} failed: {exc}")
                    LOGGER.exception("Unexpected error on item %s", video.item_id)
                    results.append(VideoResult(
                        video=video, status="failed", reason=type(exc).__name__
                    ))
                progress.update(task, advance=1)

        write_language_report(args.output_dir, results)
        print_summary(results, args.output_dir, quota_stopped=quota_stopped)
        return 0 if all(result.status == "transcribed" for result in results) else 1

    except CheckpointMismatch as exc:
        console.print(f"\n[red]✗ Cannot resume:[/] {exc}")
        return 1
    except ValueError as exc:
        console.print(f"\n[red]✗ Configuration Error:[/] {exc}")
        if "GEMINI_API_KEY" in str(exc):
            console.print(
                "\n[bold]Set your Gemini API key:[/]\n"
                "  1. Get one at [link=https://aistudio.google.com/app/api-keys]"
                "https://aistudio.google.com/app/api-keys[/link]\n"
                "  2. Add GEMINI_API_KEY=... to the .env file in the repo root"
            )
        return 1
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Cancelled by user. Completed transcripts are saved.[/]")
        return 1
    except Exception:
        console.print("\n[red]✗ Unexpected error:[/]")
        console.print_exception()
        return 1


if __name__ == "__main__":
    sys.exit(main())
