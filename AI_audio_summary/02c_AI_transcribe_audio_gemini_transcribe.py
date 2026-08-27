#!/usr/bin/env python3
"""
Audio Transcription Script using Google Gemini 3.5 Transcribe

Transcribes audio and video files from the Audio folder with Google's dedicated
speech-to-text model, producing word-level timestamps and speaker labels
alongside the text.

Three things about this model shape the whole script, and none of them are true
of ``02_AI_transcribe_audio.py``:

* **It takes no prompt.** ``gemini-3.5-transcribe`` is an ASR model reached
  through the Interactions API, not a multimodal chat model. Sending a system
  instruction is a hard ``400 Developer instruction is not enabled for this
  model``, so none of the ``prompts/`` modes apply here: no translation into
  French or English, no Hausa segmentation, no editorial apparatus. What this
  script produces is the recording's own words. Use ``02`` for anything else.

* **Timestamps and "smart" are mutually exclusive, and they move the cap.**
  Verbatim mode takes ``timestamp_granularities`` and ``diarization_mode`` and
  caps a request at 30 minutes; smart mode post-processes into clean prose
  (disfluencies removed) and allows an hour, but rejects both options with a
  ``400``. Longer files are therefore split before upload, not after failure —
  see ``_segment_paths``.

* **Its locale list is not this archive's.** The API accepts any BCP-47 string
  without validating it, so ``--language mos-BF`` returns a confident, fluent,
  wrong transcript rather than an error. Five of the thirteen languages
  catalogued in this collection — Mooré, Dioula, Ewé, Kabyè, Dendi — have no
  locale here, and they are the West African ones. ``SUPPORTED_LOCALES`` is
  checked in this script precisely because the server will not check it.
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

# Add repo root to path for shared imports
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.gemini_utils import (
    build_gemini_client,
    delete_uploaded_file,
    upload_and_wait_active,
)
from common.rate_limiter import QuotaExhaustedError, is_quota_exhausted
from common.ffmpeg_utils import get_mime_type, probe_duration_seconds
from common.log_redaction import install_credential_redaction

# Credentials ride in Omeka query strings and provider headers; keep them
# out of anything urllib3 or an SDK decides to log.
install_credential_redaction()

from rich.panel import Panel
from rich.table import Table
from rich import box

from transcriber_base import SCRIPT_DIR, TranscriberBase, console, detect_ffmpeg, make_config_table
from segments import cleanup_temp_segments, segment_header, split_audio_file, transcription_path

# Load environment variables from .env file
load_dotenv()

MODEL = "gemini-3.5-transcribe"

#: A plain request takes an hour of audio; enabling word timestamps or speaker
#: diarization drops that to thirty minutes. Both are documented limits and both
#: are enforced here before the upload rather than after it — a rejected request
#: still costs the minutes it took to push a multi-hour file over the wire.
MAX_SECONDS_PLAIN = 60 * 60
MAX_SECONDS_ANNOTATED = 30 * 60

DEFAULT_SEGMENT_MINUTES = 20

#: The BCP-47 locales the model documents. Transcribed from the supported-language
#: table at https://ai.google.dev/gemini-api/docs/transcribe, not inferred: the
#: API accepts an unlisted code silently, so this set is the only guardrail
#: between a curator and a fluent transcript of a language the model cannot hear.
SUPPORTED_LOCALES: frozenset = frozenset({
    "af-ZA", "am-ET", "ar-EG", "as-IN", "az-AZ", "be-BY", "bg-BG", "bn-BD",
    "bn-IN", "bs-BA", "ca-ES", "cmn-Hans-CN", "cs-CZ", "da-DK", "de-DE",
    "el-GR", "en-GB", "en-IN", "en-US", "es-ES", "es-US", "et-EE", "fa-IR",
    "fi-FI", "fil-PH", "fr-FR", "gl-ES", "gu-IN", "ha-NG", "he-IL", "hi-IN",
    "hr-HR", "hu-HU", "hy-AM", "id-ID", "is-IS", "it-IT", "ja-JP", "jv-ID",
    "ka-GE", "kea-CV", "kk-KZ", "km-KH", "kn-IN", "ko-KR", "ky-KG", "ln-CD",
    "lt-LT", "lv-LV", "mk-MK", "ml-IN", "mn-MN", "mr-IN", "ms-MY", "mt-MT",
    "my-MM", "nb-NO", "ne-NP", "nl-NL", "or-IN", "pa-Guru-IN", "pa-IN",
    "pl-PL", "pt-BR", "pt-PT", "ro-RO", "ru-RU", "rup-BG", "sd-Arab-IN",
    "sk-SK", "sl-SI", "sr-RS", "sv-SE", "sw-KE", "te-IN", "tg-TJ", "th-TH",
    "tr-TR", "uk-UA", "uz-UZ", "vi-VN", "yue-Hant-HK",
})

#: How this collection's catalogued languages (``iwac_config.LANGUAGE_LABELS_BY_CODE``)
#: map onto the locales above. A ``None`` is a language the model has no locale
#: for — asserting it would be worse than auto-detection, which at least does not
#: claim to be transcribing something it was never trained to hear.
IWAC_LANGUAGE_LOCALES: Dict[str, Optional[str]] = {
    "fr": "fr-FR", "en": "en-US", "ar": "ar-EG", "ha": "ha-NG",
    "de": "de-DE", "it": "it-IT", "es": "es-ES", "sl": "sl-SI",
    "mos": None,   # Mooré
    "dyu": None,   # Dioula
    "ee": None,    # Ewé
    "kbp": None,   # Kabyè
    "ddn": None,   # Dendi
}

#: The interactive shortlist: this archive's working languages, in the order a
#: curator here is likely to want them. Any locale in ``SUPPORTED_LOCALES`` can
#: still be passed with ``--language``.
IWAC_LOCALE_CHOICES: List[Tuple[str, str]] = [
    ("auto", "Auto-detect (the model switches on code-switching)"),
    ("fr-FR", "French"),
    ("ha-NG", "Hausa"),
    ("ar-EG", "Arabic"),
    ("en-US", "English (US)"),
    ("en-GB", "English (UK)"),
]

_OFFSET_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*s\s*$")


def parse_offset_seconds(value: Optional[str]) -> Optional[float]:
    """Parse an Interactions API offset (``'0.100s'``, ``'2s'``) into seconds.

    Returns ``None`` for anything that does not parse, so a change in the wire
    format degrades a timestamp to absent rather than to a wrong number.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _OFFSET_RE.match(str(value))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def format_clock(seconds: Optional[float]) -> str:
    """Format a number of seconds as ``HH:MM:SS``."""
    if seconds is None:
        return "--:--:--"
    total = int(round(seconds))
    hours, rem = divmod(max(0, total), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def validate_locales(codes: Sequence[str]) -> List[str]:
    """Return *codes* unchanged, or raise ``ValueError`` naming the unsupported ones.

    The API validates nothing here: every code tried against it — including
    ``mos-BF`` and ``dyu-CI``, which the model does not support — was accepted
    and answered normally. An unsupported code is therefore not an error the
    operator will ever see from the server, which is why it has to be one here.
    """
    unknown = [code for code in codes if code not in SUPPORTED_LOCALES]
    if unknown:
        raise ValueError(
            f"{MODEL} does not support {', '.join(unknown)}. The API accepts "
            f"any BCP-47 string without validating it, so this would have "
            f"returned a fluent transcript of a language the model cannot "
            f"hear. Omit --language to auto-detect, or pick one of: "
            f"{', '.join(sorted(SUPPORTED_LOCALES))}"
        )
    return list(codes)


def unsupported_iwac_languages() -> List[str]:
    """Catalogued IWAC language codes this model has no locale for."""
    return [code for code, locale in IWAC_LANGUAGE_LOCALES.items() if locale is None]


class GeminiTranscribeTranscriber(TranscriberBase):
    """Transcribe with ``gemini-3.5-transcribe`` via the Interactions API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        language_codes: Optional[List[str]] = None,
        smart: bool = False,
        timestamps: bool = True,
        diarize: bool = True,
        segment_minutes: int = DEFAULT_SEGMENT_MINUTES,
        timestamps_in_text: bool = False,
        requests_per_minute: Optional[int] = None,
    ):
        """
        Args:
            api_key: Gemini API key. If None, uses GEMINI_API_KEY env var.
            language_codes: BCP-47 locales to assert, or None to auto-detect.
            smart: Use smart mode (clean prose, 1 h cap, no timestamps/speakers).
            timestamps: Request word-level timestamps (verbatim mode only).
            diarize: Request speaker labels (verbatim mode only).
            segment_minutes: Segment length for recordings over the cap.
            timestamps_in_text: Prefix each speaker turn with its clock position
                in the ``.txt`` body. Off by default: that file becomes
                ``bibo:content``, the archive's indexed full text.
            requests_per_minute: Optional RPM limit for proactive throttling.
        """
        super().__init__(requests_per_minute)

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file or environment variables")

        if smart and (timestamps or diarize):
            raise ValueError(
                "smart mode rejects timestamps and diarization "
                "(400 Unknown parameter 'timestamp_granularities'); "
                "pass --smart alone, or drop it to keep them"
            )
        if segment_minutes <= 0:
            raise ValueError("segment_minutes must be positive")

        self.client = build_gemini_client(self.api_key)
        self.language_codes = list(language_codes or [])
        self.smart = smart
        self.timestamps = timestamps and not smart
        self.diarize = diarize and not smart
        self.timestamps_in_text = timestamps_in_text
        self.generator_label = f"Google {MODEL}"

        cap_minutes = self.cap_seconds // 60
        self.segment_minutes = min(segment_minutes, cap_minutes)
        if self.segment_minutes < segment_minutes:
            console.print(
                f"[yellow]⚠[/] Segment length lowered to [bold]{self.segment_minutes}[/] minutes: "
                f"this configuration caps a request at {cap_minutes} minutes."
            )

    # ---- Request configuration --------------------------------------------

    @property
    def cap_seconds(self) -> int:
        """Seconds of audio this configuration may send in one request."""
        return MAX_SECONDS_PLAIN if self.smart else MAX_SECONDS_ANNOTATED

    def _transcription_config(self) -> Dict[str, Any]:
        """Build ``generation_config.transcription_config`` for this run.

        ``timestamp_granularities`` and ``diarization_mode`` belong *inside* the
        verbatim mode object; passing them alongside a smart mode is a 400.

        No ``temperature`` and no ``thinking_level`` are set. Both are vendor
        territory, and this model rejects the ``minimal`` level every other
        pipeline here asks for (``Allowed values are: low, high``) — a reason to
        send neither rather than to branch on the model name.
        """
        mode: Dict[str, Any] = {"type": "smart"} if self.smart else {"type": "verbatim"}
        if self.timestamps:
            mode["timestamp_granularities"] = ["word"]
        if self.diarize:
            mode["diarization_mode"] = "speaker"

        config: Dict[str, Any] = {"mode": mode}
        if self.language_codes:
            config["language_codes"] = self.language_codes
        return config

    # ---- Transcription ----------------------------------------------------

    def _segment_paths(self, audio_path: Path) -> Tuple[List[Path], Optional[float]]:
        """Return the request-sized pieces of *audio_path*, and its duration.

        A file of unknown duration is split rather than gambled with: when
        ffprobe is unavailable, ``split_audio_file`` reads the length itself and
        returns the original path untouched if it fits in one segment.
        """
        duration = probe_duration_seconds(audio_path)
        if duration is not None and duration <= self.cap_seconds:
            return [audio_path], duration

        if duration is None:
            console.print(
                f"[dim]ffprobe unavailable for '{audio_path.name}'; splitting to stay "
                f"inside the {self.cap_seconds // 60}-minute cap.[/]"
            )
        else:
            console.print(
                f"[yellow]⚠[/] '[cyan]{audio_path.name}[/]' runs {duration / 60:.0f} min, over the "
                f"{self.cap_seconds // 60}-minute cap for this configuration — splitting into "
                f"{self.segment_minutes}-minute segments."
            )
        return split_audio_file(audio_path, SCRIPT_DIR / "temp_segments", self.segment_minutes), duration

    def _create_interaction(self, audio_path: Path, max_retries: int = 3):
        """Upload one audio file and transcribe it, with backoff. ``None`` on failure."""
        mime_type = get_mime_type(audio_path)
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            uploaded = None
            try:
                self.rate_limiter.wait()
                uploaded = upload_and_wait_active(self.client, audio_path, mime_type=mime_type)
                return self.client.interactions.create(
                    model=MODEL,
                    input=[{
                        "type": "audio",
                        "uri": uploaded.uri,
                        "mime_type": uploaded.mime_type or mime_type,
                    }],
                    generation_config={"transcription_config": self._transcription_config()},
                )
            except Exception as e:
                # A daily/billing quota is not a transient 429: retrying it burns
                # the run instead of saving what completed.
                if is_quota_exhausted(e):
                    raise QuotaExhaustedError(str(e)) from e
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1) + random.uniform(0, 2)
                    console.print(f"[red]✗[/] Error transcribing [cyan]{audio_path.name}[/]: {e}")
                    console.print(f"[yellow]⏳[/] Retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    console.print(
                        f"[red]✗[/] Error transcribing [cyan]{audio_path.name}[/] "
                        f"after {max_retries} attempts: {last_error}"
                    )
            finally:
                if uploaded is not None:
                    delete_uploaded_file(self.client, uploaded)

        return None

    def transcribe_audio(self, audio_path: Path) -> Optional[Tuple[str, dict]]:
        """Transcribe one file, splitting it first if it exceeds the cap.

        Returns ``(text, payload)`` where *payload* is the JSON sidecar, or
        ``None`` when any segment failed — a transcript missing its middle is
        invisible once it is a single Omeka value, so a partial result is not
        offered as a whole one.
        """
        console.print(f"[cyan]🎤[/] Transcribing: [bold]{audio_path.name}[/]")
        segment_paths, duration = self._segment_paths(audio_path)
        total = len(segment_paths)

        try:
            blocks: List[str] = []
            words: List[dict] = []

            for idx, segment_path in enumerate(segment_paths, start=1):
                if total > 1:
                    console.print(f"[dim]  segment {idx}/{total}: {segment_path.name}[/]")

                interaction = self._create_interaction(segment_path)
                if interaction is None:
                    console.print(
                        f"[red]✗[/] Segment {idx}/{total} of [cyan]{audio_path.name}[/] failed; "
                        f"the transcript would be incomplete, so nothing is written."
                    )
                    return None

                # Fixed-length segments, so segment idx starts at exactly
                # (idx - 1) * segment_minutes. Word offsets arrive relative to
                # the segment and are shifted here, which is what makes a
                # stitched sidecar's timings absolute rather than restarting.
                offset = (idx - 1) * self.segment_minutes * 60 if total > 1 else 0
                segment_words = self._collect_words(interaction, offset, idx if total > 1 else None)
                words.extend(segment_words)

                body = self._format_text(interaction, segment_words)
                if total > 1:
                    blocks.append(f"{segment_header(idx, total, self.segment_minutes)}\n{body}")
                else:
                    blocks.append(body)

            text = "\n\n".join(block for block in blocks if block.strip())
            if not text.strip():
                console.print(f"[yellow]⚠[/] Empty transcription for [cyan]{audio_path.name}[/]")
                return None

            return text, self._build_payload(audio_path, text, words, duration, total)

        finally:
            cleanup_temp_segments(audio_path, segment_paths)

    # ---- Response parsing -------------------------------------------------

    @staticmethod
    def _collect_words(interaction, offset_seconds: float, segment_index: Optional[int]) -> List[dict]:
        """Flatten ``word_info`` annotations into plain dicts with absolute times.

        Speaker ids are namespaced per segment when the file was split: the
        model's ``spk:0`` in one 20-minute segment and ``spk:0`` in the next are
        not evidence of the same person, and merging them would invent a
        continuity the diarizer never claimed.
        """
        words: List[dict] = []
        for step in getattr(interaction, "steps", None) or []:
            for content in getattr(step, "content", None) or []:
                for annotation in getattr(content, "annotations", None) or []:
                    if getattr(annotation, "type", None) != "word_info":
                        continue
                    start = parse_offset_seconds(getattr(annotation, "start_offset", None))
                    end = parse_offset_seconds(getattr(annotation, "end_offset", None))
                    speaker = getattr(annotation, "speaker", None)
                    if speaker is not None and segment_index is not None:
                        speaker = f"seg{segment_index}-{speaker}"
                    words.append({
                        "text": getattr(annotation, "text", None),
                        "speaker": speaker,
                        "start": None if start is None else round(start + offset_seconds, 3),
                        "end": None if end is None else round(end + offset_seconds, 3),
                    })
        return words

    def _format_text(self, interaction, words: List[dict]) -> str:
        """Build the text body: speaker turns when diarized, the model's own text otherwise.

        ``output_text`` is authoritative for wording and punctuation, so it is
        used whole unless speaker labels require the text to be regrouped.
        """
        output_text = (getattr(interaction, "output_text", None) or "").strip()
        if not self.diarize or not words:
            return output_text

        turns: List[Tuple[Optional[str], Optional[float], List[str]]] = []
        for word in words:
            token = (word.get("text") or "").strip()
            if not token:
                continue
            speaker = word.get("speaker")
            if not turns or turns[-1][0] != speaker:
                turns.append((speaker, word.get("start"), [token]))
            else:
                turns[-1][2].append(token)

        if not turns:
            return output_text

        lines: List[str] = []
        for speaker, start, tokens in turns:
            label = f"[Speaker {speaker}]" if speaker else "[Speaker unknown]"
            if self.timestamps_in_text:
                label = f"[{format_clock(start)}] {label}"
            lines.append(f"{label}\n{' '.join(tokens)}")
        return "\n\n".join(lines)

    def _build_payload(
        self,
        audio_path: Path,
        text: str,
        words: List[dict],
        duration: Optional[float],
        segments: int,
    ) -> dict:
        """Assemble the JSON sidecar: the word timings and the run's settings."""
        payload: dict = {
            "file": audio_path.name,
            "model": MODEL,
            "mode": "smart" if self.smart else "verbatim",
            "language_codes": self.language_codes or None,
            "timestamps": self.timestamps,
            "diarization": self.diarize,
            "segments": segments,
            "segment_minutes": self.segment_minutes if segments > 1 else None,
            "duration_seconds": None if duration is None else round(duration, 3),
            "text": text,
        }
        if words:
            payload["words"] = words
            speakers = sorted({w["speaker"] for w in words if w.get("speaker")})
            if speakers:
                payload["speakers"] = speakers
        return payload

    # ---- Saving -----------------------------------------------------------

    def save_transcribe_output(
        self,
        text: str,
        payload: dict,
        audio_path: Path,
        output_folder: str = "Transcriptions",
    ) -> Optional[Path]:
        """Write the ``.txt`` transcript and its ``_transcription.json`` sidecar."""
        extra_fields = [
            ("Mode", "smart" if self.smart else "verbatim"),
            ("Language", ", ".join(self.language_codes) if self.language_codes else "auto-detected"),
            ("Timestamps", "ON" if self.timestamps else "OFF"),
            ("Diarization", "ON" if self.diarize else "OFF"),
        ]
        txt_file = self.save_transcription(
            text,
            audio_path,
            output_folder,
            extra_fields=extra_fields,
            segment_minutes=self.segment_minutes if payload.get("segments", 1) > 1 else None,
        )
        if txt_file is None:
            return None

        json_file = SCRIPT_DIR / output_folder / f"{audio_path.stem}_transcription.json"
        try:
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            console.print(f"[green]✓[/] JSON with timestamps saved: [cyan]{json_file}[/]")
        except OSError as e:
            console.print(f"[red]✗[/] Error saving JSON sidecar: {e}")
            return None

        return txt_file

    # ---- Main orchestration -----------------------------------------------

    def transcribe_all_audio_files(
        self,
        audio_folder: str = "Audio",
        output_folder: str = "Transcriptions",
    ):
        """Transcribe every audio/video file in *audio_folder*."""
        media_files = self.get_audio_files(audio_folder)
        if not media_files:
            self.print_no_files_warning()
            return

        files_to_process = []
        files_complete = []
        for file_path, is_video in media_files:
            if transcription_path(file_path, SCRIPT_DIR / output_folder).exists():
                files_complete.append((file_path, is_video))
            else:
                files_to_process.append((file_path, is_video))

        self.print_status_table(files_complete, [], files_to_process)

        if not files_to_process:
            console.print("[green]✓[/] All files are already transcribed!")
            return

        self.print_files_table(files_to_process)
        console.print(f"\n[bold]Summary:[/] [cyan]{len(files_to_process)}[/] file(s) to process")
        console.print()
        console.rule("[bold]Starting Transcription Process", style="cyan")
        console.print()

        def _process_item(item) -> bool:
            original_file, is_video = item[0], item[1]

            if is_video:
                audio_file = self._convert_video(original_file)
                if not audio_file:
                    console.print(f"[yellow]⚠[/] Skipping [cyan]{original_file.name}[/]: video conversion failed")
                    return False
            else:
                audio_file = original_file

            result = self.transcribe_audio(audio_file)
            if not result:
                return False

            text, payload = result
            # Name the output for the original file, not the converted audio.
            return self.save_transcribe_output(text, payload, original_file, output_folder) is not None

        successful, failed = self.run_processing_loop(files_to_process, _process_item)
        self.print_summary_table(len(files_to_process), successful, failed, output_folder)


# ---- Interactive selection ------------------------------------------------

def select_language_interactive() -> Optional[str]:
    """Display the interactive language menu. Returns a locale, or None to auto-detect."""
    console.print()
    table = Table(title="🌍 Language Selection", box=box.ROUNDED)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Language", style="green")
    table.add_column("Locale", style="dim")

    for i, (code, name) in enumerate(IWAC_LOCALE_CHOICES, 1):
        table.add_row(str(i), name, code if code != "auto" else "-")

    console.print(table)
    console.print(
        f"[dim]Not offered — {MODEL} has no locale for them: "
        f"{', '.join(unsupported_iwac_languages())} "
        f"(Mooré, Dioula, Ewé, Kabyè, Dendi).[/]"
    )

    while True:
        try:
            choice = console.input(
                f"\n[bold]Select language (1-{len(IWAC_LOCALE_CHOICES)}) or Enter for auto-detect:[/] "
            ).strip()
            if not choice:
                console.print("[green]✓[/] Language: [cyan]Auto-detect[/]")
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(IWAC_LOCALE_CHOICES):
                code, name = IWAC_LOCALE_CHOICES[idx]
                console.print(f"[green]✓[/] Language: [cyan]{name}[/]")
                return None if code == "auto" else code

            console.print(f"[red]✗[/] Please enter a number between 1 and {len(IWAC_LOCALE_CHOICES)}.")
        except ValueError:
            console.print("[red]✗[/] Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            console.print("\n[green]✓[/] Language: [cyan]Auto-detect[/]")
            return None


# ---- CLI ------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description=f"Audio transcription using Google {MODEL}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "This model takes no prompt: translation and segmentation modes live "
            "in 02_AI_transcribe_audio.py. Smart mode cannot be combined with "
            "timestamps or diarization."
        ),
    )
    parser.add_argument(
        "--language",
        default=None,
        help=(
            "BCP-47 locale to assert (e.g. fr-FR, ha-NG), or 'auto' to detect. "
            "Repeat-free: pass a comma-separated list for a multilingual recording. "
            "Default: interactive selection."
        ),
    )
    parser.add_argument(
        "--smart",
        action="store_true",
        help=(
            "Smart mode: disfluencies removed and text auto-formatted, 1 h per "
            "request. Rejects timestamps and diarization."
        ),
    )
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Disable word-level timestamps (raises accuracy; keeps the 30-min cap if diarizing)",
    )
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Disable speaker diarization",
    )
    parser.add_argument(
        "--timestamps-in-text",
        action="store_true",
        help=(
            "Prefix each speaker turn in the .txt with its clock position. Off by "
            "default: that file becomes bibo:content, the archive's indexed full text."
        ),
    )
    parser.add_argument(
        "--segment-minutes",
        type=int,
        default=DEFAULT_SEGMENT_MINUTES,
        help=f"Segment length for recordings over the cap (default: {DEFAULT_SEGMENT_MINUTES})",
    )
    parser.add_argument(
        "--audio-folder",
        default="Audio",
        help="Folder containing audio files (default: Audio)",
    )
    parser.add_argument(
        "--output-folder",
        default="Transcriptions",
        help="Folder for output transcriptions (default: Transcriptions)",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=None,
        help="Rate limit: maximum requests per minute (default: no limit)",
    )
    return parser.parse_args(argv)


def resolve_language_codes(raw: Optional[str]) -> List[str]:
    """Turn the ``--language`` value into a validated locale list.

    ``None`` means the flag was absent (the caller prompts); ``'auto'`` and the
    empty list both mean auto-detection.
    """
    if raw is None or raw.strip().lower() == "auto":
        return []
    codes = [part.strip() for part in raw.split(",") if part.strip()]
    return validate_locales(codes)


def main():
    args = parse_args()

    console.print(Panel(
        f"Transcribe audio and video files using Google {MODEL}",
        title="🎤 Audio Transcription using Gemini 3.5 Transcribe",
        border_style="cyan",
    ))

    try:
        if args.language is not None:
            language_codes = resolve_language_codes(args.language)
        else:
            chosen = select_language_interactive()
            language_codes = [chosen] if chosen else []

        detect_ffmpeg()

        transcriber = GeminiTranscribeTranscriber(
            language_codes=language_codes,
            smart=args.smart,
            timestamps=not args.no_timestamps,
            diarize=not args.no_diarize,
            segment_minutes=args.segment_minutes,
            timestamps_in_text=args.timestamps_in_text,
            requests_per_minute=args.rpm,
        )

        console.print()
        console.print(make_config_table([
            ("Model", MODEL),
            ("Mode", "smart" if transcriber.smart else "verbatim"),
            ("Language", ", ".join(language_codes) if language_codes else "Auto-detect"),
            ("Timestamps", "[green]ON[/]" if transcriber.timestamps else "[dim]OFF[/]"),
            ("Diarization", "[green]ON[/]" if transcriber.diarize else "[dim]OFF[/]"),
            ("Request Cap", f"{transcriber.cap_seconds // 60} minutes"),
            ("Segment Length", f"{transcriber.segment_minutes} minutes"),
            ("Audio Folder", args.audio_folder),
            ("Output Folder", args.output_folder),
            ("Rate Limit", f"{args.rpm} RPM" if args.rpm else "None"),
        ]))
        console.print()

        transcriber.transcribe_all_audio_files(
            audio_folder=args.audio_folder,
            output_folder=args.output_folder,
        )

    except QuotaExhaustedError as e:
        console.print(f"\n[red]✗ Quota exhausted:[/] {e}")
        console.print("[dim]Completed transcriptions were saved; rerun to continue.[/]")

    except ValueError as e:
        console.print(f"\n[red]✗ Configuration Error:[/] {e}")
        if "GEMINI_API_KEY" in str(e):
            console.print("\n[bold]To use this script, set your Gemini API key:[/]")
            console.print("  1. Get a key from: [link=https://aistudio.google.com/apikey]https://aistudio.google.com/apikey[/link]")
            console.print("  2. Add to your .env file: GEMINI_API_KEY=your-api-key-here")
            console.print("  3. Run this script again")

    except Exception as e:
        console.print(f"\n[red]✗ Unexpected error:[/] {e}")


if __name__ == "__main__":
    main()
