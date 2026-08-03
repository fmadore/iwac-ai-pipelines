#!/usr/bin/env python3
"""
Audio Transcription Script using Mistral Voxtral Mini Transcribe 2

Transcribes audio and video files from the Audio folder using Voxtral's
dedicated transcription model with optional speaker diarization.
Video files are automatically converted to audio before transcription.
Voxtral supports up to 3 hours of audio per request.
"""

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Optional, Tuple

from dotenv import load_dotenv
from mistralai.client import Mistral

# Add repo root to path for shared imports
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.rate_limiter import QuotaExhaustedError, is_mistral_quota_exhausted
from common.ffmpeg_utils import get_ffmpeg_paths
from common.log_redaction import install_credential_redaction

# Credentials ride in Omeka query strings and provider headers; keep them
# out of anything urllib3 or an SDK decides to log.
install_credential_redaction()

from rich.panel import Panel
from rich.table import Table
from rich import box

from transcriber_base import SCRIPT_DIR, TranscriberBase, console, detect_ffmpeg, make_config_table
from segments import transcription_path

# Load environment variables from .env file
load_dotenv()

MODEL = "voxtral-mini-2602"

# Voxtral caps a single transcription request at 3 hours of audio.
VOXTRAL_MAX_SECONDS = 3 * 3600

# Languages supported by Voxtral (subset relevant to IWAC)
LANGUAGES = [
    ("auto", "Auto-detect"),
    ("en", "English"),
    ("fr", "French"),
    ("de", "German"),
    ("ha", "Hausa"),
    ("sw", "Swahili"),
]


def probe_duration_seconds(audio_path: Path) -> Optional[float]:
    """Return the media duration in seconds via ffprobe, or ``None``.

    Silently returns ``None`` when ffprobe is unavailable or fails — the
    duration check is a best-effort warning, not a gate.
    """
    paths = get_ffmpeg_paths()
    if not paths:
        return None
    try:
        result = subprocess.run(
            [
                paths.ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return None
        return float(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


class VoxtralTranscriber(TranscriberBase):
    def __init__(
        self,
        api_key: Optional[str] = None,
        language: Optional[str] = None,
        diarize: bool = True,
        requests_per_minute: Optional[int] = None,
    ):
        """
        Initialize the Voxtral Transcriber.

        Args:
            api_key: Mistral API key. If None, uses MISTRAL_API_KEY env var.
            language: Language code (e.g. 'en', 'fr') or None for auto-detect.
            diarize: Enable speaker diarization.
            requests_per_minute: Optional RPM limit for proactive throttling.
        """
        super().__init__(requests_per_minute)

        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in .env file or environment variables")

        self.client = Mistral(api_key=self.api_key)
        self.language = language  # None = auto-detect
        self.diarize = diarize
        self.generator_label = f"Mistral {MODEL}"

    # ---- Transcription ----------------------------------------------------

    def _warn_if_too_long(self, audio_path: Path) -> None:
        """Warn when a file exceeds Voxtral's 3-hour per-request cap."""
        duration = probe_duration_seconds(audio_path)
        if duration and duration > VOXTRAL_MAX_SECONDS:
            console.print(
                f"[yellow]⚠[/] '[cyan]{audio_path.name}[/]' is {duration / 3600:.1f} h long — "
                f"Voxtral supports at most 3 h of audio per request; "
                f"the transcription may fail or be truncated."
            )

    def transcribe_audio(self, audio_path: Path, max_retries: int = 3) -> Optional[Tuple[str, Any]]:
        """Transcribe a single audio file via Voxtral API with retry.

        Args:
            audio_path: Path to the audio file.
            max_retries: Maximum number of retry attempts.

        Returns:
            Tuple of (formatted_text, raw_response), or None on failure.
        """
        console.print(f"[cyan]🎤[/] Transcribing: [bold]{audio_path.name}[/]")
        self._warn_if_too_long(audio_path)

        last_error = None
        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait()

                with open(audio_path, "rb") as f:
                    kwargs = {
                        "model": MODEL,
                        "file": {"content": f, "file_name": audio_path.name},
                    }

                    # Language and timestamp_granularities are mutually exclusive
                    if self.language:
                        kwargs["language"] = self.language
                    else:
                        kwargs["timestamp_granularities"] = ["segment"]

                    if self.diarize:
                        kwargs["diarize"] = True

                    response = self.client.audio.transcriptions.complete(**kwargs)

                if response is None or not hasattr(response, "text"):
                    console.print(f"[yellow]⚠[/] No transcription returned for [cyan]{audio_path.name}[/]")
                    return None

                # Format with speaker labels when diarization is active
                if self.diarize and hasattr(response, "segments") and response.segments:
                    text = self._format_diarized_text(response)
                else:
                    text = response.text.strip()

                return text, response

            except Exception as e:
                if is_mistral_quota_exhausted(e):
                    raise QuotaExhaustedError(str(e)) from e
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1) + random.uniform(0, 2)
                    console.print(f"[red]✗[/] Error transcribing [cyan]{audio_path.name}[/]: {e}")
                    console.print(f"[yellow]⏳[/] Retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    console.print(f"[red]✗[/] Error transcribing [cyan]{audio_path.name}[/] after {max_retries} attempts: {last_error}")

        return None

    def _format_diarized_text(self, response) -> str:
        """Format transcription text with speaker labels from diarization segments."""
        if not response.segments:
            return response.text.strip()

        parts = []
        current_speaker = None
        current_texts = []

        for segment in response.segments:
            speaker = getattr(segment, "speaker_id", None)
            text = getattr(segment, "text", "").strip()
            if not text:
                continue

            if speaker is not None and speaker != current_speaker:
                # Flush previous speaker's accumulated text
                if current_texts:
                    label = f"[Speaker {current_speaker}]\n" if current_speaker is not None else ""
                    parts.append(label + " ".join(current_texts))
                current_speaker = speaker
                current_texts = [text]
            else:
                current_texts.append(text)

        # Flush remaining text
        if current_texts:
            label = f"[Speaker {current_speaker}]\n" if current_speaker is not None else ""
            parts.append(label + " ".join(current_texts))

        return "\n\n".join(parts) if parts else response.text.strip()

    # ---- Serialization ----------------------------------------------------

    @staticmethod
    def _serialize_segment(seg) -> dict:
        """Serialize a segment object to a plain dict."""
        if isinstance(seg, dict):
            return seg
        d = {}
        for attr in ("start", "end", "text", "speaker_id"):
            val = getattr(seg, attr, None)
            if val is not None:
                d[attr] = val
        return d

    @staticmethod
    def _serialize_word(w) -> dict:
        """Serialize a word object to a plain dict."""
        if isinstance(w, dict):
            return w
        d = {}
        for attr in ("word", "start", "end", "confidence", "speaker"):
            val = getattr(w, attr, None)
            if val is not None:
                d[attr] = val
        return d

    def _serialize_response(self, response, audio_filename: str) -> dict:
        """Serialize the full API response to a JSON-serializable dict."""
        result = {
            "file": audio_filename,
            "model": getattr(response, "model", MODEL),
            "text": response.text,
        }
        if getattr(response, "language", None):
            result["language"] = response.language
        if getattr(response, "segments", None):
            result["segments"] = [self._serialize_segment(s) for s in response.segments]
        if getattr(response, "words", None):
            result["words"] = [self._serialize_word(w) for w in response.words]
        if getattr(response, "usage", None):
            result["usage"] = {
                "prompt_audio_seconds": getattr(response.usage, "prompt_audio_seconds", None),
                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                "total_tokens": getattr(response.usage, "total_tokens", None),
            }
        return result

    # ---- Saving -----------------------------------------------------------

    def save_voxtral_transcription(
        self,
        transcription: str,
        response: Any,
        audio_path: Path,
        output_folder: str = "Transcriptions",
    ) -> Optional[Path]:
        """Save transcription as both a text file and a JSON file with timestamps."""
        lang_display = self.language if self.language else "auto-detected"
        diarize_display = "ON" if self.diarize else "OFF"

        # Text file with header — format owned by segments.write_transcription
        txt_file = self.save_transcription(
            transcription,
            audio_path,
            output_folder,
            extra_fields=[("Language", lang_display), ("Diarization", diarize_display)],
        )
        if txt_file is None:
            return None

        # Save JSON file with timestamps and structured data
        json_file = SCRIPT_DIR / output_folder / f"{audio_path.stem}_transcription.json"
        try:
            json_data = self._serialize_response(response, audio_path.name)
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            console.print(f"[green]✓[/] JSON with timestamps saved: [cyan]{json_file}[/]")
            return txt_file
        except Exception as e:
            console.print(f"[red]✗[/] Error saving transcription: {e}")
            return None

    # ---- Main orchestration -----------------------------------------------

    def transcribe_all_audio_files(
        self,
        audio_folder: str = "Audio",
        output_folder: str = "Transcriptions",
    ):
        """Transcribe all audio/video files in the specified folder."""
        media_files = self.get_audio_files(audio_folder)
        if not media_files:
            self.print_no_files_warning()
            return

        # Separate already-transcribed files from new ones
        files_to_process = []
        files_complete = []

        for file_path, is_video in media_files:
            if transcription_path(file_path, SCRIPT_DIR / output_folder).exists():
                files_complete.append((file_path, is_video))
            else:
                files_to_process.append((file_path, is_video))

        # Status table
        self.print_status_table(files_complete, [], files_to_process)

        if not files_to_process:
            console.print("[green]✓[/] All files are already transcribed!")
            return

        # Display files table
        self.print_files_table(files_to_process)
        console.print(f"\n[bold]Summary:[/] [cyan]{len(files_to_process)}[/] file(s) to process")
        console.print()
        console.rule("[bold]Starting Transcription Process", style="cyan")
        console.print()

        def _process_item(item) -> bool:
            original_file, is_video = item[0], item[1]

            # Convert video to audio if needed
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

            transcription, response = result
            # Use original file name for output (not converted audio name)
            return self.save_voxtral_transcription(transcription, response, original_file, output_folder) is not None

        successful, failed = self.run_processing_loop(files_to_process, _process_item)

        self.print_summary_table(len(files_to_process), successful, failed, output_folder)


# ---- Interactive selection ------------------------------------------------

def select_language_interactive() -> Optional[str]:
    """Display interactive language selection menu."""
    console.print()
    table = Table(title="🌍 Language Selection", box=box.ROUNDED)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Language", style="green")
    table.add_column("Code", style="dim")

    for i, (code, name) in enumerate(LANGUAGES, 1):
        table.add_row(str(i), name, code if code != "auto" else "-")

    console.print(table)

    while True:
        try:
            choice = console.input(f"\n[bold]Select language (1-{len(LANGUAGES)}) or Enter for auto-detect:[/] ").strip()
            if not choice:
                console.print("[green]✓[/] Language: [cyan]Auto-detect[/]")
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(LANGUAGES):
                code, name = LANGUAGES[idx]
                console.print(f"[green]✓[/] Language: [cyan]{name}[/]")
                return None if code == "auto" else code

            console.print(f"[red]✗[/] Please enter a number between 1 and {len(LANGUAGES)}.")
        except ValueError:
            console.print("[red]✗[/] Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            console.print("\n[green]✓[/] Language: [cyan]Auto-detect[/]")
            return None


# ---- CLI ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Audio Transcription using Mistral Voxtral Mini Transcribe 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--language",
        choices=["auto", "en", "fr", "de", "ha", "sw"],
        default=None,
        help="Audio language (default: interactive selection). 'auto' = auto-detect.",
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
        "--no-diarize",
        action="store_true",
        help="Disable speaker diarization",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=None,
        help="Rate limit: maximum requests per minute (default: no limit)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Welcome banner
    console.print(Panel(
        "Transcribe audio and video files using Mistral Voxtral Mini Transcribe 2",
        title="🎤 Audio Transcription using Voxtral",
        border_style="cyan",
    ))

    try:
        # Language selection (CLI or interactive)
        if args.language is not None:
            language = None if args.language == "auto" else args.language
            lang_name = dict(LANGUAGES).get(args.language, args.language)
            console.print(f"\n[green]✓[/] Language: [cyan]{lang_name}[/]")
        else:
            language = select_language_interactive()

        diarize = not args.no_diarize

        # Check ffmpeg availability for video conversion
        detect_ffmpeg()

        # Display configuration
        console.print()
        console.print(make_config_table([
            ("Model", MODEL),
            ("Language", language if language else "Auto-detect"),
            ("Diarization", "[green]ON[/]" if diarize else "[dim]OFF[/]"),
            ("Audio Folder", args.audio_folder),
            ("Output Folder", args.output_folder),
            ("Rate Limit", f"{args.rpm} RPM" if args.rpm else "None"),
        ]))
        console.print()

        # Initialize transcriber and run
        transcriber = VoxtralTranscriber(
            language=language,
            diarize=diarize,
            requests_per_minute=args.rpm,
        )

        transcriber.transcribe_all_audio_files(
            audio_folder=args.audio_folder,
            output_folder=args.output_folder,
        )

    except ValueError as e:
        console.print(f"\n[red]✗ Configuration Error:[/] {e}")
        console.print("\n[bold]To use this script, set your Mistral API key:[/]")
        console.print("  1. Get a key from: [link=https://console.mistral.ai/]https://console.mistral.ai/[/link]")
        console.print("  2. Add to your .env file: MISTRAL_API_KEY=your-api-key-here")
        console.print("  3. Run this script again")

    except Exception as e:
        console.print(f"\n[red]✗ Unexpected error:[/] {e}")


if __name__ == "__main__":
    main()
