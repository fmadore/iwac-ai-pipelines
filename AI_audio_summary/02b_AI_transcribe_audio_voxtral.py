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
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

from dotenv import load_dotenv
from mistralai import Mistral

# Add repo root to path for shared imports
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent))
from common.rate_limiter import RateLimiter, QuotaExhaustedError
from common.ffmpeg_utils import (
    AUDIO_FORMATS, VIDEO_FORMATS,
    get_ffmpeg_paths, setup_pydub,
    convert_video_to_audio, cleanup_files,
    has_unsafe_path_chars,
)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Initialize rich console
console = Console()

# Load environment variables from .env file
load_dotenv()

# Script directory for relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()

MODEL = "voxtral-mini-2602"

# Languages supported by Voxtral (subset relevant to IWAC)
LANGUAGES = [
    ("auto", "Auto-detect"),
    ("en", "English"),
    ("fr", "French"),
    ("de", "German"),
    ("ha", "Hausa"),
    ("sw", "Swahili"),
]


def _is_mistral_quota_exhausted(error: Exception) -> bool:
    """Check if a Mistral API error indicates quota exhaustion (not a transient rate limit)."""
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    if status != 429:
        return False
    message = str(error).lower()
    return any(ind in message for ind in ["quota", "exceeded", "billing", "per_day"])


class VoxtralTranscriber:
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
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in .env file or environment variables")

        self.client = Mistral(api_key=self.api_key)
        self.language = language  # None = auto-detect
        self.diarize = diarize
        self.rate_limiter = RateLimiter(requests_per_minute, logger=logging.getLogger(__name__))
        self._temp_audio_files: List[Path] = []

    # ---- Video conversion ------------------------------------------------

    def _convert_video(self, video_path: Path) -> Optional[Path]:
        """Convert a video file to audio, tracking the result for cleanup."""
        temp_dir = SCRIPT_DIR / "temp_converted_audio"
        console.print(f"[cyan]🎬[/] Converting video to audio: [bold]{video_path.name}[/]")
        result = convert_video_to_audio(video_path, temp_dir)
        if result:
            console.print(f"[green]✓[/] Video converted: [cyan]{result.name}[/]")
            self._temp_audio_files.append(result)
        else:
            console.print(f"[red]✗[/] Cannot convert video '[cyan]{video_path.name}[/]'")
        return result

    def cleanup_converted_audio(self):
        """Clean up temporary converted audio files."""
        if self._temp_audio_files:
            cleanup_files(self._temp_audio_files, remove_parents=True)
            for f in self._temp_audio_files:
                if not f.exists():
                    console.print(f"[dim]🧹 Cleaned up converted audio: {f.name}[/]")
            self._temp_audio_files.clear()

    # ---- File discovery ---------------------------------------------------

    def get_audio_files(self, audio_folder: str = "Audio") -> List[Tuple[Path, bool]]:
        """Get all supported audio and video files from the specified folder.

        Returns:
            List of (file_path, is_video) tuples sorted by name.
        """
        audio_path = SCRIPT_DIR / audio_folder
        if not audio_path.exists():
            console.print(f"[red]✗[/] Audio folder '[cyan]{audio_path}[/]' not found!")
            return []

        media_files = []
        skipped = []
        for file_path in audio_path.iterdir():
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in AUDIO_FORMATS or ext in VIDEO_FORMATS:
                    if has_unsafe_path_chars(file_path.name):
                        skipped.append(file_path)
                        continue
                    is_video = ext in VIDEO_FORMATS
                    media_files.append((file_path, is_video))

        if skipped:
            console.print(f"\n[yellow]⚠[/] Skipped [bold]{len(skipped)}[/] file(s) with characters that break ffmpeg ([cyan]' \" < > | * ?[/]):")
            for f in skipped:
                console.print(f"  [dim]- {f.name}[/]")
            console.print("[yellow]  → Please rename these files to remove the problematic characters and try again.[/]\n")

        return sorted(media_files, key=lambda x: x[0])

    # ---- Transcription ----------------------------------------------------

    def transcribe_audio(self, audio_path: Path, max_retries: int = 3) -> Optional[Tuple[str, Any]]:
        """Transcribe a single audio file via Voxtral API with retry.

        Args:
            audio_path: Path to the audio file.
            max_retries: Maximum number of retry attempts.

        Returns:
            Tuple of (formatted_text, raw_response), or None on failure.
        """
        console.print(f"[cyan]🎤[/] Transcribing: [bold]{audio_path.name}[/]")

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
                if _is_mistral_quota_exhausted(e):
                    raise QuotaExhaustedError(str(e))
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

    def save_transcription(
        self,
        transcription: str,
        response: Any,
        audio_path: Path,
        output_folder: str = "Transcriptions",
    ) -> Optional[Path]:
        """Save transcription as both a text file and a JSON file with timestamps."""
        output_path = SCRIPT_DIR / output_folder
        output_path.mkdir(exist_ok=True)

        stem = f"{audio_path.stem}_transcription"
        txt_file = output_path / f"{stem}.txt"
        json_file = output_path / f"{stem}.json"

        try:
            lang_display = self.language if self.language else "auto-detected"
            diarize_display = "ON" if self.diarize else "OFF"

            # Save text file with header
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(f"Transcription of: {audio_path.name}\n")
                f.write(f"Generated using: Mistral {MODEL}\n")
                f.write(f"Language: {lang_display}\n")
                f.write(f"Diarization: {diarize_display}\n")
                f.write("=" * 50 + "\n\n")
                f.write(transcription)

            console.print(f"[green]✓[/] Transcription saved: [cyan]{txt_file}[/]")

            # Save JSON file with timestamps and structured data
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
            console.print("[yellow]⚠[/] No supported audio or video files found.")
            console.print(f"[dim]Supported audio: {', '.join(AUDIO_FORMATS.keys())}[/]")
            console.print(f"[dim]Supported video: {', '.join(VIDEO_FORMATS.keys())}[/]")
            return

        # Separate already-transcribed files from new ones
        files_to_process = []
        files_complete = []

        for file_path, is_video in media_files:
            output_file = SCRIPT_DIR / output_folder / f"{file_path.stem}_transcription.txt"
            if output_file.exists():
                files_complete.append((file_path, is_video))
            else:
                files_to_process.append((file_path, is_video))

        # Status table
        if files_complete:
            status_table = Table(title="📋 Transcription Status", box=box.ROUNDED)
            status_table.add_column("Status", style="dim")
            status_table.add_column("Count", justify="right")
            status_table.add_column("Details", style="dim")
            status_table.add_row("[green]✓ Complete[/]", str(len(files_complete)), "Already transcribed")
            if files_to_process:
                status_table.add_row("[cyan]○ New[/]", str(len(files_to_process)), "Not yet transcribed")
            console.print(status_table)
            console.print()

        if not files_to_process:
            console.print("[green]✓[/] All files are already transcribed!")
            return

        # Display files table
        files_table = Table(title="📁 Files to Transcribe", box=box.ROUNDED)
        files_table.add_column("Type", style="cyan")
        files_table.add_column("Filename", style="green")

        for fp, iv in files_to_process:
            ftype = "🎬 Video" if iv else "🎵 Audio"
            files_table.add_row(ftype, fp.name)

        console.print(files_table)
        console.print(f"\n[bold]Summary:[/] [cyan]{len(files_to_process)}[/] file(s) to process")
        console.print()
        console.rule("[bold]Starting Transcription Process", style="cyan")
        console.print()

        successful = 0
        failed = 0

        try:
            for fp, iv in files_to_process:
                try:
                    console.rule(f"[dim]{fp.name}[/]", style="dim")

                    # Convert video to audio if needed
                    if iv:
                        audio_file = self._convert_video(fp)
                        if not audio_file:
                            console.print(f"[yellow]⚠[/] Skipping [cyan]{fp.name}[/]: video conversion failed")
                            failed += 1
                            continue
                    else:
                        audio_file = fp

                    result = self.transcribe_audio(audio_file)

                    if result:
                        transcription, response = result
                        # Use original file name for output (not converted audio name)
                        out = self.save_transcription(transcription, response, fp, output_folder)
                        if out:
                            successful += 1
                        else:
                            failed += 1
                    else:
                        failed += 1

                except QuotaExhaustedError:
                    console.print(f"\n[red bold]API quota exhausted — stopping all processing.[/]")
                    console.print("[red]Partial results (if any) have been saved.[/]")
                    console.print("[red]Wait for your quota to reset or upgrade your plan.[/]")
                    break

                except Exception as e:
                    console.print(f"[red]✗[/] Unexpected error processing [cyan]{fp.name}[/]: {e}")
                    failed += 1

                console.print()

        finally:
            self.cleanup_converted_audio()

        # Summary
        console.print()
        console.rule("[bold]Transcription Summary", style="cyan")

        summary = Table(title="📊 Results", box=box.ROUNDED)
        summary.add_column("Metric", style="dim")
        summary.add_column("Value", style="green")
        summary.add_row("Total files processed", str(len(files_to_process)))
        summary.add_row("Successful transcriptions", f"[green]{successful}[/]")
        summary.add_row("Failed transcriptions", f"[red]{failed}[/]" if failed > 0 else "0")
        summary.add_row("Output folder", output_folder)
        console.print(summary)

        if successful > 0:
            console.print(f"\n[green]✓[/] Transcriptions saved in the '[cyan]{output_folder}[/]' folder.")


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
        setup_pydub()
        paths = get_ffmpeg_paths()
        if paths:
            console.print(f"[green]✓[/] Detected ffmpeg: [cyan]{paths.ffmpeg}[/]")
            console.print(f"[green]✓[/] Detected ffprobe: [cyan]{paths.ffprobe}[/]")

        # Display configuration
        console.print()
        config = Table(title="⚙️ Configuration", box=box.ROUNDED)
        config.add_column("Setting", style="dim")
        config.add_column("Value", style="green")
        config.add_row("Model", MODEL)
        config.add_row("Language", language if language else "Auto-detect")
        config.add_row("Diarization", "[green]ON[/]" if diarize else "[dim]OFF[/]")
        config.add_row("Audio Folder", args.audio_folder)
        config.add_row("Output Folder", args.output_folder)
        config.add_row("Rate Limit", f"{args.rpm} RPM" if args.rpm else "None")
        console.print(config)
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
