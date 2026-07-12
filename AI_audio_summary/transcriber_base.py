"""
Shared base class for the audio transcription scripts.

Hosts the file discovery, video conversion, transcription saving, rich
status/files/summary tables, and the per-file orchestration loop that were
previously duplicated verbatim between ``02_AI_transcribe_audio.py``
(Gemini) and ``02b_AI_transcribe_audio_voxtral.py`` (Voxtral).

Each provider script subclasses :class:`TranscriberBase`, sets
``generator_label``, and supplies its own ``transcribe_audio()`` plus
provider-specific configuration.
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

# Add repo root to path for shared imports
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent))
from common.ffmpeg_utils import (
    AUDIO_FORMATS, VIDEO_FORMATS,
    get_ffmpeg_paths, setup_pydub,
    convert_video_to_audio, cleanup_files,
    has_unsafe_path_chars,
)
from common.rate_limiter import RateLimiter, QuotaExhaustedError

from rich import box
from rich.console import Console
from rich.table import Table

from segments import write_transcription

# Single console shared by the transcription scripts
console = Console()

# Directory containing the transcription scripts (Audio/, Transcriptions/, ...)
SCRIPT_DIR = Path(__file__).parent.resolve()


def detect_ffmpeg() -> bool:
    """Check pydub + ffmpeg availability once and report detected paths.

    Returns ``True`` when pydub is importable and ffmpeg/ffprobe were found
    (i.e. audio splitting is possible).
    """
    ready = setup_pydub()
    paths = get_ffmpeg_paths()
    if paths:
        console.print(f"[green]✓[/] Detected ffmpeg: [cyan]{paths.ffmpeg}[/]")
        console.print(f"[green]✓[/] Detected ffprobe: [cyan]{paths.ffprobe}[/]")
    return ready


def make_config_table(rows: Sequence[Tuple[str, str]]) -> Table:
    """Build the standard ⚙️ Configuration table from (setting, value) rows."""
    table = Table(title="⚙️ Configuration", box=box.ROUNDED)
    table.add_column("Setting", style="dim")
    table.add_column("Value", style="green")
    for setting, value in rows:
        table.add_row(setting, value)
    return table


class TranscriberBase:
    """Provider-agnostic plumbing shared by the audio transcription scripts."""

    def __init__(self, requests_per_minute: Optional[int] = None):
        # Rate limiter for proactive throttling (None = no throttling)
        self.rate_limiter = RateLimiter(requests_per_minute, logger=logging.getLogger(__name__))

        # Track temporary converted audio files for cleanup
        self._temp_audio_files: List[Path] = []

        # Written into the "Generated using:" transcription header line —
        # subclasses set e.g. "Google gemini-pro-latest" or "Mistral voxtral-mini-2602".
        self.generator_label = "unknown model"

    # ---- Provider hook -----------------------------------------------------

    def transcribe_audio(self, audio_file_path: Path, *args, **kwargs):
        """Transcribe a single audio file. Provider scripts must implement this."""
        raise NotImplementedError("Provider scripts must implement transcribe_audio()")

    # ---- File discovery ------------------------------------------------------

    def get_audio_files(self, audio_folder: str = "Audio") -> List[Tuple[Path, bool]]:
        """Get all supported audio and video files from the specified folder.

        Video files will be converted to audio during transcription. Files
        whose names contain characters that break ffmpeg are skipped with a
        warning.

        Args:
            audio_folder: Path to the audio folder (relative to script directory).

        Returns:
            List of ``(file_path, is_video)`` tuples sorted by name.
        """
        audio_path = SCRIPT_DIR / audio_folder
        if not audio_path.exists():
            console.print(f"[red]✗[/] Audio folder '[cyan]{audio_path}[/]' not found!")
            return []

        media_files = []
        skipped = []
        for file_path in audio_path.iterdir():
            if file_path.is_file():
                extension = file_path.suffix.lower()
                if extension in AUDIO_FORMATS or extension in VIDEO_FORMATS:
                    if has_unsafe_path_chars(file_path.name):
                        skipped.append(file_path)
                        continue
                    is_video = extension in VIDEO_FORMATS
                    media_files.append((file_path, is_video))

        if skipped:
            console.print(
                f"\n[yellow]⚠[/] Skipped [bold]{len(skipped)}[/] file(s) with characters "
                f"that break ffmpeg ([cyan]' \" < > | * ?[/]):"
            )
            for f in skipped:
                console.print(f"  [dim]- {f.name}[/]")
            console.print(
                "[yellow]  → Please rename these files to remove the problematic characters and try again.[/]\n"
            )

        return sorted(media_files, key=lambda x: x[0])

    # ---- Video conversion ----------------------------------------------------

    def _convert_video(self, video_file_path: Path, output_format: str = "mp3") -> Optional[Path]:
        """Convert a video file to audio, tracking the result for cleanup.

        Delegates to ``convert_video_to_audio`` from ``common.ffmpeg_utils``.
        """
        temp_dir = SCRIPT_DIR / "temp_converted_audio"
        console.print(f"[cyan]🎬[/] Converting video to audio: [bold]{video_file_path.name}[/]")
        result = convert_video_to_audio(video_file_path, temp_dir, output_format)
        if result:
            console.print(f"[green]✓[/] Video converted: [cyan]{result.name}[/]")
            self._temp_audio_files.append(result)
        else:
            console.print(f"[red]✗[/] Cannot convert video '[cyan]{video_file_path.name}[/]'")
        return result

    def cleanup_converted_audio(self) -> None:
        """Clean up temporary converted audio files."""
        if self._temp_audio_files:
            cleanup_files(self._temp_audio_files, remove_parents=True)
            for f in self._temp_audio_files:
                if not f.exists():
                    console.print(f"[dim]🧹 Cleaned up converted audio: {f.name}[/]")
            self._temp_audio_files.clear()

    # ---- Saving ----------------------------------------------------------------

    def save_transcription(
        self,
        transcription: str,
        audio_file_path: Path,
        output_folder: str = "Transcriptions",
        *,
        extra_fields: Optional[Sequence[Tuple[str, str]]] = None,
        segment_minutes: Optional[int] = None,
    ) -> Optional[Path]:
        """Save a transcription text file with the standard metadata header.

        Delegates to :func:`segments.write_transcription`, which owns the
        on-disk format.
        """
        return write_transcription(
            transcription,
            audio_file_path,
            SCRIPT_DIR / output_folder,
            generator=self.generator_label,
            extra_fields=extra_fields,
            segment_minutes=segment_minutes,
        )

    # ---- Rich tables -----------------------------------------------------------

    def print_no_files_warning(self) -> None:
        """Report that no supported media files were found."""
        console.print("[yellow]⚠[/] No supported audio or video files found in the Audio folder.")
        console.print(f"[dim]Supported audio formats: {', '.join(AUDIO_FORMATS.keys())}[/]")
        console.print(f"[dim]Supported video formats: {', '.join(VIDEO_FORMATS.keys())}[/]")

    def print_status_table(self, files_complete, files_to_retry, files_to_process) -> None:
        """Show the 📋 Transcription Status summary table.

        ``files_to_retry`` entries must have the file path at index 0 and the
        failed-segment list at index 2 (pass ``[]`` for providers without
        segment retry support).
        """
        if not (files_complete or files_to_retry):
            return

        status_table = Table(title="📋 Transcription Status", box=box.ROUNDED)
        status_table.add_column("Status", style="dim")
        status_table.add_column("Count", justify="right")
        status_table.add_column("Details", style="dim")

        if files_complete:
            status_table.add_row("[green]✓ Complete[/]", str(len(files_complete)), "Already transcribed")
        if files_to_retry:
            retry_details = ", ".join(f"{item[0].name} ({len(item[2])} failed)" for item in files_to_retry)
            status_table.add_row(
                "[yellow]⚠ Has failures[/]",
                str(len(files_to_retry)),
                retry_details[:50] + "..." if len(retry_details) > 50 else retry_details,
            )
        if files_to_process:
            status_table.add_row("[cyan]○ New[/]", str(len(files_to_process)), "Not yet transcribed")

        console.print(status_table)
        console.print()

    def print_files_table(self, items, with_status: bool = False) -> None:
        """Show the 📁 Files to Transcribe table.

        Each item is ``(file_path, is_video, ...)``; with *with_status* a
        third element ``(failed_segments, ...)`` marks a retry entry.
        """
        files_table = Table(title="📁 Files to Transcribe", box=box.ROUNDED)
        files_table.add_column("Type", style="cyan")
        files_table.add_column("Filename", style="green")
        if with_status:
            files_table.add_column("Status", style="dim")

        for item in items:
            file_path, is_video = item[0], item[1]
            file_type = "🎬 Video" if is_video else "🎵 Audio"
            if with_status:
                retry_info = item[2] if len(item) > 2 else None
                status = f"Retry {len(retry_info[0])} segment(s)" if retry_info else "New"
                files_table.add_row(file_type, file_path.name, status)
            else:
                files_table.add_row(file_type, file_path.name)

        console.print(files_table)

    def print_summary_table(self, total_files: int, successful: int, failed: int, output_folder: str) -> None:
        """Show the final 📊 Results summary table."""
        console.print()
        console.rule("[bold]Transcription Summary", style="cyan")

        summary_table = Table(title="📊 Results", box=box.ROUNDED)
        summary_table.add_column("Metric", style="dim")
        summary_table.add_column("Value", style="green")
        summary_table.add_row("Total files processed", str(total_files))
        summary_table.add_row("Successful transcriptions", f"[green]{successful}[/]")
        summary_table.add_row("Failed transcriptions", f"[red]{failed}[/]" if failed > 0 else "0")
        summary_table.add_row("Output folder", output_folder)
        console.print(summary_table)

        if successful > 0:
            console.print(f"\n[green]✓[/] Transcriptions saved in the '[cyan]{output_folder}[/]' folder.")

    # ---- Orchestration loop -------------------------------------------------------

    def run_processing_loop(self, items, handler: Callable) -> Tuple[int, int]:
        """Process each item with *handler*, tracking success/failure counts.

        *handler* receives one item (``(file_path, is_video, ...)``) and
        returns ``True`` on success. ``QuotaExhaustedError`` stops the whole
        loop immediately; any other exception fails just that file.
        Converted temp audio is always cleaned up at the end.

        Returns:
            ``(successful, failed)`` counts.
        """
        successful = 0
        failed = 0

        try:
            for item in items:
                original_file = item[0]
                try:
                    console.rule(f"[dim]{original_file.name}[/]", style="dim")
                    if handler(item):
                        successful += 1
                    else:
                        failed += 1

                except QuotaExhaustedError:
                    console.print("\n[red bold]API quota exhausted — stopping all processing.[/]")
                    console.print("[red]Partial results (if any) have been saved.[/]")
                    console.print("[red]Wait for your quota to reset or upgrade your plan.[/]")
                    break

                except Exception as e:
                    console.print(f"[red]✗[/] Unexpected error processing [cyan]{original_file.name}[/]: {e}")
                    failed += 1

                console.print()  # Add spacing between files

        finally:
            self.cleanup_converted_audio()

        return successful, failed
