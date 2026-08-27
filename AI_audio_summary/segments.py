"""
Transcription-file format for the audio transcription pipeline.

This module is the single owner of the on-disk transcription format that
``03_omeka_transcription_updater.py`` (and resume/retry mode) relies on:

* the metadata header (``Transcription of: ...`` / ``Generated using: ...``
  / optional extra fields / ``=`` * 50 separator);
* the ``[Segment N/M | HH:MM:SS–HH:MM:SS]`` segment markers;
* the ``TRANSCRIPTION FAILED (<reason>)`` failure markers;
* the ``segment_NN.<ext>`` temp-segment naming used to resume a split.

Both writing and parsing live here so the format cannot drift between the
writer and the three regexes that used to parse it independently.

The header optionally records the segment length used when the audio was
split (``Segment length: N minutes``). Resume mode compares it against the
requested ``--segment-minutes`` and refuses to retry on mismatch, because
fixed-length segment numbers only line up when the length is identical.
Files written before this field existed parse fine (the field is ``None``).
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Add repo root to path for shared imports
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.ffmpeg_utils import cleanup_files, sanitize_stem, split_audio

from rich.console import Console

console = Console()

# Header field naming what produced the transcription, e.g.
# "Generated using: Google gemini-3.7-flash". ``03`` reads it back so the
# iwac:transcriptionModel annotation it writes can be checked against what
# actually ran, rather than resting on the operator's memory of which script
# filled this folder.
GENERATOR_FIELD = "Generated using"

# Header field recording the segment length used when the audio was split.
SEGMENT_LENGTH_FIELD = "Segment length"

# A failed segment: "[Segment 3/7 | 00:40:00–01:00:00] TRANSCRIPTION FAILED (...)".
# The header may carry extra detail after the number, so match anything up to
# the closing bracket.
_FAILED_SEGMENT_RE = re.compile(r"\[Segment (\d+)[^\]]*\] TRANSCRIPTION FAILED")

# Any segment marker (used to count total segments).
_SEGMENT_MARKER_RE = re.compile(r"\[Segment (\d+)[^\]]*\]")

# Optional header field, e.g. "Segment length: 20 minutes".
_SEGMENT_LENGTH_RE = re.compile(rf"^{SEGMENT_LENGTH_FIELD}: (\d+) minutes?$", re.MULTILINE)

# Temp segment files produced by splitting: "segment_01.mp3", "segment_12.wav", ...
_SEGMENT_FILE_RE = re.compile(r"^segment_(\d+)\..+$")

_HEADER_SEPARATOR = "=" * 50


# ---------------------------------------------------------------------------
# Segment markers
# ---------------------------------------------------------------------------

def format_timestamp(total_minutes: float) -> str:
    """Format a number of minutes as ``HH:MM:SS``."""
    total_seconds = int(round(total_minutes * 60))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def segment_header(idx: int, total: int, segment_minutes: int) -> str:
    """Build a segment marker that records its position in the recording.

    Segments are fixed-length, so segment ``idx`` (1-based) starts at exactly
    ``(idx - 1) * segment_minutes`` into the original audio. The final segment
    runs to the end of the file (an unknown remainder), so its end is shown as
    ``end`` rather than an over-stated boundary.

    The leading ``[Segment N ...]`` token is what the resume/retry logic
    parses, so it must stay first.
    """
    start = format_timestamp((idx - 1) * segment_minutes)
    if idx < total:
        span = f"{start}–{format_timestamp(idx * segment_minutes)}"
    else:
        span = f"{start}–end"
    return f"[Segment {idx}/{total} | {span}]"


def failed_segment_marker(idx: int, header: str, reason: Optional[str] = None) -> str:
    """Build the ``TRANSCRIPTION FAILED`` marker for a failed segment.

    Uses the full positional *header* when the file was split, or a bare
    ``[Segment N]`` marker otherwise. The optional *reason* is a short,
    parenthesis-free code (e.g. ``RECITATION``, ``API-503``).
    """
    suffix = f" ({reason})" if reason else ""
    if header:
        return f"{header} TRANSCRIPTION FAILED{suffix}"
    return f"[Segment {idx}] TRANSCRIPTION FAILED{suffix}"


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def transcription_path(original_file: Path, output_dir: Path) -> Path:
    """Return the transcription file path for *original_file* inside *output_dir*."""
    return output_dir / f"{original_file.stem}_transcription.txt"


def write_transcription(
    transcription: str,
    original_file: Path,
    output_dir: Path,
    *,
    generator: str,
    extra_fields: Optional[Sequence[Tuple[str, str]]] = None,
    segment_minutes: Optional[int] = None,
) -> Optional[Path]:
    """Write a transcription file with the standard metadata header.

    Args:
        transcription: Transcribed text (body of the file).
        original_file: Original audio/video file (its stem names the output).
        output_dir: Directory the transcription file is written into.
        generator: Value of the ``Generated using:`` header line
            (e.g. ``"Google gemini-pro-latest"``).
        extra_fields: Optional extra ``(name, value)`` header lines
            (e.g. Voxtral's Language/Diarization fields).
        segment_minutes: When the audio was split, the segment length used —
            recorded in the header so resume mode can verify it later.

    Returns:
        The output path, or ``None`` on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = transcription_path(original_file, output_dir)

    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(f"Transcription of: {original_file.name}\n")
            f.write(f"{GENERATOR_FIELD}: {generator}\n")
            for name, value in extra_fields or []:
                f.write(f"{name}: {value}\n")
            if segment_minutes is not None:
                f.write(f"{SEGMENT_LENGTH_FIELD}: {segment_minutes} minutes\n")
            f.write(_HEADER_SEPARATOR + "\n\n")
            f.write(transcription)

        console.print(f"[green]✓[/] Transcription saved: [cyan]{output_file_path}[/]")
        return output_file_path

    except Exception as e:
        console.print(f"[red]✗[/] Error saving transcription: {e}")
        return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _split(text: str) -> Tuple[Dict[str, str], str]:
    """Split raw file text into ``(header fields, body)`` at the separator.

    One definition, because the two halves leave for different places and must
    agree on where the line between them falls: the header is provenance that
    ``03`` reads to name a model, the body is the transcript it uploads.

    A file with no separator has no header and is body in full. Never guess
    which leading lines were metadata: a wrong guess would both let transcript
    text be read as a provenance claim and silently truncate the transcript.
    """
    if _HEADER_SEPARATOR not in text:
        return {}, text.strip()

    header: Dict[str, str] = {}
    head, _, body = text.partition(_HEADER_SEPARATOR)
    for line in head.splitlines():
        name, separator, value = line.partition(":")
        if separator and name.strip():
            header[name.strip()] = value.strip()
    return header, body.strip()


def read_header(path: Path) -> Dict[str, str]:
    """Parse the metadata header of a transcription file.

    Returns the ``Name: value`` lines written above the separator, e.g.
    ``{"Transcription of": "khutba.mp3", "Generated using": "Google gemini-3.7-flash"}``.

    A file with no separator has no header and yields ``{}``: never guess which
    leading lines were metadata, because a wrong guess here would let transcript
    text be read as a provenance claim.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except Exception as e:
        console.print(f"[yellow]⚠[/] Warning: Could not read transcription header: {e}")
        return {}

    return _split(text)[0]


def read_body(path: Path) -> str:
    """Return the transcript alone — everything below the header separator.

    This is what ``03`` uploads. ``bibo:content`` is the archive's full-text
    field, exported to Hugging Face as ``OCR`` and indexed for search, so a
    header left inside it puts "Generated using: Google gemini-3.7-flash" in the
    index as though a speaker had said it. The header stays on disk, where it is
    auditable, and reaches Omeka as an ``iwac:transcriptionModel`` annotation
    instead. ``AI_youtube_transcription`` splits its own files for the same
    reason.

    Unlike :func:`read_header`, a read failure is raised rather than swallowed:
    a header that cannot be read costs an annotation, a body that cannot be read
    would upload an empty transcript over a real one.
    """
    return _split(Path(path).read_text(encoding="utf-8"))[1]


def check_existing_transcription(
    original_file: Path,
    output_dir: Path,
) -> Tuple[bool, List[int], int, Optional[int]]:
    """Check whether a transcription exists and identify any failed segments.

    Args:
        original_file: Path to the original audio/video file.
        output_dir: Directory containing transcription files.

    Returns:
        ``(exists, failed_segments, total_segments, segment_minutes)`` where
        *failed_segments* is a list of 1-indexed segment numbers that carry a
        ``TRANSCRIPTION FAILED`` marker, *total_segments* is the number of
        distinct segment markers (0 if not segmented), and *segment_minutes*
        is the segment length recorded in the header (``None`` for files
        written before this field existed, or unsplit files).
    """
    output_file_path = transcription_path(original_file, output_dir)

    if not output_file_path.exists():
        return False, [], 0, None

    try:
        with open(output_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        failed_segments = [int(m) for m in _FAILED_SEGMENT_RE.findall(content)]

        all_segments = _SEGMENT_MARKER_RE.findall(content)
        total_segments = len(set(all_segments)) if all_segments else 0

        # The segment-length field lives in the header, before the separator;
        # search only there so transcript text can never shadow it.
        header_part = content.split(_HEADER_SEPARATOR, 1)[0]
        length_match = _SEGMENT_LENGTH_RE.search(header_part)
        segment_minutes = int(length_match.group(1)) if length_match else None

        return True, failed_segments, total_segments, segment_minutes

    except Exception as e:
        console.print(f"[yellow]⚠[/] Warning: Could not read existing transcription: {e}")
        return False, [], 0, None


def update_transcription_segment(
    original_file: Path,
    segment_num: int,
    new_content: str,
    output_dir: Path,
) -> bool:
    """Replace a failed segment's marker with newly transcribed content.

    Preserves whatever header the marker had (segment number + any timestamp
    range) and removes the failure annotation, reason included.

    Returns:
        ``True`` if the marker was found and replaced.
    """
    output_file_path = transcription_path(original_file, output_dir)

    try:
        with open(output_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # The (?![0-9]) guard stops segment 2 from matching "[Segment 20 ...]".
        # Match the failed marker plus any optional "(<reason>)" annotation so
        # a successful retry removes the whole marker, reason included.
        failed_pattern = re.compile(
            rf"\[Segment {segment_num}(?![0-9])([^\]]*)\] TRANSCRIPTION FAILED(?: \([^)]*\))?"
        )

        def _replace(match):
            header = f"[Segment {segment_num}{match.group(1)}]"
            return f"{header}\n{new_content}"

        new_text, replaced = failed_pattern.subn(_replace, content)
        if replaced:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(new_text)

            console.print(f"[green]✓[/] Updated segment {segment_num} in transcription file")
            return True

        console.print(f"[yellow]⚠[/] Could not find failed marker for segment {segment_num}")
        return False

    except Exception as e:
        console.print(f"[red]✗[/] Error updating transcription file: {e}")
        return False


# ---------------------------------------------------------------------------
# Temp-segment files (format-coupled: resume relies on the segment naming)
# ---------------------------------------------------------------------------

def find_existing_segments(audio_file_path: Path, segment_root: Path) -> List[Path]:
    """Find segment files left by a previous split of *audio_file_path*.

    Args:
        audio_file_path: Path to the original audio file.
        segment_root: Root temp directory holding per-file segment folders.

    Returns:
        Existing segment paths sorted by segment number, or ``[]``.
    """
    segment_dir = segment_root / sanitize_stem(audio_file_path.stem)

    if not segment_dir.exists() or not segment_dir.is_dir():
        return []

    found: List[Tuple[int, Path]] = []
    for file_path in segment_dir.iterdir():
        if file_path.is_file():
            match = _SEGMENT_FILE_RE.match(file_path.name)
            if match:
                found.append((int(match.group(1)), file_path))

    found.sort(key=lambda item: item[0])
    return [path for _, path in found]


def split_audio_file(
    audio_file_path: Path,
    segment_root: Path,
    segment_minutes: int = 20,
) -> List[Path]:
    """Split an audio file into fixed-length segments.

    Checks for existing segments from a previous run first (resume support),
    then delegates the actual splitting to ``split_audio`` from
    ``common.ffmpeg_utils``.

    Args:
        audio_file_path: Path to the original audio file.
        segment_root: Root temp directory for segments.
        segment_minutes: Length of each segment in minutes.

    Returns:
        Segment file paths (or ``[audio_file_path]`` if no split applied).
    """
    existing = find_existing_segments(audio_file_path, segment_root)
    if existing:
        console.print(
            f"[green]✓[/] Found [bold]{len(existing)}[/] existing segment(s) for '[cyan]{audio_file_path.name}[/]'"
        )
        return existing

    output_dir = segment_root / sanitize_stem(audio_file_path.stem)
    parts = split_audio(audio_file_path, output_dir, segment_minutes)

    if len(parts) > 1:
        console.print(
            f"[green]✓[/] Split '[cyan]{audio_file_path.name}[/]' into [bold]{len(parts)}[/] segment(s) "
            f"of up to {segment_minutes} minutes each."
        )
    # parts == [audio_file_path]: no split needed or failed — logged by split_audio
    return parts


def cleanup_temp_segments(original_audio_file: Path, segment_paths: List[Path]) -> None:
    """Clean up temporary segment files and directories after transcription."""
    # Only clean up if we actually created segments (not just returned original)
    if len(segment_paths) == 1 and segment_paths[0] == original_audio_file:
        return

    to_remove = [p for p in segment_paths if p != original_audio_file]
    cleanup_files(to_remove, remove_parents=True)
    for p in to_remove:
        if not p.exists():
            console.print(f"[dim]🧹 Cleaned up segment: {p.name}[/]")
