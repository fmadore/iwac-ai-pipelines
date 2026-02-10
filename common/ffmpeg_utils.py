"""
Shared FFmpeg utilities for audio and video pipelines.

Provides FFmpeg discovery, pydub configuration, video-to-audio conversion,
audio splitting, MIME type lookups, and file cleanup helpers.

Multimodal pipelines that handle audio/video files should import from here
rather than embedding FFmpeg logic inline.

Usage:
    from common.ffmpeg_utils import (
        AUDIO_FORMATS, VIDEO_FORMATS,
        get_ffmpeg_paths, setup_pydub, is_video_file, get_mime_type,
        convert_video_to_audio, split_audio, cleanup_files,
    )
"""

import logging
import mimetypes
import os
import re
import subprocess
from pathlib import Path
from shutil import which
from typing import Dict, List, NamedTuple, Optional

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Format / MIME-type registries
# ---------------------------------------------------------------------------

AUDIO_FORMATS: Dict[str, str] = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
    ".mp4": "audio/mp4",
    ".aac": "audio/aac",
}

VIDEO_FORMATS: Dict[str, str] = {
    ".mp4": "video/mp4",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".wmv": "video/x-ms-wmv",
    ".flv": "video/x-flv",
    ".webm": "video/webm",
    ".m4v": "video/x-m4v",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".3gp": "video/3gpp",
    ".3gpp": "video/3gpp",
}

AUDIO_EXPORT_FORMAT_MAP: Dict[str, str] = {
    "m4a": "mp4",
    "mp4": "mp4",
    "mp3": "mp3",
    "wav": "wav",
    "flac": "flac",
    "ogg": "ogg",
    "webm": "webm",
    "aac": "aac",
}
"""Maps file extensions (without dot) to the pydub/ffmpeg export format name."""


# ---------------------------------------------------------------------------
# FFmpeg discovery
# ---------------------------------------------------------------------------

class FFmpegPaths(NamedTuple):
    """Resolved paths to ffmpeg and ffprobe executables."""
    ffmpeg: str
    ffprobe: str


# Module-level cache
_cached_ffmpeg_paths: Optional[FFmpegPaths] = None
_ffmpeg_searched: bool = False


def _normalize_candidate(path_str: Optional[str], exe_name: str = "ffmpeg") -> Optional[str]:
    """Resolve an env-var value to an executable path.

    If the value points to a directory, appends the platform-appropriate
    executable name.  Strips surrounding whitespace and quotes.
    """
    if not path_str:
        return None
    path_str = path_str.strip().strip('"')
    p = Path(path_str)
    if p.is_dir():
        suffix = ".exe" if os.name == "nt" else ""
        return str(p / f"{exe_name}{suffix}")
    return str(p)


def _find_executable(name: str, env_var: Optional[str] = None) -> Optional[str]:
    """Locate an executable by env var, PATH, or common Windows directories."""
    # 1. Environment variable override
    candidate = _normalize_candidate(os.environ.get(env_var) if env_var else None, name)
    if candidate and Path(candidate).exists():
        return candidate

    # 2. PATH lookup
    found = which(name) or which(f"{name}.exe")
    if found:
        return found

    # 3. Common Windows install locations
    if os.name == "nt":
        common_dirs = [
            rf"C:\Program Files\ffmpeg\bin\{name}.exe",
            rf"C:\Program Files (x86)\ffmpeg\bin\{name}.exe",
            rf"C:\ffmpeg\bin\{name}.exe",
        ]
        for loc in common_dirs:
            if Path(loc).exists():
                return loc

    return None


def get_ffmpeg_paths() -> Optional[FFmpegPaths]:
    """Discover ffmpeg and ffprobe executables.

    Results are cached at module level so the filesystem is only probed once.

    Returns:
        An ``FFmpegPaths`` namedtuple, or ``None`` if either executable
        could not be found.
    """
    global _cached_ffmpeg_paths, _ffmpeg_searched
    if _ffmpeg_searched:
        return _cached_ffmpeg_paths

    _ffmpeg_searched = True

    ffmpeg = _find_executable("ffmpeg", "FFMPEG_PATH")
    ffprobe = _find_executable("ffprobe", "FFPROBE_PATH")

    if ffmpeg and ffprobe:
        _cached_ffmpeg_paths = FFmpegPaths(ffmpeg=ffmpeg, ffprobe=ffprobe)
        LOGGER.info("Detected ffmpeg: %s", ffmpeg)
        LOGGER.info("Detected ffprobe: %s", ffprobe)
    else:
        _cached_ffmpeg_paths = None
        if not ffmpeg:
            LOGGER.warning("ffmpeg not found. Set FFMPEG_PATH or install ffmpeg.")
        if not ffprobe:
            LOGGER.warning("ffprobe not found. Set FFPROBE_PATH or install ffmpeg.")

    return _cached_ffmpeg_paths


# ---------------------------------------------------------------------------
# pydub integration
# ---------------------------------------------------------------------------

def setup_pydub() -> bool:
    """Import pydub and configure it to use discovered FFmpeg paths.

    Returns:
        ``True`` if pydub is importable **and** ffmpeg/ffprobe are available.
    """
    try:
        from pydub import AudioSegment  # noqa: F811
    except ImportError:
        LOGGER.warning("pydub is not installed; audio splitting unavailable.")
        return False

    paths = get_ffmpeg_paths()
    if not paths:
        return False

    AudioSegment.converter = paths.ffmpeg
    AudioSegment.ffmpeg = paths.ffmpeg
    AudioSegment.ffprobe = paths.ffprobe
    LOGGER.debug("pydub configured: converter=%s", paths.ffmpeg)
    return True


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_UNSAFE_PATH_RE = re.compile(r"""['"<>|*?]""")


def sanitize_stem(name: str) -> str:
    """Remove characters that break subprocess / ffmpeg path arguments.

    Strips single/double quotes and a few other shell-unsafe characters
    while preserving Unicode letters, digits, spaces, hyphens, and dots.
    """
    return _UNSAFE_PATH_RE.sub("", name)


def has_unsafe_path_chars(name: str) -> bool:
    """Return ``True`` if *name* contains characters that break ffmpeg paths."""
    return bool(_UNSAFE_PATH_RE.search(name))


# ---------------------------------------------------------------------------
# File-type helpers
# ---------------------------------------------------------------------------

def is_video_file(file_path: Path) -> bool:
    """Check whether *file_path* is a video file.

    Checks the extension against ``VIDEO_FORMATS`` first, then falls back to
    ``mimetypes.guess_type()``.
    """
    ext = file_path.suffix.lower()
    if ext in VIDEO_FORMATS:
        return True
    mime, _ = mimetypes.guess_type(str(file_path))
    return bool(mime and mime.startswith("video/"))


def get_mime_type(file_path: Path) -> Optional[str]:
    """Return the MIME type for an audio or video file.

    Checks ``AUDIO_FORMATS`` and ``VIDEO_FORMATS`` first, then falls back
    to ``mimetypes.guess_type()``.
    """
    ext = file_path.suffix.lower()
    mime = AUDIO_FORMATS.get(ext) or VIDEO_FORMATS.get(ext)
    if mime:
        return mime
    guessed, _ = mimetypes.guess_type(str(file_path))
    return guessed


# ---------------------------------------------------------------------------
# Conversion & splitting
# ---------------------------------------------------------------------------

def convert_video_to_audio(
    video_path: Path,
    output_dir: Path,
    output_format: str = "mp3",
    timeout: int = 3600,
) -> Optional[Path]:
    """Convert a video file to audio using ffmpeg.

    Args:
        video_path: Path to the source video.
        output_dir: Directory where the converted audio is written.
        output_format: Target audio format (default ``"mp3"``).
        timeout: Maximum seconds for the ffmpeg process.

    Returns:
        Path to the converted audio file, or ``None`` on failure.
    """
    paths = get_ffmpeg_paths()
    if not paths:
        LOGGER.error("Cannot convert video: ffmpeg not available.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = sanitize_stem(video_path.stem)
    output_filename = f"{safe_stem}_audio.{output_format}"
    output_path = output_dir / output_filename

    LOGGER.info("Converting %s -> %s", video_path.name, output_filename)

    if output_format == "mp3":
        cmd = [
            paths.ffmpeg,
            "-i", str(video_path),
            "-vn",
            "-acodec", "libmp3lame",
            "-ab", "192k",
            "-ar", "44100",
            "-y",
            str(output_path),
        ]
    else:
        cmd = [
            paths.ffmpeg,
            "-i", str(video_path),
            "-vn",
            "-ab", "192k",
            "-y",
            str(output_path),
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            LOGGER.error("ffmpeg conversion failed: %s", result.stderr)
            return None
        if output_path.exists():
            LOGGER.info("Video converted: %s", output_path.name)
            return output_path
        LOGGER.error("Conversion produced no output file.")
        return None
    except subprocess.TimeoutExpired:
        LOGGER.error("Video conversion timed out for %s", video_path.name)
        return None
    except Exception as exc:
        LOGGER.error("Error converting %s: %s", video_path.name, exc)
        return None


def split_audio(
    audio_path: Path,
    output_dir: Path,
    segment_minutes: int = 10,
) -> List[Path]:
    """Split an audio file into fixed-length segments using pydub.

    If the file is shorter than one segment, the original path is returned
    unchanged.  On any failure (pydub missing, ffmpeg unavailable, I/O
    error), the original path is returned so that callers can still
    proceed with the unsplit file.

    Args:
        audio_path: Source audio file.
        output_dir: Directory to write segment files into.
        segment_minutes: Maximum length of each segment in minutes.

    Returns:
        List of segment file paths (or ``[audio_path]`` if no split was
        performed).
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        LOGGER.warning("pydub not installed; skipping split for %s", audio_path.name)
        return [audio_path]

    if not setup_pydub():
        LOGGER.warning("ffmpeg not configured; skipping split for %s", audio_path.name)
        return [audio_path]

    try:
        segment_ms = segment_minutes * 60 * 1000
        audio = AudioSegment.from_file(audio_path)
        if len(audio) <= segment_ms:
            return [audio_path]

        output_dir.mkdir(parents=True, exist_ok=True)

        segments: List[Path] = []
        for i, start in enumerate(range(0, len(audio), segment_ms), start=1):
            end = min(start + segment_ms, len(audio))
            chunk = audio[start:end]
            segment_filename = f"segment_{i:02d}{audio_path.suffix}"
            segment_path = output_dir / segment_filename

            file_ext = audio_path.suffix.lstrip(".").lower()
            export_format = AUDIO_EXPORT_FORMAT_MAP.get(file_ext, file_ext)
            chunk.export(segment_path, format=export_format)
            segments.append(segment_path)

        LOGGER.info(
            "Split '%s' into %d segment(s) of up to %d min each.",
            audio_path.name, len(segments), segment_minutes,
        )
        return segments

    except Exception as exc:
        LOGGER.error("Error splitting %s: %s. Returning unsplit.", audio_path.name, exc)
        return [audio_path]


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

def cleanup_files(file_paths: List[Path], remove_parents: bool = False) -> None:
    """Delete files and optionally their parent directories.

    Directories are only removed if they become empty after file deletion.

    Args:
        file_paths: Files to delete.
        remove_parents: If ``True``, attempt to remove each file's parent
            directory (and *its* parent) when empty.
    """
    parents_to_check: List[Path] = []

    for fp in file_paths:
        try:
            if fp.exists():
                fp.unlink()
                LOGGER.debug("Removed %s", fp)
                if remove_parents:
                    parents_to_check.append(fp.parent)
        except Exception as exc:
            LOGGER.warning("Could not remove %s: %s", fp, exc)

    if not remove_parents:
        return

    # Deduplicate, then try to remove empty dirs (child first, then parent)
    seen: set = set()
    for d in parents_to_check:
        if d in seen:
            continue
        seen.add(d)
        for target in (d, d.parent):
            try:
                if target.exists() and target.is_dir():
                    target.rmdir()  # only succeeds if empty
                    LOGGER.debug("Removed empty directory %s", target)
            except OSError:
                pass
