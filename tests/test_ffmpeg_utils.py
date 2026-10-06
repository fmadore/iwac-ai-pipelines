"""Tests for the shared ffmpeg utilities: cleanup and MIME types."""

from pathlib import Path

from common.ffmpeg_utils import cleanup_files, get_mime_type


def test_cleanup_removes_file_but_keeps_parent_by_default(tmp_path):
    directory = tmp_path / "segments"
    directory.mkdir()
    segment = directory / "segment.mp3"
    segment.write_bytes(b"audio")

    cleanup_files([segment])

    assert not segment.exists()
    assert directory.is_dir()


def test_cleanup_can_remove_two_levels_of_empty_parents(tmp_path):
    outer = tmp_path / "work"
    inner = outer / "segments"
    inner.mkdir(parents=True)
    segment = inner / "segment.mp3"
    segment.write_bytes(b"audio")

    cleanup_files([segment], remove_parents=True)

    assert not segment.exists()
    assert not inner.exists()
    assert not outer.exists()


def test_cleanup_ignores_missing_files(tmp_path):
    cleanup_files([tmp_path / "missing.mp3"], remove_parents=True)

    assert tmp_path.exists()


def test_mime_type_of_a_container_in_both_tables_depends_on_the_use():
    """``.mp4`` / ``.webm`` sent for their pictures must not go out as audio."""
    assert get_mime_type(Path("clip.mp4")) == "audio/mp4"
    assert get_mime_type(Path("clip.mp4"), video=True) == "video/mp4"
    assert get_mime_type(Path("clip.webm"), video=True) == "video/webm"
    assert get_mime_type(Path("talk.mp3"), video=True) == "audio/mpeg"
    assert get_mime_type(Path("clip.MOV")) == "video/quicktime"
