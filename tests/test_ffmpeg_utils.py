"""Tests for safe cleanup helpers in the shared ffmpeg utilities."""

from common.ffmpeg_utils import cleanup_files


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
