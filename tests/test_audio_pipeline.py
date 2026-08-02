"""Tests for audio media discovery and Gemini transcription orchestration."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


PIPELINE = Path(__file__).resolve().parent.parent / "AI_audio_summary"


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, PIPELINE / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


downloader_module = load_script("audio_media_downloader", "01_omeka_media_downloader.py")
transcription_module = load_script("gemini_audio_transcriber", "02_AI_transcribe_audio.py")


def bare_transcriber():
    worker = transcription_module.AudioTranscriber.__new__(
        transcription_module.AudioTranscriber
    )
    worker.client = MagicMock()
    worker.model = "gemini-flash-latest"
    worker.rate_limiter = MagicMock()
    worker.transcription_prompt = "Transcribe."
    worker.last_failure_reason = None
    return worker


def test_media_discovery_keeps_only_supported_original_urls(tmp_path):
    client = MagicMock()
    client.get_resource.side_effect = [
        {"o:source": "recording.mp3", "o:original_url": "https://x/recording.mp3"},
        {"o:source": "scan.pdf", "o:original_url": "https://x/scan.pdf"},
        {"o:source": "missing-url.wav"},
    ]
    downloader = downloader_module.MediaDownloader(client, tmp_path)

    urls = downloader._find_media_urls({
        "o:media": [{"@id": "m1"}, {"@id": "m2"}, {"@id": "m3"}],
    })

    assert [(url, media["o:source"]) for url, media in urls] == [
        ("https://x/recording.mp3", "recording.mp3"),
    ]


def test_existing_media_is_resumed_without_download(tmp_path):
    existing = tmp_path / "recording.mp3"
    existing.write_bytes(b"complete")
    client = MagicMock()
    client.get_item.return_value = {
        "o:id": 7,
        "o:media": [{"@id": "m1"}],
    }
    client.get_resource.return_value = {
        "o:source": "recording.mp3",
        "o:original_url": "https://x/recording.mp3",
    }
    downloader = downloader_module.MediaDownloader(client, tmp_path)

    with patch.object(downloader, "download_file") as download:
        result = downloader.process_item({"o:id": 7})

    assert result == ("7", [str(existing)])
    download.assert_not_called()


def test_large_media_uses_upload_transport():
    worker = bare_transcriber()
    path = MagicMock()
    path.stat.return_value.st_size = transcription_module.INLINE_REQUEST_LIMIT_BYTES + 1
    uploaded = MagicMock()
    worker._upload_via_files_api = MagicMock(return_value=uploaded)

    assert worker._prepare_media_part(path, "audio/mpeg") == (uploaded, uploaded)
    worker._upload_via_files_api.assert_called_once_with(path, "audio/mpeg")


def test_transcription_always_deletes_uploaded_file(tmp_path, monkeypatch):
    worker = bare_transcriber()
    audio_path = tmp_path / "long.mp3"
    audio_path.write_bytes(b"audio")
    uploaded = MagicMock(name="files/123")
    worker._prepare_media_part = MagicMock(return_value=(uploaded, uploaded))
    worker._transcribe_with_retries = MagicMock(return_value="transcript")
    monkeypatch.setattr(transcription_module, "get_mime_type", lambda path: "audio/mpeg")
    cleanup = MagicMock()
    monkeypatch.setattr(transcription_module, "delete_uploaded_file", cleanup)

    assert worker.transcribe_audio(audio_path) == "transcript"

    cleanup.assert_called_once_with(worker.client, uploaded)


def test_retryable_response_is_retried_then_succeeds(monkeypatch, tmp_path):
    worker = bare_transcriber()
    worker.client.models.generate_content.side_effect = [MagicMock(), MagicMock()]
    worker._response_to_text = MagicMock(side_effect=[
        transcription_module._RetryableResponse("RECITATION"),
        "transcript",
    ])
    monkeypatch.setattr(transcription_module.random, "uniform", lambda *_: 0)
    monkeypatch.setattr(transcription_module.time, "sleep", lambda *_: None)

    result = worker._transcribe_with_retries(
        MagicMock(), "prompt", tmp_path / "audio.mp3", 2,
    )

    assert result == "transcript"
    assert worker.client.models.generate_content.call_count == 2
    assert worker.rate_limiter.wait.call_count == 2
