"""
Shared streaming file download.

``common/pdf_downloader.py`` (PDFs) and ``AI_audio_summary/01`` (audio/video)
each carried their own copy of the same routine, and the audio one was the last
place in a pipeline script still calling ``requests`` directly.

The ``.part`` temp file is the point of the exercise: these pipelines are re-run
against the same output directory, and a download interrupted halfway must not
be mistaken for a complete file on the next pass.

Usage:
    from common.downloader import stream_download

    path = stream_download(url, output_dir / "123.pdf")
"""

import logging
from pathlib import Path
from typing import Optional

import requests

LOGGER = logging.getLogger(__name__)

CHUNK_SIZE = 8192


def stream_download(
    url: str,
    file_path: Path,
    *,
    timeout: int = 30,
    chunk_size: int = CHUNK_SIZE,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Stream *url* to *file_path*, via a ``.part`` file renamed on success.

    A truncated transfer is detected against ``Content-Length`` where the server
    provides it, and the partial file is removed either way — so a failed
    download leaves nothing behind that a later run would treat as done.

    Args:
        url: Source URL.
        file_path: Final destination. Its parent must exist.
        timeout: Request timeout in seconds. Large media wants a larger value.
        chunk_size: Bytes per write.
        logger: Optional logger; falls back to the module logger.

    Returns:
        *file_path* on success, or ``None`` if the download failed.
    """
    log = logger or LOGGER
    part_path = file_path.with_suffix(file_path.suffix + ".part")

    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()

            with open(part_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        handle.write(chunk)

            expected = response.headers.get("Content-Length")
            if expected is not None and part_path.stat().st_size != int(expected):
                raise requests.RequestException(
                    f"Incomplete download: got {part_path.stat().st_size} of {expected} bytes"
                )

        part_path.rename(file_path)
        return file_path

    except requests.Timeout:
        log.error("Timeout downloading %s", url)
        part_path.unlink(missing_ok=True)
        return None
    except (requests.RequestException, OSError) as exc:
        log.error("Failed to download %s: %s", url, exc)
        part_path.unlink(missing_ok=True)
        return None
