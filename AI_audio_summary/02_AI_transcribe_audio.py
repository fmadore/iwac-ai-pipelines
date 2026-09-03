#!/usr/bin/env python3
"""
Audio Transcription Script using Google Gemini Pro
Transcribes audio and video files from the Audio folder and saves them as text files.
Video files are automatically converted to audio before transcription.
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from google.genai import types
from google.genai import errors as genai_errors

# Add repo root to path for shared imports
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.gemini_utils import (
    INLINE_REQUEST_LIMIT_BYTES,
    build_generation_config,
    build_gemini_client,
    delete_uploaded_file,
    extract_text_from_response,
    upload_and_wait_active,
)
from common.prompt_loader import select_prompt_interactive
from common.rate_limiter import QuotaExhaustedError, is_quota_exhausted
from common.ffmpeg_utils import get_mime_type
from common.log_redaction import install_credential_redaction

# Credentials ride in Omeka query strings and provider headers; keep them
# out of anything urllib3 or an SDK decides to log.
install_credential_redaction()

from rich.panel import Panel
from rich.table import Table
from rich import box

from transcriber_base import SCRIPT_DIR, TranscriberBase, console, detect_ffmpeg, make_config_table
from segments import (
    check_existing_transcription,
    cleanup_temp_segments,
    failed_segment_marker,
    segment_header,
    split_audio_file,
    update_transcription_segment,
)

# Load environment variables from .env file FIRST
load_dotenv()

#: The models this script offers, in menu order, with the note shown beside each.
#: Two of them are rolling aliases, so ``03`` cannot annotate their output: an
#: annotation naming "Gemini Pro Latest" asserts a release the run never
#: confirmed, which is why ``AI_MODEL_ITEMS`` deliberately holds no entry for
#: either. Any *pinned* id added here needs an Omeka authority item, or ``03``
#: will refuse the folder it fills — ``tests/test_audio_pipeline.py`` guards that.
ALLOWED_MODELS = {
    "gemini-pro-latest": "Higher quality, slower",
    "gemini-3.7-flash": "Faster, good quality",
    "gemini-flash-lite-latest": "Fastest, cheapest, lowest latency",
}
# Pinned, so the default is a model step 03 can annotate; the rolling aliases
# stay on offer for a run that does not go to Omeka.
DEFAULT_MODEL = "gemini-3.7-flash"

# Default transcription prompt (fallback when no prompt file is selected)
DEFAULT_PROMPT = """
        Please transcribe the audio content accurately.
        Include proper punctuation and formatting.
        If there are multiple speakers, indicate speaker changes.
        Provide a clear, readable transcription of the spoken content.
        """


class _RetryableResponse(Exception):
    """Internal signal: a response came back empty/blocked but is worth retrying.

    Gemini can return an empty or blocked result (RECITATION, a momentary blank
    candidate, or MAX_TOKENS with no recoverable text) that is *transient* — a
    fresh sample usually succeeds. Raising this routes such cases through the
    same exponential-backoff path as API errors instead of failing the segment
    outright.

    The exception message is a short, parenthesis-free reason code suitable for
    the ``TRANSCRIPTION FAILED (<reason>)`` segment marker.
    """


def _finish_reason_code(finish_reason) -> str:
    """Return the bare finish-reason name (e.g. ``MAX_TOKENS``).

    Tolerates a ``types.FinishReason`` enum, a plain string, or ``None``.
    """
    if finish_reason is None:
        return "UNKNOWN"
    name = getattr(finish_reason, "name", None)
    if name:
        return name
    text = str(finish_reason)
    return text.rsplit(".", 1)[-1] if "." in text else text


class AudioTranscriber(TranscriberBase):
    def __init__(
        self,
        api_key=None,
        model=DEFAULT_MODEL,
        requests_per_minute: Optional[int] = None,
        transcription_prompt: Optional[str] = None,
        auto_split: bool = False,
    ):
        """
        Initialize the Audio Transcriber with Gemini API.

        Args:
            api_key (str, optional): Gemini API key. If None, will use GEMINI_API_KEY environment variable.
            model (str, optional): Model to use — 'gemini-pro-latest', 'gemini-3.7-flash', or
                'gemini-flash-lite-latest'. Default is 'gemini-pro-latest'.
                The Flash slot names a pinned release where the other two roll,
                because each transcript records ``Generated using: Google
                <model>`` in its header — and "gemini-flash-latest" names no
                version a reader could look up later. Pro and Flash-Lite are
                still rolling for now; pin them when their headers start being
                cited too.
            requests_per_minute: Optional RPM limit for proactive throttling (None = no throttling)
            transcription_prompt: The transcription prompt to use (selected in ``main()``);
                falls back to ``DEFAULT_PROMPT``.
            auto_split: Whether the selected prompt auto-enables audio splitting.
        """
        super().__init__(requests_per_minute)

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file or environment variables")

        # Store the model choice
        self.model = model
        self.generator_label = f"Google {model}"

        # Initialize the Gemini client
        self.client = build_gemini_client(self.api_key)

        # Short, human-readable reason for the most recent transcription failure
        # (e.g. "RECITATION", "MAX_TOKENS", "API-503"). Surfaced in the
        # "TRANSCRIPTION FAILED (<reason>)" segment marker; None on success.
        self.last_failure_reason = None

        # Prompt is injected by main() so constructing the class never blocks on stdin.
        self.transcription_prompt = transcription_prompt or DEFAULT_PROMPT
        self.auto_split = auto_split

    def prepare_audio_for_api(self, audio_file_path):
        """
        Prepare audio file for Gemini API by reading it as bytes.

        Args:
            audio_file_path (Path): Path to the audio file

        Returns:
            tuple: (audio_bytes, mime_type)
        """
        try:
            with open(audio_file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()

            mime_type = get_mime_type(audio_file_path)
            return audio_bytes, mime_type

        except Exception as e:
            console.print(f"[red]✗[/] Error reading audio file [cyan]{audio_file_path}[/]: {e}")
            return None, None

    def transcribe_audio(self, audio_file_path, custom_prompt=None, max_retries=5):
        """
        Transcribe a single audio file using Gemini with a resilient retry mechanism.

        Retries cover not only transient API/network errors but also *empty or
        blocked* responses (RECITATION, MAX_TOKENS with no text, momentary
        blanks), which are frequently non-deterministic for audio. Partial text
        is recovered when the model truncates on MAX_TOKENS.

        Args:
            audio_file_path (Path): Path to the audio file
            custom_prompt (str, optional): Custom transcription prompt
            max_retries (int): Maximum number of attempts (default: 5)

        Returns:
            str: Transcribed text, or None if every attempt failed (in which
                 case ``self.last_failure_reason`` holds a short reason code).
        """
        console.print(f"[cyan]🎤[/] Transcribing: [bold]{audio_file_path.name}[/]")

        if max_retries <= 0:
            raise ValueError("max_retries must be positive")

        mime_type = get_mime_type(audio_file_path)
        if not mime_type:
            console.print(f"[red]✗[/] Could not determine MIME type for [cyan]{audio_file_path.name}[/]")
            return None
        uploaded_file = None
        try:
            media_part, uploaded_file = self._prepare_media_part(
                audio_file_path, mime_type,
            )
            if media_part is None:
                return None
            prompt = custom_prompt or self.transcription_prompt
            return self._transcribe_with_retries(
                media_part, prompt, audio_file_path, max_retries,
            )
        finally:
            if uploaded_file is not None:
                delete_uploaded_file(self.client, uploaded_file)
                console.print(f"[dim]🧹 Removed uploaded file: {uploaded_file.name}[/]")

    def _prepare_media_part(self, audio_file_path: Path, mime_type: str):
        """Choose inline bytes or an uploaded Files API handle by payload size."""
        try:
            file_size = audio_file_path.stat().st_size
        except OSError:
            file_size = 0
        if file_size > INLINE_REQUEST_LIMIT_BYTES:
            uploaded = self._upload_via_files_api(audio_file_path, mime_type)
            return uploaded, uploaded
        audio_bytes, _ = self.prepare_audio_for_api(audio_file_path)
        if not audio_bytes:
            return None, None
        return types.Part.from_bytes(data=audio_bytes, mime_type=mime_type), None

    def _transcribe_with_retries(
        self,
        media_part,
        prompt: str,
        audio_file_path: Path,
        max_retries: int,
    ) -> Optional[str]:
        """Generate with retry/backoff for transient API and response failures."""
        self.last_failure_reason = None
        last_error = None
        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait()
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[media_part, prompt],
                    config=build_generation_config(
                        self.model,
                        max_output_tokens=65536,
                    ),
                )
                return self._response_to_text(response, audio_file_path)
            except QuotaExhaustedError:
                raise
            except genai_errors.APIError as exc:
                if is_quota_exhausted(exc):
                    raise QuotaExhaustedError(str(exc)) from exc
                last_error = exc
                self.last_failure_reason = f"API-{getattr(exc, 'code', '?')}"
            except _RetryableResponse as exc:
                last_error = exc
                self.last_failure_reason = str(exc)
            except Exception as exc:
                last_error = exc
                self.last_failure_reason = "error"
            self._report_retry(audio_file_path, attempt, max_retries, last_error)
        return None

    def _report_retry(
        self,
        audio_file_path: Path,
        attempt: int,
        max_retries: int,
        error: Optional[Exception],
    ) -> None:
        """Report one failure and sleep only when another attempt remains."""
        if attempt >= max_retries - 1:
            console.print(
                f"[red]✗[/] Giving up on [cyan]{audio_file_path.name}[/] after "
                f"{max_retries} attempts ([yellow]{self.last_failure_reason}[/]): {error}"
            )
            return
        wait_time = 2 ** (attempt + 1) + random.uniform(0, 2)
        console.print(
            f"[red]✗[/] Transcription attempt failed for "
            f"[cyan]{audio_file_path.name}[/] "
            f"([yellow]{self.last_failure_reason}[/]): {error}"
        )
        console.print(
            f"[yellow]⏳[/] Retrying in {wait_time:.1f}s... "
            f"(attempt {attempt + 1}/{max_retries})"
        )
        time.sleep(wait_time)

    def _response_to_text(self, response, audio_file_path):
        """Extract transcription text from a Gemini response, or signal a retry.

        Mirrors the robust handling used by the OCR pipeline:

        * **MAX_TOKENS** — keep whatever was generated (a truncated segment is
          still useful) and append an inline truncation marker. If nothing was
          produced, treat it as retryable.
        * **Empty / blocked** (no candidates, RECITATION, SAFETY, OTHER, or a
          momentary blank) — raise :class:`_RetryableResponse` so the caller
          resamples instead of failing the segment outright.

        Returns:
            str: the transcription text (possibly with a truncation marker).

        Raises:
            _RetryableResponse: when a fresh attempt may succeed.
        """
        # Whole prompt rejected before any generation (rare with safety off).
        feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None) if feedback else None
        if block_reason:
            raise _RetryableResponse(f"prompt-blocked-{_finish_reason_code(block_reason)}")

        if not response.candidates:
            raise _RetryableResponse("no-candidates")

        candidate = response.candidates[0]
        code = _finish_reason_code(candidate.finish_reason)

        # MAX_TOKENS: salvage partial text rather than discarding the segment.
        if code == "MAX_TOKENS":
            partial = extract_text_from_response(response)
            if partial:
                console.print(
                    f"[yellow]⚠[/] Output truncated (MAX_TOKENS) for [cyan]{audio_file_path.name}[/] "
                    f"— partial transcription kept"
                )
                return partial + "\n\n[... TRANSCRIPTION TRUNCATED — OUTPUT EXCEEDED MAX TOKENS ...]"
            raise _RetryableResponse("MAX_TOKENS")

        text = extract_text_from_response(response)
        if text:
            return text

        # Empty text with no usable parts — RECITATION/SAFETY/OTHER or a blank.
        raise _RetryableResponse(code if code not in ("STOP", "UNKNOWN") else "empty")

    def _upload_via_files_api(self, audio_file_path, mime_type):
        """Upload a media file via the Gemini Files API and wait until it is ACTIVE.

        Used for segments/files that exceed the 20 MB inline request limit.
        The Files API supports files up to ~2 GB / 9.5 h of audio. Delegates
        the upload/poll loop to ``common.gemini_utils.upload_and_wait_active``.

        Returns:
            The ACTIVE ``File`` handle, or ``None`` on failure.
        """
        size_mb = audio_file_path.stat().st_size / (1024 * 1024)
        console.print(f"[cyan]☁[/] Uploading [bold]{audio_file_path.name}[/] ({size_mb:.1f} MB) via Files API...")
        try:
            uploaded = upload_and_wait_active(
                self.client,
                audio_file_path,
                mime_type=mime_type,
                poll_interval=2.0,
            )
        except genai_errors.APIError as e:
            if is_quota_exhausted(e):
                raise QuotaExhaustedError(str(e)) from e
            console.print(f"[red]✗[/] Files API upload failed for [cyan]{audio_file_path.name}[/]: {e}")
            return None
        except (RuntimeError, TimeoutError) as e:
            console.print(f"[red]✗[/] Files API upload failed for [cyan]{audio_file_path.name}[/]: {e}")
            return None

        console.print(f"[green]✓[/] Upload ready: [cyan]{uploaded.name}[/]")
        return uploaded

    # ---- Per-file processing --------------------------------------------------

    def retry_failed_segments(
        self,
        original_file: Path,
        is_video: bool,
        failed_segments: List[int],
        total_segments: int,
        custom_prompt: Optional[str] = None,
        output_folder: str = "Transcriptions",
        segment_minutes: int = 20,
        header_minutes: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Retry the failed segments of one file's existing transcription.

        Used by both resume mode and the interactive retry path. Splits the
        audio (or reuses existing temp segments), re-transcribes each failed
        segment, and patches the transcription file in place.

        When the transcription header records a segment length
        (*header_minutes*) that differs from the requested *segment_minutes*,
        the retry is refused: fixed-length segment numbers only line up when
        the length matches, and retrying with mismatched boundaries would
        splice the wrong audio into the file.

        Returns:
            ``(successful, failed)`` segment counts. Raises
            ``QuotaExhaustedError`` mid-file; temp segments are then kept on
            disk so the next resume run can pick up where this one stopped.
        """
        console.print(f"[cyan]🔄[/] Retrying {len(failed_segments)} failed segment(s): {failed_segments}")

        if header_minutes is not None and header_minutes != segment_minutes:
            console.print(
                f"[yellow]⚠[/] Refusing to resume '[cyan]{original_file.name}[/]': the transcription was "
                f"created with [bold]{header_minutes}[/]-minute segments but --segment-minutes is "
                f"[bold]{segment_minutes}[/]. Rerun with [cyan]--segment-minutes {header_minutes}[/] "
                f"or delete the transcription file to start over."
            )
            return 0, len(failed_segments)

        # Convert video to audio if needed
        if is_video:
            audio_file = self._convert_video(original_file)
            if not audio_file:
                console.print(f"[yellow]⚠[/] Skipping [cyan]{original_file.name}[/]: video conversion failed")
                return 0, len(failed_segments)
        else:
            audio_file = original_file

        # Split the audio to get segments (reuses existing temp segments)
        segment_paths = split_audio_file(audio_file, SCRIPT_DIR / "temp_segments", segment_minutes)

        successful = 0
        failed = 0
        for seg_num in failed_segments:
            if seg_num > len(segment_paths):
                console.print(f"[red]✗[/] Segment {seg_num} out of range")
                failed += 1
                continue

            segment_path = segment_paths[seg_num - 1]  # Convert to 0-indexed
            console.print(f"[cyan]📍[/] Retrying segment {seg_num}/{total_segments}: [bold]{segment_path.name}[/]")

            seg_transcription = self.transcribe_audio(segment_path, custom_prompt)

            if seg_transcription and update_transcription_segment(
                original_file, seg_num, seg_transcription, SCRIPT_DIR / output_folder
            ):
                successful += 1
            else:
                if not seg_transcription:
                    console.print(f"[red]✗[/] Segment {seg_num} still failed")
                failed += 1

        # Clean up segments
        cleanup_temp_segments(audio_file, segment_paths)
        return successful, failed

    def transcribe_new_file(
        self,
        original_file: Path,
        is_video: bool,
        custom_prompt: Optional[str] = None,
        split_segments: bool = False,
        segment_minutes: int = 20,
        output_folder: str = "Transcriptions",
    ) -> bool:
        """Transcribe a file with no existing transcription.

        Optionally splits the audio into fixed-length segments, adding a
        positional ``[Segment N/M | start–end]`` header per segment and a
        ``TRANSCRIPTION FAILED (<reason>)`` marker for any segment that
        exhausted its retries.

        Returns:
            ``True`` when a transcription file was saved.
        """
        # Convert video to audio if needed
        if is_video:
            audio_file = self._convert_video(original_file)
            if not audio_file:
                console.print(f"[yellow]⚠[/] Skipping [cyan]{original_file.name}[/]: video conversion failed")
                return False
        else:
            audio_file = original_file

        used_segment_minutes = None
        if split_segments:
            # Split into segments (or return original if no splitting applied)
            segment_paths = split_audio_file(audio_file, SCRIPT_DIR / "temp_segments", segment_minutes)
            combined_transcription_parts = []
            all_segments_successful = True

            total_segments = len(segment_paths)
            for idx, segment_path in enumerate(segment_paths, start=1):
                console.print(f"[cyan]📍[/] Processing segment {idx}/{total_segments}: [bold]{segment_path.name}[/]")
                seg_transcription = self.transcribe_audio(segment_path, custom_prompt)
                # Header records the segment's start–end position in the
                # recording; only added when the file was actually split.
                header = segment_header(idx, total_segments, segment_minutes) if total_segments > 1 else ""
                if seg_transcription:
                    part = f"{header}\n{seg_transcription}\n" if header else f"{seg_transcription}\n"
                    combined_transcription_parts.append(part)
                else:
                    combined_transcription_parts.append(
                        failed_segment_marker(idx, header, self.last_failure_reason)
                    )
                    all_segments_successful = False

            # Combine
            transcription = "\n".join(combined_transcription_parts).strip()

            if total_segments > 1:
                # Record the segment length in the header so resume mode can
                # verify it matches before splicing retried segments back in.
                used_segment_minutes = segment_minutes
                # Clean up temporary segments only if all were successful and
                # we actually split the file (failed ones are kept for resume).
                if all_segments_successful:
                    cleanup_temp_segments(audio_file, segment_paths)
        else:
            transcription = self.transcribe_audio(audio_file, custom_prompt)

        if not transcription:
            return False

        # Use original file name for output (not the converted audio name)
        output_file = self.save_transcription(
            transcription, original_file, output_folder, segment_minutes=used_segment_minutes
        )
        return output_file is not None

    # ---- Orchestration -----------------------------------------------------------

    def _classify_media_files(self, media_files, output_folder):
        """Partition media into new, failed-segment, and complete buckets."""
        files_to_process = []
        files_to_retry = []
        files_complete = []
        for file_path, is_video in media_files:
            exists, failed_segments, total_segments, header_minutes = (
                check_existing_transcription(file_path, SCRIPT_DIR / output_folder)
            )
            if not exists:
                files_to_process.append((file_path, is_video, None))
            elif failed_segments:
                files_to_retry.append(
                    (file_path, is_video, failed_segments, total_segments, header_minutes)
                )
            else:
                files_complete.append((file_path, is_video))
        return files_to_process, files_to_retry, files_complete

    @staticmethod
    def _select_normal_work(files_to_process, files_to_retry):
        """Optionally add failed segments to the normal new-file batch."""
        selected = files_to_process.copy()
        if not files_to_retry:
            return selected
        retry_choice = console.input(
            f"\n[bold]Found {len(files_to_retry)} file(s) with failed segments. "
            "Retry them? (Y/n):[/] "
        ).strip().lower()
        if retry_choice == "n":
            return selected
        selected.extend(
            (file_path, is_video, (failed, total, header_minutes))
            for file_path, is_video, failed, total, header_minutes in files_to_retry
        )
        return selected

    def _process_batch_item(
        self,
        item,
        *,
        custom_prompt,
        output_folder,
        segment_minutes,
        split_segments,
    ) -> bool:
        """Process one new file or retry record from a normal batch."""
        original_file, is_video, retry_info = item
        if retry_info:
            failed_segments, total_segments, header_minutes = retry_info
            _, still_failed = self.retry_failed_segments(
                original_file,
                is_video,
                failed_segments,
                total_segments,
                custom_prompt=custom_prompt,
                output_folder=output_folder,
                segment_minutes=segment_minutes,
                header_minutes=header_minutes,
            )
            return still_failed == 0
        return self.transcribe_new_file(
            original_file,
            is_video,
            custom_prompt=custom_prompt,
            split_segments=split_segments,
            segment_minutes=segment_minutes,
            output_folder=output_folder,
        )

    def transcribe_all_audio_files(
        self,
        audio_folder="Audio",
        output_folder="Transcriptions",
        custom_prompt=None,
        split_segments=False,
        segment_minutes=20,
        resume_mode=False,
    ):
        """
        Transcribe all audio and video files in the specified folder.
        Video files are automatically converted to audio before transcription.

        Args:
            audio_folder (str): Path to the audio folder
            output_folder (str): Output folder for transcriptions
            custom_prompt (str, optional): Custom transcription prompt
            split_segments (bool): Whether to split audio into segments
            segment_minutes (int): Length of each segment in minutes
            resume_mode (bool): If True, only retry failed segments in existing transcriptions
        """
        media_files = self.get_audio_files(audio_folder)

        if not media_files:
            self.print_no_files_warning()
            return

        files_to_process, files_to_retry, files_complete = self._classify_media_files(
            media_files, output_folder,
        )

        video_count = sum(1 for _, is_video, *_ in media_files if is_video)

        # Show summary of what needs to be done
        self.print_status_table(files_complete, files_to_retry, files_to_process)

        # In resume mode, only process files with failed segments
        if resume_mode:
            self._resume_failed_segments(files_to_retry, custom_prompt, output_folder, segment_minutes)
            return

        all_files_to_process = self._select_normal_work(
            files_to_process, files_to_retry,
        )

        if not all_files_to_process:
            console.print("[green]✓[/] All files are already transcribed successfully!")
            return

        # Display files table
        self.print_files_table(all_files_to_process, with_status=True)

        # Summary
        console.print(f"\n[bold]Summary:[/] [cyan]{len(all_files_to_process)}[/] file(s) to process")
        if video_count > 0:
            console.print("[dim](Video files will be converted to audio)[/]")

        console.print()
        console.rule("[bold]Starting Transcription Process", style="cyan")
        console.print()

        successful_transcriptions, failed_transcriptions = self.run_processing_loop(
            all_files_to_process,
            lambda item: self._process_batch_item(
                item,
                custom_prompt=custom_prompt,
                output_folder=output_folder,
                segment_minutes=segment_minutes,
                split_segments=split_segments,
            ),
        )

        self.print_summary_table(len(media_files), successful_transcriptions, failed_transcriptions, output_folder)

    def _resume_failed_segments(self, files_to_retry, custom_prompt, output_folder, segment_minutes):
        """Resume mode: retry only failed segments in existing transcriptions."""
        if not files_to_retry:
            console.print("[green]✓[/] No failed segments to retry. All transcriptions are complete!")
            return

        total_failed = sum(len(item[2]) for item in files_to_retry)
        console.print(
            f"[bold]Resume mode:[/] Retrying [cyan]{total_failed}[/] failed segment(s) "
            f"in [cyan]{len(files_to_retry)}[/] file(s)"
        )
        console.print()

        successful_retries = 0
        failed_retries = 0

        try:
            for original_file, is_video, failed_segments, total_segments, header_minutes in files_to_retry:
                console.rule(f"[dim]Retrying: {original_file.name}[/]", style="dim")
                try:
                    successful, failed = self.retry_failed_segments(
                        original_file, is_video, failed_segments, total_segments,
                        custom_prompt=custom_prompt, output_folder=output_folder,
                        segment_minutes=segment_minutes, header_minutes=header_minutes,
                    )
                    successful_retries += successful
                    failed_retries += failed
                except QuotaExhaustedError:
                    console.print("\n[red bold]API quota exhausted — stopping all processing.[/]")
                    console.print("[red]Partial results have been saved.[/]")
                    break
                console.print()

        finally:
            self.cleanup_converted_audio()

        # Summary
        console.print()
        console.rule("[bold]Retry Summary", style="cyan")

        summary_table = Table(title="📊 Results", box=box.ROUNDED)
        summary_table.add_column("Metric", style="dim")
        summary_table.add_column("Value", style="green")
        summary_table.add_row("Segments retried", str(successful_retries + failed_retries))
        summary_table.add_row("Successful retries", f"[green]{successful_retries}[/]")
        summary_table.add_row("Still failing", f"[red]{failed_retries}[/]" if failed_retries > 0 else "0")
        console.print(summary_table)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Audio Transcription using Google Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        choices=list(ALLOWED_MODELS),
        default=None,
        help="Model to use for transcription (default: interactive selection)"
    )
    parser.add_argument(
        "--audio-folder",
        default="Audio",
        help="Folder containing audio files (default: Audio)"
    )
    parser.add_argument(
        "--output-folder",
        default="Transcriptions",
        help="Folder for output transcriptions (default: Transcriptions)"
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split audio into 20-minute segments (segments over 20 MB are sent via the Files API)"
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Never split; answers the interactive 'Split audio?' question with no",
    )
    parser.add_argument(
        "--prompt",
        type=int,
        default=None,
        help="Prompt number from prompts/ (0 = built-in default) instead of the menu",
    )
    parser.add_argument(
        "--segment-minutes",
        type=int,
        default=20,
        help="Segment length in minutes when splitting (default: 20)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume mode: only retry failed segments in existing transcriptions"
    )
    return parser.parse_args()


def select_model_interactive():
    """
    Interactively select a model if not provided via CLI.
    """
    models_table = Table(title="🤖 Available Models", box=box.ROUNDED)
    models_table.add_column("#", style="cyan", justify="right")
    models_table.add_column("Model", style="green")
    models_table.add_column("Description", style="dim")
    keys = list(ALLOWED_MODELS)
    for number, model in enumerate(keys, start=1):
        models_table.add_row(str(number), model, ALLOWED_MODELS[model])
    console.print(models_table)

    model_choice = console.input(
        f"\n[bold]Select a model (1-{len(keys)}) or press Enter for default "
        f"({DEFAULT_MODEL}):[/] "
    ).strip()

    if model_choice.isdigit() and 1 <= int(model_choice) <= len(keys):
        return keys[int(model_choice) - 1]
    return DEFAULT_MODEL


def choose_split_mode(args, transcriber: AudioTranscriber, ffmpeg_ready: bool) -> bool:
    """Resolve CLI, prompt-driven, or interactive splitting and its dependency."""
    split_segments = args.split or transcriber.auto_split
    if transcriber.auto_split and not args.split:
        console.print("  [dim]→ Audio splitting automatically enabled for detailed transcription[/]")
    if args.no_split:
        split_segments = False
    elif not split_segments:
        split_choice = console.input(
            "\n[bold]Split audio into 20-minute segments for improved accuracy? (y/N):[/] "
        ).strip().lower()
        split_segments = split_choice in {"y", "yes"}
    if split_segments and not ffmpeg_ready:
        console.print(
            "[yellow]⚠[/] Audio splitting requires 'pydub' and ffmpeg; "
            "proceeding without splitting."
        )
        return False
    return split_segments


def show_transcription_configuration(args, selected_model: str, split_segments: bool) -> None:
    """Display the resolved batch configuration."""
    config_rows = [
        ("Model", selected_model),
        ("Audio Folder", args.audio_folder),
        ("Output Folder", args.output_folder),
        ("Split Segments", "Yes" if split_segments else "No"),
    ]
    if split_segments:
        config_rows.append(("Segment Length", f"{args.segment_minutes} minutes"))
    config_rows.append(
        ("Resume Mode", "[cyan]Yes (retry failed only)[/]" if args.resume else "No")
    )
    console.print()
    console.print(make_config_table(config_rows))
    console.print()


def build_transcriber(args) -> tuple[AudioTranscriber, str]:
    """Resolve model and prompt, then construct the non-interactive worker."""
    selected_model = args.model or select_model_interactive()
    verb = "Using" if args.model else "Selected"
    console.print(f"[green]✓[/] {verb}: [cyan]{selected_model}[/]")
    transcription_prompt, prompt_number = select_prompt_interactive(
        SCRIPT_DIR / "prompts",
        console,
        default_prompt=DEFAULT_PROMPT,
        title="Available Transcription Prompts",
        preselected=args.prompt,
    )
    return AudioTranscriber(
        model=selected_model,
        transcription_prompt=transcription_prompt,
        auto_split=prompt_number == 1,
    ), selected_model


def main() -> int:
    """Run the audio transcription CLI."""
    args = parse_args()

    # Display welcome banner
    console.print(Panel(
        "Transcribe audio and video files using Google Gemini AI",
        title="🎤 Audio Transcription using Google Gemini",
        border_style="cyan"
    ))

    try:
        if args.segment_minutes <= 0:
            raise ValueError("--segment-minutes must be positive")
        transcriber, selected_model = build_transcriber(args)
        split_segments = choose_split_mode(args, transcriber, detect_ffmpeg())
        show_transcription_configuration(args, selected_model, split_segments)

        # Transcribe all audio files (optionally split into segments)
        transcriber.transcribe_all_audio_files(
            audio_folder=args.audio_folder,
            output_folder=args.output_folder,
            split_segments=split_segments,
            segment_minutes=args.segment_minutes,
            resume_mode=args.resume
        )
        return 0

    except ValueError as e:
        console.print(f"\n[red]✗ Configuration Error:[/] {e}")
        console.print("\n[bold]To use this script, you need to set your Gemini API key:[/]")
        console.print("  1. Get your API key from: [link=https://aistudio.google.com/app/api-keys]https://aistudio.google.com/app/api-keys[/link]")
        console.print("  2. Edit the .env file in this directory")
        console.print("  3. Replace 'your-api-key-here' with your actual API key")
        console.print("  4. Save the file and run this script again")
        return 1

    except Exception as e:
        console.print(f"\n[red]✗ Unexpected error:[/] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
