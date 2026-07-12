#!/usr/bin/env python3
"""
Video Processing Script using Google Gemini
Processes video files from the video folder and saves summaries/transcriptions as text files.

Supports:
- Video summarization
- Full transcription with periodic visual descriptions
- Both Gemini Pro and Gemini Flash models
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

# Add repo root to path for shared imports
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent))
from common.gemini_utils import (
    INLINE_REQUEST_LIMIT_BYTES,
    build_generation_config,
    delete_uploaded_file,
    extract_text_from_response,
    upload_and_wait_active,
)
from common.ffmpeg_utils import VIDEO_FORMATS, get_mime_type
from common.prompt_loader import select_prompt_interactive
from common.rate_limiter import RateLimiter, QuotaExhaustedError, is_quota_exhausted
from common.retry import retry_with_backoff

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Initialize rich console
console = Console()

# Load environment variables from .env file FIRST
load_dotenv()

# Script directory for relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()

# Default prompt (fallback when no prompt file is selected)
DEFAULT_PROMPT = """
        Please analyze this video and provide a comprehensive summary of its content.
        Include information about what is shown visually and any spoken content.
        """

# Transient Gemini API errors worth retrying with backoff
RETRYABLE_API_CODES = (429, 500, 503)


def _is_retryable_api_error(exc: BaseException) -> bool:
    """Retry predicate: transient Gemini API errors only."""
    return isinstance(exc, genai_errors.APIError) and getattr(exc, "code", 0) in RETRYABLE_API_CODES


class VideoProcessor:
    def __init__(
        self,
        api_key=None,
        model="gemini-pro-latest",
        requests_per_minute: Optional[int] = None,
        processing_prompt: Optional[str] = None,
    ):
        """
        Initialize the Video Processor with Gemini API.

        Args:
            api_key (str, optional): Gemini API key. If None, will use GEMINI_API_KEY environment variable.
            model (str, optional): Model to use. Either 'gemini-pro-latest' or 'gemini-flash-latest'.
                                   Default is 'gemini-pro-latest'.
            requests_per_minute: Optional RPM limit for proactive throttling (None = no throttling)
            processing_prompt: The processing prompt to use (selected in ``main()``);
                falls back to ``DEFAULT_PROMPT``.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file or environment variables")

        # Store the model choice
        self.model = model

        # Initialize the Gemini client
        self.client = genai.Client(api_key=self.api_key)

        # Rate limiter for proactive throttling
        self.rate_limiter = RateLimiter(requests_per_minute, logger=logging.getLogger(__name__))

        # Prompt is injected by main() so constructing the class never blocks on stdin.
        self.processing_prompt = processing_prompt or DEFAULT_PROMPT

    def get_video_files(self, video_folder="video"):
        """
        Get all supported video files from the specified folder.

        Args:
            video_folder (str): Path to the video folder (relative to script directory)

        Returns:
            list: List of video file paths
        """
        # Resolve video folder relative to script directory
        video_path = SCRIPT_DIR / video_folder
        if not video_path.exists():
            console.print(f"[red]✗[/] Video folder '[cyan]{video_path}[/]' not found!")
            return []

        video_files = []
        for file_path in video_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in VIDEO_FORMATS:
                video_files.append(file_path)

        return sorted(video_files)

    def _generate_from_part(self, media_part) -> Optional[str]:
        """Generate text from one media part (inline bytes or an uploaded file).

        Single generation path shared by the inline and Files API transports.
        Transient API errors (429/500/503) are retried with exponential
        backoff; quota exhaustion always propagates as ``QuotaExhaustedError``.

        Returns:
            The generated text, or ``None`` on failure.
        """
        def _call() -> str:
            self.rate_limiter.wait()
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[media_part, self.processing_prompt],
                    config=build_generation_config(
                        self.model,
                        temperature=0.2,
                        max_output_tokens=65536,
                    ),
                )
            except genai_errors.APIError as e:
                if is_quota_exhausted(e):
                    raise QuotaExhaustedError(str(e)) from e
                raise
            # Safely skips thought-parts and empty candidates (bare
            # response.text.strip() breaks on both).
            return extract_text_from_response(response)

        try:
            text = retry_with_backoff(
                max_retries=3,
                base_delay=5.0,
                exceptions=(genai_errors.APIError,),
                is_retryable=_is_retryable_api_error,
            )(_call)()
        except QuotaExhaustedError:
            raise
        except Exception as e:
            console.print(f"  [red]✗[/] Error generating content from video: {e}")
            return None

        if not text:
            console.print("  [red]✗[/] Empty response from Gemini")
            return None
        return text

    def _upload_video(self, video_file_path) -> Optional[object]:
        """Upload a video via the Files API and wait until it is ACTIVE.

        Delegates the upload/poll loop (with timeout and FAILED handling) to
        ``common.gemini_utils.upload_and_wait_active``.

        Returns:
            The ACTIVE file object, or ``None`` on failure.
        """
        console.print("  [cyan]☁[/] Uploading video to Gemini Files API...")
        try:
            uploaded_file = upload_and_wait_active(
                self.client,
                video_file_path,
                poll_interval=5.0,
            )
        except genai_errors.APIError as e:
            if is_quota_exhausted(e):
                raise QuotaExhaustedError(str(e)) from e
            console.print(f"  [red]✗[/] Error uploading video file: {e}")
            return None
        except (RuntimeError, TimeoutError) as e:
            console.print(f"  [red]✗[/] Error uploading video file: {e}")
            return None

        console.print("  [green]✓[/] Video ready for processing.")
        return uploaded_file

    def process_video(self, video_file_path):
        """
        Process a single video file using the appropriate method based on file size.

        Small files are sent inline; anything above the 20 MB request cap is
        uploaded via the Files API and always cleaned up afterwards (uploads
        are potentially GBs; never leak them on failure paths).

        Args:
            video_file_path (Path): Path to the video file

        Returns:
            str: Generated text or None if error
        """
        console.print(f"[cyan]🎬[/] Processing: [bold]{video_file_path.name}[/]")

        # Check file size
        file_size = video_file_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        console.print(f"  [dim]File size: {file_size_mb:.1f} MB[/]")

        if file_size <= INLINE_REQUEST_LIMIT_BYTES:
            # Small file — send bytes inline
            mime_type = get_mime_type(video_file_path)
            if not mime_type:
                console.print(f"  [red]✗[/] Unsupported video format: {video_file_path.suffix}")
                return None

            console.print("  [cyan]→[/] Sending to Gemini API (inline mode)...")
            with open(video_file_path, "rb") as f:
                video_bytes = f.read()
            media_part = types.Part.from_bytes(data=video_bytes, mime_type=mime_type)
            return self._generate_from_part(media_part)

        # Large file — upload first
        console.print(
            f"  [dim]File exceeds {INLINE_REQUEST_LIMIT_BYTES / (1024 * 1024):.0f} MB, "
            f"using Files API upload...[/]"
        )
        uploaded_file = self._upload_video(video_file_path)
        if not uploaded_file:
            return None

        try:
            console.print("  [cyan]→[/] Generating content from video...")
            return self._generate_from_part(uploaded_file)
        finally:
            delete_uploaded_file(self.client, uploaded_file)
            console.print("  [dim]🧹 Cleaned up uploaded file.[/]")

    def save_output(self, output_text, video_file_path, output_folder="output"):
        """
        Save processed output to a text file.

        Args:
            output_text (str): Generated text
            video_file_path (Path): Original video file path
            output_folder (str): Output folder for results (relative to script directory)
        """
        # Create output folder relative to script directory if it doesn't exist
        output_path = SCRIPT_DIR / output_folder
        output_path.mkdir(exist_ok=True)

        # Create output filename
        output_filename = video_file_path.stem + "_processed.txt"
        output_file_path = output_path / output_filename

        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                # Write header with metadata
                f.write(f"Video Processing Output: {video_file_path.name}\n")
                f.write(f"Generated using: Google {self.model}\n")
                f.write("=" * 60 + "\n\n")
                f.write(output_text)

            console.print(f"  [green]✓[/] Output saved: [cyan]{output_file_path}[/]")
            return output_file_path

        except Exception as e:
            console.print(f"  [red]✗[/] Error saving output: {e}")
            return None

    def process_all_video_files(self, video_folder="video", output_folder="output"):
        """
        Process all video files in the specified folder.

        Args:
            video_folder (str): Path to the video folder
            output_folder (str): Output folder for results
        """
        video_files = self.get_video_files(video_folder)

        if not video_files:
            console.print("[yellow]⚠[/] No supported video files found in the video folder.")
            console.print(f"[dim]Supported formats: {', '.join(VIDEO_FORMATS.keys())}[/]")
            return

        # Display files table
        files_table = Table(title="📁 Videos to Process", box=box.ROUNDED)
        files_table.add_column("Filename", style="green")
        files_table.add_column("Size", justify="right", style="dim")
        for file_path in video_files:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            files_table.add_row(file_path.name, f"{size_mb:.1f} MB")
        console.print(files_table)

        console.print(f"\n[bold]Summary:[/] [cyan]{len(video_files)}[/] file(s) to process")
        console.print()
        console.rule("[bold]Starting Video Processing", style="cyan")
        console.print()

        successful_processes = 0
        failed_processes = 0

        for video_file in video_files:
            try:
                console.rule(f"[dim]{video_file.name}[/]", style="dim")
                output_text = self.process_video(video_file)

                if output_text:
                    output_file = self.save_output(output_text, video_file, output_folder)
                    if output_file:
                        successful_processes += 1
                    else:
                        failed_processes += 1
                else:
                    failed_processes += 1

            except QuotaExhaustedError:
                console.print("\n[red bold]API quota exhausted — stopping all processing.[/]")
                console.print("[red]Partial results (if any) have been saved.[/]")
                console.print("[red]Wait for your quota to reset or upgrade your plan.[/]")
                break

            except Exception as e:
                console.print(f"[red]✗[/] Unexpected error processing [cyan]{video_file.name}[/]: {e}")
                failed_processes += 1

            console.print()  # Add spacing between files

        # Summary
        console.print()
        console.rule("[bold]Processing Summary", style="cyan")

        summary_table = Table(title="📊 Results", box=box.ROUNDED)
        summary_table.add_column("Metric", style="dim")
        summary_table.add_column("Value", style="green")
        summary_table.add_row("Total files processed", str(len(video_files)))
        summary_table.add_row("Successful", f"[green]{successful_processes}[/]")
        summary_table.add_row("Failed", f"[red]{failed_processes}[/]" if failed_processes > 0 else "0")
        summary_table.add_row("Output folder", output_folder)
        console.print(summary_table)

        if successful_processes > 0:
            console.print(f"\n[green]✓[/] Output saved in the '[cyan]{output_folder}[/]' folder.")


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Video Processing using Google Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python AI_video_summary.py
  python AI_video_summary.py --model gemini-flash-latest
  python AI_video_summary.py --video-folder my_videos --output-folder results
  python AI_video_summary.py --rpm 5
        """
    )
    parser.add_argument(
        "--model",
        choices=["gemini-pro-latest", "gemini-flash-latest"],
        default=None,
        help="Model to use for processing (default: interactive selection)"
    )
    parser.add_argument(
        "--video-folder", "--video-dir",
        dest="video_folder",
        default="video",
        help="Folder containing video files (default: video)"
    )
    parser.add_argument(
        "--output-folder",
        default="output",
        help="Folder for output files (default: output)"
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=None,
        help="Rate limit: maximum requests per minute (default: no limit)"
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
    models_table.add_row("1", "gemini-pro-latest", "Higher quality, best for detailed transcription")
    models_table.add_row("2", "gemini-flash-latest", "Faster, good for summaries")
    console.print(models_table)

    model_choice = console.input(
        "\n[bold]Select a model (1 or 2) or press Enter for default (gemini-pro-latest):[/] "
    ).strip()

    if model_choice == '2':
        return 'gemini-flash-latest'
    return 'gemini-pro-latest'


def main():
    """
    Main function to run the video processing script.
    """
    args = parse_args()

    # Display welcome banner
    console.print(Panel(
        "Summarize or transcribe video files using Google Gemini AI",
        title="🎬 Video Processing using Google Gemini",
        border_style="cyan"
    ))

    try:
        # Select model via CLI or interactive
        if args.model:
            selected_model = args.model
            console.print(f"\n[green]✓[/] Using model: [cyan]{selected_model}[/]")
        else:
            selected_model = select_model_interactive()
            console.print(f"[green]✓[/] Selected: [cyan]{selected_model}[/]")

        # Select the processing prompt up front, then inject it into the
        # processor (constructing the class never blocks on stdin).
        processing_prompt, _prompt_number = select_prompt_interactive(
            SCRIPT_DIR / "prompts",
            console,
            default_prompt=DEFAULT_PROMPT,
            title="Available Processing Modes",
        )

        # Display configuration
        console.print()
        config_table = Table(title="⚙️ Configuration", box=box.ROUNDED)
        config_table.add_column("Setting", style="dim")
        config_table.add_column("Value", style="green")
        config_table.add_row("Model", selected_model)
        config_table.add_row("Video Folder", args.video_folder)
        config_table.add_row("Output Folder", args.output_folder)
        config_table.add_row("Rate Limit", f"{args.rpm} RPM" if args.rpm else "None")
        console.print(config_table)
        console.print()

        # Initialize processor and run
        processor = VideoProcessor(
            model=selected_model,
            requests_per_minute=args.rpm,
            processing_prompt=processing_prompt,
        )

        processor.process_all_video_files(
            video_folder=args.video_folder,
            output_folder=args.output_folder
        )

    except ValueError as e:
        console.print(f"\n[red]✗ Configuration Error:[/] {e}")
        console.print("\n[bold]To use this script, you need to set your Gemini API key:[/]")
        console.print("  1. Get your API key from: [link=https://aistudio.google.com/app/api-keys]https://aistudio.google.com/app/api-keys[/link]")
        console.print("  2. Create or edit a .env file in this directory")
        console.print("  3. Add: GEMINI_API_KEY=your-api-key-here")
        console.print("  4. Save the file and run this script again")

    except Exception:
        console.print("\n[red]✗ Unexpected error:[/]")
        console.print_exception()


if __name__ == "__main__":
    main()
