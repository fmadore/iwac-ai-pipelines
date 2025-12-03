#!/usr/bin/env python3
"""
Audio Transcription Script using Google Gemini 3.0 Pro
Transcribes audio and video files from the Audio folder and saves them as text files.
Video files are automatically converted to audio before transcription.
"""

import argparse
import mimetypes
import os
import subprocess
import tempfile
import time
from pathlib import Path
from shutil import which

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file FIRST
load_dotenv()

# Script directory for relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()

# Add ffmpeg to PATH if specified in environment variables
if os.environ.get("FFMPEG_PATH"):
    ffmpeg_dir = os.path.dirname(os.environ.get("FFMPEG_PATH"))
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

# Configure pydub before any import to suppress warnings
AudioSegment = None
if os.environ.get("FFMPEG_PATH") or os.environ.get("FFPROBE_PATH"):
    # Set environment variables that pydub will check during import
    if os.environ.get("FFMPEG_PATH") and not os.environ.get("PATH_TO_FFMPEG"):
        os.environ["PATH_TO_FFMPEG"] = os.path.dirname(os.environ.get("FFMPEG_PATH"))

try:
    from pydub import AudioSegment
    from pydub.utils import which as pydub_which
    # Override pydub's ffmpeg detection
    if os.environ.get("FFMPEG_PATH"):
        AudioSegment.converter = os.environ.get("FFMPEG_PATH")
        AudioSegment.ffmpeg = os.environ.get("FFMPEG_PATH")
    if os.environ.get("FFPROBE_PATH"):
        AudioSegment.ffprobe = os.environ.get("FFPROBE_PATH")
except ImportError:  # pragma: no cover
    AudioSegment = None

class AudioTranscriber:
    def __init__(self, api_key=None, model='gemini-3-pro-preview'):
        """
        Initialize the Audio Transcriber with Gemini API.
        
        Args:
            api_key (str, optional): Gemini API key. If None, will use GEMINI_API_KEY environment variable.
            model (str, optional): Model to use. Either 'gemini-3-pro-preview' or 'gemini-2.5-flash'. Default is 'gemini-3-pro-preview'.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file or environment variables")
        
        # Store the model choice
        self.model = model
        
        # Initialize the Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        # Supported audio formats
        self.supported_formats = {
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.m4a': 'audio/mp4',
            '.flac': 'audio/flac',
            '.ogg': 'audio/ogg',
            '.webm': 'audio/webm',
            '.mp4': 'audio/mp4',
            '.aac': 'audio/aac'
        }
        
        # Supported video formats (will be converted to audio)
        self.video_formats = {
            '.mp4': 'video/mp4',
            '.mkv': 'video/x-matroska',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.wmv': 'video/x-ms-wmv',
            '.flv': 'video/x-flv',
            '.webm': 'video/webm',
            '.m4v': 'video/x-m4v',
            '.mpeg': 'video/mpeg',
            '.mpg': 'video/mpeg',
            '.3gp': 'video/3gpp'
        }
        
        # Track temporary converted audio files for cleanup
        self._temp_audio_files = []
        
        # Default transcription prompt (fallback)
        self.default_prompt = """
        Please transcribe the audio content accurately. 
        Include proper punctuation and formatting.
        If there are multiple speakers, indicate speaker changes.
        Provide a clear, readable transcription of the spoken content.
        """
        
        # Load transcription prompt from file or user selection
        self.transcription_prompt, self.auto_split = self.select_prompt()
        
        # Internal flag for ffmpeg availability
        self._ffmpeg_checked = False
        self._ffmpeg_available = False

    def ensure_ffmpeg(self):
        """Ensure ffmpeg and ffprobe are discoverable for pydub.

        Attempts to locate executables on PATH. If found, assigns them
        to AudioSegment.* attributes so pydub can use them. This runs once.
        """
        if self._ffmpeg_checked or not AudioSegment:
            return self._ffmpeg_available

        self._ffmpeg_checked = True
        # 1. Environment variables override
        ffmpeg_path = os.environ.get("FFMPEG_PATH")
        ffprobe_path = os.environ.get("FFPROBE_PATH")

        def normalize_candidate(p):
            if not p:
                return None
            p = p.strip().strip('"')
            path_obj = Path(p)
            if path_obj.is_dir():
                # Append executable name for directory input
                exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
                return str(path_obj / exe)
            return str(path_obj)

        ffmpeg_path = normalize_candidate(ffmpeg_path)
        ffprobe_path = normalize_candidate(ffprobe_path)

        # 2. PATH lookup if not provided via env
        if not ffmpeg_path:
            ffmpeg_path = which("ffmpeg") or which("ffmpeg.exe")
        if not ffprobe_path:
            ffprobe_path = which("ffprobe") or which("ffprobe.exe")

        # Common Windows install locations to probe if not on PATH
        if not ffmpeg_path:
            common_dirs = [
                r"C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
                r"C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe",
                r"C:\\ffmpeg\\bin\\ffmpeg.exe",
            ]
            for candidate in common_dirs:
                if Path(candidate).exists():
                    ffmpeg_path = candidate
                    break

        if not ffprobe_path:
            common_probe = [
                r"C:\\Program Files\\ffmpeg\\bin\\ffprobe.exe",
                r"C:\\Program Files (x86)\\ffmpeg\\bin\\ffprobe.exe",
                r"C:\\ffmpeg\\bin\\ffprobe.exe",
            ]
            for candidate in common_probe:
                if Path(candidate).exists():
                    ffprobe_path = candidate
                    break

        if ffmpeg_path:
            AudioSegment.converter = ffmpeg_path
            AudioSegment.ffmpeg = ffmpeg_path
        if ffprobe_path:
            AudioSegment.ffprobe = ffprobe_path

        self._ffmpeg_available = bool(ffmpeg_path and ffprobe_path)

        if not self._ffmpeg_available:
            print("Warning: ffmpeg/ffprobe not fully available. Splitting may fail.")
            if not ffmpeg_path:
                print("  - ffmpeg not found. Set FFMPEG_PATH in .env or install via 'winget install Gyan.FFmpeg'.")
            if not ffprobe_path:
                print("  - ffprobe not found. Set FFPROBE_PATH in .env (usually same folder as ffmpeg).")
        else:
            print(f"Detected ffmpeg: {ffmpeg_path}")
            print(f"Detected ffprobe: {ffprobe_path}")
            if os.environ.get("FFMPEG_PATH") or os.environ.get("FFPROBE_PATH"):
                print("(Using paths from environment variables)")
        return self._ffmpeg_available
    
    def is_video_file(self, file_path):
        """
        Check if a file is a video file based on its extension and MIME type.
        
        Args:
            file_path (Path): Path to the file
            
        Returns:
            bool: True if the file is a video file
        """
        extension = file_path.suffix.lower()
        
        # Check by extension first
        if extension in self.video_formats:
            return True
        
        # Use mimetypes as fallback
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and mime_type.startswith('video/'):
            return True
        
        return False
    
    def convert_video_to_audio(self, video_file_path, output_format='mp3'):
        """
        Convert a video file to audio using ffmpeg.
        
        Args:
            video_file_path (Path): Path to the video file
            output_format (str): Output audio format (default: mp3)
            
        Returns:
            Path: Path to the converted audio file, or None if conversion failed
        """
        if not self.ensure_ffmpeg():
            print(f"Cannot convert video '{video_file_path.name}': ffmpeg not available")
            return None
        
        try:
            # Create temp directory for converted audio files
            temp_dir = SCRIPT_DIR / "temp_converted_audio"
            temp_dir.mkdir(exist_ok=True)
            
            # Create output filename
            output_filename = f"{video_file_path.stem}_audio.{output_format}"
            output_path = temp_dir / output_filename
            
            # Get ffmpeg path
            ffmpeg_path = getattr(AudioSegment, 'converter', None) or which("ffmpeg") or "ffmpeg"
            
            print(f"Converting video to audio: {video_file_path.name} -> {output_filename}")
            
            # Build ffmpeg command
            # -i: input file
            # -vn: no video
            # -acodec: audio codec (libmp3lame for mp3, copy for m4a)
            # -ab: audio bitrate
            # -y: overwrite output file
            if output_format == 'mp3':
                cmd = [
                    ffmpeg_path,
                    '-i', str(video_file_path),
                    '-vn',
                    '-acodec', 'libmp3lame',
                    '-ab', '192k',
                    '-ar', '44100',
                    '-y',
                    str(output_path)
                ]
            else:
                # For other formats, let ffmpeg choose the codec
                cmd = [
                    ffmpeg_path,
                    '-i', str(video_file_path),
                    '-vn',
                    '-ab', '192k',
                    '-y',
                    str(output_path)
                ]
            
            # Run ffmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout for very long videos
            )
            
            if result.returncode != 0:
                print(f"Error converting video: {result.stderr}")
                return None
            
            if output_path.exists():
                print(f"Video converted successfully: {output_path.name}")
                self._temp_audio_files.append(output_path)
                return output_path
            else:
                print(f"Conversion failed: output file not created")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"Video conversion timed out for {video_file_path.name}")
            return None
        except Exception as e:
            print(f"Error converting video {video_file_path.name}: {e}")
            return None
    
    def cleanup_converted_audio(self):
        """
        Clean up temporary converted audio files.
        """
        for temp_file in self._temp_audio_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    print(f"Cleaned up converted audio: {temp_file.name}")
            except Exception as e:
                print(f"Warning: Could not clean up {temp_file}: {e}")
        
        self._temp_audio_files.clear()
        
        # Clean up temp directory if empty
        temp_dir = SCRIPT_DIR / "temp_converted_audio"
        if temp_dir.exists() and temp_dir.is_dir():
            try:
                temp_dir.rmdir()
            except OSError:
                pass  # Directory not empty
    
    def get_audio_files(self, audio_folder="Audio"):
        """
        Get all supported audio and video files from the specified folder.
        Video files will be converted to audio during transcription.
        
        Args:
            audio_folder (str): Path to the audio folder (relative to script directory)
            
        Returns:
            list: List of tuples (file_path, is_video)
        """
        # Resolve audio folder relative to script directory
        audio_path = SCRIPT_DIR / audio_folder
        if not audio_path.exists():
            print(f"Audio folder '{audio_path}' not found!")
            return []
        
        media_files = []
        for file_path in audio_path.iterdir():
            if file_path.is_file():
                extension = file_path.suffix.lower()
                if extension in self.supported_formats:
                    media_files.append((file_path, False))  # Audio file
                elif extension in self.video_formats:
                    media_files.append((file_path, True))   # Video file
        
        return sorted(media_files, key=lambda x: x[0])
    
    def prepare_audio_for_api(self, audio_file_path):
        """
        Prepare audio file for Gemini API by reading it as bytes.
        
        Args:
            audio_file_path (Path): Path to the audio file
            
        Returns:
            tuple: (audio_bytes, mime_type)
        """
        try:
            with open(audio_file_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            
            # Get MIME type
            file_extension = audio_file_path.suffix.lower()
            mime_type = self.supported_formats.get(file_extension)
            
            if not mime_type:
                # Fallback to mimetypes module
                mime_type, _ = mimetypes.guess_type(str(audio_file_path))
            
            return audio_bytes, mime_type
        
        except Exception as e:
            print(f"Error reading audio file {audio_file_path}: {e}")
            return None, None
    
    def transcribe_audio(self, audio_file_path, custom_prompt=None, max_retries=3):
        """
        Transcribe a single audio file using Gemini 3.0 Pro with retry mechanism.
        
        Args:
            audio_file_path (Path): Path to the audio file
            custom_prompt (str, optional): Custom transcription prompt
            max_retries (int): Maximum number of retry attempts (default: 3)
            
        Returns:
            str: Transcribed text or None if error
        """
        print(f"Transcribing: {audio_file_path.name}")
        
        # Prepare audio data
        audio_bytes, mime_type = self.prepare_audio_for_api(audio_file_path)
        if not audio_bytes or not mime_type:
            return None
        
        # Create audio part for the API (do this once, outside retry loop)
        audio_part = types.Part.from_bytes(
            data=audio_bytes,
            mime_type=mime_type
        )
        
        # Use custom prompt or default
        prompt = custom_prompt or self.transcription_prompt
        
        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(max_retries):
            try:
                # Generate transcription using the selected model
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[prompt, audio_part],
                    config=types.GenerateContentConfig(
                        temperature=0.1,  # Low temperature for more consistent transcription
                        max_output_tokens=65536,
                    )
                )
                
                # Handle case where response.text is None (e.g., content blocked, empty response)
                if response.text is None:
                    print(f"Warning: No transcription returned for {audio_file_path.name} (response was empty)")
                    # Don't retry for empty responses - likely a content issue, not transient
                    return None
                
                return response.text.strip()
            
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 2^attempt seconds (2s, 4s, 8s...)
                    wait_time = 2 ** (attempt + 1)
                    print(f"Error transcribing {audio_file_path.name}: {e}")
                    print(f"Retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"Error transcribing {audio_file_path.name} after {max_retries} attempts: {last_error}")
        
        return None
    
    def save_transcription(self, transcription, audio_file_path, output_folder="Transcriptions"):
        """
        Save transcription to a text file.
        
        Args:
            transcription (str): Transcribed text
            audio_file_path (Path): Original audio file path
            output_folder (str): Output folder for transcriptions (relative to script directory)
        """
        # Create output folder relative to script directory if it doesn't exist
        output_path = SCRIPT_DIR / output_folder
        output_path.mkdir(exist_ok=True)
        
        # Create output filename
        output_filename = audio_file_path.stem + "_transcription.txt"
        output_file_path = output_path / output_filename
        
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                # Write header with metadata
                f.write(f"Transcription of: {audio_file_path.name}\n")
                f.write(f"Generated using: Google {self.model}\n")
                f.write("=" * 50 + "\n\n")
                f.write(transcription)
            
            print(f"Transcription saved: {output_file_path}")
            return output_file_path
        
        except Exception as e:
            print(f"Error saving transcription: {e}")
            return None
    
    def split_audio_file(self, audio_file_path, segment_minutes=10, temp_root="temp_segments"):
        """
        Split an audio file into fixed-length segments.

        Args:
            audio_file_path (Path): Path to original audio file
            segment_minutes (int): Length of each segment in minutes
            temp_root (str): Root temp directory for segments (relative to script directory)

        Returns:
            list[Path]: List of segment file paths (or [original] if splitting failed)
        """
        if not AudioSegment:
            print("pydub not installed; skipping splitting for", audio_file_path.name)
            return [audio_file_path]

        try:
            # Ensure ffmpeg tools available
            if not self.ensure_ffmpeg():
                print("ffmpeg not configured; skipping splitting for", audio_file_path.name)
                return [audio_file_path]

            segment_ms = segment_minutes * 60 * 1000
            audio = AudioSegment.from_file(audio_file_path)
            if len(audio) <= segment_ms:
                # No need to split
                return [audio_file_path]

            # Prepare temp directory relative to script directory
            parent_dir = SCRIPT_DIR / temp_root / audio_file_path.stem
            parent_dir.mkdir(parents=True, exist_ok=True)

            segments = []
            for i, start in enumerate(range(0, len(audio), segment_ms), start=1):
                end = min(start + segment_ms, len(audio))
                chunk = audio[start:end]
                segment_filename = f"segment_{i:02d}{audio_file_path.suffix}"
                segment_path = parent_dir / segment_filename
                
                # Map file extensions to ffmpeg format names
                file_ext = audio_file_path.suffix.lstrip('.').lower()
                format_mapping = {
                    'm4a': 'mp4',
                    'mp4': 'mp4',
                    'mp3': 'mp3',
                    'wav': 'wav',
                    'flac': 'flac',
                    'ogg': 'ogg',
                    'webm': 'webm',
                    'aac': 'aac'
                }
                export_format = format_mapping.get(file_ext, file_ext)
                
                chunk.export(segment_path, format=export_format)
                segments.append(segment_path)

            print(f"Split '{audio_file_path.name}' into {len(segments)} segment(s) of up to {segment_minutes} minutes each.")
            return segments
        except Exception as e:
            print(f"Error splitting {audio_file_path.name}: {e}. Proceeding without splitting.")
            return [audio_file_path]

    def cleanup_temp_segments(self, original_audio_file, segment_paths, temp_root="temp_segments"):
        """
        Clean up temporary segment files and directories after successful transcription.
        
        Args:
            original_audio_file (Path): Path to the original audio file
            segment_paths (list[Path]): List of segment file paths that were created
            temp_root (str): Root temp directory for segments
        """
        try:
            # Only clean up if we actually created segments (not just returned original)
            if len(segment_paths) == 1 and segment_paths[0] == original_audio_file:
                return
            
            # Remove individual segment files
            for segment_path in segment_paths:
                if segment_path.exists() and segment_path != original_audio_file:
                    segment_path.unlink()
                    print(f"Cleaned up segment: {segment_path.name}")
            
            # Remove the segment directory if it's empty
            segment_dir = SCRIPT_DIR / temp_root / original_audio_file.stem
            if segment_dir.exists() and segment_dir.is_dir():
                try:
                    segment_dir.rmdir()  # Only removes if empty
                    print(f"Cleaned up directory: {segment_dir}")
                except OSError:
                    # Directory not empty, leave it
                    pass
            
            # Clean up root temp directory if it's empty
            temp_root_path = SCRIPT_DIR / temp_root
            if temp_root_path.exists() and temp_root_path.is_dir():
                try:
                    temp_root_path.rmdir()  # Only removes if empty
                except OSError:
                    # Directory not empty, leave it
                    pass
                    
        except Exception as e:
            print(f"Warning: Could not clean up temporary files: {e}")

    def transcribe_all_audio_files(self, audio_folder="Audio", output_folder="Transcriptions", custom_prompt=None, split_segments=False, segment_minutes=10):
        """
        Transcribe all audio and video files in the specified folder.
        Video files are automatically converted to audio before transcription.
        
        Args:
            audio_folder (str): Path to the audio folder
            output_folder (str): Output folder for transcriptions
            custom_prompt (str, optional): Custom transcription prompt
            split_segments (bool): Whether to split audio into segments
            segment_minutes (int): Length of each segment in minutes
        """
        media_files = self.get_audio_files(audio_folder)
        
        if not media_files:
            print("No supported audio or video files found in the Audio folder.")
            print(f"Supported audio formats: {', '.join(self.supported_formats.keys())}")
            print(f"Supported video formats: {', '.join(self.video_formats.keys())}")
            return
        
        audio_count = sum(1 for _, is_video in media_files if not is_video)
        video_count = sum(1 for _, is_video in media_files if is_video)
        
        print(f"Found {len(media_files)} file(s) to transcribe:")
        print(f"  - {audio_count} audio file(s)")
        print(f"  - {video_count} video file(s) (will be converted to audio)")
        print()
        for file_path, is_video in media_files:
            file_type = "[VIDEO]" if is_video else "[AUDIO]"
            print(f"  {file_type} {file_path.name}")
        
        print("\nStarting transcription process...\n")
        
        successful_transcriptions = 0
        failed_transcriptions = 0
        
        try:
            for original_file, is_video in media_files:
                try:
                    # Convert video to audio if needed
                    if is_video:
                        audio_file = self.convert_video_to_audio(original_file)
                        if not audio_file:
                            print(f"Skipping {original_file.name}: video conversion failed")
                            failed_transcriptions += 1
                            continue
                    else:
                        audio_file = original_file
                    
                    if split_segments:
                        # Split into segments (or return original if no splitting applied)
                        segment_paths = self.split_audio_file(audio_file, segment_minutes=segment_minutes)
                        combined_transcription_parts = []
                        all_segments_successful = True
                        
                        for idx, segment_path in enumerate(segment_paths, start=1):
                            print(f"Processing segment {idx}/{len(segment_paths)}: {segment_path.name}")
                            seg_transcription = self.transcribe_audio(segment_path, custom_prompt)
                            if seg_transcription:
                                header = f"[Segment {idx}]" if len(segment_paths) > 1 else ""
                                combined_transcription_parts.append(f"{header}\n{seg_transcription}\n")
                            else:
                                combined_transcription_parts.append(f"[Segment {idx}] TRANSCRIPTION FAILED")
                                all_segments_successful = False

                        # Combine
                        transcription = "\n".join(combined_transcription_parts).strip()
                        
                        # Clean up temporary segments if all were successful and we actually split the file
                        if all_segments_successful and len(segment_paths) > 1:
                            self.cleanup_temp_segments(audio_file, segment_paths)
                    else:
                        transcription = self.transcribe_audio(audio_file, custom_prompt)

                    if transcription:
                        # Use original file name for output (not the converted audio name)
                        output_file = self.save_transcription(transcription, original_file, output_folder)
                        if output_file:
                            successful_transcriptions += 1
                        else:
                            failed_transcriptions += 1
                    else:
                        failed_transcriptions += 1

                except Exception as e:
                    print(f"Unexpected error processing {original_file.name}: {e}")
                    failed_transcriptions += 1

                print()  # Add spacing between files
        
        finally:
            # Clean up converted audio files
            self.cleanup_converted_audio()
        
        # Summary
        print("=" * 50)
        print("TRANSCRIPTION SUMMARY")
        print("=" * 50)
        print(f"Total files processed: {len(media_files)}")
        print(f"Successful transcriptions: {successful_transcriptions}")
        print(f"Failed transcriptions: {failed_transcriptions}")
        
        if successful_transcriptions > 0:
            print(f"\nTranscriptions saved in the '{output_folder}' folder.")
    
    def get_available_prompts(self, prompts_folder="prompts"):
        """
        Get all available prompt files from the prompts folder.
        
        Args:
            prompts_folder (str): Path to the prompts folder (relative to script directory)
            
        Returns:
            list: List of tuples (number, description, filepath)
        """
        # Resolve prompts folder relative to script directory
        prompts_path = SCRIPT_DIR / prompts_folder
        if not prompts_path.exists():
            print(f"Warning: Prompts folder '{prompts_path}' not found.")
            return []
        
        prompt_files = []
        for file_path in prompts_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() == '.md':
                # Extract description from filename (remove number prefix and extension)
                name_part = file_path.stem
                if '_' in name_part:
                    number_part, description = name_part.split('_', 1)
                    try:
                        number = int(number_part)
                        description = description.replace('_', ' ').title()
                        prompt_files.append((number, description, file_path))
                    except ValueError:
                        # If no number prefix, use filename as description
                        description = name_part.replace('_', ' ').title()
                        prompt_files.append((0, description, file_path))
                else:
                    description = name_part.replace('_', ' ').title()
                    prompt_files.append((0, description, file_path))
        
        return sorted(prompt_files, key=lambda x: x[0])
    
    def display_prompt_menu(self, available_prompts):
        """
        Display the prompt selection menu.
        
        Args:
            available_prompts (list): List of available prompts
        """
        print("\n" + "=" * 50)
        print("PROMPT SELECTION")
        print("=" * 50)
        print("Available transcription prompts:")
        print()
        
        for number, description, _ in available_prompts:
            if number > 0:
                print(f"{number}. {description}")
            else:
                print(f"   {description}")
        
        print()
    
    def load_prompt_content(self, prompt_file_path):
        """
        Load prompt content from a markdown file.
        
        Args:
            prompt_file_path (Path): Path to the prompt file
            
        Returns:
            str: The prompt content or default prompt if error
        """
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract content after the first markdown header
            lines = content.split('\n')
            prompt_lines = []
            found_header = False
            
            for line in lines:
                if line.startswith('# ') and not found_header:
                    found_header = True
                    continue
                elif found_header and line.strip():
                    prompt_lines.append(line)
            
            if prompt_lines:
                return '\n'.join(prompt_lines).strip()
            else:
                return content.strip()
                
        except Exception as e:
            print(f"Error loading prompt from '{prompt_file_path}': {e}")
            return self.default_prompt
    
    def select_prompt(self, prompts_folder="prompts"):
        """
        Allow user to select a transcription prompt.
        
        Args:
            prompts_folder (str): Path to the prompts folder
            
        Returns:
            tuple: (prompt_content, auto_split) where auto_split is True if splitting should be enabled automatically
        """
        available_prompts = self.get_available_prompts(prompts_folder)
        
        if not available_prompts:
            print("No prompt files found. Using default prompt.")
            return self.default_prompt, False
        
        # Display menu
        self.display_prompt_menu(available_prompts)
        
        # Get user selection
        numbered_prompts = [(num, desc, path) for num, desc, path in available_prompts if num > 0]
        
        if not numbered_prompts:
            print("No numbered prompts found. Using default prompt.")
            return self.default_prompt, False
        
        while True:
            try:
                choice = input(f"Select a prompt (1-{len(numbered_prompts)}) or press Enter for default: ").strip()
                
                if not choice:
                    print("Using default prompt.")
                    return self.default_prompt, False
                
                choice_num = int(choice)
                
                # Find the prompt with the selected number
                selected_prompt = None
                for num, desc, path in numbered_prompts:
                    if num == choice_num:
                        selected_prompt = (num, desc, path)
                        break
                
                if selected_prompt:
                    prompt_num, description, prompt_path = selected_prompt
                    print(f"Selected: {description}")
                    # Auto-enable splitting for prompt #1 (full audio transcription)
                    auto_split = (prompt_num == 1)
                    if auto_split:
                        print("  → Audio splitting automatically enabled for detailed transcription")
                    return self.load_prompt_content(prompt_path), auto_split
                else:
                    print(f"Invalid choice. Please select a number between 1 and {len(numbered_prompts)}.")
                    
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nUsing default prompt.")
                return self.default_prompt, False

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
        choices=["gemini-3-pro-preview", "gemini-2.5-flash"],
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
        help="Split audio into 10-minute segments for improved accuracy"
    )
    parser.add_argument(
        "--segment-minutes",
        type=int,
        default=10,
        help="Segment length in minutes when splitting (default: 10)"
    )
    return parser.parse_args()


def select_model_interactive():
    """
    Interactively select a model if not provided via CLI.
    """
    print("\nAvailable models:")
    print("1. gemini-3-pro-preview (Higher quality, slower)")
    print("2. gemini-2.5-flash (Faster, good quality)")
    model_choice = input("\nSelect a model (1 or 2) or press Enter for default (gemini-3-pro-preview): ").strip()
    
    if model_choice == '2':
        return 'gemini-2.5-flash'
    return 'gemini-3-pro-preview'


def main():
    """
    Main function to run the audio transcription script.
    """
    args = parse_args()
    
    print("Audio Transcription using Google Gemini")
    print("=" * 50)
    
    try:
        # Select model via CLI or interactive
        if args.model:
            selected_model = args.model
            print(f"\nUsing model: {selected_model}")
        else:
            selected_model = select_model_interactive()
            print(f"Selected: {selected_model}")
        
        # Initialize transcriber (prompt chosen interactively)
        transcriber = AudioTranscriber(model=selected_model)

        # Determine if splitting should be used
        if args.split:
            # CLI flag takes precedence
            split_segments = True
            if not AudioSegment:
                print("Warning: Audio splitting requires 'pydub'. Install via 'pip install pydub' and ensure ffmpeg is available. Proceeding without splitting.")
                split_segments = False
        elif transcriber.auto_split:
            # Auto-split is enabled for this prompt
            split_segments = True
            if not AudioSegment:
                print("Warning: Audio splitting requires 'pydub'. Install via 'pip install pydub' and ensure ffmpeg is available. Proceeding without splitting.")
                split_segments = False
        else:
            # Ask user if they want to split audio into 10-minute segments
            split_choice = input("\nSplit audio into 10-minute segments for improved accuracy? (y/N): ").strip().lower()
            split_segments = split_choice in {"y", "yes"}
            if split_segments and not AudioSegment:
                print("You chose to split audio, but 'pydub' is not installed. Install via 'pip install pydub' and ensure ffmpeg is available on PATH. Proceeding without splitting.")
                split_segments = False

        # Transcribe all audio files (optionally split into segments)
        transcriber.transcribe_all_audio_files(
            audio_folder=args.audio_folder,
            output_folder=args.output_folder,
            split_segments=split_segments,
            segment_minutes=args.segment_minutes
        )
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nTo use this script, you need to set your Gemini API key:")
        print("1. Get your API key from: https://aistudio.google.com/app/api-keys")
        print("2. Edit the .env file in this directory")
        print("3. Replace 'your-api-key-here' with your actual API key")
        print("4. Save the file and run this script again")
        
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
