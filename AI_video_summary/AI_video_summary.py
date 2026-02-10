#!/usr/bin/env python3
"""
Video Processing Script using Google Gemini
Processes video files from the video folder and saves summaries/transcriptions as text files.

Supports:
- Video summarization
- Full transcription with periodic visual descriptions
- Both Gemini 3.0 Pro Preview and Gemini 3 Flash models
"""

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

# Add repo root to path for shared imports
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent))
from common.gemini_utils import SAFETY_SETTINGS_NONE, get_thinking_level
from common.ffmpeg_utils import VIDEO_FORMATS, get_mime_type
from common.rate_limiter import RateLimiter, QuotaExhaustedError, is_quota_exhausted

# Load environment variables from .env file FIRST
load_dotenv()

# Script directory for relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()


class VideoProcessor:
    def __init__(self, api_key=None, model='gemini-3-pro-preview', requests_per_minute: Optional[int] = None):
        """
        Initialize the Video Processor with Gemini API.

        Args:
            api_key (str, optional): Gemini API key. If None, will use GEMINI_API_KEY environment variable.
            model (str, optional): Model to use. Either 'gemini-3-pro-preview' or 'gemini-3-flash-preview'.
                                   Default is 'gemini-3-pro-preview'.
            requests_per_minute: Optional RPM limit for proactive throttling (None = no throttling)
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
        
        # Default prompt (fallback)
        self.default_prompt = """
        Please analyze this video and provide a comprehensive summary of its content.
        Include information about what is shown visually and any spoken content.
        """
        
        # Load prompt from file or user selection
        self.processing_prompt = self.select_prompt()
        
        # Maximum file size for inline upload (20MB)
        self.max_inline_size = 20 * 1024 * 1024

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
            print(f"Video folder '{video_path}' not found!")
            return []
        
        video_files = []
        for file_path in video_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in VIDEO_FORMATS:
                video_files.append(file_path)
        
        return sorted(video_files)
    
    def upload_video_file(self, video_file_path):
        """
        Upload a video file to the Gemini Files API.
        
        Args:
            video_file_path (Path): Path to the video file
            
        Returns:
            File object or None if error
        """
        try:
            print(f"  Uploading video to Gemini Files API...")
            uploaded_file = self.client.files.upload(file=str(video_file_path))
            
            # Poll until the video is processed (state becomes ACTIVE)
            print(f"  Waiting for video processing...")
            while not uploaded_file.state or uploaded_file.state.name != "ACTIVE":
                print(f"    Processing... (state: {uploaded_file.state})")
                time.sleep(5)
                uploaded_file = self.client.files.get(name=uploaded_file.name)
            
            print(f"  Video ready for processing.")
            return uploaded_file
            
        except Exception as e:
            print(f"  Error uploading video file: {e}")
            return None
    
    def process_video_inline(self, video_file_path, max_retries=3):
        """
        Process a small video file by sending bytes inline.

        Args:
            video_file_path (Path): Path to the video file
            max_retries (int): Maximum retry attempts for transient errors

        Returns:
            str: Generated text or None if error
        """
        print(f"  Reading video file...")
        with open(video_file_path, 'rb') as f:
            video_bytes = f.read()

        mime_type = get_mime_type(video_file_path)
        if not mime_type:
            print(f"  Unsupported video format: {video_file_path.suffix}")
            return None

        base_delay = 5
        for attempt in range(max_retries):
            try:
                print(f"  Sending to Gemini API (inline mode)...")
                self.rate_limiter.wait()
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=types.Content(
                        parts=[
                            types.Part(
                                inline_data=types.Blob(data=video_bytes, mime_type=mime_type)
                            ),
                            types.Part(text=self.processing_prompt)
                        ]
                    ),
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=65536,
                        thinking_config=types.ThinkingConfig(
                            thinking_level=get_thinking_level(self.model)
                        ),
                        safety_settings=SAFETY_SETTINGS_NONE,
                    )
                )

                return response.text.strip()

            except genai_errors.APIError as e:
                if is_quota_exhausted(e):
                    raise QuotaExhaustedError(str(e))
                error_code = getattr(e, 'code', 0)
                is_retryable = error_code in [429, 500, 503]
                if attempt < max_retries - 1 and is_retryable:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"  Retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(f"  Error processing video inline: {e}")
                    return None

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"  Error: {e} — retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(f"  Error processing video inline: {e}")
                    return None

        return None
    
    def process_video_uploaded(self, video_file_path, max_retries=3):
        """
        Process a video file by uploading to Files API first.

        Args:
            video_file_path (Path): Path to the video file
            max_retries (int): Maximum retry attempts for transient errors

        Returns:
            str: Generated text or None if error
        """
        # Upload the video
        uploaded_file = self.upload_video_file(video_file_path)
        if not uploaded_file:
            return None

        base_delay = 5
        for attempt in range(max_retries):
            try:
                print(f"  Generating content from video...")
                self.rate_limiter.wait()
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[uploaded_file, self.processing_prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=65536,
                        thinking_config=types.ThinkingConfig(
                            thinking_level=get_thinking_level(self.model)
                        ),
                        safety_settings=SAFETY_SETTINGS_NONE,
                    )
                )

                # Optionally delete the uploaded file after processing
                try:
                    self.client.files.delete(name=uploaded_file.name)
                    print(f"  Cleaned up uploaded file.")
                except Exception:
                    pass  # Non-critical if deletion fails

                return response.text.strip()

            except genai_errors.APIError as e:
                if is_quota_exhausted(e):
                    raise QuotaExhaustedError(str(e))
                error_code = getattr(e, 'code', 0)
                is_retryable = error_code in [429, 500, 503]
                if attempt < max_retries - 1 and is_retryable:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"  Retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(f"  Error processing uploaded video: {e}")
                    return None

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"  Error: {e} — retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(f"  Error processing uploaded video: {e}")
                    return None

        return None
    
    def process_video(self, video_file_path):
        """
        Process a single video file using the appropriate method based on file size.
        
        Args:
            video_file_path (Path): Path to the video file
            
        Returns:
            str: Generated text or None if error
        """
        print(f"Processing: {video_file_path.name}")
        
        # Check file size
        file_size = video_file_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        print(f"  File size: {file_size_mb:.1f} MB")
        
        if file_size <= self.max_inline_size:
            # Small file - process inline
            return self.process_video_inline(video_file_path)
        else:
            # Large file - upload first
            print(f"  File exceeds {self.max_inline_size / (1024*1024):.0f}MB, using Files API upload...")
            return self.process_video_uploaded(video_file_path)
    
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
            with open(output_file_path, 'w', encoding='utf-8') as f:
                # Write header with metadata
                f.write(f"Video Processing Output: {video_file_path.name}\n")
                f.write(f"Generated using: Google {self.model}\n")
                f.write("=" * 60 + "\n\n")
                f.write(output_text)
            
            print(f"  Output saved: {output_file_path}")
            return output_file_path
        
        except Exception as e:
            print(f"  Error saving output: {e}")
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
            print("No supported video files found in the video folder.")
            print(f"Supported formats: {', '.join(VIDEO_FORMATS.keys())}")
            return
        
        print(f"Found {len(video_files)} video file(s) to process:")
        for file_path in video_files:
            print(f"  - {file_path.name}")
        
        print("\nStarting video processing...\n")
        
        successful_processes = 0
        failed_processes = 0
        
        for video_file in video_files:
            try:
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
                print("\n" + "="*60)
                print("API QUOTA EXHAUSTED — stopping all processing.")
                print("Partial results (if any) have been saved.")
                print("Wait for your quota to reset or upgrade your plan.")
                print("="*60)
                break

            except Exception as e:
                print(f"  Unexpected error processing {video_file.name}: {e}")
                failed_processes += 1

            print()  # Add spacing between files
        
        # Summary
        print("=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total files processed: {len(video_files)}")
        print(f"Successful: {successful_processes}")
        print(f"Failed: {failed_processes}")
        
        if successful_processes > 0:
            print(f"\nOutput saved in the '{output_folder}' folder.")
    
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
        print("\n" + "=" * 60)
        print("PROMPT SELECTION")
        print("=" * 60)
        print("Available processing modes:")
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
        Allow user to select a processing prompt.
        
        Args:
            prompts_folder (str): Path to the prompts folder
            
        Returns:
            str: The selected prompt content
        """
        available_prompts = self.get_available_prompts(prompts_folder)
        
        if not available_prompts:
            print("No prompt files found. Using default prompt.")
            return self.default_prompt
        
        # Display menu
        self.display_prompt_menu(available_prompts)
        
        # Get user selection
        numbered_prompts = [(num, desc, path) for num, desc, path in available_prompts if num > 0]
        
        if not numbered_prompts:
            print("No numbered prompts found. Using default prompt.")
            return self.default_prompt
        
        while True:
            try:
                choice = input(f"Select a mode (1-{len(numbered_prompts)}) or press Enter for default: ").strip()
                
                if not choice:
                    print("Using default prompt.")
                    return self.default_prompt
                
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
                    return self.load_prompt_content(prompt_path)
                else:
                    print(f"Invalid choice. Please select a number between 1 and {len(numbered_prompts)}.")
                    
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nUsing default prompt.")
                return self.default_prompt


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
  python AI_video_summary.py --model gemini-3-flash-preview
  python AI_video_summary.py --video-folder my_videos --output-folder results
        """
    )
    parser.add_argument(
        "--model",
        choices=["gemini-3-pro-preview", "gemini-3-flash-preview"],
        default=None,
        help="Model to use for processing (default: interactive selection)"
    )
    parser.add_argument(
        "--video-folder",
        default="video",
        help="Folder containing video files (default: video)"
    )
    parser.add_argument(
        "--output-folder",
        default="output",
        help="Folder for output files (default: output)"
    )
    return parser.parse_args()


def select_model_interactive():
    """
    Interactively select a model if not provided via CLI.
    """
    print("\nAvailable models:")
    print("1. gemini-3-pro-preview (Higher quality, best for detailed transcription)")
    print("2. gemini-3-flash-preview (Faster, good for summaries)")
    model_choice = input("\nSelect a model (1 or 2) or press Enter for default (gemini-3-pro-preview): ").strip()
    
    if model_choice == '2':
        return 'gemini-3-flash-preview'
    return 'gemini-3-pro-preview'


def main():
    """
    Main function to run the video processing script.
    """
    args = parse_args()
    
    print("Video Processing using Google Gemini")
    print("=" * 60)
    
    try:
        # Select model via CLI or interactive
        if args.model:
            selected_model = args.model
            print(f"\nUsing model: {selected_model}")
        else:
            selected_model = select_model_interactive()
            print(f"Selected: {selected_model}")
        
        # Initialize processor (prompt chosen interactively)
        processor = VideoProcessor(model=selected_model)
        
        # Process all video files
        processor.process_all_video_files(
            video_folder=args.video_folder,
            output_folder=args.output_folder
        )
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nTo use this script, you need to set your Gemini API key:")
        print("1. Get your API key from: https://aistudio.google.com/app/api-keys")
        print("2. Create or edit a .env file in this directory")
        print("3. Add: GEMINI_API_KEY=your-api-key-here")
        print("4. Save the file and run this script again")
        
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
