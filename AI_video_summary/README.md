# AI Video Summary

Process video files using Google Gemini's multimodal capabilities for summarization and transcription with visual descriptions.

## Features

- **Video Summarization**: Generate comprehensive summaries of video content including visual and audio elements
- **Full Transcription with Visual Descriptions**: Create detailed transcripts with periodic descriptions of what appears on screen
- **Automatic file size handling**: Small files (<20MB) are processed inline; larger files use the Gemini Files API
- **Multiple model support**: Choose between Gemini 3.0 Pro Preview (higher quality) and Gemini 2.5 Flash (faster)

## Supported Video Formats

- MP4, MPEG, MOV, AVI, FLV, MPG, WebM, WMV, 3GPP

## Setup

### 1. Install Dependencies

```bash
pip install google-genai python-dotenv
```

### 2. Configure API Key

Create or edit a `.env` file in the repository root:

```
GEMINI_API_KEY=your-api-key-here
```

Get your API key from: https://aistudio.google.com/app/api-keys

### 3. Add Video Files

Place video files in the `video/` folder (or specify a custom folder with `--video-folder`).

## Usage

### Interactive Mode

```bash
python AI_video_summary.py
```

This will:
1. Prompt you to select a model (Gemini 3.0 Pro Preview or 2.5 Flash)
2. Display available processing modes (summary or full transcription)
3. Process all videos in the `video/` folder
4. Save outputs to the `output/` folder

### Command Line Options

```bash
# Use a specific model
python AI_video_summary.py --model gemini-2.5-flash

# Specify custom input/output folders
python AI_video_summary.py --video-folder my_videos --output-folder results

# Full example
python AI_video_summary.py --model gemini-3-pro-preview --video-folder recordings --output-folder transcripts
```

### Available Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model to use (`gemini-3-pro-preview` or `gemini-2.5-flash`) | Interactive selection |
| `--video-folder` | Folder containing video files | `video` |
| `--output-folder` | Folder for output files | `output` |

## Processing Modes

### 1. Video Summary
Generates a structured summary including:
- Overview of the video content
- Setting and context
- Key participants
- Main content themes
- Notable moments
- Conclusion

Best for: Quick content analysis, cataloging, research documentation.

### 2. Video Transcription with Visual Descriptions
Creates a detailed transcript with:
- Timestamped speaker turns
- Verbatim transcription of spoken content
- Periodic visual descriptions (every 30-60 seconds)
- On-screen text and graphics notation
- Non-speech audio events

Best for: Archival purposes, accessibility, detailed analysis.

## Model Selection Guide

| Model | Best For | Speed | Quality |
|-------|----------|-------|---------|
| `gemini-3-pro-preview` | Detailed transcription, complex scenes | Slower | Higher |
| `gemini-2.5-flash` | Summaries, shorter videos | Faster | Good |

## File Size Handling

- **Small files (≤20MB)**: Processed inline for faster results
- **Large files (>20MB)**: Automatically uploaded to Gemini Files API, with polling until processing completes

## Output

Processed files are saved as text files in the output folder with metadata headers:

```
Video Processing Output: example_video.mp4
Generated using: Google gemini-3-pro-preview
============================================================

[Content follows...]
```

## Technical Details

- Video sampling rate: 1 frame per second (default)
- Token usage: ~300 tokens per second of video at default resolution
- Maximum video length: Up to 2 hours (2M context models) or 1 hour (1M context models)

## Folder Structure

```
AI_video_summary/
├── AI_video_summary.py      # Main script
├── README.md                 # This file
├── prompts/
│   ├── 1_video_summary.md              # Summary prompt
│   └── 2_video_transcription_description.md  # Transcription prompt
├── video/                    # Input folder (place videos here)
└── output/                   # Output folder (generated automatically)
```

## Troubleshooting

### "GEMINI_API_KEY not found"
Ensure your `.env` file exists and contains a valid API key.

### Video processing takes too long
- Use `gemini-2.5-flash` for faster processing
- Consider using shorter video clips

### "Unsupported video format"
Convert your video to MP4 format using ffmpeg:
```bash
ffmpeg -i input.mkv -c:v libx264 -c:a aac output.mp4
```

### Large file upload fails
Ensure you have a stable internet connection. The script will poll for processing status automatically.
