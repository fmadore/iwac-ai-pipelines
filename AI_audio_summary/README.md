# Audio Transcription Pipeline

AI-powered audio and video transcription using Google Gemini for oral histories, interviews, sermons, and multilingual recordings.

## Overview

This pipeline downloads and transcribes audio and video files using Google Gemini's multimodal capabilities. It can fetch media directly from Omeka S collections or process local files. Video files are automatically converted to audio before transcription. It supports multiple formats, offers different transcription modes (verbatim, translation, segmented summaries), and can automatically split long recordings for improved accuracy.

## Features

- **Omeka S integration**: Download audio/video from item sets or individual items
- **Multiple audio formats**: MP3, WAV, M4A, FLAC, OGG, WebM, MP4, AAC, WMA
- **Video support**: MP4, MKV, AVI, MOV, WMV, FLV, WebM, M4V, MPEG, MPG, 3GP (auto-converted to audio)
- **Flexible prompts**: Full transcription, translation to English, or Hausa-specific segmentation
- **Long audio handling**: Automatic splitting into 10-minute segments for better accuracy
- **Speaker detection**: Labels speaker changes with timestamps
- **Multilingual support**: Handles code-switching and multiple languages in one recording
- **Interactive or CLI**: Choose settings interactively or pass arguments for batch processing

## Prerequisites

### Required

- Python 3.8+
- Google Gemini API key ([Get one free](https://aistudio.google.com/app/api-keys))

### Optional (for audio splitting)

- **pydub**: `pip install pydub`
- **ffmpeg**: Required by pydub for audio processing
  - Windows: `winget install Gyan.FFmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: `apt install ffmpeg`

## Installation

1. Install dependencies:
```bash
pip install google-genai python-dotenv pydub requests tqdm
```

2. Configure your API keys in the project root `.env` file:
```env
# Required for transcription
GEMINI_API_KEY=your_gemini_api_key_here

# Required for Omeka S media download
OMEKA_BASE_URL=https://your-omeka-instance.org
OMEKA_KEY_IDENTITY=your_key_identity
OMEKA_KEY_CREDENTIAL=your_key_credential

# Optional: Explicit ffmpeg paths (if not on PATH)
FFMPEG_PATH=C:\path\to\ffmpeg.exe
FFPROBE_PATH=C:\path\to\ffprobe.exe
```

## Scripts

### 1. Media Downloader (`01_omeka_media_downloader.py`)

Downloads audio and video files from Omeka S collections.

```bash
python 01_omeka_media_downloader.py
```

The script will prompt you to:
1. Choose between downloading from an item set or a single item
2. Enter the item set ID or item ID

**Features:**
- Downloads from entire item sets or individual items
- Handles multiple media files per item
- Skips already downloaded files
- Uses item titles for descriptive filenames
- Concurrent downloads with progress bar

**Supported formats:**
- Audio: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac`, `.wma`
- Video: `.mp4`, `.mkv`, `.avi`, `.mov`, `.wmv`, `.flv`, `.webm`, `.m4v`, `.mpeg`, `.mpg`, `.3gp`

### 2. Audio Transcription (`02_AI_transcribe_audio.py`)

Transcribes audio and video files using Google Gemini.

#### Interactive Mode (Recommended)

```bash
cd AI_audio_summary
python 02_AI_transcribe_audio.py
```

The script will prompt you to:
1. Select a Gemini model (Pro or Flash)
2. Choose a transcription prompt
3. Decide whether to split long audio files

#### Command Line Options

```bash
# Specify model directly
python 02_AI_transcribe_audio.py --model gemini-3-pro-preview
python 02_AI_transcribe_audio.py --model gemini-2.5-flash

# Enable audio splitting
python 02_AI_transcribe_audio.py --split

# Custom segment length (default: 10 minutes)
python 02_AI_transcribe_audio.py --split --segment-minutes 15

# Custom folders
python 02_AI_transcribe_audio.py --audio-folder MyAudio --output-folder MyOutput
```

## Workflow

### Complete Pipeline (Omeka S to Transcription)

1. Run the media downloader to fetch files from Omeka S:
   ```bash
   python 01_omeka_media_downloader.py
   ```
2. Files are saved to `Audio/` folder
3. Run the transcription script:
   ```bash
   python 02_AI_transcribe_audio.py
   ```
4. Find transcriptions in the `Transcriptions/` folder

### Local Files Only

1. Place audio or video files in the `Audio/` folder
2. Run the transcription script
3. Select your options
4. Video files are automatically converted to audio (requires ffmpeg)
5. Find transcriptions in the `Transcriptions/` folder

## Available Prompts

### 1. Full Audio Transcription
Verbatim transcription with:
- Timestamped speaker labels (`[00:00:01] Speaker 1:`)
- Non-speech events (`[laughter]`, `[applause]`, `[overlapping speech]`)
- Unclear audio markers (`[inaudible 00:01:23]`, `[Kandahar?]`)
- Preserved code-switching (no translation)
- Normalized numbers and dates

**Best for**: Oral history interviews, academic research, archival transcription

### 2. Full Audio Translation (to English)
Same format as above, but:
- All content translated into English
- Original language noted when code-switching occurs
- Culturally specific terms include original in parentheses: "religious endowment (waqf)"

**Best for**: Multilingual recordings where English output is needed

### 3. Hausa Segmentation & Summary
Creates structured summaries rather than full transcription:
- Segments audio by thematic/speaker breaks
- Provides English summaries for each segment
- Includes 5-8 keywords per segment
- Timestamps for each segment

**Best for**: Quick analysis of Hausa-language content, research cataloging

## Model Selection

| Model | Speed | Quality | Cost | Best For |
|-------|-------|---------|------|----------|
| `gemini-3-pro-preview` | Slower | Higher | Higher | Complex audio, multiple speakers, background noise |
| `gemini-2.5-flash` | Faster | Good | Lower | Clear recordings, single speaker, bulk processing |

## Audio Splitting

For long recordings (> 10 minutes), splitting improves accuracy by:
- Preventing context window issues
- Allowing the model to focus on smaller chunks
- Enabling recovery if one segment fails

**Auto-enabled** for prompt #1 (Full Audio Transcription)

**Requirements**: pydub + ffmpeg must be installed

### Segment Output Format

When splitting is enabled, each segment is labeled:
```
[Segment 1]
[00:00:01] Speaker 1: ...

[Segment 2]
[00:10:00] Speaker 1: ...
```

Temporary segment files are automatically cleaned up after successful transcription.

## Directory Structure

```
AI_audio_summary/
├── 01_omeka_media_downloader.py  # Download media from Omeka S
├── 02_AI_transcribe_audio.py     # Main transcription script
├── README.md                     # This file
├── prompts/                      # Transcription prompt templates
│   ├── 1_full_audio_transcription.md
│   ├── 2_full_audio_translation.md
│   └── 3_hausa_prompt.md
├── Audio/                        # Input: Place audio/video files here
├── Transcriptions/               # Output: Transcriptions saved here
├── log/                          # Download logs
├── temp_segments/                # Temporary: Split audio segments (auto-cleaned)
└── temp_converted_audio/         # Temporary: Converted video audio (auto-cleaned)
```

## Output Format

Each transcription file includes:
```
Transcription of: interview_2024.mp3
Generated using: Google gemini-3-pro-preview
==================================================

[00:00:01] Speaker 1: Welcome to today's interview...
```

## Customizing Prompts

To create a new transcription mode:

1. Create a new `.md` file in `prompts/`
2. Name it with a number prefix: `4_my_custom_prompt.md`
3. Include instructions in Markdown format
4. The script will automatically detect and offer it as an option

## Troubleshooting

### "GEMINI_API_KEY not found"
- Ensure `.env` file exists in the project root with `GEMINI_API_KEY=your_key`

### "pydub not installed" (when trying to split)
```bash
pip install pydub
```

### "ffmpeg not found" (when trying to split or convert video)
- Install ffmpeg and ensure it's on PATH, or set `FFMPEG_PATH` in `.env`
- Windows: `winget install Gyan.FFmpeg`

### Video file not converting
- Ensure ffmpeg is installed and accessible
- Check that the video file is not corrupted
- Verify the video format is supported (mp4, mkv, avi, mov, wmv, flv, webm, m4v, mpeg, mpg, 3gp)

### Audio file not detected
- Check file extension is supported (mp3, wav, m4a, flac, ogg, webm, mp4, aac)
- Ensure file is in the `Audio/` folder (relative to the script)

### Transcription quality issues
- Try the Pro model instead of Flash
- Enable audio splitting for long recordings
- Check audio quality (reduce background noise if possible)

### Omeka S download issues
- Verify `OMEKA_BASE_URL`, `OMEKA_KEY_IDENTITY`, and `OMEKA_KEY_CREDENTIAL` in `.env`
- Check that the item set or item ID exists
- Ensure the item has audio/video media attachments
- Check `log/media_download.log` for detailed error messages

## API Costs

Gemini API pricing varies. As of late 2024:
- **Gemini 2.5 Flash**: ~$0.075 per 1M input tokens (audio)
- **Gemini 3.0 Pro**: ~$1.25 per 1M input tokens (audio)

A typical 1-hour interview might cost $0.10-0.50 with Flash or $1-5 with Pro.

Check [Google AI Studio](https://aistudio.google.com/) for current pricing.

## Technical Notes

- The transcription script uses `google.genai.Client()` directly (not the shared `llm_provider.py`) because it requires multimodal audio processing
- Audio is sent as bytes with MIME type detection
- Temperature is set to 0.1 for consistent transcription output
- Max output tokens: 65,536 (sufficient for ~30 minutes of dense audio)
- The media downloader uses concurrent downloads (2 workers by default) to speed up item set processing
