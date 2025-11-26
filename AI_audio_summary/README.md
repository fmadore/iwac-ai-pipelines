# Audio Transcription Pipeline

AI-powered audio transcription using Google Gemini for oral histories, interviews, sermons, and multilingual recordings.

## Overview

This pipeline transcribes audio files using Google Gemini's multimodal capabilities. It supports multiple audio formats, offers different transcription modes (verbatim, translation, segmented summaries), and can automatically split long recordings for improved accuracy.

## Features

- **Multiple audio formats**: MP3, WAV, M4A, FLAC, OGG, WebM, MP4, AAC
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
pip install google-genai python-dotenv pydub
```

2. Configure your API key in the project root `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Explicit ffmpeg paths (if not on PATH)
FFMPEG_PATH=C:\path\to\ffmpeg.exe
FFPROBE_PATH=C:\path\to\ffprobe.exe
```

## Usage

### Interactive Mode (Recommended)

```bash
cd AI_audio_summary
python 02_AI_transcribe_audio.py
```

The script will prompt you to:
1. Select a Gemini model (Pro or Flash)
2. Choose a transcription prompt
3. Decide whether to split long audio files

### Command Line Options

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

### Workflow

1. Place audio files in the `Audio/` folder
2. Run the script
3. Select your options
4. Find transcriptions in the `Transcriptions/` folder

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
├── 02_AI_transcribe_audio.py   # Main transcription script
├── README.md                   # This file
├── prompts/                    # Transcription prompt templates
│   ├── 1_full_audio_transcription.md
│   ├── 2_full_audio_translation.md
│   └── 3_hausa_prompt.md
├── Audio/                      # Input: Place audio files here
├── Transcriptions/             # Output: Transcriptions saved here
└── temp_segments/              # Temporary: Split audio segments (auto-cleaned)
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

### "ffmpeg not found" (when trying to split)
- Install ffmpeg and ensure it's on PATH, or set `FFMPEG_PATH` in `.env`
- Windows: `winget install Gyan.FFmpeg`

### Audio file not detected
- Check file extension is supported (mp3, wav, m4a, flac, ogg, webm, mp4, aac)
- Ensure file is in the `Audio/` folder (relative to the script)

### Transcription quality issues
- Try the Pro model instead of Flash
- Enable audio splitting for long recordings
- Check audio quality (reduce background noise if possible)

## API Costs

Gemini API pricing varies. As of late 2024:
- **Gemini 2.5 Flash**: ~$0.075 per 1M input tokens (audio)
- **Gemini 3.0 Pro**: ~$1.25 per 1M input tokens (audio)

A typical 1-hour interview might cost $0.10-0.50 with Flash or $1-5 with Pro.

Check [Google AI Studio](https://aistudio.google.com/) for current pricing.

## Technical Notes

- This script uses `google.genai.Client()` directly (not the shared `llm_provider.py`) because it requires multimodal audio processing
- Audio is sent as bytes with MIME type detection
- Temperature is set to 0.1 for consistent transcription output
- Max output tokens: 65,536 (sufficient for ~30 minutes of dense audio)
