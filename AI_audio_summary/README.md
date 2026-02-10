# Audio Transcription

Transcribe audio and video recordings using Google Gemini's multimodal capabilities.

## Why This Tool?

Oral histories, interviews, sermons, and radio broadcasts contain irreplaceable information that remains unsearchable as audio. Transcription unlocks this content for full-text search, quotation, and analysis. This pipeline handles multilingual recordings, speaker changes, and non-speech events.

## How It Works

```
Omeka S or local files → Download media → AI transcription → Update database
```

1. **Download** (`01_omeka_media_downloader.py`): Fetch audio/video from Omeka S
2. **Transcribe** (`02_AI_transcribe_audio.py`): Generate timestamped transcripts
3. **Update** (`03_omeka_transcription_updater.py`): Store transcripts in Omeka S

## Quick Start

```bash
python 01_omeka_media_downloader.py   # Download from Omeka S item set
python 02_AI_transcribe_audio.py      # Transcribe (interactive prompts)
python 03_omeka_transcription_updater.py  # Update Omeka S
```

Or process local files by placing them in `Audio/` and running step 2.

## Transcription Modes

| Mode | Output | Best For |
|------|--------|----------|
| **Full Transcription** | Verbatim with timestamps | Oral histories, academic research |
| **Full Translation (English)** | Translated to English | Multilingual recordings |
| **Full Translation (French)** | Translated to French | Francophone research output |
| **Hausa Segmentation** | Segment summaries + keywords | Quick analysis, cataloging |

Select mode interactively or edit prompt files in `prompts/`.

## Supported Formats

**Audio**: MP3, WAV, M4A, FLAC, OGG, WebM, AAC
**Video**: MP4, MKV, AVI, MOV, WMV, FLV (auto-converted to audio)

## Model Selection

| Model | Speed | Quality | Cost |
|-------|-------|---------|------|
| `gemini-3-flash-preview` | Faster | Good | ~$0.50-2/hour |
| `gemini-3-pro-preview` | Slower | Higher | ~$5-15/hour |

Use Flash for clear recordings, Pro for noisy audio or multiple speakers.

## Long Audio Handling

For recordings over 10 minutes, enable splitting:

```bash
python 02_AI_transcribe_audio.py --split
```

This improves accuracy and allows recovery if segments fail. Failed segments are marked and can be retried with `--resume`.

## Output Format

```
Transcription of: interview_2024.mp3
Generated using: Google gemini-3-pro-preview
==================================================

[00:00:01] Speaker 1: Welcome to today's interview...
[00:01:23] Speaker 2: Thank you for having me...
[laughter]
[00:02:15] Speaker 1: Let's begin with...
```

## Limitations

**Speaker identification**: The model labels speaker changes but doesn't identify individuals by name. You'll see "Speaker 1", "Speaker 2", etc.

**Background noise**: Heavy background noise, overlapping speech, or poor recording quality reduces accuracy.

**Accents and dialects**: Regional accents or code-switching may be transcribed with varying accuracy.

**Unclear audio**: Marked as `[inaudible 00:01:23]` rather than guessed.

**Non-speech sounds**: Captured as `[laughter]`, `[applause]`, etc., but subtle sounds may be missed.

## Requirements

**Optional** (for audio splitting and video conversion):
```bash
pip install pydub
```

FFmpeg must also be installed. The shared `common/ffmpeg_utils.py` module auto-discovers it via `FFMPEG_PATH`/`FFPROBE_PATH` env vars, `PATH`, or common Windows install locations. To install ffmpeg:

```bash
# Windows
winget install Gyan.FFmpeg
# macOS
brew install ffmpeg
# Linux
apt install ffmpeg
```

## Configuration

Create `.env` in project root:

```bash
GEMINI_API_KEY=your_key

# For Omeka S integration
OMEKA_BASE_URL=https://your-instance.org
OMEKA_KEY_IDENTITY=your_key
OMEKA_KEY_CREDENTIAL=your_credential
```

## Customization

Create custom transcription modes by adding `.md` files to `prompts/`:
- Name with number prefix: `4_my_custom_prompt.md`
- The script auto-detects new prompts

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Segment failed | Use `--resume --split` to retry only failed segments |
| pydub not installed | `pip install pydub` (required for splitting) |
| ffmpeg not found | Install ffmpeg or set `FFMPEG_PATH`/`FFPROBE_PATH` in `.env`. See `common/ffmpeg_utils.py` for discovery logic |
| Poor transcription quality | Try Pro model, reduce background noise |
| Video not converting | Verify ffmpeg is found (`get_ffmpeg_paths()`) and the video format is in `VIDEO_FORMATS` |
