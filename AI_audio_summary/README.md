# Audio Transcription

Transcribe audio and video recordings using Google Gemini or Mistral Voxtral.

## Why This Tool?

Oral histories, interviews, sermons, and radio broadcasts contain irreplaceable information that remains unsearchable as audio. Transcription unlocks this content for full-text search, quotation, and analysis. This pipeline handles multilingual recordings, speaker changes, and non-speech events.

## How It Works

```
Omeka S or local files → Download media → AI transcription → Update database
```

1. **Download** (`01_omeka_media_downloader.py`): Fetch audio/video from Omeka S
2. **Transcribe**: Generate transcripts using one of:
   - `02_AI_transcribe_audio.py` — Google Gemini (prompt-based, multimodal)
   - `02b_AI_transcribe_audio_voxtral.py` — Mistral Voxtral (dedicated transcription model with diarization)
3. **Update** (`03_omeka_transcription_updater.py`): Store transcripts in Omeka S

## Quick Start

```bash
python 01_omeka_media_downloader.py          # Download from Omeka S item set
python 02_AI_transcribe_audio.py             # Transcribe with Gemini
# or
python 02b_AI_transcribe_audio_voxtral.py    # Transcribe with Voxtral
python 03_omeka_transcription_updater.py     # Update Omeka S
```

Or process local files by placing them in `Audio/` and running step 2.

## Provider Comparison

| Feature | Gemini (`02_`) | Voxtral (`02b_`) |
|---------|---------------|-------------------|
| **Approach** | Prompt-based multimodal | Dedicated transcription model |
| **Speaker diarization** | Via prompt instructions | Native `--no-diarize` toggle |
| **Language selection** | Via prompt | `--language` flag (en, fr, de, ha, sw) |
| **Max audio length** | ~10 min per segment | Up to 3 hours per request |
| **Audio splitting** | Optional (`--split`) | Not needed |
| **Transcription modes** | Multiple prompts (verbatim, translation, segmentation) | Single mode (verbatim) |
| **Output files** | `.txt` only | `.txt` + `.json` (with timestamps) |
| **Cost** | ~$0.50-15/hour depending on model | $0.003/min (~$0.18/hour) |

**Use Gemini** when you need translation, custom prompts, or Hausa segmentation.
**Use Voxtral** for straightforward transcription with speaker diarization at lower cost.

## Transcription Modes (Gemini only)

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

### Gemini

| Model | Speed | Quality | Cost |
|-------|-------|---------|------|
| `gemini-3-flash-preview` | Faster | Good | ~$0.50-2/hour |
| `gemini-3-pro-preview` | Slower | Higher | ~$5-15/hour |

Use Flash for clear recordings, Pro for noisy audio or multiple speakers.

### Voxtral

| Model | Max Duration | Diarization | Cost |
|-------|-------------|-------------|------|
| `voxtral-mini-2602` | 3 hours | Yes (on by default) | $0.003/min |

Supports 13 languages. Language can be auto-detected or specified explicitly.

## Long Audio Handling

### Gemini

For recordings over 10 minutes, enable splitting:

```bash
python 02_AI_transcribe_audio.py --split
```

This improves accuracy and allows recovery if segments fail. Failed segments are marked and can be retried with `--resume`.

### Voxtral

No splitting needed — Voxtral handles up to 3 hours per request natively.

## Output Format

### Gemini

```
Transcription of: interview_2024.mp3
Generated using: Google gemini-3-pro-preview
==================================================

[00:00:01] Speaker 1: Welcome to today's interview...
[00:01:23] Speaker 2: Thank you for having me...
[laughter]
[00:02:15] Speaker 1: Let's begin with...
```

### Voxtral

Two files are saved per transcription:

**Text file** (`_transcription.txt`) — readable transcript with speaker labels:

```
Transcription of: interview_2024.mp3
Generated using: Mistral voxtral-mini-2602
Language: auto-detected
Diarization: ON
==================================================

[Speaker speaker_0]
Welcome to today's interview. We're here with...

[Speaker speaker_1]
Thank you for having me. I'm happy to be here...
```

**JSON file** (`_transcription.json`) — structured data with timestamps and speaker IDs:

```json
{
  "file": "interview_2024.mp3",
  "model": "voxtral-mini-2602",
  "text": "Welcome to today's interview...",
  "language": "en",
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "Welcome to today's interview.",
      "speaker_id": "speaker_0"
    }
  ],
  "usage": {
    "prompt_audio_seconds": 3600,
    "prompt_tokens": 1200,
    "total_tokens": 1200
  }
}
```

## Limitations

**Speaker identification**: Both models label speaker changes but don't identify individuals by name.

**Background noise**: Heavy background noise, overlapping speech, or poor recording quality reduces accuracy.

**Accents and dialects**: Regional accents or code-switching may be transcribed with varying accuracy.

**Overlapping speech**: Voxtral typically transcribes one speaker when speech overlaps.

**API constraints**: Voxtral's `language` and `timestamp_granularities` parameters are mutually exclusive — when a specific language is set, segment timestamps are disabled.

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
GEMINI_API_KEY=your_key      # For Gemini transcription
MISTRAL_API_KEY=your_key     # For Voxtral transcription

# For Omeka S integration
OMEKA_BASE_URL=https://your-instance.org
OMEKA_KEY_IDENTITY=your_key
OMEKA_KEY_CREDENTIAL=your_credential
```

## Customization

Create custom transcription modes (Gemini only) by adding `.md` files to `prompts/`:
- Name with number prefix: `4_my_custom_prompt.md`
- The script auto-detects new prompts

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Segment failed (Gemini) | Use `--resume --split` to retry only failed segments |
| pydub not installed | `pip install pydub` (required for splitting and video conversion) |
| ffmpeg not found | Install ffmpeg or set `FFMPEG_PATH`/`FFPROBE_PATH` in `.env`. See `common/ffmpeg_utils.py` for discovery logic |
| Poor transcription quality | Try Gemini Pro model, reduce background noise, or specify `--language` with Voxtral |
| Quota exhausted | Pipeline stops immediately and saves completed transcriptions; wait for daily reset |
| Video not converting | Verify ffmpeg is found (`get_ffmpeg_paths()`) and the video format is in `VIDEO_FORMATS` |
