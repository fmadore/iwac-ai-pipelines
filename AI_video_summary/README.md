# Video Processing

Summarize or transcribe video files with visual descriptions using Google Gemini's multimodal capabilities.

## Why This Tool?

Video content combines speech, on-screen text, and visual information. Standard audio transcription misses what's shown on screen—speakers, locations, displayed documents, gestures. This pipeline captures both audio and visual elements for comprehensive documentation.

## How It Works

```
Video files → Gemini multimodal processing → Summary or transcript with visual descriptions
```

The script sends video directly to Gemini, which analyzes both audio and visual streams.

## Quick Start

1. Place video files in the `video/` folder
2. Run the script:
   ```bash
   python AI_video_summary.py
   ```
3. Select model and processing mode
4. Find outputs in `output/`

Or specify options directly:
```bash
python AI_video_summary.py --model gemini-3-flash-preview
```

## Processing Modes

| Mode | Output | Best For |
|------|--------|----------|
| **Summary** | Structured overview (setting, participants, themes) | Quick content analysis, cataloging |
| **Transcription** | Timestamped transcript + periodic visual descriptions | Archival, accessibility, detailed analysis |

## Supported Formats

MP4, MPEG, MOV, AVI, FLV, MPG, WebM, WMV, 3GPP

## Model Selection

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| `gemini-3-flash-preview` | Faster | Good | Summaries, shorter videos |
| `gemini-3-pro-preview` | Slower | Higher | Detailed transcription, complex scenes |

## Output Format

```
Video Processing Output: example_video.mp4
Generated using: Google gemini-3-pro-preview
============================================================

[Summary or transcript content...]
```

## Limitations

**Visual description frequency**: Descriptions occur every 30-60 seconds; fast-changing visuals between descriptions may be missed.

**On-screen text**: Small or stylized text may not be captured accurately.

**Speaker identification**: Speakers are labeled generically ("Speaker 1") unless visually identifiable.

**Processing time**: Video analysis is slower than audio-only; expect several minutes for longer videos.

**File size**: Files over 20MB are automatically uploaded via Gemini Files API (handled transparently).

## Configuration

Create `.env` in project root:

```bash
GEMINI_API_KEY=your_key
```

## Customization

Edit prompt files in `prompts/` to adjust:
- `1_video_summary.md` — Summary structure and focus areas
- `2_video_transcription_description.md` — Visual description frequency and detail level

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Unsupported format | Convert to MP4: `ffmpeg -i input.mkv output.mp4` |
| Processing timeout | Use Flash model for faster results |
| Upload fails | Check internet connection; script retries automatically |
