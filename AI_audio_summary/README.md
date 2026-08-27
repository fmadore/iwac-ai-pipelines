# Audio Transcription

Transcribe audio and video recordings using Google Gemini 3.5 Transcribe,
Google Gemini, or Mistral Voxtral.

## Why This Tool?

Oral histories, interviews, sermons, and radio broadcasts contain irreplaceable information that remains unsearchable as audio. Transcription unlocks this content for full-text search, quotation, and analysis. This pipeline handles multilingual recordings, speaker changes, and non-speech events.

## How It Works

```
Omeka S or local files → Download media → AI transcription → Update database
```

1. **Download** (`01_omeka_media_downloader.py`): Fetch audio/video from Omeka S
2. **Transcribe**: Generate transcripts using one of:
   - `02c_AI_transcribe_audio_gemini_transcribe.py` — **Google Gemini 3.5
     Transcribe** (dedicated speech-to-text; word timestamps and speaker labels).
     The default for verbatim work.
   - `02_AI_transcribe_audio.py` — Google Gemini (prompt-based, multimodal). The
     only route to translation and Hausa segmentation.
   - `02b_AI_transcribe_audio_voxtral.py` — Mistral Voxtral (dedicated transcription model with diarization)
3. **Update** (`03_omeka_transcription_updater.py`): Store transcripts in Omeka S
   as `bibo:content`, annotated with the model that produced them. Only the
   transcript is uploaded — each file's metadata header stays on disk, because
   `bibo:content` is the archive's indexed full text

## Quick Start

```bash
python 01_omeka_media_downloader.py                    # Download from Omeka S item set
python 02c_AI_transcribe_audio_gemini_transcribe.py    # Transcribe (default)
python 03_omeka_transcription_updater.py               # Update Omeka S (--dry-run to preview)
```

The other two transcribers, when the default will not do:

```bash
python 02_AI_transcribe_audio.py             # translation, Hausa segmentation, unlisted languages
python 02b_AI_transcribe_audio_voxtral.py    # Voxtral, 3 h per request, no splitting
python 03_omeka_transcription_updater.py --no-model-annotation   # Voxtral output: no authority item
```

Or process local files by placing them in `Audio/` and running step 2.

## Provider Comparison

| Feature | Gemini 3.5 Transcribe (`02c_`) | Gemini (`02_`) | Voxtral (`02b_`) |
|---------|-------------------------------|---------------|-------------------|
| **Approach** | Dedicated speech-to-text | Prompt-based multimodal | Dedicated transcription model |
| **Prompt** | **None accepted** (hard 400) | Any file in `prompts/` | None |
| **Speaker diarization** | Native, up to 8 speakers | Via prompt instructions | Native `--no-diarize` toggle |
| **Timestamps** | Word-level, in a JSON sidecar | Asked for in the prompt | Segment-level, JSON sidecar |
| **Language selection** | 82 BCP-47 locales, validated locally | Via prompt (any language) | `--language` flag (en, fr, de, ha, sw) |
| **Max audio length** | 30 min with timestamps, 1 h without | Up to 9.5 h per request (Files API); 20-min segments by default | Up to 3 hours per request |
| **Audio splitting** | Automatic above the cap | Optional (`--split`, 20-min segments) | Not needed |
| **Transcription modes** | Verbatim or smart | Multiple prompts (verbatim, translation, segmentation) | Single mode (verbatim) |
| **Output files** | `.txt` + `.json` (word timings) | `.txt` only | `.txt` + `.json` (with timestamps) |
| **Cost** | $0.003/min in + $0.002/min out (~$0.30/hour) | ~$0.50-15/hour depending on model | $0.003/min (~$0.18/hour) |

**Use Gemini 3.5 Transcribe** for verbatim transcription of French, Hausa, Arabic
or English. It is the only route here that gives word-level timings, and its
speaker labels come from a diarizer rather than from asking a chat model to
notice speaker changes.

**Use Gemini** when you need translation into French or English, Hausa
segmentation, a custom prompt, or a language `02c_` has no locale for — which,
in this collection, is most of the local ones.

**Use Voxtral** for a recording between one and three hours that you would
rather not split.

## Language coverage — read this before choosing `02c_`

`gemini-3.5-transcribe` documents 82 BCP-47 locales. Set against the thirteen
languages catalogued in this collection (`LANGUAGE_LABELS_BY_CODE` in
`common/iwac_config.py`):

| Catalogued | Locale |
|---|---|
| Français, Haoussa, Arabe, Anglais | `fr-FR`, `ha-NG`, `ar-EG`, `en-US`/`en-GB` |
| Allemand, Italien, Espagnol, Slovène | `de-DE`, `it-IT`, `es-ES`, `sl-SI` |
| **Mooré, Dioula, Ewé, Kabyè, Dendi** | **none** |

The five with no locale are the West African ones. They are not a rounding error
here: the sibling YouTube corpus is 8 Mooré-dominant items in 44, all of them
catalogued `Français`.

**The API does not enforce that list.** Every unsupported code tried against it —
`mos-BF`, `dyu-CI`, `ee-GH`, `kbp-TG` — was accepted and answered normally, with
no warning and no error. So `--language` is validated inside `02c_` against
`SUPPORTED_LOCALES` before a request is built. Without that check, asserting
Mooré returns a fluent, plausible, French-ish transcript, and nothing downstream
can tell it from a real one.

Omitting `--language` is safe and often better: the model auto-detects, and
switches when speakers code-switch.

## Verbatim vs smart (`02c_` only)

| | verbatim (default) | smart (`--smart`) |
|---|---|---|
| Output | word for word, fillers kept | disfluencies removed, auto-formatted |
| Word timestamps | yes | **rejected** (400) |
| Speaker labels | yes | **rejected** (400) |
| Cap per request | 30 min | 1 hour |

The two are mutually exclusive at the API, so `--smart` combined with either
option fails at start-up rather than after the upload. Timestamps also cost a
little accuracy, as the vendor documentation warns: on a test clip, verbatim
with timestamps rendered *Aujourd'hui* as *Aujourd'*, which smart and plain
verbatim both got right.

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

### Gemini 3.5 Transcribe

| Model | Cap per request | Diarization | Cost |
|-------|-----------------|-------------|------|
| `gemini-3.5-transcribe` | 30 min (timestamps on), 1 h (smart) | Yes, up to 8 speakers | $0.003/min in + $0.002/min out |

Speaker attribution beyond two voices is documented as experimental. The model
id is pinned rather than rolling, which is what lets step 03 stamp it: authority
item 113077, registered in `AI_MODEL_ITEMS` as `gemini-3.5-transcribe`.

`--rpm` throttles proactively; a daily quota raises `QuotaExhaustedError`, which
saves what completed and stops instead of retrying a limit that will not clear.

### Gemini

| Model | Speed | Quality | Cost |
|-------|-------|---------|------|
| `gemini-flash-lite-latest` | Fastest | Good | ~$0.20-1/hour |
| `gemini-3.7-flash` | Faster | Good | ~$0.50-2/hour |
| `gemini-pro-latest` | Slower | Higher | ~$5-15/hour |

Use Flash-Lite for clean, single-speaker recordings on a budget, Flash for general use, Pro for noisy audio or multiple speakers.

### Voxtral

| Model | Max Duration | Diarization | Cost |
|-------|-------------|-------------|------|
| `voxtral-mini-2602` | 3 hours | Yes (on by default) | $0.003/min |

Supports 13 languages. Language can be auto-detected or specified explicitly.

## Long Audio Handling

### Gemini 3.5 Transcribe

Splitting is automatic and not optional: 30 minutes is a hard per-request limit
whenever timestamps or diarization are on. Anything longer is cut into
`--segment-minutes` pieces (default 20, clamped to whatever the active cap
allows) before upload, and a file whose duration cannot be probed is split
rather than gambled with.

Two things are then corrected client-side, because the API restarts both in
every request:

- **Word offsets are shifted** by each segment's start position, so the sidecar
  reports positions in the whole recording rather than three transcripts that
  each begin at zero.
- **Speaker ids are namespaced per segment** (`seg2-spk:0`). `spk:0` in one
  segment and `spk:0` in the next are not evidence of the same person, and
  merging them would invent a continuity the diarizer never claimed. Diarization
  identity does not carry across a split.

A segment that fails withholds the whole file rather than writing a transcript
missing its middle — that gap is invisible once the text is a single Omeka value.

### Gemini

Gemini can transcribe up to **9.5 hours** of audio per request. Audio is sent
**inline** (capped at 20 MB total) for small files, and automatically uploaded
via the **Files API** for anything larger — so segments are not limited by the
20 MB inline cap.

For long recordings, enable splitting into 20-minute segments:

```bash
python 02_AI_transcribe_audio.py --split                 # 20-min segments (default)
python 02_AI_transcribe_audio.py --split --segment-minutes 45   # longer segments
```

Twenty-minute segments mean fewer split points — and fewer boundary
artifacts — than shorter chunks, while keeping per-segment retries cheap.
Splitting also allows recovery if segments fail: failed segments are marked
and can be retried with `--resume`.

### Voxtral

No splitting needed — Voxtral handles up to 3 hours per request natively.

## Provenance

Every value step 3 writes carries an `iwac:transcriptionModel` annotation naming
the model behind it, so a transcript's origin survives outside the header on
disk. The model is read from the transcripts' own `Generated using:` line, and
`--model` overrides it.

Only some of what this pipeline produces can be named that way. An annotation
points at an Omeka authority item, and three of the five models here have none:

| Header | Annotated as | Why |
|---|---|---|
| `Google gemini-3.5-transcribe` | Gemini 3.5 Transcribe | Pinned release, item 113077 |
| `Google gemini-3.7-flash` | Gemini 3.7 Flash | Pinned release, item 111774 |
| `Google gemini-pro-latest` | — | Rolling alias: reports its version as "Gemini Pro Latest", so a run through it cannot say which release answered |
| `Google gemini-flash-lite-latest` | — | Same |
| `Mistral voxtral-mini-2602` | — | No authority item yet |

So the default transcriber (`02c_`) needs no flag: its header names a pinned id
with an authority item, and the annotation is written without being asked for.

For the three that cannot be cited, pass `--no-model-annotation` to upload the
text without provenance, or `--model <key>` to assert the pinned release an
alias resolved to on the day the run happened. Whichever it is, it is a flag the
operator passes on purpose: the step stops rather than writing text whose model
nothing records.

One annotation covers the whole batch, so a `Transcriptions/` folder holding two
models' output is refused. Move each model's transcripts into their own folder
and run the step once per folder.

## Output Format

### Gemini 3.5 Transcribe

Two files per recording, as with Voxtral.

**Text file** (`_transcription.txt`) — speaker turns, no timestamps in the body:

```
Transcription of: entretien_2024.wav
Generated using: Google gemini-3.5-transcribe
Mode: verbatim
Language: fr-FR
Timestamps: ON
Diarization: ON
==================================================

[Speaker spk:0]
Bonjour, je m'appelle Fatoumata Traoré et je suis membre de la communauté
musulmane de Ouagadougou.

[Speaker spk:1]
Merci beaucoup madame. Pouvez-vous nous parler de la construction de la grande
mosquée en 1990.
```

Clock positions stay out of the body by default because that file becomes
`bibo:content`, the archive's indexed full text — a timestamp there is indexed
as though someone had said it. `--timestamps-in-text` prefixes each turn with
its position when a navigable reading copy is what you want.

**JSON file** (`_transcription.json`) — the word timings, and the settings that
produced them:

```json
{
  "file": "entretien_2024.wav",
  "model": "gemini-3.5-transcribe",
  "mode": "verbatim",
  "language_codes": ["fr-FR"],
  "timestamps": true,
  "diarization": true,
  "segments": 1,
  "duration_seconds": 22.34,
  "text": "[Speaker spk:0]
Bonjour, je m'appelle Fatoumata Traoré...",
  "words": [
    {"text": "Bonjour,", "speaker": "spk:0", "start": 0.1, "end": 1.0},
    {"text": "je", "speaker": "spk:0", "start": 1.0, "end": 1.2}
  ],
  "speakers": ["spk:0", "spk:1"]
}
```

### Gemini

```
Transcription of: interview_2024.mp3
Generated using: Google gemini-pro-latest
==================================================

[00:00:01] Speaker 1: Welcome to today's interview...
[00:01:23] Speaker 2: Thank you for having me...
[laughter]
[00:02:15] Speaker 1: Let's begin with...
```

When the audio is split (`--split`), each block is prefixed with a header
giving its position in the original recording — `[Segment <n>/<total> |
<start>–<end>]`, with the start time computed from the segment length. The
final segment shows `–end`:

```
[Segment 1/3 | 00:00:00–00:20:00]
[00:00:01] Speaker 1: Welcome to today's interview...

[Segment 2/3 | 00:20:00–00:40:00]
...

[Segment 3/3 | 00:40:00–end]
...
```

A segment that fails is marked `[Segment <n>/<total> | <start>–<end>] TRANSCRIPTION FAILED` and can be re-run with `--resume` (which preserves the header).

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

**API constraints**: Voxtral's `language` and `timestamp_granularities` parameters are mutually exclusive — when a specific language is set, segment timestamps are disabled. Gemini 3.5 Transcribe's `smart` mode is likewise incompatible with timestamps and diarization.

**No prompt on `02c_`**: `gemini-3.5-transcribe` rejects a system instruction outright (`400 Developer instruction is not enabled for this model`). Translation, Hausa segmentation and any custom mode therefore stay on `02_`, which is a multimodal chat model and can be told what to do.

**Local languages on `02c_`**: Mooré, Dioula, Ewé, Kabyè and Dendi have no locale, and the API will not say so — see *Language coverage* above.

**Provenance**: step 03 annotates only what it can name. `02c_` and `gemini-3.7-flash` output carries `iwac:transcriptionModel`; Voxtral output and anything produced through a rolling alias must be uploaded with `--no-model-annotation`, and its model then survives only in the on-disk transcript header. See *Provenance* above.

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
| Poor transcription quality | Try Gemini Pro model, reduce background noise, or specify `--language` |
| `does not support <code>` on `02c_` | The locale is not in the model's table; omit `--language` to auto-detect |
| `smart mode rejects timestamps` | Pass `--smart` on its own, or drop it to keep timestamps and speakers |
| Fluent transcript of the wrong language | Almost certainly a local language `02c_` cannot hear — re-run on `02_` |
| Quota exhausted | Pipeline stops immediately and saves completed transcriptions; wait for daily reset |
| Video not converting | Verify ffmpeg is found (`get_ffmpeg_paths()`) and the video format is in `VIDEO_FORMATS` |
