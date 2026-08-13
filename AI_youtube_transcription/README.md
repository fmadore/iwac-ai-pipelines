# YouTube Transcription

Transcribe YouTube-hosted audiovisual items with Google Gemini, straight from
their URLs.

## Why This Tool?

Since 2026-08-12 the collection holds embedded YouTube videos alongside its
deposited recordings — sermons, interviews, ceremonies and TV segments ingested
from public channels. They are unsearchable as video, and unlike the deposited
recordings there is nothing to download: a YouTube item's media carries no file
at all (Omeka's `youtube` ingester stores only thumbnail derivatives, so
`o:original_url`, `o:media_type` and `o:size` are null).

Gemini accepts a YouTube watch URL directly. Google's servers fetch and decode
the video, so this pipeline never touches the stream: no download, no `ffmpeg`
re-encode, no Files API upload, no local media on disk. That removes most of what
`AI_audio_summary/01_omeka_media_downloader.py` exists to do, and it is why this
pipeline is genuinely thinner than its sibling rather than a copy of it.

## How It Works

```
Omeka S (template 23) → work list → Gemini (URL, no download) → bibo:content
                                            └ languages heard → dcterms:language
```

1. **Fetch** (`01_omeka_youtube_fetcher.py`): read items, pull the watch URL off
   `fabio:hasURL`, write a work list. Reads metadata only.
2. **Transcribe** (`02_AI_transcribe_youtube.py`): detect the spoken languages
   from sampled windows, then transcribe. One request per video, or one per
   window for videos longer than the budget.
3. **Update** (`03_omeka_transcription_updater.py`): write `bibo:content` with an
   `iwac:transcriptionModel` annotation naming the model.
4. **Correct the record** (`04_omeka_language_updater.py`, optional): append the
   languages actually heard to `dcterms:language`.

## Quick Start

```bash
python 01_omeka_youtube_fetcher.py                       # item set 108260 by default
python 02_AI_transcribe_youtube.py                       # prompts for model and prompt
python 03_omeka_transcription_updater.py --dry-run       # preview the writes
python 03_omeka_transcription_updater.py --model gemini-3.5-flash-lite
python 04_omeka_language_updater.py --dry-run            # preview the language fixes
```

One item end to end:

```bash
python 01_omeka_youtube_fetcher.py --item-id 108353
python 02_AI_transcribe_youtube.py --model gemini-3.5-flash-lite --prompt 1
python 03_omeka_transcription_updater.py --model gemini-3.5-flash-lite --dry-run
```

The model passed to `03` must be the one that produced the transcripts — it is
what the `iwac:transcriptionModel` annotation will name. `03` warns when the
transcript headers disagree with it, and when a folder holds output from two
models (one annotation is written for the whole batch, so those need separate runs).

## Scope: which items are picked up

`01` defaults to the YouTube item sets in `common/iwac_config.py`
(`YOUTUBE_VIDEO_ITEM_SETS`, currently `108260` — YouTube videos Burkina Faso) and
filters to resource template **23**. Resource class 38 holds two populations that
behave differently: deposited recordings on template 19, which have real media
files and belong to `AI_audio_summary`, and embedded YouTube videos on template
23. Filtering by class alone would hand file-less items to a downloader.

**Items that already hold `bibo:content` are skipped.** Pass
`--include-transcribed` to re-transcribe them.

| Flag | Effect |
|---|---|
| `--item-set-id 108260` | Read this item set. Repeatable; overrides the default. |
| `--item-id 108353` | Read single items. Repeatable; skips the item-set scope. |
| `--template-id 23` | Resource template to filter on. |
| `--all-templates` | Do not filter by template — items are still filtered by whether `fabio:hasURL` parses as a canonical watch URL. |
| `--include-transcribed` | Include items that already have `bibo:content`. |

Only `youtube.com/watch?v=<id>` and `youtu.be/<id>` are accepted — the same two
forms Omeka's own ingester accepts. A `/shorts/`, `/live/` or `/embed/` link is
reported and skipped rather than sent to the API to fail.

## Automatic language detection

`dcterms:language` is catalogued per item and is not reliable enough to prompt
from on this material: the collection's 46 YouTube items are *all* catalogued
`Français`, and the first one tested is dominated by **Mooré**. An unprompted
model transcribing Mooré tends to render it as approximate French, which reads as
a clean transcript and is not one.

So `02` makes a cheap first pass before transcribing: two 45-second windows,
sampled at 10% and 55% of the runtime, answered against a small JSON schema. Two
windows rather than one because these videos open on a channel jingle or a French
title card as often as on the speech; ~9k tokens, under 5% of a 33-minute
transcription request.

The detected languages are then named in the transcription prompt, written into
the transcript header, and compared against the catalogue record. A disagreement
is reported per item and collected in `Transcriptions/_language_report.json` —
a `dcterms:language` that contradicts the audio is a metadata correction this
pipeline can see and does not make.

Comparison is by ISO code, not by name: the authority labels are French
(`Français`, `Haoussa`, `Ewé`, `Mooré`) while detection answers in English, so a
naive comparison reports `Français` against `French` as a mismatch — which is
exactly what the first live run did before `CATALOGUED_LANGUAGE_CODES` was added.

```bash
python 02_AI_transcribe_youtube.py --no-detect-language   # let the model work it out
python 02_AI_transcribe_youtube.py --language French      # assert it instead
```

## Correcting `dcterms:language` from what was heard (step 04)

The detection pass is evidence about the record, and `04` is what acts on it:

```bash
python 04_omeka_language_updater.py --dry-run
python 04_omeka_language_updater.py
```

`dcterms:language` is a *link* to an authority item, not a literal, so this writes
resource links through `common/omeka_link_updater.py`. Three rules shape it:

- **It only ever appends.** A catalogued language the samples did not contain is
  reported for a human to check, never removed. Detection hears 90 seconds; that
  is enough to prove a language is present and not enough to prove one is absent,
  and deleting `Français` from an item because two windows happened to be in
  Mooré would be discarding a curator's judgement on the strength of a sample.
- **`occasional` languages are skipped** by default — an isolated phrase, a
  quotation, a line of Qur'anic recitation. Cataloguing Arabic for one
  *bismillah* would drown the field it is meant to describe.
  `--include-occasional` overrides.
- **A language with no authority record is reported, not invented.** Creating one
  is a curatorial act: `04` prints what is missing, with the item set and template
  to create it in, and links nothing for it. `LANGUAGE_LABELS_BY_CODE` in
  `common/iwac_config.py` is deliberately exactly the labels that exist — adding a
  code without its record only moves the failure later. Records exist today for
  Allemand, Anglais, Arabe, Dendi, Dioula, Espagnol, Ewé, Français, Haoussa,
  Italien, Kabyè, Mooré and Slovène, and *not* for Peul, Bambara, Zarma, Yoruba or
  Wolof.

  Dioula shows the intended loop: the first full run found it on one item, `04`
  reported it as unlinkable, the record was created by hand (item 108359), the
  code was added to the map, and a re-run linked it. **Record first, then code** —
  the map is not the place to express an intention.

Authority items are resolved by *title* at runtime rather than from a table of
IDs: the IDs are assigned per installation and the ones here are scattered
(Français 8355, Espagnol 26353, Ewé 66720, Kabyè 79081).

A real example from the first run: item 108309 is catalogued `Français`, is
dominated by Mooré, and `04` adds the Mooré link while leaving `Français` in
place and flagging the item for review.

## Models

| Model | Notes |
|---|---|
| `gemini-3.5-flash-lite` | **Default.** Cheapest and fastest. |
| `gemini-3.6-flash` | Better on non-French speech in testing. |

Flash-Lite is the default because cost here scales with runtime, the corpus is
9.3 h of video today with more channels to come, and it is catalogued as
overwhelmingly French. On French speech the two are close.

**They are not close on the local languages, and the gap is not subtle.** From the
first full-corpus run — 44 videos, 9.1 h, Flash-Lite, zero API failures:

| | items | result |
|---|---:|---|
| French-dominant | 36 | all sound; median 156 wpm, worst legitimate 12-gram repeat 7× |
| Mooré-dominant | 8 | 5 sound, **3 degenerate loops** — 575×, 1,266× and 2,744× repeats of one clause, each run to the output cap |

Re-running those three on `gemini-3.6-flash` fixed all three: no looping (1–6×
repeats), 61–85 wpm, on-screen text interleaved and honest `[inaudible]` markers.
So the failure is model capability on Mooré, not the video or the prompt.

Naming the language is what makes Flash-Lite attempt Mooré at all — without
detection it renders the speech as French — but attempting is not managing.
**The recommended run is therefore hybrid:** Flash-Lite over the corpus, then
re-run whatever `_language_report.json` flags as non-French with
`--model gemini-3.6-flash` into its own output directory, and upload the two
folders separately so each carries the right `iwac:transcriptionModel`. Read even
the 3.6 Flash output as a lead rather than a quotation.

```bash
python 02_AI_transcribe_youtube.py --model gemini-3.6-flash \
    --work-list work/retry.json --output-dir Transcriptions_flash
```

## The loop guard

The looping failure above is the one this pipeline works hardest to catch, because
nothing else can see it. A model that cannot render a language does not fail
visibly: it emits one plausible clause over and over until it hits the output cap,
producing a file of the right shape, marked `Chunks: 1/1`, non-empty, and entirely
worthless. Length checks, window-completeness and `finish_reason` all pass it.
Without a guard it lands in `bibo:content` as archive full text, is exported to
Hugging Face as `OCR`, and is indexed for search.

So `looping_reason()` measures the most repeated 12-word window and rejects the
text above 20 repeats. The threshold sits in a wide observed gap: across 46 real
transcripts the sound ones peaked at **7** repeats (a prayer formula, a station
ident) and the broken ones scored **575 or more**. It is enforced twice —

- in `02`, where a looping window is discarded and re-drawn, but only **twice**
  (`LOOP_MAX_ATTEMPTS`): a loop runs to the output cap every time, so each retry
  costs a full 65k-token generation, and it signals a model that cannot render the
  language rather than an unlucky sample. A window that keeps looping is left out,
  which makes the transcript incomplete and so ineligible for upload;
- in `03`, which refuses such a file outright — and unlike incompleteness this is
  **not** overridable by `--include-incomplete`, because a looping transcript is
  not a partial transcript.

Both are **pinned** releases rather than the `gemini-flash-latest` rolling
aliases `AI_video_summary` uses, because `03` stamps provenance. A rolling alias
reports its own version as the string "Gemini Flash Latest", so a run through one
cannot confirm which model produced the text, and an annotation naming a model
the run never confirmed is provenance in name only. Each key here has an Omeka
authority item in `AI_MODEL_ITEMS`; a test enforces that.

No `temperature` is set. It is vendor-owned, and on a 40-minute transcription a
lowered one is what makes a model loop on a single paragraph for the rest of the
recording.

## Cost and the request budget

Video payload costs **~93–103 tokens per second of runtime** at the default 1 fps,
measured across four videos of this corpus: 32 tokens of audio plus ~61–71 of
frames. That is the documented *low* media-resolution rate rather than the ~300/s
default one, so Gemini already serves these YouTube videos at the cheap resolution
and raising `media_resolution` would only cost more.

| | |
|---|---|
| Measured input rate | ~93–103 tok/s at 1 fps (32 audio + ~61–71 frames) |
| Measured output rate | 3.77 tok/s (Flash-Lite), 4.90 tok/s (3.6 Flash), no thinking tokens at `minimal` |
| A 1M context window | ≈ 2.7 hours of video |
| Default window (`--chunk-minutes`) | 45 min, ≈ 280k tokens — a margin, not the limit |
| Longest video in the collection today | 33.6 min |
| Whole corpus (46 videos) | 9.26 h ≈ 3.5–3.9M input + ~0.13M output tokens |

Paid-tier cost for the whole corpus, detection pass included, at the prices
current on 2026-08-12 — check them rather than trusting this table:

| model | input | output | corpus |
|---|---|---|---|
| `gemini-3.5-flash-lite` | $0.30/M | $2.50/M | **≈ $1.45** |
| `gemini-3.6-flash` | $1.50/M | $7.50/M | **≈ $6.80** |

Input is ~80% of the bill, so `--fps` moves the cost far more than anything on the
output side. The hybrid run — Flash-Lite over everything, then 3.6 Flash on only
what the language report flags as non-French — lands near the Flash-Lite figure.

Both models have a **free tier**, and the corpus would fit in two free days under
the 8 h/day YouTube cap. Prefer paying: free-tier requests are used to improve
Google's products, and these are whole archival recordings — the same reason every
OpenRouter request in this repo is pinned to `data_collection: "deny"`.

`--fps` is the lever that lowers this further, and it is a real trade-off:

| `--fps` | tok/s | 33-min video |
|---|---|---|
| 1.0 (default) | 103.0 | 208k |
| 0.5 | 67.5 | 136k |
| 0.2 | 46.2 | 93k |

Frames are where the on-screen lower-thirds live, and on this material those
captions are what name the speakers — the sample transcript attributes turns to
"Tené Justine Kientega" and "El hadj Adama Nikiéma" because the prompt reads them
off the screen. Drop frames and the transcript loses those names.

The **free tier caps YouTube input at 8 hours per day**, which the 9.3 h corpus
exceeds; `01` warns when the collected runtime does. Hitting the cap raises
`QuotaExhaustedError`, which saves what completed and stops rather than retrying a
daily limit as though it were a transient 429.

## Long videos

Videos longer than `--chunk-minutes` are re-requested at different
`VideoMetadata` offsets — the same segmentation the audio pipeline does with
`ffmpeg`, without the download, because the same URL is simply requested at a
different window. Each window after the first re-sends
`--chunk-overlap-seconds` (default 15) of the previous one, so no utterance falls
in a gap, and the prompt is told two things that are not optional:

- timestamps are absolute positions in the full recording, or every window after
  the first restarts at `[00:00:00]` and the stitched transcript claims the whole
  recording happened in its first 45 minutes;
- skip any utterance beginning before the nominal boundary, which is how the
  duplicated overlap is resolved rather than merely moved.

Nothing in the collection triggers this today (the longest video is 33.6 min), so
treat it as tested-by-unit-test rather than proven in production. Existing
deposited extents run to `PT571M`, so future uploads will exercise it.

## Resume

`02` keeps a checkpoint in `Transcriptions/.youtube_transcription_checkpoint.json`
whose context pins the model, the prompt hash, the window geometry, the frame rate
and whether language detection ran. Re-running skips videos already transcribed;
changing any of those refuses to resume, because a corpus half-transcribed under
one prompt and half under another has nothing on the outside recording which item
got which. `--force` replaces the existing output.

A checkpoint entry only counts when its transcript is still on disk.

## What reaches Omeka

`bibo:content`, as one literal, with an `iwac:transcriptionModel` annotation
(property 315, "AI Model - Transcription"). The property has been declared in the
IWAC vocabulary since it was first uploaded and nothing wrote it until now — the
audio pipeline's `03` step stamps no provenance, so a transcription's model was
recorded nowhere.

Two deliberate differences from `AI_audio_summary/03`:

- **The file's header is stripped, not uploaded.** `bibo:content` is the archive's
  full-text field, exported to Hugging Face as `OCR` and indexed for search;
  "Generated using: Google gemini-3.6-flash" inside it would be indexed as though
  a speaker had said it. The header stays on disk, where it is auditable.
- **Transcripts are named `<item_id>.txt`**, so no `dcterms:identifier` lookup is
  needed — one fewer request per file, and no chance of matching the wrong item.

**Incomplete transcripts are held back.** When a window fails, the header records
`Chunks: 2/3` and `03` refuses the file: a transcript missing its middle third is
invisible once it is a single Omeka value. `--include-incomplete` overrides this.
No failure markers are ever written into the body, for the same reason.

The write goes through `common/omeka_text_updater.py`: the whole item is fetched
and PATCHed back (Omeka deletes any property missing from the payload), unchanged
items are skipped, and every pre-write payload is appended to `backups/` before
its PATCH — the only route back from an overwrite.

```bash
python 03_omeka_transcription_updater.py --dry-run     # report, write nothing
python 03_omeka_transcription_updater.py --yes         # unattended
python 03_omeka_transcription_updater.py --no-backup   # not recommended
```

## Transcription modes

| Prompt | Output |
|---|---|
| `1_full_video_transcription.md` | Verbatim, in the language spoken, code-switches tagged |
| `2_video_transcription_french.md` | Translated into French, original language marked per passage |
| `3_video_transcription_english.md` | Translated into English, original language marked per passage |

All three transcribe on-screen text that carries information the audio does not
— speaker captions, place and date cards, banners — and none of them describe the
picture otherwise: this is a transcript, not a shot list. Reading the screen is
what lets the transcript attribute turns to named people, and it is also where the
numbers come from: on the Mooré video the phone numbers and dates are legible on
the poster and unintelligible in the model's rendering of the speech.

**The editorial apparatus is French in prompt 1, whatever the language spoken** —
`Locuteur 1 :`, `[à l'écran : …]`, `[applaudissements]`, `[en mooré]`. This is a
francophone archive and the brackets are read by francophone researchers; only the
transcribed speech itself is in the original language. Prompt 3 uses English
markers because its whole output is English.

Select interactively, or with `--prompt 1`. Edit the files in `prompts/`; the
prompt's hash is recorded in every transcript header and in the checkpoint, so
editing one invalidates resume rather than silently mixing two versions.

## Limitations

- **Public videos only.** Gemini cannot fetch a private, unlisted, removed or
  region-blocked video; the API answers `400 INVALID_ARGUMENT` with no detail.
  Those items are reported as `unavailable` and need a deposited media file and
  the `AI_audio_summary` path instead. This is the one case this pipeline cannot
  serve at all.
- **Non-French speech is the hard case, and it is not rare here.** All 46 items are
  catalogued `Français`; the two tested so far are a French one with tagged Mooré
  passages and a Mooré-dominated one. The transcripts do render Mooré as Mooré,
  which is the right behaviour, but the orthography is not reliable and both models
  degrade on sustained local-language speech — Flash-Lite into syllable fragments,
  3.6 Flash into approximate romanisation. Treat a non-French passage as a lead,
  not a quotation, and expect the numbers in it (dates, prices, phone numbers) to
  be where it fails first.
- **Nothing verifies the transcript against the audio.** Length, timestamps,
  window completeness and degenerate repetition are checked; accuracy is not. A
  fluent transcript of a passage the model could not hear looks exactly like a
  correct one, and the loop guard only catches the failure loud enough to measure.
- **The catalogue is wrong about language on ~16% of this set.** 8 of 44 items are
  Mooré-dominant and all 44 are catalogued `Français`. That is worth knowing before
  treating `dcterms:language` as a filter for anything.

## Requirements

- `GEMINI_API_KEY` in `.env`
- `OMEKA_BASE_URL`, `OMEKA_KEY_IDENTITY`, `OMEKA_KEY_CREDENTIAL` for steps 01 and 03

## Output

```
AI_youtube_transcription/
├── work/youtube_videos.json                  # step 01's work list
├── Transcriptions/
│   ├── 108353.txt                            # header + transcript
│   ├── _language_report.json                 # step 04's input: heard vs catalogued
│   └── .youtube_transcription_checkpoint.json
├── backups/                                  # pre-write Omeka payloads (03 and 04)
└── log/
```
