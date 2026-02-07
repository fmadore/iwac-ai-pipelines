# Handwritten Text Recognition (HTR)

Transcribe handwritten manuscripts in French, Arabic, or other languages using Gemini vision models.

## Why This Tool?

Handwritten documents—letters, manuscripts, field notes—are often the most valuable and least accessible items in an archive. Traditional OCR fails on handwriting. AI vision models can read diverse handwriting styles, recognize multiple scripts, and handle mixed-language documents.

## How It Works

```
PDF scans → Page-by-page PDF processing → AI transcription → Text output
```

The script extracts individual pages from each PDF and sends them directly to Gemini's vision model with language-specific instructions, outputting the transcribed text. No image conversion is needed.

## Quick Start

1. Place PDF files in the `PDF/` folder
2. Run the script:
   ```bash
   python 02_gemini_htr_processor.py
   ```
3. Select language mode (French, Arabic, or Multilingual)
4. Select model (Flash or Pro)
5. Find transcriptions in `OCR_Results/`

## Language Modes

| Mode | Script Direction | Best For |
|------|------------------|----------|
| **French** | Left-to-right | French manuscripts, European handwriting |
| **Arabic** | Right-to-left | Arabic manuscripts, traditional orthography |
| **Multilingual** | Auto-detected | Mixed scripts, other languages |

Each mode uses a specialized prompt (`htr_system_prompt_*.md`) with language-specific rules.

## Supported Models

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| Gemini 3 Flash | Faster | Good | Clear handwriting, bulk processing |
| Gemini 3 Pro | Slower | Higher | Difficult scripts, faded ink |

## Output

Transcriptions are saved to `OCR_Results/` as `.txt` files with page markers for multi-page documents.

**Multilingual mode** includes a header:
```
[LANGUAGE DETECTED: Russian]
[WRITING SYSTEM: Cyrillic]
[TEXT DIRECTION: Left-to-right]

[transcribed text]
```

## Limitations

**Handwriting variability**: Results depend heavily on handwriting legibility. Hasty or idiosyncratic scripts produce lower accuracy.

**Faded or damaged documents**: Very faint text or damaged pages may be partially or incorrectly transcribed.

**Mixed scripts**: Documents mixing Latin, Arabic, and other scripts in the same line may have inconsistent results.

**No spell-checking**: The model transcribes what it sees, including the writer's spelling errors.

**Marginalia and annotations**: Side notes may be missed or transcribed out of order depending on layout complexity.

## Requirements

### Configuration

Create `.env` in project root:

```bash
GEMINI_API_KEY=your_key
```

## Customization

Edit the prompt files to adjust transcription behavior:
- `htr_system_prompt_french.md` — French de-hyphenation, spacing rules
- `htr_system_prompt_arabic.md` — Right-to-left handling, ligatures
- `htr_system_prompt_multilingual.md` — Auto-detection protocol

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Empty transcriptions | Try different model (Pro for difficult scripts) |
| Safety/copyright blocks | Check logs; some content may be filtered |
| Wrong language detected | Use specific language mode instead of Multilingual |
