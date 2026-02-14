# OCR Post-Correction

Fix errors in machine-generated OCR text using AI, with optional coordinate preservation for ALTO XML.

## Why This Tool?

Raw OCR output contains systematic errors: `rn` misread as `m`, `li` as `d`, encoding artifacts like `cafÃ©`. These errors compound in downstream tasks like named entity recognition and full-text search. AI correction fixes character-level errors while preserving historical spelling and document structure.

## How It Works

Two correction approaches are available:

### Plain Text Correction
```
Raw OCR text → AI correction → Clean text
```
Fixes errors and reconstructs paragraphs. Best for display and search.

### ALTO XML Correction
```
ALTO XML → AI correction (token-aligned) → ALTO XML with coordinates preserved
```
Corrects text while maintaining word-level coordinates. Essential when coordinates matter for display or annotation.

## Quick Start

### Plain Text Workflow

```bash
python 01_extract_ocr_text.py      # Extract text from Omeka S
python 02_correct_ocr_text.py      # Apply AI corrections
python 03_update_database.py       # Update Omeka S
```

### ALTO XML Workflow

```bash
# Place ALTO XML files in ALTO/ folder
python 02_correct_alto_xml.py --model gemini-flash
# Corrected files saved to ALTO_Corrected/
```

## Supported Models

| Model | Speed | Best For |
|-------|-------|----------|
| `gemini-flash` | Fast | Bulk processing |
| `gemini-pro` | Slower | Complex documents |
| `gpt-5-mini` | Fast | General use |
| `gpt-5.1` | Slower | Higher quality |
| `mistral-large` | Medium | European languages |
| `ministral-14b` | Fast | Budget option |

## What Gets Corrected

**Fixed**:
- Character recognition errors: `Tlie` → `The`, `rn` → `m`
- Encoding issues: `cafÃ©` → `café`
- Obvious OCR artifacts: `gouvernernent` → `gouvernement`

**Preserved**:
- Historical spellings: `connexion`, `shew`, `to-day`
- Original language (no translation)
- Document structure and formatting

## ALTO XML: Coordinate Preservation

The ALTO correction enforces strict token alignment:

```xml
<!-- Before -->
<String CONTENT="Tlie" HPOS="100" VPOS="50" WIDTH="40"/>
<String CONTENT="gouvernernent" HPOS="145" VPOS="50" WIDTH="120"/>

<!-- After: only CONTENT changes -->
<String CONTENT="The" HPOS="100" VPOS="50" WIDTH="40"/>
<String CONTENT="gouvernement" HPOS="145" VPOS="50" WIDTH="120"/>
```

**Limitation**: Token count must match. Words needing merge (`ilest` → `il est`) or split are flagged but left unchanged to preserve coordinates.

## Limitations

**Silent errors**: Unlike garbled OCR, AI corrections appear as fluent text. Errors may be harder to spot.

**Over-correction**: Historical or unusual spellings may be "corrected" to modern forms despite instructions to preserve them.

**Token alignment (ALTO)**: Cannot merge or split words without breaking coordinate mapping.

**Context window**: Very long documents are chunked, potentially losing cross-page context.

## Configuration

Create `.env` in project root:

```bash
OMEKA_BASE_URL=https://your-instance.com/api
OMEKA_KEY_IDENTITY=your_key
OMEKA_KEY_CREDENTIAL=your_credential

GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key
MISTRAL_API_KEY=your_key
```

## Customization

- `ocr_correction_prompt.md` — Plain text correction rules
- `alto_correction_prompt.md` — Token-aligned ALTO correction rules

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Token count mismatch | Script falls back to original; check for merged/split words |
| Historical spellings changed | Adjust prompt with more examples |
| ALTO namespace errors | Script auto-detects v2/v3/v4; check XML validity |
| API rate limits | Built-in retry handles this |
