# OCR Extraction

Extract text from PDF scans using AI-powered document processing with Gemini or Mistral.

## Why This Tool?

Traditional OCR tools struggle with historical documents, poor scan quality, and complex layouts like newspaper columns. AI vision models understand document structure and can handle French typography rules, de-hyphenation, and reading order—producing research-grade text from challenging sources.

## How It Works

```
Omeka S → Download PDFs → AI OCR processing → Update database with extracted text
```

1. **Download** (`01_omeka_pdf_downloader.py`): Fetch PDFs from Omeka S collection
2. **Extract** (`02_gemini_ocr_processor.py` or `02_mistral_ocr_processor.py`): AI reads each page
3. **Update** (`03_omeka_content_updater.py`): Store extracted text in Omeka S

## Quick Start

```bash
python 01_omeka_pdf_downloader.py   # Download PDFs from item set
python 02_gemini_ocr_processor.py   # Extract text (or use Mistral version)
python 03_omeka_content_updater.py  # Update Omeka S items
```

## Provider Comparison

| Provider | Approach | Best For |
|----------|----------|----------|
| **Gemini** | Vision model with custom prompts | French typography, column layouts, guided extraction |
| **Mistral** | Dedicated OCR endpoint | Fast raw extraction, Markdown output |

### Gemini Models

| Model | Speed | Quality | Cost | Notes |
|-------|-------|---------|------|-------|
| Gemini 3 Flash | Fast | Good | Low | Recommended for most documents |
| Gemini 3 Pro | Slower | Higher | Higher | Complex layouts, poor scans |

**Recommendation**: Start with Gemini Flash. Switch to Pro for challenging documents or frequent copyright blocks.

## Output

Extracted text is saved to `OCR_Results/` as `.txt` files, one per PDF. Files include page markers:

```
--- Page 1 ---
[extracted text]

--- Page 2 ---
[extracted text]
```

## Limitations

**Copyright detection**: Gemini may refuse to process pages it identifies as copyrighted material (RECITATION error). The script logs these and continues. Gemini Pro is less restrictive than Flash.

**Column reading order**: Multi-column layouts are generally handled well, but unusual arrangements may produce out-of-order text.

**Image-heavy pages**: Pages with mostly images and little text may return sparse results.

**Handwritten text**: This pipeline is optimized for printed text. Use `AI_htr_extraction/` for handwritten documents.

**Non-Latin scripts**: Arabic text extraction works but may need prompt adjustments for optimal results.

## Configuration

Create `.env` in project root:

```bash
OMEKA_BASE_URL=https://your-instance.com/api
OMEKA_KEY_IDENTITY=your_key
OMEKA_KEY_CREDENTIAL=your_credential

GEMINI_API_KEY=your_key    # For Gemini OCR
MISTRAL_API_KEY=your_key   # For Mistral OCR
```

## Customization

Edit `ocr_system_prompt.md` to adjust extraction behavior:
- Reading order rules (columns, sidebars, captions)
- French typography spacing (`:`, `;`, `!`, `?`)
- De-hyphenation rules
- What to preserve vs. normalize

## Troubleshooting

| Problem | Solution |
|---------|----------|
| RECITATION errors | Try Gemini Pro instead of Flash |
| Empty output files | Check PDF isn't corrupted or password-protected |
| Page too large | Script auto-uploads large files; no action needed |
| Out-of-order text | Adjust reading order rules in prompt |
| API rate limits (transient) | Built-in retry with backoff handles this automatically |
| Quota exhausted (daily limit) | Pipeline stops immediately and saves partial results; wait for reset or use `--rpm` to throttle |

## Rate Limiting

The Gemini free tier has strict limits (5 RPM, 100 requests/day). The script handles this automatically:

- **Quota exhaustion**: When daily quota is hit, the pipeline stops immediately and saves any partial results — no wasted retries
- **Proactive throttling**: Use `--rpm` to space requests and avoid hitting limits:
  ```bash
  python 02_gemini_ocr_processor.py --rpm 5   # Free tier: 5 requests/minute
  ```
- **Transient 429s**: Retried with exponential backoff + jitter (5s base delay)

## Performance

- **Processing speed**: ~20-40 seconds per page
- **Cost**: ~$0.54 per 450-page document (Gemini Flash)
- **Token usage**: Fixed 258 tokens per page for PDF processing
