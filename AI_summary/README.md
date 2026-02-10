# Document Summarization

Generate French summaries of text content from Omeka S collections using AI.

## Why This Tool?

Large document collections are difficult to browse. Researchers need to open each item to understand its contents. AI-generated summaries enable quick scanning and improve search relevance—users can find documents by concept rather than exact keyword matches.

## How It Works

```
Omeka S → Extract text → AI summarization → Update database with summaries
```

1. **Extract** (`01_extract_omeka_content.py`): Pull OCR text from Omeka S items
2. **Summarize** (`02_AI_generate_summaries.py`): Generate French summaries
3. **Update** (`03_omeka_update_summaries.py`): Store summaries in Omeka S

## Quick Start

```bash
python 01_extract_omeka_content.py    # Extract text from item set
python 02_AI_generate_summaries.py    # Generate summaries
python 03_omeka_update_summaries.py   # Update Omeka S items
```

The summarization script prompts you to select a model interactively, or use `--model`:

```bash
python 02_AI_generate_summaries.py --model gemini-flash
```

## Supported Models

| Model | Provider | Speed | Cost |
|-------|----------|-------|------|
| `gemini-flash` | Google | Fast | Low |
| `gpt-5-mini` | OpenAI | Fast | Low |
| `ministral-14b` | Mistral | Fast | Very low ($0.2/M tokens) |

All models produce comparable summary quality for this task.

## Output

Summaries are saved to `Summaries_FR_TXT/` as `.txt` files, then uploaded to the `dcterms:abstract` field in Omeka S.

## Limitations

**Summarization is lossy**: Summaries compress information. Important details may be omitted, especially from long or complex documents.

**Hallucination risk**: AI may occasionally include information not present in the source text. Summaries are aids for discovery, not substitutes for reading originals.

**Language**: Summaries are generated in French regardless of source language. The prompt assumes French input.

**RAG optimization**: Summaries are designed to be concise and keyword-rich for retrieval-augmented generation. They prioritize searchability over narrative flow.

## Configuration

Create `.env` in project root:

```bash
OMEKA_BASE_URL=https://your-instance.com/api
OMEKA_KEY_IDENTITY=your_key
OMEKA_KEY_CREDENTIAL=your_credential

# At least one AI provider
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key
MISTRAL_API_KEY=your_key
```

## Customization

Edit `summary_prompt.md` to adjust:
- Summary language
- Length and style
- What to emphasize (people, events, themes)
- Output format

The `{text}` placeholder is replaced with document content at runtime.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Empty summaries | Check source text exists in `TXT/` folder |
| Missing prompt error | Ensure `summary_prompt.md` exists in script directory |
| API authentication | Verify correct API key in `.env` |
| Items not updating | Check Omeka S credentials have write access |
