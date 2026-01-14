# NotebookLM Exporter

Export newspaper articles from Omeka S to Markdown files optimized for [Google NotebookLM](https://notebooklm.google.com/).

## Why NotebookLM?

NotebookLM is an AI research assistant that can analyze, summarize, and answer questions about uploaded documents. By exporting IWAC articles to NotebookLM, you can:

- Ask questions across multiple newspaper issues
- Find connections between articles and themes
- Generate audio overviews of your research materials
- Explore people, places, and events mentioned in the collection

## How It Works

```
Omeka S collection → Export to Markdown → Upload to NotebookLM → AI-assisted research
```

The script fetches articles from Omeka S and formats them for NotebookLM's ingestion, automatically splitting large collections into multiple files.

## Quick Start

```bash
python omeka_items_to_md.py
```

Choose from three export modes:
1. **Whole IWAC collection** — All predefined item sets, organized by country
2. **Single Item Set** — One collection by ID
3. **Subject lookup** — All articles referencing a specific authority

Or use command line:
```bash
python omeka_items_to_md.py all              # Whole collection
python omeka_items_to_md.py 60638            # Single item set
python omeka_items_to_md.py subject:12345    # Subject lookup
```

## Output

Files are saved to `extracted_articles/`:

```
extracted_articles/
├── Benin/
│   ├── L_Observateur_60638.md
│   └── La_Croix_du_Benin_61062.md
├── Burkina Faso/
│   └── ...
└── subject_Islam_12345.md
```

Large collections (>250 articles) are automatically split:
```
newspaper_name_60638_part1.md
newspaper_name_60638_part2.md
```

## Using with NotebookLM

1. Export articles using this script
2. Open [NotebookLM](https://notebooklm.google.com/)
3. Create a new notebook
4. Upload the `.md` files as sources
5. Start asking questions about your collection

**Tips**:
- Upload multiple part files for large collections
- Use specific questions referencing dates, people, or places
- Try the "Audio Overview" feature for podcast-style summaries

## Limitations

**File size**: NotebookLM has a ~500k word limit per source. The script auto-splits at 250 articles to stay within limits.

**Article types only**: Exports `bibo:Article` and `bibo:Issue` items; other item types are skipped.

**Text content only**: Images and media attachments are not included.

**Omeka S dependency**: Requires API access to an Omeka S instance with the IWAC collection.

## Configuration

Create `.env` in project root:

```bash
OMEKA_BASE_URL=https://your-instance.com/api
OMEKA_KEY_IDENTITY=your_key
OMEKA_KEY_CREDENTIAL=your_credential
```

## Customization

Edit `COUNTRY_ITEM_SETS` in the script to modify which item sets belong to which country for whole-collection exports.

Adjust `MAX_ITEMS_PER_FILE` to change the split threshold (default: 250).

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Missing environment variables | Check `.env` contains all Omeka credentials |
| Empty output files | Verify item set contains `bibo:Article` items |
| Large exports timing out | Export individual item sets instead |
