# NotebookLM Exporter

This script exports newspaper articles from an Omeka S digital archive to Markdown files optimized for use with [Google NotebookLM](https://notebooklm.google.com/). NotebookLM is an AI-powered research tool that can analyze, summarize, and answer questions about uploaded documents.

## Why NotebookLM?

NotebookLM allows researchers to have AI-assisted conversations with their source materials. By exporting IWAC articles to a NotebookLM-friendly format, you can:

- Ask questions about newspaper content across multiple issues
- Generate summaries and find connections between articles
- Explore themes, people, and events mentioned in the collection
- Create audio overviews of your research materials

## Features

- **Three export modes**: Whole collection, single Item Set, or subject-based reverse lookup
- **Automatic file splitting**: Large collections are split into multiple files (max 250 articles each) to respect NotebookLM's ~500k word limit
- **Country-organized output**: Whole collection exports are organized into country subfolders
- **Clean Markdown formatting**: Articles are formatted with clear headings, metadata, and separators
- **Rich console output**: Professional terminal UI with progress bars, spinners, panels, and color-coded status indicators
- **Real-time progress tracking**: Visual progress bars with ETA for long-running exports

## Files Structure

```
NotebookLM/
├── omeka_items_to_md.py      # Main export script
├── README.md                  # This file
└── extracted_articles/        # Output directory
    ├── Benin/                 # Country subfolders (whole collection mode)
    ├── Burkina Faso/
    ├── Côte d'Ivoire/
    ├── Niger/
    ├── Togo/
    └── *.md                   # Individual exports (single set/subject mode)
```

## Prerequisites

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Omeka S API Configuration (required)
OMEKA_BASE_URL=https://your-omeka-instance.com/api
OMEKA_KEY_IDENTITY=your_api_key_identity
OMEKA_KEY_CREDENTIAL=your_api_key_credential

# Optional: Change output format (default: md)
NOTEBOOKLM_EXPORT_EXT=md  # or "txt" for plain text
```

### Required Python Packages

```bash
pip install requests python-dotenv rich
```

## Usage

### Interactive Mode

Run without arguments to get an interactive prompt with a styled menu:

```bash
python omeka_items_to_md.py
```

You'll see a welcome panel and be asked to choose:
1. **Whole IWAC collection** - Export all predefined Item Sets organized by country
2. **Single Item Set** - Export one Item Set by its numeric ID
3. **Subject reverse lookup** - Export all articles referencing a specific subject/authority item

### Command Line Mode

```bash
# Export the entire IWAC collection
python omeka_items_to_md.py all

# Export a single Item Set by ID
python omeka_items_to_md.py 60638

# Export articles referencing a subject authority
python omeka_items_to_md.py subject:12345
python omeka_items_to_md.py s:12345
python omeka_items_to_md.py --subject 12345
```

## Export Modes Explained

### Mode 1: Whole IWAC Collection

Exports all Item Sets defined in the `COUNTRY_ITEM_SETS` dictionary at the top of the script. Output is organized by country:

```
extracted_articles/
├── Benin/
│   ├── L_Observateur_60638.md
│   └── La_Croix_du_Benin_61062.md
├── Burkina Faso/
│   └── ...
└── ...
```

### Mode 2: Single Item Set

Provide an Item Set ID to export just that collection:

```bash
python omeka_items_to_md.py 60638
```

Output: `extracted_articles/<newspaper_title>_<set_id>.md`

### Mode 3: Subject Reverse Lookup

Find all articles that reference a specific subject authority (person, place, organization, topic):

```bash
python omeka_items_to_md.py subject:12345
```

This is useful for creating thematic collections, e.g., "all articles mentioning Burkina Faso" or "all articles about a specific person."

Output: `extracted_articles/<subject_title>_subject_<item_id>.md`

## Output Format

Each exported Markdown file contains articles in this format:

```markdown
# Article Title

**Newspaper:** L'Observateur
**Date:** 1998-02-16

Full article text content goes here...

---

# Next Article Title

**Newspaper:** La Croix du Benin
**Date:** 1999-03-20

Another article's content...

---
```

### Multi-part Files

Collections with more than 250 articles are automatically split:

```
newspaper_name_60638_part1.md  (articles 1-250)
newspaper_name_60638_part2.md  (articles 251-500)
newspaper_name_60638_part3.md  (articles 501+)
```

## Configuration

### Modifying Country/Item Set Mappings

Edit the `COUNTRY_ITEM_SETS` dictionary in `omeka_items_to_md.py` to customize which Item Sets belong to which country:

```python
COUNTRY_ITEM_SETS: Dict[str, List[str]] = {
    "Benin": ["60638", "61062", "2185", ...],
    "Burkina Faso": ["2199", "2200", ...],
    # Add your own countries/sets here
}
```

### Changing the Article Limit per File

Modify `MAX_ITEMS_PER_FILE` in the `main()` function:

```python
MAX_ITEMS_PER_FILE = 250  # Default: 250 articles per file
```

## Using with NotebookLM

1. **Export your articles** using one of the modes above
2. **Open [NotebookLM](https://notebooklm.google.com/)**
3. **Create a new notebook**
4. **Upload the generated `.md` files** as sources
5. **Start asking questions** about your newspaper collection!

### Tips for NotebookLM

- Keep individual files under 500,000 words (the script handles this automatically)
- Upload multiple part files for large collections
- Use specific questions referencing dates, people, or places for best results
- The "Audio Overview" feature can create podcast-style summaries of your sources

## Technical Notes

- **Filtering**: Only `bibo:Article` and `bibo:Issue` types are exported (other item types are skipped)
- **Encoding**: All files are UTF-8 encoded
- **Whitespace**: Content is normalized to clean up OCR artifacts and ensure consistent formatting
- **Pagination**: The script automatically handles Omeka's paginated API responses
- **Rate limiting**: Consider API rate limits when exporting large collections
- **Console output**: Uses the `rich` library for professional terminal UI with colors, progress bars, and spinners

## Troubleshooting

### "Missing required environment variables"

Ensure your `.env` file contains all required Omeka credentials:
- `OMEKA_BASE_URL`
- `OMEKA_KEY_IDENTITY`
- `OMEKA_KEY_CREDENTIAL`

### Empty output files

- Check that the Item Set contains `bibo:Article` or `bibo:Issue` items
- Verify the Item Set ID is correct
- Check API credentials have read access to the collection

### Large exports timing out

- Try exporting individual Item Sets instead of the whole collection
- Check your network connection
- The script includes progress tracking to monitor long-running exports

## See Also

- [Google NotebookLM](https://notebooklm.google.com/) - AI research assistant
- [Omeka S Documentation](https://omeka.org/s/docs/user-manual/) - Content management system
- [Islam West Africa Collection](https://islam.zmo.de/s/westafrica/) - The IWAC digital archive
