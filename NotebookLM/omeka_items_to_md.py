"""
Omeka item to NotebookLM-ready Markdown exporter.

This script fetches newspaper articles from an Omeka S digital archive and
exports them to consolidated UTF-8 Markdown files optimized for ingestion into
Google's NotebookLM AI research tool.

=== Overview ===

The Islam West Africa Collection (IWAC) holds digitised West African newspaper
articles from the 1960s onwards. This script helps researchers export a
selection of them in a format that NotebookLM can process for analysis,
summarisation and question-answering.

=== Export Modes ===

Input
- Option A: Export the whole IWAC collection (predefined country -> item set IDs)
- Option B: Export a single Item Set by ID
- Environment: OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL in .env

How it works
1) Loads credentials from .env
2) In "Whole IWAC" mode: iterate predefined Item Set IDs per country
3) In "Single Item Set" mode: process only the provided Item Set ID
4) For each item set, list items (paginated), keep only "bibo:Article" items
5) For each article, extracts:
     - TITLE: o:title
     - NEWSPAPER: dcterms:publisher (display_title)
     - DATE: dcterms:date (first @value)
     - CONTENT: bibo:content (first @value)
6) Writes a Markdown-friendly .md file with clear separators per article

Option C: Items by subject Item ID (reverse links)
- Provide an Omeka Item ID, the script looks up its "@reverse.dcterms:subject"
    references, fetches those items, keeps only "bibo:Article", and exports them.

Multi-part Output
- If more than 250 articles are found (to respect NotebookLM's 500k word limit),
    the output is automatically split into multiple files:
    - <item-title>_articles_part1.md
    - <item-title>_articles_part2.md
    - etc.

Output
- File path: NotebookLM/extracted_articles/<item-title>_articles.md
    (or _part1.md, _part2.md, etc. for large collections)
- Encoding: UTF-8
- Layout: one H1 naming the file's source (item set, or subject + publisher,
    plus the part number when split), then one H2 per article:

        # Item Set: <set title> (ID <set id>) — Part 1 of 3

        ## <title>
        **Journal :** <publisher(s)> | **Date :** <date> | **Pays :** <country> | **Item :** [<id>](<url>)

        <article body>

        ---

Optional AI summaries
- ``--with-summaries`` inlines each article's stored ``bibo:shortDescription``
    as a "Résumé (IA)" line. The summary is read from the item JSON already
    fetched, so it costs no extra API call and no model call — this script never
    generates text. It is off by default because it puts machine-written prose
    into a corpus of primary sources, which NotebookLM will cite as readily as
    it cites the newspaper.

Notes
- The format favors readability and simple chunking for tools like NotebookLM.
- French labels are used (Journal, Date, Pays) since most content is in French.
- The item link makes a NotebookLM citation traceable back to the Omeka record.
- Country is extracted from the publisher's "Spatial Coverage" (dcterms:spatial) field.
- Only the first value is taken for multi-valued fields (date/content) to avoid
    duplications and keep the file compact.
- Maximum 250 articles per file to stay within NotebookLM's word limits.
"""

import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich import box

# Initialize rich console for styled output
console = Console()

# Shared Omeka client
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.omeka_client import OmekaClient
from common.checkpoint import atomic_write_text
from common.iwac_config import item_page_url
from common.log_redaction import install_credential_redaction

# Credentials ride in Omeka query strings and provider headers; keep them
# out of anything urllib3 or an SDK decides to log.
install_credential_redaction()

# Lightweight aliases to make intent clearer when reading types
JSONObj = Dict[str, Any]
JSONLike = Union[Dict[str, Any], List[Any]]

# -----------------------------------------------------------------------------
# Configuration: Country -> Item Set IDs mapping for "Whole IWAC" export
# 
# This dictionary maps West African countries to their corresponding Omeka Item Set IDs.
# Each Item Set contains newspaper collections from that country.
# Fill these lists with the Omeka Item Set IDs (as strings) that belong to each
# country. Leave a list empty to skip that country.
#
# Example: Item Set "60638" might contain "L'Observateur" newspaper from Benin
COUNTRY_ITEM_SETS: Dict[str, List[str]] = {
    "Benin": [
        "60638", "61062", "2185", "5502", "75959", "2186", "2188", "2187", "2191", "2190", "2189", "4922", "76053", "76081", "76059", "5501", "76070", "75960", "76071", "61063", "5500", "76072", "76073"
    ],
    "Burkina Faso": [
        "2199", "2200", "23448", "23273", "23449", "5503", "2215", "2214", "2207", "2209", "2210", "2213", "2201", "75969"
    ],
    "Côte d'Ivoire": [
        "43051", "76357", "62076", "31882", "57945", "63444", "76253", "61684", "76239", "48249", "57943", "57944", "61320", "15845", "76364", "73533", "61289", "45390", "76534", "77882"
    ],
    "Niger": [
        "62021",
    ],
    "Togo": [
        "67437", "25304", "67399", "9458", "67407", "67460", "67480", "67430", "5498", "67436", "67456", "5499"
    ],
}
MAX_ITEMS_PER_FILE = 250

# Fail loudly on malformed entries: a non-numeric string would otherwise be
# silently skipped at export time, dropping whole collections.
for _country, _set_ids in COUNTRY_ITEM_SETS.items():
    for _sid in _set_ids:
        if not isinstance(_sid, str) or not _sid.isdigit():
            raise ValueError(
                f"COUNTRY_ITEM_SETS[{_country!r}] contains a malformed item-set ID: {_sid!r}"
            )


def sanitize_filename(name: str) -> str:
    """Return a filesystem-safe filename across Windows/macOS/Linux.

    Args:
        name: Candidate filename (usually an Omeka title).

    Returns:
        A sanitized string (no path separators or illegal characters) trimmed to
        a reasonable length. Falls back to "untitled" if blank.
    """
    name = name.strip()
    # Replace path separators and illegal chars with underscore
    name = re.sub(r"[\\/\:*?\"<>|]", "_", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name)
    # Trim and limit length
    return name[:180] or "untitled"


def normalize_md_whitespace(text: str) -> str:
    """Normalize whitespace to keep Markdown tidy and consistent.

    This function cleans up common formatting issues in text content:
    - Standardizes different line ending types (Windows CRLF, Mac CR) to Unix LF
    - Converts non-breaking spaces (often from OCR/HTML) to regular spaces
    - Removes trailing whitespace at the end of lines
    - Collapses multiple consecutive blank lines into single blank lines
    - Removes leading and trailing blank lines from the entire text

    Args:
        text: Raw text content that may have inconsistent whitespace.

    Returns:
        Cleaned text with normalized whitespace, ready for Markdown output.
    """
    if not text:
        return text
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # NBSP to normal space
    text = text.replace("\u00A0", " ")
    # Trim trailing spaces at line ends
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    # Collapse 2+ blank lines to a single blank line
    text = re.sub(r"(\n[\t ]*){3,}", "\n\n", text)
    # Remove leading/trailing blank lines
    return text.strip()


def extract_first_value(prop: Any) -> Optional[str]:
    """Extract the first "@value" from an Omeka JSON-LD multi-valued property.

    Omeka stores many properties as JSON-LD arrays of value objects, even for
    single values. For example, a date might be stored as:
        [{"@value": "1998-02-16", "@type": "literal"}]
    
    This function extracts just the first actual value, which is often sufficient
    for our purposes and avoids duplication in the output.

    Args:
        prop: A property value from Omeka JSON (could be list, dict, string, etc.).

    Returns:
        The first @value found as a string, or None if no @value is found.
        
    Example:
        extract_first_value([{"@value": "1998-02-16"}, {"@value": "1998"}]) 
        # Returns: "1998-02-16"
    """
    if isinstance(prop, list):
        for entry in prop:
            if isinstance(entry, dict) and "@value" in entry:
                return str(entry["@value"]).strip()
    return None


def extract_publishers(article: JSONObj) -> List[str]:
    """Extract unique publisher names from an article's dcterms:publisher property.

    Publishers in Omeka are stored as related items with display_title fields.
    This function extracts the human-readable names of newspapers/publications
    that published this article.

    Args:
        article: Omeka JSON object representing a bibo:Article item.

    Returns:
        List of unique publisher display names (e.g., ["L'Observateur", "Le Matin"]).
        Returns empty list if no publishers are found.
        
    Example JSON structure:
        "dcterms:publisher": [
            {"display_title": "L'Observateur", "@id": "..."},
            {"display_title": "Le Matin", "@id": "..."}
        ]
    """
    publishers: List[str] = []
    pub_list = article.get("dcterms:publisher")
    if isinstance(pub_list, list):
        for p in pub_list:
            if isinstance(p, dict):
                name = p.get("display_title")
                if name and name not in publishers:
                    publishers.append(str(name))
    return publishers


def extract_summary(article: JSONObj, language: str = "fr") -> Optional[str]:
    """Extract the AI-written summary in one language from bibo:shortDescription.

    Since 2026-08-06 the property carries two literals, tagged ``@language``
    ``fr`` and ``en``. The summaries written before then carry no tag at all
    and are French, so an untagged literal counts as French — the same rule the
    write side applies through ``PropertyTarget(adopt_untagged=True)``.

    Args:
        article: Omeka JSON object representing a bibo:Article item.
        language: BCP-47 tag to return ("fr" or "en").

    Returns:
        The summary text, or None when the article carries none in that
        language. Nothing is generated here: this reads a field the AI_summary
        pipeline already wrote.
    """
    values = article.get("bibo:shortDescription")
    if not isinstance(values, list):
        return None

    untagged: Optional[str] = None
    for entry in values:
        if not isinstance(entry, dict) or "@value" not in entry:
            continue
        text = str(entry["@value"]).strip()
        if not text:
            continue
        tag = entry.get("@language")
        if tag == language:
            return text
        if not tag and untagged is None:
            untagged = text

    # An untagged literal is a legacy French summary; it answers "fr" only.
    return untagged if language == "fr" else None


def extract_country_from_item_set(item_set: JSONObj) -> Optional[str]:
    """Extract the country from an item set's dcterms:spatial field.

    Args:
        item_set: Omeka JSON object representing an Item Set.

    Returns:
        The country name from the item set's spatial coverage, or None if not found.
    """
    spatial_list = item_set.get("dcterms:spatial")
    if not isinstance(spatial_list, list):
        return None
    
    for s in spatial_list:
        if isinstance(s, dict):
            # Try display_title first (linked item), then @value (literal)
            country = s.get("display_title") or s.get("@value")
            if country:
                return str(country).strip()
    
    return None


# Cache for item set countries to avoid repeated API calls
_item_set_country_cache: Dict[str, Optional[str]] = {}


def extract_article_country(article: JSONObj, client: OmekaClient) -> Optional[str]:
    """Extract the country from an article's item set's dcterms:spatial field.

    Fetches the article's item set from Omeka and looks up its Spatial Coverage field.
    Results are cached to avoid repeated API calls for the same item set.

    Args:
        article: Omeka JSON object representing a bibo:Article item.
        client: OmekaClient instance.

    Returns:
        The country name from the item set's spatial coverage, or None if not found.
    """
    # Get the article's item set(s)
    item_set_list = article.get("o:item_set")
    if not isinstance(item_set_list, list) or not item_set_list:
        return None

    # Get the first item set's ID
    first_set = item_set_list[0]
    if not isinstance(first_set, dict):
        return None

    # Get item set ID from o:id or @id
    set_id = first_set.get("o:id")
    if set_id:
        set_id = str(set_id)
    else:
        set_id_url = first_set.get("@id")
        if set_id_url:
            m = re.search(r"/item_sets/(\d+)$", set_id_url)
            set_id = m.group(1) if m else None

    if not set_id:
        return None

    # Check cache first
    if set_id in _item_set_country_cache:
        return _item_set_country_cache[set_id]

    # Fetch the item set
    item_set = client.get_item_set(int(set_id))
    if not item_set:
        _item_set_country_cache[set_id] = None
        return None

    # Extract country from item set's spatial coverage
    country = extract_country_from_item_set(item_set)
    _item_set_country_cache[set_id] = country
    return country


def format_article(
    article: JSONObj,
    client: Optional[OmekaClient] = None,
    country: Optional[str] = None,
    include_summary: bool = False,
) -> str:
    """Convert a single newspaper article into NotebookLM-friendly Markdown format.

    Creates a standardized Markdown block for each article with:
    - Level 2 heading with the article title (level 1 is the file itself)
    - A single metadata line: Journal, Date, Pays, and a link back to the Omeka
      record, so a NotebookLM citation stays traceable to its source
    - Optionally the AI-written summary, explicitly labelled as such
    - The full article content (cleaned of extra whitespace)
    - A horizontal rule separator for visual separation

    This format is optimized for NotebookLM ingestion and human readability.
    French labels are used (Journal, Date, Pays) as most content is in French.

    Args:
        article: Omeka JSON object representing a bibo:Article with fields like
                o:title, dcterms:date, dcterms:publisher, bibo:content.
        client: Optional OmekaClient instance, used to resolve the article's
                country and to build the item link from its base URL.
        country: Optional country name to include in metadata. If not provided,
                 will attempt to extract from publisher's spatial coverage.
        include_summary: Emit the stored ``bibo:shortDescription`` summary above
                the body. Off by default: it is machine-written text entering a
                corpus of primary sources, and NotebookLM will cite it as
                readily as it cites the newspaper.

    Returns:
        Formatted Markdown string ready to write to file, including trailing
        horizontal rule and newlines for proper separation between articles.

    Example output:
        ## Article Title Here
        **Journal :** L'Observateur | **Date :** 1998-02-16 | **Pays :** Bénin | **Item :** [2233](https://islam.zmo.de/s/afrique_ouest/item/2233)

        **Résumé (IA) :** One or two sentences written by the summary pipeline.

        Article content goes here with proper formatting...

        ---
    """
    title = article.get("o:title") or "No title"
    date = extract_first_value(article.get("dcterms:date")) or "Inconnu"
    content = extract_first_value(article.get("bibo:content")) or ""
    content = normalize_md_whitespace(content)
    publishers = extract_publishers(article)
    publisher_str = "; ".join(publishers) if publishers else "Inconnu"

    # Use provided country, or try to extract from article's item set if client is provided
    if not country and client:
        country = extract_article_country(article, client)

    meta = [f"**Journal :** {publisher_str}", f"**Date :** {date}"]
    if country:
        meta.append(f"**Pays :** {country}")
    item_id = article.get("o:id")
    if item_id:
        # The link is what makes a NotebookLM answer checkable against the
        # archive; without a client we still emit the bare ID to cite.
        if client:
            url = item_page_url(client.base_url, item_id)
            meta.append(f"**Item :** [{item_id}]({url})")
        else:
            meta.append(f"**Item :** {item_id}")

    lines = [f"## {title}", " | ".join(meta), ""]
    if include_summary:
        summary = extract_summary(article)
        if summary:
            # Labelled, so a grounded answer leaning on it is visibly doing so.
            lines.append(f"**Résumé (IA) :** {normalize_md_whitespace(summary)}")
            lines.append("")
    if content:
        lines.append(content)
        lines.append("")
    # Markdown horizontal rule between articles
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def fetch_item(client: OmekaClient, item_id: str) -> Optional[JSONObj]:
    """Fetch a single Item by ID.

    Args:
        client: OmekaClient instance.
        item_id: Numeric string Item ID.

    Returns:
        The item JSON or None.
    """
    return client.get_item(int(item_id))


def parse_id_from_at_id(at_id: str) -> Optional[str]:
    """Extract numeric ID from an Omeka '@id' URL reference.

    Omeka JSON-LD uses '@id' fields containing full URLs to reference related items.
    This function extracts just the numeric ID portion for API calls.

    Args:
        at_id: Full Omeka URL like "https://example.org/api/items/23601" or
               "https://example.org/api/resources/5717"

    Returns:
        Just the numeric ID as a string (e.g., "23601") or None if no valid ID found.
        
    Example:
        parse_id_from_at_id("https://example.org/api/items/23601") → "23601"
        parse_id_from_at_id("https://example.org/api/resources/5717") → "5717"
    """
    if not isinstance(at_id, str):
        return None
    m = re.search(r"/(?:items|resources)/(\d+)$", at_id)
    return m.group(1) if m else None


def fetch_resource(client: OmekaClient, resource_id: str) -> Optional[JSONObj]:
    """Fetch a generic resource by ID via the '/resources' endpoint.

    Useful when reverse links use '/api/resources/<id>' rather than '/items'.
    """
    url = f"{client.base_url}/resources/{resource_id}"
    return client.get_resource(url)


def fetch_item_or_resource(client: OmekaClient, id_str: str) -> Optional[JSONObj]:
    """Try fetching an Item by ID, with fallback to the generic Resource endpoint.

    Args:
        client: OmekaClient instance.
        id_str: Numeric ID as a string.

    Returns:
        The JSON object for the item/resource, or None if both endpoints fail.
    """
    it = fetch_item(client, id_str)
    if isinstance(it, dict):
        return it
    return fetch_resource(client, id_str)


def _reference_item_id(reference: JSONObj) -> Optional[str]:
    """Extract an Omeka item ID from a reverse-link representation."""
    item_id = reference.get("o:id")
    if isinstance(item_id, int):
        return str(item_id)
    if isinstance(item_id, str) and item_id.isdigit():
        return item_id
    at_id = reference.get("@id")
    return parse_id_from_at_id(at_id) if isinstance(at_id, str) else None


def _fetch_subject_reference(
    client: OmekaClient,
    reference: Any,
    seen_ids: set[str],
) -> tuple[str, Optional[JSONObj]]:
    """Resolve one reverse link and classify its article-fetch outcome."""
    if not isinstance(reference, dict):
        return "invalid", None
    ref_types = reference.get("@type")
    if isinstance(ref_types, list) and "bibo:Article" not in ref_types:
        return "non_article", None
    resource_id = _reference_item_id(reference)
    if not resource_id:
        return "invalid", None
    if resource_id in seen_ids:
        return "duplicate", None
    seen_ids.add(resource_id)

    item = fetch_item_or_resource(client, resource_id)
    if not isinstance(item, dict):
        return "fetch_error", None
    item_types = item.get("@type", [])
    if not isinstance(item_types, list) or "bibo:Article" not in item_types:
        return "non_article", None
    return "article", item


def _show_subject_fetch_stats(stats: Counter) -> None:
    stats_table = Table(show_header=False, box=box.SIMPLE)
    stats_table.add_column("Metric", style="dim")
    stats_table.add_column("Value", style="green")
    stats_table.add_row("Articles collected", str(stats["article"]))
    stats_table.add_row("Skipped (non-article)", str(stats["non_article"]))
    stats_table.add_row("Duplicates", str(stats["duplicate"]))
    stats_table.add_row("Fetch errors", str(stats["fetch_error"]))
    console.print(stats_table)


def fetch_articles_with_subject(client: OmekaClient, subject_item_id: str) -> Tuple[Optional[JSONObj], List[JSONObj]]:
    """Find all newspaper articles that reference a specific subject/topic item.

    This function implements "reverse lookup" - given a subject authority record
    (like a person, place, or topic), it finds all articles that cite it via
    dcterms:subject relationships.

    Args:
        client: OmekaClient instance.
        subject_item_id: Numeric ID of the subject authority item.

    Returns:
        Tuple of (subject_item_json_or_None, [list_of_article_items]).
        The subject will be None if not found; articles list may be empty.
    """
    subject_item = fetch_item(client, subject_item_id)
    if not subject_item:
        return None, []

    reverse = subject_item.get("@reverse") or {}
    refs = reverse.get("dcterms:subject") or [] if isinstance(reverse, dict) else []
    if not isinstance(refs, list):
        refs = []
    total_refs = len(refs)
    console.print(f"[cyan]ℹ[/] Found {total_refs} reverse subject references.")

    article_items: List[JSONObj] = []
    seen_ids: set[str] = set()
    stats = Counter()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching articles...", total=total_refs)
        
        for reference in refs:
            status, item = _fetch_subject_reference(client, reference, seen_ids)
            stats[status] += 1
            if item is not None:
                article_items.append(item)
            progress.update(
                task,
                advance=1,
                description=(
                    "Fetching articles... "
                    f"[dim](added={len(article_items)}, skipped={stats['non_article']})[/]"
                ),
            )

    _show_subject_fetch_stats(stats)

    return subject_item, article_items


def fetch_item_set(client: OmekaClient, set_id: str) -> Optional[JSONObj]:
    """Fetch a single Item Set resource by ID.

    Args:
        client: OmekaClient instance.
        set_id: Numeric string ID for the item set.

    Returns:
        The item set JSON object, or None if not found.
    """
    return client.get_item_set(int(set_id))


def fetch_items_in_set(client: OmekaClient, set_id: str) -> List[JSONObj]:
    """Fetch all items that belong to a specific Omeka Item Set.

    Args:
        client: OmekaClient instance.
        set_id: Numeric string ID for the target item set.

    Returns:
        Complete list of item JSON objects from the set.
    """
    with console.status("[cyan]Fetching items from set...[/]", spinner="dots"):
        return client.get_items(int(set_id))


def process_item_set(
    client: OmekaClient,
    set_id: str,
    out_dir: str,
    max_items_per_file: int,
    country_label: Optional[str] = None,
    file_ext: str = "md",
    include_summary: bool = False,
) -> Tuple[int, List[str]]:
    """Process a single Item Set: fetch items, filter for articles, and export to file(s).

    Args:
        client: OmekaClient instance.
        set_id: The Omeka Item Set ID to process (as string).
        out_dir: Base output directory for generated files.
        max_items_per_file: Maximum articles per file before auto-splitting.
        country_label: Optional country name for subfolder organization and logging.
        file_ext: Output file extension ("md" or "txt").
        include_summary: Emit each article's stored AI summary.

    Returns:
        Tuple of (total_article_count, [list_of_written_file_paths]).
    """
    # Fetch the Item Set metadata to get its human-readable title
    item_set = fetch_item_set(client, set_id)
    if not item_set:
        console.print(f"[yellow]⚠[/] Skipping Item Set {set_id}: not found or inaccessible.")
        return 0, []

    # Extract and sanitize the set title for use in filenames
    set_title = extract_first_value(item_set.get("dcterms:title")) or f"item_set_{set_id}"
    safe_set_title = sanitize_filename(set_title)
    
    # Extract country from item set's spatial coverage
    country = extract_country_from_item_set(item_set)

    # Create progress messages with optional country prefix
    prefix = f"[cyan]{country_label}[/] » " if country_label else ""
    console.print(f"{prefix}Processing Item Set {set_id}: [bold]{set_title}[/]")
    
    # Fetch all items in this set
    items = fetch_items_in_set(client, set_id)
    console.print(f"  [dim]Found {len(items)} items. Filtering for articles...[/]")

    # Filter for newspaper articles only (exclude bibo:Issue which are full newspaper editions)
    articles: List[JSONObj] = []
    for it in items:
        types = it.get("@type", [])
        if isinstance(types, list) and "bibo:Article" in types:
            articles.append(it)

    if not articles:
        console.print(f"  [yellow]⚠[/] No bibo:Article items found in Item Set {set_id}.")
        return 0, []

    console.print(f"  [green]✓[/] {len(articles)} articles found")

    # Prepare file naming and output organization
    header_title = f"Item Set: {set_title} (ID {set_id})"
    # Include set ID in filename to avoid collisions across sets with similar titles
    file_stub = f"{safe_set_title}_{set_id}"

    written_files: List[str] = []
    total_articles = len(articles)
    
    # Determine output directory (with optional country subfolder)
    target_dir = out_dir
    if country_label:
        country_dir = sanitize_filename(country_label)
        target_dir = os.path.join(out_dir, country_dir)
        os.makedirs(target_dir, exist_ok=True)
        
    # Normalize file extension (remove any leading dot)
    file_ext = (file_ext or "md").lower().lstrip(".")

    # Write articles to file(s), splitting if necessary
    if total_articles <= max_items_per_file:
        # Small collection: write all articles to a single file
        out_path = os.path.join(target_dir, f"{file_stub}_articles.{file_ext}")
        write_articles_to_file(
            articles,
            out_path,
            header_title,
            client=client,
            country=country,
            include_summary=include_summary,
        )
        written_files.append(out_path)
        console.print(f"  [green]✓[/] Wrote {total_articles} articles → [dim]{os.path.basename(out_path)}[/]")
    else:
        # Large collection: split into multiple files to stay within NotebookLM limits
        num_parts = (total_articles + max_items_per_file - 1) // max_items_per_file
        console.print(f"  [cyan]ℹ[/] Splitting {total_articles} articles into {num_parts} parts...")

        for part_num in range(1, num_parts + 1):
            start_idx = (part_num - 1) * max_items_per_file
            end_idx = min(start_idx + max_items_per_file, total_articles)
            part_articles = articles[start_idx:end_idx]
            out_path = os.path.join(target_dir, f"{file_stub}_articles_part{part_num}.{file_ext}")
            write_articles_to_file(
                part_articles,
                out_path,
                header_title,
                part_num,
                client=client,
                country=country,
                num_parts=num_parts,
                include_summary=include_summary,
            )
            written_files.append(out_path)
            console.print(f"    Part {part_num}: {len(part_articles)} articles → [dim]{os.path.basename(out_path)}[/]")

    return total_articles, written_files


def process_subject_items(
    client: OmekaClient,
    subject_item_id: str,
    out_dir: str,
    max_items_per_file: int,
    file_ext: str = "md",
    include_summary: bool = False,
) -> Tuple[int, List[str]]:
    """Process reverse-linked items for a given subject Item ID and export to file(s).

    Args:
        client: OmekaClient instance.
        subject_item_id: Item ID used as dcterms:subject by articles.
        out_dir: Output directory.
        max_items_per_file: Max articles per file before splitting.
        file_ext: Output extension (md|txt).
        include_summary: Emit each article's stored AI summary.

    Returns:
        (article_count, [written_files])
    """
    console.print(f"[cyan]ℹ[/] Looking up subject Item {subject_item_id}...")
    subject, articles = fetch_articles_with_subject(client, subject_item_id)
    if not subject:
        console.print(f"[red]✗[/] Subject item {subject_item_id} not found or inaccessible.")
        return 0, []

    subj_title = subject.get("o:title") or extract_first_value(subject.get("dcterms:title")) or f"item_{subject_item_id}"
    safe_subj_title = sanitize_filename(str(subj_title))

    if not articles:
        console.print(f"[yellow]⚠[/] No bibo:Article items reference subject {subject_item_id}.")
        return 0, []

    # Group articles by publisher
    console.print(f"[cyan]ℹ[/] Grouping {len(articles)} articles by publisher...")
    articles_by_publisher: Dict[str, List[JSONObj]] = {}
    for art in articles:
        publishers = extract_publishers(art)
        # Use first publisher or "Unknown" if none
        publisher_name = publishers[0] if publishers else "Unknown"
        if publisher_name not in articles_by_publisher:
            articles_by_publisher[publisher_name] = []
        articles_by_publisher[publisher_name].append(art)
    
    # Display publisher breakdown
    pub_table = Table(title="📰 Articles by Publisher", box=box.ROUNDED)
    pub_table.add_column("Publisher", style="cyan")
    pub_table.add_column("Articles", style="green", justify="right")
    for pub_name, pub_articles in sorted(articles_by_publisher.items(), key=lambda x: -len(x[1])):
        pub_table.add_row(pub_name, str(len(pub_articles)))
    console.print(pub_table)

    written: List[str] = []
    total = len(articles)
    file_ext = (file_ext or "md").lower().lstrip(".")
    
    # Create a subfolder for this subject
    subject_dir = os.path.join(out_dir, safe_subj_title)
    os.makedirs(subject_dir, exist_ok=True)
    console.print(f"[cyan]ℹ[/] Creating folder: [dim]{os.path.basename(subject_dir)}[/]")

    # Write each publisher's articles to a separate file (or multiple files if too large)
    for publisher_name, pub_articles in articles_by_publisher.items():
        safe_pub_name = sanitize_filename(publisher_name)
        file_stub = safe_pub_name  # Just use publisher name since we're in a subject folder
        header_title = f"Subject: {subj_title} - Publisher: {publisher_name}"
        pub_count = len(pub_articles)
        
        if pub_count <= max_items_per_file:
            out_path = os.path.join(subject_dir, f"{file_stub}_articles.{file_ext}")
            write_articles_to_file(
                pub_articles,
                out_path,
                header_title,
                client=client,
                include_summary=include_summary,
            )
            written.append(out_path)
            console.print(f"[green]✓[/] {publisher_name}: {pub_count} articles → [dim]{os.path.basename(out_path)}[/]")
        else:
            num_parts = (pub_count + max_items_per_file - 1) // max_items_per_file
            console.print(f"[cyan]ℹ[/] {publisher_name}: Splitting {pub_count} articles into {num_parts} parts...")
            for part_num in range(1, num_parts + 1):
                start_idx = (part_num - 1) * max_items_per_file
                end_idx = min(start_idx + max_items_per_file, pub_count)
                part_articles = pub_articles[start_idx:end_idx]
                out_path = os.path.join(subject_dir, f"{file_stub}_articles_part{part_num}.{file_ext}")
                write_articles_to_file(
                    part_articles,
                    out_path,
                    header_title,
                    part_num,
                    client=client,
                    num_parts=num_parts,
                    include_summary=include_summary,
                )
                written.append(out_path)
                console.print(f"  Part {part_num}: {len(part_articles)} articles → [dim]{os.path.basename(out_path)}[/]")

    return total, written


def write_articles_to_file(
    articles: List[JSONObj],
    file_path: str,
    header_title: str,
    part_num: int = None,
    client: Optional[OmekaClient] = None,
    country: Optional[str] = None,
    num_parts: Optional[int] = None,
    include_summary: bool = False,
) -> None:
    """Write a batch of articles to a single Markdown file under one H1 header.

    The header gives the file a document root, so the articles below it are H2
    siblings rather than a flat run of H1s. Split exports name their part in
    that header, which is otherwise the only thing distinguishing them once
    NotebookLM has ingested the sources.

    Args:
        articles: List of article JSON objects to write.
        file_path: Full path to the output file.
        header_title: Title for the file's H1 (the item set, or subject and
            publisher, the articles were drawn from).
        part_num: Part number for multi-part exports, named in the header.
        client: OmekaClient instance for fetching publisher country.
        country: Optional country name to include in all articles' metadata.
        num_parts: Total number of parts, when known, for "Part 2 of 5".
        include_summary: Emit each article's stored AI summary (see
            :func:`format_article`).
    """
    heading = header_title
    if part_num:
        heading += f" — Part {part_num}"
        if num_parts:
            heading += f" of {num_parts}"

    body = "".join(
        format_article(article, client, country, include_summary)
        for article in articles
    )
    atomic_write_text(Path(file_path), f"# {heading}\n\n{body}")


@dataclass(frozen=True)
class ExportRequest:
    """One validated exporter mode and its optional Omeka identifier."""

    mode: str
    item_id: Optional[str] = None


#: CLI spellings that turn on the stored-summary line. Not argparse: this
#: entry point reads nothing and writes nothing to Omeka, so the flag is a
#: presentation choice rather than one of the consent gates write_guard exists
#: to enforce.
SUMMARY_FLAGS = ("--with-summaries", "--with-summary", "--summaries")


def extract_summary_flag(argv: List[str]) -> Tuple[List[str], bool]:
    """Split the summary flag out of argv, returning the remaining arguments.

    Keeps the compact positional syntax ("all", "12345", "subject:678") free of
    flag handling, so the flag can appear on either side of the mode.
    """
    remaining = [a for a in argv if a.strip().lower() not in SUMMARY_FLAGS]
    return remaining, len(remaining) != len(argv)


def parse_cli_request(argv: List[str]) -> Optional[ExportRequest]:
    """Parse the exporter’s compact backwards-compatible CLI syntax."""
    if not argv:
        return None
    first = argv[0].strip()
    lowered = first.lower()
    if lowered in ("all", "--all"):
        return ExportRequest("all")
    if lowered.startswith(("subject:", "s:")):
        item_id = first.split(":", 1)[1]
        return ExportRequest("subject", item_id) if item_id.isdigit() else None
    if lowered == "--subject" and len(argv) > 1 and argv[1].strip().isdigit():
        return ExportRequest("subject", argv[1].strip())
    if first.isdigit():
        return ExportRequest("item_set", first)
    return None


def prompt_export_request() -> Optional[ExportRequest]:
    """Prompt for an export mode when no valid CLI request was supplied."""
    console.print("[bold]Choose export mode:[/]")
    console.print("  [cyan]1[/] Whole IWAC collection (all countries)")
    console.print("  [cyan]2[/] Single Item Set by ID")
    console.print("  [cyan]3[/] Articles by subject Item ID (reverse lookup)")
    mode = console.input("\n[bold]Enter choice (1/2/3):[/] ").strip().lower()
    if mode in ("1", "all", "a"):
        return ExportRequest("all")
    if mode in ("3", "subject", "s"):
        item_id = console.input("[bold]Enter the subject Item ID:[/] ").strip()
        if item_id.isdigit():
            return ExportRequest("subject", item_id)
        console.print("[red]✗[/] Subject Item ID must be a number.")
        return None
    item_id = console.input("[bold]Enter the Omeka Item Set ID:[/] ").strip()
    if item_id.isdigit():
        return ExportRequest("item_set", item_id)
    console.print("[red]✗[/] Item Set ID must be a number.")
    return None


def prompt_include_summaries() -> bool:
    """Ask whether to inline the stored AI summaries (interactive mode only)."""
    console.print(
        "\n[dim]Articles carry an AI-written summary (bibo:shortDescription). "
        "Including it costs no extra API or model calls, but adds machine-written "
        "text NotebookLM will cite alongside the newspapers.[/]"
    )
    answer = console.input("[bold]Include AI summaries? (y/N):[/] ").strip().lower()
    return answer in ("y", "yes", "o", "oui")


def export_file_extension() -> str:
    """Resolve the supported Markdown/plain-text output extension."""
    extension = os.getenv("NOTEBOOKLM_EXPORT_EXT", "md").lower().lstrip(".")
    if extension in ("md", "txt"):
        return extension
    console.print(
        f"[yellow]⚠[/] Unrecognized NOTEBOOKLM_EXPORT_EXT='{extension}', "
        "defaulting to 'md'."
    )
    return "md"


def show_export_configuration(
    out_dir: str, file_ext: str, include_summary: bool = False
) -> None:
    config_table = Table(show_header=False, box=box.ROUNDED, title="⚙️ Configuration")
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Output directory", out_dir)
    config_table.add_row("File format", f".{file_ext}")
    config_table.add_row("Max articles/file", str(MAX_ITEMS_PER_FILE))
    config_table.add_row(
        "AI summaries", "included" if include_summary else "omitted"
    )
    console.print(config_table)
    console.print()


def show_written_files(files: List[str]) -> None:
    for path in files:
        console.print(f"  [dim]{path}[/]")


def export_whole_collection(
    client: OmekaClient,
    out_dir: str,
    file_ext: str,
    include_summary: bool = False,
) -> Tuple[int, List[str]]:
    """Export every configured country/item-set pair."""
    console.rule("[bold cyan]Whole IWAC Collection Export[/]")
    grand_total = 0
    all_written: List[str] = []
    for country, set_ids in COUNTRY_ITEM_SETS.items():
        if not set_ids:
            console.print(f"[dim]Skip {country}: no Item Set IDs configured.[/]")
            continue
        console.rule(f"[bold]{country}[/]", style="dim")
        for set_id in set_ids:
            count, files = process_item_set(
                client,
                set_id,
                out_dir,
                MAX_ITEMS_PER_FILE,
                country_label=country,
                file_ext=file_ext,
                include_summary=include_summary,
            )
            grand_total += count
            all_written.extend(files)
    return grand_total, all_written


def show_whole_export_summary(article_count: int, files: List[str]) -> None:
    console.print()
    summary_table = Table(title="📊 Export Summary", box=box.ROUNDED)
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value", style="green bold")
    summary_table.add_row("Total articles exported", str(article_count))
    summary_table.add_row("Files created", str(len(files)))
    console.print(summary_table)
    if files:
        console.print("\n[bold]Files created:[/]")
        show_written_files(files)
    else:
        console.print(
            "[yellow]⚠[/] No files were created. Ensure COUNTRY_ITEM_SETS is configured."
        )


def export_one_request(
    client: OmekaClient,
    request: ExportRequest,
    out_dir: str,
    file_ext: str,
    include_summary: bool = False,
) -> None:
    """Run a subject or single-item-set request and display its result."""
    assert request.item_id is not None
    is_subject = request.mode == "subject"
    console.rule(
        "[bold cyan]Subject-based Export[/]"
        if is_subject else "[bold cyan]Single Item Set Export[/]"
    )
    processor = process_subject_items if is_subject else process_item_set
    count, files = processor(
        client,
        request.item_id,
        out_dir,
        MAX_ITEMS_PER_FILE,
        file_ext=file_ext,
        include_summary=include_summary,
    )
    if files:
        console.print(Panel(
            f"[green]✓[/] Exported [bold]{count}[/] articles to {len(files)} file(s)",
            title="✅ Export Complete",
            border_style="green",
        ))
        show_written_files(files)
        return
    noun = "subject" if is_subject else "Item Set"
    console.print(f"[yellow]⚠[/] No files were created for the specified {noun}.")


def main() -> int:
    """Main entry point for the Omeka to NotebookLM Markdown exporter.
    
    This function handles:
    1. Loading environment variables and configuration
    2. Parsing command-line arguments or prompting for user input
    3. Dispatching to the appropriate export mode:
       - Whole IWAC collection (all countries and their Item Sets)
       - Single Item Set by ID  
       - Articles referencing a specific subject Item ID
    4. Providing summary output of what was exported
    
    The script supports three main export modes:
    - Mode 1: Export entire IWAC collection (uses COUNTRY_ITEM_SETS mapping)
    - Mode 2: Export single Item Set by providing its numeric ID
    - Mode 3: Export articles that reference a subject authority by its ID
    
    CLI usage examples:
        python script.py all                    # Export whole collection
        python script.py 12345                  # Export Item Set 12345
        python script.py subject:67890          # Export articles about subject 67890
        python script.py --subject 67890        # Alternative subject syntax
        python script.py 12345 --with-summaries # Inline the stored AI summaries
    """
    # Welcome banner
    console.print(Panel(
        "[bold]Export newspaper articles from Omeka S to NotebookLM-ready Markdown[/]\n"
        "[dim]Optimized for Google NotebookLM ingestion with automatic file splitting[/]",
        title="📰 Omeka → NotebookLM Exporter",
        border_style="cyan"
    ))
    
    try:
        client = OmekaClient.from_env()
    except ValueError as e:
        console.print(f"[red]✗[/] {e}")
        return 1

    # Set up output directory structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "extracted_articles")
    os.makedirs(out_dir, exist_ok=True)
    file_ext = export_file_extension()

    argv, include_summary = extract_summary_flag(sys.argv[1:])
    request = parse_cli_request(argv)
    if request is None and argv:
        console.print(
            f"[yellow]⚠[/] Unrecognized CLI arguments '{' '.join(argv)}'. "
            "Switching to interactive mode..."
        )
    if request is None:
        request = prompt_export_request()
        if request is None:
            return 1
        # The flag is the non-interactive route; interactive runs get the ask.
        include_summary = include_summary or prompt_include_summaries()

    show_export_configuration(out_dir, file_ext, include_summary)

    if request.mode == "all":
        article_count, files = export_whole_collection(
            client, out_dir, file_ext, include_summary
        )
        show_whole_export_summary(article_count, files)
    else:
        export_one_request(client, request, out_dir, file_ext, include_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
