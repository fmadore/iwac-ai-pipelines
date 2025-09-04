"""
Omeka item to NotebookLM-ready Markdown exporter.

This script fetches newspaper articles from an Omeka S digital archive and
exports them to consolidated UTF-8 Markdown files optimized for ingestion into
Google's NotebookLM AI research tool.

=== Overview ===

The IWAC (Institut des mondes africains) project digitizes West African newspapers
from the colonial and post-independence periods. This script helps researchers
export collections of articles in a format that NotebookLM can easily process
for analysis, summarization, and question-answering.

=== Export Modes ===

Three main export modes are supported:em to NotebookLM-ready Markdown exporter.

This script fetches all newspaper articles linked to a single Omeka item and
exports them to one consolidated UTF-8 Markdown file that’s easy to ingest into
NotebookLM.

Input
- Option A: Export the whole IWAC collection (predefined country -> item set IDs)
- Option B: Export a single Item Set by ID
- Environment: OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL in .env

How it works
1) Loads credentials from .env
2) In "Whole IWAC" mode: iterate predefined Item Set IDs per country
3) In "Single Item Set" mode: process only the provided Item Set ID
4) For each item set, list items (paginated), keep "bibo:Article" (and "bibo:Issue")
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
- Layout per article (Markdown):

        # <title>
        **Newspaper:** <publisher(s)>
        **Date:** <date>

        <article body>

        ---

Notes
- The format favors readability and simple chunking for tools like NotebookLM.
- Only the first value is taken for multi-valued fields (date/content) to avoid
    duplications and keep the file compact.
- Maximum 250 articles per file to stay within NotebookLM's word limits.
"""

import os
import re
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple

import requests
from dotenv import load_dotenv

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
        "43051", "76357", "62076", "31882", "57945", "63444", "76253", "61684", "76239", "48249", "57943", "57944", "61320", "15845", "76364", "73533", "61289", "45390"
    ],
    "Niger": [
        "62021",
    ],
    "Togo": [
        "67437", "25304", "67399", "9458", "67407", "67460", "67480", "67430", "5498", "67436", "67456", "5499"
    ],
}


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


def load_env() -> Dict[str, str]:
    """Load Omeka credentials from .env and validate presence.

    Required keys in .env:
        - OMEKA_BASE_URL
        - OMEKA_KEY_IDENTITY
        - OMEKA_KEY_CREDENTIAL

    Returns:
        A dict with keys: {"base", "kid", "kcr"}.

    Side effects:
        Exits the program with code 1 if any required variables are missing.
    """
    load_dotenv()
    base = os.getenv("OMEKA_BASE_URL")
    kid = os.getenv("OMEKA_KEY_IDENTITY")
    kcr = os.getenv("OMEKA_KEY_CREDENTIAL")
    missing = [k for k, v in {
        "OMEKA_BASE_URL": base,
        "OMEKA_KEY_IDENTITY": kid,
        "OMEKA_KEY_CREDENTIAL": kcr,
    }.items() if not v]
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}")
        print("Create a .env file with: OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL")
        sys.exit(1)
    return {"base": base.rstrip("/"), "kid": kid, "kcr": kcr}


def get_json(url: str, params: Dict[str, str]) -> Optional[JSONLike]:
    """Perform a GET request to the Omeka API and parse the JSON response.

    This is a wrapper around requests.get() that handles common errors and
    provides consistent error reporting. Used for all Omeka API calls.

    Args:
        url: Complete Omeka API URL (e.g., "https://example.org/api/items/123").
        params: Query parameters including authentication credentials.

    Returns:
        Parsed JSON response (dict or list) on success, or None if any error occurs
        (network issues, HTTP errors, invalid JSON, etc.).
    """
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


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


def format_article(article: JSONObj) -> str:
    """Convert a single newspaper article into NotebookLM-friendly Markdown format.

    Creates a standardized Markdown block for each article with:
    - Level 1 heading with the article title
    - Bold metadata lines for Newspaper and Date
    - The full article content (cleaned of extra whitespace)
    - A horizontal rule separator for visual separation

    This format is optimized for NotebookLM ingestion and human readability.

    Args:
        article: Omeka JSON object representing a bibo:Article with fields like
                o:title, dcterms:date, dcterms:publisher, bibo:content.

    Returns:
        Formatted Markdown string ready to write to file, including trailing
        horizontal rule and newlines for proper separation between articles.
        
    Example output:
        # Article Title Here
        **Newspaper:** L'Observateur
        **Date:** 1998-02-16
        
        Article content goes here with proper formatting...
        
        ---
    """
    title = article.get("o:title") or "No title"
    date = extract_first_value(article.get("dcterms:date")) or "Unknown"
    content = extract_first_value(article.get("bibo:content")) or ""
    content = normalize_md_whitespace(content)
    publishers = extract_publishers(article)
    publisher_str = "; ".join(publishers) if publishers else "Unknown"

    lines = [f"# {title}"]
    lines.append(f"**Newspaper:** {publisher_str}")
    lines.append(f"**Date:** {date}")
    lines.append("")
    if content:
        lines.append(content)
        lines.append("")
    # Markdown horizontal rule between articles
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def fetch_item(env: Dict[str, str], item_id: str) -> Optional[JSONObj]:
    """Fetch a single Item by ID.

    Args:
        env: Environment dict from load_env().
        item_id: Numeric string Item ID.

    Returns:
        The item JSON or None.
    """
    url = f"{env['base']}/items/{item_id}"
    params = {
        "key_identity": env["kid"],
        "key_credential": env["kcr"],
    }
    data = get_json(url, params)
    return data if isinstance(data, dict) else None


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


def fetch_resource(env: Dict[str, str], resource_id: str) -> Optional[JSONObj]:
    """Fetch a generic resource by ID via the '/resources' endpoint.

    Useful when reverse links use '/api/resources/<id>' rather than '/items'.
    """
    url = f"{env['base']}/resources/{resource_id}"
    params = {
        "key_identity": env["kid"],
        "key_credential": env["kcr"],
    }
    data = get_json(url, params)
    return data if isinstance(data, dict) else None


def fetch_item_or_resource(env: Dict[str, str], id_str: str) -> Optional[JSONObj]:
    """Try fetching an Item by ID, with fallback to the generic Resource endpoint.

    Some Omeka installations use different endpoints for the same content.
    This function tries the more specific /items endpoint first, then falls back
    to the generic /resources endpoint if that fails.

    Args:
        env: Environment dict containing API credentials and base URL.
        id_str: Numeric ID as a string.

    Returns:
        The JSON object for the item/resource, or None if both endpoints fail.
    """
    it = fetch_item(env, id_str)
    if isinstance(it, dict):
        return it
    return fetch_resource(env, id_str)


def fetch_articles_with_subject(env: Dict[str, str], subject_item_id: str) -> Tuple[Optional[JSONObj], List[JSONObj]]:
    """Find all newspaper articles that reference a specific subject/topic item.

    This function implements "reverse lookup" - given a subject authority record
    (like a person, place, or topic), it finds all articles that cite it via
    dcterms:subject relationships.

    How it works:
    1. Fetch the subject item to access its @reverse.dcterms:subject property
    2. This property lists all items that reference this subject
    3. For each reference, fetch the full item record
    4. Filter to keep only bibo:Article and bibo:Issue items
    5. Return both the original subject and the filtered articles

    Args:
        env: Environment dict containing API credentials.
        subject_item_id: Numeric ID of the subject authority item.

    Returns:
        Tuple of (subject_item_json_or_None, [list_of_article_items]).
        The subject will be None if not found; articles list may be empty.
        
    Example use case:
        If item 12345 represents "Burkina Faso" as a place authority,
        this will find all newspaper articles about Burkina Faso.
    """
    subject_item = fetch_item(env, subject_item_id)
    if not subject_item:
        return None, []

    # Extract reverse subject references - this is where Omeka stores "what items point to this one"
    reverse = subject_item.get("@reverse") or {}
    refs = []
    if isinstance(reverse, dict):
        refs = reverse.get("dcterms:subject") or []
    if not isinstance(refs, list):
        refs = []
    total_refs = len(refs)
    print(f"Found {total_refs} reverse subject references.")

    # Process each reference to collect actual article items
    article_items: List[JSONObj] = []
    seen_ids: set[str] = set()  # Avoid duplicate processing
    skipped_non_article = 0      # Track items that aren't articles
    fetch_fail = 0               # Track API failures  
    dupes = 0                    # Track duplicate IDs encountered
    start_ts = time.perf_counter()  # For progress reporting
    
    for idx, ref in enumerate(refs, start=1):
        if not isinstance(ref, dict):
            continue
        
        # Optimization: Skip obvious non-articles using embedded type information
        # This avoids unnecessary API calls for items we know we don't want
        ref_types = ref.get("@type")
        if isinstance(ref_types, list) and "bibo:Article" not in ref_types and "bibo:Issue" not in ref_types:
            skipped_non_article += 1
            # Update progress display (overwrite same line for live updates)
            elapsed = max(time.perf_counter() - start_ts, 1e-6)
            rate = idx / elapsed
            remain = max(total_refs - idx, 0)
            eta = remain / rate if rate > 0 else 0
            eta_m, eta_s = divmod(int(eta), 60)
            print(
                f"  [{idx}/{total_refs}] added={len(article_items)} skippedType={skipped_non_article} dup={dupes} fetchErr={fetch_fail} ETA={eta_m:02d}:{eta_s:02d}",
                end="\r",
                flush=True,
            )
            continue
            
        # Extract the numeric ID from various possible locations in the reference
        rid = None
        if isinstance(ref.get("o:id"), int):
            rid = str(ref["o:id"]) 
        elif isinstance(ref.get("o:id"), str) and ref["o:id"].isdigit():
            rid = ref["o:id"]
        elif isinstance(ref.get("@id"), str):
            rid = parse_id_from_at_id(ref["@id"])
        if not rid:
            continue
            
        # Skip if we've already processed this ID
        if rid in seen_ids:
            dupes += 1
            # live progress update
            elapsed = max(time.perf_counter() - start_ts, 1e-6)
            rate = idx / elapsed
            remain = max(total_refs - idx, 0)
            eta = remain / rate if rate > 0 else 0
            eta_m, eta_s = divmod(int(eta), 60)
            print(
                f"  [{idx}/{total_refs}] added={len(article_items)} skippedType={skipped_non_article} dup={dupes} fetchErr={fetch_fail} ETA={eta_m:02d}:{eta_s:02d}",
                end="\r",
                flush=True,
            )
            continue
        seen_ids.add(rid)
        
        # Fetch the full item record to get complete content and confirm type
        item = fetch_item_or_resource(env, rid)
        if not isinstance(item, dict):
            fetch_fail += 1
            # Update progress display
            elapsed = max(time.perf_counter() - start_ts, 1e-6)
            rate = idx / elapsed
            remain = max(total_refs - idx, 0)
            eta = remain / rate if rate > 0 else 0
            eta_m, eta_s = divmod(int(eta), 60)
            print(
                f"  [{idx}/{total_refs}] added={len(article_items)} skippedType={skipped_non_article} dup={dupes} fetchErr={fetch_fail} ETA={eta_m:02d}:{eta_s:02d}",
                end="\r",
                flush=True,
            )
            continue
            
        # Final type check: only keep actual articles and issues
        types = item.get("@type", [])
        if isinstance(types, list) and ("bibo:Article" in types or "bibo:Issue" in types):
            article_items.append(item)
            
        # Live progress update (overwrites previous line)
        elapsed = max(time.perf_counter() - start_ts, 1e-6)
        rate = idx / elapsed
        remain = max(total_refs - idx, 0)
        eta = remain / rate if rate > 0 else 0
        eta_m, eta_s = divmod(int(eta), 60)
        msg = (
            f"  [{idx}/{total_refs}] added={len(article_items)} "
            f"skippedType={skipped_non_article} dup={dupes} fetchErr={fetch_fail} "
            f"ETA={eta_m:02d}:{eta_s:02d}"
        )
        print(msg, end="\r", flush=True)
        
        # Every 50 items, print a newline snapshot to keep some history visible
        if idx % 50 == 0 or idx == 1:
            print("\n" + msg)

    # Finish line to move past carriage return
    print()
    elapsed_total = time.perf_counter() - start_ts
    print(
        f"Collected {len(article_items)} article/issue items from {total_refs} references in {elapsed_total:.1f}s. "
        f"(skippedType={skipped_non_article}, dup={dupes}, fetchErr={fetch_fail})"
    )

    return subject_item, article_items


def fetch_item_set(env: Dict[str, str], set_id: str) -> Optional[JSONObj]:
    """Fetch a single Item Set resource by ID.

    Args:
        env: Environment dict from load_env().
        set_id: Numeric string ID for the item set.

    Returns:
        The item set JSON object, or None if not found.
    """
    url = f"{env['base']}/item_sets/{set_id}"
    params = {
        "key_identity": env["kid"],
        "key_credential": env["kcr"],
    }
    data = get_json(url, params)
    return data if isinstance(data, dict) else None


def fetch_items_in_set(env: Dict[str, str], set_id: str, per_page: int = 100) -> List[JSONObj]:
    """Fetch all items that belong to a specific Omeka Item Set using pagination.

    Item Sets in Omeka can contain hundreds or thousands of items, so we need
    to paginate through the results. This function handles the pagination
    automatically and returns all items in one list.

    Args:
        env: Environment dict containing API credentials from load_env().
        set_id: Numeric string ID for the target item set.
        per_page: Number of items to request per API call (default 100).

    Returns:
        Complete list of item JSON objects from the set. May be empty if
        the set doesn't exist or contains no items.
        
    Note:
        This can be memory-intensive for very large sets. Consider adding
        a streaming/callback approach for sets with 10,000+ items.
    """
    base_items_url = f"{env['base']}/items"
    all_items: List[JSONObj] = []
    page = 1
    
    # Paginate through all items in the set
    while True:
        params = {
            "key_identity": env["kid"],
            "key_credential": env["kcr"],
            "item_set_id": set_id,
            "per_page": str(per_page),
            "page": str(page),
        }
        data = get_json(base_items_url, params)
        
        # Stop if we get no data or an error
        if not isinstance(data, list) or not data:
            break
            
        # Filter to ensure we only get dict objects (valid items)
        page_items = [d for d in data if isinstance(d, dict)]
        all_items.extend(page_items)
        
        # If we got fewer items than requested, we've reached the end
        if len(page_items) < per_page:
            break
        page += 1
    return all_items


def process_item_set(
    env: Dict[str, str],
    set_id: str,
    out_dir: str,
    max_items_per_file: int,
    country_label: Optional[str] = None,
    file_ext: str = "md",
) -> Tuple[int, List[str]]:
    """Process a single Item Set: fetch items, filter for articles, and export to file(s).

    This is the main processing function for Item Set-based exports. It:
    1. Fetches the Item Set metadata to get its title
    2. Retrieves all items in the set (with pagination)
    3. Filters for newspaper articles (bibo:Article and bibo:Issue types)
    4. Exports articles to Markdown file(s), splitting if needed for size limits
    5. Organizes output by country if specified

    Args:
        env: Loaded environment dict with API credentials and base URL.
        set_id: The Omeka Item Set ID to process (as string).
        out_dir: Base output directory for generated files.
        max_items_per_file: Maximum articles per file before auto-splitting.
        country_label: Optional country name for subfolder organization and logging.
        file_ext: Output file extension ("md" or "txt").

    Returns:
        Tuple of (total_article_count, [list_of_written_file_paths]).
        
    Example:
        If set_id "12345" contains 500 articles and max_items_per_file is 250,
        this will create two files: "newspaper_name_part1.md" and "newspaper_name_part2.md"
    """
    # Fetch the Item Set metadata to get its human-readable title
    item_set = fetch_item_set(env, set_id)
    if not item_set:
        print(f"- Skipping Item Set {set_id}: not found or inaccessible.")
        return 0, []

    # Extract and sanitize the set title for use in filenames
    set_title = extract_first_value(item_set.get("dcterms:title")) or f"item_set_{set_id}"
    safe_set_title = sanitize_filename(set_title)

    # Create progress messages with optional country prefix
    prefix = f"[{country_label}] " if country_label else ""
    print(f"{prefix}Listing items in Item Set {set_id} (\"{set_title}\")...")
    
    # Fetch all items in this set
    items = fetch_items_in_set(env, set_id, per_page=100)
    print(f"{prefix}Found {len(items)} items. Filtering bibo:Article and bibo:Issue…")

    # Filter for newspaper articles only
    articles: List[JSONObj] = []
    for idx, it in enumerate(items, start=1):
        types = it.get("@type", [])
        if isinstance(types, list) and ("bibo:Article" in types or "bibo:Issue" in types):
            articles.append(it)
            # Show progress every 50 items to avoid flooding console
            if idx % 50 == 0:
                print(f"  Processed {idx}/{len(items)}…")

    if not articles:
        print(f"{prefix}No bibo:Article items found in Item Set {set_id}.")
        return 0, []

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
        write_articles_to_file(articles, out_path, header_title)
        written_files.append(out_path)
        print(f"{prefix}Wrote {total_articles} articles -> {os.path.basename(out_path)}")
    else:
        # Large collection: split into multiple files to stay within NotebookLM limits
        num_parts = (total_articles + max_items_per_file - 1) // max_items_per_file
        print(f"{prefix}Total articles ({total_articles}) exceed {max_items_per_file}; splitting into {num_parts} parts…")
        
        for part_num in range(1, num_parts + 1):
            start_idx = (part_num - 1) * max_items_per_file
            end_idx = min(start_idx + max_items_per_file, total_articles)
            part_articles = articles[start_idx:end_idx]
            out_path = os.path.join(target_dir, f"{file_stub}_articles_part{part_num}.{file_ext}")
            write_articles_to_file(part_articles, out_path, header_title, part_num)
            written_files.append(out_path)
            print(f"  Part {part_num}: {len(part_articles)} articles -> {os.path.basename(out_path)}")

    return total_articles, written_files


def process_subject_items(
    env: Dict[str, str],
    subject_item_id: str,
    out_dir: str,
    max_items_per_file: int,
    file_ext: str = "md",
) -> Tuple[int, List[str]]:
    """Process reverse-linked items for a given subject Item ID and export to file(s).

    Args:
        env: Loaded env dict.
        subject_item_id: Item ID used as dcterms:subject by articles.
        out_dir: Output directory.
        max_items_per_file: Max articles per file before splitting.
        file_ext: Output extension (md|txt).

    Returns:
        (article_count, [written_files])
    """
    print(f"Looking up subject Item {subject_item_id}…")
    subject, articles = fetch_articles_with_subject(env, subject_item_id)
    if not subject:
        print(f"- Subject item {subject_item_id} not found or inaccessible.")
        return 0, []

    subj_title = subject.get("o:title") or extract_first_value(subject.get("dcterms:title")) or f"item_{subject_item_id}"
    safe_title = sanitize_filename(str(subj_title))
    header_title = f"Subject: {subj_title} (ID {subject_item_id})"
    file_stub = f"{safe_title}_subject_{subject_item_id}"

    if not articles:
        print(f"No bibo:Article items reference subject {subject_item_id}.")
        return 0, []

    written: List[str] = []
    total = len(articles)
    file_ext = (file_ext or "md").lower().lstrip(".")

    if total <= max_items_per_file:
        out_path = os.path.join(out_dir, f"{file_stub}_articles.{file_ext}")
        write_articles_to_file(articles, out_path, header_title)
        written.append(out_path)
        print(f"Wrote {total} articles -> {os.path.basename(out_path)}")
    else:
        num_parts = (total + max_items_per_file - 1) // max_items_per_file
        print(f"Total articles ({total}) exceed {max_items_per_file}; splitting into {num_parts} parts…")
        for part_num in range(1, num_parts + 1):
            start_idx = (part_num - 1) * max_items_per_file
            end_idx = min(start_idx + max_items_per_file, total)
            part_articles = articles[start_idx:end_idx]
            out_path = os.path.join(out_dir, f"{file_stub}_articles_part{part_num}.{file_ext}")
            write_articles_to_file(part_articles, out_path, header_title, part_num)
            written.append(out_path)
            print(f"  Part {part_num}: {len(part_articles)} articles -> {os.path.basename(out_path)}")

    return total, written


def write_articles_to_file(articles: List[JSONObj], file_path: str, header_title: str, part_num: int = None) -> None:
    """Write a batch of articles to a single Markdown file (no top header).
    
    Args:
        articles: List of article JSON objects to write.
        file_path: Full path to the output file.
        header_title: Unused (kept for backward compatibility).
        part_num: Part number for multi-part exports (unused in content).
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for art in articles:
            f.write(format_article(art))


def main():
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
    """
    print("\n=== Omeka -> Markdown Export (NotebookLM-ready) ===")
    env = load_env()

    # Configuration: Maximum items per file to respect NotebookLM's 500k word limit
    # NotebookLM performs better with smaller, focused documents rather than huge files
    MAX_ITEMS_PER_FILE = 250

    # Set up output directory structure
    # Creates "extracted_articles" folder next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "extracted_articles")
    os.makedirs(out_dir, exist_ok=True)

    # Determine output file format (Markdown by default, can override with env var)
    # Set NOTEBOOKLM_EXPORT_EXT=txt in .env to export as plain text instead
    file_ext = os.getenv("NOTEBOOKLM_EXPORT_EXT", "md").lower().lstrip(".")
    if file_ext not in ("md", "txt"):
        print(f"Unrecognized NOTEBOOKLM_EXPORT_EXT='{file_ext}', defaulting to 'md'.")
        file_ext = "md"

    # Parse command-line arguments for batch processing
    # Supports several convenient formats for different use cases:
    # - "all" or "--all"                     => whole collection export
    # - numeric ID (e.g. "12345")            => single item set export  
    # - "subject:<id>" | "s:<id>"            => subject-based export
    # - "--subject <id>"                     => alternative subject syntax
    cli_arg = sys.argv[1].strip() if len(sys.argv) > 1 else None
    export_whole = False
    single_set_id: Optional[str] = None
    subject_item_id: Optional[str] = None
    if cli_arg:
        # Parse the command-line argument to determine export mode
        low = cli_arg.lower()
        if low in ("all", "--all"):
            export_whole = True
        elif low.startswith("subject:") or low.startswith("s:"):
            # Extract ID from "subject:12345" or "s:12345" format
            maybe_id = cli_arg.split(":", 1)[1]
            if maybe_id.isdigit():
                subject_item_id = maybe_id
        elif low == "--subject" and len(sys.argv) > 2 and sys.argv[2].strip().isdigit():
            # Handle "--subject 12345" format (two separate arguments)
            subject_item_id = sys.argv[2].strip()
        elif cli_arg.isdigit():
            # Simple numeric ID = item set export
            single_set_id = cli_arg
        else:
            print(f"Unrecognized CLI arg '{cli_arg}'. Ignoring and switching to interactive mode…")

    # If no valid CLI args, prompt user for interactive mode selection
    if not cli_arg:
        # Interactive mode: prompt user to choose export type
        mode = input("Choose export mode: [1] Whole IWAC, [2] Item Set by ID, [3] Items with dcterms:subject Item ID: ").strip().lower()
        if mode in ("1", "all", "a", "one"):
            export_whole = True
        elif mode in ("3", "subject", "s"):
            subject_item_id = input("Enter the subject Item ID: ").strip()
            if not subject_item_id.isdigit():
                print("Subject Item ID must be a number.")
                sys.exit(1)
        else:
            # Default to item set mode
            single_set_id = input("Enter the Omeka Item Set ID to export: ").strip()
            if not single_set_id.isdigit():
                print("Item Set ID must be a number.")
                sys.exit(1)

    # Execute the selected export mode
    if export_whole:
        # Mode 1: Export entire IWAC collection organized by country
        print("\n=== Whole IWAC Collection export ===")
        grand_total = 0
        all_written: List[str] = []
        
        # Process each country's Item Sets in the predefined order
        for country in ["Benin", "Burkina Faso", "Côte d'Ivoire", "Niger", "Togo"]:
            set_ids = COUNTRY_ITEM_SETS.get(country, [])
            if not set_ids:
                print(f"[Skip] {country}: no Item Set IDs configured.")
                continue
            print(f"\n-- {country} --")
            
            # Process each Item Set for this country
            for sid in set_ids:
                if not isinstance(sid, str) or not sid.isdigit():
                    print(f"- Skipping invalid Item Set ID '{sid}' for {country}.")
                    continue
                count, files = process_item_set(env, sid, out_dir, MAX_ITEMS_PER_FILE, country_label=country, file_ext=file_ext)
                grand_total += count
                all_written.extend(files)

        print("\n=== Summary ===")
        print(f"Total articles exported: {grand_total}")
        if all_written:
            print("Files created:")
            for p in all_written:
                print(f"  {p}")
        else:
            print("No files were created. Ensure COUNTRY_ITEM_SETS is configured.")
        return

    # Mode 3: Export articles that reference a specific subject authority
    if subject_item_id:
        count, files = process_subject_items(env, subject_item_id, out_dir, MAX_ITEMS_PER_FILE, file_ext=file_ext)
        if files:
            print(f"\nDone. Exported {count} articles to:")
            for p in files:
                print(f"  {p}")
        else:
            print("No files were created for the specified subject.")
        return

    # Mode 2: Export a single Item Set by its ID
    assert single_set_id is not None
    count, files = process_item_set(env, single_set_id, out_dir, MAX_ITEMS_PER_FILE, file_ext=file_ext)
    if files:
        print(f"\nDone. Exported {count} articles to:")
        for p in files:
            print(f"  {p}")
    else:
        print("No files were created for the specified Item Set.")


if __name__ == "__main__":
    main()
