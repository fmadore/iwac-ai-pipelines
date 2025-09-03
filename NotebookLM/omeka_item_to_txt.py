"""
Omeka item to NotebookLM-ready TXT exporter.

This script fetches all newspaper articles linked to a single Omeka item and
exports them to one consolidated UTF-8 text file that’s easy to ingest into
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
6) Writes a structured, human-readable TXT with clear separators per article

Alternative flow (legacy Item mode)
- Previous versions supported exporting articles linked to a single Item via
    "@reverse.dcterms:subject". That flow has been removed in favor of the
    explicit Item Set export modes above.

Multi-part Output
- If more than 500 articles are found (to respect NotebookLM's 500k word limit),
    the output is automatically split into multiple files:
    - <item-title>_articles_part1.txt
    - <item-title>_articles_part2.txt
    - etc.

Output
- File path: NotebookLM/extracted_articles/<item-title>_articles.txt
    (or _part1.txt, _part2.txt, etc. for large collections)
- Encoding: UTF-8
- Layout per article:

        TITLE: <title>
        NEWSPAPER: <publisher(s)>
        DATE: <date>
        CONTENT:
        <article body>

        ==============================================================================

Notes
- The format favors readability and simple chunking for tools like NotebookLM.
- Only the first value is taken for multi-valued fields (date/content) to avoid
    duplications and keep the file compact.
- Maximum 500 articles per file to stay within NotebookLM's word limits.
"""

import os
import re
import sys
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple

import requests
from dotenv import load_dotenv

# Lightweight aliases to make intent clearer when reading types
JSONObj = Dict[str, Any]
JSONLike = Union[Dict[str, Any], List[Any]]

# -----------------------------------------------------------------------------
# Configuration: Country -> Item Set IDs mapping for "Whole IWAC" export
# Fill these lists with the Omeka Item Set IDs (as strings) that belong to each
# country. Leave a list empty to skip that country.
COUNTRY_ITEM_SETS: Dict[str, List[str]] = {
    "Benin": [
        "60638", "61062", "2185", "5502", "75959", "2186", "2188", "2187", "2191", "2190", "2189", "4922", "76053", "76081", "76059", "5501", "76070", "75960", "76071", "61063", "5500", "76072", "76073"
    ],
    "Burkina Faso": [
        "2199", "2200", "23448", "23273", "23449", "5503", "2215", "2214", "2207", "2209", "2210", "2213", "2201", "75969"
    ],
    "Côte d'Ivoire": [
        "43051",
    ],
    "Niger": [
        "62021",
    ],
    "Togo": [
        # "56789",
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
    """Perform a GET request and parse JSON.

    Args:
        url: Absolute Omeka API URL.
        params: Query parameters (auth keys, etc.).

    Returns:
        Parsed JSON (dict or list) on success, or None if any error occurs.
    """
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def extract_first_value(prop: Any) -> Optional[str]:
    """Extract the first "@value" from a multi-valued property.

    Many Omeka JSON-LD properties are arrays of objects like:
        [{"@value": "1998-02-16"}, ...]

    Args:
        prop: A property value from the Omeka JSON (any type).

    Returns:
        The first @value as a string, or None if not found.
    """
    if isinstance(prop, list):
        for entry in prop:
            if isinstance(entry, dict) and "@value" in entry:
                return str(entry["@value"]).strip()
    return None


def extract_publishers(article: JSONObj) -> List[str]:
    """Return publisher display titles from dcterms:publisher.

    Args:
        article: Omeka JSON object for a bibo:Article.

    Returns:
        Unique list of publisher display titles (may be empty).
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
    """Build a plain-text block for a single bibo:Article.

    The block is designed to be readable and chunk-friendly for NotebookLM.

    Args:
        article: Omeka JSON object representing a bibo:Article.

    Returns:
        Formatted string containing TITLE, NEWSPAPER, DATE, and CONTENT, followed
        by a consistent visual separator line.
    """
    title = article.get("o:title") or "No title"
    date = extract_first_value(article.get("dcterms:date")) or "Unknown"
    content = extract_first_value(article.get("bibo:content")) or ""
    publishers = extract_publishers(article)
    publisher_str = "; ".join(publishers) if publishers else "Unknown"

    lines = [
        f"TITLE: {title}",
        f"NEWSPAPER: {publisher_str}",
        f"DATE: {date}",
    ]
    if content:
        lines.append("CONTENT:")
        lines.append(content)
    lines.append("")
    lines.append("=" * 80)
    lines.append("")
    return "\n".join(lines)


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
    """Fetch all items that belong to the given Item Set (paginated).

    Args:
        env: Environment dict from load_env().
        set_id: Numeric string ID for the item set.
        per_page: Page size to request from Omeka API.

    Returns:
        List of item JSON objects (may be empty).
    """
    base_items_url = f"{env['base']}/items"
    all_items: List[JSONObj] = []
    page = 1
    while True:
        params = {
            "key_identity": env["kid"],
            "key_credential": env["kcr"],
            "item_set_id": set_id,
            "per_page": str(per_page),
            "page": str(page),
        }
        data = get_json(base_items_url, params)
        if not isinstance(data, list) or not data:
            break
        # Ensure dict objects only
        page_items = [d for d in data if isinstance(d, dict)]
        all_items.extend(page_items)
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
) -> Tuple[int, List[str]]:
    """Process a single Item Set ID: fetch, filter, and export to TXT.

    Args:
        env: Loaded environment dict with base URL and API keys.
        set_id: The Omeka Item Set ID (string).
        out_dir: Output directory for TXT files.
        max_items_per_file: Max articles per file before splitting.
        country_label: Optional country name for logging; not used in filenames.

    Returns:
        (article_count, [written_file_paths])
    """
    item_set = fetch_item_set(env, set_id)
    if not item_set:
        print(f"- Skipping Item Set {set_id}: not found or inaccessible.")
        return 0, []

    set_title = extract_first_value(item_set.get("dcterms:title")) or f"item_set_{set_id}"
    safe_set_title = sanitize_filename(set_title)

    prefix = f"[{country_label}] " if country_label else ""
    print(f"{prefix}Listing items in Item Set {set_id} (\"{set_title}\")...")
    items = fetch_items_in_set(env, set_id, per_page=100)
    print(f"{prefix}Found {len(items)} items. Filtering bibo:Article and bibo:Issue…")

    articles: List[JSONObj] = []
    for idx, it in enumerate(items, start=1):
        types = it.get("@type", [])
        if isinstance(types, list) and ("bibo:Article" in types or "bibo:Issue" in types):
            articles.append(it)
            if idx % 50 == 0:
                # Periodic progress ping without flooding the console
                print(f"  Processed {idx}/{len(items)}…")

    if not articles:
        print(f"{prefix}No bibo:Article items found in Item Set {set_id}.")
        return 0, []

    header_title = f"Item Set: {set_title} (ID {set_id})"
    # Include set ID in file name to avoid collisions across countries/sets with similar titles
    file_stub = f"{safe_set_title}_{set_id}"

    written_files: List[str] = []
    total_articles = len(articles)
    # If a country label is provided, write into a country subfolder
    target_dir = out_dir
    if country_label:
        country_dir = sanitize_filename(country_label)
        target_dir = os.path.join(out_dir, country_dir)
        os.makedirs(target_dir, exist_ok=True)
    if total_articles <= max_items_per_file:
        txt_path = os.path.join(target_dir, f"{file_stub}_articles.txt")
        write_articles_to_file(articles, txt_path, header_title)
        written_files.append(txt_path)
        print(f"{prefix}Wrote {total_articles} articles -> {os.path.basename(txt_path)}")
    else:
        num_parts = (total_articles + max_items_per_file - 1) // max_items_per_file
        print(f"{prefix}Total articles ({total_articles}) exceed {max_items_per_file}; splitting into {num_parts} parts…")
        for part_num in range(1, num_parts + 1):
            start_idx = (part_num - 1) * max_items_per_file
            end_idx = min(start_idx + max_items_per_file, total_articles)
            part_articles = articles[start_idx:end_idx]
            txt_path = os.path.join(target_dir, f"{file_stub}_articles_part{part_num}.txt")
            write_articles_to_file(part_articles, txt_path, header_title, part_num)
            written_files.append(txt_path)
            print(f"  Part {part_num}: {len(part_articles)} articles -> {os.path.basename(txt_path)}")

    return total_articles, written_files


def write_articles_to_file(articles: List[JSONObj], file_path: str, header_title: str, part_num: int = None) -> None:
    """Write a batch of articles to a single TXT file.
    
    Args:
        articles: List of article JSON objects to write.
        file_path: Full path to the output file.
        header_title: Title for the file header.
        part_num: Part number for multi-part exports (None for single file).
    """
    with open(file_path, "w", encoding="utf-8") as f:
        part_suffix = f" (Part {part_num})" if part_num else ""
        header = (
            f"{header_title}{part_suffix}\n"
            f"Exported: {datetime.now().isoformat(timespec='seconds')}\n"
            f"Articles: {len(articles)}\n"
        )
        f.write(header + "\n" + ("-" * 80) + "\n\n")
        for art in articles:
            f.write(format_article(art))


def main():
    print("\n=== Omeka -> TXT Export (NotebookLM-ready) ===")
    env = load_env()

    # Configuration: Maximum items per file to respect NotebookLM's 500k word limit
    MAX_ITEMS_PER_FILE = 500

    # Output directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "extracted_articles")
    os.makedirs(out_dir, exist_ok=True)

    # Convenience CLI args (optional):
    # - "all" or "--all" => whole collection
    # - a number           => single item set ID
    cli_arg = sys.argv[1].strip() if len(sys.argv) > 1 else None
    export_whole = False
    single_set_id: Optional[str] = None
    if cli_arg:
        if cli_arg.lower() in ("all", "--all"):
            export_whole = True
        elif cli_arg.isdigit():
            single_set_id = cli_arg
        else:
            print(f"Unrecognized CLI arg '{cli_arg}'. Ignoring and switching to interactive mode…")

    if not cli_arg:
        ans = input("Export the whole IWAC Collection? (y/N): ").strip().lower()
        export_whole = ans in ("y", "yes")

    if not export_whole and not single_set_id:
        single_set_id = input("Enter the Omeka Item Set ID to export: ").strip()
        if not single_set_id.isdigit():
            print("Item Set ID must be a number.")
            sys.exit(1)

    if export_whole:
        print("\n=== Whole IWAC Collection export ===")
        grand_total = 0
        all_written: List[str] = []
        # Iterate in the order provided above
        for country in ["Benin", "Burkina Faso", "Côte d'Ivoire", "Niger", "Togo"]:
            set_ids = COUNTRY_ITEM_SETS.get(country, [])
            if not set_ids:
                print(f"[Skip] {country}: no Item Set IDs configured.")
                continue
            print(f"\n-- {country} --")
            for sid in set_ids:
                if not isinstance(sid, str) or not sid.isdigit():
                    print(f"- Skipping invalid Item Set ID '{sid}' for {country}.")
                    continue
                count, files = process_item_set(env, sid, out_dir, MAX_ITEMS_PER_FILE, country_label=country)
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

    # Single Item Set export path
    assert single_set_id is not None
    count, files = process_item_set(env, single_set_id, out_dir, MAX_ITEMS_PER_FILE)
    if files:
        print(f"\nDone. Exported {count} articles to:")
        for p in files:
            print(f"  {p}")
    else:
        print("No files were created for the specified Item Set.")


if __name__ == "__main__":
    main()
