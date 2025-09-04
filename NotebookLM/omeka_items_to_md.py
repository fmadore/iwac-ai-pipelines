"""
Omeka item to NotebookLM-ready Markdown exporter.

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
    """Normalize whitespace to keep Markdown tidy.

    - Convert CRLF/CR to LF
    - Replace non-breaking spaces with regular spaces
    - Trim trailing spaces
    - Collapse 2+ consecutive blank lines to a single blank line
    - Strip leading/trailing blank lines
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
    """Build a Markdown block for a single bibo:Article.

    The block is designed to be readable and chunk-friendly for NotebookLM.

    Args:
        article: Omeka JSON object representing a bibo:Article.

    Returns:
        Markdown string with a level-1 heading for the title, bold metadata
        lines (Newspaper, Date), the content body, and a horizontal rule.
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
    """Extract numeric ID from an Omeka '@id' URL.

    Supports both '/api/items/<id>' and '/api/resources/<id>' forms.
    Example: https://example.org/api/items/23601 -> '23601'
             https://example.org/api/resources/5717 -> '5717'
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
    """Try fetching an Item by ID, falling back to the generic Resource endpoint.

    Returns the JSON object or None if neither fetch succeeds.
    """
    it = fetch_item(env, id_str)
    if isinstance(it, dict):
        return it
    return fetch_resource(env, id_str)


def fetch_articles_with_subject(env: Dict[str, str], subject_item_id: str) -> Tuple[Optional[JSONObj], List[JSONObj]]:
    """Return the subject item and all bibo:Article items that reference it via dcterms:subject.

    Strategy: read '@reverse.dcterms:subject' from the subject item, then fetch
    each referenced item to confirm type and gather full content.

    Args:
        env: Environment dict.
        subject_item_id: The Item ID used as the subject authority.

    Returns:
        (subject_item_json_or_None, [article_items])
    """
    subject_item = fetch_item(env, subject_item_id)
    if not subject_item:
        return None, []

    reverse = subject_item.get("@reverse") or {}
    refs = []
    if isinstance(reverse, dict):
        refs = reverse.get("dcterms:subject") or []
    if not isinstance(refs, list):
        refs = []
    total_refs = len(refs)
    print(f"Found {total_refs} reverse subject references.")

    article_items: List[JSONObj] = []
    seen_ids: set[str] = set()
    skipped_non_article = 0
    fetch_fail = 0
    dupes = 0
    start_ts = time.perf_counter()
    for idx, ref in enumerate(refs, start=1):
        if not isinstance(ref, dict):
            continue
        # Prefer embedded types when available to avoid a fetch if clearly not an article.
        ref_types = ref.get("@type")
        if isinstance(ref_types, list) and "bibo:Article" not in ref_types and "bibo:Issue" not in ref_types:
            # Skip clearly non-article/issue items
            skipped_non_article += 1
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
        # Get numeric ID to fetch full record
        rid = None
        if isinstance(ref.get("o:id"), int):
            rid = str(ref["o:id"]) 
        elif isinstance(ref.get("o:id"), str) and ref["o:id"].isdigit():
            rid = ref["o:id"]
        elif isinstance(ref.get("@id"), str):
            rid = parse_id_from_at_id(ref["@id"])
        if not rid:
            continue
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
        # Try as /items/<id>, then fallback to /resources/<id>
        item = fetch_item_or_resource(env, rid)
        if not isinstance(item, dict):
            fetch_fail += 1
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
        types = item.get("@type", [])
        if isinstance(types, list) and ("bibo:Article" in types or "bibo:Issue" in types):
            article_items.append(item)
        # live progress update (single-line)
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
        # Every 50, drop a newline snapshot to keep some history
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
    file_ext: str = "md",
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
    # Normalize file extension (no leading dot)
    file_ext = (file_ext or "md").lower().lstrip(".")

    if total_articles <= max_items_per_file:
        out_path = os.path.join(target_dir, f"{file_stub}_articles.{file_ext}")
        write_articles_to_file(articles, out_path, header_title)
        written_files.append(out_path)
        print(f"{prefix}Wrote {total_articles} articles -> {os.path.basename(out_path)}")
    else:
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
    print("\n=== Omeka -> Markdown Export (NotebookLM-ready) ===")
    env = load_env()

    # Configuration: Maximum items per file to respect NotebookLM's 500k word limit
    MAX_ITEMS_PER_FILE = 250

    # Output directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "extracted_articles")
    os.makedirs(out_dir, exist_ok=True)

    # Output file extension (default to md). Override via env NOTEBOOKLM_EXPORT_EXT (md|txt).
    file_ext = os.getenv("NOTEBOOKLM_EXPORT_EXT", "md").lower().lstrip(".")
    if file_ext not in ("md", "txt"):
        print(f"Unrecognized NOTEBOOKLM_EXPORT_EXT='{file_ext}', defaulting to 'md'.")
        file_ext = "md"

    # Convenience CLI args (optional):
    # - "all" or "--all"                     => whole collection
    # - a number                               => single item set ID
    # - "subject:<id>" | "s:<id>" | --subject <id> => export items referencing subject item
    cli_arg = sys.argv[1].strip() if len(sys.argv) > 1 else None
    export_whole = False
    single_set_id: Optional[str] = None
    subject_item_id: Optional[str] = None
    if cli_arg:
        low = cli_arg.lower()
        if low in ("all", "--all"):
            export_whole = True
        elif low.startswith("subject:") or low.startswith("s:"):
            maybe_id = cli_arg.split(":", 1)[1]
            if maybe_id.isdigit():
                subject_item_id = maybe_id
        elif low == "--subject" and len(sys.argv) > 2 and sys.argv[2].strip().isdigit():
            subject_item_id = sys.argv[2].strip()
        elif cli_arg.isdigit():
            single_set_id = cli_arg
        else:
            print(f"Unrecognized CLI arg '{cli_arg}'. Ignoring and switching to interactive mode…")

    if not cli_arg:
        mode = input("Choose export mode: [1] Whole IWAC, [2] Item Set by ID, [3] Items with dcterms:subject Item ID: ").strip().lower()
        if mode in ("1", "all", "a", "one"):
            export_whole = True
        elif mode in ("3", "subject", "s"):
            subject_item_id = input("Enter the subject Item ID: ").strip()
            if not subject_item_id.isdigit():
                print("Subject Item ID must be a number.")
                sys.exit(1)
        else:
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

    # Subject-based export path
    if subject_item_id:
        count, files = process_subject_items(env, subject_item_id, out_dir, MAX_ITEMS_PER_FILE, file_ext=file_ext)
        if files:
            print(f"\nDone. Exported {count} articles to:")
            for p in files:
                print(f"  {p}")
        else:
            print("No files were created for the specified subject.")
        return

    # Single Item Set export path
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
