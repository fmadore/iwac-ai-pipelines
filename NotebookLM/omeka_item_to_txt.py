"""
Omeka item to NotebookLM-ready TXT exporter.

This script fetches all newspaper articles linked to a single Omeka item and
exports them to one consolidated UTF-8 text file that’s easy to ingest into
NotebookLM.

Input
- An Omeka "item ID" (CLI arg or interactive prompt)
- Environment: OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL in .env

How it works
1) Loads credentials from .env
2) Downloads the main item: GET {OMEKA_BASE_URL}/items/{item_id}
3) Looks at "@reverse.dcterms:subject" to find related items
4) Keeps only items with type "bibo:Article"
5) For each article, extracts:
     - TITLE: o:title
     - NEWSPAPER: dcterms:publisher (display_title)
     - DATE: dcterms:date (first @value)
     - CONTENT: bibo:content (first @value)
6) Writes a structured, human-readable TXT with clear separators per article

Alternative flow (Item Set)
- Instead of a single item, you can choose an Item Set ID. The script fetches
    all items in that set (paginated), keeps those of type "bibo:Article", and
    produces the same NotebookLM-friendly TXT. The Item Set title is taken from
    dcterms:title[0].@value.

Output
- File path: NotebookLM/extracted_articles/<item-title>_articles.txt
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
"""

import os
import re
import sys
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

import requests
from dotenv import load_dotenv

# Lightweight aliases to make intent clearer when reading types
JSONObj = Dict[str, Any]
JSONLike = Union[Dict[str, Any], List[Any]]


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


def main():
    print("\n=== Omeka -> TXT Export (NotebookLM-ready) ===")
    env = load_env()

    # Mode selection: Item (reverse subjects) or Item Set.
    mode = None
    cli_arg = sys.argv[1].strip() if len(sys.argv) > 1 else None
    if cli_arg and cli_arg.isdigit():
        # Backward-compatible: single numeric arg implies Item mode with that ID
        mode = "item"
        item_id = cli_arg
        print(f"Using Item mode with ID from arguments: {item_id}")
    else:
        print("Choose source mode:")
        print("  1) Item (via @reverse.dcterms:subject) [default]")
        print("  2) Item Set (all items in the set)")
        choice = input("Enter 1 or 2: ").strip() or "1"
        mode = "set" if choice == "2" else "item"

    # Prepare output directory relative to this script. NotebookLM can then
    # ingest the generated TXT directly from NotebookLM/extracted_articles.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "extracted_articles")
    os.makedirs(out_dir, exist_ok=True)

    params = {
        "key_identity": env["kid"],
        "key_credential": env["kcr"],
    }

    articles: List[Dict[str, Any]] = []
    header_title = ""
    header_prefix = ""
    file_stub = ""

    if mode == "item":
        # Determine the target item ID from CLI fallback or prompt.
        if not cli_arg:
            item_id = input("Enter the Omeka item ID: ").strip()
        if not item_id.isdigit():
            print("Item ID must be a number.")
            sys.exit(1)

        # Fetch the main Omeka item to discover related content.
        item_url = f"{env['base']}/items/{item_id}"
        item = get_json(item_url, params)
        if not item or not isinstance(item, dict):
            print("Could not fetch the main item. Exiting.")
            sys.exit(1)

        item_title = item.get("o:title") or f"omeka_item_{item_id}"
        safe_title = sanitize_filename(item_title)

        # Collect candidate related items from @reverse.dcterms:subject.
        reverse = item.get("@reverse", {})
        related = reverse.get("dcterms:subject") or []
        candidate_urls: List[str] = []
        for entry in related:
            if isinstance(entry, dict) and "@id" in entry:
                candidate_urls.append(entry["@id"])

        if not candidate_urls:
            print("No related items found under dcterms:subject.")
            sys.exit(0)

        print(f"Found {len(candidate_urls)} related items. Fetching articles...")
        for idx, url in enumerate(candidate_urls, start=1):
            data = get_json(url, params)
            if not isinstance(data, dict):
                continue
            types = data.get("@type", [])
            if isinstance(types, list) and "bibo:Article" in types:
                articles.append(data)
                title = data.get("o:title", "Untitled")
                print(f"  [{idx}/{len(candidate_urls)}] Article: {title}")

        header_title = f"Item: {item_title} (ID {item_id})"
        header_prefix = "Item"
        file_stub = safe_title

    else:  # mode == "set"
        set_id = input("Enter the Omeka item set ID: ").strip()
        if not set_id.isdigit():
            print("Item set ID must be a number.")
            sys.exit(1)

        # Fetch item set (for title) and then all items in the set.
        item_set = fetch_item_set(env, set_id)
        if not item_set:
            print("Could not fetch the item set. Exiting.")
            sys.exit(1)

        set_title = extract_first_value(item_set.get("dcterms:title")) or f"item_set_{set_id}"
        safe_set_title = sanitize_filename(set_title)

        print("Listing items in the set (may take a moment)...")
        items = fetch_items_in_set(env, set_id, per_page=100)
        print(f"Found {len(items)} items in the set. Filtering bibo:Article and bibo:Issue...")

        for idx, it in enumerate(items, start=1):
            types = it.get("@type", [])
            if isinstance(types, list) and ("bibo:Article" in types or "bibo:Issue" in types):
                articles.append(it)
                title = it.get("o:title", "Untitled")
                label = "Article" if "bibo:Article" in types else ("Issue" if "bibo:Issue" in types else "Item")
                print(f"  [{idx}/{len(items)}] {label}: {title}")

        header_title = f"Item Set: {set_title} (ID {set_id})"
        header_prefix = "Item Set"
        file_stub = safe_set_title

    if not articles:
        print("No bibo:Article items found.")
        sys.exit(0)

    # Build TXT: one header for the whole export, then a block per article.
    txt_path = os.path.join(out_dir, f"{file_stub}_articles.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        header = (
            f"{header_title}\n"
            f"Exported: {datetime.now().isoformat(timespec='seconds')}\n"
            f"Articles: {len(articles)}\n"
        )
        f.write(header + "\n" + ("-" * 80) + "\n\n")
        for art in articles:
            f.write(format_article(art))

    print(f"\nDone. Wrote {len(articles)} articles to:\n{txt_path}")


if __name__ == "__main__":
    main()
