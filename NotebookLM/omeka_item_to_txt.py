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


def main():
    print("\n=== Omeka Item -> Single TXT Export (NotebookLM-ready) ===")
    env = load_env()

    # Determine the target item ID from the CLI (preferred) or prompt.
    if len(sys.argv) > 1:
        item_id = sys.argv[1].strip()
        print(f"Using item ID from arguments: {item_id}")
    else:
        item_id = input("Enter the Omeka item ID: ").strip()
    if not item_id.isdigit():
        print("Item ID must be a number.")
        sys.exit(1)

    # Prepare output directory relative to this script. NotebookLM can then
    # ingest the generated TXT directly from NotebookLM/extracted_articles.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "extracted_articles")
    os.makedirs(out_dir, exist_ok=True)

    # Fetch the main Omeka item to discover related content.
    item_url = f"{env['base']}/items/{item_id}"
    params = {
        "key_identity": env["kid"],
        "key_credential": env["kcr"],
    }
    item = get_json(item_url, params)
    if not item:
        print("Could not fetch the main item. Exiting.")
        sys.exit(1)

    item_title = item.get("o:title") or f"omeka_item_{item_id}"
    safe_title = sanitize_filename(item_title)

    # Collect candidate related items from @reverse.dcterms:subject.
    # Omeka places reverse references under the "@reverse" key; we filter for
    # items that actually are of type bibo:Article a few lines below.
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

    articles: List[Dict[str, Any]] = []
    for idx, url in enumerate(candidate_urls, start=1):
        data = get_json(url, params)
        if not data:
            continue
        types = data.get("@type", [])
        if isinstance(types, list) and "bibo:Article" in types:
            articles.append(data)
            title = data.get("o:title", "Untitled")
            print(f"  [{idx}/{len(candidate_urls)}] Article: {title}")

    if not articles:
        print("No bibo:Article items found.")
        sys.exit(0)

    # Build TXT: one header for the whole export, then a block per article.
    txt_path = os.path.join(out_dir, f"{safe_title}_articles.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        header = f"Item: {item_title} (ID {item_id})\nExported: {datetime.now().isoformat(timespec='seconds')}\nArticles: {len(articles)}\n"
        f.write(header + "\n" + ("-" * 80) + "\n\n")
        for art in articles:
            f.write(format_article(art))

    print(f"\nDone. Wrote {len(articles)} articles to:\n{txt_path}")


if __name__ == "__main__":
    main()
