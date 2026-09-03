#!/usr/bin/env python3
"""
Collect the YouTube-hosted audiovisual items to transcribe.

This is the step that ``AI_audio_summary/01_omeka_media_downloader.py`` replaces
with a download, and it downloads nothing: a YouTube item's media carries no file
at all (the core ``youtube`` ingester stores only thumbnail derivatives, so
``o:original_url``, ``o:media_type`` and ``o:size`` are null), and Gemini fetches
the video from the URL itself. So this step only reads metadata and writes the
work list step ``02`` consumes.

Resource class 38 holds two populations since 2026-08-12 — deposited recordings
on template 19, which have real files and belong to the audio pipeline, and
embedded YouTube videos on template 23. They are separated here by template, and
then again by whether ``fabio:hasURL`` actually parses as a canonical watch URL.

Usage:
    python 01_omeka_youtube_fetcher.py
    python 01_omeka_youtube_fetcher.py --item-set-id 108260
    python 01_omeka_youtube_fetcher.py --item-id 108263 --item-id 108265
    python 01_omeka_youtube_fetcher.py --all-templates --include-transcribed

Requirements:
    - Environment variables: OMEKA_BASE_URL, OMEKA_KEY_IDENTITY, OMEKA_KEY_CREDENTIAL
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Add repo root to path for shared imports, and this pipeline's own directory
# for the sibling format module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.console_utils import count_table, key_value_table
from common.iwac_config import YOUTUBE_VIDEO_ITEM_SETS, YOUTUBE_VIDEO_TEMPLATE_ID
from common.log_redaction import install_credential_redaction
from common.omeka_client import OmekaClient

from youtube_source import VideoWork, format_hms, parse_iso_duration, parse_video_id, write_work_list

console = Console()

SCRIPT_DIR = Path(__file__).parent.resolve()
WORK_LIST_PATH = SCRIPT_DIR / "work" / "youtube_videos.json"

#: Omeka returns at most 100 resources per page whatever is requested.
PAGE_SIZE = 100


def setup_logging(log_folder: Path) -> None:
    """Configure logging to file and console."""
    log_folder.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_folder / "youtube_fetch.log", mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    # Credentials ride in Omeka query strings; keep them out of anything
    # urllib3 or an SDK decides to log.
    install_credential_redaction()


def first_value(item: Dict[str, Any], term: str) -> str:
    """Read the first value of a property as text.

    Follows the project's extraction priority — ``display_title`` for a linked
    resource, then ``@value`` for a literal, then ``@id`` for a URI, which is
    where ``fabio:hasURL`` keeps its URL.
    """
    for entry in item.get(term) or []:
        if not isinstance(entry, dict):
            continue
        for key in ("display_title", "@value", "@id"):
            value = entry.get(key)
            if value:
                return str(value)
    return ""


def fetch_items_by_template(client: OmekaClient, template_id: int) -> List[Dict[str, Any]]:
    """Every item on one resource template."""
    return client.get_items(resource_template_id=template_id)


def collect_items(
    client: OmekaClient,
    *,
    item_ids: List[int],
    item_set_ids: List[int],
    template_id: Optional[int],
) -> List[Dict[str, Any]]:
    """Fetch the candidate items for the requested scope, de-duplicated."""
    items: Dict[int, Dict[str, Any]] = {}

    for item_id in item_ids:
        item = client.get_item(item_id)
        if item:
            items[int(item["o:id"])] = item
        else:
            console.print(f"[yellow]⚠[/] Item {item_id} could not be fetched — skipped")

    for item_set_id in item_set_ids:
        extra = {"resource_template_id": template_id} if template_id else {}
        for item in client.get_items(item_set_id, **extra):
            items[int(item["o:id"])] = item

    if template_id and not item_ids and not item_set_ids:
        for item in fetch_items_by_template(client, template_id):
            items[int(item["o:id"])] = item

    return [items[key] for key in sorted(items)]


def to_video_work(item: Dict[str, Any]) -> Optional[VideoWork]:
    """Resolve one item to a fetchable video, or ``None`` if it is not one."""
    raw_url = first_value(item, "fabio:hasURL")
    video_id = parse_video_id(raw_url)
    if not video_id:
        return None
    return VideoWork(
        item_id=int(item["o:id"]),
        video_id=video_id,
        # Rebuilt from the id rather than passed through: a stored URL may carry
        # a `&t=` or tracking parameters, and the request should not.
        url=f"https://www.youtube.com/watch?v={video_id}",
        title=str(item.get("o:title") or first_value(item, "dcterms:title")),
        identifier=first_value(item, "dcterms:identifier"),
        duration_seconds=parse_iso_duration(first_value(item, "dcterms:extent")),
        language=first_value(item, "dcterms:language"),
        has_content=bool((item.get("bibo:content") or [])),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect YouTube-hosted Omeka items into a transcription work list.",
    )
    parser.add_argument(
        "--item-set-id", type=int, action="append", default=[], dest="item_set_ids",
        help=f"Item set to read (repeatable). Default: {', '.join(YOUTUBE_VIDEO_ITEM_SETS)}.",
    )
    parser.add_argument(
        "--item-id", type=int, action="append", default=[], dest="item_ids",
        help="Single item to read (repeatable). Skips the item-set scope.",
    )
    parser.add_argument(
        "--template-id", type=int, default=YOUTUBE_VIDEO_TEMPLATE_ID,
        help=f"Resource template of YouTube items (default: {YOUTUBE_VIDEO_TEMPLATE_ID}).",
    )
    parser.add_argument(
        "--all-templates", action="store_true",
        help="Do not filter by resource template. Items are still filtered by "
             "whether fabio:hasURL is a canonical YouTube watch URL.",
    )
    parser.add_argument(
        "--include-transcribed", action="store_true",
        help="Include items that already hold bibo:content (they are skipped by default).",
    )
    parser.add_argument(
        "--output", type=Path, default=WORK_LIST_PATH,
        help=f"Where the work list is written (default: {WORK_LIST_PATH.name} under work/).",
    )
    return parser.parse_args()


def report(videos: List[VideoWork], skipped: List[VideoWork], rejected: List[int]) -> None:
    """Print what was collected, and what was left out and why."""
    if videos:
        table = Table(title=f"🎬 Videos to transcribe ({len(videos)})", box=box.ROUNDED)
        table.add_column("Item", style="cyan", justify="right")
        table.add_column("Identifier", style="dim")
        table.add_column("Duration", justify="right", style="green")
        table.add_column("Catalogued", style="dim")
        table.add_column("Title")
        for video in videos:
            table.add_row(
                str(video.item_id),
                video.identifier or "—",
                format_hms(video.duration_seconds),
                video.language or "—",
                (video.title[:52] + "…") if len(video.title) > 53 else video.title,
            )
        console.print(table)

    total_seconds = sum(video.duration_seconds or 0 for video in videos)
    unknown = sum(1 for video in videos if video.duration_seconds is None)
    console.print()
    console.print(count_table([
        ("Videos collected", len(videos)),
        ("Total runtime", f"{total_seconds / 3600:.2f} h" if total_seconds else "0 h"),
        ("Duration unrecorded", unknown or None),
        ("Already transcribed (skipped)", len(skipped) or None),
        ("No usable YouTube URL", len(rejected) or None),
    ], title="Summary"))

    if total_seconds > 8 * 3600:
        console.print(
            f"\n[yellow]⚠[/] {total_seconds / 3600:.1f} h of video exceeds the free tier's "
            "8 h/day YouTube cap — a free-tier run will stop partway through with "
            "QuotaExhaustedError and can be resumed the next day."
        )
    if rejected:
        console.print(
            f"\n[yellow]⚠[/] {len(rejected)} item(s) had no canonical watch URL in "
            f"fabio:hasURL: {rejected[:10]}{' …' if len(rejected) > 10 else ''}"
        )
        console.print(
            "[dim]  Gemini accepts only youtube.com/watch?v=<id> and youtu.be/<id>. "
            "A /shorts/, /live/ or /embed/ link has to be normalised on the item first.[/]"
        )


def main() -> int:
    args = parse_args()
    setup_logging(SCRIPT_DIR / "log")

    console.print(Panel(
        "Read YouTube-hosted audiovisual items from Omeka S and write the work "
        "list for step 02. Nothing is downloaded — Gemini fetches the video itself.",
        title="🎬 Omeka S YouTube Fetcher",
        border_style="cyan",
    ))

    item_set_ids = args.item_set_ids
    if not item_set_ids and not args.item_ids:
        item_set_ids = [int(value) for value in YOUTUBE_VIDEO_ITEM_SETS]
    template_id = None if args.all_templates else args.template_id

    console.print()
    console.print(key_value_table([
        ("Item sets", ", ".join(str(value) for value in item_set_ids) or None),
        ("Items", ", ".join(str(value) for value in args.item_ids) or None),
        ("Template", str(template_id) if template_id else "any"),
        ("Already transcribed", "included" if args.include_transcribed else "skipped"),
        ("Work list", str(args.output)),
    ]))
    console.print()

    try:
        client = OmekaClient.from_env()
    except ValueError as exc:
        console.print(f"[red]Configuration Error:[/] {exc}")
        return 1

    try:
        with console.status("[cyan]Fetching items from Omeka...[/]"):
            items = collect_items(
                client,
                item_ids=args.item_ids,
                item_set_ids=item_set_ids,
                template_id=template_id,
            )
    except Exception as exc:
        console.print(f"[red]✗[/] Could not fetch items: {exc}")
        logging.exception("Item fetch failed")
        return 1

    if not items:
        console.print("[yellow]No items found for the requested scope.[/]")
        return 0

    videos: List[VideoWork] = []
    skipped: List[VideoWork] = []
    rejected: List[int] = []
    for item in items:
        video = to_video_work(item)
        if video is None:
            rejected.append(int(item["o:id"]))
            continue
        if video.has_content and not args.include_transcribed:
            skipped.append(video)
            continue
        videos.append(video)

    report(videos, skipped, rejected)

    if not videos:
        console.print(
            "\n[yellow]Nothing to transcribe.[/] "
            "Pass --include-transcribed to re-transcribe items that already have content."
        )
        return 0

    path = write_work_list(args.output, videos, scope={
        "item_set_ids": item_set_ids,
        "item_ids": args.item_ids,
        "template_id": template_id,
        "include_transcribed": args.include_transcribed,
        "omeka": client.base_url,
    })
    console.print(f"\n[green]✓[/] Work list written: [cyan]{path}[/]")
    logging.info("Wrote %d video(s) to %s", len(videos), path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
