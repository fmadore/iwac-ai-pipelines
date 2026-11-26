#!/usr/bin/env python3
"""
Structured OCR for scholarly publications, via Mistral Document AI (OCR 4.1).

Each PDF in ``PDF/`` is sent to the OCR endpoint with ``include_blocks=True``,
which returns paragraph-level blocks carrying a structural label — ``text``,
``title``, ``table``, ``references``, ``header``, ``footer`` and the rest — in
reading order. Two files come back out per document:

    OCR_Results/<item_id>.txt    plain text for bibo:content (step 03)
    OCR_Results/<item_id>.json   the typed blocks, with bboxes and roles

The JSON sidecar is the point of this pipeline. It is what a later citation pass
reads: the footnotes and the bibliography are already separated from the body
there, so extracting cited works never has to guess where the apparatus starts.

Why this is not a flag on ``AI_ocr_extraction/02_mistral_ocr_processor.py``:
that script drops every page's header and footer from page 2 on, which is right
for a newspaper's running head and destructive here. On a 33-page article from
the Cahiers du CERLESHS, the page feet held 53 substantive footnotes — 7,388
characters, including full citations — against 32 folio numbers. So a foot is
kept unless it is a folio number or repeats across the document; see
``common.mistral_ocr.classify_blocks``.

Oversized scans are handled rather than skipped: Mistral rejects uploads over
50 MB, and four documents in the backlog exceed that (up to 276 MB). They are
split by page range, OCR'd part by part, and stitched back with their original
page numbers.

Usage:
    python 02_mistral_blocks_processor.py
    python 02_mistral_blocks_processor.py --item-id 5312    # pilot on one document
    python 02_mistral_blocks_processor.py --limit 3
    python 02_mistral_blocks_processor.py --force           # redo completed documents
    python 02_mistral_blocks_processor.py --rpm 30          # proactive throttling

Requirements:
    - MISTRAL_API_KEY in the environment / .env
    - PDFs in PDF/ (named <item_id>.pdf by 01_fetch_publication_pdfs.py)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.artifacts import artifact_matches, commit_artifact, invalidate_artifact
from common.checkpoint import atomic_write_text, sha256_file
from common.console_utils import key_value_table
from common.log_redaction import configure_logging
from common.mistral_ocr import (
    BATCH_PRICE_PER_1000_PAGES,
    BATCH_TERMINAL_STATUSES,
    MISTRAL_OCR_MODEL,
    PRICE_PER_1000_PAGES,
    MistralOcrClient,
    classify_blocks,
    render_plain_text,
)
from common.rate_limiter import QuotaExhaustedError

console = Console()

SCRIPT_DIR = Path(__file__).resolve().parent
PDF_DIR = SCRIPT_DIR / "PDF"
RESULTS_DIR = SCRIPT_DIR / "OCR_Results"
#: One manifest per submitted batch job: what was sent, so the results can be
#: collected later, by ``--batch-job``, from another process.
BATCH_DIR = RESULTS_DIR / "batch_jobs"
LOG_DIR = SCRIPT_DIR / "log"

#: Bumped when the block→text rules change, so ``--force`` is not the only way
#: to tell an old sidecar from one this version would produce.
SIDECAR_VERSION = 1


def build_sidecar(
    pdf_path: Path, result, blocks, text: str, elapsed: Optional[float],
    batch_job_id: Optional[str] = None,
) -> Dict:
    """Assemble the JSON written beside the plain text."""
    role_counts = Counter(b.role for b in blocks)
    type_counts = Counter(b.type for b in blocks)

    pages = []
    by_page: Dict[int, List] = {}
    for block in blocks:
        by_page.setdefault(block.page_index, []).append(block)

    for raw in result.pages:
        index = raw.get("index", 0)
        pages.append({
            "index": index,
            "dimensions": raw.get("dimensions"),
            "tables": raw.get("tables") or [],
            "hyperlinks": raw.get("hyperlinks") or [],
            "blocks": [
                {
                    "type": b.type,
                    "role": b.role,
                    "content": b.content,
                    "bbox": b.bbox,
                }
                for b in by_page.get(index, [])
            ],
        })

    return {
        "sidecar_version": SIDECAR_VERSION,
        "source_pdf": pdf_path.name,
        "source_bytes": pdf_path.stat().st_size,
        "model": result.model,
        "pages_processed": result.pages_processed,
        "upload_parts": result.parts,
        "warnings": result.warnings,
        "elapsed_seconds": round(elapsed, 2) if elapsed is not None else None,
        "batch_job_id": batch_job_id,
        "text_characters": len(text),
        "block_roles": dict(role_counts),
        "block_types": dict(type_counts),
        "pages": pages,
    }


def process_one(
    client: MistralOcrClient, pdf_path: Path, results_dir: Path
) -> Optional[Dict]:
    """OCR one PDF and write its ``.txt`` and ``.json``. Returns the sidecar."""
    console.print()
    console.rule(f"[bold]📄 {pdf_path.name}[/]")
    size_mb = pdf_path.stat().st_size / 1e6
    console.print(f"  [dim]Size:[/] {size_mb:,.1f} MB")

    invalidate_artifact(results_dir / f"{pdf_path.stem}.txt")
    source_hash = sha256_file(pdf_path)
    started = time.time()
    result = client.process_pdf(pdf_path)
    return write_outputs(pdf_path, result, results_dir, source_hash, time.time() - started)


def write_outputs(
    pdf_path: Path, result, results_dir: Path, source_hash: str,
    elapsed: Optional[float], batch_job_id: Optional[str] = None,
) -> Optional[Dict]:
    """Classify one OCR result's blocks and write ``.txt``, ``.json`` and the artifact.

    Shared by the direct and the batch path, so both write the same files.
    """
    output_file = results_dir / f"{pdf_path.stem}.txt"
    for warning in result.warnings:
        console.print(f"  [yellow]⚠[/] {warning}")

    blocks = classify_blocks(result.pages)
    text = render_plain_text(blocks)

    if not text.strip():
        console.print("  [red]✗[/] OCR produced no text — nothing written")
        logging.error("No text extracted from %s", pdf_path.name)
        return None

    sidecar = build_sidecar(pdf_path, result, blocks, text, elapsed, batch_job_id)

    output_file.write_text(text, encoding="utf-8")
    (results_dir / f"{pdf_path.stem}.json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    commit_artifact(output_file, context={"pipeline": "publication-ocr-v2", "model_key": result.model,
                                        "model_id": result.model}, source_sha256=source_hash,
                    companions=[results_dir / f"{pdf_path.stem}.json"])

    roles = sidecar["block_roles"]
    console.print(
        f"  [green]✓[/] {result.pages_processed} pages"
        + (f" in {elapsed:,.1f}s" if elapsed is not None else "")
        + f" — {len(text):,} characters"
    )
    console.print(
        f"  [dim]Blocks:[/] {roles.get('body', 0)} body · "
        f"[cyan]{roles.get('apparatus', 0)} apparatus[/] · "
        f"{roles.get('furniture', 0)} furniture dropped"
    )
    logging.info(
        "%s: %d pages, %d chars, roles=%s",
        pdf_path.name, result.pages_processed, len(text), roles,
    )
    return sidecar


def _block_table(sidecars: List[Dict]) -> Table:
    """Aggregate what the structural labels actually found across the run."""
    totals: Counter = Counter()
    for sidecar in sidecars:
        for page in sidecar["pages"]:
            for block in page["blocks"]:
                totals[(block["type"], block["role"])] += len(block["content"] or "")

    table = Table(title="Structure found", box=box.ROUNDED)
    table.add_column("Block type", style="cyan")
    table.add_column("Role", style="dim")
    table.add_column("Characters", justify="right")
    for (btype, role), chars in sorted(totals.items(), key=lambda kv: -kv[1]):
        style = "green" if role == "apparatus" else None
        table.add_row(btype, role, f"{chars:,}", style=style)
    return table


def _summary(sidecars: List[Dict], failed: int, price_per_1000: float, elapsed: float) -> None:
    console.print()
    console.rule("[bold cyan]Summary")
    total_pages = sum(s["pages_processed"] for s in sidecars)
    console.print(key_value_table([
        ("Documents processed", len(sidecars)),
        ("Failed", failed),
        ("Pages", f"{total_pages:,}"),
        ("Characters", f"{sum(s['text_characters'] for s in sidecars):,}"),
        ("Billed (approx.)", f"~${total_pages * price_per_1000 / 1000:,.2f}"),
        ("Elapsed", f"{elapsed:,.1f}s"),
    ], title="Run", value_style="cyan"))

    if sidecars:
        console.print()
        console.print(_block_table(sidecars))
        console.print(
            f"\n[green]✓[/] Results in [cyan]{RESULTS_DIR}[/] "
            "([dim].txt for step 03, .json for the citation pass[/])"
        )
        console.print("[dim]Next: python 03_omeka_content_updater.py --dry-run[/]")


def run_direct(client: MistralOcrClient, pdf_files: List[Path]) -> int:
    """One synchronous OCR call per document, at the list price."""
    sidecars: List[Dict] = []
    failed = 0
    quota_hit = False
    started = time.time()

    for pdf_path in pdf_files:
        try:
            sidecar = process_one(client, pdf_path, RESULTS_DIR)
            if sidecar:
                sidecars.append(sidecar)
            else:
                failed += 1
        except QuotaExhaustedError as exc:
            console.print(f"\n[red]✗ Quota exhausted[/] — stopping. {exc}")
            logging.error("Quota exhausted on %s: %s", pdf_path.name, exc)
            quota_hit = True
            break
        except Exception as exc:
            console.print(f"  [red]✗[/] {pdf_path.name} failed: {exc}")
            logging.error("Failed on %s: %s", pdf_path.name, exc, exc_info=True)
            failed += 1

    _summary(sidecars, failed, PRICE_PER_1000_PAGES, time.time() - started)
    if quota_hit:
        return 1
    return 0 if failed == 0 else 1


# --- Batch mode --------------------------------------------------------------


def _manifest_path(job_id: str) -> Path:
    return BATCH_DIR / f"{job_id}.json"


def _save_manifest(manifest: Dict) -> None:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(_manifest_path(manifest["job_id"]),
                      json.dumps(manifest, ensure_ascii=False, indent=2))


def submit_batch(client: MistralOcrClient, pdf_files: List[Path], pdf_dir: Path) -> Dict:
    """Upload every document, queue one job, and save its manifest before anything else."""
    documents = []
    for pdf_path in pdf_files:
        # The artifact is invalidated now, not at collection: from here on the
        # file on disk is no longer the result of the current request.
        invalidate_artifact(RESULTS_DIR / f"{pdf_path.stem}.txt")
        documents.append({"pdf_path": pdf_path, "source_sha256": sha256_file(pdf_path)})

    console.print(f"[cyan]Uploading {len(documents)} document(s) to Mistral…[/]")
    manifest = client.submit_batch(documents, metadata={"pipeline": "publication-ocr-v2"})
    manifest["pdf_dir"] = str(pdf_dir)
    manifest["submitted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _save_manifest(manifest)
    pages = sum(doc["pages"] for doc in manifest["documents"].values())
    console.print(Panel(
        f"Job [bold]{manifest['job_id']}[/] queued — {len(documents)} documents, {pages:,} pages, "
        f"~${pages * BATCH_PRICE_PER_1000_PAGES / 1000:,.2f} at the batch price.\n"
        f"Manifest: {_manifest_path(manifest['job_id'])}\n"
        "Mistral answers within 24 hours. Stopping this script does not stop the job:\n"
        f"collect it later with [cyan]--batch-job {manifest['job_id']}[/].",
        title="Batch submitted", border_style="green",
    ))
    logging.info("Submitted batch %s: %d documents, %d pages", manifest["job_id"], len(documents), pages)
    return manifest


def wait_for_job(client: MistralOcrClient, job_id: str, poll_seconds: int):
    """Poll until the job reaches a terminal state; return the job."""
    last = None
    while True:
        job = client.batch_job(job_id)
        status = str(getattr(job.status, "value", job.status))
        line = (f"{status}: {job.completed_requests}/{job.total_requests} requests "
                f"({job.failed_requests} failed)")
        if line != last:
            console.print(f"  [dim]{time.strftime('%H:%M:%S')}[/] {line}")
            last = line
        if status in BATCH_TERMINAL_STATUSES:
            return job
        time.sleep(poll_seconds)


def collect_batch(client: MistralOcrClient, manifest: Dict, job) -> int:
    """Write every complete document of a finished job; report the rest."""
    results, errors = client.collect_batch(manifest, job)
    pdf_dir = Path(manifest.get("pdf_dir") or PDF_DIR)
    sidecars: List[Dict] = []
    failed = 0

    for stem, doc in manifest["documents"].items():
        pdf_path = pdf_dir / doc["pdf"]
        if stem in errors:
            console.print(f"  [red]✗[/] {doc['pdf']}: {errors[stem]}")
            logging.error("Batch %s: %s failed: %s", manifest["job_id"], doc["pdf"], errors[stem])
            failed += 1
            continue
        # The artifact records the hash of what was sent; a PDF replaced since
        # submission would be stamped with text from another file.
        if not pdf_path.exists() or sha256_file(pdf_path) != doc["source_sha256"]:
            console.print(f"  [red]✗[/] {doc['pdf']}: changed or missing since submission — not written")
            logging.error("Batch %s: %s changed since submission", manifest["job_id"], doc["pdf"])
            failed += 1
            continue
        console.print()
        console.rule(f"[bold]📄 {pdf_path.name}[/]")
        sidecar = write_outputs(pdf_path, results[stem], RESULTS_DIR, doc["source_sha256"],
                                None, batch_job_id=manifest["job_id"])
        if sidecar:
            sidecars.append(sidecar)
        else:
            failed += 1

    client.cleanup_batch(manifest, job)
    manifest["collected_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["status"] = str(getattr(job.status, "value", job.status))
    manifest["written"] = [s["source_pdf"] for s in sidecars]
    manifest["errors"] = errors
    _save_manifest(manifest)

    _summary(sidecars, failed, BATCH_PRICE_PER_1000_PAGES, 0.0)
    if failed:
        console.print("[yellow]⚠[/] Re-run with --batch (or without it) to retry the failed documents.")
    return 0 if failed == 0 else 1


def run_batch(client: MistralOcrClient, manifest: Dict, poll_seconds: int) -> int:
    job_id = manifest["job_id"]
    if manifest.get("collected_at"):
        console.print(f"[yellow]⚠[/] Batch {job_id} was already collected on {manifest['collected_at']}.")
        return 0
    try:
        job = wait_for_job(client, job_id, poll_seconds)
    except KeyboardInterrupt:
        console.print(f"\n[yellow]⚠[/] Stopped waiting. The job keeps running; "
                      f"collect it with --batch-job {job_id}")
        return 130
    except QuotaExhaustedError as exc:
        console.print(f"[red]✗ Quota exhausted[/] while polling: {exc}")
        return 1
    return collect_batch(client, manifest, job)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Structured OCR for scholarly PDFs via Mistral Document AI.",
    )
    parser.add_argument(
        "--item-id", type=int, action="append", dest="item_ids",
        help="Process only these item ids (matched against <id>.pdf). Repeatable.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N PDFs.")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process documents that already have results.",
    )
    parser.add_argument("--rpm", type=int, default=None, help="Requests per minute limit.")
    parser.add_argument(
        "--model", default=MISTRAL_OCR_MODEL,
        help=f"OCR model id (default: pinned {MISTRAL_OCR_MODEL}).",
    )
    parser.add_argument(
        "--pdf-dir", type=Path, default=PDF_DIR, help=f"Source directory (default: {PDF_DIR})."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--batch", action="store_true",
        help="Send everything as one Batch API job: half the price, answered within 24 h.",
    )
    mode.add_argument(
        "--batch-job", metavar="JOB_ID",
        help="Wait for and collect a batch job submitted earlier (its manifest is in OCR_Results/batch_jobs/).",
    )
    parser.add_argument(
        "--poll", type=int, default=60, help="Seconds between batch status checks (default: 60).",
    )
    args = parser.parse_args()
    configure_logging(log_file=LOG_DIR / "ocr_blocks.log", terminal=False)

    console.print(Panel(
        "[bold]Structured OCR for scholarly publications[/]\n"
        "Mistral Document AI with block extraction — body, footnotes and\n"
        "bibliography separated into a JSON sidecar beside the plain text.",
        title="📚 Publication Blocks Processor",
        border_style="cyan",
    ))
    console.print()

    load_dotenv()
    import os

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        console.print("[red]✗[/] MISTRAL_API_KEY not found in the environment!")
        return 1

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.batch_job:
        path = _manifest_path(args.batch_job)
        if not path.exists():
            console.print(f"[red]✗[/] No manifest for job {args.batch_job} in {BATCH_DIR}")
            return 1
        manifest = json.loads(path.read_text(encoding="utf-8"))
        client = MistralOcrClient(api_key, model=manifest["model"], logger=logging.getLogger(__name__))
        return run_batch(client, manifest, args.poll)

    pdf_files = sorted(args.pdf_dir.glob("*.pdf"))
    if args.item_ids:
        wanted = {str(i) for i in args.item_ids}
        pdf_files = [p for p in pdf_files if p.stem.split("_")[0] in wanted]
    if not args.force:
        context = {"pipeline": "publication-ocr-v2", "model_key": args.model, "model_id": args.model}
        pdf_files = [p for p in pdf_files if not artifact_matches(RESULTS_DIR / f"{p.stem}.txt", context, sha256_file(p))]
    if args.limit:
        pdf_files = pdf_files[: args.limit]

    console.print(key_value_table([
        ("Model", args.model),
        ("Source", str(args.pdf_dir)),
        ("Output", str(RESULTS_DIR)),
        ("Documents", len(pdf_files)),
        ("Mode", "Batch API (half price, ≤ 24 h)" if args.batch else "direct"),
        ("Rate limit", f"{args.rpm} rpm" if args.rpm else "none (paid tier)"),
    ]))
    console.print()

    if not pdf_files:
        console.print(
            "[yellow]⚠[/] Nothing to process. "
            "Run 01_fetch_publication_pdfs.py first, or pass --force to redo."
        )
        return 0

    client = MistralOcrClient(
        api_key, model=args.model, requests_per_minute=args.rpm,
        logger=logging.getLogger(__name__),
    )

    if not args.batch:
        return run_direct(client, pdf_files)

    try:
        manifest = submit_batch(client, pdf_files, args.pdf_dir)
    except QuotaExhaustedError as exc:
        console.print(f"[red]✗ Quota exhausted[/] — nothing was queued. {exc}")
        return 1
    return run_batch(client, manifest, args.poll)


if __name__ == "__main__":
    sys.exit(main())
