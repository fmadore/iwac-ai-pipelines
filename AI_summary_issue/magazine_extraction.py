"""Shared building blocks for the magazine article extraction pipeline.

Everything the Gemini (02_AI_generate_summaries_issue.py) and Mistral
(02_Mistral_generate_summaries_issue.py) scripts have in common lives here:

- Pydantic models for the structured outputs (PageArticle, PageExtraction,
  ConsolidatedArticle, MagazineIndex)
- Prompt loading helpers
- Markdown formatting of intermediate and final results
- The step-1 page loop (page_*.json cache check, per-page JSON/MD saves,
  summary table, consolidated JSON/MD writes), parameterized by a
  provider-supplied ``extract_page(page_num) -> PageExtraction`` callable
- The step-2 consolidation save logic (prompt splitting on
  '{extracted_content}', JSON/MD saves), parameterized by a provider-supplied
  ``consolidate(system_prompt, extracted_json) -> MagazineIndex`` callable
- Cache lifecycle: the per-page ``page_*.json`` cache is deleted only AFTER
  step 2 has successfully written the final index, so a failed consolidation
  never forces re-paying the per-page API cost on the next run
- Error accounting: pages that fail extraction become explicit placeholders;
  when more than ``ERROR_RATE_THRESHOLD`` of pages failed, step 2 is skipped
  entirely (cache kept) instead of silently polluting the final TOC

Provider SDKs are deliberately NOT imported here — each 02_ script keeps its
own client initialization and generate calls. pypdf is imported lazily so this
module can be imported without any provider SDK installed.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from pydantic import BaseModel, Field

# Rich console for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich import box

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from common.rate_limiter import QuotaExhaustedError  # noqa: E402

# Shared console — the provider scripts import this so the whole pipeline
# writes through a single rich console.
console = Console()

# Step 2 is skipped (and the page cache kept) when more than this fraction of
# pages failed extraction in step 1.
ERROR_RATE_THRESHOLD = 0.20


class TooManyExtractionErrors(RuntimeError):
    """Raised when step 1 produced too many failed pages to trust consolidation."""


# ------------------------------------------------------------------
# Pydantic Models for Structured Outputs
# ------------------------------------------------------------------

class PageArticle(BaseModel):
    """An article found on a single page."""
    titre: str = Field(description="Titre exact de l'article tel qu'imprimé sur la page")
    auteurs: Optional[List[str]] = Field(default=None, description="Liste des auteurs de l'article (ex: ['Jean Dupont', 'La Rédaction']). Null si non mentionné.")
    continuation: Optional[str] = Field(default=None, description="Indication de continuation: 'suite page X' ou null si aucune")
    resume: str = Field(description="Résumé bref de 2-3 phrases du contenu visible sur cette page")


class PageExtraction(BaseModel):
    """Extraction result for a single PDF page."""
    page_number: int = Field(description="Numéro de la page analysée")
    is_cover: bool = Field(default=False, description="True si c'est une page de couverture")
    has_articles: bool = Field(default=True, description="True si des articles sont présents")
    articles: List[PageArticle] = Field(default_factory=list, description="Liste des articles trouvés sur cette page")
    other_content: Optional[str] = Field(default=None, description="Description des autres contenus non-articles (publicités, annonces, etc.)")


class ConsolidatedArticle(BaseModel):
    """A consolidated article after merging fragmented pages."""
    titre: str = Field(description="Titre exact complet de l'article")
    auteurs: Optional[List[str]] = Field(default=None, description="Liste des auteurs de l'article. Null si non mentionné.")
    pages: str = Field(description="Numéros de pages, ex: '1-3' ou '1, 3, 5'")
    resume: str = Field(description="Résumé global consolidé de 4-6 phrases")


class MagazineIndex(BaseModel):
    """Final consolidated index of all articles in the magazine."""
    articles: List[ConsolidatedArticle] = Field(description="Liste des articles consolidés du magazine")


# ------------------------------------------------------------------
# Prompt Loading
# ------------------------------------------------------------------

def load_extraction_prompt() -> str:
    """Load the prompt for step 1 (page-by-page extraction).

    Note: With structured outputs, the prompt focuses on instructions
    while the schema defines the output format.
    """
    prompt_file = Path(SCRIPT_DIR) / 'summary_prompt_issue.md'
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt template not found: {prompt_file}") from None
    except Exception as e:
        raise RuntimeError(f"Failed to read prompt template {prompt_file}: {e}") from e


def load_consolidation_prompt() -> str:
    """Load the prompt for step 2 (consolidation).

    Note: With structured outputs, the prompt focuses on instructions
    while the schema defines the output format.
    """
    prompt_file = Path(SCRIPT_DIR) / 'consolidation_prompt_issue.md'
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Consolidation prompt template not found: {prompt_file}") from None
    except Exception as e:
        raise RuntimeError(f"Failed to read consolidation prompt template {prompt_file}: {e}") from e


def build_consolidation_system_prompt() -> str:
    """Build the step-2 system prompt from the consolidation template.

    Strips the '{extracted_content}' placeholder section (the JSON is passed
    as the user message instead) and appends the JSON input/output note.
    """
    consolidation_prompt = load_consolidation_prompt()
    system_prompt = consolidation_prompt.split('{extracted_content}')[0].strip()
    system_prompt += (
        "\n\nL'entrée est fournie au format JSON structuré. Consolidez les articles "
        "et retournez le résultat au format JSON selon le schéma fourni."
    )
    return system_prompt


# ------------------------------------------------------------------
# Markdown Formatting
# ------------------------------------------------------------------

def format_extraction_to_markdown(extraction: PageExtraction) -> str:
    """Convert a PageExtraction to markdown format for intermediate files."""
    lines = [f"## Page : {extraction.page_number}\n"]

    if extraction.is_cover:
        lines.append("Page de couverture du magazine.\n")
        return "\n".join(lines)

    if not extraction.has_articles:
        lines.append("Aucun article identifié sur cette page.\n")
        if extraction.other_content:
            lines.append(f"\n### Autres contenus\n{extraction.other_content}\n")
        return "\n".join(lines)

    for i, article in enumerate(extraction.articles, 1):
        lines.append(f"\n### Article {i}")
        lines.append(f"- Titre : {article.titre}")
        if article.auteurs:
            lines.append(f"- Auteur(s) : {', '.join(article.auteurs)}")
        lines.append(f"- Résumé :\n  {article.resume}")

    if extraction.other_content:
        lines.append(f"\n### Autres contenus\n{extraction.other_content}")

    return "\n".join(lines)


def format_index_to_markdown(index: MagazineIndex) -> str:
    """Convert a MagazineIndex to the final markdown format."""
    lines = ["# Index des articles du magazine\n"]

    for article in index.articles:
        lines.append(f"\n## {article.titre}")
        if article.auteurs:
            lines.append(f"- Auteur(s) : {', '.join(article.auteurs)}")
        lines.append(f"- Pages : {article.pages}")
        lines.append("- Résumé :")
        lines.append(f"  {article.resume}")

    return "\n".join(lines)


# ------------------------------------------------------------------
# PDF Helpers
# ------------------------------------------------------------------

class PdfPageSource:
    """Serve single-page PDF bytes from a document parsed once.

    Local optimization over calling ``common.pdf_utils.extract_pdf_page`` in a
    loop, which re-parses the whole source PDF for every page. The per-page
    output is identical: a minimal single-page PDF document as bytes.

    pypdf is imported lazily so this module stays importable without it.
    """

    def __init__(self, pdf_path: Path) -> None:
        from pypdf import PdfReader  # lazy: keep module importable without pypdf

        self.pdf_path = pdf_path
        self._reader = PdfReader(str(pdf_path))

    def __len__(self) -> int:
        return len(self._reader.pages)

    def page_bytes(self, page_index: int) -> bytes:
        """Extract a single page as bytes (0-indexed)."""
        import io

        from pypdf import PdfWriter

        try:
            writer = PdfWriter()
            writer.add_page(self._reader.pages[page_index])
            buf = io.BytesIO()
            writer.write(buf)
            return buf.getvalue()
        except Exception as exc:
            logging.error(f"Error extracting page {page_index + 1} from {self.pdf_path}: {exc}")
            raise


# ------------------------------------------------------------------
# User Interaction
# ------------------------------------------------------------------

def get_input_pdfs(script_dir: Path) -> List[Path]:
    """Get the list of all PDFs to process."""
    # Use the default PDF folder
    default_pdf_dir = script_dir / "PDF"

    if not default_pdf_dir.exists():
        console.print(f"[red]✗[/] PDF folder does not exist: {default_pdf_dir}")
        raise FileNotFoundError(f"PDF folder does not exist: {default_pdf_dir}")

    pdf_files = sorted(default_pdf_dir.glob('*.pdf'))

    if not pdf_files:
        console.print(f"[red]✗[/] No PDF files found in {default_pdf_dir}")
        raise FileNotFoundError(f"No PDF files found in {default_pdf_dir}")

    logging.info(f"{len(pdf_files)} PDF file(s) found in {default_pdf_dir}")
    return pdf_files


# ------------------------------------------------------------------
# Step-1 Cache
# ------------------------------------------------------------------

def step1_cache_dir(output_dir: Path) -> Path:
    """Directory holding the per-page ``page_*.json`` / ``page_*.md`` cache."""
    return output_dir / "step1_page_extractions"


def cleanup_step1_cache(output_dir: Path) -> None:
    """Delete the per-page cache files.

    Only call this AFTER step 2 has successfully written the final index —
    the cache is what makes a re-run after a step-2 failure free.
    """
    step1_dir = step1_cache_dir(output_dir)
    if not step1_dir.exists():
        return
    for f in step1_dir.glob('page_*.*'):
        f.unlink()
    try:
        step1_dir.rmdir()
    except OSError:
        pass


# ------------------------------------------------------------------
# Pipeline Steps
# ------------------------------------------------------------------

def run_step1(
    extract_page: Callable[[int], Optional[PageExtraction]],
    *,
    total_pages: int,
    output_dir: Path,
    magazine_id: str,
    model_label: str,
    schema_note: str,
) -> Tuple[Path, List[PageExtraction], int]:
    """Step 1: page-by-page extraction, provider-agnostic skeleton.

    Handles the JSON cache check, per-page JSON/MD saves, progress bar,
    summary table, and consolidated JSON/MD writes. The actual API call is
    delegated to *extract_page*, which receives the 1-indexed page number and
    returns a :class:`PageExtraction` (or raises).

    Pages that fail become explicit error placeholders and are counted; the
    per-page cache is NOT deleted here (see :func:`cleanup_step1_cache`).
    A :class:`QuotaExhaustedError` from *extract_page* propagates immediately
    so the caller can stop the whole batch — already-processed pages stay
    cached for the next run.

    Returns:
        Tuple of (consolidated JSON file path, list of PageExtraction
        objects, number of pages that failed extraction).
    """
    step1_dir = step1_cache_dir(output_dir)
    step1_dir.mkdir(parents=True, exist_ok=True)

    all_extractions: List[PageExtraction] = []
    all_markdown: List[str] = []

    console.print(f"\n[bold cyan]Step 1:[/] Processing {total_pages} pages with [green]{model_label}[/]")
    console.print(f"[dim]Using structured outputs ({schema_note})[/]")
    logging.info(f"Step 1: Processing {total_pages} pages with {model_label} (structured output)...")

    success_count = 0
    cached_count = 0
    error_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False
    ) as progress:
        task = progress.add_task("[cyan]Extracting articles...", total=total_pages)

        for page_num in range(1, total_pages + 1):
            page_json_file = step1_dir / f"page_{page_num:03d}.json"
            page_md_file = step1_dir / f"page_{page_num:03d}.md"

            # Check if page has already been processed (JSON cache)
            if page_json_file.exists():
                logging.info(f"Page {page_num} already processed, loading from cache...")
                try:
                    with open(page_json_file, 'r', encoding='utf-8') as f:
                        extraction = PageExtraction.model_validate_json(f.read())
                    all_extractions.append(extraction)
                    all_markdown.append(format_extraction_to_markdown(extraction))
                    cached_count += 1
                    progress.update(task, advance=1, description=f"[dim]Page {page_num}/{total_pages} (cached)[/]")
                    continue
                except Exception as e:
                    logging.warning(f"Failed to load cached page {page_num}, re-processing: {e}")

            progress.update(task, description=f"[cyan]Page {page_num}/{total_pages}[/]")

            # Generate structured extraction via the provider callable
            try:
                extraction = extract_page(page_num)

                if extraction:
                    all_extractions.append(extraction)
                    markdown = format_extraction_to_markdown(extraction)
                    all_markdown.append(markdown)

                    # Save both JSON and markdown for debugging/resumption
                    with open(page_json_file, 'w', encoding='utf-8') as f:
                        f.write(extraction.model_dump_json(indent=2, ensure_ascii=False))
                    with open(page_md_file, 'w', encoding='utf-8') as f:
                        f.write(markdown)

                    logging.info(f"Page {page_num} processed and saved")
                    success_count += 1
                else:
                    logging.error(f"No extraction generated for page {page_num}")
                    placeholder = PageExtraction(
                        page_number=page_num,
                        has_articles=False,
                        other_content="Error processing this page."
                    )
                    all_extractions.append(placeholder)
                    all_markdown.append(format_extraction_to_markdown(placeholder))
                    error_count += 1

            except QuotaExhaustedError:
                console.print(
                    f"[red]✗ Quota exhausted at page {page_num}/{total_pages}[/] — stopping. "
                    f"Processed pages remain cached in [dim]{step1_dir}[/] for resumption."
                )
                logging.error(f"Quota exhausted during step 1 at page {page_num}/{total_pages} of {magazine_id}.")
                raise

            except Exception as e:
                logging.error(f"Failed to extract or process page {page_num} after retries: {e}")
                placeholder = PageExtraction(
                    page_number=page_num,
                    has_articles=False,
                    other_content=f"Error: {str(e)}"
                )
                all_extractions.append(placeholder)
                all_markdown.append(format_extraction_to_markdown(placeholder))
                error_count += 1

            progress.update(task, advance=1)

    # Display step 1 summary
    step1_summary = Table(box=box.SIMPLE, show_header=False)
    step1_summary.add_column("Status", style="bold")
    step1_summary.add_column("Count", justify="right")
    step1_summary.add_row("[green]✓ Processed[/]", str(success_count))
    if cached_count > 0:
        step1_summary.add_row("[dim]↺ Cached[/]", str(cached_count))
    if error_count > 0:
        step1_summary.add_row("[red]✗ Errors[/]", str(error_count))
    console.print(step1_summary)

    # Save consolidated JSON (for step 2)
    consolidated_json_file = output_dir / f"{magazine_id}_step1_consolidated.json"
    with open(consolidated_json_file, 'w', encoding='utf-8') as f:
        json.dump([e.model_dump() for e in all_extractions], f, ensure_ascii=False, indent=2)

    # Save consolidated markdown (for human review)
    consolidated_md_file = output_dir / f"{magazine_id}_step1_consolidated.md"
    with open(consolidated_md_file, 'w', encoding='utf-8') as f:
        f.write(f"# Page-by-page extraction - Magazine {magazine_id}\n\n")
        f.write('\n---\n'.join(all_markdown))

    console.print(f"[green]✓[/] Step 1 complete → [dim]{consolidated_json_file.name}[/]")
    logging.info(f"Step 1 complete. Consolidated files: {consolidated_json_file}, {consolidated_md_file}")

    return consolidated_json_file, all_extractions, error_count


def run_step2(
    consolidate: Callable[[str, str], Optional[MagazineIndex]],
    *,
    extractions: List[PageExtraction],
    output_dir: Path,
    magazine_id: str,
    model_label: str,
    schema_note: str,
    extraction_errors: int = 0,
) -> Path:
    """Step 2: magazine-level consolidation, provider-agnostic skeleton.

    Builds the system prompt (splitting the template on '{extracted_content}')
    and the extracted-content JSON, delegates the LLM call to *consolidate*,
    then saves the final index as JSON and markdown. When *extraction_errors*
    is non-zero, an ``"extraction_errors"`` count is included in the final
    index JSON so downstream consumers can see the TOC is incomplete.

    Returns:
        Path to the final markdown index file.
    """
    console.print(f"\n[bold cyan]Step 2:[/] Consolidating articles with [green]{model_label}[/]")
    console.print(f"[dim]Using structured outputs ({schema_note})[/]")
    logging.info(f"Step 2: Consolidating articles at magazine level with {model_label} (structured output)...")

    # Prepare extracted content as JSON for the model
    extracted_json = json.dumps([e.model_dump() for e in extractions], ensure_ascii=False, indent=2)

    # Load the consolidation prompt as system instruction
    system_prompt = build_consolidation_system_prompt()

    # Generate consolidation via the provider callable (retries inside)
    try:
        with console.status("[cyan]Generating article index...", spinner="dots"):
            index = consolidate(system_prompt, extracted_json)

        if not index:
            console.print("[red]✗[/] Failed to generate consolidated output")
            logging.error("Failed to generate consolidated output")
            raise RuntimeError("Step 2 consolidation failed - no output generated")

        # Save the final result as JSON
        final_json_file = output_dir / f"{magazine_id}_final_index.json"
        with open(final_json_file, 'w', encoding='utf-8') as f:
            if extraction_errors:
                payload = index.model_dump()
                payload["extraction_errors"] = extraction_errors
                f.write(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                f.write(index.model_dump_json(indent=2, ensure_ascii=False))

        # Save the final result as markdown (for human readability)
        final_md_file = output_dir / f"{magazine_id}_final_index.md"
        with open(final_md_file, 'w', encoding='utf-8') as f:
            f.write(format_index_to_markdown(index))

        console.print(f"[green]✓[/] Step 2 complete → [dim]{final_md_file.name}[/]")
        logging.info(f"Step 2 complete. Final index: {final_md_file}")
        return final_md_file

    except Exception as e:
        console.print(f"[red]✗[/] Step 2 failed: {e}")
        logging.error(f"Step 2 failed after retries: {e}")
        raise


def run_extraction_pipeline(
    *,
    extract_page: Callable[[int], Optional[PageExtraction]],
    consolidate: Callable[[str, str], Optional[MagazineIndex]],
    total_pages: int,
    output_dir: Path,
    magazine_id: str,
    step1_model_label: str,
    step2_model_label: str,
    schema_note: str,
) -> Path:
    """Run both steps for one magazine and manage the cache lifecycle.

    - Runs step 1 (page loop) and step 2 (consolidation) with the
      provider-supplied callables.
    - If more than ``ERROR_RATE_THRESHOLD`` of pages failed in step 1, step 2
      is skipped entirely and :class:`TooManyExtractionErrors` is raised; the
      per-page cache stays intact so a re-run only retries the failed pages.
    - The per-page cache is deleted only AFTER step 2 has successfully
      written the final index.

    Returns:
        Path to the final markdown index file.
    """
    _step1_file, extractions, error_count = run_step1(
        extract_page,
        total_pages=total_pages,
        output_dir=output_dir,
        magazine_id=magazine_id,
        model_label=step1_model_label,
        schema_note=schema_note,
    )

    if error_count:
        error_rate = error_count / total_pages if total_pages else 1.0
        if error_rate > ERROR_RATE_THRESHOLD:
            console.print(Panel(
                f"[red bold]{error_count}/{total_pages} pages failed extraction ({error_rate:.0%}).[/]\n"
                "Step 2 (consolidation) skipped — the resulting index would be unreliable.\n"
                f"The per-page cache in [bold]{step1_cache_dir(output_dir)}[/] has been kept:\n"
                "fix the underlying issue and re-run to retry only the failed pages.",
                title="✗ Too many extraction errors",
                border_style="red",
            ))
            logging.error(
                f"Magazine {magazine_id}: {error_count}/{total_pages} pages failed extraction "
                f"({error_rate:.0%} > {ERROR_RATE_THRESHOLD:.0%}) — step 2 skipped, cache kept."
            )
            raise TooManyExtractionErrors(
                f"{error_count}/{total_pages} pages failed extraction — consolidation skipped"
            )
        console.print(
            f"[yellow]⚠ {error_count}/{total_pages} pages failed extraction — "
            f'the final index will be flagged with "extraction_errors": {error_count}.[/]'
        )
        logging.warning(f"Magazine {magazine_id}: {error_count}/{total_pages} pages failed extraction.")

    final_file = run_step2(
        consolidate,
        extractions=extractions,
        output_dir=output_dir,
        magazine_id=magazine_id,
        model_label=step2_model_label,
        schema_note=schema_note,
        extraction_errors=error_count,
    )

    # Success: the final index exists — the per-page cache is no longer needed.
    cleanup_step1_cache(output_dir)

    return final_file
