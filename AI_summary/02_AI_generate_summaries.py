"""Bilingual (French + English) summary generation using the shared text-model registry.

Each document is read once and rendered twice: ``summary_fr`` and ``summary_en``
report the same facts, each idiomatic in its own language. Step 03 writes them to
``bibo:shortDescription`` as two ``@language``-tagged literals on one property.

Optimized for cost-effective document summarization with low reasoning effort.
"""

import argparse
import os
import sys
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.llm_provider import (  # noqa: E402
    LEGACY_CLI_MODEL_KEYS,
    TEXT_ECONOMY_MODELS,
    BaseLLMClient,
    LLMConfig,
    build_llm_client,
    get_model_option,
    summary_from_option,
)
from common.checkpoint import (  # noqa: E402
    CheckpointMismatch,
    JsonCheckpoint,
    atomic_write_text,
    sha256_text,
)
from common.log_redaction import install_credential_redaction

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Credentials ride in Omeka query strings and provider headers; keep them
# out of anything urllib3 or an SDK decides to log.
install_credential_redaction()
load_dotenv()
console = Console()

# Restricted to the cost-effective tiers — summarization does not need a flagship.
ALLOWED_MODEL_KEYS = TEXT_ECONOMY_MODELS
LEGACY_MODEL_KEYS = LEGACY_CLI_MODEL_KEYS

#: This pipeline pins its own default rather than taking the shared
#: ``DEFAULT_TEXT_MODEL_KEY``. Both are in ``TEXT_ECONOMY_MODELS``; Luna is
#: chosen for throughput — the measured full-corpus sentiment pass ran 2.7 h on
#: Luna against 31.5 h on DeepSeek V4 Flash 0731, which has no middle reasoning
#: level and rounds up to ``high``. Override with ``--model``.
DEFAULT_MODEL_KEY = "gpt-5.6-luna"

#: Written as ``@language`` on the two Omeka literals in step 03.
FRENCH_DIR = "Summaries_FR_TXT"
ENGLISH_DIR = "Summaries_EN_TXT"

#: Concurrent model calls. Summarization is one independent request per file, so
#: this is pure wall-clock: a serial corpus pass is ~4 s x 12,305 = ~14 hours.
#: Kept modest because the ceiling is the provider's rate limit, not the CPU —
#: raise with --workers if your account's limits allow.
DEFAULT_WORKERS = 6


class BilingualSummary(BaseModel):
    """The two renderings of one reading of a document."""

    summary_fr: str = Field(description="Résumé en français, quelques phrases concises.")
    summary_en: str = Field(description="The same summary in English, idiomatic.")

# ------------------------------------------------------------------
# Prompt Loading
# ------------------------------------------------------------------
def load_prompt_template() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(script_dir, 'summary_prompt.md')
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        if '{text}' not in content:
            logging.warning("Prompt template missing '{text}' placeholder.")
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt template not found: {prompt_file}") from None
    except Exception as e:
        raise RuntimeError(f"Failed to read prompt template {prompt_file}: {e}") from e

def split_prompt_template(template: str) -> str:
    """Return the instruction portion of the template for the system prompt.

    The template ends with a '**Texte:** {text}' block that belongs in the
    user message; sending the whole template as the system prompt duplicated
    every instruction on every request.
    """
    instructions = template.split('{text}')[0]
    instructions = instructions.rstrip()
    for suffix in ('**Texte:**', '---'):
        if instructions.endswith(suffix):
            instructions = instructions[: -len(suffix)].rstrip()
    return instructions


# ------------------------------------------------------------------
# Generation Helper
# ------------------------------------------------------------------
def _clean(summary: str) -> str:
    """Strip the stray markdown emphasis models emit despite the prompt."""
    return summary.strip().replace('*', '')


def generate_summary(
    llm_client: BaseLLMClient, text: str, system_prompt: str
) -> Optional[Tuple[str, str]]:
    """Generate the French and English summaries of *text*.

    Returns ``(french, english)``, or ``None`` when either is missing — a pair
    with one empty half is a failed generation, not a partial success, because
    step 03 would then write a single-language value and mark the item done.
    """
    if not text.strip():
        return None
    user_prompt = f"**Texte:**\n{text}"
    try:
        result = llm_client.generate_structured(system_prompt, user_prompt, BilingualSummary)
    except Exception as exc:
        logging.error(f"Summary generation error: {exc}")
        return None

    french, english = _clean(result.summary_fr), _clean(result.summary_en)
    if not french or not english:
        missing = "French" if not french else "English"
        logging.error(f"Model returned no {missing} summary.")
        return None
    return french, english

# ------------------------------------------------------------------
# File Processing
# ------------------------------------------------------------------
def process_txt_files(
    llm_client: BaseLLMClient,
    input_dir: str,
    french_dir: str,
    english_dir: str,
    system_prompt: str,
    checkpoint: JsonCheckpoint,
    workers: int = 1,
) -> tuple[int, int, int]:
    """Process text files, resuming only exact input/provenance matches.

    Files are independent, so *workers* > 1 simply overlaps the model calls.
    The checkpoint is guarded by a lock: ``mark()`` rewrites the whole manifest,
    and two threads doing that at once would interleave into a corrupt file.
    """
    if not os.path.exists(input_dir):
        console.print(f"[red]✗[/red] Input directory not found: {input_dir}")
        return 0, 0, 0
    os.makedirs(french_dir, exist_ok=True)
    os.makedirs(english_dir, exist_ok=True)
    txt_files = sorted(f for f in os.listdir(input_dir) if f.endswith('.txt'))
    if not txt_files:
        console.print("[yellow]⚠[/yellow] No .txt files to process.")
        return 0, 0, 0

    console.print(
        f"\n[cyan]📁 Processing {len(txt_files)} files[/cyan]"
        f"{f' [dim]({workers} workers)[/dim]' if workers > 1 else ''}\n"
    )

    checkpoint_lock = threading.Lock()

    def handle(fname: str) -> str:
        """Return 'success', 'error' or 'skipped' for one file."""
        input_path = os.path.join(input_dir, fname)
        french_path = os.path.join(french_dir, fname)
        english_path = os.path.join(english_dir, fname)

        with open(input_path, 'r', encoding='utf-8') as infile:
            original_text = infile.read()
        if not original_text.strip():
            tqdm.write(f"  [yellow]⚠[/yellow] Skipped (empty): {fname}")
            return "skipped_empty"

        source_fingerprint = sha256_text(original_text)
        # Both halves must be on disk: a run interrupted between the two writes
        # must regenerate, not resume with a missing translation.
        with checkpoint_lock:
            resumable = checkpoint.matches(fname, source_fingerprint)
        if resumable and os.path.exists(french_path) and os.path.exists(english_path):
            return "skipped"

        summary = generate_summary(llm_client, original_text, system_prompt)
        if not summary:
            tqdm.write(f"  [red]✗[/red] No summary: {fname}")
            return "error"

        french, english = summary
        atomic_write_text(Path(french_path), french)
        atomic_write_text(Path(english_path), english)
        with checkpoint_lock:
            checkpoint.mark(fname, source_fingerprint)
        return "success"

    success_count = 0
    error_count = 0
    skipped_count = 0
    bar = dict(desc="Generating Summaries",
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    def tally(fname: str, outcome: str) -> None:
        nonlocal success_count, error_count, skipped_count
        if outcome == "success":
            success_count += 1
        elif outcome == "skipped":
            skipped_count += 1
        elif outcome == "error":
            error_count += 1

    if workers <= 1:
        for fname in tqdm(txt_files, **bar):
            try:
                tally(fname, handle(fname))
            except Exception as e:
                tqdm.write(f"  [red]✗[/red] Error processing {fname}: {e}")
                error_count += 1
        return success_count, error_count, skipped_count

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(handle, f): f for f in txt_files}
        for future in tqdm(as_completed(futures), total=len(futures), **bar):
            fname = futures[future]
            try:
                tally(fname, future.result())
            except Exception as e:
                tqdm.write(f"  [red]✗[/red] Error processing {fname}: {e}")
                error_count += 1

    return success_count, error_count, skipped_count

def main():
    parser = argparse.ArgumentParser(
        description="Generate French and English summaries for extracted texts"
    )
    parser.add_argument(
        "--model",
        choices=ALLOWED_MODEL_KEYS + LEGACY_MODEL_KEYS,
        default=DEFAULT_MODEL_KEY,
        help=f"Model key (default: {DEFAULT_MODEL_KEY})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Replace output/checkpoint even when model or prompt provenance differs",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Concurrent model calls (default: {DEFAULT_WORKERS}; 1 = serial)",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(script_dir, 'TXT')
        french_dir = os.path.join(script_dir, FRENCH_DIR)
        english_dir = os.path.join(script_dir, ENGLISH_DIR)

        system_prompt = split_prompt_template(load_prompt_template())

        # Display header
        console.print(Panel.fit(
            "[bold blue]📝 Bilingual Summary Generation Pipeline[/bold blue]",
            border_style="blue",
            box=box.DOUBLE
        ))
        console.print()

        # Get model selection (restricted to cost-effective models)
        model_option = get_model_option(args.model, allowed_keys=ALLOWED_MODEL_KEYS)

        os.makedirs(french_dir, exist_ok=True)
        os.makedirs(english_dir, exist_ok=True)
        checkpoint_path = Path(french_dir) / ".summary_checkpoint.json"
        existing_outputs = (
            list(Path(french_dir).glob("*.txt")) + list(Path(english_dir).glob("*.txt"))
        )
        if existing_outputs and not checkpoint_path.exists() and not args.force:
            raise CheckpointMismatch(
                f"Existing summaries have no provenance checkpoint: {french_dir}. "
                "Use --force to replace them."
            )
        checkpoint = JsonCheckpoint.open(
            checkpoint_path,
            {
                # Bumped from "french-summary-v2": a monolingual checkpoint must
                # not resume a bilingual run, or every item it covers would keep
                # its French file and never get an English one.
                "pipeline": "bilingual-summary-v3",
                "model_key": model_option.key,
                "model_id": model_option.model,
                "prompt_sha256": sha256_text(system_prompt),
            },
            reset=args.force,
        )

        # Configure for cost-effective summarization
        config = LLMConfig(
            reasoning_effort="low",      # OpenAI: quick summarization
            text_verbosity="low",        # OpenAI: concise output
            thinking_level="MINIMAL",    # Gemini Flash: minimal thinking for speed
            # No temperature: MODEL_REGISTRY holds each vendor's recommendation.
        )

        # Display configuration table
        config_table = Table(title="⚙️  Configuration", box=box.ROUNDED, show_header=False)
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")
        config_table.add_row("AI Model", summary_from_option(model_option))
        config_table.add_row("Input Directory", input_dir)
        config_table.add_row("Output (French)", french_dir)
        config_table.add_row("Output (English)", english_dir)
        config_table.add_row("Reasoning Effort", config.reasoning_effort or "default")
        config_table.add_row("Text Verbosity", config.text_verbosity or "default")
        config_table.add_row("Thinking Level", config.thinking_level or "default")
        config_table.add_row("Workers", str(args.workers))
        console.print(config_table)

        llm_client = build_llm_client(model_option, config=config)
        success_count, error_count, skipped_count = process_txt_files(
            llm_client, input_dir, french_dir, english_dir, system_prompt, checkpoint,
            workers=args.workers,
        )
        
        # Display results
        console.print()
        if error_count == 0 and success_count > 0:
            console.print(Panel.fit(
                f"[bold green]✓ Completed successfully![/bold green]\n\n"
                f"[green]📄 {success_count} documents summarized (FR + EN)[/green]\n"
                f"[dim]↷ {skipped_count} resumed from checkpoint[/dim]",
                border_style="green",
                box=box.ROUNDED
            ))
        elif success_count > 0:
            console.print(Panel.fit(
                f"[bold yellow]⚠ Completed with warnings[/bold yellow]\n\n"
                f"[green]✓ {success_count} files summarized[/green]\n"
                f"[dim]↷ {skipped_count} resumed from checkpoint[/dim]\n"
                f"[red]✗ {error_count} files failed[/red]",
                border_style="yellow",
                box=box.ROUNDED
            ))
        elif skipped_count:
            console.print(Panel.fit(
                f"[bold green]✓ Already complete[/bold green]\n\n"
                f"[dim]↷ {skipped_count} files matched the checkpoint[/dim]",
                border_style="green",
                box=box.ROUNDED,
            ))
        else:
            console.print(Panel.fit(
                "[bold red]✗ No files processed[/bold red]",
                border_style="red",
                box=box.ROUNDED
            ))
            
    except (FileNotFoundError, ValueError) as err:
        console.print(f"\n[red]✗ Error:[/red] {err}")
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
    except Exception as exc:
        console.print(f"\n[red]✗ Unexpected failure:[/red] {exc}")
        console.print_exception()

if __name__ == '__main__':
    main()
