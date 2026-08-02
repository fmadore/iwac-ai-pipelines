"""French summary generation using the shared text-model registry.

Optimized for cost-effective document summarization with low reasoning effort.
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
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
    DEFAULT_TEXT_MODEL_KEY,
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

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
console = Console()

# Restricted to the cost-effective tiers — summarization does not need a flagship.
ALLOWED_MODEL_KEYS = TEXT_ECONOMY_MODELS
LEGACY_MODEL_KEYS = LEGACY_CLI_MODEL_KEYS

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
def generate_summary(llm_client: BaseLLMClient, text: str, system_prompt: str) -> Optional[str]:
    """Generate a summary using the configured LLM client."""
    if not text.strip():
        return None
    user_prompt = f"**Texte:**\n{text}"
    try:
        raw_output = llm_client.generate(system_prompt, user_prompt)
        if raw_output:
            return raw_output.strip().replace('*', '')
        logging.error("Model returned empty summary.")
        return None
    except Exception as exc:
        logging.error(f"Summary generation error: {exc}")
        return None

# ------------------------------------------------------------------
# File Processing
# ------------------------------------------------------------------
def process_txt_files(
    llm_client: BaseLLMClient,
    input_dir: str,
    output_dir: str,
    system_prompt: str,
    checkpoint: JsonCheckpoint,
) -> tuple[int, int, int]:
    """Process text files, resuming only exact input/provenance matches."""
    if not os.path.exists(input_dir):
        console.print(f"[red]✗[/red] Input directory not found: {input_dir}")
        return 0, 0, 0
    os.makedirs(output_dir, exist_ok=True)
    txt_files = sorted(f for f in os.listdir(input_dir) if f.endswith('.txt'))
    if not txt_files:
        console.print("[yellow]⚠[/yellow] No .txt files to process.")
        return 0, 0, 0
    
    console.print(f"\n[cyan]📁 Processing {len(txt_files)} files[/cyan]\n")
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    for fname in tqdm(txt_files, desc="Generating Summaries", 
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)
        try:
            with open(input_path, 'r', encoding='utf-8') as infile:
                original_text = infile.read()
            if not original_text.strip():
                tqdm.write(f"  [yellow]⚠[/yellow] Skipped (empty): {fname}")
                continue
            source_fingerprint = sha256_text(original_text)
            if checkpoint.matches(fname, source_fingerprint) and os.path.exists(output_path):
                skipped_count += 1
                continue
            summary = generate_summary(llm_client, original_text, system_prompt)
            if summary:
                atomic_write_text(Path(output_path), summary)
                checkpoint.mark(fname, source_fingerprint)
                success_count += 1
            else:
                tqdm.write(f"  [red]✗[/red] No summary: {fname}")
                error_count += 1
        except Exception as e:
            tqdm.write(f"  [red]✗[/red] Error processing {fname}: {e}")
            error_count += 1
    
    return success_count, error_count, skipped_count

def main():
    parser = argparse.ArgumentParser(description="Generate French summaries for extracted texts")
    parser.add_argument(
        "--model",
        choices=ALLOWED_MODEL_KEYS + LEGACY_MODEL_KEYS,
        default=DEFAULT_TEXT_MODEL_KEY,
        help=f"Model key (default: {DEFAULT_TEXT_MODEL_KEY})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Replace output/checkpoint even when model or prompt provenance differs",
    )
    args = parser.parse_args()

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(script_dir, 'TXT')
        output_dir = os.path.join(script_dir, 'Summaries_FR_TXT')

        system_prompt = split_prompt_template(load_prompt_template())
        
        # Display header
        console.print(Panel.fit(
            "[bold blue]📝 French Summary Generation Pipeline[/bold blue]",
            border_style="blue",
            box=box.DOUBLE
        ))
        console.print()
        
        # Get model selection (restricted to cost-effective models)
        model_option = get_model_option(args.model, allowed_keys=ALLOWED_MODEL_KEYS)

        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = Path(output_dir) / ".summary_checkpoint.json"
        existing_outputs = list(Path(output_dir).glob("*.txt"))
        if existing_outputs and not checkpoint_path.exists() and not args.force:
            raise CheckpointMismatch(
                f"Existing summaries have no provenance checkpoint: {output_dir}. "
                "Use --force to replace them."
            )
        checkpoint = JsonCheckpoint.open(
            checkpoint_path,
            {
                "pipeline": "french-summary-v2",
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
        config_table.add_row("Output Directory", output_dir)
        config_table.add_row("Reasoning Effort", config.reasoning_effort or "default")
        config_table.add_row("Text Verbosity", config.text_verbosity or "default")
        config_table.add_row("Thinking Level", config.thinking_level or "default")
        console.print(config_table)
        
        llm_client = build_llm_client(model_option, config=config)
        success_count, error_count, skipped_count = process_txt_files(
            llm_client, input_dir, output_dir, system_prompt, checkpoint
        )
        
        # Display results
        console.print()
        if error_count == 0 and success_count > 0:
            console.print(Panel.fit(
                f"[bold green]✓ Completed successfully![/bold green]\n\n"
                f"[green]📄 {success_count} files summarized[/green]\n"
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
