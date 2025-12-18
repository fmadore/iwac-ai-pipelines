"""French Summary Generation using OpenAI GPT-5 mini or Gemini 3 Flash.

Optimized for cost-effective document summarization with low reasoning effort.
"""

import os
import sys
import logging
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
    sys.path.append(REPO_ROOT)

from common.llm_provider import (  # noqa: E402
    BaseLLMClient,
    LLMConfig,
    build_llm_client,
    get_model_option,
    summary_from_option,
)

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
console = Console()

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
        raise FileNotFoundError(f"Prompt template not found: {prompt_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to read prompt template {prompt_file}: {e}")

PROMPT_TEMPLATE = load_prompt_template()

# ------------------------------------------------------------------
# Generation Helper
# ------------------------------------------------------------------
def generate_summary(llm_client: BaseLLMClient, text: str) -> Optional[str]:
    """Generate a summary using the configured LLM client."""
    if not text.strip():
        return None
    system_prompt = PROMPT_TEMPLATE
    user_prompt = system_prompt.format(text=text)
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
def process_file(llm_client: BaseLLMClient, input_file_path: str, output_file_path: str):
    """Process a single text file and generate its summary."""
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            original_text = infile.read()
        if not original_text.strip():
            logging.warning(f"Empty file skipped: {input_file_path}")
            return
        logging.info(f"Summarizing: {os.path.basename(input_file_path)}")
        summary = generate_summary(llm_client, original_text)
        if summary:
            with open(output_file_path, 'w', encoding='utf-8') as out:
                out.write(summary)
            logging.info(f"Saved: {os.path.basename(output_file_path)}")
        else:
            logging.error(f"No summary produced for {input_file_path}")
    except FileNotFoundError:
        logging.error(f"File not found: {input_file_path}")
    except IOError as e:
        logging.error(f"IO error {input_file_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error {input_file_path}: {e}")

def process_txt_files(llm_client: BaseLLMClient, input_dir: str, output_dir: str) -> tuple[int, int]:
    """Process all text files in input directory. Returns (success_count, error_count)."""
    if not os.path.exists(input_dir):
        console.print(f"[red]✗[/red] Input directory not found: {input_dir}")
        return 0, 0
    os.makedirs(output_dir, exist_ok=True)
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    if not txt_files:
        console.print("[yellow]⚠[/yellow] No .txt files to process.")
        return 0, 0
    
    console.print(f"\n[cyan]📁 Processing {len(txt_files)} files[/cyan]\n")
    
    success_count = 0
    error_count = 0
    
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
            summary = generate_summary(llm_client, original_text)
            if summary:
                with open(output_path, 'w', encoding='utf-8') as out:
                    out.write(summary)
                success_count += 1
            else:
                tqdm.write(f"  [red]✗[/red] No summary: {fname}")
                error_count += 1
        except Exception as e:
            tqdm.write(f"  [red]✗[/red] Error processing {fname}: {e}")
            error_count += 1
    
    return success_count, error_count

def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(script_dir, 'TXT')
        output_dir = os.path.join(script_dir, 'Summaries_FR_TXT')
        
        # Display header
        console.print(Panel.fit(
            "[bold blue]📝 French Summary Generation Pipeline[/bold blue]",
            border_style="blue",
            box=box.DOUBLE
        ))
        console.print()
        
        # Get model selection (restricted to cost-effective models)
        model_option = get_model_option(None, allowed_keys=["gpt-5-mini", "gemini-flash", "ministral-14b"])
        
        # Configure for cost-effective summarization
        config = LLMConfig(
            reasoning_effort="low",      # OpenAI: quick summarization
            text_verbosity="low",        # OpenAI: concise output
            thinking_level="MINIMAL",    # Gemini 3 Flash: minimal thinking for speed
            temperature=0.3              # Gemini: some creativity for natural summaries
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
        config_table.add_row("Temperature", str(config.temperature) if config.temperature else "default")
        console.print(config_table)
        
        llm_client = build_llm_client(model_option, config=config)
        success_count, error_count = process_txt_files(llm_client, input_dir, output_dir)
        
        # Display results
        console.print()
        if error_count == 0 and success_count > 0:
            console.print(Panel.fit(
                f"[bold green]✓ Completed successfully![/bold green]\n\n"
                f"[green]📄 {success_count} files summarized[/green]",
                border_style="green",
                box=box.ROUNDED
            ))
        elif success_count > 0:
            console.print(Panel.fit(
                f"[bold yellow]⚠ Completed with warnings[/bold yellow]\n\n"
                f"[green]✓ {success_count} files summarized[/green]\n"
                f"[red]✗ {error_count} files failed[/red]",
                border_style="yellow",
                box=box.ROUNDED
            ))
        else:
            console.print(Panel.fit(
                f"[bold red]✗ No files processed[/bold red]",
                border_style="red",
                box=box.ROUNDED
            ))
            
    except (FileNotFoundError, ValueError) as err:
        console.print(f"\n[red]✗ Error:[/red] {err}")
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
    except Exception as exc:
        console.print(f"\n[red]✗ Unexpected failure:[/red] {exc}")

if __name__ == '__main__':
    main()
