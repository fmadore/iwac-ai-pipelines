"""
Shared prompt discovery and interactive selection for pipelines that keep
several alternative prompt ``.md`` files in a ``prompts/`` directory
(audio transcription, video summaries).

Prompt files are named ``<number>_<description>.md``; the number orders the
menu and lets scripts attach behavior to specific prompts (e.g. prompt #1
enabling audio splitting).

Usage:
    from common.prompt_loader import discover_prompts, load_prompt_md, select_prompt_interactive
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from rich import box
from rich.console import Console
from rich.table import Table

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptOption:
    number: int  # 0 = unnumbered (shown but not selectable by number)
    description: str
    path: Path


def discover_prompts(prompts_dir: Path) -> List[PromptOption]:
    """Scan *prompts_dir* for ``.md`` prompt files.

    Filenames like ``01_full_transcription.md`` produce number 1 and the
    title-cased description "Full Transcription"; files without a numeric
    prefix get number 0.
    """
    if not prompts_dir.exists():
        LOGGER.warning("Prompts folder not found: %s", prompts_dir)
        return []

    options: List[PromptOption] = []
    for file_path in prompts_dir.iterdir():
        if not (file_path.is_file() and file_path.suffix.lower() == ".md"):
            continue
        name_part = file_path.stem
        number = 0
        description = name_part
        if "_" in name_part:
            number_part, rest = name_part.split("_", 1)
            if number_part.isdigit():
                number = int(number_part)
                description = rest
        options.append(PromptOption(number, description.replace("_", " ").title(), file_path))

    return sorted(options, key=lambda opt: opt.number)


def load_prompt_md(prompt_path: Path) -> str:
    """Load a prompt from a markdown file.

    Returns the content after the first ``# `` header (the header is a
    human-facing title, not part of the prompt); falls back to the whole
    file when there is no header.
    """
    content = prompt_path.read_text(encoding="utf-8")

    lines = content.split("\n")
    prompt_lines: List[str] = []
    found_header = False
    for line in lines:
        if line.startswith("# ") and not found_header:
            found_header = True
            continue
        if found_header and line.strip():
            prompt_lines.append(line)

    if prompt_lines:
        return "\n".join(prompt_lines).strip()
    return content.strip()


def select_prompt_interactive(
    prompts_dir: Path,
    console: Console,
    *,
    default_prompt: str,
    title: str = "Available Prompts",
) -> Tuple[str, Optional[int]]:
    """Show a numbered menu of prompts and return the chosen content.

    Returns:
        (prompt_content, prompt_number) — number is ``None`` when the
        default prompt was used, so callers can attach behavior to
        specific prompt numbers.
    """
    options = discover_prompts(prompts_dir)
    numbered = [opt for opt in options if opt.number > 0]

    if not numbered:
        console.print("[yellow]⚠[/] No numbered prompt files found. Using default prompt.")
        return default_prompt, None

    table = Table(title=f"📝 {title}", box=box.ROUNDED)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Description", style="green")
    for opt in options:
        table.add_row(str(opt.number) if opt.number > 0 else "-", opt.description)
    console.print()
    console.print(table)
    console.print()

    while True:
        try:
            choice = console.input(
                f"[bold]Select a prompt (1-{len(numbered)}) or press Enter for default:[/] "
            ).strip()
            if not choice:
                console.print("[dim]Using default prompt.[/]")
                return default_prompt, None

            choice_num = int(choice)
            selected = next((opt for opt in numbered if opt.number == choice_num), None)
            if selected is None:
                console.print(f"[red]✗[/] Invalid choice. Please select 1-{len(numbered)}.")
                continue

            console.print(f"[green]✓[/] Selected: [cyan]{selected.description}[/]")
            try:
                return load_prompt_md(selected.path), selected.number
            except OSError as exc:
                console.print(f"[red]✗[/] Error loading prompt '{selected.path}': {exc}")
                return default_prompt, None

        except ValueError:
            console.print("[red]✗[/] Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            console.print("\n[dim]Using default prompt.[/]")
            return default_prompt, None
