"""
Shared rich console furniture for the pipeline scripts.

Every pipeline printed the same three things with slightly different code: a
five-column progress bar, a two-column "Setting / Value" configuration table,
and a two-column "Metric / Count" summary. Twenty-two hand-built ``Progress``
stacks had drifted apart in column order and spinner choice; this module makes
them one definition.

Usage:
    from common.console_utils import count_table, key_value_table, standard_progress

    with standard_progress(console) as progress:
        task = progress.add_task("[cyan]Working...", total=len(items))
        ...
        progress.update(task, advance=1)

    console.print(key_value_table([("Model", "gemini-flash"), ("Items", "42")]))
"""

from typing import Iterable, Optional, Sequence, Tuple

from rich import box
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Rows are (label, value) pairs; values may carry rich markup.
Rows = Iterable[Tuple[str, str]]


def standard_progress(
    console: Optional[Console] = None,
    *,
    show_eta: bool = False,
    **kwargs,
) -> Progress:
    """Return the project's standard progress bar.

    Spinner, description, bar, percentage, elapsed time — the combination every
    pipeline had reimplemented. Extra keyword arguments are forwarded to
    :class:`~rich.progress.Progress` (e.g. ``transient=True``).

    Args:
        show_eta: append a remaining-time column. Off by default because on a
            short run it is noise; worth turning on for the ones measured in
            days, where "how long is left" is the operator's actual question
            and rich's running estimate beats any constant in the source.
    """
    columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ]
    if show_eta:
        columns.append(TextColumn("[dim]left[/]"))
        columns.append(TimeRemainingColumn())

    return Progress(*columns, console=console, **kwargs)


def key_value_table(
    rows: Rows,
    *,
    title: Optional[str] = "Configuration",
    key_header: str = "Setting",
    value_header: str = "Value",
    value_style: str = "green",
) -> Table:
    """Build the standard two-column settings table.

    Rows whose value is ``None`` are skipped, so callers can express optional
    settings inline instead of guarding each ``add_row``.
    """
    table = Table(title=title, box=box.ROUNDED)
    table.add_column(key_header, style="dim")
    table.add_column(value_header, style=value_style)
    for key, value in rows:
        if value is None:
            continue
        table.add_row(key, str(value))
    return table


def count_table(rows: Rows, *, title: Optional[str] = None) -> Table:
    """Build the standard right-aligned "Metric / Count" summary table."""
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("Metric", style="dim")
    table.add_column("Count", justify="right")
    for label, value in rows:
        if value is None:
            continue
        table.add_row(label, str(value))
    return table


def print_file_table(
    console: Console,
    paths: Sequence,
    *,
    title: Optional[str] = None,
    name_style: str = "cyan",
) -> None:
    """Print the numbered "files to process" table with sizes in MB."""
    table = Table(title=title or f"📚 Files to process ({len(paths)})", box=box.ROUNDED)
    table.add_column("#", style="dim", width=4)
    table.add_column("Filename", style=name_style)
    table.add_column("Size", justify="right", style="green")
    for index, path in enumerate(paths, start=1):
        size_mb = path.stat().st_size / (1024 * 1024)
        table.add_row(str(index), path.name, f"{size_mb:.2f} MB")
    console.print(table)
