import csv
import os
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich import box

# Initialize rich console
console = Console()

# Shared Omeka client
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.omeka_client import OmekaClient


def update_item_fields(client: OmekaClient, item_id: str, spatial_ids_str: str | None, subject_ids_str: str | None) -> dict:
    """
    Updates an Omeka item with new spatial and subject links.
    Preserves existing data and avoids adding duplicate links.

    Returns:
        dict with keys: 'modified', 'spatial_added', 'subject_added', 'error'
    """
    result = {'modified': False, 'spatial_added': 0, 'subject_added': 0, 'error': False}

    item_data = client.get_item(int(item_id))

    if not item_data:
        result['error'] = True
        return result

    modified = False

    # --- Handle Spatial IDs (dcterms:spatial, property_id: 40) ---
    if spatial_ids_str:
        if 'dcterms:spatial' not in item_data or not isinstance(item_data['dcterms:spatial'], list):
            item_data['dcterms:spatial'] = []

        existing_spatial_resource_ids = set()
        for entry in item_data['dcterms:spatial']:
            if isinstance(entry, dict) and 'value_resource_id' in entry:
                try:
                    existing_spatial_resource_ids.add(int(entry['value_resource_id']))
                except ValueError:
                    pass

        new_spatial_ids_to_add = [s_id.strip() for s_id in spatial_ids_str.split('|') if s_id.strip()]
        for s_id_val_str in new_spatial_ids_to_add:
            try:
                s_id_int = int(s_id_val_str)
                if s_id_int not in existing_spatial_resource_ids:
                    item_data['dcterms:spatial'].append({
                        "type": "resource:item",
                        "property_id": 40,
                        "property_label": "Spatial Coverage",
                        "is_public": True,
                        "value_resource_id": s_id_int,
                        "value_resource_name": "items"
                    })
                    modified = True
                    result['spatial_added'] += 1
                    existing_spatial_resource_ids.add(s_id_int)
            except ValueError:
                pass

    # --- Handle Subject IDs (dcterms:subject, property_id: 3) ---
    if subject_ids_str:
        if 'dcterms:subject' not in item_data or not isinstance(item_data['dcterms:subject'], list):
            item_data['dcterms:subject'] = []

        existing_subject_resource_ids = set()
        for entry in item_data['dcterms:subject']:
            if isinstance(entry, dict) and 'value_resource_id' in entry:
                try:
                    existing_subject_resource_ids.add(int(entry['value_resource_id']))
                except ValueError:
                    pass

        new_subject_ids_to_add = [s_id.strip() for s_id in subject_ids_str.split('|') if s_id.strip()]
        for s_id_val_str in new_subject_ids_to_add:
            try:
                s_id_int = int(s_id_val_str)
                if s_id_int not in existing_subject_resource_ids:
                    item_data['dcterms:subject'].append({
                        "type": "resource:item",
                        "property_id": 3,
                        "property_label": "Subject",
                        "is_public": True,
                        "value_resource_id": s_id_int,
                        "value_resource_name": "items"
                    })
                    modified = True
                    result['subject_added'] += 1
                    existing_subject_resource_ids.add(s_id_int)
            except ValueError:
                pass

    if modified:
        if client.update_item(int(item_id), item_data):
            result['modified'] = True
        else:
            result['error'] = True

    return result


def main():
    # Display welcome banner
    console.print(Panel(
        "[bold]Omeka S Database Update[/bold]\n"
        "Updates items with reconciled spatial and subject entity links",
        title="Omeka Update Pipeline",
        border_style="cyan"
    ))

    # Initialize shared Omeka client
    try:
        client = OmekaClient.from_env()
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")

    if not os.path.isdir(output_dir):
        console.print(f"[red]Output directory not found: {output_dir}[/]")
        return

    # Find the latest main reconciled CSV file
    reconciled_csv_files = [
        f for f in os.listdir(output_dir)
        if f.endswith('_reconciled.csv') and '_ambiguous_authorities' not in f and '_unreconciled' not in f
    ]

    if not reconciled_csv_files:
        console.print(f"[red]No '*_reconciled.csv' files found in {output_dir}[/]")
        return

    latest_reconciled_csv_filename = max(reconciled_csv_files, key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
    input_csv_path = os.path.join(output_dir, latest_reconciled_csv_filename)

    # Display configuration
    config_table = Table(title="Configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="green")
    config_table.add_row("Input file", latest_reconciled_csv_filename)
    config_table.add_row("Omeka URL", client.base_url)
    console.print(config_table)
    console.print()

    # Statistics tracking
    stats = {
        'total': 0,
        'modified': 0,
        'skipped': 0,
        'errors': 0,
        'spatial_added': 0,
        'subject_added': 0
    }

    try:
        with open(input_csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames:
                console.print(f"[red]CSV file is empty or header is missing[/]")
                return

            required_columns = ['o:id', 'Spatial AI Reconciled ID', 'Subject AI Reconciled ID']
            missing_cols = [col for col in required_columns if col not in reader.fieldnames]
            if missing_cols:
                console.print(f"[red]Missing required columns: {', '.join(missing_cols)}[/]")
                return

            rows_to_process = list(reader)

        stats['total'] = len(rows_to_process)

        console.rule("[bold cyan]Processing Items")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Updating Omeka items...", total=len(rows_to_process))

            for row in rows_to_process:
                item_id = row.get('o:id')
                spatial_ids = row.get('Spatial AI Reconciled ID')
                subject_ids = row.get('Subject AI Reconciled ID')

                if not item_id:
                    stats['skipped'] += 1
                    progress.update(task, advance=1)
                    continue

                result = update_item_fields(client, item_id, spatial_ids, subject_ids)

                if result['error']:
                    stats['errors'] += 1
                elif result['modified']:
                    stats['modified'] += 1
                    stats['spatial_added'] += result['spatial_added']
                    stats['subject_added'] += result['subject_added']
                else:
                    stats['skipped'] += 1

                progress.update(task, advance=1)

    except FileNotFoundError:
        console.print(f"[red]Input CSV file not found: {input_csv_path}[/]")
        return
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/]")
        console.print_exception()
        return

    # Display summary
    console.print()
    summary_table = Table(title="Update Summary", box=box.ROUNDED)
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Count", justify="right")
    summary_table.add_row("Total items processed", str(stats['total']))
    summary_table.add_row("Items modified", f"[green]{stats['modified']}[/]")
    summary_table.add_row("Items skipped (no changes)", f"[dim]{stats['skipped']}[/]")
    summary_table.add_row("Errors", f"[red]{stats['errors']}[/]" if stats['errors'] > 0 else "[dim]0[/]")
    summary_table.add_row("Spatial links added", f"[cyan]{stats['spatial_added']}[/]")
    summary_table.add_row("Subject links added", f"[cyan]{stats['subject_added']}[/]")
    console.print(summary_table)

    console.print()
    console.print(Panel(
        f"[green]{chr(10003)}[/] Update complete!\n\n"
        f"Modified [cyan]{stats['modified']}[/] items with "
        f"[cyan]{stats['spatial_added']}[/] spatial and [cyan]{stats['subject_added']}[/] subject links",
        title="Process Complete",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
