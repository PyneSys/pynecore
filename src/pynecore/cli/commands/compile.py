import tomllib
import time
from pathlib import Path
from datetime import datetime

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..app import app, app_state
from ..utils.api_error_handler import APIErrorHandler
from pynecore.pynesys.compiler import PyneComp
from ...utils.file_utils import copy_mtime

__all__ = []

console = Console()


def _print_usage(compiler: PyneComp, sleep: float = 0):
    """Print usage statistics in a nicely formatted table."""
    usage = compiler.get_usage()

    # Create the main usage table
    table = Table(title="API Usage Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Period", style="cyan", no_wrap=True)
    table.add_column("Used", style="bold red", justify="right")
    table.add_column("Limit", style="bold green", justify="right")
    table.add_column("Remaining", style="bold yellow", justify="right")
    table.add_column("Reset At", style="dim")

    # Add daily usage
    table.add_row(
        "Daily",
        str(usage.daily.used),
        str(usage.daily.limit),
        str(usage.daily.remaining),
        usage.daily.reset_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    )

    # Add hourly usage
    table.add_row(
        "Hourly",
        str(usage.hourly.used),
        str(usage.hourly.limit),
        str(usage.hourly.remaining),
        usage.hourly.reset_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    )

    console.print(table)

    # Show API key expiration information
    try:
        # Sleep to wait for usage is saved in the background of the DB
        if sleep > 0:
            time.sleep(sleep)

        token_info = compiler.validate_api_key()
        if token_info.valid and token_info.expiration:
            print()
            expiry_table = Table(title="API Key Information", show_header=True, header_style="bold magenta")
            expiry_table.add_column("Property", style="cyan", no_wrap=True)
            expiry_table.add_column("Value", style="white")

            # Calculate days remaining
            now = datetime.now()
            days_remaining = (token_info.expiration - now).days
            hours_remaining = (token_info.expiration - now).total_seconds() / 3600

            # Format expiration time
            expiry_str = token_info.expiration.strftime("%Y-%m-%d %H:%M:%S")

            # Add rows
            expiry_table.add_row("Expires At", expiry_str)

            if days_remaining > 0:
                expiry_table.add_row("Days Remaining", f"{days_remaining} days")
            elif hours_remaining > 0:
                expiry_table.add_row("Hours Remaining", f"{hours_remaining:.1f} hours")
            else:
                expiry_table.add_row("Status", "[red]EXPIRED[/red]")

            console.print(expiry_table)
    except (AttributeError, ValueError, TypeError) as e:
        print()
        console.print(f"[yellow]⚠[/yellow] Could not validate API key expiration: {str(e)}")
        console.print("[dim]API key information unavailable[/dim]")


# noinspection PyShadowingBuiltins
@app.command()
def compile(
        script: Path = typer.Argument(
            None, file_okay=True, dir_okay=False,
            help="Path to Pine Script file (.pine extension) or name of script in [cyan]scripts[/cyan] directory"
        ),
        output: Path | None = typer.Option(
            None, "--output", "-o",
            help="Output Python file path (defaults to same name with .py extension)"
        ),
        strict: bool = typer.Option(
            False, "--strict", '-s',
            help="Enable strict compilation mode with enhanced error checking"
        ),
        force: bool = typer.Option(
            False, "--force", "-f",
            help="Force recompilation even if output file is up-to-date"
        ),
        api_key: str | None = typer.Option(
            None, "--api-key", "-a",
            help="PyneSys API key (overrides configuration file)",
            envvar="PYNESYS_API_KEY"
        ),
        show_usage: bool = typer.Option(
            False, "--usage", "-u",
            help="Print API usage statistics after compilation"
        )
):
    """
    Compile Pine Script to Python using PyneSys API.

    The system automatically searches for the workdir folder in the current and parent directories.
    If not found, it creates or uses a workdir folder in the current directory.

    Only version 6 is supported (no v4 or v5 support).
    API key can be provided via --api-key or via config file: [cyan]workdir/config/api.toml[/cyan].

    If no script is provided, will print usage statistics and exit.
    """

    # Read api.toml configuration
    api_config = {}
    try:
        with open(app_state.config_dir / 'api.toml', 'rb') as f:
            api_config = tomllib.load(f)['api']
    except KeyError:
        console.print("[red]Invalid API config file (api.toml)![/red]")
        raise typer.Exit(1)
    except FileNotFoundError:
        pass

    # Override API key if provided
    if api_key:
        api_config['api_key'] = api_key

    # Create the compiler instance
    compiler = PyneComp(**api_config)

    # Ensure script was provided
    if not script:
        _print_usage(compiler)
        raise typer.Exit(0)

    # Ensure .py extension
    if script.suffix != ".pine":
        script = script.with_suffix(".pine")
    # Expand script path
    if len(script.parts) == 1:
        script = app_state.scripts_dir / script

    # Determine output path
    if output is None:
        output = script.with_suffix('.py')

    # Check if compilation is needed (smart compilation)
    if not compiler.needs_compilation(script, output) and not force:
        console.print(f"[green]✓[/green] Output file is up-to-date: {output}")
        console.print("[dim]Use --force to recompile anyway[/dim]")
        return

    # Compile script
    with APIErrorHandler(console):
        with Progress(
                SpinnerColumn(finished_text="[green]✓"),
                TextColumn("[progress.description]{task.description}"),
                console=console
        ) as progress:
            task = progress.add_task("Compiling Pine Script...", total=1)

            # Compile the .pine file to .py
            out_path = compiler.compile(script, output, force=force, strict=strict)

            progress.update(task, completed=1)

        # Preserve modification time from source file
        copy_mtime(script, output)

        console.print(f"The compiled script is located at: [cyan]{out_path}[/cyan]")

        if show_usage:
            print()
            _print_usage(compiler, 1.0)
