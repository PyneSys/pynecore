import tomllib
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..app import app, app_state
from ..utils.api_error_handler import APIErrorHandler
from pynecore.pynesys.compiler import PyneComp
from ...utils.file_utils import copy_mtime

__all__ = []

console = Console()


# noinspection PyShadowingBuiltins
@app.command()
def compile(
        script: Path = typer.Argument(
            ..., file_okay=True, dir_okay=False,
            help="Path to Pine Script file (.pine extension)"
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
        )
):
    """
    Compile Pine Script to Python using PyneSys API.

    CONFIGURATION:
        Default config: workdir/config/api.toml
        Fallback config: ~/.pynecore/api.toml

    REQUIREMENTS:
        - Pine Script version 6 only (version 5 not supported)
        - Valid PyneSys API key required (get one at [blue]https://app.pynesys.io[/])
    """

    # Ensure .py extension
    if script.suffix != ".pine":
        script = script.with_suffix(".pine")
    # Expand script path
    if len(script.parts) == 1:
        script = app_state.scripts_dir / script

    # Determine output path
    if output is None:
        output = script.with_suffix('.py')

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
