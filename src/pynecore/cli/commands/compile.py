from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..app import app
from ..utils.api_error_handler import handle_api_errors
from ...core.compiler import create_compilation_service
from ...utils.file_utils import preserve_mtime

__all__ = []

console = Console()


# noinspection PyShadowingBuiltins
@app.command()
def compile(
        script_path: Path = typer.Argument(
            ...,
            help="Path to Pine Script file (.pine extension)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True
        ),
        output: Optional[Path] = typer.Option(
            None,
            "--output", "-o",
            help="Output Python file path (defaults to same name with .py extension)"
        ),
        strict: bool = typer.Option(
            False,
            "--strict",
            help="Enable strict compilation mode with enhanced error checking"
        ),
        force: bool = typer.Option(
            False,
            "--force",
            help="Force recompilation even if output file is up-to-date"
        ),
        api_key: Optional[str] = typer.Option(
            None,
            "--api-key",
            help="PyneSys API key (overrides configuration file)",
            envvar="PYNESYS_API_KEY"
        )
):
    """
    Compile Pine Script to Python using PyneSys API.

    USAGE:
        pyne compile <file.pine>                    # Compile single file
        pyne compile <file.pine> --force            # Force recompile even if up-to-date
        pyne compile <file.pine> --strict           # Enable strict compilation mode
        pyne compile <file.pine> --output <path>    # Specify output file path

    CONFIGURATION:
        Default config: workdir/config/api.toml
        Fallback config: ~/.pynecore/api.toml

        Config format (TOML only):
        [api]
        pynesys_api_key = "your_api_key_here"
        base_url = "https://api.pynesys.io/"  # optional
        timeout = 30  # optional, seconds

    SMART COMPILATION:
        - Automatically skips recompilation if output is newer than input
        - Use --force to override this behavior
        - Preserves file modification timestamps

    REQUIREMENTS:
        - Pine Script version 6 only (version 5 not supported)
        - Valid PyneSys API key required
        - Input file must have .pine extension
        - Output defaults to same name with .py extension

    Use 'pyne api configure' to set up your API configuration.
    """
    try:
        # Create compilation service
        compilation_service = create_compilation_service(
            api_key=api_key,

        )

        # Determine output path
        if output is None:
            output = script_path.with_suffix('.py')

        # Check if compilation is needed (smart compilation)
        if not compilation_service.needs_compilation(script_path, output) and not force:
            console.print(f"[green]✓[/green] Output file is up-to-date: {output}")
            console.print("[dim]Use --force to recompile anyway[/dim]")
            return

        # Compile script
        with handle_api_errors(console):
            with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
            ) as progress:
                progress.add_task("Compiling Pine Script...", total=None)

                # Compile the .pine file
                compiled_path = compilation_service.compile_file(
                    script_path,
                    output,
                    force=force,
                    strict=strict
                )

            # Preserve modification time from source file
            preserve_mtime(script_path, output)

            console.print(f"[green]Compilation successful![/green] Your Pine Script is now ready: "
                          f"[cyan]{compiled_path}[/cyan]")

    except ValueError as e:
        error_msg = str(e)
        if "No configuration file found" in error_msg or "Configuration file not found" in error_msg:
            # No API configuration found - show helpful setup message
            console.print("[yellow]⚠️  No API configuration found[/yellow]")
            console.print()
            console.print("[bold]Quick setup (takes just few minutes):[/bold]")
            console.print("1. 🌐 Get your API key at [blue][link=https://pynesys.io]https://pynesys.io[/link][/blue]")
            console.print("2. 🔧 Run [cyan]pyne api configure[/cyan] to save your configuration")
            console.print()
            console.print("[dim]💬 Need assistance? Our docs are here: https://pynesys.io/docs[/dim]")
        elif "this file format isn't supported" in error_msg:
            # File format error - show the friendly message directly
            console.print(f"[red]{e}[/red]")
        else:
            console.print(f"[red]Attention:[/red] {e}")
        raise typer.Exit(1)
