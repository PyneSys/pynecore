import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..app import app, app_state
from ...api import PynesysAPIClient, ConfigManager, APIError, AuthError, RateLimitError, CompilationError
from ...utils.file_utils import should_compile, preserve_mtime

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
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to TOML configuration file (defaults to workdir/config/api_config.toml)"
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="PyneSys API key (overrides configuration file)",
        envvar="PYNESYS_API_KEY"
    )
):
    """Compile Pine Script to Python using PyneSys API.
    
    USAGE:
        pyne compile <file.pine>                    # Compile single file
        pyne compile <file.pine> --force            # Force recompile even if up-to-date
        pyne compile <file.pine> --strict           # Enable strict compilation mode
        pyne compile <file.pine> --output <path>    # Specify output file path
    
    CONFIGURATION:
        Default config: workdir/config/api_config.toml
        Fallback config: ~/.pynecore/api_config.toml
        Custom config: --config /path/to/config.toml
        
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
        # Load configuration
        if api_key:
            from ...api.config import APIConfig
            config = APIConfig(api_key=api_key)
        else:
            config = ConfigManager.load_config(config_path)
        
        # Read Pine Script content
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
        except Exception as e:
            console.print(f"[red]Error reading script file: {e}[/red]")
            raise typer.Exit(1)
        
        # Determine output path
        if output is None:
            output = script_path.with_suffix('.py')
        
        # Check if compilation is needed (smart compilation)
        if not should_compile(script_path, output, force):
            console.print(f"[green]✓[/green] Output file is up-to-date: {output}")
            console.print("[dim]Use --force to recompile anyway[/dim]")
            return
        
        # Compile script
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Compiling Pine Script...", total=None)
            
            try:
                # Use synchronous client for CLI
                client = PynesysAPIClient(config.api_key, config.base_url, config.timeout)
                result = client.compile_script_sync(script_content, strict=strict)
                
                if result.success:
                    # Write compiled Python code to output file
                    try:
                        output.parent.mkdir(parents=True, exist_ok=True)
                        with open(output, 'w', encoding='utf-8') as f:
                            f.write(result.compiled_code)
                        
                        # Preserve modification time from source file
                        preserve_mtime(script_path, output)
                        
                        progress.update(task, description="[green]Compilation successful![/green]")
                        console.print(f"[green]✓[/green] Compiled successfully to: {output}")
                        
                        if result.warnings:
                            console.print("[yellow]Warnings:[/yellow]")
                            for warning in result.warnings:
                                console.print(f"  [yellow]•[/yellow] {warning}")
                                
                    except Exception as e:
                        console.print(f"[red]Error writing output file: {e}[/red]")
                        raise typer.Exit(1)
                        
                else:
                    progress.update(task, description="[red]Compilation failed[/red]")
                    console.print(f"[red]✗[/red] Compilation failed: {result.error}")
                    
                    if result.details:
                        console.print("[red]Details:[/red]")
                        for detail in result.details:
                            console.print(f"  [red]•[/red] {detail}")
                    
                    raise typer.Exit(1)
                    
            except AuthError:
                progress.update(task, description="[red]Authentication failed[/red]")
                console.print("[red]✗[/red] Authentication failed. Please check your API key.")
                console.print("[yellow]Hint:[/yellow] Use 'pyne api configure' to set up your API key.")
                raise typer.Exit(1)
                
            except RateLimitError:
                progress.update(task, description="[red]Rate limit exceeded[/red]")
                console.print("[red]✗[/red] You've reached your API rate limit.")
                console.print("[yellow]💡 To increase your limits, consider upgrading your subscription at [blue][link=https://pynesys.io]https://pynesys.io[/link][/blue]")
                console.print("[dim]You can also try again in a few minutes/hours when your rate limit resets.[/dim]")
                raise typer.Exit(1)
                
            except CompilationError as e:
                progress.update(task, description="[red]Compilation error[/red]")
                console.print(f"[red]✗[/red] Compilation error: {e}")
                raise typer.Exit(1)
                
            except APIError as e:
                progress.update(task, description="[red]API error[/red]")
                console.print(f"[red]✗[/red] API error: {e}")
                raise typer.Exit(1)
                
    except ValueError as e:
        error_msg = str(e)
        if "No configuration file found" in error_msg or "Configuration file not found" in error_msg:
            # No API configuration found - show helpful setup message
            console.print("[yellow]⚠️  No API configuration found[/yellow]")
            console.print()
            console.print("To get started with PyneSys API:")
            console.print("1. 🌐 Visit [blue][link=https://pynesys.io]https://pynesys.io[/link][/blue] to get your API key")
            console.print("2. 🔧 Run [cyan]pyne api configure[/cyan] to set up your configuration")
            console.print()
            console.print("[dim]Need help? Check our documentation at https://pynesys.io/docs[/dim]")
        else:
            console.print(f"[red]Configuration error: {e}[/red]")
            console.print("[yellow]Hint:[/yellow] Use 'pyne api configure' to set up your API configuration.")
        raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)
