import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..app import app, app_state
from ...api import ConfigManager, APIError, AuthError, RateLimitError, CompilationError
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
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to TOML configuration file (defaults to workdir/config/api.toml)"
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
        Default config: workdir/config/api.toml
        Fallback config: ~/.pynecore/api.toml
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
        # Create compilation service
        compilation_service = create_compilation_service(
            api_key=api_key,
            config_path=config_path
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
        try:
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
            
            console.print(f"[green]Compilation successful![/green] Your Pine Script is now ready: [cyan]{compiled_path}[/cyan]")
                
        except CompilationError as e:
            console.print(f"[red]❌ Oops! Compilation encountered an issue:[/red] {str(e)}")
            if e.validation_errors:
                console.print("[red]Validation errors:[/red]")
                for error in e.validation_errors:
                    console.print(f"  [red]• {error}[/red]")
            raise typer.Exit(1)
            
        except AuthError as e:
            console.print(f"[red]🔐 Authentication issue:[/red] {str(e)}")
            console.print("[yellow]🚀 Quick fix:[/yellow] Run [cyan]'pyne api configure'[/cyan] to set up your API key and get back on track!")
            raise typer.Exit(1)
            
        except RateLimitError as e:
            console.print("[red]🚦 Rate Limit Exceeded:[/red] You've hit your compilation limit")
            if e.retry_after:
                console.print(f"[yellow]⏰ Please try again in {e.retry_after} seconds[/yellow]")
            console.print("[yellow]💡 To increase your limits, consider upgrading your subscription at [blue][link=https://pynesys.io]https://pynesys.io[/link][/blue]")
            raise typer.Exit(1)
            
        except APIError as e:
            error_msg = str(e).lower()
            
            # Handle specific API error scenarios based on HTTP status codes
            if "400" in error_msg or "bad request" in error_msg:
                if "compilation fails" in error_msg or "script is too large" in error_msg:
                    console.print("[red]📝 Script Issue:[/red] Your Pine Script couldn't be compiled")
                    console.print("[yellow]💡 Common fixes:[/yellow]")
                    console.print("  • Check if your script is too large (try breaking it into smaller parts)")
                    console.print("  • Verify your Pine Script syntax is correct")
                    console.print("  • Make sure you're using Pine Script v6 syntax")
                else:
                    console.print(f"[red]⚠️  Request Error:[/red] {str(e)}")
                    console.print("[yellow]💡 This usually means there's an issue with the request format[/yellow]")
                    
            elif "401" in error_msg or "authentication" in error_msg or "no permission" in error_msg:
                console.print("[red]🔐 Authentication Failed:[/red] Your API credentials aren't working")
                console.print("[yellow]🚀 Quick fixes:[/yellow]")
                console.print("  • Check if your API key is valid and active")
                console.print("  • Verify your token type is allowed for compilation")
                console.print("[blue]🔑 Get a new API key at [link=https://pynesys.io]https://pynesys.io[/link][/blue]")
                console.print("[blue]⚙️  Then run [cyan]'pyne api configure'[/cyan] to update your configuration[/blue]")
                
            elif "404" in error_msg or "not found" in error_msg:
                console.print("[red]🔍 Not Found:[/red] The API endpoint or user wasn't found")
                console.print("[yellow]💡 This might indicate:[/yellow]")
                console.print("  • Your account may not exist or be accessible")
                console.print("  • There might be a temporary service issue")
                
            elif "422" in error_msg or "validation error" in error_msg:
                console.print("[red]📋 Validation Error:[/red] Your request data has validation issues")
                console.print("[yellow]💡 Common causes:[/yellow]")
                console.print("  • Invalid Pine Script syntax or structure")
                console.print("  • Missing required parameters")
                console.print("  • Incorrect data format")
                console.print(f"[dim]Details: {str(e)}[/dim]")
                
            elif "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
                console.print("[red]🚦 Rate Limit Exceeded:[/red] You've hit your compilation limit")
                console.print("[yellow]⏰ What you can do:[/yellow]")
                console.print("  • Wait a bit before trying again")
                console.print("  • Consider upgrading your plan for higher limits")
                console.print("[blue]💎 Upgrade at [link=https://pynesys.io/pricing]https://pynesys.io/pricing[/link][/blue]")
                
            elif "500" in error_msg or "server" in error_msg or "internal" in error_msg:
                console.print("[red]🔧 Server Error:[/red] Something went wrong on our end")
                console.print("[yellow]😅 Don't worry, it's not you![/yellow]")
                console.print("  • This is a temporary server issue")
                console.print("  • Please try again in a few moments")
                
            elif "unsupported pinescript version" in error_msg:
                console.print("[red]📌 Version Issue:[/red] Your Pine Script version isn't supported")
                if "version 5" in error_msg:
                    console.print("[yellow]🔄 Pine Script v5 → v6 Migration:[/yellow]")
                    console.print("  • Update your script to Pine Script version 6")
                    console.print("  • Most v5 scripts need minimal changes")
                    console.print("[blue]📖 Migration guide: [link=https://www.tradingview.com/pine-script-docs/en/v6/migration_guides/v5_to_v6_migration_guide.html]Pine Script v5→v6 Guide[/link][/blue]")
                else:
                    console.print("[yellow]💡 Only Pine Script version 6 is currently supported[/yellow]")
                    
            elif "api key" in error_msg:
                console.print("[red]🔑 API Key Issue:[/red] There's a problem with your API key")
                console.print("[blue]🔑 Get your API key at [link=https://pynesys.io]https://pynesys.io[/link][/blue]")
                console.print("[blue]⚙️  Then run [cyan]'pyne api configure'[/cyan] to set up your configuration[/blue]")
                
            else:
                # Generic API error fallback
                console.print(f"[red]🌐 API Error:[/red] {str(e)}")
                console.print("[yellow]💡 If this persists, please check:[/yellow]")
                console.print("  • Your internet connection")
                console.print("  • API service status")
                console.print("[blue]📞 Need help? [link=https://pynesys.io/support]Contact Support[/link][/blue]")
                
            raise typer.Exit(1)
                
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
