"""API configuration and management commands."""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..app import app
from ...api import ConfigManager, APIConfig, PynesysAPIClient, APIError, AuthError


def format_expires_in(seconds: int) -> str:
    """Format expires_in seconds into human-readable format.
    
    Args:
        seconds: Number of seconds until expiration
        
    Returns:
        Formatted string like "2 days, 5 hours, 30 minutes"
    """
    if seconds <= 0:
        return "Expired"
    
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    
    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    
    if not parts:
        return "Less than 1 minute"
    
    return ", ".join(parts)

__all__ = []

console = Console()
api_app = typer.Typer(help="PyneSys API configuration and management commands for authentication and connection testing")
app.add_typer(api_app, name="api")


@api_app.command()
def configure(
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="PyneSys API key",
        prompt="Enter your PyneSys API key",
        hide_input=True
    ),
    base_url: str = typer.Option(
        "https://api.pynesys.io",
        "--base-url",
        help="API base URL"
    ),
    timeout: int = typer.Option(
        30,
        "--timeout",
        help="Request timeout in seconds"
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Configuration file path (defaults to ~/.pynecore/config.json)"
    )
):
    """Configure PyneSys API settings and validate your API key.
    
    This command sets up your PyneSys API configuration including:
    - API key for authentication
    - Request timeout settings
    
    The configuration is saved to ~/.pynecore/config.json by default.
    You can specify a custom config path using --config option.
    
    The API key will be validated during configuration to ensure it's working.
    """
    try:
        # Create configuration
        config = APIConfig(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )
        
        # Test the API key
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Validating API key...", total=None)
            
            try:
                client = PynesysAPIClient(config.api_key, config.base_url, config.timeout)
                result = client.verify_token_sync()
                
                if result.valid:
                    progress.update(task, description="[green]API key validated![/green]")
                    console.print(f"[green]✓[/green] API key is valid")
                    console.print(f"[blue]User ID:[/blue] {result.user_id}")
                    console.print(f"[blue]Token Type:[/blue] {result.token_type}")
                    if result.expires_at:
                        console.print(f"[blue]Expires:[/blue] {result.expires_at}")
                else:
                    progress.update(task, description="[red]API key validation failed[/red]")
                    console.print("[red]✗[/red] The provided API key appears to be invalid.")
                    console.print("[yellow]💡 Please check your API key at [blue][link=https://pynesys.io]https://pynesys.io[/link][/blue] and try again.")
                    console.print("[dim]Make sure you've copied the complete API key without any extra spaces.[/dim]")
                    if result.message:
                        console.print(f"[dim]Details: {result.message}[/dim]")
                    raise typer.Exit(1)
                    
            except AuthError:
                progress.update(task, description="[red]Invalid API key[/red]")
                console.print("[red]✗[/red] The provided API key appears to be invalid.")
                console.print("[yellow]💡 Please check your API key at [blue][link=https://pynesys.io]https://pynesys.io[/link][/blue] and try again.")
                console.print("[dim]Make sure you've copied the complete API key without any extra spaces.[/dim]")
                raise typer.Exit(1)
                
            except APIError as e:
                progress.update(task, description="[red]API error[/red]")
                console.print(f"[red]✗[/red] API error: {e}")
                raise typer.Exit(1)
        
        # Save configuration
        ConfigManager.save_config(config, config_path)
        
        config_file = config_path or ConfigManager.get_default_config_path()
        console.print(f"[green]✓[/green] Configuration saved to: {config_file}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@api_app.command()
def status(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Configuration file path"
    )
):
    """Check API configuration and connection status.
    
    This command displays:
    - Current API configuration (base URL, timeout, masked API key)
    - API connection test results
    - User information (User ID, token type)
    - Token expiration details (expires at, expires in human-readable format)
    
    Use this command to verify your API setup is working correctly
    and to check when your API token will expire.
    """
    try:
        # Load configuration
        config = ConfigManager.load_config(config_path)
        
        # Display configuration info
        table = Table(title="API Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Base URL", config.base_url)
        table.add_row("Timeout", f"{config.timeout}s")
        table.add_row("API Key", f"{config.api_key[:8]}...{config.api_key[-4:]}" if len(config.api_key) > 12 else "***")
        
        console.print(table)
        
        # Test connection
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Testing API connection...", total=None)
            
            try:
                client = PynesysAPIClient(config.api_key, config.base_url, config.timeout)
                result = client.verify_token_sync()
                
                if result.valid:
                    progress.update(task, description="[green]Connection successful![/green]")
                    console.print(f"[green]✓[/green] API connection is working")
                    console.print(f"[blue]User ID:[/blue] {result.user_id}")
                    console.print(f"[blue]Token Type:[/blue] {result.token_type}")
                    if result.expires_at:
                        console.print(f"[blue]Expires At:[/blue] {result.expires_at}")
                    if result.expires_in:
                        console.print(f"[blue]Expires In:[/blue] {format_expires_in(result.expires_in)}")
                    if result.expiration:
                        console.print(f"[blue]Expiration:[/blue] {result.expiration}")
                else:
                    progress.update(task, description="[red]Connection failed[/red]")
                    console.print(f"[red]✗[/red] API connection failed: {result.message}")
                    
            except AuthError:
                progress.update(task, description="[red]Authentication failed[/red]")
                console.print("[red]✗[/red] Authentication failed. API key may be invalid or expired.")
                
            except APIError as e:
                progress.update(task, description="[red]API error[/red]")
                console.print(f"[red]✗[/red] API error: {e}")
                
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
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@api_app.command()
def reset(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Configuration file path"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Skip confirmation prompt"
    )
):
    """Reset API configuration by removing the configuration file.
    
    This command will:
    - Delete the API configuration file
    - Remove all stored API settings (API key, timeout)
    - Require you to run 'pyne api configure' again to set up the API
    
    Use --force to skip the confirmation prompt.
    This is useful when you want to start fresh with API configuration
    or switch to a different API key.
    """
    config_file = config_path or ConfigManager.get_default_config_path()
    
    if not config_file.exists():
        console.print(f"[yellow]No configuration file found at: {config_file}[/yellow]")
        return
    
    if not force:
        typer.confirm(
            f"Are you sure you want to delete the configuration file at {config_file}?",
            abort=True
        )
    
    try:
        config_file.unlink()
        console.print(f"[green]✓[/green] Configuration file deleted: {config_file}")
        console.print("[yellow]Use 'pyne api configure' to set up a new configuration.[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error deleting configuration: {e}[/red]")
        raise typer.Exit(1)