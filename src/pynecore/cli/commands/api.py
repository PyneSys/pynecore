"""API configuration and management commands."""
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..app import app
from pynecore.pynesys.api import APIClient, APIError, AuthError


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
api_app = typer.Typer(
    help="PyneSys API management commands for connection testing.")
app.add_typer(api_app, name="api")


@api_app.command()
def status(
        api_key: str | None = typer.Option(
            None, "--api-key", "-a",
            help="Test a specific API key directly (without saving to config)")
):
    """Check API configuration and connection status.

    This command displays:
    - Current API configuration (timeout, masked API key)
    - API connection test results
    - User information (User ID, token type)
    - Token expiration details (expires at, expires in human-readable format)

    Use this command to verify your API setup is working correctly
    and to check when your API token will expire.

    If --api-key is provided, it will test that specific key directly without
    saving it to the configuration. Otherwise, it loads the saved configuration
    from ~/.pynecore/api.toml.
    """
    try:
        if api_key:
            # Test the provided API key directly
            config = APIConfig(
                api_key=api_key,
                base_url="https://api.pynesys.io",
                timeout=30
            )

            # Display test configuration info
            table = Table(title="API Key Test")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Timeout", f"{config.timeout}s")
            table.add_row("API Key",
                          f"{config.api_key[:8]}...{config.api_key[-4:]}" if len(config.api_key) > 12 else "***")
            table.add_row("Mode", "[yellow]Direct Test (not saved)[/yellow]")

            console.print(table)
        else:
            # Load configuration from file
            config = ConfigManager.load_config(None)

            # Display configuration info
            table = Table(title="API Configuration")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Timeout", f"{config.timeout}s")
            table.add_row("API Key",
                          f"{config.api_key[:8]}...{config.api_key[-4:]}" if len(config.api_key) > 12 else "***")
            table.add_row("Mode", "[green]Saved Configuration[/green]")

            console.print(table)

        # Test connection
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
        ) as progress:
            task = progress.add_task("Testing API connection...", total=None)

            try:
                client = APIClient(config.api_key, config.base_url, config.timeout)
                result = client.verify_token()

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
        if not api_key and ("No configuration file found" in error_msg or "Configuration file not found" in error_msg):
            # No API configuration found and no API key provided - show helpful setup message
            console.print("[yellow]⚠️  No API configuration found[/yellow]")
            console.print()
            console.print("To get started with PyneSys API:")
            console.print(
                "1. 🌐 Visit [blue][link=https://pynesys.io]https://pynesys.io[/link][/blue] to get your API key")
            console.print("2. 🔧 Run [cyan]pyne api configure[/cyan] to set up your configuration")
            console.print("3. 🧪 Or test a key directly with [cyan]pyne api status --api-key YOUR_KEY[/cyan]")
            console.print()
            console.print("[dim]Need help? Check our documentation at https://pynesys.io/docs[/dim]")
        else:
            console.print(f"[red]Configuration error: {e}[/red]")
            console.print("[yellow]Hint:[/yellow] Use 'pyne api configure' to set up your API configuration.")
        raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
