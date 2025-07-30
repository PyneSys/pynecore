import typer
from typer import Typer, Argument, Option
from pathlib import Path
from typing import Optional
import shutil
import subprocess
import sys

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

from ..app import app, app_state
from ..template_engine import TemplateEngine, get_plugin_templates_dir, get_default_template_variables
from ..plugin_manager import plugin_manager

app_plugins = Typer(help="Plugin management commands")
app.add_typer(app_plugins, name="plugin")
console = Console()


@app_plugins.command()
def list(
    plugin_type: Optional[str] = Option(None, help="Filter by plugin type (provider, indicator, strategy)"),
    show_errors: bool = Option(False, "--show-errors", help="Show plugins with errors")
):
    """List available and installed plugins"""
    console.print("[bold blue]Discovering PyneCore plugins...[/bold blue]")
    plugin_manager.list_plugins(plugin_type=plugin_type, show_errors=show_errors)


@app_plugins.command()
def create(
    plugin_name: str = Argument(..., help="Name of the plugin to create"),
    plugin_type: str = Option("provider", help="Type of plugin (provider, indicator, etc.)"),
    output_dir: Path = Option(Path.cwd(), help="Output directory for plugin"),
    force: bool = Option(False, "--force", help="Overwrite existing plugin directory")
):
    """Create a new plugin template"""
    # Validate plugin type
    supported_types = ["provider"]  # Can be extended later
    if plugin_type not in supported_types:
        console.print(f"[red]Error: Unsupported plugin type '{plugin_type}'[/red]")
        console.print(f"Supported types: {', '.join(supported_types)}")
        raise typer.Exit(1)
    
    # Validate plugin name
    if not plugin_name.replace('-', '').replace('_', '').isalnum():
        console.print(f"[red]Error: Plugin name '{plugin_name}' contains invalid characters[/red]")
        console.print("Plugin names should only contain letters, numbers, hyphens, and underscores")
        raise typer.Exit(1)
    
    # Determine output directory
    plugin_dir = output_dir / f"pynecore-{plugin_name.lower().replace('_', '-')}-{plugin_type}"
    
    # Check if directory already exists
    if plugin_dir.exists():
        if not force:
            if not Confirm.ask(f"Directory '{plugin_dir}' already exists. Overwrite?"):
                console.print("[yellow]Plugin creation cancelled[/yellow]")
                raise typer.Exit(0)
        
        # Remove existing directory
        shutil.rmtree(plugin_dir)
    
    # Get template directory
    template_dir = get_plugin_templates_dir() / plugin_type
    if not template_dir.exists():
        console.print(f"[red]Error: No template found for plugin type '{plugin_type}'[/red]")
        raise typer.Exit(1)
    
    # Initialize template engine
    template_engine = TemplateEngine()
    template_variables = get_default_template_variables(plugin_name, plugin_type)
    template_engine.set_variables(template_variables)
    
    console.print(f"[bold green]Creating {plugin_type} plugin '{plugin_name}'...[/bold green]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating plugin files...", total=None)
        
        try:
            # Create plugin directory
            plugin_dir.mkdir(parents=True, exist_ok=True)
            
            # Process all template files
            _process_template_directory(template_dir, plugin_dir, template_engine, progress, task)
            
            progress.update(task, description="Plugin created successfully!")
            
        except Exception as e:
            console.print(f"[red]Error creating plugin: {e}[/red]")
            # Clean up on error
            if plugin_dir.exists():
                shutil.rmtree(plugin_dir)
            raise typer.Exit(1)
    
    # Show success message with next steps
    _show_creation_success(plugin_name, plugin_type, plugin_dir)


def _process_template_directory(template_dir: Path, output_dir: Path, template_engine: TemplateEngine, progress, task):
    """Recursively process template directory"""
    # Skip patterns for files/directories that shouldn't be processed
    skip_patterns = {
        '__pycache__',
        '.pyc',
        '.pyo',
        '.pyd',
        '.so',
        '.dylib',
        '.dll',
        '.git',
        '.svn',
        '.hg',
        '.DS_Store',
        'Thumbs.db'
    }
    
    for item in template_dir.rglob("*"):
        # Skip if any part of the path contains skip patterns
        if any(skip_pattern in str(item) for skip_pattern in skip_patterns):
            continue
            
        # Skip if file extension is in skip patterns
        if item.suffix in {'.pyc', '.pyo', '.pyd', '.so', '.dylib', '.dll'}:
            continue
            
        if item.is_file():
            # Calculate relative path from template directory
            rel_path = item.relative_to(template_dir)
            
            # Process path template variables (e.g., {{plugin_name_snake}}_provider)
            output_path_str = str(rel_path)
            output_path_str = template_engine.render(output_path_str)
            output_path = output_dir / output_path_str
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Update progress
            progress.update(task, description=f"Processing {rel_path}...")
            
            try:
                # Render template file
                template_engine.render_file(item, output_path)
            except UnicodeDecodeError:
                # If we can't decode the file as text, copy it as binary
                shutil.copy2(item, output_path)


def _show_creation_success(plugin_name: str, plugin_type: str, plugin_dir: Path):
    """Show success message with next steps"""
    plugin_name_kebab = plugin_name.lower().replace('_', '-')
    
    success_text = f"""
[bold green]✓ Plugin created successfully![/bold green]

[bold]Plugin Details:[/bold]
• Name: {plugin_name}
• Type: {plugin_type}
• Directory: {plugin_dir}
• Package: pynecore-{plugin_name_kebab}-{plugin_type}

[bold]Next Steps:[/bold]

1. [cyan]Navigate to the plugin directory:[/cyan]
   cd "{plugin_dir}"

2. [cyan]Customize the implementation:[/cyan]
   • Edit src/{plugin_name.lower().replace('-', '_')}_provider/provider.py
   • Update configuration in providers.toml
   • Modify pyproject.toml with your details

3. [cyan]Install in development mode:[/cyan]
   pip install -e ".[dev]"

4. [cyan]Test your plugin:[/cyan]
   pytest

5. [cyan]Use with PyneCore:[/cyan]
   pyne data download --provider {plugin_name.lower().replace('-', '_')} --list-symbols

[bold]Documentation:[/bold]
See README.md in the plugin directory for detailed implementation guidance.
    """
    
    panel = Panel(
        success_text.strip(),
        title=f"Plugin '{plugin_name}' Created",
        border_style="green"
    )
    
    console.print(panel)


@app_plugins.command()
def install(
    plugin_name: str = Argument(..., help="Name of the plugin to install from PyPI"),
    upgrade: bool = Option(False, "--upgrade", "-U", help="Upgrade if already installed")
):
    """Install a plugin from PyPI"""
    # Construct expected package name
    if not plugin_name.startswith("pynecore-"):
        package_name = f"pynecore-{plugin_name}"
    else:
        package_name = plugin_name
    
    console.print(f"[bold blue]Installing plugin: {package_name}[/bold blue]")
    
    # Prepare pip command
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append(package_name)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Installing {package_name}...", total=None)
            
            # Run pip install
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                progress.update(task, description="Installation completed!")
                console.print(f"[green]✓ Successfully installed {package_name}[/green]")
                
                # Try to discover the new plugin
                console.print("[blue]Discovering new plugin...[/blue]")
                plugin_manager.discover_plugins()
                
            else:
                console.print(f"[red]✗ Failed to install {package_name}[/red]")
                console.print(f"[red]Error: {result.stderr}[/red]")
                raise typer.Exit(1)
                
    except Exception as e:
        console.print(f"[red]Error during installation: {e}[/red]")
        raise typer.Exit(1)


@app_plugins.command()
def info(
    plugin_name: str = Argument(..., help="Name of the plugin to show info for")
):
    """Show plugin information and status"""
    console.print(f"[bold blue]Getting information for plugin: {plugin_name}[/bold blue]")
    plugin_manager.show_plugin_info(plugin_name)