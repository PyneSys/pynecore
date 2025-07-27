"""Plugin Manager for PyneCore

This module handles plugin discovery, registration, and management using entry points.
"""

from typing import Dict, List, Any, Optional, Type
from pathlib import Path
import importlib.metadata
import importlib.util
import sys
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel


@dataclass
class PluginInfo:
    """Information about a discovered plugin"""
    name: str
    version: str
    description: str
    entry_point: str
    plugin_type: str
    module_name: str
    class_name: str
    installed: bool = False
    loaded: bool = False
    error: Optional[str] = None


class PluginManager:
    """Manages PyneCore plugins using entry points"""
    
    def __init__(self):
        self.console = Console()
        self._discovered_plugins: Dict[str, PluginInfo] = {}
        self._loaded_plugins: Dict[str, Any] = {}
        
        # Entry point groups for different plugin types
        self.entry_point_groups = {
            "provider": "pynecore.providers",
            "indicator": "pynecore.indicators",
            "strategy": "pynecore.strategies",
        }
    
    def discover_plugins(self, plugin_type: Optional[str] = None) -> Dict[str, PluginInfo]:
        """Discover plugins using entry points and built-in providers
        
        Args:
            plugin_type: Optional plugin type to filter by (provider, indicator, strategy)
            
        Returns:
            Dictionary of discovered plugins
        """
        discovered = {}
        
        # Add built-in providers if we're looking for providers or all types
        if plugin_type is None or plugin_type == "provider":
            discovered.update(self._discover_builtin_providers())
        
        # Determine which entry point groups to check
        if plugin_type and plugin_type in self.entry_point_groups:
            groups_to_check = {plugin_type: self.entry_point_groups[plugin_type]}
        else:
            groups_to_check = self.entry_point_groups
        
        for ptype, group_name in groups_to_check.items():
            try:
                # Get entry points for this group
                entry_points = importlib.metadata.entry_points().select(group=group_name)
                
                for entry_point in entry_points:
                    try:
                        # Get distribution info
                        dist = entry_point.dist
                        
                        plugin_info = PluginInfo(
                            name=entry_point.name,
                            version=dist.version if dist else "unknown",
                            description=self._get_package_description(dist),
                            entry_point=f"{entry_point.module}:{entry_point.attr}",
                            plugin_type=ptype,
                            module_name=entry_point.module,
                            class_name=entry_point.attr,
                            installed=True
                        )
                        
                        discovered[entry_point.name] = plugin_info
                        
                    except Exception as e:
                        # Create error entry for failed plugin
                        error_info = PluginInfo(
                            name=entry_point.name,
                            version="unknown",
                            description="Failed to load plugin info",
                            entry_point=f"{entry_point.module}:{entry_point.attr}",
                            plugin_type=ptype,
                            module_name=entry_point.module,
                            class_name=entry_point.attr,
                            installed=True,
                            error=str(e)
                        )
                        discovered[entry_point.name] = error_info
                        
            except Exception as e:
                self.console.print(f"[red]Error discovering plugins for group {group_name}: {e}[/red]")
        
        self._discovered_plugins.update(discovered)
        return discovered
    
    def _discover_builtin_providers(self) -> Dict[str, PluginInfo]:
        """Discover built-in providers
        
        Returns:
            Dictionary of built-in provider info
        """
        builtin_plugins = {}
        
        try:
            # Import built-in providers info
            from ..providers import _builtin_providers
            
            for name, provider_class in _builtin_providers.items():
                # Get version from the main package
                try:
                    import pynecore
                    version = getattr(pynecore, '__version__', 'unknown')
                except:
                    version = 'unknown'
                
                # Get description from provider class docstring
                description = "Built-in provider"
                if provider_class.__doc__:
                    # Get first line of docstring
                    description = provider_class.__doc__.strip().split('\n')[0]
                
                plugin_info = PluginInfo(
                    name=name,
                    version=version,
                    description=description,
                    entry_point=f"{provider_class.__module__}:{provider_class.__name__}",
                    plugin_type="provider",
                    module_name=provider_class.__module__,
                    class_name=provider_class.__name__,
                    installed=True,
                    loaded=True  # Built-in providers are always "loaded"
                )
                
                builtin_plugins[name] = plugin_info
                
        except ImportError:
            # Built-in providers not available
            pass
        
        return builtin_plugins
    
    def _get_package_description(self, dist) -> str:
        """Get package description from distribution metadata"""
        if not dist:
            return "No description available"
        
        try:
            metadata = dist.metadata
            return metadata.get("Summary", metadata.get("Description", "No description available"))
        except Exception:
            return "No description available"
    
    def load_plugin(self, plugin_name: str) -> Optional[Any]:
        """Load a specific plugin by name
        
        Args:
            plugin_name: Name of the plugin to load
            
        Returns:
            Loaded plugin class or None if failed
        """
        if plugin_name in self._loaded_plugins:
            return self._loaded_plugins[plugin_name]
        
        # Discover plugins if not already done
        if plugin_name not in self._discovered_plugins:
            self.discover_plugins()
        
        if plugin_name not in self._discovered_plugins:
            self.console.print(f"[red]Plugin '{plugin_name}' not found[/red]")
            return None
        
        plugin_info = self._discovered_plugins[plugin_name]
        
        if plugin_info.error:
            self.console.print(f"[red]Plugin '{plugin_name}' has errors: {plugin_info.error}[/red]")
            return None
        
        try:
            # Load the module
            module = importlib.import_module(plugin_info.module_name)
            
            # Get the plugin class
            plugin_class = getattr(module, plugin_info.class_name)
            
            # Cache the loaded plugin
            self._loaded_plugins[plugin_name] = plugin_class
            plugin_info.loaded = True
            
            return plugin_class
            
        except Exception as e:
            error_msg = f"Failed to load plugin: {e}"
            plugin_info.error = error_msg
            self.console.print(f"[red]{error_msg}[/red]")
            return None
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider plugin names
        
        Returns:
            List of provider plugin names
        """
        providers = self.discover_plugins("provider")
        return [name for name, info in providers.items() if not info.error]
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get information about a specific plugin
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            PluginInfo object or None if not found
        """
        if plugin_name not in self._discovered_plugins:
            self.discover_plugins()
        
        return self._discovered_plugins.get(plugin_name)
    
    def list_plugins(self, plugin_type: Optional[str] = None, show_errors: bool = False) -> None:
        """Display a formatted list of available plugins
        
        Args:
            plugin_type: Optional plugin type to filter by
            show_errors: Whether to show plugins with errors
        """
        plugins = self.discover_plugins(plugin_type)
        
        if not plugins:
            self.console.print("[yellow]No plugins found[/yellow]")
            return
        
        # Display status meanings
        self.console.print("[bold blue]Status Meanings:[/bold blue]")
        self.console.print("  [green]✓ Loaded[/green]    - Plugin is loaded and ready to use")
        self.console.print("  [yellow]○ Available[/yellow] - Plugin is installed and will load on demand")
        if show_errors:
            self.console.print("  [red]✗ Error[/red]     - Plugin has errors and cannot be loaded")
        self.console.print()
        
        # Filter out error plugins if requested
        if not show_errors:
            plugins = {name: info for name, info in plugins.items() if not info.error}
        
        # Group by plugin type
        by_type = {}
        for name, info in plugins.items():
            if info.plugin_type not in by_type:
                by_type[info.plugin_type] = []
            by_type[info.plugin_type].append((name, info))
        
        for ptype, plugin_list in by_type.items():
            # Create table for this plugin type
            table = Table(title=f"{ptype.title()} Plugins")
            table.add_column("Name", style="cyan")
            table.add_column("Version", style="green")
            table.add_column("Description", style="white")
            table.add_column("Status", style="yellow")
            
            for name, info in sorted(plugin_list):
                status = "✓ Loaded" if info.loaded else "○ Available"
                if info.error:
                    status = "✗ Error"
                
                description = info.description
                if info.error and show_errors:
                    description += f" (Error: {info.error})"
                
                table.add_row(name, info.version, description, status)
            
            self.console.print(table)
            self.console.print()
    
    def show_plugin_info(self, plugin_name: str) -> None:
        """Display detailed information about a specific plugin
        
        Args:
            plugin_name: Name of the plugin to show info for
        """
        plugin_info = self.get_plugin_info(plugin_name)
        
        if not plugin_info:
            self.console.print(f"[red]Plugin '{plugin_name}' not found[/red]")
            return
        
        # Create info panel
        info_text = f"""
[bold]Name:[/bold] {plugin_info.name}
[bold]Version:[/bold] {plugin_info.version}
[bold]Type:[/bold] {plugin_info.plugin_type}
[bold]Entry Point:[/bold] {plugin_info.entry_point}
[bold]Description:[/bold] {plugin_info.description}
[bold]Installed:[/bold] {'Yes' if plugin_info.installed else 'No'}
[bold]Loaded:[/bold] {'Yes' if plugin_info.loaded else 'No'}
        """
        
        if plugin_info.error:
            info_text += f"\n[bold red]Error:[/bold red] {plugin_info.error}"
        
        panel = Panel(
            info_text.strip(),
            title=f"Plugin Information: {plugin_name}",
            border_style="blue"
        )
        
        self.console.print(panel)
    
    def validate_plugin_structure(self, plugin_path: Path) -> bool:
        """Validate that a plugin directory has the correct structure
        
        Args:
            plugin_path: Path to the plugin directory
            
        Returns:
            True if structure is valid, False otherwise
        """
        required_files = [
            "pyproject.toml",
            "README.md",
        ]
        
        for file_name in required_files:
            if not (plugin_path / file_name).exists():
                self.console.print(f"[red]Missing required file: {file_name}[/red]")
                return False
        
        # Check for src directory structure
        src_dir = plugin_path / "src"
        if not src_dir.exists():
            self.console.print(f"[red]Missing src directory[/red]")
            return False
        
        return True


# Global plugin manager instance
plugin_manager = PluginManager()