from typer import Typer, Option, Argument, Exit, secho, colors

from ..app import app

__all__ = []

app_plugin = Typer(help="Plugin management commands")
app.add_typer(app_plugin, name="plugin")

# Plugin group names and display labels
PLUGIN_GROUPS = {
    'provider': 'pyne.provider',
}


@app_plugin.command("list")
def list_plugins(
        plugin_type: str = Option(
            None, '--type', '-t',
            help="Filter by plugin type (e.g. 'provider')",
        ),
):
    """
    List all installed PyneCore plugins.
    """
    from ...core.plugin import discover_plugins

    if plugin_type and plugin_type not in PLUGIN_GROUPS:
        secho(f"Unknown plugin type: {plugin_type}", fg=colors.RED, err=True)
        secho(f"Available types: {', '.join(PLUGIN_GROUPS)}", fg=colors.YELLOW, err=True)
        raise Exit(1)

    groups = {plugin_type: PLUGIN_GROUPS[plugin_type]} if plugin_type else PLUGIN_GROUPS
    found_any = False

    for type_name, group in groups.items():
        plugins = discover_plugins(group)
        if not plugins:
            continue
        found_any = True
        secho(f"\n  {type_name} plugins:", fg=colors.BRIGHT_WHITE, bold=True)
        for name, ep in sorted(plugins.items()):
            try:
                cls = ep.load()
                display_name = getattr(cls, 'plugin_name', '') or name
                version = getattr(cls, 'plugin_version', '')
                version_str = f" v{version}" if version and version != "0.0.0" else ""
                secho(f"    {name:20s} {display_name}{version_str}  ({ep.value})")
            except Exception as e:
                secho(f"    {name:20s} (failed to load: {e})", fg=colors.RED)

    if not found_any:
        secho("No plugins installed.", fg=colors.YELLOW)

    secho("")


@app_plugin.command("info")
def plugin_info(
        name: str = Argument(..., help="Plugin name (e.g. 'ccxt', 'capitalcom')"),
):
    """
    Show detailed information about an installed plugin.
    """
    from ...core.plugin import discover_plugins

    for type_name, group in PLUGIN_GROUPS.items():
        plugins = discover_plugins(group)
        if name in plugins:
            ep = plugins[name]
            try:
                cls = ep.load()
            except Exception as e:
                secho(f"Failed to load plugin '{name}': {e}", fg=colors.RED, err=True)
                raise Exit(1)

            secho(f"\n  Plugin: {name}", fg=colors.BRIGHT_WHITE, bold=True)
            secho(f"  Type:          {type_name}")
            secho(f"  Display name:  {getattr(cls, 'plugin_name', '') or name}")
            secho(f"  Version:       {getattr(cls, 'plugin_version', 'unknown')}")
            secho(f"  Entry point:   {ep.value}")
            secho(f"  Min PyneCore:  {getattr(cls, 'min_pynecore_version', '') or 'any'}")

            config_cls = getattr(cls, 'Config', None)
            if config_cls:
                import dataclasses
                fields = dataclasses.fields(config_cls)
                if fields:
                    secho(f"\n  Config fields:")
                    for f in fields:
                        default = f"= {f.default!r}" if f.default is not dataclasses.MISSING else "(required)"
                        secho(f"    {f.name:20s} {default}")

            secho("")
            return

    secho(f"Plugin '{name}' not found.", fg=colors.RED, err=True)
    secho(f"Install it with: pip install pynecore-{name}", fg=colors.YELLOW, err=True)
    raise Exit(1)
