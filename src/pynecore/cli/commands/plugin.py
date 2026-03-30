from typer import Typer, Option, Argument, Exit, secho, colors

from ..app import app

__all__ = []

app_plugin = Typer(help="Plugin management commands")
app.add_typer(app_plugin, name="plugin")


def _get_capabilities(cls: type) -> list[str]:
    """Determine plugin capabilities from its class hierarchy."""
    from ...core.plugin import ProviderPlugin, CLIPlugin

    caps = []
    if isinstance(cls, type) and issubclass(cls, ProviderPlugin):
        caps.append('provider')
    # Future: ExtensionPlugin, LiveProviderPlugin checks will go here
    if isinstance(cls, type) and issubclass(cls, CLIPlugin):
        caps.append('cli')
    return caps


@app_plugin.command("list")
def list_plugins(
        plugin_type: str = Option(
            None, '--type', '-t',
            help="Filter by capability (e.g. 'provider', 'cli')",
        ),
):
    """
    List all installed PyneCore plugins.
    """
    from ...core.plugin import discover_plugins, get_plugin_metadata

    plugins = discover_plugins()
    if not plugins:
        secho("No plugins installed.", fg=colors.YELLOW)
        secho("")
        return

    # Collect data first to calculate column widths
    rows = []
    errors = []
    for name, ep in sorted(plugins.items()):
        try:
            cls = ep.load()
            meta = get_plugin_metadata(ep)
            caps = _get_capabilities(cls)

            if plugin_type and plugin_type not in caps:
                continue

            display_name = getattr(cls, 'plugin_name', '') or name
            version = f"v{meta['version']}" if meta['version'] else ''
            caps_str = ', '.join(caps) if caps else 'library'
            rows.append((name, display_name, version, caps_str))
        except Exception as e:
            errors.append((name, str(e)))

    if not rows and not errors:
        secho("No plugins found for the given filter.", fg=colors.YELLOW)
        secho("")
        return

    # Calculate column widths
    w_name = max((len(r[0]) for r in rows), default=0)
    w_disp = max((len(r[1]) for r in rows), default=0)
    w_ver = max((len(r[2]) for r in rows), default=0)

    secho(f"\n  Installed plugins:\n", fg=colors.BRIGHT_WHITE, bold=True)

    for name, display_name, version, caps_str in rows:
        secho(f"    {name:<{w_name}}   {display_name:<{w_disp}}   {version:<{w_ver}}   [{caps_str}]")

    for name, error in errors:
        secho(f"    {name:<{w_name}}   (failed to load: {error})", fg=colors.RED)

    secho("")
    secho("  Use 'pyne plugin info <name>' for details.", dim=True)
    secho("")


@app_plugin.command("info")
def plugin_info(
        name: str = Argument(..., help="Plugin name (e.g. 'ccxt', 'capitalcom')"),
):
    """
    Show detailed information about an installed plugin.
    """
    from ...core.plugin import discover_plugins, get_plugin_metadata
    import dataclasses

    plugins = discover_plugins()
    if name not in plugins:
        secho(f"Plugin '{name}' not found.", fg=colors.RED, err=True)
        secho(f"Install it with: pip install pynesys-pynecore-{name}  (official)", fg=colors.YELLOW, err=True)
        secho(f"             or: pip install pynecore-{name}  (3rd party)", fg=colors.YELLOW, err=True)
        raise Exit(1)

    ep = plugins[name]
    try:
        cls = ep.load()
    except Exception as e:
        secho(f"Failed to load plugin '{name}': {e}", fg=colors.RED, err=True)
        raise Exit(1)

    meta = get_plugin_metadata(ep)
    caps = _get_capabilities(cls)

    secho(f"\n  Plugin: {name}", fg=colors.BRIGHT_WHITE, bold=True)
    secho(f"  Package:       {meta['package']}")
    secho(f"  Version:       {meta['version'] or 'unknown'}")
    secho(f"  Description:   {meta['description'] or '-'}")
    secho(f"  Min PyneCore:  {'>=' + meta['min_pynecore'] if meta['min_pynecore'] else 'any'}")
    secho(f"  Capabilities:  {', '.join(caps) if caps else 'library'}")
    secho(f"  Entry point:   {ep.value}")

    config_cls = getattr(cls, 'Config', None)
    if config_cls and dataclasses.is_dataclass(config_cls):
        fields = dataclasses.fields(config_cls)
        if fields:
            secho(f"\n  Config fields (defaults):")
            for f in fields:
                default = f"= {f.default!r}" if f.default is not dataclasses.MISSING else "(required)"
                secho(f"    {f.name:20s} {default}")

    secho("")
