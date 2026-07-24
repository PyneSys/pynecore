from typer import Typer, Option, Argument, Exit, secho, colors

from ..app import app

__all__ = []

app_plugin = Typer(help="Plugin management commands")
app.add_typer(app_plugin, name="plugin")


def _get_capabilities(cls: type) -> list[str]:
    """Determine plugin capabilities from its class hierarchy."""
    from ...core.plugin import ProviderPlugin, CLIPlugin
    from ...core.plugin.broker import BrokerPlugin

    caps = []
    if isinstance(cls, type) and issubclass(cls, ProviderPlugin):
        caps.append('provider')
    if isinstance(cls, type) and issubclass(cls, BrokerPlugin):
        caps.append('broker')
    if isinstance(cls, type) and issubclass(cls, CLIPlugin):
        caps.append('cli')
    return caps


@app_plugin.command("list")
def list_plugins(
        plugin_type: str = Option(
            None, '--type', '-t',
            help="Filter by capability (e.g. 'provider', 'cli')",
        ),
        as_json: bool = Option(
            False, '--json',
            help="Machine readable output for tooling",
        ),
):
    """
    List all installed PyneCore plugins.
    """
    from ...core.plugin import (
        discover_plugin_entry_points, get_plugin_metadata, get_plugin_package, get_plugin_summary
    )

    plugins = discover_plugin_entry_points()
    if not plugins and not as_json:
        secho("No plugins installed.", fg=colors.YELLOW)
        secho("")
        return

    # Collect data first to calculate column widths. A plugin name may be
    # declared by several packages — list every one of them, flagged.
    rows = []
    errors = []
    for name, eps in sorted(plugins.items()):
        conflict = len(eps) > 1
        for ep in eps:
            try:
                cls = ep.load()
                meta = get_plugin_metadata(ep)
                caps = _get_capabilities(cls)

                if plugin_type and plugin_type not in caps:
                    continue

                display_name = getattr(cls, 'plugin_name', '') or name
                version = f"v{meta['version']}" if meta['version'] else ''
                caps_str = ', '.join(caps) if caps else 'library'
                summary = get_plugin_summary(cls) or meta['description']
                rows.append((name, display_name, version, caps_str, summary,
                             meta['package'], conflict))
            except Exception as e:
                errors.append((f"{name} ({get_plugin_package(ep)})" if conflict else name, str(e)))

    if as_json:
        import json
        secho(json.dumps({
            'plugins': [
                {
                    'name': name,
                    'display_name': display_name,
                    'version': version.lstrip('v'),
                    'capabilities': caps_str.split(', '),
                    'summary': summary,
                    'package': package,
                    'conflict': conflict,
                }
                for name, display_name, version, caps_str, summary, package, conflict in rows
            ],
            'errors': [{'name': name, 'error': error} for name, error in errors],
        }))
        return

    if not rows and not errors:
        secho("No plugins found for the given filter.", fg=colors.YELLOW)
        secho("")
        return

    # Calculate column widths
    w_name = max((len(r[0]) for r in rows), default=0)
    w_disp = max((len(r[1]) for r in rows), default=0)
    w_ver = max((len(r[2]) for r in rows), default=0)

    secho(f"\n  Installed plugins:\n", fg=colors.BRIGHT_WHITE, bold=True)

    indent = 4 + w_name + 3
    conflicting: dict[str, list[str]] = {}
    for name, display_name, version, caps_str, summary, package, conflict in rows:
        prefix = "  ! " if conflict else "    "
        secho(f"{prefix}{name:<{w_name}}   {display_name:<{w_disp}}   {version:<{w_ver}}   [{caps_str}]",
              fg=colors.YELLOW if conflict else None)
        if summary:
            secho(f"{' ' * indent}└─ {summary}", dim=True)
        if conflict:
            conflicting.setdefault(name, []).append(package)

    for name, error in errors:
        secho(f"    {name:<{w_name}}   (failed to load: {error})", fg=colors.RED)

    if conflicting:
        secho("")
        secho("  ! Name conflicts:", fg=colors.YELLOW, bold=True)
        for name, packages in sorted(conflicting.items()):
            secho(f"    '{name}' is declared by: {', '.join(packages)}", fg=colors.YELLOW)
        secho("    Loading such a plugin fails until all but one package is uninstalled.", dim=True)

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
    from ...core.plugin import (
        discover_plugin_entry_points, get_plugin_metadata, get_plugin_description, get_plugin_package
    )
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.padding import Padding
    import dataclasses

    candidates = discover_plugin_entry_points().get(name)
    if not candidates:
        secho(f"Plugin '{name}' not found.", fg=colors.RED, err=True)
        secho(f"Install it with: pip install pynesys-pynecore-{name}  (official)", fg=colors.YELLOW, err=True)
        secho(f"             or: pip install pynecore-{name}  (3rd party)", fg=colors.YELLOW, err=True)
        raise Exit(1)

    if len(candidates) > 1:
        secho(f"Plugin name '{name}' is ambiguous — declared by {len(candidates)} installed packages:",
              fg=colors.RED, err=True)
        for candidate in candidates:
            secho(f"    {get_plugin_package(candidate)}  ({candidate.value})", fg=colors.YELLOW, err=True)
        secho("Uninstall all but the one you want to use.", fg=colors.YELLOW, err=True)
        raise Exit(1)

    ep = candidates[0]
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

    description = get_plugin_description(cls)
    if description:
        secho("\n  Details:", fg=colors.BRIGHT_WHITE, bold=True)
        Console().print(Padding(Markdown(description), (0, 0, 0, 2)))

    if 'broker' in caps:
        from rich.table import Table
        from rich import box
        from ...core.broker.models import CapabilityLevel

        # ``get_capabilities`` is an instance method but well-behaved broker
        # plugins return a static dataclass with no I/O. Try a default
        # construction first; if the plugin requires its Config dataclass
        # (e.g. Capital.com asserts on it), retry with a default-constructed
        # Config. Fall back to a warning if even that fails — long-term fix
        # is to make ``get_capabilities`` a classmethod.
        ex_caps = None
        intro_error: Exception | None = None
        try:
            ex_caps = cls().get_capabilities()  # type: ignore[call-arg]
        except Exception as e1:
            intro_error = e1
            cfg_cls = getattr(cls, 'Config', None)
            if cfg_cls is not None and dataclasses.is_dataclass(cfg_cls):
                try:
                    ex_caps = cls(config=cfg_cls()).get_capabilities()  # type: ignore[call-arg]
                    intro_error = None
                except Exception as e2:
                    intro_error = e2
        if intro_error is not None:
            secho(
                f"\n  Exchange Capabilities:  (cannot introspect: {intro_error})",
                fg=colors.YELLOW,
            )
        else:
            assert ex_caps is not None
            secho("\n  Exchange Capabilities:", fg=colors.BRIGHT_WHITE, bold=True)
            level_color = {
                CapabilityLevel.NATIVE: "green",
                CapabilityLevel.PARTIAL_NATIVE: "yellow",
                CapabilityLevel.SOFTWARE: "cyan",
                CapabilityLevel.UNSUPPORTED: "dim white",
            }
            table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2, 0, 2))
            table.add_column("Capability", style="white")
            table.add_column("Level")
            # noinspection PyDataclass
            for f in dataclasses.fields(ex_caps):
                value = getattr(ex_caps, f.name)
                colour = level_color.get(value, "cyan")
                table.add_row(f.name, f"[{colour}]{value.value}[/{colour}]")
            Console().print(Padding(table, (0, 0, 0, 2)))

    config_cls: type | None = getattr(cls, 'Config', None)
    if config_cls is not None and dataclasses.is_dataclass(config_cls):
        # noinspection PyDataclass
        fields = dataclasses.fields(config_cls)
        if fields:
            secho(f"\n  Config fields (defaults):")
            for f in fields:
                default = f"= {f.default!r}" if f.default is not dataclasses.MISSING else "(required)"
                secho(f"    {f.name:20s} {default}")

    secho("")
