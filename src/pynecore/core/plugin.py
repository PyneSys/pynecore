"""
Plugin discovery and loading via Python entry points.

All PyneCore plugins (providers, extensions, CLI commands) are discovered
through :pep:`631` entry points declared in ``pyproject.toml``.  This module
provides a thin, general-purpose API over :mod:`importlib.metadata`.

Example ``pyproject.toml`` for a provider plugin::

    [project.entry-points."pyne.provider"]
    myexchange = "mypackage:MyExchangeProvider"

Discovery example::

    plugins = discover_plugins("pyne.provider")
    # {"ccxt": <EntryPoint ...>, "myexchange": <EntryPoint ...>}

    cls = load_plugin("pyne.provider", "ccxt")
    # <class 'pynecore.providers.ccxt.CCXTProvider'>
"""

# noinspection PyProtectedMember
from importlib.metadata import entry_points, EntryPoint


class PluginNotFoundError(ImportError):
    """Raised when a requested plugin is not installed."""


def discover_plugins(group: str) -> dict[str, EntryPoint]:
    """
    Return all installed entry points for a plugin group.

    :param group: Entry point group name (e.g. ``"pyne.provider"``).
    :return: Mapping of plugin name to its :class:`EntryPoint`.
    """
    return {ep.name: ep for ep in entry_points(group=group)}


def load_plugin(group: str, name: str) -> type:
    """
    Load and return a plugin class by name.

    The actual import happens lazily — only when this function is called.

    :param group: Entry point group name (e.g. ``"pyne.provider"``).
    :param name: Plugin name as declared in the entry point.
    :return: The plugin class.
    :raises PluginNotFoundError: If no plugin with the given name is installed.
    """
    eps = discover_plugins(group)
    if name not in eps:
        short_group = group.replace("pyne.", "")
        raise PluginNotFoundError(
            f"Plugin '{name}' not found for group '{group}'. "
            f"Install it with: pip install pynecore-{name}\n"
            f"Available {short_group} plugins: {', '.join(sorted(eps)) or '(none)'}"
        )
    return eps[name].load()


def get_available_plugin_names(group: str) -> list[str]:
    """
    Return a sorted list of all available plugin names for a group.

    :param group: Entry point group name (e.g. ``"pyne.provider"``).
    :return: Sorted list of plugin names.
    """
    return sorted(discover_plugins(group))
