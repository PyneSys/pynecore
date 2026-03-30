"""
Plugin discovery and loading via Python entry points.

All PyneCore plugins register under a single entry point group
(``pyne.plugin``) in their ``pyproject.toml``.  The plugin class hierarchy
determines capabilities (Provider, Extension, etc.).

Example ``pyproject.toml`` for a provider plugin::

    [project.entry-points."pyne.plugin"]
    myexchange = "mypackage:MyExchangeProvider"

Discovery::

    plugins = discover_plugins()
    cls = load_plugin("capitalcom")

Metadata (name, version, description) is read from the package's
``pyproject.toml`` via :mod:`importlib.metadata`, not from class attributes.
"""

import re

# noinspection PyProtectedMember
from importlib.metadata import entry_points, EntryPoint

PLUGIN_GROUP = 'pyne.plugin'


class PluginNotFoundError(ImportError):
    """Raised when a requested plugin is not installed."""


def discover_plugins() -> dict[str, EntryPoint]:
    """
    Return all installed plugins.

    :return: Mapping of plugin name to its :class:`EntryPoint`.
    """
    return {ep.name: ep for ep in entry_points(group=PLUGIN_GROUP)}


def load_plugin(name: str) -> type:
    """
    Load and return a plugin class by name.

    :param name: Plugin name as declared in the entry point.
    :return: The plugin class.
    :raises PluginNotFoundError: If no plugin with the given name is installed.
    """
    eps = discover_plugins()
    if name not in eps:
        raise PluginNotFoundError(
            f"Plugin '{name}' not found. "
            f"Install it with: pip install pynesys-pynecore-{name}  (official) "
            f"or: pip install pynecore-{name}  (3rd party)\n"
            f"Available plugins: {', '.join(sorted(eps)) or '(none)'}"
        )
    return eps[name].load()


def get_available_plugin_names() -> list[str]:
    """
    Return a sorted list of all available plugin names.

    :return: Sorted list of plugin names.
    """
    return sorted(discover_plugins())


def get_plugin_metadata(ep: EntryPoint) -> dict[str, str]:
    """
    Extract plugin metadata from its package distribution.

    :param ep: The entry point of the plugin.
    :return: Dict with ``name``, ``version``, ``description``, ``min_pynecore``.
    """
    meta = ep.dist.metadata
    return {
        'name': ep.name,
        'package': meta['Name'] or '',
        'version': meta['Version'] or '',
        'description': meta['Summary'] or '',
        'min_pynecore': _parse_min_pynecore(ep),
    }


def _parse_min_pynecore(ep: EntryPoint) -> str:
    """
    Extract the minimum PyneCore version from the package dependencies.

    Parses ``pynesys-pynecore>=X.Y`` from the ``Requires-Dist`` list.

    :param ep: The entry point of the plugin.
    :return: Version string (e.g. ``"6.5"``) or ``""`` if not found.
    """
    requires = ep.dist.requires
    if not requires:
        return ''
    for req in requires:
        m = re.match(r'pynesys-pynecore(?:\[.*?])?>=([.\d]+)', req)
        if m:
            return m.group(1)
    return ''
