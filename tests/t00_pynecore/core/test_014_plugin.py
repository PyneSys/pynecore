"""
Tests for the plugin discovery and loading system.
"""

from pynecore.core.plugin import (
    discover_plugins,
    load_plugin,
    get_available_plugin_names,
    get_plugin_metadata,
    PluginNotFoundError,
)
from pynecore.core.plugin import Plugin
from pynecore.providers.provider import Provider


def __test_discover_plugins_returns_dict__():
    """discover_plugins returns a dict of installed plugins"""
    result = discover_plugins()
    assert isinstance(result, dict)


def __test_discover_ccxt__():
    """discover_plugins finds CCXT provider via pyne.plugin entry point"""
    result = discover_plugins()
    assert "ccxt" in result, (
        f"CCXT not found in pyne.plugin entry points. "
        f"Available: {list(result.keys())}. "
        f"Make sure pynecore is installed with: pip install -e pynecore/"
    )


def __test_load_plugin_ccxt__():
    """load_plugin loads the CCXTProvider class"""
    from pynecore.providers.ccxt import CCXTProvider

    cls = load_plugin("ccxt")
    assert cls is CCXTProvider


def __test_load_plugin_not_found__():
    """load_plugin raises PluginNotFoundError for missing plugin"""
    try:
        load_plugin("nonexistent_provider_xyz")
        assert False, "Should have raised PluginNotFoundError"
    except PluginNotFoundError as e:
        assert "nonexistent_provider_xyz" in str(e)
        assert "pip install" in str(e)


def __test_get_available_plugin_names__():
    """get_available_plugin_names returns sorted list including ccxt"""
    names = get_available_plugin_names()
    assert isinstance(names, list)
    assert "ccxt" in names
    assert names == sorted(names)


def __test_ccxt_is_provider__():
    """CCXTProvider inherits from Plugin and Provider"""
    cls = load_plugin("ccxt")
    assert issubclass(cls, Plugin)
    assert issubclass(cls, Provider)


def __test_plugin_metadata__():
    """get_plugin_metadata extracts metadata from pyproject.toml"""
    plugins = discover_plugins()
    assert "ccxt" in plugins
    meta = get_plugin_metadata(plugins["ccxt"])
    assert meta['name'] == 'ccxt'
    assert meta['version']  # should be non-empty
    assert meta['package'] == 'pynesys-pynecore'


def __test_plugin_base_defaults__():
    """Plugin base class has sensible defaults for CLI methods"""
    assert Plugin.cli() is None
    assert Plugin.cli_params('run') == []
    assert Plugin.Config is None
