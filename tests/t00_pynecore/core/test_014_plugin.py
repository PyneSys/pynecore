"""
Tests for the plugin discovery and loading system.
"""

from pynecore.core.plugin import (
    discover_plugins,
    load_plugin,
    get_available_plugin_names,
    PluginNotFoundError,
)


def __test_discover_plugins_returns_dict__():
    """discover_plugins returns a dict (possibly empty for unknown group)"""
    result = discover_plugins("pyne.nonexistent_group_for_testing")
    assert isinstance(result, dict)
    assert len(result) == 0


def __test_discover_providers__():
    """discover_plugins finds CCXT provider via entry_points"""
    result = discover_plugins("pyne.provider")
    assert "ccxt" in result, (
        f"CCXT not found in pyne.provider entry points. "
        f"Available: {list(result.keys())}. "
        f"Make sure pynecore is installed with: pip install -e pynecore/"
    )


def __test_load_plugin_ccxt__():
    """load_plugin loads the CCXTProvider class"""
    from pynecore.providers.ccxt import CCXTProvider

    cls = load_plugin("pyne.provider", "ccxt")
    assert cls is CCXTProvider


def __test_load_plugin_not_found__():
    """load_plugin raises PluginNotFoundError for missing plugin"""
    try:
        load_plugin("pyne.provider", "nonexistent_provider_xyz")
        assert False, "Should have raised PluginNotFoundError"
    except PluginNotFoundError as e:
        assert "nonexistent_provider_xyz" in str(e)
        assert "pip install" in str(e)


def __test_get_available_plugin_names__():
    """get_available_plugin_names returns sorted list including ccxt"""
    names = get_available_plugin_names("pyne.provider")
    assert isinstance(names, list)
    assert "ccxt" in names
    assert names == sorted(names)


def __test_get_available_plugin_names_empty_group__():
    """get_available_plugin_names returns empty list for unknown group"""
    names = get_available_plugin_names("pyne.nonexistent_group_for_testing")
    assert names == []
