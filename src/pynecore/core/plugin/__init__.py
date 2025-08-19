"""Plugin system for PyneCore

This module provides plugin discovery, registration, and management capabilities
that can be used by any PyneCore application, not just the CLI.
"""

from .plugin_manager import PluginManager, PluginInfo, plugin_manager

__all__ = ['PluginManager', 'PluginInfo', 'plugin_manager']