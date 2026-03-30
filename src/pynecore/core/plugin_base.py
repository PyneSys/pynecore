"""
Base class for all PyneCore plugins.

Every plugin inherits from :class:`Plugin` (directly or via :class:`Provider`,
:class:`Extension`, etc.).  The class hierarchy determines capabilities:

- ``Provider(Plugin)`` — offline OHLCV data provider
- ``Extension(Plugin)`` — hook-based script extension
- ``LiveProvider(Plugin)`` — WebSocket/streaming data
- ``Plugin`` directly — CLI-only plugin

CLI methods (``cli()``, ``cli_params()``) are optional with sensible defaults.
Plugin metadata (name, version) comes from the package's ``pyproject.toml``
via :mod:`importlib.metadata`, not from class attributes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import click
    import typer


class Plugin:
    """Base class for all PyneCore plugins."""

    Config: type | None = None
    """Override with a ``@dataclass`` for plugin configuration."""

    plugin_name: str = ""
    """Optional display name override.  If empty, the entry point name is used."""

    @staticmethod
    def cli() -> typer.Typer | None:
        """
        Return a Typer app for plugin subcommands.

        Override to add commands like ``pyne <plugin_name> <subcommand>``.
        Return ``None`` (default) if the plugin has no CLI commands.
        """
        return None

    @staticmethod
    def cli_params(command_name: str) -> list[click.Parameter]:
        """
        Return extra parameters for an existing command.

        Override to inject flags/options into commands like ``pyne run``.
        Return ``[]`` (default) if the plugin has no parameter hooks.

        :param command_name: The command to extend (e.g. ``"run"``).
        """
        return []
