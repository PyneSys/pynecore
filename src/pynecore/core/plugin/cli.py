from __future__ import annotations

from typing import TYPE_CHECKING

from . import Plugin

if TYPE_CHECKING:
    import click
    import typer


class CLIPlugin(Plugin):
    """
    Plugin that provides CLI commands and/or parameter hooks.

    Override :meth:`cli` to add subcommands (``pyne <name> ...``).
    Override :meth:`cli_params` to inject flags into existing commands.
    """

    @staticmethod
    def cli() -> typer.Typer | None:
        """
        Return a Typer app for plugin subcommands.

        Override to add commands like ``pyne <plugin_name> <subcommand>``.

        :return: A Typer app, or ``None`` if the plugin has no CLI commands.
        """
        return None

    # noinspection PyUnusedLocal
    @staticmethod
    def cli_params(command_name: str) -> list[click.Parameter]:
        """
        Return extra parameters for an existing command.

        Override to inject flags/options into commands like ``pyne run``.

        :param command_name: The command to extend (e.g. ``"run"``).
        :return: List of Click parameters, or ``[]`` if no hooks for this command.
        """
        return []
