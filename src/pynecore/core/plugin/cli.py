from typing import TYPE_CHECKING, Any
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from . import Plugin, ConfigT

if TYPE_CHECKING:
    import typer


@dataclass(slots=True)
class CLIOption:
    """
    Declarative description of a CLI option injected into a built-in command.

    Deliberately backend-agnostic: the plugin describes the option, PyneCore
    builds the concrete parser object. Plugins must not construct parser
    objects themselves — Click is not a PyneCore dependency, and since Typer
    0.26 it is not a Typer dependency either (Typer vendors a reduced fork of
    it), so an externally built ``click.Option`` is not even parseable by the
    CLI anymore.

    :ivar decls: Option string, or a tuple of them (``"--live"``,
        ``("--output", "-o")``). The parameter name is derived from the first
        long option, as usual.
    :ivar help: Help text shown in ``--help``.
    :ivar default: Value used when the option is not given on the command line.
    :ivar is_flag: Boolean switch that takes no value.
    :ivar type: Converter applied to the raw string value (``int``, ``float``,
        ``pathlib.Path``, or any single-argument callable). Ignored when
        ``choices`` is set.
    :ivar choices: Restrict the value to this set.
    :ivar metavar: Value placeholder in the help output.
    :ivar required: Fail when the option is missing.
    :ivar multiple: Allow the option to be repeated, collecting a tuple.
    :ivar hidden: Keep the option out of the help output.
    :ivar envvar: Environment variable to read the value from.
    :ivar rich_help_panel: Help panel to group the option under.
    """

    decls: str | tuple[str, ...]
    help: str = ""
    default: Any = None
    is_flag: bool = False
    type: Callable[[str], Any] | None = None
    choices: Sequence[str] | None = None
    metavar: str | None = None
    required: bool = False
    multiple: bool = False
    hidden: bool = False
    envvar: str | None = None
    rich_help_panel: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.decls, str):
            self.decls = (self.decls,)
        else:
            self.decls = tuple(self.decls)
        if not self.decls:
            raise ValueError("CLIOption needs at least one option string")


class CLIPlugin(Plugin[ConfigT]):
    """
    Plugin that provides CLI commands and/or parameter hooks.

    Override :meth:`cli` to add subcommands (``pyne <name> ...``).
    Override :meth:`cli_params` to inject flags into existing commands.
    """

    @staticmethod
    def cli() -> 'typer.Typer | None':
        """
        Return a Typer app for plugin subcommands.

        Override to add commands like ``pyne <plugin_name> <subcommand>``.

        :return: A Typer app, or ``None`` if the plugin has no CLI commands.
        """
        return None

    # noinspection PyUnusedLocal
    @staticmethod
    def cli_params(command_name: str) -> list[CLIOption]:
        """
        Return extra parameters for an existing command.

        Override to inject flags/options into commands like ``pyne run``.
        Nested subcommands are addressed by their space-separated path, so
        ``pyne data download`` is matched as ``"data download"``.

        :param command_name: The command to extend (e.g. ``"run"`` or
            ``"data download"``).
        :return: List of option specs, or ``[]`` if no hooks for this command.
        """
        return []
