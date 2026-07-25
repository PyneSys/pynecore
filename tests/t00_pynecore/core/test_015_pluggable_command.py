"""Tests for the PluggableCommand CLI parameter injection system."""

import pytest
import typer
from typer.testing import CliRunner

from pynecore.core.plugin import CLIOption
from pynecore.cli.pluggable import PluggableCommand


@pytest.fixture(autouse=True)
def _reset_plugin_param_registry():
    """Isolate tests: the registry is class-level (survives Typer rebuilds), so it
    must be cleared between tests that reuse the same command leaf name."""
    PluggableCommand._plugin_param_registry.clear()
    yield
    PluggableCommand._plugin_param_registry.clear()


def _make_app():
    """Create a minimal Typer app with a PluggableCommand for testing."""
    test_app = typer.Typer()

    @test_app.command(cls=PluggableCommand)
    def greet(ctx: typer.Context, name: str = "world"):
        plugin_p = getattr(ctx, "plugin_params", {})
        greeting = plugin_p.get("greeting", "Hello")
        loud = plugin_p.get("loud", False)
        msg = f"{greeting} {name}"
        if loud:
            msg = msg.upper()
        typer.echo(msg)

    return test_app


def _get_command(test_app: typer.Typer) -> PluggableCommand:
    """Get the underlying PluggableCommand from a Typer app."""
    cmd = typer.main.get_command(test_app)
    assert isinstance(cmd, PluggableCommand)
    return cmd


def __test_command_type__():
    """PluggableCommand is used when cls= is passed to @app.command()."""
    app = _make_app()
    cmd = _get_command(app)
    assert isinstance(cmd, PluggableCommand)


def __test_no_plugin_params_default__():
    """Without registered plugin params, the command works normally."""
    app = _make_app()

    result = CliRunner().invoke(app, ["--name", "PyneCore"])
    assert result.exit_code == 0
    assert "Hello PyneCore" in result.output


def __test_register_option__():
    """A registered plugin option is parsed and available via ctx.plugin_params."""
    app = _make_app()
    cmd = _get_command(app)

    ok = cmd.register_plugin_param(
        CLIOption("--greeting", default="Hello", help="Greeting word"),
    )
    assert ok is True

    result = CliRunner().invoke(app, ["--greeting", "Ahoy", "--name", "Sailor"])
    assert result.exit_code == 0
    assert "Ahoy Sailor" in result.output


def __test_register_flag__():
    """A registered boolean flag works correctly."""
    app = _make_app()
    cmd = _get_command(app)

    cmd.register_plugin_param(
        CLIOption("--loud", is_flag=True, default=False, help="Shout"),
    )

    result = CliRunner().invoke(app, ["--loud", "--name", "test"])
    assert result.exit_code == 0
    assert "HELLO TEST" in result.output


def __test_default_values__():
    """Plugin params use their default when not provided on the command line."""
    app = _make_app()
    cmd = _get_command(app)

    cmd.register_plugin_param(
        CLIOption("--greeting", default="Hi", help="Greeting word"),
    )

    result = CliRunner().invoke(app, ["--name", "there"])
    assert result.exit_code == 0
    assert "Hi there" in result.output


def __test_typed_and_choice_params__():
    """A converter type and a choice set are both honoured."""
    received = {}

    test_app = typer.Typer()

    @test_app.command(cls=PluggableCommand)
    def show(ctx: typer.Context):
        received.update(getattr(ctx, "plugin_params", {}))

    cmd = _get_command(test_app)
    cmd.register_plugin_param(CLIOption("--count", type=int, default=1))
    cmd.register_plugin_param(CLIOption("--field", choices=("open", "close")))

    result = CliRunner().invoke(test_app, ["--count", "7", "--field", "close"])
    assert result.exit_code == 0
    assert received == {"count": 7, "field": "close"}

    bad = CliRunner().invoke(test_app, ["--field", "nonexistent"])
    assert bad.exit_code != 0


def __test_conflict_with_core_param__():
    """Registering a param that conflicts with a core param returns False."""
    app = _make_app()
    cmd = _get_command(app)

    ok = cmd.register_plugin_param(
        CLIOption("--name", default="x", help="Conflict"),
    )
    assert ok is False


def __test_conflict_between_plugins__():
    """Second registration of the same param name returns False."""
    app = _make_app()
    cmd = _get_command(app)

    ok1 = cmd.register_plugin_param(CLIOption("--extra", default="a"))
    ok2 = cmd.register_plugin_param(CLIOption("--extra", default="b"))
    assert ok1 is True
    assert ok2 is False


def __test_conflict_option_string__():
    """Option string conflict (e.g. --name vs --nickname/-n/--name) is detected."""
    test_app = typer.Typer()

    @test_app.command(cls=PluggableCommand)
    def cmd(
            time_from: str = typer.Option("", "--from", "-f"),
    ):
        typer.echo(time_from)

    plug_cmd = _get_command(test_app)

    ok = plug_cmd.register_plugin_param(CLIOption("--from", default="x"))
    assert ok is False

    ok2 = plug_cmd.register_plugin_param(CLIOption("-f", default="x"))
    assert ok2 is False

    ok3 = plug_cmd.register_plugin_param(CLIOption(("--other", "-o"), default="y"))
    assert ok3 is True


def __test_help_shows_plugin_params__():
    """Plugin params appear in --help output."""
    app = _make_app()
    cmd = _get_command(app)

    cmd.register_plugin_param(
        CLIOption("--live", is_flag=True, default=False, help="Enable live trading"),
    )

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--live" in result.output
    assert "Enable live trading" in result.output


def __test_plugin_params_not_passed_to_callback__():
    """Plugin params are NOT passed as kwargs to the callback function."""
    received_kwargs = {}

    test_app = typer.Typer()

    @test_app.command(cls=PluggableCommand)
    def strict(name: str = "x"):
        received_kwargs["name"] = name

    cmd = _get_command(test_app)
    cmd.register_plugin_param(CLIOption("--extra", default="val"))

    result = CliRunner().invoke(test_app, ["--extra", "test"])
    assert result.exit_code == 0
    assert "extra" not in received_kwargs


def __test_multiple_plugin_params__():
    """Multiple plugin params from different 'plugins' work together."""
    app = _make_app()
    cmd = _get_command(app)

    cmd.register_plugin_param(CLIOption("--greeting", default="Hello"))
    cmd.register_plugin_param(CLIOption("--loud", is_flag=True, default=False))

    result = CliRunner().invoke(app, ["--greeting", "YO", "--loud", "--name", "dev"])
    assert result.exit_code == 0
    assert "YO DEV" in result.output


def __test_get_params_includes_help__():
    """get_params always includes the --help option at the end."""
    app = _make_app()
    cmd = _get_command(app)

    cmd.register_plugin_param(CLIOption("--extra", default="x"))

    ctx = typer.Context(cmd)
    param_names = [p.name for p in cmd.get_params(ctx)]

    assert "extra" in param_names
    assert "help" in param_names


def __test_empty_decls_rejected__():
    """An option spec without any option string is a programming error."""
    with pytest.raises(ValueError):
        CLIOption(())
