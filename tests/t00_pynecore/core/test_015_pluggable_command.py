"""Tests for the PluggableCommand CLI parameter injection system."""

import click
import typer
from click.testing import CliRunner

from pynecore.cli.pluggable import PluggableCommand


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


def _get_click_cmd(test_app: typer.Typer) -> PluggableCommand:
    """Get the underlying PluggableCommand from a Typer app."""
    return typer.main.get_command(test_app)


def __test_command_type__():
    """PluggableCommand is used when cls= is passed to @app.command()."""
    app = _make_app()
    cmd = _get_click_cmd(app)
    assert isinstance(cmd, PluggableCommand)


def __test_no_plugin_params_default__():
    """Without registered plugin params, the command works normally."""
    app = _make_app()
    cmd = _get_click_cmd(app)

    result = CliRunner().invoke(cmd, ["--name", "PyneCore"])
    assert result.exit_code == 0
    assert "Hello PyneCore" in result.output


def __test_register_option__():
    """A registered plugin option is parsed and available via ctx.plugin_params."""
    app = _make_app()
    cmd = _get_click_cmd(app)

    ok = cmd.register_plugin_param(
        click.Option(["--greeting"], default="Hello", help="Greeting word"),
    )
    assert ok is True

    result = CliRunner().invoke(cmd, ["--greeting", "Ahoy", "--name", "Sailor"])
    assert result.exit_code == 0
    assert "Ahoy Sailor" in result.output


def __test_register_flag__():
    """A registered boolean flag works correctly."""
    app = _make_app()
    cmd = _get_click_cmd(app)

    cmd.register_plugin_param(
        click.Option(["--loud"], is_flag=True, default=False, help="Shout"),
    )

    result = CliRunner().invoke(cmd, ["--loud", "--name", "test"])
    assert result.exit_code == 0
    assert "HELLO TEST" in result.output


def __test_default_values__():
    """Plugin params use their default when not provided on the command line."""
    app = _make_app()
    cmd = _get_click_cmd(app)

    cmd.register_plugin_param(
        click.Option(["--greeting"], default="Hi", help="Greeting word"),
    )

    result = CliRunner().invoke(cmd, ["--name", "there"])
    assert result.exit_code == 0
    assert "Hi there" in result.output


def __test_conflict_with_core_param__():
    """Registering a param that conflicts with a core param returns False."""
    app = _make_app()
    cmd = _get_click_cmd(app)

    ok = cmd.register_plugin_param(
        click.Option(["--name"], default="x", help="Conflict"),
    )
    assert ok is False


def __test_conflict_between_plugins__():
    """Second registration of the same param name returns False."""
    app = _make_app()
    cmd = _get_click_cmd(app)

    ok1 = cmd.register_plugin_param(
        click.Option(["--extra"], default="a"),
    )
    ok2 = cmd.register_plugin_param(
        click.Option(["--extra"], default="b"),
    )
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

    plug_cmd = _get_click_cmd(test_app)

    ok = plug_cmd.register_plugin_param(
        click.Option(["--from"], default="x"),
    )
    assert ok is False

    ok2 = plug_cmd.register_plugin_param(
        click.Option(["-f"], default="x"),
    )
    assert ok2 is False

    ok3 = plug_cmd.register_plugin_param(
        click.Option(["--other", "-o"], default="y"),
    )
    assert ok3 is True


def __test_help_shows_plugin_params__():
    """Plugin params appear in --help output."""
    app = _make_app()
    cmd = _get_click_cmd(app)

    cmd.register_plugin_param(
        click.Option(["--live"], is_flag=True, default=False, help="Enable live trading"),
    )

    result = CliRunner().invoke(cmd, ["--help"])
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

    cmd = _get_click_cmd(test_app)
    cmd.register_plugin_param(
        click.Option(["--extra"], default="val"),
    )

    result = CliRunner().invoke(cmd, ["--extra", "test"])
    assert result.exit_code == 0
    assert "extra" not in received_kwargs


def __test_multiple_plugin_params__():
    """Multiple plugin params from different 'plugins' work together."""
    app = _make_app()
    cmd = _get_click_cmd(app)

    cmd.register_plugin_param(
        click.Option(["--greeting"], default="Hello"),
    )
    cmd.register_plugin_param(
        click.Option(["--loud"], is_flag=True, default=False),
    )

    result = CliRunner().invoke(cmd, ["--greeting", "YO", "--loud", "--name", "dev"])
    assert result.exit_code == 0
    assert "YO DEV" in result.output


def __test_get_params_includes_help__():
    """get_params always includes the --help option at the end."""
    app = _make_app()
    cmd = _get_click_cmd(app)

    cmd.register_plugin_param(click.Option(["--extra"], default="x"))

    ctx = click.Context(cmd)
    params = cmd.get_params(ctx)
    param_names = [p.name for p in params]

    assert "extra" in param_names
    assert "help" in param_names
