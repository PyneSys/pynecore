import logging
from pathlib import Path

import typer
from typer.core import TyperGroup

from ..app import app, app_state
from ..pluggable import PluggableCommand
from ..utils.error_hook import setup_global_error_logging

# Import commands
from . import run, data, compile, benchmark, debug, plugin

__all__ = ['run', 'data', 'compile', 'benchmark', 'debug', 'plugin']

logger = logging.getLogger(__name__)


@app.callback()
def setup(
        ctx: typer.Context,
        workdir: Path = typer.Option(
            app_state.workdir,
            "--workdir", "-w",
            envvar="PYNE_WORK_DIR",
            help="Working directory",
            file_okay=False, dir_okay=True,
            resolve_path=True,
        ),
        recreate_demo: bool = typer.Option(
            False,
            "--recreate-demo",
            help="Recreate demo.py and demo.ohlcv files even if workdir exists",
        ),
        recreate_provider_config: bool = typer.Option(
            False,
            "--recreate-provider-config",
            help="Recreate provider.toml file even if workdir exists",
        ),
        recreate_api_config: bool = typer.Option(
            False,
            "--recreate-api-config",
            help="Recreate api.toml file even if workdir exists",
        )
):
    """
    Pyne Command Line Interface
    """
    if ctx.resilient_parsing:
        return

    # If no subcommand is provided, show complete help like --help
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)

    typer.echo("")

    # Check if workdir is available
    workdir_existed = Path(workdir).exists()
    if not workdir_existed:
        typer.echo(f"Working directory '{workdir}' does not exist.")
        typer.confirm("Do you want to create it?", abort=True)

        # Create workdir
        Path(workdir).mkdir(parents=True, exist_ok=False)

    # Create the .pyne marker file so the directory is recognized as a workdir by content
    marker_file = Path(workdir) / '.pyne'
    if not marker_file.exists():
        marker_file.write_text(
            "# PyneCore workdir marker. The version refers to the workdir structure, "
            "not your project.\n"
            'workdir_version = "1.0"\n'
        )

    # Create scripts directory
    scripts_dir = Path(workdir) / 'scripts' / 'lib'
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # Create demo.py file only if we created the workdir in this run or recreate_demo is True
    if not workdir_existed or recreate_demo:
        demo_file = Path(workdir) / 'scripts' / 'demo.py'
        if not demo_file.exists() or recreate_demo:
            if recreate_demo and demo_file.exists():
                typer.echo("Recreating demo.py...")
            with demo_file.open('w') as f:
                f.write('''"""
@pyne
Simple Pyne code Demo

A basic demo showing a 12 and 26 period EMA crossover system.
"""
from pynecore import Series
from pynecore.lib import script, input, plot, color, ta


@script.indicator(
    title="Simple EMA Crossover Demo",
    shorttitle="EMA Demo",
    overlay=True
)
def main(
    src: Series[float] = input.source("close", title="Price Source"),
    fast_length: int = input.int(12, title="Fast EMA Length"),
    slow_length: int = input.int(26, title="Slow EMA Length")
):
    """
    A simple EMA crossover demo
    """
    # Calculate EMAs
    fast_ema = ta.ema(src, fast_length)
    slow_ema = ta.ema(src, slow_length)

    # Plot our indicators
    plot(fast_ema, title="Fast EMA", color=color.blue)
    plot(slow_ema, title="Slow EMA", color=color.red)
''')

    # Create data directory
    data_dir = Path(workdir) / 'data'
    data_dir.mkdir(exist_ok=True)

    # Create demo.ohlcv file only if we created the workdir in this run or recreate_demo is True
    if not workdir_existed or recreate_demo:
        from datetime import datetime, timedelta, time as dt_time, UTC
        from ...core.ohlcv import OHLCVWriter
        from ...core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
        from ...types.ohlcv import OHLCV
        import random

        demo_file = data_dir / 'demo.ohlcv'
        if not demo_file.exists() or recreate_demo:
            if recreate_demo and demo_file.exists():
                typer.echo("Recreating demo.ohlcv and demo.toml...")
                # Remove existing files to start fresh
                demo_file.unlink(missing_ok=True)
                toml_path = demo_file.with_suffix(".toml")
                toml_path.unlink(missing_ok=True)
            # Create opening hours, session starts and ends for 24/7 trading
            opening_hours = []
            session_starts = []
            session_ends = []

            for day in range(7):  # 0 = Monday, 6 = Sunday
                opening_hours.append(SymInfoInterval(
                    day=day,
                    start=dt_time(0, 0, 0),
                    end=dt_time(23, 59, 59)
                ))
                session_starts.append(SymInfoSession(
                    day=day,
                    time=dt_time(0, 0, 0)
                ))
                session_ends.append(SymInfoSession(
                    day=day,
                    time=dt_time(23, 59, 59)
                ))

            # Create symbol info
            syminfo = SymInfo(
                prefix="DEMO",
                description="Demo Benchmark Asset",
                ticker="DEMOBTC",
                currency="BTC",
                basecurrency="DEMO",
                period="1D",
                type="crypto",
                volumetype="base",
                mintick=0.01,
                pricescale=100,
                minmove=1,
                pointvalue=1.0,
                mincontract=0.0001,
                opening_hours=opening_hours,
                session_starts=session_starts,
                session_ends=session_ends,
                timezone="UTC",
                avg_spread=None,
                taker_fee=0.001,
                maker_fee=0.0005
            )

            # Save symbol info
            toml_path = demo_file.with_suffix(".toml")
            syminfo.save_toml(toml_path)

            # Generate synthetic OHLCV data (2000 candles) with random walk.
            # The anchor is UTC so the daily grid is an exact 86400000 ms step
            # on every machine, matching the "1D" period declared below.
            start_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=UTC)
            base_price = 100.0
            base_volatility = 0.015  # Long-run standard deviation of the daily return

            # Use fixed seed for reproducibility
            random.seed(42)

            with OHLCVWriter(demo_file, syminfo.period,
                             minmove=syminfo.minmove, pricescale=syminfo.pricescale) as writer:
                close_price = base_price
                volatility = base_volatility
                bar_return = 0.0
                trend = 0.0

                for i in range(2000):
                    bar_time = start_time + timedelta(days=i)
                    # OHLCV timestamps are Unix milliseconds.
                    timestamp = int(bar_time.timestamp() * 1000)

                    # Volatility clusters: it decays back towards the base level but is
                    # pushed up by the previous move, so calm and turbulent stretches alternate
                    volatility = min(0.85 * volatility + 0.07 * base_volatility
                                     + 0.10 * abs(bar_return), 4 * base_volatility)
                    # Slowly drifting trend component, giving multi-month up and down legs
                    trend = 0.98 * trend + random.gauss(0.0, 0.0006)

                    # A 24/7 asset normally opens where the previous bar closed,
                    # a gap is a rare event
                    open_price = close_price
                    if random.random() < 0.02:
                        open_price *= 1 + random.gauss(0.0, 0.012)

                    # The move happens inside the bar, between its open and close
                    bar_return = random.gauss(0.0012 + trend, volatility)
                    close_price = open_price * (1 + bar_return)

                    # Wicks extend beyond the body, so the range always contains it
                    body_high = max(open_price, close_price)
                    body_low = min(open_price, close_price)
                    high_price = body_high * (1 + abs(random.gauss(0.0, 0.5)) * volatility)
                    low_price = body_low * (1 - abs(random.gauss(0.0, 0.5)) * volatility)

                    # Prices sit on the mintick grid declared in the .toml. Rounding is
                    # monotonic, so it cannot invalidate the OHLC ordering.
                    open_price = round(open_price / syminfo.mintick) * syminfo.mintick
                    high_price = round(high_price / syminfo.mintick) * syminfo.mintick
                    low_price = round(low_price / syminfo.mintick) * syminfo.mintick
                    close_price = round(close_price / syminfo.mintick) * syminfo.mintick

                    # Wide-range bars trade more, weekends are quieter even on a 24/7 market
                    bar_range = (high_price - low_price) / open_price
                    volume = 1_000_000 * (0.5 + 35 * bar_range) * random.uniform(0.75, 1.25)
                    if bar_time.weekday() >= 5:
                        volume *= 0.7

                    ohlcv = OHLCV(
                        timestamp=timestamp,
                        open=open_price,
                        high=high_price,
                        low=low_price,
                        close=close_price,
                        volume=volume,
                        extra_fields=None
                    )
                    writer.write(ohlcv)
    # Create output and logs directory
    output_dir = Path(workdir) / 'output' / 'logs'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create cache directory (machine-written runtime state: auth sessions, etc.)
    cache_dir = Path(workdir) / 'cache'
    cache_dir.mkdir(exist_ok=True)

    # Create config directory
    config_dir = Path(workdir) / 'config'
    config_dir.mkdir(exist_ok=True)

    # Generate per-plugin config files for all installed plugins
    from ...core.plugin import discover_plugins
    from ...core.config import ensure_config

    plugins_dir = config_dir / 'plugins'
    plugins_dir.mkdir(exist_ok=True)

    for name, ep in discover_plugins().items():
        config_path = plugins_dir / f'{name}.toml'
        if not config_path.exists() or recreate_provider_config:
            try:
                plugin_cls = ep.load()
                if hasattr(plugin_cls, 'Config') and plugin_cls.Config is not None:
                    ensure_config(plugin_cls.Config, config_path)
            except Exception as e:
                logger.warning("Failed to load plugin config '%s': %s", name, e)

    # Create api.toml file for PyneSys API (if not exists)
    api_file = config_dir / 'api.toml'
    if not api_file.exists() or recreate_api_config:
        with api_file.open('w') as f:
            f.write("""[api]
api_key = ""
timeout = 30
""")

    # Set workdir in app_state
    app_state.workdir = workdir

    # Setup global error logging
    setup_global_error_logging(workdir / "output" / "logs" / "error.log")


# ---------------------------------------------------------------------------
# CLIPlugin loading: subcommands and parameter hooks
# ---------------------------------------------------------------------------
_BUILTIN_COMMANDS = {'run', 'data', 'compile', 'benchmark', 'debug', 'plugin'}
# Commands plugins may inject parameters into, as space-separated paths so a
# nested subcommand (e.g. ``data download``) can be reached as well.
_PLUGGABLE_COMMANDS = ('run', 'data download')


def _resolve_command(group: TyperGroup, path: str) -> PluggableCommand | None:
    """Walk a space-separated command path down to a pluggable command."""
    parts = path.split()
    node = group
    for part in parts[:-1]:
        child = node.commands.get(part)
        if not isinstance(child, TyperGroup):
            return None
        node = child
    leaf = node.commands.get(parts[-1])
    return leaf if isinstance(leaf, PluggableCommand) else None


def _register_cli_plugins():
    """Load CLIPlugin subcommands and parameter hooks from installed plugins."""
    from ...core.plugin import discover_plugins, CLIPlugin

    for name, ep in discover_plugins().items():
        try:
            plugin_cls = ep.load()

            if not (isinstance(plugin_cls, type) and issubclass(plugin_cls, CLIPlugin)):
                continue

            # 1. CLI subcommand registration
            cli_app = plugin_cls.cli()
            if cli_app is not None:
                if name in _BUILTIN_COMMANDS:
                    typer.secho(
                        f"Warning: plugin '{name}' CLI name conflicts with "
                        f"built-in command, skipping",
                        fg="yellow", err=True,
                    )
                else:
                    app.add_typer(cli_app, name=name)

            # 2. Parameter hook registration
            for cmd_path in _PLUGGABLE_COMMANDS:
                params = plugin_cls.cli_params(cmd_path)
                if not params:
                    continue

                group = typer.main.get_command(app)
                assert isinstance(group, TyperGroup)
                command = _resolve_command(group, cmd_path)
                if command is None:
                    continue

                for param in params:
                    if not command.register_plugin_param(param):
                        typer.secho(
                            f"Warning: plugin '{name}' param '{param.decls[0]}' "
                            f"conflicts on '{cmd_path}'",
                            fg="yellow", err=True,
                        )

        except Exception as e:
            logger.warning("Failed to load CLI plugin '%s': %s", name, e)


_register_cli_plugins()
