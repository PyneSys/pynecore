import sys
from pathlib import Path

import typer

from ..app import app, app_state
from ..utils.error_hook import setup_global_error_logging

from ...providers import available_providers

# Import commands
from . import run, data, compile, benchmark

__all__ = ['run', 'data', 'compile', 'benchmark']


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
        from datetime import datetime, timedelta, time as dt_time
        from ...core.ohlcv_file import OHLCVWriter
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

            # Generate synthetic OHLCV data (2000 candles) with random walk
            start_time = datetime(2020, 1, 1, 0, 0, 0)
            base_price = 100.0

            # Use fixed seed for reproducibility
            random.seed(42)

            with OHLCVWriter(demo_file) as writer:
                current_price = base_price

                for i in range(20000):
                    timestamp = int((start_time + timedelta(days=i)).timestamp())

                    # Random walk with slight upward bias
                    change_percent = random.gauss(0.0002, 0.02)  # 0.02% mean, 2% std dev
                    current_price *= (1 + change_percent)

                    # Generate OHLC with some volatility
                    daily_volatility = random.uniform(0.01, 0.03)

                    open_price = current_price
                    high_price = open_price * (1 + daily_volatility * random.random())
                    low_price = open_price * (1 - daily_volatility * random.random())
                    close_price = low_price + (high_price - low_price) * random.random()

                    # Update current price for next candle
                    current_price = close_price

                    # Volume with some randomness
                    volume = 1000000 * (1 + random.uniform(-0.5, 1.0))

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

    # Create config directory
    config_dir = Path(workdir) / 'config'
    config_dir.mkdir(exist_ok=True)

    # Create providers.toml file for all supported providers (if not exists)
    providers_file = config_dir / 'providers.toml'
    if not providers_file.exists() or recreate_provider_config:
        with providers_file.open('w') as f:
            for provider in available_providers:
                f.write(f"[{provider}]\n")
                provider_module = __import__(f"pynecore.providers.{provider}", fromlist=[''])
                provider_class = getattr(
                    provider_module,
                    [p for p in dir(provider_module) if p.endswith('Provider')][0]
                )
                for key, value in provider_class.config_keys.items():
                    if key.startswith('#'):  # Comments
                        f.write(f'{key}\n')
                    else:
                        if isinstance(value, str):
                            f.write(f'{key} = "{value}"\n')
                        elif isinstance(value, bool):
                            f.write(f'{key} = {str(value).lower()}\n')
                        elif isinstance(value, int) or isinstance(value, float):
                            f.write(f'{key} = {value}\n')
                        else:
                            raise ValueError(f"Unsupported type for {key}: {type(value)}")
                f.write("\n")

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
