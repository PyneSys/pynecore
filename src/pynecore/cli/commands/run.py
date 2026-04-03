import os
import queue
import threading
import time
import sys
import tomllib

from pathlib import Path
from datetime import datetime, timedelta, UTC

from typer import Option, Argument, secho, Exit, colors
from rich.progress import (Progress, SpinnerColumn, TextColumn, BarColumn,
                           ProgressColumn, Task, TimeElapsedColumn, TimeRemainingColumn)
from rich.text import Text
from rich.console import Console

from ..app import app, app_state
from ..pluggable import PluggableCommand

from ...utils.rich.date_column import DateColumn
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.data_converter import DataConverter, DataFormatError, ConversionError

from pynecore.core.syminfo import SymInfo
from pynecore.core.script_runner import ScriptRunner
from pynecore.pynesys.compiler import PyneComp
from pynecore.core.provider_string import is_provider_string, parse_provider_string
from ...cli.utils.api_error_handler import APIErrorHandler

__all__ = []

console = Console()


class CustomTimeElapsedColumn(ProgressColumn):
    """Custom time elapsed column showing milliseconds."""

    def render(self, task: Task) -> Text:
        """Render the time elapsed with milliseconds."""
        elapsed = task.elapsed
        if elapsed is None:
            return Text("--:--.-", style="cyan")

        minutes = int(elapsed // 60)
        seconds = elapsed % 60

        return Text(f"{minutes:02d}:{seconds:06.3f}", style="cyan")


class CustomTimeRemainingColumn(ProgressColumn):
    """Custom time remaining column showing milliseconds."""

    def render(self, task: Task) -> Text:
        """Render the time remaining with milliseconds."""
        remaining = task.time_remaining
        if remaining is None:
            return Text("--:--.-", style="cyan")

        minutes = int(remaining // 60)
        seconds = remaining % 60

        return Text(f"{minutes:02d}:{seconds:06.3f}", style="cyan")


def _parse_time_value(value: str | None, *, allow_bars: bool = False) -> datetime | int | None:
    """
    Parse a --from or --to parameter value.

    :param value: The raw string value.
    :param allow_bars: If True, allow negative numbers as bar counts.
    :return: A datetime, a negative int (bar count), or None.
    """
    if value is None:
        return None
    value = value.strip()

    # Negative number = bar count (only for --from in provider mode)
    if allow_bars and value.startswith('-'):
        try:
            bars = int(value)
            return bars
        except ValueError:
            pass

    # Positive number = days back
    try:
        days = int(value)
        if days < 0:
            secho("Error: Days cannot be negative (use negative numbers only with provider mode for bar count)",
                  err=True, fg=colors.RED)
            raise Exit(1)
        return (datetime.now(UTC) - timedelta(days=days)).replace(second=0, microsecond=0)
    except ValueError:
        pass

    # Date string
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        secho(f"Error: Invalid date or number: '{value}'", err=True, fg=colors.RED)
        raise Exit(1)


def _download_provider_data(provider_str: str, time_from_str: str | None) -> tuple[Path, SymInfo]:
    """
    Download historical data from a provider and return the OHLCV path and SymInfo.

    :param provider_str: Provider string (e.g. "ccxt:BYBIT:BTC/USDT:USDT@1D").
    :param time_from_str: The --from parameter value (date, days, or -bars).
    :return: Tuple of (ohlcv_path, syminfo).
    """
    from pynecore.core.plugin import load_plugin, ProviderPlugin
    from pynecore.core.config import ensure_config
    from pynecore.lib.timeframe import in_seconds

    ps = parse_provider_string(provider_str, require_timeframe=True)

    # Load provider plugin
    provider_class = load_plugin(ps.provider)
    if not issubclass(provider_class, ProviderPlugin):
        secho(f"Plugin '{ps.provider}' is not a data provider.", err=True, fg=colors.RED)
        raise Exit(1)

    # Parse --from (required in provider mode)
    if not time_from_str:
        secho("Error: --from / -f is required in provider mode.\n"
              "  Examples: -f 30 (30 days back), -f -500 (500 bars back), -f 2025-01-01",
              err=True, fg=colors.RED)
        raise Exit(1)

    time_from_value = _parse_time_value(time_from_str, allow_bars=True)
    time_to_dt = datetime.now(UTC).replace(second=0, microsecond=0)

    # Convert bar count to time range
    if isinstance(time_from_value, int) and time_from_value < 0:
        bar_count = abs(time_from_value)
        tf_seconds = in_seconds(ps.timeframe)
        time_from_dt = time_to_dt - timedelta(seconds=tf_seconds * bar_count)
    else:
        time_from_dt = time_from_value

    # Load config
    config = None
    if hasattr(provider_class, 'Config') and provider_class.Config is not None:
        config = ensure_config(provider_class.Config,
                               app_state.config_dir / 'plugins' / f'{ps.provider}.toml')

    # Create provider instance
    provider_instance: ProviderPlugin = provider_class(
        symbol=ps.symbol, timeframe=ps.timeframe,
        ohlv_dir=app_state.data_dir, config=config
    )

    # Fetch symbol info
    with Progress(SpinnerColumn(finished_text="[green]✓"), TextColumn("{task.description}")) as progress:
        task = progress.add_task("Fetching symbol info...", total=1)
        syminfo = provider_instance.get_symbol_info(force_update=not provider_instance.is_symbol_info_exists())
        progress.update(task, completed=1)

    # Download OHLCV data (always fresh in provider mode)
    with provider_instance as ohlcv_writer:
        ohlcv_writer.seek(0)
        ohlcv_writer.truncate()

        time_from_dl = time_from_dt.replace(tzinfo=None) if time_from_dt.tzinfo else time_from_dt
        time_to_dl = time_to_dt.replace(tzinfo=None) if time_to_dt.tzinfo else time_to_dt

        total_seconds = int((time_to_dl - time_from_dl).total_seconds())

        with Progress(
                SpinnerColumn(finished_text="[green]✓"),
                TextColumn("{task.description}"),
                DateColumn(time_from_dl),
                BarColumn(),
                TimeElapsedColumn(),
                "/",
                TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Downloading OHLCV data...", total=total_seconds)

            def cb_progress(current_time: datetime):
                elapsed_seconds = int((current_time - time_from_dl).total_seconds())
                progress.update(task, completed=elapsed_seconds)

            provider_instance.download_ohlcv(time_from_dl, time_to_dl, on_progress=cb_progress)

    return provider_instance.ohlcv_path, syminfo


@app.command(cls=PluggableCommand)
def run(
        script: Path = Argument(..., dir_okay=False, file_okay=True, help="Script to run (.py or .pine)"),
        data: str = Argument(...,
                             help="Data file (*.ohlcv, *.csv) or provider string "
                                  "(e.g. ccxt:BYBIT:BTC/USDT:USDT@1D)"),
        time_from: str | None = Option(None, '--from', '-f',
                                       metavar="[DATE|DAYS|-BARS]",
                                       help="Start: date (2025-01-01), days back (30), "
                                            "or -N bars back (-500). Required in provider mode."),
        time_to: str | None = Option(None, '--to', '-t',
                                     metavar="[DATE|DAYS]",
                                     help="End: date or days from start (default: end of data or now)"),
        plot_path: Path | None = Option(None, "--plot", "-pp",
                                        help="Path to save the plot data",
                                        rich_help_panel="Out Path Options"),
        strat_path: Path | None = Option(None, "--strat", "-sp",
                                         help="Path to save the strategy statistics",
                                         rich_help_panel="Out Path Options"
                                         ),
        trade_path: Path | None = Option(None, "--trade", "-tp",
                                         help="Path to save the trade data",
                                         rich_help_panel="Out Path Options"),
        api_key: str | None = Option(None, "--api-key", "-a",
                                     help="PyneSys API key for compilation (overrides configuration file)",
                                     envvar="PYNESYS_API_KEY",
                                     rich_help_panel="Compilation Options"),
        security: list[str] | None = Option(None, "--security", "-sec",
                                            help='Security data: "TIMEFRAME=data_name" or '
                                                 '"SYMBOL:TIMEFRAME=data_name"',
                                            rich_help_panel="Security Options"),

):
    """
    Run a script (.py or .pine)

    The system automatically searches for the workdir folder in the current and parent directories.
    If not found, it creates or uses a workdir folder in the current directory.

    If [bold]script[/] path is a name without full path, it will be searched in the [italic]"workdir/scripts"[/] directory.
    Similarly, if [bold]data[/] path is a name without full path, it will be searched in the [italic]"workdir/data"[/] directory.
    The [bold]plot_path[/], [bold]strat_path[/], and [bold]trade_path[/] work the same way - if they are names without full paths,
    they will be saved in the [italic]"workdir/output"[/] directory.

    [bold]Data Source:[/bold]
    The [bold]data[/] argument accepts either a file path or a provider string:
    \b
      File mode:     pyne run script.py data.csv
      Provider mode: pyne run script.py ccxt:BYBIT:BTC/USDT:USDT@1D -f -500

    In provider mode, historical data is downloaded automatically. The --from/-f parameter
    is required and accepts: date (2025-01-01), days back (30), or -N bars back (-500).

    [bold]Pine Script Support:[/bold]
    Pine Script (.pine) files are automatically compiled to Python (.py) before execution.
    A valid [bold]PyneSys API[/bold] key is required. Get one at [blue]https://pynesys.io[/blue].

    [bold]Data Formats:[/bold]
    Supports CSV, TXT, JSON, and OHLCV data files. Non-OHLCV files are automatically converted.
    """  # noqa

    # Expand script path
    if len(script.parts) == 1:
        script = app_state.scripts_dir / script

    # If no script suffix, try .pine 1st
    if script.suffix == "":
        script = script.with_suffix(".pine")
    # If doesn't exist, try .py
    if not script.exists():
        script = script.with_suffix(".py")

    # Check if script exists
    if not script.exists():
        secho(f"Script file '{script}' not found!", fg="red", err=True)
        raise Exit(1)

    # Handle .pine files - compile them first
    if script.suffix == ".pine":
        # Read api.toml configuration
        api_config = {}
        try:
            with open(app_state.config_dir / 'api.toml', 'rb') as f:
                api_config = tomllib.load(f)['api']
        except KeyError:
            console.print("[red]Invalid API config file (api.toml)![/red]")
            raise Exit(1)
        except FileNotFoundError:
            pass

        # Override API key if provided
        if api_key:
            api_config['api_key'] = api_key

        # Override API URL if provided via environment variable
        api_url = os.getenv("PYNESYS_API_URL")
        if api_url:
            api_config['base_url'] = api_url

        if api_config.get('api_key'):
            # Create the compiler instance
            compiler = PyneComp(**api_config)

            # Determine output path for compiled file
            out_path = script.with_suffix(".py")

            # Check if compilation is needed
            if compiler.needs_compilation(script, out_path):
                with APIErrorHandler(console):
                    with Progress(
                            SpinnerColumn(finished_text="[green]✓"),
                            TextColumn("[progress.description]{task.description}"),
                            console=console
                    ) as progress:
                        task = progress.add_task("Compiling Pine Script...", total=1)

                        # Compile the .pine file
                        compiler.compile(script, out_path)

                        progress.update(task, completed=1)

            # Update script to point to the compiled file
            script = out_path

        # Go back to normal .py file
        else:
            script = script.with_suffix(".py")
            # Check if script exists
            if not script.exists():
                secho(f"Script file '{script}' not found!", fg="red", err=True)
                raise Exit(1)

    # --- Data resolution: provider string or file path ---
    provider_mode = is_provider_string(data)

    if provider_mode:
        # Provider mode: download historical data, get syminfo
        data_path, syminfo = _download_provider_data(data, time_from)
    else:
        # File mode: resolve path, convert if needed
        data_path = Path(data)

        if len(data_path.parts) == 1:
            data_path = app_state.data_dir / data_path

        if data_path.suffix == "":
            ohlcv_path = data_path.with_suffix(".ohlcv")
            csv_path = data_path.with_suffix(".csv")
            if ohlcv_path.exists():
                data_path = ohlcv_path
            elif csv_path.exists():
                data_path = csv_path
            else:
                data_path = ohlcv_path

        if data_path.suffix != ".ohlcv":
            try:
                converter = DataConverter()
                if converter.is_conversion_required(data_path):
                    detected_symbol, detected_provider = DataConverter.guess_symbol_from_filename(data_path)
                    if not detected_symbol:
                        detected_symbol = data_path.stem.upper()
                    with Progress(
                            SpinnerColumn(finished_text="[green]✓"),
                            TextColumn("[progress.description]{task.description}"),
                            console=console
                    ) as progress:
                        task = progress.add_task(f"Converting {data_path.suffix} to OHLCV format...", total=1)
                        converter.convert_to_ohlcv(
                            data_path, provider=detected_provider,
                            symbol=detected_symbol, force=True
                        )
                        data_path = data_path.with_suffix(".ohlcv")
                        progress.update(task, completed=1)
                else:
                    data_path = data_path.with_suffix(".ohlcv")
            except (DataFormatError, ConversionError) as e:
                secho(f"Conversion failed: {e}", fg="red", err=True)
                secho("Please convert the file manually:", fg="red")
                secho(f"pyne data convert-from {data_path}", fg="yellow")
                raise Exit(1)

        if not data_path.exists():
            secho(f"Data file not found: {data_path.name}", fg="red", err=True)
            raise Exit(1)

        try:
            syminfo = SymInfo.load_toml(data_path.with_suffix(".toml"))
        except FileNotFoundError:
            secho(f"Symbol info file '{data_path.with_suffix('.toml')}' not found!", fg="red", err=True)
            raise Exit(1)

    # --- Output paths ---
    if plot_path and plot_path.suffix != ".csv":
        plot_path = plot_path.with_suffix(".csv")
    if not plot_path:
        plot_path = app_state.output_dir / f"{script.stem}.csv"

    if strat_path and strat_path.suffix != ".csv":
        strat_path = strat_path.with_suffix(".csv")
    if not strat_path:
        strat_path = app_state.output_dir / f"{script.stem}_strat.csv"

    if trade_path and trade_path.suffix != ".csv":
        trade_path = trade_path.with_suffix(".csv")
    if not trade_path:
        trade_path = app_state.output_dir / f"{script.stem}_trade.csv"

    # --- Open data and run ---
    with OHLCVReader(data_path) as reader:
        # Parse time range
        time_from_dt = _parse_time_value(time_from) if time_from and not provider_mode else None
        time_to_dt = _parse_time_value(time_to) if time_to else None

        if not time_from_dt:
            time_from_dt = reader.start_datetime
        if not time_to_dt:
            time_to_dt = reader.end_datetime

        time_from_ts = int(time_from_dt.timestamp())
        time_to_ts = int(time_to_dt.timestamp())

        # Remove timezone for display purposes
        time_from_display = time_from_dt.replace(tzinfo=None)
        time_to_display = time_to_dt.replace(tzinfo=None)

        total_seconds = int((time_to_display - time_from_display).total_seconds())

        # Get the iterator using the correct UTC timestamps
        size = reader.get_size(time_from_ts, time_to_ts)
        ohlcv_iter = reader.read_from(time_from_ts, time_to_ts)

        # Parse security data mappings
        security_data: dict[str, str | Path] | None = None
        if security:
            security_data = {}
            for entry in security:
                if '=' not in entry:
                    secho(
                        f"Invalid --security format: '{entry}'. "
                        f"Expected 'TIMEFRAME=data_name' or 'SYMBOL:TIMEFRAME=data_name'",
                        fg="red", err=True,
                    )
                    raise Exit(1)
                key, value = entry.split('=', 1)
                sec_path = Path(value)
                if len(sec_path.parts) == 1:
                    sec_path = app_state.data_dir / sec_path
                if sec_path.suffix:
                    sec_path = sec_path.with_suffix('')
                ohlcv_check = sec_path.with_suffix('.ohlcv')
                if not ohlcv_check.exists():
                    secho(
                        f"Security data not found: {ohlcv_check}",
                        fg="red", err=True,
                    )
                    raise Exit(1)
                security_data[key] = str(sec_path)

        # Add lib directory to Python path for library imports
        lib_dir = app_state.scripts_dir / "lib"
        lib_path_added = False
        if lib_dir.exists() and lib_dir.is_dir():
            sys.path.insert(0, str(lib_dir))
            lib_path_added = True

        # Show loading spinner while importing
        with Progress(
                SpinnerColumn(finished_text="[green]✓"),
                TextColumn("{task.description}"),
        ) as loading_progress:
            loading_task = loading_progress.add_task("Loading PyneCore...", total=1)

            try:
                # Create script runner (this is where the import happens)
                runner = ScriptRunner(script, ohlcv_iter, syminfo, last_bar_index=size - 1,
                                      plot_path=plot_path, strat_path=strat_path, trade_path=trade_path,
                                      security_data=security_data)
            finally:
                # Remove lib directory from Python path
                if lib_path_added:
                    sys.path.remove(str(lib_dir))

            # Mark as completed
            loading_progress.update(loading_task, completed=1)

        # Now run with the main progress bar
        with Progress(
                SpinnerColumn(finished_text="[green]✓"),
                TextColumn("{task.description}"),
                DateColumn(time_from_display),
                BarColumn(),
                CustomTimeElapsedColumn(),
                "/",
                CustomTimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                description="Running script...",
                total=total_seconds,
            )

            # Create queue for progress updates
            progress_queue = queue.Queue()
            stop_event = threading.Event()

            def progress_worker():
                """Worker thread that updates progress bar at 60Hz"""
                last_update = 0
                while not stop_event.is_set():
                    try:
                        # Drain all pending updates
                        current_time = None
                        while True:
                            try:
                                current_time = progress_queue.get_nowait()
                            except queue.Empty:
                                break

                        # Update progress if we have new data
                        if current_time is not None:
                            if current_time == datetime.max:
                                current_time = time_to_display
                            elapsed_seconds = int((current_time - time_from_display).total_seconds())
                            # Only update if time changed (to avoid redundant updates)
                            if elapsed_seconds != last_update:
                                progress.update(task, completed=elapsed_seconds)
                                last_update = elapsed_seconds
                    except Exception:  # noqa
                        pass  # Ignore any errors in worker thread

                    # Wait ~33.33ms (30Hz refresh rate)
                    time.sleep(1 / 30)

            # Start worker thread
            worker = threading.Thread(target=progress_worker, daemon=True)
            worker.start()

            def cb_progress(current_time: datetime | None):
                """Callback that just puts timestamp in queue - near zero overhead"""
                try:
                    progress_queue.put_nowait(current_time)
                except queue.Full:
                    pass  # If queue is full, skip this update

            try:
                # Run the script
                runner.run(on_progress=cb_progress)

                # Ensure final progress update
                progress_queue.put(time_to_display)
                time.sleep(0.05)  # Give worker thread time to process final update

                progress.update(task, completed=total_seconds)
            finally:
                # Stop worker thread
                stop_event.set()
                worker.join(timeout=0.1)  # Wait max 100ms for thread to finish

                # Final update to ensure completion
                progress.refresh()
