import os
import queue
import threading
import time
import sys
import tomllib

from pathlib import Path
from datetime import datetime

from typer import Option, Argument, secho, Exit
from rich.progress import (Progress, SpinnerColumn, TextColumn, BarColumn,
                           ProgressColumn, Task)
from rich.text import Text
from rich.console import Console

from ..app import app, app_state

from ...utils.rich.date_column import DateColumn
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.data_converter import DataConverter, DataFormatError, ConversionError

from pynecore.core.syminfo import SymInfo
from pynecore.core.script_runner import ScriptRunner
from pynecore.pynesys.compiler import PyneComp
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


@app.command()
def run(
        script: Path = Argument(..., dir_okay=False, file_okay=True, help="Script to run (.py or .pine)"),
        data: Path = Argument(..., dir_okay=False, file_okay=True,
                              help="Data file to use (*.ohlcv)"),
        time_from: datetime | None = Option(None, '--from', '-f',
                                            formats=["%Y-%m-%d", "%Y-%m-%d %H:%M:%S"],
                                            help="Start date (UTC), if not specified, will use the "
                                                 "first date in the data"),
        time_to: datetime | None = Option(None, '--to', '-t',
                                          formats=["%Y-%m-%d", "%Y-%m-%d %H:%M:%S"],
                                          help="End date (UTC), if not specified, will use the last "
                                               "date in the data"),
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

):
    """
    Run a script (.py or .pine)

    The system automatically searches for the workdir folder in the current and parent directories.
    If not found, it creates or uses a workdir folder in the current directory.

    If [bold]script[/] path is a name without full path, it will be searched in the [italic]"workdir/scripts"[/] directory.
    Similarly, if [bold]data[/] path is a name without full path, it will be searched in the [italic]"workdir/data"[/] directory.
    The [bold]plot_path[/], [bold]strat_path[/], and [bold]trade_path[/] work the same way - if they are names without full paths,
    they will be saved in the [italic]"workdir/output"[/] directory.
    
    [bold]Pine Script Support:[/bold]
    Also Pine Script (.pine) files could be automatically compiled to Python (.py) before execution, if the 
    file is newer than the [italic]py[/] file or if the [italic].py[/] file doesn't exist. The compiled [italic].py[/] file will be saved 
    into the same folder as the original [italic].pine[/] file.
    A valid [bold]PyneSys API[/bold] key is required for Pine Script compilation. You can get one at [blue]https://pynesys.io[/blue].
    
    [bold]Data Support:[/bold]
    Supports CSV, TXT, JSON, and OHLCV data files. Non-OHLCV files are automatically converted. Symbol is auto-detected from filename.
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

    # Expand data path first - convert relative paths to absolute paths in workdir/data
    if len(data.parts) == 1:
        data = app_state.data_dir / data

    # Store the original suffix to check what user provided
    original_suffix = data.suffix

    # Check file format and extension
    if data.suffix == "":
        # No extension provided - check if .ohlcv exists, otherwise look for .csv
        ohlcv_path = data.with_suffix(".ohlcv")
        csv_path = data.with_suffix(".csv")

        if ohlcv_path.exists():
            data = ohlcv_path
        elif csv_path.exists():
            data = csv_path
        else:
            # Default to .ohlcv for error message
            data = ohlcv_path

    # Now handle conversion if needed
    if data.suffix != ".ohlcv":
        # Has extension but not .ohlcv - automatically convert
        try:
            converter = DataConverter()

            # Check if conversion is needed
            if converter.is_conversion_required(data):
                # Auto-detect symbol and provider from filename
                detected_symbol, detected_provider = DataConverter.guess_symbol_from_filename(data)

                if not detected_symbol:
                    detected_symbol = data.stem.upper()

                with Progress(
                        SpinnerColumn(finished_text="[green]✓"),
                        TextColumn("[progress.description]{task.description}"),
                        console=console
                ) as progress:
                    task = progress.add_task(f"Converting {data.suffix} to OHLCV format...", total=1)

                    # Perform conversion with smart defaults
                    converter.convert_to_ohlcv(
                        data,
                        provider=detected_provider,
                        symbol=detected_symbol,
                        force=True
                    )

                    # After conversion, the OHLCV file has the same name but .ohlcv extension
                    data = data.with_suffix(".ohlcv")

                    progress.update(task, completed=1)
            else:
                # File is already up-to-date, use existing OHLCV file
                data = data.with_suffix(".ohlcv")

        except (DataFormatError, ConversionError) as e:
            secho(f"Conversion failed: {e}", fg="red", err=True)
            secho("Please convert the file manually:", fg="red")
            secho(f"pyne data convert-from {data}", fg="yellow")
            raise Exit(1)

    # Final check if data exists
    if not data.exists():
        secho(f"Data file not found: {data.name}", fg="red", err=True)
        raise Exit(1)

    # Ensure .csv extension for plot path
    if plot_path and plot_path.suffix != ".csv":
        plot_path = plot_path.with_suffix(".csv")
    if not plot_path:
        plot_path = app_state.output_dir / f"{script.stem}.csv"

    # Ensure .csv extension for strategy path
    if strat_path and strat_path.suffix != ".csv":
        strat_path = strat_path.with_suffix(".csv")
    if not strat_path:
        strat_path = app_state.output_dir / f"{script.stem}_strat.csv"

    # Ensure .csv extension for trade path
    if trade_path and trade_path.suffix != ".csv":
        trade_path = trade_path.with_suffix(".csv")
    if not trade_path:
        trade_path = app_state.output_dir / f"{script.stem}_trade.csv"

    # Get symbol info for the data
    try:
        syminfo = SymInfo.load_toml(data.with_suffix(".toml"))
    except FileNotFoundError:
        secho(f"Symbol info file '{data.with_suffix('.toml')}' not found!", fg="red", err=True)
        raise Exit(1)

    # Open data file
    with OHLCVReader(data) as reader:
        if not time_from:
            time_from = reader.start_datetime
        if not time_to:
            time_to = reader.end_datetime

        # Convert to UTC timestamps BEFORE removing timezone info
        # This ensures we use the correct UTC timestamps for the OHLCV reader
        time_from_ts = int(time_from.timestamp())
        time_to_ts = int(time_to.timestamp())

        # Now we can safely remove timezone for display purposes
        time_from = time_from.replace(tzinfo=None)
        time_to = time_to.replace(tzinfo=None)

        total_seconds = int((time_to - time_from).total_seconds())

        # Get the iterator using the correct UTC timestamps
        size = reader.get_size(time_from_ts, time_to_ts)
        ohlcv_iter = reader.read_from(time_from_ts, time_to_ts)

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
                                      plot_path=plot_path, strat_path=strat_path, trade_path=trade_path)
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
                DateColumn(time_from),
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
                                current_time = time_to
                            elapsed_seconds = int((current_time - time_from).total_seconds())
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
                progress_queue.put(time_to)
                time.sleep(0.05)  # Give worker thread time to process final update

                progress.update(task, completed=total_seconds)
            finally:
                # Stop worker thread
                stop_event.set()
                worker.join(timeout=0.1)  # Wait max 100ms for thread to finish

                # Final update to ensure completion
                progress.refresh()
