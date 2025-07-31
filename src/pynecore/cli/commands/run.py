from pathlib import Path
from datetime import datetime
import queue
import threading
import time
import sys
from typing import Optional

from typer import Option, Argument, secho, Exit
from rich.progress import (Progress, SpinnerColumn, TextColumn, BarColumn,
                           ProgressColumn, Task)
from rich.text import Text
from rich.console import Console

from ..app import app, app_state

from ...utils.rich.date_column import DateColumn
from pynecore.core.ohlcv_file import OHLCVReader

from pynecore.core.syminfo import SymInfo
from pynecore.core.script_runner import ScriptRunner
from ...core.compiler import create_compilation_service
from ...api.exceptions import APIError, AuthError, RateLimitError, CompilationError
from ...api.config import ConfigManager
from ...cli.utils.api_error_handler import handle_api_errors

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
        force: bool = Option(
            False,
            "--force",
            help="Force recompilation for .pine files (ignore smart compilation)",
            rich_help_panel="Compilation Options"
        ),
        strict: bool = Option(
            False,
            "--strict",
            help="Enable strict compilation mode for .pine files",
            rich_help_panel="Compilation Options"
        ),
        api_key: Optional[str] = Option(
            None,
            "--api-key",
            help="PyneSys API key (overrides configuration file)",
            envvar="PYNESYS_API_KEY",
            rich_help_panel="Compilation Options"
        ),
        config_path: Optional[Path] = Option(
            None,
            "--config",
            help="Path to TOML configuration file",
            rich_help_panel="Compilation Options"
        ),
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
    When running a .pine file, it will be automatically compiled to Python before execution.
    Use the compilation options (--force, --strict) to control the compilation process.
    A valid PyneSys API key is required for Pine Script compilation.
    
    [bold]Smart Compilation:[/bold]
    The system checks for changes in .pine files and only recompiles when necessary.
    Use --force to bypass this check and force recompilation.
    """  # noqa
    # Handle script file extension and path
    original_script = script
    compiled_file = None
    
    # Support both .py and .pine files
    if script.suffix not in [".py", ".pine"]:
        # Check if the file exists with the given extension first
        if len(script.parts) == 1:
            full_script_path = app_state.scripts_dir / script
        else:
            full_script_path = script
            
        if full_script_path.exists():
            # File exists but has unsupported extension
            console.print(f"[red]This file format isn't supported:[/red] {script.suffix}")
            console.print("[yellow]✨ Currently supported formats:[/yellow] .py, .pine")
            if script.suffix in [".ohlcv", ".csv", ".json"]:
                console.print(f"[blue]💡 Heads up:[/blue] {script.suffix} files are data files, not executable scripts.")
            raise Exit(1)
        
        # File doesn't exist, try .pine first, then .py
        pine_script = script.with_suffix(".pine")
        py_script = script.with_suffix(".py")
        
        if len(script.parts) == 1:
            pine_script = app_state.scripts_dir / pine_script
            py_script = app_state.scripts_dir / py_script
            
        if pine_script.exists():
            script = pine_script
        elif py_script.exists():
            script = py_script
        else:
            script = py_script  # Default to .py for error message
    
    # Expand script path if it's just a filename
    if len(script.parts) == 1:
        script = app_state.scripts_dir / script
    
    # Check if script exists
    if not script.exists():
        secho(f"Script file '{script}' not found!", fg="red", err=True)
        raise Exit(1)
    
    # Handle .pine files - compile them first
    if script.suffix == ".pine":
        try:
            # Create compilation service
            compilation_service = create_compilation_service(
                api_key=api_key,
                config_path=config_path
            )
            
            # Determine output path for compiled file
            compiled_file = script.with_suffix(".py")
            
            # Check if compilation is needed
            if compilation_service.needs_compilation(script, compiled_file) or force:
                with handle_api_errors(console):
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console
                    ) as progress:
                        progress.add_task("Compiling Pine Script...", total=None)
                        
                        # Compile the .pine file
                        compiled_path = compilation_service.compile_file(
                            script,
                            compiled_file,
                            force=force,
                            strict=strict
                        )
                        
                    console.print(f"[green]Compilation successful![/green] Ready to run: [cyan]{compiled_path}[/cyan]")
                    compiled_file = compiled_path
                        
            else:
                console.print(f"[green]⚡ Using cached version:[/green] [cyan]{compiled_file}[/cyan]")
                console.print("[dim]Use --force to recompile[/dim]")
            
            # Update script to point to the compiled file
            script = compiled_file
            
        except ValueError as e:
            error_msg = str(e)
            if "No configuration file found" in error_msg or "Configuration file not found" in error_msg:
                console.print("[yellow]⚠️  No API configuration found[/yellow]")
                console.print()
                console.print("[bold]Quick setup (takes few minutes):[/bold]")
                console.print("1. 🌐 Get your API key at [blue][link=https://pynesys.io]https://pynesys.io[/link][/blue]")
                console.print("2. 🔧 Run [cyan]pyne api configure[/cyan] to save your configuration")
                console.print()
                console.print("[dim]💬 Need assistance? Our docs are here: https://pynesys.io/docs[/dim]")
            else:
                console.print(f"[red] Attention:[/red] {e}")
            raise Exit(1)
    
    # Ensure we have a .py file at this point
    if script.suffix != ".py":
        console.print(f"[red]This file format isn't supported:[/red] {script.suffix}")
        console.print("[yellow]✨ Currently supported formats:[/yellow] .py, .pine")
        if script.suffix in [".ohlcv", ".csv", ".json"]:
            console.print(f"[blue]💡 Heads up:[/blue] {script.suffix} files are data files, not executable scripts.")
        raise Exit(1)

    # Check file format and extension
    if data.suffix == "":
        # No extension, add .ohlcv
        data = data.with_suffix(".ohlcv")
    elif data.suffix != ".ohlcv":
        # Has extension but not .ohlcv
        secho(f"Cannot run with '{data.suffix}' files. The PyneCore runtime requires .ohlcv format.",
              fg="red", err=True)
        secho("If you're trying to use a different data format, please convert it first:", fg="red")
        symbol_placeholder = "YOUR_SYMBOL"
        timeframe_placeholder = "YOUR_TIMEFRAME"
        secho(f"pyne data convert-from {data} --symbol {symbol_placeholder} --timeframe {timeframe_placeholder}",
              fg="yellow")
        raise Exit(1)

    # Expand data path
    if len(data.parts) == 1:
        data = app_state.data_dir / data
    # Check if data exists
    if not data.exists():
        secho(f"Data file '{data}' not found!", fg="red", err=True)
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
        time_from = time_from.replace(tzinfo=None)
        time_to = time_to.replace(tzinfo=None)

        total_seconds = int((time_to - time_from).total_seconds())

        # Get the iterator
        size = reader.get_size(int(time_from.timestamp()), int(time_to.timestamp()))
        ohlcv_iter = reader.read_from(int(time_from.timestamp()), int(time_to.timestamp()))

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
