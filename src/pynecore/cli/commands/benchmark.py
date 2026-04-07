import time
import gc
from pathlib import Path
import statistics

from typer import Option, Argument, secho, Exit
from rich.table import Table
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from ..app import app, app_state
from ...core.ohlcv_file import OHLCVReader
from ...core.syminfo import SymInfo
from ...core.script_runner import ScriptRunner

__all__ = []


@app.command(name="benchmark")
def benchmark(
        script: Path = Argument("demo", help="Script to benchmark"),
        data: Path = Argument("demo", help="Data file to use"),
        iterations: int = Option(10, "--iterations", "-i", help="Number of iterations to run"),
        candles: int = Option(5000, "--candles", "-c", help="Number of candles to process"),
        warmup: int = Option(2, "--warmup", "-w", help="Number of warmup iterations"),
        no_output: bool = Option(True, "--no-output", help="Don't write CSV output"),
):
    """
    Benchmark script execution performance

    This command runs a script multiple times to measure performance.
    By default, it uses demo.py and demo.ohlcv files from the workdir.
    """

    console = Console()

    # Process script path - use same logic as run command
    # Ensure .py extension
    if script.suffix != ".py":
        script = script.with_suffix(".py")
    # Expand script path
    if len(script.parts) == 1:
        script = app_state.scripts_dir / script
    # Check if script exists
    if not script.exists():
        secho(f"Script file '{script}' not found!", fg="red", err=True)
        raise Exit(1)

    # Process data path - similar logic
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

    if not data.exists():
        secho(f"Data file '{data}' not found!", fg="red", err=True)
        raise Exit(1)

    # Load symbol info
    try:
        syminfo = SymInfo.load_toml(data.with_suffix(".toml"))
    except FileNotFoundError:
        secho(f"Symbol info file '{data.with_suffix('.toml')}' not found!", fg="red", err=True)
        raise Exit(1)

    secho(f"\nBenchmarking: {script.name}", fg="cyan")
    secho(f"Data: {data.name} ({candles} candles)", fg="cyan")
    secho(f"Iterations: {iterations} (warmup: {warmup})", fg="cyan")
    secho("")

    # Timing results
    import_times = []
    run_times = []
    total_times = []

    # Total iterations including warmup
    total_iterations = warmup + iterations

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Running benchmark...", total=total_iterations)

        for i in range(total_iterations):
            # Force garbage collection before each run
            gc.collect()

            # Time the whole process
            total_start = time.perf_counter()

            # Time the import
            import_start = time.perf_counter()

            # Open data file and get iterator
            with OHLCVReader(data) as reader:
                start_ts = reader.start_timestamp
                assert start_ts is not None
                # Get only the requested number of candles
                ohlcv_list = []
                # Use read_from to get all data from the beginning
                for idx, ohlcv in enumerate(reader.read_from(start_ts, reader.end_timestamp)):
                    if idx >= candles:
                        break
                    ohlcv_list.append(ohlcv)

                # Create runner
                runner = ScriptRunner(
                    script,
                    iter(ohlcv_list),
                    syminfo,
                    last_bar_index=len(ohlcv_list) - 1,
                    plot_path=None if no_output else app_state.output_dir / f"benchmark_{i}.csv"
                )

                import_time = time.perf_counter() - import_start

                # Time the execution
                run_start = time.perf_counter()
                runner.run()
                run_time = time.perf_counter() - run_start

                total_time = time.perf_counter() - total_start

                # Only collect times after warmup
                if i >= warmup:
                    import_times.append(import_time)
                    run_times.append(run_time)
                    total_times.append(total_time)

            progress.update(task, advance=1)

            # Clean up output files if no_output
            if no_output and (app_state.output_dir / f"benchmark_{i}.csv").exists():
                (app_state.output_dir / f"benchmark_{i}.csv").unlink()

    # Calculate statistics
    avg_import = statistics.mean(import_times)
    avg_run = statistics.mean(run_times)
    avg_total = statistics.mean(total_times)

    min_run = min(run_times)
    max_run = max(run_times)
    std_run = statistics.stdev(run_times) if len(run_times) > 1 else 0

    # Candles per second
    candles_per_second = candles / avg_run if avg_run > 0 else 0

    # Create results table
    table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right", style="green")

    table.add_row("Script", script.name)
    table.add_row("Candles", str(candles))
    table.add_row("Iterations", str(iterations))
    table.add_row("", "")
    table.add_row("Avg Import Time", f"{avg_import * 1000:.2f} ms")
    table.add_row("Avg Run Time", f"{avg_run * 1000:.2f} ms")
    table.add_row("Avg Total Time", f"{avg_total * 1000:.2f} ms")
    table.add_row("", "")
    table.add_row("Min Run Time", f"{min_run * 1000:.2f} ms")
    table.add_row("Max Run Time", f"{max_run * 1000:.2f} ms")
    table.add_row("Std Dev", f"{std_run * 1000:.2f} ms")
    table.add_row("", "")
    table.add_row("Candles/Second", f"{candles_per_second:.0f}")
    table.add_row("Time per Candle", f"{avg_run / candles * 1000:.3f} ms")

    console.print("")
    console.print(table)
