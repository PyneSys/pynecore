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
from pynecore.core.aggregator import validate_aggregation
from pynecore.lib.log import logger as pyne_logger
from pynecore.lib.timeframe import in_seconds

from pynecore.core.syminfo import SymInfo
from pynecore.core.script_runner import ScriptRunner
from pynecore.pynesys.compiler import PyneComp
from pynecore.core.provider_string import is_provider_string, parse_provider_string
from pynecore.core.live_runner import live_ohlcv_generator
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
    value: str = value.strip()

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


class _ProviderData:
    """Result of provider data download, including the provider instance for live mode."""

    def __init__(self, ohlcv_path: Path, syminfo: 'SymInfo', provider_instance=None,
                 parsed_string=None, time_from_ts: int | None = None):
        self.ohlcv_path = ohlcv_path
        self.syminfo = syminfo
        self.provider_instance = provider_instance
        self.parsed_string = parsed_string
        # Exact start timestamp that yields exactly the requested bar
        # count in ``-N bars`` mode — None when the caller should use
        # the file's natural start (date/days mode, no bar target).
        self.time_from_ts = time_from_ts


def _download_provider_data(provider_str: str, time_from_str: str | None) -> _ProviderData:
    """
    Download historical data from a provider and return the result.

    :param provider_str: Provider string (e.g. "ccxt:BYBIT:BTC/USDT:USDT@1D").
    :param time_from_str: The --from parameter value (date, days, or -bars).
    :return: _ProviderData with ohlcv_path, syminfo, and provider instance.
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

    # Default to -500 bars if --from not specified in provider mode
    if not time_from_str:
        time_from_str = "-500"

    time_from_value = _parse_time_value(time_from_str, allow_bars=True)
    time_to_dt = datetime.now(UTC).replace(second=0, microsecond=0)

    # Convert bar count to time range. ``bar_count`` being set signals
    # the "-N bars" mode — we then guarantee at least N *real* bars
    # (exchange-provided, non-gap-fill) after download, extending the
    # from-timestamp on miss. Date/days ranges are left untouched.
    tf_seconds = in_seconds(ps.timeframe)
    bar_count: int | None = None
    if isinstance(time_from_value, int) and time_from_value < 0:
        bar_count = abs(time_from_value)
        # Pad the request by one bar to absorb the still-forming current
        # bar that closed-bars-only providers (e.g. Capital.com) filter
        # out of history responses. Without this, every ``-N`` run
        # against a now-aligned end-time would burn a wasted retry pass
        # (``real_bars == N - 1`` on first attempt → retry → success).
        time_from_dt = time_to_dt - timedelta(seconds=tf_seconds * (bar_count + 1))
    else:
        assert isinstance(time_from_value, datetime)
        time_from_dt = time_from_value

    # Load config
    config = None
    config_cls: type | None = getattr(provider_class, 'Config', None)
    if config_cls is not None:
        config = ensure_config(config_cls,
                               app_state.config_dir / 'plugins' / f'{ps.provider}.toml')

    # Create provider instance
    provider_instance: ProviderPlugin = provider_class(
        symbol=ps.symbol, timeframe=ps.timeframe,
        ohlcv_dir=app_state.data_dir, config=config
    )

    # Fetch symbol info
    with Progress(SpinnerColumn(finished_text="[green]✓"), TextColumn("{task.description}")) as progress:
        task = progress.add_task("Fetching symbol info...", total=1)
        syminfo = provider_instance.get_symbol_info(force_update=not provider_instance.is_symbol_info_exists())
        progress.update(task, completed=1)

    # Download OHLCV data (always fresh in provider mode). In bar-count
    # mode we may re-download with an extended ``from`` until we hit the
    # target — some feeds omit minutes with no ticks (CFD quiet hours,
    # illiquid futures), and ``--from -500`` must mean 500 real bars.
    # The Progress wrapper lives outside the retry loop so a gap-driven
    # second pass updates the same spinner instead of stamping a
    # duplicate ``Downloading OHLCV data...`` line.
    max_retries = 4
    with Progress(
            SpinnerColumn(finished_text="[green]✓"),
            TextColumn("{task.description}"),
            DateColumn(),
            BarColumn(),
            TimeElapsedColumn(),
            "/",
            TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            "Downloading OHLCV data...", total=1, start_time=time_from_dt,
        )
        for attempt in range(max_retries + 1):
            with provider_instance as ohlcv_writer:
                ohlcv_writer.seek(0)
                ohlcv_writer.truncate()

                time_from_dl = time_from_dt.replace(tzinfo=None) if time_from_dt.tzinfo else time_from_dt
                time_to_dl = time_to_dt.replace(tzinfo=None) if time_to_dt.tzinfo else time_to_dt

                total_seconds = int((time_to_dl - time_from_dl).total_seconds())

                progress.update(
                    task, total=total_seconds, completed=0,
                    start_time=time_from_dl,
                )

                def cb_progress(current_time: datetime):
                    elapsed_seconds = int((current_time - time_from_dl).total_seconds())
                    progress.update(task, completed=elapsed_seconds)

                provider_instance.download_ohlcv(time_from_dl, time_to_dl, on_progress=cb_progress)

            if bar_count is None:
                break

            # Count real bars (``volume >= 0``) in the requested range —
            # gap-fill rows (``volume == -1``) emitted by the OHLCV writer
            # don't count against the target.
            with OHLCVReader(provider_instance.ohlcv_path) as r:  # type: ignore[arg-type]
                real_bars = sum(1 for _ in r.read_from(
                    int(time_from_dt.timestamp()),
                    int(time_to_dt.timestamp()),
                    skip_gaps=True,
                ))

            if real_bars >= bar_count:
                break
            if attempt == max_retries:
                secho(
                    f"Warning: requested {bar_count} bars, only got {real_bars} "
                    f"real bars after {max_retries + 1} attempts (feed has "
                    f"sparse coverage).",
                    fg=colors.YELLOW,
                )
                break

            # Extend the range by the shortfall plus a proportional buffer so
            # we likely finish in one extra pass instead of retrying again.
            missing = bar_count - real_bars
            gap_ratio = missing / real_bars if real_bars else 1.0
            extra_bars = int(missing * (1.0 + gap_ratio)) + 10
            time_from_dt -= timedelta(seconds=tf_seconds * extra_bars)

    assert provider_instance.ohlcv_path is not None

    # For bar-count mode, pin the start timestamp to the N-th last real
    # bar so the reader serves *exactly* N bars, not the over-fetched
    # surplus we used to guarantee coverage through gaps.
    exact_from_ts: int | None = None
    if bar_count is not None:
        with OHLCVReader(provider_instance.ohlcv_path) as r:  # type: ignore[arg-type]
            real_ts = [b.timestamp for b in r.read_from(
                0, int(time_to_dt.timestamp()), skip_gaps=True,
            )]
        if len(real_ts) >= bar_count:
            exact_from_ts = real_ts[-bar_count]

    return _ProviderData(
        ohlcv_path=provider_instance.ohlcv_path,
        syminfo=syminfo,
        provider_instance=provider_instance,
        parsed_string=ps,
        time_from_ts=exact_from_ts,
    )


@app.command(cls=PluggableCommand)
def run(
        script: Path = Argument(..., dir_okay=False, file_okay=True, help="Script to run (.py or .pine)"),
        data: str = Argument(...,
                             help="Data file (*.ohlcv, *.csv) or provider string "
                                  "(e.g. ccxt:BYBIT:BTC/USDT:USDT@1D)"),
        time_from: str | None = Option(None, '--from', '-f',
                                       metavar="[DATE|DAYS|-BARS]",
                                       help="Start: date (2025-01-01), days back (30), "
                                            "or -N bars back (-500). Default: -500 bars in provider mode."),
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
        live: bool = Option(False, "--live", "-l",
                            help="Continue with live data after historical phase "
                                 "(provider mode only)"),
        broker: bool = Option(False, "--broker",
                              help="Enable live broker trading — requires a provider plugin that "
                                   "subclasses BrokerPlugin. Implies --live.",
                              rich_help_panel="Live Options"),
        run_label: str | None = Option(None, "--run-label",
                                       help="Optional label to distinguish parallel instances of the "
                                            "same strategy+account+symbol+timeframe. Stored in the "
                                            "broker run_id as ``...#<label>``.",
                                       rich_help_panel="Live Options"),
        shutdown_timeout: float = Option(120.0, "--shutdown-timeout",
                                         help="Max seconds to wait for graceful shutdown "
                                              "(0 = wait forever)",
                                         rich_help_panel="Live Options"),
        no_log_ohlcv: bool = Option(False, "--no-log-ohlcv",
                                    help="Disable per-bar OHLCV log lines in live mode "
                                         "(default: enabled).",
                                    rich_help_panel="Live Options"),
        security: list[str] | None = Option(None, "--security", "-sec",
                                            help='Security data: "TIMEFRAME=data_name" or '
                                                 '"SYMBOL:TIMEFRAME=data_name"',
                                            rich_help_panel="Security Options"),
        timeframe: str | None = Option(None, "--timeframe", "-tf",
                                       help="Chart timeframe (TradingView format, e.g. '60', '1D'). "
                                            "When larger than data timeframe: aggregates on-the-fly, "
                                            "or activates bar magnifier if strategy uses "
                                            "use_bar_magnifier=true."),

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
    accepts: date (2025-01-01), days back (30), or -N bars back (-500). Default: -500 bars.

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
    provider_data = None

    if live and not provider_mode:
        secho("Error: --live is only available in provider mode.", err=True, fg=colors.RED)
        raise Exit(1)

    if provider_mode:
        # Provider mode: download historical data, get syminfo
        provider_data = _download_provider_data(data, time_from)
        data_path, syminfo = provider_data.ohlcv_path, provider_data.syminfo
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

    # Validate and process --timeframe option
    magnifier_mode = False
    if timeframe:
        chart_tf: str = timeframe.upper()
        try:
            in_seconds(chart_tf)
        except (ValueError, AssertionError):
            secho(f"Invalid timeframe: {chart_tf}. Must be a valid TradingView format "
                  f"(e.g. '1', '5', '60', '1D', '1W', '1M').", fg="red", err=True)
            raise Exit(1)

        data_tf = syminfo.period
        if chart_tf != data_tf:
            try:
                validate_aggregation(data_tf, chart_tf)
            except ValueError as e:
                secho(str(e), fg="red", err=True)
                raise Exit(1)
            # Override syminfo period to the chart timeframe
            syminfo.period = chart_tf
            magnifier_mode = True

    # --- Open data and run ---
    with OHLCVReader(data_path) as reader:
        # Parse time range
        time_from_dt = _parse_time_value(time_from) if time_from and not provider_mode else None
        time_to_dt = _parse_time_value(time_to) if time_to else None

        if not time_from_dt:
            # Provider bar-count mode pins the start to the N-th last
            # real bar; otherwise use the file's natural start.
            if provider_data is not None and provider_data.time_from_ts is not None:
                time_from_dt = datetime.fromtimestamp(
                    provider_data.time_from_ts, UTC,
                )
            else:
                time_from_dt = reader.start_datetime
        if not time_to_dt:
            time_to_dt = reader.end_datetime

        assert isinstance(time_from_dt, datetime) and isinstance(time_to_dt, datetime)
        time_from_ts = int(time_from_dt.timestamp())
        time_to_ts = int(time_to_dt.timestamp())

        # Remove timezone for display purposes
        time_from_display = time_from_dt.replace(tzinfo=None)
        time_to_display = time_to_dt.replace(tzinfo=None)

        total_seconds = int((time_to_display - time_from_display).total_seconds())

        # Get the iterator using the correct UTC timestamps
        size = reader.get_size(time_from_ts, time_to_ts)
        magnifier_iter = None
        if magnifier_mode:
            # Sub-TF data goes to magnifier; ohlcv_iter is unused (replaced in ScriptRunner)
            magnifier_iter = reader.read_from(time_from_ts, time_to_ts)
            ohlcv_iter = iter([])
        else:
            ohlcv_iter = reader.read_from(time_from_ts, time_to_ts)

        # --broker implies --live.
        if broker:
            live = True

        # Broker mode: verify plugin capability up front.
        broker_plugin = None
        broker_event_loop = None
        broker_event_loop_thread = None
        broker_store = None
        broker_store_ctx = None
        if broker:
            if not provider_data:
                secho("--broker requires a provider string (ccxt:EXCHANGE:SYMBOL@TIMEFRAME).",
                      err=True, fg=colors.RED)
                raise Exit(1)
            from pynecore.core.plugin.broker import BrokerPlugin
            if not isinstance(provider_data.provider_instance, BrokerPlugin):
                secho(
                    f"Plugin '{provider_data.parsed_string.provider}' is not a BrokerPlugin "
                    f"— broker mode requires an exchange-backed plugin.",
                    err=True, fg=colors.RED,
                )
                raise Exit(1)
            broker_plugin = provider_data.provider_instance
            import asyncio as _asyncio
            broker_event_loop = _asyncio.new_event_loop()
            # Drive the loop on a dedicated daemon thread. Broker plugin
            # coroutines are submitted from the (synchronous) Pine script
            # thread via ``run_coroutine_threadsafe``, which requires the
            # target loop to actually be running — without this pump every
            # broker call would park forever.
            broker_event_loop_thread = threading.Thread(
                target=broker_event_loop.run_forever,
                daemon=True,
                name="broker-event-loop",
            )
            broker_event_loop_thread.start()

            # Open the unified broker storage and register a new run instance.
            # The plugin's account_id is already populated by this point —
            # _download_provider_data has driven authentication through the
            # provider side, and the plugin stashes the identifier during
            # its session setup.
            from pynecore.core.broker.run_identity import RunIdentity
            from pynecore.core.broker.storage import BrokerStore
            store_path = app_state.workdir / "output" / "logs" / "broker.sqlite"
            broker_store = BrokerStore(
                store_path, plugin_name=broker_plugin.plugin_name,
            )
            identity = RunIdentity(
                strategy_id=script.stem,
                symbol=str(syminfo.ticker),
                timeframe=str(syminfo.period or ""),
                account_id=broker_plugin.account_id,
                label=run_label,
            )
            try:
                broker_store_ctx = broker_store.open_run(
                    identity,
                    script_source=script.read_text(encoding='utf-8'),
                    script_path=script,
                )
            except RuntimeError as e:
                secho(str(e), err=True, fg=colors.RED)
                broker_store.close()
                raise Exit(1)

        # The broker run is now open; every subsequent startup step
        # (security parsing, live iterator chaining, ScriptRunner import)
        # must run under the same try/finally that also wraps runner.run(),
        # otherwise an early raise would leave the active runs row with a
        # NULL ``ended_ts_ms`` and block the next startup of the same bot
        # until the stale-cleanup window (5 min) expires.
        try:
            # Validate live mode capability up front (the live iterator is
            # created only AFTER ``Loading PyneCore`` finishes, so its eager
            # ``WS connect`` log doesn't race the spinner).
            if live and provider_data:
                from pynecore.core.plugin.live_provider import LiveProviderPlugin

                if not isinstance(provider_data.provider_instance, LiveProviderPlugin):
                    secho(f"Plugin '{provider_data.parsed_string.provider}' does not support live data.",
                          err=True, fg=colors.RED)
                    raise Exit(1)
                size = 0

            # Parse security data mappings
            security_data: dict[str, str | Path] | None = None
            if security:
                sec_map: dict[str, str | Path] = {}
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
                    sec_map[key] = str(sec_path)
                security_data = sec_map

            # Add lib directory to Python path for library imports
            lib_dir = app_state.scripts_dir / "lib"
            lib_path_added = False
            if lib_dir.exists() and lib_dir.is_dir():
                sys.path.insert(0, str(lib_dir))
                lib_path_added = True

            # Set live mode flags before ScriptRunner creation
            if live:
                from pynecore import lib as _lib
                _lib._is_live = True
                _lib._strategy_suppressed = True

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
                                          security_data=security_data,
                                          magnifier_iter=magnifier_iter,
                                          broker_plugin=broker_plugin,
                                          broker_event_loop=broker_event_loop,
                                          broker_store_ctx=broker_store_ctx,
                                          log_ohlcv=live and not no_log_ohlcv)
                finally:
                    # Remove lib directory from Python path
                    if lib_path_added:
                        sys.path.remove(str(lib_dir))

                # Mark as completed
                loading_progress.update(loading_task, completed=1)

            # Now that the script is loaded, start the live OHLCV stream.
            # ``live_ohlcv_generator`` eager-starts the WS connect (and
            # blocks until subscribed), so doing it AFTER the spinner keeps
            # the ``[BROKER] WS connect …`` lines below ``✓ Loading PyneCore``
            # in the user-visible startup log.
            if live and provider_data:
                import itertools
                live_iter = live_ohlcv_generator(
                    provider=provider_data.provider_instance,
                    symbol=provider_data.parsed_string.symbol,
                    timeframe=provider_data.parsed_string.timeframe,
                    last_historical_timestamp=time_to_ts,
                    shutdown_timeout=shutdown_timeout,
                    event_loop=broker_event_loop,
                )
                runner.ohlcv_iter = itertools.chain(runner.ohlcv_iter, live_iter)

            # Start broker-side I/O (watch_orders task + startup reconcile).
            # No-op when ``broker_plugin`` is None.
            runner.start_broker()

            if live:
                # Share the Pine logger's Console with the live Progress so
                # `logger.info(...)` lines render above the spinner rather
                # than colliding with it. The PineRichHandler owns the only
                # Console; reusing it keeps Rich's Live-intercept coherent.
                live_console = None
                for _h in pyne_logger.handlers:
                    _h_console = getattr(_h, 'console', None)
                    if _h_console is not None:
                        live_console = _h_console
                        break

                # Latest quote snapshot — the tick hook updates these so the
                # spinner text carries the current bid/ask even between bar
                # closes.
                spinner_state = {
                    'time': None,
                    'bid': None,
                    'ask': None,
                }

                # Derive fixed price decimals from ``syminfo.mintick`` so
                # the spinner prices keep a constant width (``1.16830`` /
                # ``1.16837`` instead of ``1.1683`` / ``1.16837``). Matches
                # the same computation in ScriptRunner for the OHLCV log.
                _mintick = getattr(syminfo, 'mintick', 0) or 0
                if _mintick > 0:
                    _tick_str = f"{_mintick:.20f}".rstrip('0').rstrip('.')
                    price_decimals = (
                        len(_tick_str.split('.')[1])
                        if '.' in _tick_str else 0
                    )
                else:
                    price_decimals = 2

                def _spinner_text() -> str:
                    t = spinner_state['time']
                    head = f"Live — {t:%Y-%m-%d %H:%M:%S}" if t else "Live streaming..."
                    bid = spinner_state['bid']
                    ask = spinner_state['ask']
                    d = price_decimals
                    if bid is not None and ask is not None:
                        return (f"{head}  [green]▼ {bid:.{d}f}[/]  "
                                f"[red]▲ {ask:.{d}f}[/]")
                    if bid is not None:
                        return f"{head}  [green]{bid:.{d}f}[/]"
                    return head

                # Live mode: spinner instead of progress bar (no known end time)
                with Progress(
                        SpinnerColumn(),
                        TextColumn("{task.description}"),
                        CustomTimeElapsedColumn(),
                        console=live_console,
                ) as progress:
                    task = progress.add_task(description="Live streaming...", total=None)

                    def cb_progress_live(current_time: datetime | None):
                        if current_time is not None:
                            spinner_state['time'] = current_time
                            progress.update(task, description=_spinner_text())

                    def cb_tick_live(candle):
                        extra = candle.extra_fields or {}
                        # Prefer the live quote snapshot over ``candle.close``:
                        # on a closed OHLC bar ``candle.close`` is the previous
                        # period's bid-side close, not the current quote.
                        spinner_state['bid'] = extra.get('bid_close', candle.close)
                        spinner_state['ask'] = extra.get('ask_close')
                        spinner_state['time'] = datetime.fromtimestamp(
                            candle.timestamp, UTC,
                        ).replace(tzinfo=None)
                        progress.update(task, description=_spinner_text())

                    try:
                        runner.run(on_progress=cb_progress_live,
                                   on_tick=cb_tick_live)
                    except KeyboardInterrupt:
                        secho("\nLive streaming stopped.", fg=colors.YELLOW)

            else:
                # Batch mode: progress bar with time range
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
                        """Worker thread that updates progress bar at 30Hz"""
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
                                    elapsed_seconds = int(
                                        (current_time - time_from_display).total_seconds())
                                    if elapsed_seconds != last_update:
                                        progress.update(task, completed=elapsed_seconds)
                                        last_update = elapsed_seconds
                            except Exception:  # noqa
                                pass

                            time.sleep(1 / 30)

                    # Start worker thread
                    worker = threading.Thread(target=progress_worker, daemon=True)
                    worker.start()

                    def cb_progress(current_time: datetime | None):
                        """Callback that just puts timestamp in queue"""
                        try:
                            progress_queue.put_nowait(current_time)
                        except queue.Full:
                            pass

                    try:
                        runner.run(on_progress=cb_progress)

                        progress_queue.put(time_to_display)
                        time.sleep(0.05)

                        progress.update(task, completed=total_seconds)
                    finally:
                        stop_event.set()
                        worker.join(timeout=0.1)
                        progress.refresh()
        finally:
            # Close the broker storage run cleanly — happy-path lezárás.
            # Crash-path (SIGKILL, OOM) a storage stale-run cleanupjára bízott.
            if broker_store_ctx is not None:
                broker_store_ctx.close()
            if broker_store is not None:
                broker_store.close()
            # Stop the broker event-loop pump thread before process exit.
            # ``call_soon_threadsafe`` is required because the loop runs on
            # a different thread than the one issuing the stop.
            if broker_event_loop is not None:
                try:
                    broker_event_loop.call_soon_threadsafe(broker_event_loop.stop)
                except RuntimeError:
                    pass
                if broker_event_loop_thread is not None:
                    broker_event_loop_thread.join(timeout=5.0)
                broker_event_loop.close()
