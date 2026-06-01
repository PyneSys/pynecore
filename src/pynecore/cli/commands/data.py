import sys
from typing import TYPE_CHECKING, TypeAlias, cast
from pathlib import Path
from enum import Enum
from datetime import datetime, timedelta, UTC

from typer import Typer, Option, Argument, Exit, secho, colors, confirm, BadParameter

from rich import print as rprint
from rich.progress import (Progress, SpinnerColumn, TextColumn, BarColumn,
                           TimeElapsedColumn, TimeRemainingColumn)

from ..app import app, app_state
from ...core.plugin import discover_plugins, load_plugin, PluginNotFoundError
from ...core.plugin import ProviderPlugin, ProviderError
from ...core.provider_string import is_provider_string, parse_provider_string
from ...lib.timeframe import in_seconds
from ...core.data_converter import DataConverter, SupportedFormats as InputFormats
from ...core.ohlcv_file import OHLCVReader
from ...core.aggregator import validate_aggregation, aggregate_ohlcv
from ...core.syminfo import SymInfo

from ...utils.rich.date_column import DateColumn

__all__ = ['parse_date_or_days', 'validate_timeframe']

app_data = Typer(help="OHLCV related commands")
app.add_typer(app_data, name="data")

# Trick to avoid type checking errors
if TYPE_CHECKING:
    DateOrDays: TypeAlias = datetime
else:
    # DateOrDays is either a datetime or a number of days
    DateOrDays = str


def _list_provider_names() -> list[str]:
    """Return the sorted names of all installed data-provider plugins."""
    names = []
    for name, ep in discover_plugins().items():
        try:
            cls = ep.load()
            if isinstance(cls, type) and issubclass(cls, ProviderPlugin):
                names.append(name)
        except Exception:  # noqa
            pass
    return sorted(names)


def _resolve_provider_class(provider_name: str) -> type[ProviderPlugin]:
    """
    Load a data-provider plugin by name.

    :param provider_name: Provider entry-point name (e.g. ``"ccxt"``).
    :return: The provider plugin class.
    :raises ValueError: If the name is unknown or refers to a non-provider
        plugin. The caller's ``except (ImportError, ValueError)`` prints it.
    """
    try:
        cls = load_plugin(provider_name)
    except PluginNotFoundError:
        names = ', '.join(_list_provider_names()) or '(none)'
        raise ValueError(f"Unknown provider '{provider_name}'. Available providers: {names}")
    if not (isinstance(cls, type) and issubclass(cls, ProviderPlugin)):
        raise ValueError(f"Plugin '{provider_name}' is not a data provider.")
    return cast(type[ProviderPlugin], cls)


# Available output formats
class OutputFormat(Enum):
    CSV = 'csv'
    JSON = 'json'


def _typer_validate_timeframe(value: str) -> str:
    """Typer callback wrapper: convert ValueError into a clean BadParameter."""
    try:
        return validate_timeframe(value)
    except ValueError as e:
        raise BadParameter(str(e))


def _typer_parse_date_or_days(value: str) -> "datetime | str":
    """Typer callback wrapper: convert ValueError into a clean BadParameter."""
    try:
        return parse_date_or_days(value)
    except ValueError as e:
        raise BadParameter(str(e))


# TV-compatible timeframe validation function
def validate_timeframe(value: str) -> str:
    """
    Validate TV-compatible timeframe string.

    :param value: Timeframe string to validate
    :return: Validated timeframe string
    :raises ValueError: If timeframe is invalid
    """
    value = value.upper()
    try:
        # Test if it's a valid TV timeframe by trying to convert to seconds
        in_seconds(value)
    except (ValueError, AssertionError):
        raise ValueError(
            f"Invalid timeframe: {value}. Must be a valid timeframe in TradingView format "
            f"(e.g. '1', '5', '60', '1D', '1W', '1M')."
        )
    return value


def parse_date_or_days(value: str) -> datetime | str:
    """
    Parse a date, a number of days, ``"continue"``, or ``"now"``.

    :param value: User-supplied string.
    :return: A ``datetime`` (UTC, seconds/microseconds zeroed) or the ``"continue"`` sentinel.
    :raises ValueError: If the value matches none of the accepted forms. Typer
        converts this into a clean CLI error; in-process callers (e.g. the
        symbol-browser wizard) can catch it directly.
    """
    if value == 'continue':
        return value
    if not value:
        return datetime.now(UTC).replace(second=0, microsecond=0)
    if value.strip().lower() == 'now':
        return datetime.now(UTC).replace(second=0, microsecond=0)
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        pass
    try:
        days = int(value)
    except ValueError:
        raise ValueError(f"Invalid date format or days number: {value}")
    if days < 0:
        raise ValueError("Days cannot be negative")
    return (datetime.now(UTC) - timedelta(days=days)).replace(second=0, microsecond=0)


def _format_date_default(value, fallback: str) -> str:
    """Convert a parsed ``--from`` / ``--to`` value back into a user-friendly
    string for the symbol-browser wizard's default fields.

    ``parse_date_or_days`` accepts the result as input, so the round-trip is
    semantically loss-free.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        ref = value if value.tzinfo else value.replace(tzinfo=UTC)
        if abs((datetime.now(UTC) - ref).total_seconds()) < 60:
            return "now"
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return fallback


@app_data.command()
def download(
        provider: str = Argument(..., show_default=False,
                                 help="Provider name (e.g. 'ccxt') or a full provider string "
                                      "(e.g. 'ccxt:BYBIT:BTC/USDT:USDT@1D')"),
        symbol: str | None = Option(None, '--symbol', '-s', show_default=False,
                                    help="Symbol (e.g. BYBIT:BTC/USDT:USDT). Ignored when the "
                                         "provider string already contains a symbol"),
        list_symbols: bool = Option(False, '--list-symbols', '-ls',
                                    help="List available symbols of the provider"),
        list_brokers: bool = Option(False, '--list-brokers', '-lb',
                                    help="List available brokers/exchanges of a multi-broker provider"),
        timeframe: str = Option('1D', '--timeframe', '-tf', callback=_typer_validate_timeframe,
                                help="Timeframe in TradingView format (e.g., '1', '5S', '1D', '1W'). "
                                     "Ignored when the provider string contains an @timeframe"),
        time_from: DateOrDays = Option("continue", '--from', '-f',
                                       callback=_typer_parse_date_or_days, formats=[],
                                       metavar="[%Y-%m-%d|%Y-%m-%d %H:%M:%S|NUMBER]|continue",
                                       help="Start date or days back from now, or 'continue' to resume last download,"
                                            " or one year if no data"),
        time_to: DateOrDays = Option(datetime.now(UTC).replace(second=0, microsecond=0), '--to', '-t',
                                     callback=_typer_parse_date_or_days, formats=[],
                                     metavar="[%Y-%m-%d|%Y-%m-%d %H:%M:%S|NUMBER]",
                                     help="End date or days from start date"),
        show_info: bool = Option(False, '--symbol-info', '-si', help="Show symbol info"),
        force_save_info: bool = Option(False, '--force-save-info', '-fi',
                                       help="Force save symbol info"),
        truncate: bool = Option(False, '--truncate', '-tr',
                                help="Truncate file before downloading, all data will be lost"),
        chunk_size: int | None = Option(None, '--chunk-size', '-cs',
                                        help="Number of bars to download per API request. "
                                             "Overrides automatic detection based on exchange limits. "
                                             "Useful for exchanges with timeframe-specific limits (e.g., Bitget 1w: 12). "
                                             "Lower values = slower but safer, higher values = faster but may hit API limits."),
        extra_data: bool = Option(False, '--extra-data/--no-extra-data', '-ed',
                                  help="Also download provider extra fields (ask/bid/spread) into a "
                                       ".extra.csv sidecar. Off by default: extra fields cost extra "
                                       "requests and slow every later backtest that loads them. "
                                       "Ignored by providers that have no extra fields."),
):
    """
    Download historical OHLCV data

    The provider can be given either as a bare name plus ``-s``/``-tf`` flags
    (``pyne data download ccxt -s BYBIT:BTC/USDT:USDT -tf 1D``) or as a single
    provider string in the same syntax as ``pyne run``
    (``pyne data download ccxt:BYBIT:BTC/USDT:USDT@1D``). Both forms are equivalent.
    """
    try:
        from ...core.config import ensure_config

        # Resolve the provider plugin from either the bare name or the leading
        # segment of a provider string.
        string_mode = is_provider_string(provider)
        provider_name = (provider.split(':', 1)[0] if string_mode else provider).lower()
        provider_class = _resolve_provider_class(provider_name)

        # --list-brokers needs only the provider plugin
        if list_brokers:
            try:
                brokers = provider_class.get_list_of_brokers()
            except NotImplementedError:
                secho(f"Provider '{provider_name}' does not support listing brokers.",
                      err=True, fg=colors.RED)
                raise Exit(1)
            except ProviderError as e:
                secho(f"Error: {e}", err=True, fg=colors.RED)
                raise Exit(1)
            ordered = sorted(brokers)
            id_width = max((len(b.id) for b in ordered), default=0)
            for b in ordered:
                print(f"{b.id:<{id_width}}  {b.name}".rstrip() if b.name else b.id)
            return

        # In string mode, derive broker/symbol/timeframe from the provider
        # string; the broker is re-folded into the symbol so multi-broker
        # providers (which split it off internally) receive what they expect.
        if string_mode:
            ps = parse_provider_string(provider, multi_broker=provider_class.multi_broker)
            if ps.symbol and symbol:
                raise ValueError("Symbol given both in the provider string and via "
                                 "-s/--symbol; use only one.")
            resolved_symbol = ps.symbol or symbol
            if ps.timeframe is not None:
                timeframe = validate_timeframe(ps.timeframe)
            if ps.broker:
                symbol = f"{ps.broker}:{resolved_symbol}" if resolved_symbol else ps.broker
            else:
                symbol = resolved_symbol

        config = None
        config_cls: type | None = getattr(provider_class, 'Config', None)
        if config_cls is not None:
            config = ensure_config(config_cls,
                                   app_state.config_dir / 'plugins' / f'{provider_name}.toml')

        def _fetch_symbols(sym: str | None) -> "tuple[ProviderPlugin, list[str], Path | None]":
            """Construct the provider for ``sym`` and, when it is a selector-only
            value (the provider leaves ``self.symbol`` as ``None``), fetch its
            symbol list for the TUI. Shown behind a spinner.

            :return: ``(provider_instance, symbol_list, tui_ohlcv_dir)``.
            """
            slist: list[str] = []
            tdir: Path | None = None
            with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                          transient=True) as progress:
                progress.add_task(description="Fetching market data...", total=None)
                if sym is None:
                    inst: ProviderPlugin = provider_class(symbol=None, timeframe=timeframe,
                                                          config=config)
                    tdir = app_state.data_dir
                else:
                    inst = provider_class(symbol=sym, timeframe=timeframe,
                                          ohlcv_dir=app_state.data_dir, config=config)
                if inst.symbol is None and sys.stdin.isatty():
                    slist = inst.get_list_of_symbols()
                    tdir = app_state.data_dir
            return inst, slist, tdir

        def _browse_symbols(inst: "ProviderPlugin", slist: list[str], tdir: Path,
                            *, can_go_back: bool) -> bool:
            """Launch the symbol browser TUI for ``inst``.

            :param can_go_back: When True, ESC returns to the broker picker
                instead of quitting the command.
            :return: True if the user asked to go back to the broker list.
            """
            from ..utils.symbol_browser import SymbolBrowser
            browser = SymbolBrowser(
                inst,
                slist,
                ohlcv_dir=tdir,
                default_timeframe=timeframe,
                default_from=_format_date_default(time_from, "continue"),
                default_to=_format_date_default(time_to, "now"),
                default_chunk_size=chunk_size,
                default_extra_data=extra_data,
                can_go_back=can_go_back,
            )
            browser.run()
            return browser.go_back

        provider_instance: ProviderPlugin | None = None
        symbols_list: list[str] = []
        tui_ohlcv_dir: Path | None = None

        # Multi-broker providers (CCXT, cTrader) need a broker chosen before a
        # symbol can be browsed. When none was given and we're interactive, drop
        # into a broker picker first, then the symbol browser. The two form a
        # loop: ESC in the symbol browser returns to the broker list rather than
        # exiting, and a failure opening the chosen broker (e.g. an exchange that
        # needs API credentials) returns to the picker with the error shown so
        # another broker can be tried. Only the picker itself exits the command.
        if (provider_class.multi_broker and symbol is None
                and not list_symbols and sys.stdin.isatty()):
            try:
                with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                              transient=True) as progress:
                    progress.add_task(description="Fetching available brokers...", total=None)
                    brokers = provider_class.get_list_of_brokers()
            except (NotImplementedError, ProviderError) as e:
                secho(f"Error: {e}", err=True, fg=colors.RED)
                raise Exit(1)
            from ..utils.broker_picker import BrokerPicker
            display_name = getattr(provider_class, 'plugin_name', provider_name)
            picker = BrokerPicker(sorted(brokers), provider_name=display_name)
            while True:
                chosen = picker.run()
                if chosen is None:
                    return
                try:
                    provider_instance, symbols_list, tui_ohlcv_dir = _fetch_symbols(chosen)
                except (NotImplementedError, ProviderError) as e:
                    picker.error = str(e)
                    continue
                picker.error = None
                assert tui_ohlcv_dir is not None
                if _browse_symbols(provider_instance, symbols_list, tui_ohlcv_dir,
                                   can_go_back=True):
                    continue
                return

        # If list_symbols is True, we show the available symbols then exit
        if list_symbols:
            try:
                with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                              transient=True) as progress:
                    progress.add_task(description="Fetching market data...", total=None)
                    provider_instance = provider_class(symbol=symbol, config=config)
                    symbols = provider_instance.get_list_of_symbols()
            except (NotImplementedError, ProviderError) as e:
                secho(f"Error: {e}", err=True, fg=colors.RED)
                raise Exit(1)
            for s in symbols:
                print(s)
            return

        # Some providers (e.g. CCXT) accept a selector-only ``--symbol`` (just
        # the exchange name): the constructor recognises it and leaves
        # ``self.symbol`` as ``None``. That signals "we know which backend but
        # not which instrument yet" — drop into the TUI to let the user pick.
        # Otherwise proceed to the download path.
        try:
            provider_instance, symbols_list, tui_ohlcv_dir = _fetch_symbols(symbol)
        except NotImplementedError as e:
            secho(f"Error: {e}", err=True, fg=colors.RED)
            secho("Pass a symbol explicitly with -s/--symbol.",
                  err=True, fg=colors.YELLOW)
            raise Exit(1)
        except ProviderError as e:
            secho(f"Error: {e}", err=True, fg=colors.RED)
            raise Exit(1)

        if provider_instance.symbol is None:
            if not sys.stdin.isatty():
                secho("Error: Symbol is required "
                      "(or use --list-symbols for non-interactive listing).",
                      err=True, fg=colors.RED)
                raise Exit(1)
            assert tui_ohlcv_dir is not None
            _browse_symbols(provider_instance, symbols_list, tui_ohlcv_dir,
                            can_go_back=False)
            return

        # Download symbol info if not exists
        if force_save_info or not provider_instance.is_symbol_info_exists():
            with Progress(SpinnerColumn(finished_text="[green]✓"), TextColumn("{task.description}")) as progress:
                # Get symbol info task
                task = progress.add_task(description="Fetching symbol info...", total=1)
                sym_info = provider_instance.get_symbol_info(force_update=force_save_info)

                # Complete task
                progress.update(task, completed=1)

                # Print symbol info
                if show_info:
                    rprint(sym_info)
        else:  # We have symbol info, just show it
            sym_info = provider_instance.get_symbol_info()
            if show_info:
                rprint(sym_info)

        # Open the OHLCV file and start downloading
        with provider_instance as ohlcv_writer:
            # Truncate file if overwrite is True
            if truncate:
                ohlcv_writer.seek(0)
                ohlcv_writer.truncate()

            # If the start date is "continue" (default), we resume from the last download
            resolved_from: datetime = time_from
            fetch_all = False
            if time_from == "continue":
                end_ts = ohlcv_writer.end_timestamp
                interval = ohlcv_writer.interval
                if end_ts and interval:  # Resume from last download
                    resolved_from = datetime.fromtimestamp(end_ts, UTC)
                    # We need to add one interval to the start date to avoid downloading the same data
                    resolved_from += timedelta(seconds=interval)
                elif getattr(provider_class, 'fetch_all_by_default', False):
                    resolved_from = datetime.fromtimestamp(0, UTC)
                    fetch_all = True
                else:  # No data, download one year as default
                    resolved_from = datetime.now(UTC) - timedelta(days=365)

            # We need to remove timezone info
            resolved_from = resolved_from.replace(tzinfo=None)
            time_to = time_to.replace(tzinfo=None)

            # We cannot download data from the future otherwise it would take very long
            if time_to > datetime.now(UTC).replace(tzinfo=None):
                time_to = datetime.now(UTC).replace(tzinfo=None)

            # Check time range (skip for fetch_all providers)
            if not fetch_all and time_to < resolved_from:
                secho("Error: End date (to) must be greater than start date (from)!", err=True, fg=colors.RED)
                raise Exit(1)

            # If the start date is before the start of the existing file, we truncate the file
            if ohlcv_writer.start_timestamp and not fetch_all:
                if resolved_from < ohlcv_writer.start_datetime.replace(tzinfo=None):
                    secho(f"The start date (from: {resolved_from}) is before the start of the "
                          f"existing file ({ohlcv_writer.start_datetime.replace(tzinfo=None)}).\n"
                          f"If you continue, the file will be truncated.",
                          fg=colors.YELLOW)
                    confirm("Do you want to continue?", abort=True)
                    # Truncate file
                    ohlcv_writer.seek(0)
                    ohlcv_writer.truncate()

            # fetch_all provider: use spinner-only progress (no time-based progress bar)
            if fetch_all:
                with Progress(
                        SpinnerColumn(finished_text="[green]✓"),
                        TextColumn("{task.description}"),
                        TimeElapsedColumn(),
                ) as progress:
                    progress.add_task(
                        description="Downloading all available OHLCV data...",
                        total=None,
                    )
                    provider_instance.download_ohlcv(resolved_from, time_to, on_progress=None,
                                                     limit=chunk_size, with_extra=extra_data)
            else:
                total_seconds = int((time_to - resolved_from).total_seconds())

                # Get OHLCV data
                with Progress(
                        SpinnerColumn(finished_text="[green]✓"),
                        TextColumn("{task.description}"),
                        DateColumn(resolved_from),
                        BarColumn(),
                        TimeElapsedColumn(),
                        "/",
                        TimeRemainingColumn(),
                ) as progress:
                    task = progress.add_task(
                        description="Downloading OHLCV data...",
                        total=total_seconds,
                    )

                    def cb_progress(current_time: datetime):
                        """ Callback to update progress """
                        elapsed_seconds = int((current_time - resolved_from).total_seconds())
                        progress.update(task, completed=elapsed_seconds)

                    # Start downloading
                    provider_instance.download_ohlcv(resolved_from, time_to, on_progress=cb_progress,
                                                     limit=chunk_size, with_extra=extra_data)

    except ProviderError as e:
        secho(f"Error: {e}", err=True, fg=colors.RED)
        raise Exit(1)
    except (ImportError, ValueError) as e:
        secho(str(e), err=True, fg=colors.RED)
        raise Exit(2)


@app_data.command()
def convert_to(
        ohlcv_path: Path = Argument(..., dir_okay=False, file_okay=True,
                                    help="Data file to convert (*.ohlcv)"),
        fmt: OutputFormat = Option(
            'csv', '--format', '-f',
            case_sensitive=False,
            help="Output format"),
        as_datetime: bool = Option(False, '--as-datetime', '-dt',
                                   help="Save timestamp as datetime instead of UNIX timestamp"),
):
    """
    Convert downloaded data from pyne's OHLCV format to another format
    """
    # Check file format and extension
    if ohlcv_path.suffix == "":
        # No extension, add .ohlcv
        ohlcv_path = ohlcv_path.with_suffix(".ohlcv")

    # Expand data path
    if len(ohlcv_path.parts) == 1:
        ohlcv_path = app_state.data_dir / ohlcv_path
    # Check if data exists
    if not ohlcv_path.exists():
        secho(f"Data file '{ohlcv_path}' not found!", fg="red", err=True)
        raise Exit(1)

    out_path = None
    with Progress(SpinnerColumn(finished_text="[green]✓"), TextColumn("{task.description}")) as progress:
        # Convert
        with OHLCVReader(str(ohlcv_path)) as ohlcv_reader:
            if fmt.value == OutputFormat.CSV.value:
                task = progress.add_task(description="Converting to CSV...", total=1)
                out_path = str(ohlcv_path.with_suffix('.csv'))
                ohlcv_reader.save_to_csv(out_path, as_datetime=as_datetime)

            elif fmt.value == OutputFormat.JSON.value:
                task = progress.add_task(description="Converting to JSON...", total=1)
                out_path = str(ohlcv_path.with_suffix('.json'))
                ohlcv_reader.save_to_json(out_path, as_datetime=as_datetime)

            else:
                raise ValueError(f"Unsupported format: {fmt}")

            # Complete task
            progress.update(task, completed=1)

    if out_path:
        secho(f'Data file converted successfully to "{out_path}"!')


@app_data.command()
def convert_from(
        file_path: Path = Argument(..., help="Path to CSV/JSON/TXT file to convert"),
        provider: str = Option(None, '--provider', '-p',
                               help="Data provider, can be any name"),
        symbol: str | None = Option(None, '--symbol', '-s', show_default=False,
                                    help="Symbol (default: from file name)"),
        tz: str = Option('UTC', '--timezone', '-tz', help="Timezone"),
):
    """
    Convert data from other sources to pyne's OHLCV format
    """
    # Expand file path if only filename is provided (look in workdir/data)
    if len(file_path.parts) == 1:
        file_path = app_state.data_dir / file_path

    # Check if file exists
    if not file_path.exists():
        secho(f'File "{file_path}" not found!', fg=colors.RED, err=True)
        raise Exit(1)

    # Auto-detect symbol and provider from filename if not provided
    detected_symbol, detected_provider = DataConverter.guess_symbol_from_filename(file_path)

    if symbol is None:
        symbol = detected_symbol

    if provider is None and detected_provider is not None:
        provider = detected_provider

    # Fallback: peek into CSV content (Databento exports carry the symbol as a column)
    if (symbol is None or provider is None) and file_path.suffix.lower() == '.csv':
        content_symbol, content_provider = DataConverter.guess_symbol_from_csv_content(file_path)
        if symbol is None and content_symbol:
            symbol = content_symbol
        if provider is None and content_provider:
            provider = content_provider

    # Ensure we have required parameters
    if symbol is None:
        secho(f"Error: Could not detect symbol from filename '{file_path.name}'!", fg=colors.RED, err=True)
        secho("Please provide a symbol using --symbol option.", fg=colors.YELLOW, err=True)
        raise Exit(1)

    # Auto-detect file format
    fmt = file_path.suffix[1:].lower()
    if fmt not in InputFormats:
        raise ValueError(f"Unsupported file format: {file_path}")

    # Use the enhanced DataConverter for automatic conversion
    converter = DataConverter()

    try:
        with Progress(SpinnerColumn(finished_text="[green]✓"), TextColumn("{task.description}")) as progress:
            task = progress.add_task(description=f"Converting {fmt.upper()} to OHLCV format...", total=1)

            # Perform conversion with automatic TOML generation
            converter.convert_to_ohlcv(
                file_path=Path(file_path),
                provider=provider,
                symbol=symbol,
                timezone=tz,
                force=True
            )

            progress.update(task, completed=1)

    except Exception as e:
        secho(f"Error: {e}", err=True, fg=colors.RED)
        raise Exit(1)

    secho(f'Data file converted successfully to "{file_path}".')
    secho(f'A configuration file was automatically generated for you at "{file_path.with_suffix(".toml")}". '
          f'Please check it and adjust it to match your needs.')


@app_data.command()
def aggregate(
        source: Path = Argument(..., help="Source .ohlcv file (searches in workdir/data/ if only name given)"),
        timeframe: str = Option(..., '--timeframe', '-tf', callback=_typer_validate_timeframe,
                                help="Target timeframe in TradingView format (e.g., '60', '1D', '1W')"),
        output: Path | None = Option(None, '--output', '-o',
                                     help="Custom output path (auto-generated if not specified)"),
):
    """
    Aggregate OHLCV data from a lower timeframe to a higher one.

    Combines multiple smaller candles into larger timeframe candles.
    For example: daily candles → weekly candles, or 5-minute → 1-hour.

    The source timeframe is read from the .toml metadata file.
    Only upscaling is supported (small → large timeframe).
    """
    # Resolve source path
    if len(source.parts) == 1:
        source = app_state.data_dir / source
    if source.suffix == "":
        source = source.with_suffix(".ohlcv")

    if not source.exists():
        secho(f"Error: Source file not found: {source}", err=True, fg=colors.RED)
        raise Exit(1)

    if source.suffix != '.ohlcv':
        secho(f"Error: Source must be .ohlcv format, got: {source.suffix}", err=True, fg=colors.RED)
        raise Exit(1)

    # Read source timeframe from TOML
    toml_path = source.with_suffix('.toml')
    if not toml_path.exists():
        secho(f"Error: Metadata file not found: {toml_path}", err=True, fg=colors.RED)
        raise Exit(1)

    try:
        syminfo = SymInfo.load_toml(toml_path)
    except Exception as e:
        secho(f"Error reading metadata: {e}", err=True, fg=colors.RED)
        raise Exit(1)

    source_tf = syminfo.period

    # Validate timeframe compatibility
    try:
        validate_aggregation(source_tf, timeframe)
    except ValueError as e:
        secho(f"Error: {e}", err=True, fg=colors.RED)
        raise Exit(1)

    # Generate output path if not specified
    if output is None:
        # Replace the timeframe suffix in the filename: symbol_1D.ohlcv → symbol_1W.ohlcv
        stem = source.stem
        # If the stem ends with the source timeframe, replace it
        if stem.endswith(f"_{source_tf}"):
            new_stem = stem[:-len(source_tf)] + timeframe
        else:
            new_stem = f"{stem}_{timeframe}"
        out_path: Path = source.parent / f"{new_stem}.ohlcv"
    else:
        out_path = output

    if len(out_path.parts) == 1:
        out_path = app_state.data_dir / out_path

    if out_path.suffix == "":
        out_path = out_path.with_suffix(".ohlcv")

    # Confirm before overwriting existing file
    if out_path.exists():
        secho(f"Target file already exists: {out_path.name}", fg=colors.YELLOW)
        confirm("Overwrite?", abort=True)

    # Perform aggregation
    with Progress(
            SpinnerColumn(finished_text="[green]✓"),
            TextColumn("{task.description}"),
    ) as progress:
        progress.add_task(
            description=f"Aggregating {source_tf} → {timeframe}...",
            total=None,
        )

        try:
            # Use data timezone from TOML for correct day/week/month boundaries
            from zoneinfo import ZoneInfo
            data_tz = ZoneInfo(syminfo.timezone) if syminfo.timezone else None
            source_count, target_count = aggregate_ohlcv(
                source, out_path, timeframe, tz=data_tz)
        except Exception as e:
            secho(f"Error during aggregation: {e}", err=True, fg=colors.RED)
            raise Exit(1)

    # Copy and update TOML for the target file
    target_toml = out_path.with_suffix('.toml')
    try:
        syminfo.period = timeframe
        syminfo.save_toml(target_toml)
    except Exception as e:
        secho(f"Warning: Could not write metadata: {e}", fg=colors.YELLOW)

    secho(f"Aggregated {source_count:,} → {target_count:,} candles ({source_tf} → {timeframe})")
    secho(f'Output: "{out_path}"')
