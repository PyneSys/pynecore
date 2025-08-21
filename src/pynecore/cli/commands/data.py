from typing import TYPE_CHECKING, TypeAlias
from pathlib import Path
from enum import Enum
from datetime import datetime, timedelta, UTC

from typer import Typer, Option, Argument, Exit, secho, colors, confirm

from rich import print as rprint
from rich.console import Console
from rich.progress import (Progress, SpinnerColumn, TextColumn, BarColumn,
                           TimeElapsedColumn, TimeRemainingColumn)

from ..app import app, app_state
from ...providers import available_providers
from ...providers.provider import Provider
from ...lib.timeframe import in_seconds
from ...core.data_converter import DataConverter, SupportedFormats as InputFormats
from ...core.ohlcv_file import OHLCVReader

from ...utils.rich.date_column import DateColumn
from pynecore.core.ohlcv_file import OHLCVReader, OHLCVWriter
from pynecore.core.aggregator import TimeframeAggregator

__all__ = []

app_data = Typer(help="OHLCV related commands")
app.add_typer(app_data, name="data")

# Available intervals (The same fmt as described in timeframe.period)
TimeframeEnum = Enum('Timeframe', {name: name for name in ('1', '5', '15', '30', '60', '240', '1D', '1W', '1M')})

# Trick to avoid type checking errors
if TYPE_CHECKING:
    DateOrDays: TypeAlias = datetime


    class AvailableProvidersEnum(Enum):
        ...

else:
    # DateOrDays is either a datetime or a number of days
    DateOrDays = str

    # Create an enum from available providers
    AvailableProvidersEnum = Enum('Provider', {name.upper(): name.lower() for name in available_providers})


# Available output formats
class OutputFormat(Enum):
    CSV = 'csv'
    JSON = 'json'


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
    Parse a date or a number of days
    """
    if value == 'continue':
        return value
    if not value:
        return datetime.now(UTC).replace(second=0, microsecond=0)
    try:
        # Is it a date?
        return datetime.fromisoformat(str(value))
    except ValueError:
        try:
            # Not a date, maybe it's a number of days
            days = int(value)
            if days < 0:
                secho("Error: Days cannot be negative", err=True, fg=colors.RED)
                raise Exit(1)
            return (datetime.now(UTC) - timedelta(days=days)).replace(second=0, microsecond=0)
        except ValueError:
            secho(f"Error: Invalid date fmt or days number: {value}", err=True, fg=colors.RED)
            raise Exit(1)


@app_data.command()
def download(
        provider: AvailableProvidersEnum = Argument(..., case_sensitive=False, show_default=False,
                                                    help="Data provider"),
        symbol: str | None = Option(None, '--symbol', '-s', show_default=False,
                                    help="Symbol (e.g. BYBIT:BTC/USDT:USDT)"),
        list_symbols: bool = Option(False, '--list-symbols', '-ls',
                                    help="List available symbols of the provider"),
        timeframe: str = Option('1D', '--timeframe', '-tf', callback=validate_timeframe,
                                help="Timeframe in TradingView format (e.g., '1', '5S', '1D', '1W')"),
        time_from: DateOrDays = Option("continue", '--from', '-f',
                                       callback=parse_date_or_days, formats=[],
                                       metavar="[%Y-%m-%d|%Y-%m-%d %H:%M:%S|NUMBER]|continue",
                                       help="Start date or days back from now, or 'continue' to resume last download,"
                                            " or one year if no data"),
        time_to: DateOrDays = Option(datetime.now(UTC).replace(second=0, microsecond=0), '--to', '-t',
                                     callback=parse_date_or_days, formats=[],
                                     metavar="[%Y-%m-%d|%Y-%m-%d %H:%M:%S|NUMBER]",
                                     help="End date or days from start date"),
        show_info: bool = Option(False, '--symbol-info', '-si', help="Show symbol info"),
        force_save_info: bool = Option(False, '--force-save-info', '-fi',
                                       help="Force save symbol info"),
        truncate: bool = Option(False, '--truncate', '-tr',
                                help="Truncate file before downloading, all data will be lost"),
):
    """
    Download historical OHLCV data
    """
    # Import provider module from
    provider_module = __import__(f"pynecore.providers.{provider.value}", fromlist=[''])
    provider_class = getattr(provider_module, [p for p in dir(provider_module) if p.endswith('Provider')][0])

    try:
        # If list_symbols is True, we show the available symbols then exit
        if list_symbols:
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
                progress.add_task(description="Fetching market data...", total=None)
                provider_instance: Provider = provider_class(symbol=symbol, config_dir=app_state.config_dir)
                symbols = provider_instance.get_list_of_symbols()
            with (console := Console()).pager():
                for s in symbols:
                    console.print(s)
            return

        if not symbol:
            secho("Error: Symbol is required!", err=True, fg=colors.RED)
            raise Exit(1)

        # Create provider instance
        provider_instance: Provider = provider_class(symbol=symbol, timeframe=timeframe,
                                                     ohlv_dir=app_state.data_dir)

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
            if time_from == "continue":
                if ohlcv_writer.end_timestamp:  # Resume from last download
                    time_from = datetime.fromtimestamp(ohlcv_writer.end_timestamp, UTC)
                    # We need to add one interval to the start date to avoid downloading the same data
                    time_from += timedelta(seconds=ohlcv_writer.interval)
                else:  # No data, download one year as default
                    time_from = datetime.now(UTC) - timedelta(days=365)

            # We need to remove timezone info
            time_from = time_from.replace(tzinfo=None)
            time_to = time_to.replace(tzinfo=None)

            # We cannot download data from the future otherwise it would take very long
            if time_to > datetime.now(UTC).replace(tzinfo=None):
                time_to = datetime.now(UTC).replace(tzinfo=None)

            # Check time range
            if time_to < time_from:
                secho("Error: End date (to) must be greater than start date (from)!", err=True, fg=colors.RED)
                raise Exit(1)

            # If the start date is before the start of the existing file, we truncate the file
            if ohlcv_writer.start_timestamp:
                if time_from < ohlcv_writer.start_datetime.replace(tzinfo=None):
                    secho(f"The start date (from: {time_from}) is before the start of the "
                          f"existing file ({ohlcv_writer.start_datetime.replace(tzinfo=None)}).\n"
                          f"If you continue, the file will be truncated.",
                          fg=colors.YELLOW)
                    confirm("Do you want to continue?", abort=True)
                    # Truncate file
                    ohlcv_writer.seek(0)
                    ohlcv_writer.truncate()

            total_seconds = int((time_to - time_from).total_seconds())

            # Get OHLCV data
            with Progress(
                    SpinnerColumn(finished_text="[green]✓"),
                    TextColumn("{task.description}"),
                    DateColumn(time_from),
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
                    elapsed_seconds = int((current_time - time_from).total_seconds())
                    progress.update(task, completed=elapsed_seconds)

                # Start downloading
                provider_instance.download_ohlcv(time_from, time_to, on_progress=cb_progress)

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
        source_file: Path = Argument(...,
                                     help="Source .ohlcv file (auto-searches in workdir/data/ if only filename given)"),
        target_timeframe: TimeframeEnum = Option(..., '--target-timeframe', '-tf',  # type: ignore
                                                 help="Target timeframe: 1,5,15,30,60,240 (minutes), 1D (daily), 1W (weekly), 1M (monthly). Must be larger than source."),
        output: Path | None = Option(None, '--output', '-o',
                                     help="Custom output file path (auto-generated if not specified)"),
        truncate: bool = Option(False, '--truncate', '-tr',
                                help="Truncate/overwrite existing target file"),
        force: bool = Option(False, '--force', '-f',
                             help="Force re-aggregation even if target file is newer than source"),

):
    """
    Aggregate OHLCV candlestick data from smaller to larger timeframes.
    
    WHAT IT DOES:
    Combines multiple small timeframe candles into fewer large timeframe candles.
    For example: 12 five-minute candles → 1 one-hour candle.
    
    AGGREGATION RULES:
    • Open: First candle's open price
    • High: Highest price across all candles
    • Low: Lowest price across all candles  
    • Close: Last candle's close price
    • Volume: Sum of all volumes
    
    SUPPORTED TIMEFRAMES:
    Minutes: 1, 5, 15, 30, 60, 240 (where 60=1hour, 240=4hours)
    Daily: 1D | Weekly: 1W | Monthly: 1M
    
    IMPORTANT: Only upscaling supported (small → large timeframes).
    Downscaling (1H → 5min) is impossible without creating artificial data.
    
    FILE HANDLING:
    • Source file must have matching .toml metadata file
    • Output filename auto-generated if not specified
    • Skips aggregation if target is newer (use --force to override)
    
    EXAMPLES:
      # Basic aggregation (searches workdir/data/ automatically)
      pyne data aggregate btc_5.ohlcv --target-timeframe 60
      
      # Full path with custom output
      pyne data aggregate /path/to/data_1.ohlcv -tf 5 -o custom_5min.ohlcv
      
      # Force overwrite existing target
      pyne data aggregate data_60.ohlcv --target-timeframe 1D --force
      
      # Truncate existing file (start fresh)
      pyne data aggregate data_5.ohlcv -tf 1D --truncate
    """

    # Expand data path - if only filename is provided, look in workdir/data
    if len(source_file.parts) == 1:
        source_file = app_state.data_dir / source_file

    # Validate source file
    if not source_file.exists():
        secho(f"Error: Source file not found: {source_file}", err=True, fg=colors.RED)
        raise Exit(1)

    if source_file.suffix != '.ohlcv':
        secho(f"Error: Source file must be .ohlcv format, got: {source_file.suffix}",
              err=True, fg=colors.RED)
        raise Exit(1)

    # Check for required .toml metadata file
    source_toml = source_file.with_suffix('.toml')
    if not source_toml.exists():
        secho(f"Error: Required .toml metadata file not found: {source_toml}",
              err=True, fg=colors.RED)
        secho("Aggregation requires symbol metadata. Please ensure the .toml file exists.",
              err=True, fg=colors.YELLOW)
        raise Exit(1)

    # Extract source timeframe from filename
    source_timeframe = _extract_timeframe_from_filename(source_file)
    if not source_timeframe:
        secho(f"Error: Cannot determine source timeframe from filename: {source_file.name}",
              err=True, fg=colors.RED)
        raise Exit(1)

    # Generate target file path if not provided
    if output is None:
        output = _generate_target_filename(source_file, target_timeframe.value)

    # Check if target file is newer than source (unless force is used)
    if not force and not truncate and output.exists():
        source_mtime = source_file.stat().st_mtime
        target_mtime = output.stat().st_mtime
        if target_mtime > source_mtime:
            secho(f"Target file is newer than source. Use --force to re-aggregate.",
                  fg=colors.YELLOW)
            return

    try:
        # Create aggregator
        aggregator = TimeframeAggregator(source_timeframe, target_timeframe.value)

        # Show aggregation info
        rprint(f"[bold]Aggregating OHLCV Data[/bold]")
        rprint(f"Source: {source_file}")
        rprint(f"Target: {output}")
        rprint(f"Timeframe: {source_timeframe} → {target_timeframe.value}")
        rprint(f"Window size: {aggregator.window_size} candles")

        # Get source file size for progress tracking
        with OHLCVReader(source_file) as reader:
            total_candles = reader.size

        # Perform aggregation with progress tracking
        with Progress(
                SpinnerColumn(finished_text="[green]✓"),
                TextColumn("{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                "/",
                TimeRemainingColumn(),
        ) as progress:

            task = progress.add_task(
                description="Aggregating candles...",
                total=total_candles
            )

            # Perform aggregation
            result = aggregator.aggregate_file(
                source_path=source_file,
                target_path=output,
                truncate=truncate
            )

            # Update progress to completion
            progress.update(task, completed=total_candles)

        # Handle .toml metadata file
        source_toml = source_file.with_suffix('.toml')
        target_toml = output.with_suffix('.toml')

        if source_toml.exists():
            try:
                # Load source symbol info
                from ...core.syminfo import SymInfo
                syminfo = SymInfo.load_toml(source_toml)

                # Update period to match target timeframe
                syminfo.period = target_timeframe.value

                # Save updated symbol info to target location
                syminfo.save_toml(target_toml)

                rprint(f"Symbol info copied: {target_toml}")
            except Exception as e:
                secho(f"Warning: Could not copy symbol info: {e}", fg=colors.YELLOW)

        # Show results
        rprint(f"\n[bold green]Aggregation Complete![/bold green]")
        rprint(f"Processed: {result.candles_processed:,} candles")
        rprint(f"Aggregated: {result.candles_aggregated:,} candles")
        rprint(f"Duration: {result.duration_seconds:.2f} seconds")
        rprint(f"Output: {result.target_path}")
        if source_toml.exists():
            rprint(f"Metadata: {target_toml}")

    except ValueError as e:
        secho(f"Error: {e}", err=True, fg=colors.RED)
        raise Exit(1)
    except Exception as e:
        secho(f"Unexpected error: {e}", err=True, fg=colors.RED)
        raise Exit(2)


def _extract_timeframe_from_filename(file_path: Path) -> str | None:
    """
    Extract timeframe from OHLCV filename.
    :param file_path: Path to OHLCV file
    :return: Timeframe string or None if not found
    """
    name_parts = file_path.stem.split('_')
    if len(name_parts) >= 2:
        timeframe = name_parts[-1]  # Last part should be timeframe

        # Convert common timeframe formats to PyneCore format
        # e.g., '5min' -> '5', '1hour' -> '1H', '1day' -> '1D'
        if timeframe.endswith('min'):
            return timeframe[:-3]  # Remove 'min' suffix
        elif timeframe.endswith('hour') or timeframe.endswith('h'):
            if timeframe.endswith('hour'):
                return timeframe[:-4] + 'H'  # Replace 'hour' with 'H'
            else:
                return timeframe[:-1] + 'H'  # Replace 'h' with 'H'
        elif timeframe.endswith('day') or timeframe.endswith('d'):
            if timeframe.endswith('day'):
                return timeframe[:-3] + 'D'  # Replace 'day' with 'D'
            else:
                return timeframe[:-1] + 'D'  # Replace 'd' with 'D'
        elif timeframe.endswith('week') or timeframe.endswith('w'):
            if timeframe.endswith('week'):
                return timeframe[:-4] + 'W'  # Replace 'week' with 'W'
            else:
                return timeframe[:-1] + 'W'  # Replace 'w' with 'W'
        elif timeframe.endswith('month') or timeframe.endswith('m'):
            if timeframe.endswith('month'):
                return timeframe[:-5] + 'M'  # Replace 'month' with 'M'
            else:
                return timeframe[:-1] + 'M'  # Replace 'm' with 'M'

        return timeframe  # Return as-is if no conversion needed
    return None


def _generate_target_filename(source_path: Path, target_timeframe: str) -> Path:
    """
    Generate target filename by replacing timeframe.
    :param source_path: Path to source file
    :param target_timeframe: Target timeframe string
    :return: Generated target file path
    """
    name_parts = source_path.stem.split('_')
    if len(name_parts) >= 2:
        name_parts[-1] = target_timeframe  # Replace timeframe
        new_name = '_'.join(name_parts) + '.ohlcv'
        return source_path.parent / new_name

    # Fallback: append target timeframe
    return source_path.parent / f"{source_path.stem}_{target_timeframe}.ohlcv"
