<!--
---
weight: 10004
title: "Practical Examples"
description: "Real-world examples showing how to use PyneCore programmatically"
icon: "integration_instructions"
date: "2025-08-19"  
lastmod: "2025-08-19"
draft: false
toc: true
categories: ["Advanced", "API"]
tags: ["examples", "practical", "workflows", "automation", "data-download"]
---
-->

# Practical Examples

This page demonstrates real-world applications of PyneCore's programmatic API. These examples show how to build
automation systems, batch processing workflows, and trading applications using Python code instead of CLI commands.

## Data Management Examples

### Download Market Data

Programmatically download data equivalent to `pyne data download ccxt --symbol "BYBIT:BTC/USDT:USDT"`.

```python
from pynecore.providers.ccxt import CCXTProvider
from pynecore.core.ohlcv_file import OHLCVWriter
from datetime import datetime, UTC, timedelta
from pathlib import Path


def download_crypto_data(symbol: str, days_back: int = 365, timeframe: str = "1D"):
    """
    Download cryptocurrency data for the specified period
    """

    # Calculate date range
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=days_back)

    # Initialize provider
    provider = CCXTProvider(
        symbol=symbol,
        config_dir=Path("./config")  # Contains ccxt.toml with exchange credentials
    )

    # Create output filename
    safe_symbol = symbol.replace(":", "_").replace("/", "_")
    output_file = Path(f"{safe_symbol}_{timeframe}.ohlcv")

    print(f"Downloading {symbol} data from {start_date.date()} to {end_date.date()}")

    # Download and save data
    with OHLCVWriter(output_file, truncate=True) as writer:
        candle_count = 0
        for candle in provider.download_ohlcv(
                timeframe=timeframe,
                since=start_date,
                until=end_date
        ):
            writer.write(candle)
            candle_count += 1

            if candle_count % 100 == 0:
                print(f"Downloaded {candle_count} candles...")

    # Save symbol information  
    syminfo = provider.get_syminfo(timeframe=timeframe)
    syminfo_file = output_file.with_suffix(".toml")
    syminfo.save_toml(syminfo_file)

    print(f"✓ Downloaded {candle_count} candles to {output_file}")
    print(f"✓ Saved symbol info to {syminfo_file}")

    return output_file, syminfo_file


# Usage
if __name__ == "__main__":
    download_crypto_data("BYBIT:BTC/USDT:USDT", days_back=365)
```

### Convert CSV to Binary OHLCV

Equivalent to `pyne data convert-from data.csv --symbol BTCUSD`.

```python
from pynecore.core.data_converter import DataConverter
from pynecore.core.ohlcv_file import OHLCVWriter
from pathlib import Path


def convert_csv_with_auto_analysis(csv_path: Path, symbol: str = None, provider: str = None):
    """
    Convert CSV file using enhanced DataConverter with automatic analysis.
    Equivalent to: pyne data convert-from data.csv
    """
    converter = DataConverter()

    # Auto-detect symbol and provider if not provided
    if not symbol or not provider:
        detected_symbol, detected_provider = DataConverter.guess_symbol_from_filename(csv_path)
        symbol = symbol or detected_symbol or csv_path.stem.upper()
        provider = provider or detected_provider or "custom"

    print(f"Converting {csv_path} with symbol: {symbol}, provider: {provider}")

    # Convert with automatic TOML generation and analysis
    converter.convert_to_ohlcv(
        file_path=csv_path,
        provider=provider,
        symbol=symbol,
        timezone="UTC",
        force=True
    )

    # The converter automatically creates both .ohlcv and .toml files
    ohlcv_path = csv_path.with_suffix(".ohlcv")
    toml_path = csv_path.with_suffix(".toml")

    print(f"✓ Converted to {ohlcv_path}")
    print(f"✓ Auto-generated configuration: {toml_path}")

    # Show advanced analysis results if available
    with OHLCVWriter(ohlcv_path) as writer:
        if writer.analyzed_tick_size:
            print(f"✓ Detected tick size: {writer.analyzed_tick_size}")
            print(f"✓ Detected price scale: {writer.analyzed_price_scale}")

    return ohlcv_path, toml_path


# Usage examples
# Auto-detection from filename
convert_csv_with_auto_analysis(Path("BTCUSDT_1h.csv"))

# Manual specification  
convert_csv_with_auto_analysis(
    Path("market_data.csv"),
    symbol="AAPL",
    provider="yahoo"
)


# Legacy manual conversion (for custom requirements)
def manual_csv_conversion(csv_path: Path, timezone: str = "UTC"):
    """
    Manual conversion with custom settings
    """
    with OHLCVWriter(csv_path.with_suffix(".ohlcv"), truncate=True) as writer:
        writer.load_from_csv(
            csv_path,
            timestamp_column="date",  # Adjust as needed
            tz=timezone
        )

        # Access automatic analysis results
        print(f"Tick size analysis: {writer.analyzed_tick_size}")
        print(f"Price scale: {writer.analyzed_price_scale}")

    print(f"✓ Manual conversion completed: {csv_path.with_suffix('.ohlcv')}")


manual_csv_conversion(Path("custom_data.csv"))
```

### Automatic Data Format Handling

Handle multiple data formats automatically, just like the enhanced CLI does.

```python
from pynecore.core.data_converter import DataConverter
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.syminfo import SymInfo
from pathlib import Path


def run_script_with_any_format(script_path: Path, data_path: Path):
    """
    Run script with automatic data conversion, equivalent to:
    pyne run script.py data.csv  (where CLI auto-converts CSV to OHLCV)
    """
    original_data_path = data_path

    # Check if conversion is needed
    if data_path.suffix.lower() in ['.csv', '.json', '.txt']:
        print(f"Detected {data_path.suffix} file, converting automatically...")

        converter = DataConverter()

        # Only convert if needed (file is newer than existing OHLCV)
        if converter.is_conversion_required(data_path):
            # Auto-detect symbol and provider
            detected_symbol, detected_provider = DataConverter.guess_symbol_from_filename(data_path)

            print(f"Auto-detected: Symbol={detected_symbol}, Provider={detected_provider}")

            # Convert with smart defaults
            converter.convert_to_ohlcv(
                file_path=data_path,
                provider=detected_provider or "auto-detected",
                symbol=detected_symbol or data_path.stem.upper(),
                force=True
            )
            print(f"✓ Converted {data_path.suffix} to OHLCV format")
        else:
            print("✓ Using existing up-to-date OHLCV file")

        # Update path to point to OHLCV file
        data_path = data_path.with_suffix('.ohlcv')

    # Load symbol information (auto-generated or existing)
    syminfo_path = data_path.with_suffix('.toml')
    if not syminfo_path.exists():
        raise FileNotFoundError(f"Symbol info not found: {syminfo_path}")

    syminfo = SymInfo.load_toml(syminfo_path)

    # Run script normally
    print(f"Running {script_path.name} with {original_data_path.name}")
    with OHLCVReader(data_path) as reader:
        print(f"Data range: {reader.start_datetime} to {reader.end_datetime} ({reader.size:,} bars)")

        runner = ScriptRunner(script_path, reader, syminfo)
        runner.run()

    print("✓ Script execution completed!")


def batch_run_multiple_formats(script_path: Path, data_files: list[Path]):
    """
    Run the same script on multiple data files of different formats
    """

    for data_file in data_files:
        print(f"\n{'=' * 50}")
        print(f"Processing: {data_file}")
        try:
            run_script_with_any_format(script_path, data_file)
        except Exception as e:
            print(f"❌ Error processing {data_file}: {e}")
            continue


# Usage examples
script = Path("./strategies/bollinger_bands.py")

# Single file with automatic conversion
run_script_with_any_format(script, Path("BTCUSDT_1h.csv"))

# Batch processing with mixed formats
data_files = [
    Path("BTCUSDT_1h.csv"),  # CSV - will be auto-converted
    Path("ETHUSD_daily.json"),  # JSON - will be auto-converted  
    Path("GBPUSD_1D.ohlcv"),  # OHLCV - used directly
    Path("data.txt"),  # TXT - will be auto-converted
]

batch_run_multiple_formats(script, data_files)
```

### Advanced Data Analysis During Import

Leverage PyneCore's advanced analysis capabilities during data import.

```python
from pynecore.core.ohlcv_file import OHLCVWriter
from pynecore.core.syminfo import SymInfo
from pathlib import Path


def analyze_and_import_data(data_path: Path, symbol: str = None):
    """
    Import data with advanced analysis and automatic TOML generation
    """
    print(f"Importing and analyzing {data_path}...")

    # Import with analysis
    ohlcv_path = data_path.with_suffix('.ohlcv')
    with OHLCVWriter(ohlcv_path, truncate=True) as writer:
        # Load data based on file type
        if data_path.suffix.lower() == '.csv':
            writer.load_from_csv(data_path, tz='UTC')
        elif data_path.suffix.lower() == '.json':
            writer.load_from_json(data_path, tz='UTC')
        elif data_path.suffix.lower() == '.txt':
            writer.load_from_txt(data_path, tz='UTC')

        # Access analysis results
        analysis_results = {
            'bars_imported': writer.size,
            'date_range': f"{writer.start_datetime} to {writer.end_datetime}",
            'tick_size': writer.analyzed_tick_size,
            'price_scale': writer.analyzed_price_scale,
            'min_move': writer.analyzed_min_move,
        }

        print(f"📊 Analysis Results:")
        print(f"   Bars imported: {analysis_results['bars_imported']:,}")
        print(f"   Date range: {analysis_results['date_range']}")
        print(f"   Detected tick size: {analysis_results['tick_size']}")
        print(f"   Price scale: {analysis_results['price_scale']}")
        print(f"   Minimum move: {analysis_results['min_move']}")

    # Create accurate TOML configuration using analysis results
    if analysis_results['tick_size'] and analysis_results['price_scale']:
        symbol_name = symbol or data_path.stem.upper()

        syminfo = SymInfo(
            prefix="AUTO",
            description=f"{symbol_name} - Auto-analyzed",
            ticker=symbol_name,
            currency="USD",  # Adjust as needed
            period="1D",  # Adjust based on your data
            type="crypto",  # Adjust as needed
            mintick=analysis_results['tick_size'],
            pricescale=analysis_results['price_scale'],
            minmove=analysis_results['min_move'] or 1,
            pointvalue=1.0,
            timezone="UTC",
            opening_hours=[],
            session_starts=[],
            session_ends=[]
        )

        toml_path = ohlcv_path.with_suffix('.toml')
        syminfo.save_toml(toml_path)

        print(f"✓ Generated accurate TOML configuration: {toml_path}")
        print(f"✓ Using analyzed tick size: {analysis_results['tick_size']}")

    return ohlcv_path, analysis_results


# Usage examples
analysis = analyze_and_import_data(Path("BTCUSDT_trades.csv"), "BTC/USDT")

# Batch analysis of multiple files
data_files = [
    Path("ETHUSDT_1h.csv"),
    Path("ADAUSDT_daily.json"),
    Path("market_data.txt")
]

for file_path in data_files:
    try:
        print(f"\n{'=' * 60}")
        ohlcv_file, analysis = analyze_and_import_data(file_path)
    except Exception as e:
        print(f"❌ Error analyzing {file_path}: {e}")
```

## Script Execution Examples

### Batch Strategy Testing

Run multiple strategies across different time periods and symbols.

```python
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.syminfo import SymInfo
from pathlib import Path
from datetime import datetime, UTC
import pandas as pd


def batch_strategy_test(
        strategies: list[Path],
        data_files: list[Path],
        time_ranges: list[tuple[datetime, datetime]],
        output_dir: Path
):
    """
    Test multiple strategies across different data and time periods
    """

    results = []
    output_dir.mkdir(exist_ok=True)

    for strategy_path in strategies:
        for data_path in data_files:
            syminfo_path = data_path.with_suffix(".toml")
            syminfo = SymInfo.load_toml(syminfo_path)

            for start_time, end_time in time_ranges:
                print(f"Testing {strategy_path.stem} on {data_path.stem} "
                      f"from {start_time.date()} to {end_time.date()}")

                with OHLCVReader(data_path) as reader:
                    # Get data for time range
                    data_iter = reader.read_from(
                        int(start_time.timestamp()),
                        int(end_time.timestamp())
                    )

                    # Setup output paths
                    test_id = f"{strategy_path.stem}_{data_path.stem}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}"
                    trade_path = output_dir / f"{test_id}_trades.csv"
                    strat_path = output_dir / f"{test_id}_stats.csv"

                    # Run strategy
                    try:
                        runner = ScriptRunner(
                            strategy_path,
                            data_iter,
                            syminfo,
                            trade_path=trade_path,
                            strat_path=strat_path
                        )
                        runner.run()

                        # Record successful test
                        results.append({
                            'strategy': strategy_path.stem,
                            'symbol': data_path.stem,
                            'start_date': start_time.date(),
                            'end_date': end_time.date(),
                            'status': 'success',
                            'trade_file': trade_path,
                            'stats_file': strat_path
                        })

                    except Exception as e:
                        print(f"❌ Error: {e}")
                        results.append({
                            'strategy': strategy_path.stem,
                            'symbol': data_path.stem,
                            'start_date': start_time.date(),
                            'end_date': end_time.date(),
                            'status': 'error',
                            'error': str(e)
                        })

    # Save test summary
    summary_df = pd.DataFrame(results)
    summary_path = output_dir / "test_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Test summary saved to {summary_path}")

    return results


# Usage
strategies = [Path("./strategies/bollinger_bands.py"), Path("./strategies/moving_average.py")]
data_files = [Path("./data/BTCUSD_1D.ohlcv"), Path("./data/ETHUSD_1D.ohlcv")]
time_ranges = [
    (datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 6, 30, tzinfo=UTC)),
    (datetime(2023, 7, 1, tzinfo=UTC), datetime(2023, 12, 31, tzinfo=UTC)),
]

results = batch_strategy_test(strategies, data_files, time_ranges, Path("./test_results"))
```

### Real-time Progress Monitoring

Monitor script execution progress with detailed feedback.

```python
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.syminfo import SymInfo
from pathlib import Path
from datetime import datetime
import time


def run_with_detailed_progress(script_path: Path, data_path: Path):
    """
    Run script with detailed progress information
    """

    syminfo = SymInfo.load_toml(data_path.with_suffix(".toml"))

    # Progress tracking variables
    start_time = time.time()
    bar_count = 0
    last_update = 0

    def progress_callback(current_dt: datetime):
        nonlocal bar_count, last_update
        bar_count += 1

        # Update every 100 bars or every 5 seconds
        current_time = time.time()
        if bar_count % 100 == 0 or (current_time - last_update) > 5:
            elapsed = current_time - start_time
            rate = bar_count / elapsed if elapsed > 0 else 0

            print(f"Processed {bar_count:,} bars | "
                  f"Rate: {rate:.1f} bars/sec | "
                  f"Current: {current_dt.strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"Elapsed: {elapsed:.1f}s")

            last_update = current_time

    print(f"Starting execution: {script_path.name} on {data_path.name}")

    with OHLCVReader(data_path) as reader:
        print(f"Data range: {reader.start_datetime} to {reader.end_datetime}")
        print(f"Total bars: {reader.size:,}")

        runner = ScriptRunner(
            script_path,
            reader,
            syminfo,
            last_bar_index=reader.size - 1
        )

        execution_start = time.time()
        runner.run(on_progress=progress_callback)
        execution_time = time.time() - execution_start

        print(f"\n✓ Execution completed in {execution_time:.2f}s")
        print(f"✓ Average rate: {bar_count / execution_time:.1f} bars/sec")


# Usage
run_with_detailed_progress(
    Path("./strategies/my_strategy.py"),
    Path("./data/BTCUSD_1H.ohlcv")
)
```

## Data Processing Examples

### Multi-Timeframe Analysis

Process the same data on multiple timeframes using resampling.

```python
from pynecore.core.ohlcv_file import OHLCVReader, OHLCVWriter
from pynecore.core.resampler import resample_ohlcv
from pathlib import Path


def create_multiple_timeframes(source_data: Path, timeframes: list[str]):
    """
    Create multiple timeframe versions of the same data
    """

    base_name = source_data.stem
    results = {}

    with OHLCVReader(source_data) as reader:
        source_ohlcv = list(reader)  # Load all data

    for timeframe in timeframes:
        if timeframe == "1":  # Skip if same as source
            results[timeframe] = source_data
            continue

        print(f"Creating {timeframe} timeframe from {base_name}")

        # Resample to new timeframe
        resampled_data = resample_ohlcv(source_ohlcv, timeframe)

        # Write to new file
        output_path = source_data.parent / f"{base_name}_{timeframe}.ohlcv"
        with OHLCVWriter(output_path, truncate=True) as writer:
            for candle in resampled_data:
                writer.write(candle)

        # Update syminfo for new timeframe
        syminfo_source = source_data.with_suffix(".toml")
        syminfo_output = output_path.with_suffix(".toml")

        if syminfo_source.exists():
            syminfo = SymInfo.load_toml(syminfo_source)
            syminfo.period = timeframe
            syminfo.save_toml(syminfo_output)

        results[timeframe] = output_path
        print(f"✓ Created {output_path} with {len(resampled_data)} bars")

    return results


# Usage - create 5min, 15min, 1H, 4H, and 1D from 1min data
timeframes = ["5", "15", "60", "240", "1D"]
tf_files = create_multiple_timeframes(
    Path("./data/BTCUSD_1.ohlcv"),
    timeframes
)

# Run strategy on all timeframes
for tf, file_path in tf_files.items():
    print(f"\nRunning strategy on {tf} timeframe...")
    # Run your strategy here
```

### Data Quality Checks

Validate and clean OHLCV data programmatically.

```python
from pynecore.core.ohlcv_file import OHLCVReader, OHLCVWriter
from pynecore.types.ohlcv import OHLCV
from pathlib import Path
from datetime import datetime


def validate_and_clean_data(input_path: Path, output_path: Path = None):
    """
    Validate OHLCV data and fix common issues
    """

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_cleaned.ohlcv")

    issues = {
        'gaps': 0,
        'invalid_ohlc': 0,
        'zero_volume': 0,
        'negative_prices': 0,
        'total_bars': 0
    }

    cleaned_data = []

    with OHLCVReader(input_path) as reader:
        print(f"Validating {reader.size:,} bars from {input_path.name}")

        prev_timestamp = None
        expected_interval = None

        for candle in reader:
            issues['total_bars'] += 1

            # Check for gaps
            if prev_timestamp and expected_interval:
                expected_ts = prev_timestamp + expected_interval
                if candle.timestamp > expected_ts + expected_interval:
                    issues['gaps'] += 1
                    print(f"Gap detected at {datetime.fromtimestamp(candle.timestamp)}")

            # Set interval from first two bars
            if prev_timestamp and expected_interval is None:
                expected_interval = candle.timestamp - prev_timestamp

            # Validate OHLC relationships
            if not (candle.low <= candle.open <= candle.high and
                    candle.low <= candle.close <= candle.high):
                issues['invalid_ohlc'] += 1
                print(f"Invalid OHLC at {datetime.fromtimestamp(candle.timestamp)}: "
                      f"O:{candle.open} H:{candle.high} L:{candle.low} C:{candle.close}")

                # Fix by adjusting high/low
                fixed_candle = OHLCV(
                    timestamp=candle.timestamp,
                    open=candle.open,
                    high=max(candle.high, candle.open, candle.close),
                    low=min(candle.low, candle.open, candle.close),
                    close=candle.close,
                    volume=candle.volume
                )
                cleaned_data.append(fixed_candle)
            else:
                # Check for negative prices
                if any(price < 0 for price in [candle.open, candle.high, candle.low, candle.close]):
                    issues['negative_prices'] += 1
                    continue  # Skip negative price bars

                # Check for zero volume (might be gap-filled)
                if candle.volume == 0:
                    issues['zero_volume'] += 1

                cleaned_data.append(candle)

            prev_timestamp = candle.timestamp

    # Write cleaned data
    with OHLCVWriter(output_path, truncate=True) as writer:
        for candle in cleaned_data:
            writer.write(candle)

    # Print validation report
    print(f"\n📊 Data Validation Report:")
    print(f"   Total bars processed: {issues['total_bars']:,}")
    print(f"   Gaps detected: {issues['gaps']}")
    print(f"   Invalid OHLC fixed: {issues['invalid_ohlc']}")
    print(f"   Zero volume bars: {issues['zero_volume']}")
    print(f"   Negative price bars removed: {issues['negative_prices']}")
    print(f"   Clean bars written: {len(cleaned_data):,}")
    print(f"   Output: {output_path}")

    return issues, output_path


# Usage
issues, clean_file = validate_and_clean_data(Path("./data/BTCUSD_1D.ohlcv"))
```

## Integration Examples

### Automated Trading System Setup

Complete setup for an automated trading system using PyneCore.

```python
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.syminfo import SymInfo
from pynecore.providers.ccxt import CCXTProvider
from pathlib import Path
from datetime import datetime, UTC, timedelta
import schedule
import time


class TradingSystem:
    def __init__(self, config_dir: Path, strategies_dir: Path, data_dir: Path):
        self.config_dir = config_dir
        self.strategies_dir = strategies_dir
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)

    def update_data(self, symbol: str, timeframe: str = "1D"):
        """
        Update market data for a symbol
        """
        provider = CCXTProvider(symbol=symbol, config_dir=self.config_dir)

        safe_symbol = symbol.replace(":", "_").replace("/", "_")
        data_file = self.data_dir / f"{safe_symbol}_{timeframe}.ohlcv"

        # Download last 7 days to ensure we have latest data
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=7)

        print(f"Updating {symbol} data...")

        with OHLCVWriter(data_file, truncate=False) as writer:
            if data_file.exists():
                # Append new data
                writer.seek_to_timestamp(int(start_date.timestamp()))

            for candle in provider.download_ohlcv(
                    timeframe=timeframe,
                    since=start_date,
                    until=end_date
            ):
                writer.write(candle)

        # Update syminfo if it doesn't exist
        syminfo_file = data_file.with_suffix(".toml")
        if not syminfo_file.exists():
            syminfo = provider.get_syminfo(timeframe=timeframe)
            syminfo.save_toml(syminfo_file)

        return data_file

    def run_strategy(self, strategy_name: str, symbol: str, timeframe: str = "1D"):
        """
        Run a strategy on current data
        """

        # Update data first
        data_file = self.update_data(symbol, timeframe)

        # Load strategy
        strategy_path = self.strategies_dir / f"{strategy_name}.py"
        if not strategy_path.exists():
            raise FileNotFoundError(f"Strategy not found: {strategy_path}")

        # Load symbol info
        syminfo = SymInfo.load_toml(data_file.with_suffix(".toml"))

        # Setup output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("./trading_results") / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run strategy on recent data (last 100 bars for signals)
        with OHLCVReader(data_file) as reader:
            recent_data = list(reader)[-100:]  # Last 100 bars

            runner = ScriptRunner(
                strategy_path,
                recent_data,
                syminfo,
                trade_path=output_dir / "trades.csv",
                strat_path=output_dir / "stats.csv",
                plot_path=output_dir / "signals.csv"
            )

            print(f"Running {strategy_name} on {symbol}...")
            runner.run()

            print(f"✓ Results saved to {output_dir}")

            # Return latest signals/trades for decision making
            return self._parse_latest_signals(output_dir / "signals.csv")

    def _parse_latest_signals(self, signals_file: Path):
        """
        Parse latest trading signals from output
        """
        # Implementation depends on your strategy outputs
        # This is a placeholder for signal processing
        if signals_file.exists():
            print(f"Latest signals available in: {signals_file}")
        return {}


# Usage
trading_system = TradingSystem(
    config_dir=Path("./config"),
    strategies_dir=Path("./strategies"),
    data_dir=Path("./data")
)

# Schedule automated runs
schedule.every().day.at("09:00").do(
    trading_system.run_strategy,
    "bollinger_bands",
    "BYBIT:BTC/USDT:USDT"
)

schedule.every().day.at("15:00").do(
    trading_system.run_strategy,
    "mean_reversion",
    "BYBIT:ETH/USDT:USDT"
)

# Run scheduler
print("Trading system started. Press Ctrl+C to stop.")
while True:
    schedule.run_pending()
    time.sleep(60)
```

These practical examples demonstrate how to achieve programmatically everything that's available through the CLI, plus
additional automation and integration capabilities that are only possible through the Python API.