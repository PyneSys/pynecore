<!--
---
weight: 10001
title: "Programmatic Quick Start"
description: "Quick start guide for PyneCore's programmatic Python API"
icon: "tips_and_updates"
date: "2025-08-19"
lastmod: "2025-08-19"
draft: false
toc: true
categories: ["Advanced", "API"]
tags: ["quick-start", "api", "ohlcv", "syminfo", "scriptrunner"]
---
-->

# Programmatic Quick Start

This guide shows you how to use PyneCore's Python API to execute scripts, download data, and manage OHLCV files
programmatically. Every CLI command has a programmatic equivalent, plus additional automation capabilities that are only
possible through the Python API.

## Quick Start

```python
from pathlib import Path
from pynecore.core.syminfo import SymInfo
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.script_runner import ScriptRunner

# Load symbol information from TOML file
syminfo = SymInfo.load_toml(Path("data.toml"))

# Load OHLCV data (no .open() method - use constructor directly)
with OHLCVReader(Path("data.ohlcv")) as reader:
    # Create script runner with correct parameter names
    runner = ScriptRunner(
        script_path=Path("strategy.py"),
        ohlcv_iter=reader,  # Pass reader directly as iterator
        syminfo=syminfo
    )

    # Execute the script
    runner.run()
```

## Automatic Data Conversion

PyneCore can automatically convert CSV, JSON, and TXT files to OHLCV format with smart analysis, just like the CLI:

```python
from pynecore.core.data_converter import DataConverter
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.syminfo import SymInfo
from pynecore.core.script_runner import ScriptRunner
from pathlib import Path


# Automatic conversion (equivalent to: pyne run script.py data.csv)
def run_with_auto_conversion(script_path: Path, data_path: Path):
    """
    Run script with automatic data conversion if needed
    """

    # Convert non-OHLCV files automatically
    if data_path.suffix.lower() in ['.csv', '.json', '.txt']:
        converter = DataConverter()

        # Check if conversion is needed
        if converter.is_conversion_required(data_path):
            # Auto-detect symbol and provider
            symbol, provider = DataConverter.guess_symbol_from_filename(data_path)

            # Convert with automatic TOML generation
            converter.convert_to_ohlcv(
                file_path=data_path,
                provider=provider or "auto",
                symbol=symbol or data_path.stem.upper(),
                force=True
            )
            print(f"✓ Auto-converted {data_path.suffix} to OHLCV format")

        # Update path to OHLCV file
        data_path = data_path.with_suffix('.ohlcv')

    # Load auto-generated or existing symbol info
    syminfo = SymInfo.load_toml(data_path.with_suffix('.toml'))

    # Run script normally
    with OHLCVReader(data_path) as reader:
        runner = ScriptRunner(script_path, reader, syminfo)
        runner.run()


# Usage with any format
run_with_auto_conversion(
    Path("my_strategy.py"),
    Path("BTCUSDT_1h.csv")  # Automatically converted to OHLCV
)
```

**Supported Auto-Detection Patterns:**

- `BTCUSDT.csv` → Symbol: BTC/USDT
- `BYBIT_ETHUSDT_1h.csv` → Symbol: ETH/USDT, Provider: bybit
- `ccxt_BINANCE_ADAUSDT.json` → Symbol: ADA/USDT, Provider: binance

## Strategy Outputs

Save script results to files for analysis and visualization:

```python
# Configure output paths (all optional)
runner = ScriptRunner(
    script_path=Path("strategy.py"),
    ohlcv_iter=reader,
    syminfo=syminfo,
    plot_path=Path("plots.csv"),  # Chart plot data
    trade_path=Path("trades.csv"),  # Trading strategy results  
    strat_path=Path("strategy.csv")  # Strategy statistics
)
```

**Output File Formats:**

- **Plot files**: CSV with timestamp, plot series data
- **Trade files**: CSV with trade details, P&L, timestamps
- **Strategy files**: CSV with performance metrics and statistics

## Core Components

### SymInfo Class

Symbol information management with two creation methods:

```python
# Method 1: Load from TOML file (recommended)
syminfo = SymInfo.load_toml(Path("symbol_data.toml"))

# Method 2: Create directly (requires all parameters)
syminfo = SymInfo(
    prefix="EXCHANGE",
    ticker="BTCUSD",
    currency="USD",
    period="1D",
    type="crypto",
    # ... many more required fields
)
```

**Key Attributes:**

- `ticker`: Symbol ticker (e.g., "BTC/USDT")
- `period`: Timeframe period (e.g., "1D", "1H")
- `currency`: Quote currency for calculations
- `timezone`: For timestamp conversion (default: 'UTC')
- `opening_hours`, `session_starts`, `session_ends`: Trading session data

### OHLCVReader Class

Fast memory-mapped OHLCV data reader:

```python
# Context manager usage (recommended)
with OHLCVReader(Path("data.ohlcv")) as reader:
    # Get data info
    size = reader.size
    start_time = reader.start_datetime
    end_time = reader.end_datetime

    # Read specific time ranges
    data_iter = reader.read_from(
        start_timestamp=int(start_time.timestamp()),
        end_timestamp=int(end_time.timestamp()),
        skip_gaps=True  # Skip gaps in data (default)
    )

```

**Key Methods:**

- `get_size(start_timestamp=None, end_timestamp=None) -> int`: Count records in range
- `read_from(start_timestamp, end_timestamp, skip_gaps=True)`: Get data iterator for time range
- Properties: `start_datetime`, `end_datetime`, `size`

### OHLCVWriter Class

Create and populate OHLCV files with advanced analysis capabilities:

```python
from pynecore.core.ohlcv_file import OHLCVWriter

# Create new OHLCV file with truncate option
with OHLCVWriter(Path("output.ohlcv"), truncate=True) as writer:
    # Import from various formats
    writer.load_from_csv(Path("data.csv"), tz="UTC")
    # writer.load_from_json(Path("data.json"), tz="UTC")
    # writer.load_from_txt(Path("data.txt"), tz="UTC")  # New: tab/semicolon/pipe delimited

    # Access automatic analysis results
    print(f"Detected tick size: {writer.analyzed_tick_size}")
    print(f"Price scale: {writer.analyzed_price_scale}")
    print(f"Minimum move: {writer.analyzed_min_move}")
```

**New Features:**

- **Truncate parameter**: `truncate=True` clears existing file content
- **TXT file support**: Handle tab, semicolon, or pipe-delimited files
- **Automatic analysis**: Detects tick size, price scale, and trading hours
- **Smart gap filling**: Automatically fills missing data points

### ScriptRunner Class

Execute PyneCore scripts with OHLCV data:

```python
runner = ScriptRunner(
    script_path: Path,  # Path to .py script file
ohlcv_iter: Iterable[OHLCV],  # Data iterator (e.g., OHLCVReader)
syminfo: SymInfo,  # Symbol information

# Optional output paths (keyword-only)
plot_path: Path | None = None,  # Save plot data
trade_path: Path | None = None,  # Save trade results (NOT equity_path!)
strat_path: Path | None = None,  # Save strategy statistics

    # Advanced options
update_syminfo_every_run: bool = False,  # For parallel execution
last_bar_index: int = 0  # Index of final bar
)

# Execute script
runner.run()

```

## Time Range Processing

Process specific time periods efficiently:

```python
from datetime import datetime, UTC

# Define time range
start_time = datetime(2023, 1, 1, tzinfo=UTC)
end_time = datetime(2023, 12, 31, tzinfo=UTC)

with OHLCVReader(Path("data.ohlcv")) as reader:
    # Get subset of data
    data_subset = reader.read_from(
        start_timestamp=int(start_time.timestamp()),
        end_timestamp=int(end_time.timestamp())
    )

    # Check data size before processing
    subset_size = reader.get_size(
        start_timestamp=int(start_time.timestamp()),
        end_timestamp=int(end_time.timestamp())
    )
    print(f"Processing {subset_size} candles")

    # Run script on subset
    runner = ScriptRunner(Path("script.py"), data_subset, syminfo)
    runner.run()
```

## Common Pitfalls & Tips

### File Path Management

```python
# ✅ Always use Path objects for consistency
script_path = Path("strategies/my_strategy.py")
data_path = Path("data/BTCUSD_1D.ohlcv")

# ❌ Avoid mixing strings and Path objects
runner = ScriptRunner("script.py", reader, syminfo)  # Works but inconsistent
```

### Memory Considerations

```python
# ✅ Use context managers for automatic cleanup
with OHLCVReader(Path("large_data.ohlcv")) as reader:
    runner = ScriptRunner(script_path, reader, syminfo)
    runner.run()
# File automatically closed after use

# ✅ Process data in chunks for very large datasets
with OHLCVReader(Path("large_data.ohlcv")) as reader:
    start_ts = reader.start_timestamp
    chunk_size = 86400 * 30  # 30 days in seconds

    while start_ts < reader.end_timestamp:
        end_ts = min(start_ts + chunk_size, reader.end_timestamp)
        chunk_data = reader.read_from(start_ts, end_ts)
        runner = ScriptRunner(script_path, chunk_data, syminfo)
        runner.run()
        start_ts = end_ts
```

### Data Conversion Best Practices

```python
from pynecore.core.data_converter import DataConverter

# ✅ Use DataConverter for automatic conversion with analysis
converter = DataConverter()

# Check if conversion is needed (smart caching)
if converter.is_conversion_required(Path("data.csv")):
    converter.convert_to_ohlcv(
        file_path=Path("data.csv"),
        force=True  # Force fresh conversion
    )

# ✅ Manual conversion with custom analysis
with OHLCVWriter(Path("manual.ohlcv"), truncate=True) as writer:
    writer.load_from_csv(Path("custom.csv"), tz="America/New_York")

    # Use analysis results for accurate symbol configuration
    if writer.analyzed_tick_size:
        print(f"Optimal tick size: {writer.analyzed_tick_size}")
        # Create SymInfo with detected parameters
```

### Error Handling

```python
from pynecore.core.data_converter import DataFormatError, ConversionError

try:
    # Automatic data conversion with error handling
    converter = DataConverter()

    if data_path.suffix != '.ohlcv':
        converter.convert_to_ohlcv(
            file_path=data_path,
            force=True
        )
        data_path = data_path.with_suffix('.ohlcv')

    # Script execution
    with OHLCVReader(data_path) as reader:
        runner = ScriptRunner(script_path, reader, syminfo)
        runner.run()

except FileNotFoundError:
    print("Data or script file not found")
except ImportError as e:
    print(f"Script error - check @pyne decorator: {e}")
except ValueError as e:
    print(f"Data format error: {e}")
except DataFormatError as e:
    print(f"Unsupported data format: {e}")
except ConversionError as e:
    print(f"Data conversion failed: {e}")
```

### Performance Optimization

- Use `skip_gaps=True` in `read_from()` to avoid processing invalid data
- Set appropriate `last_bar_index` for better progress tracking
- Consider `update_syminfo_every_run=False` for single-threaded usage
- Process data in time chunks for memory efficiency with large datasets
- Use `DataConverter.is_conversion_required()` to avoid unnecessary conversions
- Set `truncate=True` in OHLCVWriter for clean file creation, `truncate=False` for data appending
- Leverage automatic tick size analysis for accurate symbol configuration