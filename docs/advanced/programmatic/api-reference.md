<!--
---
weight: 10002
title: "API Reference"
description: "Complete API reference for PyneCore's programmatic Python interface"
icon: "api"
date: "2025-08-19"
lastmod: "2025-08-27"
draft: false
toc: true
categories: ["Advanced", "API"]
tags: ["api-reference", "classes", "methods", "ohlcv", "syminfo", "scriptrunner"]
---
-->

# Programmatic API Reference

This page provides a complete reference for PyneCore's programmatic Python API. Use this reference when building
applications, automation systems, or integrations that need to execute PyneCore scripts programmatically.

## Core Classes Overview

| Class           | Purpose                       | CLI Equivalent                |
|-----------------|-------------------------------|-------------------------------|
| `ScriptRunner`  | Execute PyneCore scripts      | `pyne run`                    |
| `OHLCVReader`   | Read binary OHLCV data        | Used internally by `pyne run` |
| `OHLCVWriter`   | Write binary OHLCV data       | `pyne data download`          |
| `DataConverter` | Convert CSV/JSON/TXT to OHLCV | `pyne data convert-from`      |
| `SymInfo`       | Symbol metadata management    | TOML files in workdir         |
| `PyneComp`      | Pine Script compilation       | `pyne compile`                |
| Data Providers  | Download market data          | `pyne data download`          |

---

## ScriptRunner

Execute PyneCore scripts with OHLCV data and collect results.

### Constructor

```python
ScriptRunner(
    script_path: Path,  # Path to .py script file
ohlcv_iter: Iterable[OHLCV],  # Data iterator (e.g., OHLCVReader)
syminfo: SymInfo,  # Symbol information
*,
# Output file paths (keyword-only)
plot_path: Path | None = None,  # Save plot data (CSV)
strat_path: Path | None = None,  # Save strategy statistics (CSV)  
trade_path: Path | None = None,  # Save trade results (CSV)

    # Advanced options
update_syminfo_every_run: bool = False,  # For parallel execution
last_bar_index: int = 0  # Index of final bar
)
```

**Parameters:**

- `script_path`: Must contain `@pyne` magic comment and decorated `main()` function
- `ohlcv_iter`: Any iterable yielding `OHLCV` objects (typically `OHLCVReader`)
- `syminfo`: Symbol configuration loaded from TOML or created manually
- `plot_path`: Save chart indicators and plots (optional)
- `strat_path`: Save strategy performance metrics (optional, strategies only)
- `trade_path`: Save individual trade details (optional, strategies only)

**Raises:**

- `ImportError`: Script missing `@pyne` comment or decorated `main()` function
- `FileNotFoundError`: Script file not found
- `OSError`: Cannot create output files

### Methods

#### run()

```python
def run(self, on_progress: Callable[[datetime], None] | None = None) -> None
```

Execute the script on all data. Equivalent to CLI `pyne run script.py data.ohlcv`.

**Parameters:**

- `on_progress`: Optional callback receiving current datetime on each bar

**Example:**

```python
def progress_callback(current_dt: datetime):
    print(f"Processing: {current_dt}")


runner.run(on_progress=progress_callback)
```

#### run_iter()

```python
def run_iter(self, on_progress: Callable[[datetime], None] | None = None)
        -> Iterator[tuple[OHLCV, dict[str, Any]] | tuple[OHLCV, dict[str, Any], list[Trade]]]
```

Execute script and yield results for each bar. For strategies, also yields closed trades.

**Returns:** Iterator yielding `(ohlcv_data, plot_results)` or `(ohlcv_data, plot_results, closed_trades)`

---

## OHLCVReader

Fast memory-mapped reader for binary OHLCV data files.

### Constructor

```python
OHLCVReader(path: str | Path)
```

**Parameters:**

- `path`: Path to `.ohlcv` binary file

### Context Manager Usage

```python
with OHLCVReader(Path("data.ohlcv")) as reader:
    # Use reader methods
    for candle in reader:
        print(candle)
```

### Properties

```python
@property
def size(self) -> int  # Number of records


    def start_datetime(self) -> datetime  # First record timestamp


    def end_datetime(self) -> datetime  # Last record timestamp  


    def start_timestamp(self) -> int  # First record as Unix timestamp


    def end_timestamp(self) -> int  # Last record as Unix timestamp


    def interval(self) -> int  # Seconds between records
```

### Methods

#### read()

```python
def read(self, position: int) -> OHLCV
```

Read single candle at specific position (0-indexed).

#### read_from()

```python
def read_from(
        self,
        start_timestamp: int,
        end_timestamp: int | None = None,
        skip_gaps: bool = True
) -> Iterator[OHLCV]
```

Read candles in timestamp range. Most commonly used method.

**Parameters:**

- `start_timestamp`: Unix timestamp start
- `end_timestamp`: Unix timestamp end (None = read to end)
- `skip_gaps`: Skip gap-filled records (volume = -1)

**Example:**

```python
from datetime import datetime, UTC

start = datetime(2024, 1, 1, tzinfo=UTC)
end = datetime(2024, 6, 30, tzinfo=UTC)

with OHLCVReader(data_path) as reader:
    data_iter = reader.read_from(
        int(start.timestamp()),
        int(end.timestamp())
    )
    runner = ScriptRunner(script_path, data_iter, syminfo)
    runner.run()
```

#### get_size()

```python
def get_size(self, start_timestamp: int | None = None, end_timestamp: int | None = None) -> int
```

Count records in timestamp range.

#### get_positions()

```python
def get_positions(self, start_timestamp: int | None = None, end_timestamp: int | None = None) -> tuple[int, int]
```

Get start and end positions for timestamp range.

#### Export Methods

```python
def save_to_csv(self, path: str, as_datetime: bool = False) -> None


    def save_to_json(self, path: str, as_datetime: bool = False) -> None
```

Export data to CSV or JSON format.

---

## OHLCVWriter

Write OHLCV data to binary format with automatic gap filling.

### Constructor

```python
OHLCVWriter(path: str | Path, truncate: bool = False)
```

**Parameters:**

- `path`: Path to output `.ohlcv` file
- `truncate`: If True, truncate existing file on open (default: False)

### Context Manager Usage

```python
with OHLCVWriter(Path("output.ohlcv"), truncate=True) as writer:
    writer.write(OHLCV(timestamp, open, high, low, close, volume))
```

### Properties

```python
@property
def size(self) -> int  # Number of records written


    def start_datetime(self) -> datetime  # First record timestamp


    def end_datetime(self) -> datetime  # Last record timestamp


    def is_open(self) -> bool  # File open status


# Advanced analysis properties (available after writing data)
def analyzed_tick_size(self) -> float | None  # Auto-detected minimum price movement


    def analyzed_price_scale(self) -> int | None  # Auto-detected price scale


    def analyzed_min_move(self) -> int | None  # Auto-detected minimum move in ticks
```

### Methods

#### write()

```python
def write(self, candle: OHLCV) -> None
```

Write single OHLCV candle. Automatically fills gaps with previous close price and -1 volume.

#### seek_to_timestamp() / seek()

```python
def seek_to_timestamp(self, timestamp: int) -> None


    def seek(self, position: int) -> None  
```

Move write position for data insertion/overwriting.

**Important:** When using `seek()` to overwrite data, timestamps must remain in chronological order. Writing a timestamp
earlier than subsequent records will raise a `ValueError`.

#### truncate()

```python
def truncate(self) -> None
```

Remove all data after current position.

#### Data Import Methods

```python
def load_from_csv(
        self,
        path: str | Path,
        timestamp_format: str | None = None,
        timestamp_column: str | None = None,
        date_column: str | None = None,
        time_column: str | None = None,
        tz: str | None = None
) -> None


def load_from_json(
        self,
        path: str | Path,
        timestamp_format: str | None = None,
        timestamp_field: str | None = None,
        date_field: str | None = None,
        time_field: str | None = None,
        tz: str | None = None,
        mapping: dict[str, str] | None = None
) -> None


def load_from_txt(
        self,
        path: str | Path,
        timestamp_format: str | None = None,
        timestamp_column: str | None = None,
        date_column: str | None = None,
        time_column: str | None = None,
        tz: str | None = None
) -> None
```

Import data from CSV, JSON, or TXT (tab/semicolon/pipe-delimited) files with flexible column/field mapping.

---

## DataConverter

Enhanced automatic data file conversion with smart analysis and TOML generation.

### Constructor

```python
from pynecore.core.data_converter import DataConverter

converter = DataConverter()
```

### Static Methods

#### guess_symbol_from_filename()

```python
@staticmethod
def guess_symbol_from_filename(file_path: Path) -> tuple[str | None, str | None]
```

Auto-detect symbol and provider from filename patterns.

**Returns:** `(symbol, provider)` tuple

**Examples:**

- `BTCUSDT.csv` → `("BTC/USDT", None)`
- `BYBIT_ETHUSDT_1h.csv` → `("ETH/USDT", "bybit")`
- `ccxt_BINANCE_ADAUSDT.json` → `("ADA/USDT", "binance")`

#### is_conversion_required()

```python
@staticmethod
def is_conversion_required(source_path: Path, ohlcv_path: Path | None = None) -> bool
```

Check if conversion is needed based on file freshness.

### Methods

#### convert_to_ohlcv()

```python
def convert_to_ohlcv(
        self,
        file_path: Path,
        *,
        force: bool = False,
        provider: str | None = None,
        symbol: str | None = None,
        timezone: str = "UTC"
) -> None
```

Convert CSV/JSON/TXT to OHLCV format with automatic TOML configuration generation.

**Parameters:**

- `file_path`: Source data file path
- `force`: Force conversion even if OHLCV file exists and is newer
- `provider`: Data provider name (auto-detected if None)
- `symbol`: Symbol name (auto-detected if None)
- `timezone`: Timezone for timestamp conversion

**Features:**

- **Automatic symbol detection** from filename patterns
- **Tick size analysis** during conversion
- **Trading hours detection** from data patterns
- **TOML configuration generation** with detected parameters
- **Support for multiple formats**: CSV, JSON, TXT (tab/semicolon/pipe delimited)

**Example:**

```python
converter = DataConverter()

# Convert with automatic detection
converter.convert_to_ohlcv(
    file_path=Path("BTCUSDT_1h.csv"),
    force=True
)

# Generates:
# - BTCUSDT_1h.ohlcv (binary data)
# - BTCUSDT_1h.toml (symbol configuration with detected parameters)
```

---

## SymInfo

Symbol metadata container loaded from TOML files or created programmatically.

### Class Methods

#### load_toml()

```python
@classmethod
def load_toml(cls, path: Path) -> SymInfo
```

Load symbol information from TOML file (recommended approach).

```python
syminfo = SymInfo.load_toml(Path("BTCUSD_1D.toml"))
```

### Constructor (Direct Creation)

```python
SymInfo(
    prefix: str,  # Exchange prefix (e.g., "BYBIT") 
description: str,  # Human readable description
ticker: str,  # Symbol ticker (e.g., "BTC/USDT")
currency: str,  # Quote currency (e.g., "USDT")
basecurrency: str | None,  # Base currency (e.g., "BTC")
period: str,  # Timeframe (e.g., "1D", "1H")
type: Literal[...],  # Asset type ("crypto", "stock", etc.)
mintick: float,  # Minimum price movement
pricescale: int,  # Price scale factor  
minmove: int,  # Minimum move in ticks
pointvalue: float,  # Value per point
opening_hours: list[SymInfoInterval],
session_starts: list[SymInfoSession],
session_ends: list[SymInfoSession],
timezone: str = 'UTC',
# ... many other optional fields
)
```

**Note:** Direct construction requires many parameters. Use `load_toml()` instead.

### Methods

#### save_toml()

```python
def save_toml(self, path: Path) -> None
```

Save symbol information to TOML file.

### Key Attributes

- `symbol`: Combined ticker symbol
- `timeframe`: Trading timeframe
- `timezone`: Timezone for timestamps
- `currency`: Quote currency
- `type`: Asset type classification

---

## PyneComp (Pine Script Compilation)

Compile Pine Script files to Python for execution.

### Constructor

```python
from pynecore.pynesys.compiler import PyneComp

compiler = PyneComp(
    api_key: str,  # PyneSys API key
api_url: str = "https://api.pynesys.io",
timeout: int = 30
)
```

### Methods

#### compile()

```python  
def compile(self, input_path: Path, output_path: Path) -> None
```

Compile .pine file to .py file.

#### needs_compilation()

```python
def needs_compilation(self, pine_path: Path, py_path: Path) -> bool
```

Check if compilation is needed (Pine file newer than Python file).

**Example:**

```python
from pynecore.pynesys.compiler import PyneComp

compiler = PyneComp(api_key="your-api-key")
pine_file = Path("strategy.pine")
py_file = Path("strategy.py")

if compiler.needs_compilation(pine_file, py_file):
    compiler.compile(pine_file, py_file)

# Now use the compiled Python file
with OHLCVReader(data_path) as reader:
    runner = ScriptRunner(py_file, reader, syminfo)
    runner.run()
```

---

## Data Providers

Download historical market data programmatically. Equivalent to `pyne data download`.

### Available Providers

```python
from pynecore.providers.ccxt import CCXTProvider
from pynecore.providers.capitalcom import CapitalComProvider
```

### CCXT Provider Example

```python
from pynecore.providers.ccxt import CCXTProvider
from pynecore.core.ohlcv_file import OHLCVWriter
from datetime import datetime, UTC
from pathlib import Path

# Initialize provider
provider = CCXTProvider(
    symbol="BYBIT:BTC/USDT:USDT",
    config_dir=Path("./config")  # Directory containing ccxt.toml
)

# Set date range
start_date = datetime(2024, 1, 1, tzinfo=UTC)
end_date = datetime(2024, 12, 31, tzinfo=UTC)

# Download and save data
with OHLCVWriter(Path("BTCUSD_1D.ohlcv"), truncate=True) as writer:
    for candle in provider.download_ohlcv(
            timeframe="1D",
            since=start_date,
            until=end_date
    ):
        writer.write(candle)

# Save symbol information
syminfo = provider.get_syminfo(timeframe="1D")
syminfo.save_toml(Path("BTCUSD_1D.toml"))
```

### Provider Methods

```python
def get_list_of_symbols(self) -> list[str]  # List available symbols


    def get_syminfo(self, timeframe: str) -> SymInfo  # Get symbol metadata  


    def download_ohlcv(self, timeframe: str, since: datetime, until: datetime)  # Download data iterator
```

---

## OHLCV Data Type

The core data structure for candlestick data.

```python
from pynecore.types.ohlcv import OHLCV

# Create OHLCV record
candle = OHLCV(
    timestamp=1609459200,  # Unix timestamp
    open=29000.0,
    high=29500.0,
    low=28800.0,
    close=29200.0,
    volume=1250.5,
    extra_fields={}  # Optional additional data
)

# Access fields
print(f"Close: {candle.close}")
print(f"Volume: {candle.volume}")
```

---

## Error Handling

Common exceptions and their meanings:

### ImportError

- Script missing `@pyne` magic comment
- `main()` function not decorated with `@script.indicator` or `@script.strategy`

### FileNotFoundError

- Script, data, or symbol info files not found
- Incorrect file paths

### ValueError

- Invalid OHLCV data format
- Malformed timestamps or data values
- CSV/JSON parsing errors

### APIError (from pynecore.pynesys)

- Invalid PyneSys API key
- Network/compilation errors

**Example Error Handling:**

```python
try:
    with OHLCVReader(data_path) as reader:
        runner = ScriptRunner(script_path, reader, syminfo)
        runner.run()
except FileNotFoundError as e:
    print(f"File not found: {e}")
except ImportError as e:
    print(f"Script error: {e}")
except ValueError as e:
    print(f"Data error: {e}")
```

---

## New Features Examples

### Automatic Data Conversion (CLI-equivalent)

```python
from pynecore.core.data_converter import DataConverter
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.syminfo import SymInfo
from pynecore.core.ohlcv_file import OHLCVReader
from pathlib import Path


def run_script_with_any_format(script_path: Path, data_path: Path):
    """
    Programmatic equivalent of: pyne run script.py data.csv
    Handles automatic conversion like the CLI does.
    """
    converter = DataConverter()

    # Convert non-OHLCV files automatically
    if data_path.suffix != ".ohlcv":
        if converter.is_conversion_required(data_path):
            print(f"Auto-converting {data_path.suffix} to OHLCV...")

            converter.convert_to_ohlcv(
                file_path=data_path,
                force=True  # Ensure fresh conversion
            )

        # Point to converted file
        data_path = data_path.with_suffix(".ohlcv")

    # Load auto-generated or existing symbol info
    syminfo = SymInfo.load_toml(data_path.with_suffix(".toml"))

    # Run script normally
    with OHLCVReader(data_path) as reader:
        runner = ScriptRunner(script_path, reader, syminfo)
        runner.run()


# Usage
run_script_with_any_format(
    script_path=Path("my_strategy.py"),
    data_path=Path("BTCUSDT_1h.csv")  # Automatic conversion
)
```

### Advanced Data Analysis During Conversion

```python
from pynecore.core.ohlcv_file import OHLCVWriter
from pathlib import Path

# Convert with automatic tick size and trading hours analysis
with OHLCVWriter(Path("analyzed_data.ohlcv"), truncate=True) as writer:
    writer.load_from_csv(Path("market_data.csv"))

    # Access analysis results
    print(f"Detected tick size: {writer.analyzed_tick_size}")
    print(f"Price scale: {writer.analyzed_price_scale}")
    print(f"Min move: {writer.analyzed_min_move}")

# Analysis results can be used to generate accurate TOML configurations
```