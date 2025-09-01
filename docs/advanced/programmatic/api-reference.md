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

Complete API reference for PyneCore's programmatic Python interface.

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

Executes PyneCore scripts with OHLCV data.

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
update_syminfo_every_run: bool = False,  # Update syminfo on each run (for multiple scripts)
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

Executes the script on all data.

**Parameters:**

- `on_progress`: Optional callback receiving current datetime on each bar


#### run_iter()

```python
def run_iter(self, on_progress: Callable[[datetime], None] | None = None)
        -> Iterator[tuple[OHLCV, dict[str, Any]] | tuple[OHLCV, dict[str, Any], list[Trade]]]
```

Executes script and yields results for each bar.

**Returns:** Iterator yielding `(ohlcv_data, plot_results)` or `(ohlcv_data, plot_results, closed_trades)`

---

## OHLCVReader

Memory-mapped reader for binary OHLCV data files using MMAP.

### Constructor

```python
OHLCVReader(path: str | Path)
```

**Parameters:**

- `path`: Path to `.ohlcv` binary file


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

Reads single candle at specific position.

#### read_from()

```python
def read_from(
        self,
        start_timestamp: int,
        end_timestamp: int | None = None,
        skip_gaps: bool = True
) -> Iterator[OHLCV]
```

Reads candles in timestamp range.

**Parameters:**

- `start_timestamp`: Unix timestamp start
- `end_timestamp`: Unix timestamp end (None = read to end)
- `skip_gaps`: Skip gap-filled records (volume = -1)


#### get_size()

```python
def get_size(self, start_timestamp: int | None = None, end_timestamp: int | None = None) -> int
```

Counts records in timestamp range.

#### get_positions()

```python
def get_positions(self, start_timestamp: int | None = None, end_timestamp: int | None = None) -> tuple[int, int]
```

Gets start and end positions for timestamp range.

#### Export Methods

```python
def save_to_csv(self, path: str, as_datetime: bool = False) -> None


    def save_to_json(self, path: str, as_datetime: bool = False) -> None
```

Exports data to CSV or JSON format.

---

## OHLCVWriter

Writes OHLCV data to binary format with automatic gap filling.

### Constructor

```python
OHLCVWriter(path: str | Path, truncate: bool = False)
```

**Parameters:**

- `path`: Path to output `.ohlcv` file
- `truncate`: If True, truncate existing file on open (default: False)


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

Writes single OHLCV candle with automatic gap filling.

#### seek_to_timestamp() / seek()

```python
def seek_to_timestamp(self, timestamp: int) -> None


    def seek(self, position: int) -> None  
```

Moves write position for data insertion/overwriting.

**Important:** When using `seek()` to overwrite data, timestamps must remain in chronological order. Writing a timestamp
earlier than subsequent records will raise a `ValueError`.

#### truncate()

```python
def truncate(self) -> None
```

Removes all data after current position.

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

Imports data from CSV, JSON, or TXT files.

---

## DataConverter

Converts data files to OHLCV format with automatic analysis.

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

Auto-detects symbol and provider from filename.

**Returns:** `(symbol, provider)` tuple

**Pattern Examples:** `BTCUSDT.csv` → `("BTC/USDT", None)`, `BYBIT_ETHUSDT_1h.csv` → `("ETH/USDT", "bybit")`

#### is_conversion_required()

```python
@staticmethod
def is_conversion_required(source_path: Path, ohlcv_path: Path | None = None) -> bool
```

Checks if conversion is needed.

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

Converts CSV/JSON/TXT to OHLCV format.

**Parameters:**

- `file_path`: Source data file path
- `force`: Force conversion even if OHLCV file exists and is newer
- `provider`: Data provider name (auto-detected if None)
- `symbol`: Symbol name (auto-detected if None)
- `timezone`: Timezone for timestamp conversion



---

## SymInfo

Symbol metadata container.

### Class Methods

#### load_toml()

```python
@classmethod
def load_toml(cls, path: Path) -> SymInfo
```

Loads symbol information from TOML file.


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


### Methods

#### save_toml()

```python
def save_toml(self, path: Path) -> None
```

Saves symbol information to TOML file.

### Key Attributes

- `symbol`: Combined ticker symbol
- `timeframe`: Trading timeframe
- `timezone`: Timezone for timestamps
- `currency`: Quote currency
- `type`: Asset type classification

---

## PyneComp (Pine Script Compilation)

Compiles Pine Script files to Python.

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

Compiles .pine file to .py file.

#### needs_compilation()

```python
def needs_compilation(self, pine_path: Path, py_path: Path) -> bool
```

Checks if compilation is needed.


---

## Data Providers

Downloads historical market data.

### Available Providers

- `pynecore.providers.ccxt.CCXTProvider`
- `pynecore.providers.capitalcom.CapitalComProvider`

### Provider Methods

```python
def get_list_of_symbols(self) -> list[str]  # List available symbols


    def get_syminfo(self, timeframe: str) -> SymInfo  # Get symbol metadata  


    def download_ohlcv(self, timeframe: str, since: datetime, until: datetime)  # Download data iterator
```

---

## OHLCV Data Type

Core data structure for candlestick data.

```python
OHLCV(
    timestamp: int,  # Unix timestamp
    open: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    extra_fields: dict = {}  # Optional
)
```

---

## Error Handling

Common exceptions:

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


---

