<!--
---
weight: 10001
title: "Core Components"
description: "Detailed documentation of PyneCore's programmatic API components"
icon: "layers"
date: "2025-08-06"
lastmod: "2025-08-06"
draft: false
toc: true
categories: ["Advanced", "API"]
tags: ["components", "api", "ohlcv", "syminfo", "scriptrunner"]
---
-->

# Core Components

PyneCore's programmatic API consists of three main components that work together to execute Pine scripts programmatically.

## OHLCVReader

The `OHLCVReader` class provides efficient access to OHLCV market data stored in PyneCore's optimized binary format.

### Basic Usage

```python
from pynecore.core.ohlcv_file import OHLCVReader
from pathlib import Path

# Open an OHLCV file
ohlcv_file = Path("data/BTCUSD_1D.ohlcv")
with OHLCVReader.open(ohlcv_file) as reader:
    # Get basic information
    print(f"Data size: {reader.get_size()} bars")
    print(f"Start: {reader.start_datetime}")
    print(f"End: {reader.end_datetime}")
    
    # Iterate through data
    for bar in reader:
        print(f"Time: {bar.timestamp}, Close: {bar.close}")
```

### Key Methods

#### `open(file_path: Path) -> OHLCVReader`
Opens an OHLCV file for reading. Returns a context manager that automatically handles file cleanup.

```python
with OHLCVReader.open(Path("data.ohlcv")) as reader:
    # Use reader here
    pass
# File is automatically closed
```

#### `__iter__() -> Iterator[OHLCV]`
Provides iteration over all OHLCV bars in the file.

```python
for bar in reader:
    print(f"OHLC: {bar.open}, {bar.high}, {bar.low}, {bar.close}")
```

#### `read_from(start_index: int) -> Iterator[OHLCV]`
Reads from a specific bar index, useful for partial data processing.

```python
# Start reading from the 100th bar
for bar in reader.read_from(100):
    process_bar(bar)
```

#### `get_size() -> int`
Returns the total number of bars in the file.

```python
total_bars = reader.get_size()
print(f"Processing {total_bars} bars")
```

### Properties

#### `start_datetime: datetime`
First bar's timestamp as a Python datetime object.

#### `end_datetime: datetime`
Last bar's timestamp as a Python datetime object.

#### `interval: int`
Time interval between bars in seconds.

```python
print(f"Data covers {reader.start_datetime} to {reader.end_datetime}")
print(f"Interval: {reader.interval} seconds")
```

## SymInfo

The `SymInfo` class contains symbol metadata required for script execution. It provides information about the trading symbol, including its name, exchange, and other relevant details.

### Basic Usage

```python
from pynecore.core.syminfo import SymInfo

# Create symbol information
syminfo = SymInfo(
    symbol="BTCUSD",
    exchange="BINANCE",
    timeframe="1D"
)

# Access symbol properties
print(f"Symbol: {syminfo.symbol}")
print(f"Exchange: {syminfo.exchange}")
print(f"Timeframe: {syminfo.timeframe}")
```

### Key Attributes

#### `symbol: str`
The trading symbol (e.g., "BTCUSD", "EURUSD", "AAPL").

#### `exchange: str`
The exchange name (e.g., "BINANCE", "NYSE", "NASDAQ").

#### `timeframe: str`
The timeframe (e.g., "1D", "1H", "5m", "15s").

#### Additional Properties
SymInfo may contain additional metadata fields depending on your data source:

```python
# Example with additional fields
syminfo = SymInfo(
    symbol="AAPL",
    exchange="NASDAQ",
    timeframe="1H",
    description="Apple Inc.",
    currency="USD",
    sector="Technology"
)
```

### Loading from File

If you have symbol information stored in a file:

```python
# Load from TOML file
syminfo = SymInfo.load_from_file(Path("symbols/BTCUSD.toml"))
```

## ScriptRunner

The `ScriptRunner` class is the core component that executes Pine scripts with the provided data and configuration.

### Basic Usage

```python
from pynecore.core.script_runner import ScriptRunner
from pathlib import Path

# Initialize the script runner
runner = ScriptRunner(
    script_path="indicators/cwr.py",
    ohlcv_iter=reader,
    syminfo=syminfo,
    plot_path="output/plot_data.csv",  # Optional: save plot data
    trade_path="output/trades.csv",    # Optional: save trade data
    update_syminfo_every_run=True,
    last_bar_index=None  # Process all bars
)

# Execute the script
result = runner.run()
print(f"Script execution completed: {result}")
```

### Constructor Parameters

#### Required Parameters

- **`script_path: str`**: Path to the Pine script file (.py or .pine)
- **`ohlcv_iter`**: Iterator providing OHLCV data (typically from OHLCVReader)
- **`syminfo: SymInfo`**: Symbol information object

#### Optional Parameters

- **`plot_path: str | None = None`**: Path to save plot data as CSV
- **`strat_path: str | None = None`**: Path to save strategy statistics as CSV
- **`trade_path: str | None = None`**: Path to save trade data as CSV
- **`update_syminfo_every_run: bool = True`**: Whether to update symbol info on each run
- **`last_bar_index: int | None = None`**: Limit processing to specific bar index

### Key Methods

#### `run() -> bool`
Execute the complete script and return success status.

```python
success = runner.run()
if success:
    print("Script executed successfully")
else:
    print("Script execution failed")
```

#### `run_iter() -> Iterator`
Execute script iteratively, yielding results per bar. Useful for progress tracking.

```python
for result in runner.run_iter():
    # Process intermediate results
    print(f"Processed bar: {result}")
```

### Output Files

Depending on the script type and parameters, ScriptRunner can generate several output files:

#### Plot Data (`plot_path`)
Contains indicator values and plot data in CSV format.

#### Trade Data (`trade_path`)
For strategy scripts, contains individual trade records.

#### Strategy Statistics (`strat_path`)
For strategy scripts, contains performance metrics and statistics.

#### Parallel Execution
When running multiple scripts in parallel, always set `update_syminfo_every_run=True`:

```python
runner = ScriptRunner(
    script_path="script.py",
    ohlcv_iter=reader,
    syminfo=syminfo,
    update_syminfo_every_run=True  # Required for parallel execution
)
```

## Component Interaction

Here's how the three components work together:

```python
# 1. Load data
with OHLCVReader.open(data_path) as reader:
    # 2. Create symbol info
    syminfo = SymInfo(symbol="BTCUSD", exchange="BINANCE", timeframe="1D")
    
    # 3. Run script
    runner = ScriptRunner(
        script_path="my_script.py",
        ohlcv_iter=reader,  # Data flows from reader to runner
        syminfo=syminfo     # Symbol info is passed to script
    )
    
    # 4. Execute
    runner.run()
```

The data flow is:
1. `OHLCVReader` provides market data
2. `SymInfo` provides symbol metadata
3. `ScriptRunner` combines both to execute the Pine script
4. Results are saved to specified output files

## Next Steps

- See [Basic Examples](basic-examples.md) for simple usage patterns
- Check [Advanced Examples](advanced-examples.md) for complex scenarios
- Review [Best Practices](best-practices.md) for optimization tips
```