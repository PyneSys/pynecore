<!--
---
weight: 10001
title: "Quick Start"
description: "Quick start guide for PyneCore's programmatic API"
icon: "play"
date: "2025-08-11"
lastmod: "2025-08-11"
draft: false
toc: true
categories: ["Advanced", "API"]
tags: ["quick-start", "api", "ohlcv", "syminfo", "scriptrunner"]
---
-->


# Programmatic Usage

Execute PyneCore scripts programmatically using Python code instead of the CLI. This guide provides API references and working examples for programmatic usage.

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

## Strategy Outputs

Save script results to files for analysis and visualization:

```python
# Configure output paths (all optional)
runner = ScriptRunner(
    script_path=Path("strategy.py"),
    ohlcv_iter=reader,
    syminfo=syminfo,
    plot_path=Path("plots.csv"),      # Chart plot data
    trade_path=Path("trades.csv"),    # Trading strategy results  
    strat_path=Path("strategy.csv")   # Strategy statistics
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
- `symbol`, `timeframe`: Basic identification
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

### ScriptRunner Class

Execute PyneCore scripts with OHLCV data:

```python
runner = ScriptRunner(
    script_path: Path,           # Path to .py script file
    ohlcv_iter: Iterable[OHLCV], # Data iterator (e.g., OHLCVReader)
    syminfo: SymInfo,            # Symbol information
    
    # Optional output paths (keyword-only)
    plot_path: Path | None = None,    # Save plot data
    trade_path: Path | None = None,   # Save trade results (NOT equity_path!)
    strat_path: Path | None = None,   # Save strategy statistics
    
    # Advanced options
    update_syminfo_every_run: bool = False,  # For parallel execution
    last_bar_index: int = 0                  # Index of final bar
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

# ✅ Process data in chunks for very very large datasets
start_ts = reader.start_timestamp
chunk_size = 86400 * 30  # 30 days in seconds

while start_ts < reader.end_timestamp:
    end_ts = min(start_ts + chunk_size, reader.end_timestamp)
    chunk_data = reader.read_from(start_ts, end_ts)
    # Process chunk...
    start_ts = end_ts
```

### Error Handling
```python
try:
    with OHLCVReader(Path("data.ohlcv")) as reader:
        runner = ScriptRunner(script_path, reader, syminfo)
        runner.run()
except FileNotFoundError:
    print("Data file not found")
except ImportError as e:
    print(f"Script error: {e}")
except ValueError as e:
    print(f"Data format error: {e}")
```

### Performance Optimization
- Use `skip_gaps=True` in `read_from()` to avoid processing invalid data
- Set appropriate `last_bar_index` for better progress tracking
- Consider `update_syminfo_every_run=False` for single-threaded usage
- Process data in time chunks for memory efficiency with large datasets