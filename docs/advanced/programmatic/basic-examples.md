<!--
---
weight: 10003
title: "Basic Examples"
description: "Basic examples for PyneCore's programmatic API"
icon: "code"
date: "2025-08-19"
lastmod: "2025-08-19"
draft: false
toc: true
categories: ["Advanced", "API"]
tags: ["basic-examples", "api", "ohlcv", "syminfo", "scriptrunner"]
---
-->

# Basic Programmatic Examples

This page provides simple, focused examples that demonstrate PyneCore's core programmatic functionality. Each example shows the Python API equivalent of common CLI operations.

## 1) Basic Script Execution

Equivalent to: `pyne run my_strategy.py BTCUSD_1D.ohlcv`

**Note**: The CLI now supports automatic conversion for CSV/JSON/TXT files. Programmatically, you can achieve the same with the DataConverter class (see Example 6).

```python
from pathlib import Path
from pynecore.core.syminfo import SymInfo
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.script_runner import ScriptRunner

# File paths
script_path = Path("my_strategy.py")
data_path = Path("BTCUSD_1D.ohlcv")
syminfo_path = data_path.with_suffix(".toml")

# Load symbol information
syminfo = SymInfo.load_toml(syminfo_path)

# Run strategy on full dataset
with OHLCVReader(data_path) as reader:
    runner = ScriptRunner(script_path, reader, syminfo)
    runner.run()
    
print("Strategy execution completed!")
```

## 2) Run Script with Output Files

Save results to CSV files for analysis:

```python
# Create output directory
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

with OHLCVReader(data_path) as reader:
    runner = ScriptRunner(
        script_path, reader, syminfo,
        plot_path=output_dir / "indicators.csv",   # Chart data
        trade_path=output_dir / "trades.csv",      # Trade details (strategies only)
        strat_path=output_dir / "performance.csv", # Statistics (strategies only)
        last_bar_index=reader.size - 1             # For progress tracking
    )
    runner.run()
    
print(f"Results saved to {output_dir}/")
```

## 3) Process Specific Date Range

Run script on data from specific time period:

```python
from datetime import datetime, UTC

# Define date range
start_date = datetime(2024, 1, 1, tzinfo=UTC)
end_date = datetime(2024, 6, 30, tzinfo=UTC)

with OHLCVReader(data_path) as reader:
    # Get data for specific period
    data_subset = reader.read_from(
        int(start_date.timestamp()), 
        int(end_date.timestamp())
    )
    
    # Run strategy on subset
    runner = ScriptRunner(script_path, data_subset, syminfo)
    runner.run()
    
print(f"Processed data from {start_date.date()} to {end_date.date()}")
```

## 4) Progress Monitoring

Monitor script execution with progress callbacks:

```python
def show_progress(current_datetime):
    """Called for each processed bar"""
    print(f"Processing: {current_datetime.strftime('%Y-%m-%d %H:%M:%S')}", end='\r')

with OHLCVReader(data_path) as reader:
    print(f"Processing {reader.size:,} bars from {reader.start_datetime} to {reader.end_datetime}")
    
    runner = ScriptRunner(
        script_path, reader, syminfo,
        last_bar_index=reader.size - 1  # Required for progress tracking
    )
    
    # Run with progress feedback
    runner.run(on_progress=show_progress)
    
print("\n✓ Execution completed!")
```

## 5) Download Market Data

Download data programmatically (equivalent to `pyne data download`):

```python
from pynecore.providers.ccxt import CCXTProvider
from pynecore.core.ohlcv_file import OHLCVWriter
from datetime import datetime, UTC, timedelta

# Setup provider (requires ccxt.toml config file)
provider = CCXTProvider(
    symbol="BYBIT:BTC/USDT:USDT",
    config_dir=Path("./config")
)

# Download last 30 days
end_date = datetime.now(UTC)
start_date = end_date - timedelta(days=30)

print(f"Downloading BTC/USDT data from {start_date.date()} to {end_date.date()}")

# Save to binary OHLCV file
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

print("✓ Data download completed!")
```

## 6) Data Format Conversion

Convert between CSV, JSON, TXT, and binary OHLCV formats using the enhanced DataConverter:

```python
from pynecore.core.data_converter import DataConverter
from pynecore.core.ohlcv_file import OHLCVWriter, OHLCVReader
from pathlib import Path

# Enhanced automatic conversion (equivalent to: pyne data convert-from)
converter = DataConverter()

# Convert CSV/JSON/TXT to OHLCV with automatic symbol detection
file_path = Path("BTCUSDT_1h.csv")
converter.convert_to_ohlcv(
    file_path=file_path,
    provider="binance",     # Optional: auto-detected from filename
    symbol="BTC/USDT",     # Optional: auto-detected from filename  
    timezone="UTC",
    force=True             # Overwrite existing files
)

print("✓ Data converted with automatic TOML configuration generated")

# Manual CSV conversion with custom settings
with OHLCVWriter(Path("manual_output.ohlcv"), truncate=True) as writer:
    writer.load_from_csv(
        Path("custom_data.csv"),
        timestamp_column="timestamp",  # or "date", "time"
        tz="UTC"
    )

# Load TXT files (tab, semicolon, or pipe-delimited)
with OHLCVWriter(Path("txt_output.ohlcv"), truncate=True) as writer:
    writer.load_from_txt(
        Path("data.txt"),
        timestamp_column="datetime",
        tz="Europe/London"
    )

# Export binary OHLCV to CSV/JSON
with OHLCVReader(Path("manual_output.ohlcv")) as reader:
    # Export to CSV with human-readable dates
    reader.save_to_csv("exported.csv", as_datetime=True)
    
    # Export to JSON
    reader.save_to_json("exported.json", as_datetime=True)
    
print("✓ Data exported to CSV and JSON formats")
```

## 7) Automatic Data Conversion (CLI-equivalent)

Handle non-OHLCV files automatically, just like the CLI does:

```python
from pathlib import Path
from pynecore.core.data_converter import DataConverter
from pynecore.core.syminfo import SymInfo
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.script_runner import ScriptRunner

def run_with_any_data_format(script_path: Path, data_path: Path):
    """
    Run script with any data format, converting automatically if needed.
    Equivalent to: pyne run my_strategy.py data.csv
    """
    
    # Check if data needs conversion
    if data_path.suffix != ".ohlcv":
        converter = DataConverter()
        
        if converter.is_conversion_required(data_path):
            print(f"Converting {data_path.suffix} to OHLCV format...")
            
            # Auto-detect symbol and provider from filename
            detected_symbol, detected_provider = DataConverter.guess_symbol_from_filename(data_path)
            
            # Convert with automatic TOML generation
            converter.convert_to_ohlcv(
                file_path=data_path,
                provider=detected_provider or "unknown",
                symbol=detected_symbol or data_path.stem.upper(),
                force=True
            )
            
            print("✓ Conversion completed with automatic configuration")
        
        # Update path to point to OHLCV file
        data_path = data_path.with_suffix(".ohlcv")
    
    # Load symbol information (created during conversion or existing)
    syminfo = SymInfo.load_toml(data_path.with_suffix(".toml"))
    
    # Run script normally
    with OHLCVReader(data_path) as reader:
        runner = ScriptRunner(script_path, reader, syminfo)
        runner.run()
    
    print("✓ Script execution completed!")

# Example usage
script_file = Path("my_strategy.py")
data_file = Path("BTCUSDT_1h.csv")  # Could be .csv, .json, .txt, or .ohlcv

run_with_any_data_format(script_file, data_file)
```

## 8) Compile Pine Script

Compile Pine Script files to Python (equivalent to `pyne compile`):

```python
from pynecore.pynesys.compiler import PyneComp

# Initialize compiler with API key
compiler = PyneComp(api_key="your-pynesys-api-key")

pine_file = Path("my_indicator.pine")
py_file = Path("my_indicator.py")

# Check if compilation is needed
if compiler.needs_compilation(pine_file, py_file):
    print(f"Compiling {pine_file.name}...")
    compiler.compile(pine_file, py_file)
    print(f"✓ Compiled to {py_file.name}")
else:
    print("Python file is up to date")

# Now run the compiled script
with OHLCVReader(data_path) as reader:
    runner = ScriptRunner(py_file, reader, syminfo)
    runner.run()
```

## 9) Error Handling Best Practices

Handle common errors gracefully:

```python
try:
    # Load symbol information
    syminfo = SymInfo.load_toml(data_path.with_suffix(".toml"))
    
    # Execute script
    with OHLCVReader(data_path) as reader:
        runner = ScriptRunner(script_path, reader, syminfo)
        runner.run()
        
except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
    print("   Check that .ohlcv and .toml files exist")
    
except ImportError as e:
    print(f"❌ Script import error: {e}")
    print("   Ensure script has '@pyne' comment and decorated main() function")
    
except ValueError as e:
    print(f"❌ Data format error: {e}")
    print("   Check OHLCV file format and timestamps")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    print("   Check script syntax and data integrity")
```