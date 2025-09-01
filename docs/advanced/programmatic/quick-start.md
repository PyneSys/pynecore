<!--
---
weight: 10001
title: "Programmatic Quick Start"
description: "Quick start guide for PyneCore's programmatic Python API"
icon: "tips_and_updates"
date: "2025-08-19"
lastmod: "2025-09-01"
draft: false
toc: true
categories: ["Advanced", "API"]
tags: ["quick-start", "api", "ohlcv", "syminfo", "scriptrunner"]
---
-->

# Programmatic Quick Start

PyneCore's Python API lets you execute scripts, download data, and manage OHLCV files programmatically. Every CLI command has a programmatic equivalent.

## Basic Script Execution

Run a PyneCore script with OHLCV data:

```python
from pathlib import Path
from pynecore.core.syminfo import SymInfo
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.script_runner import ScriptRunner

# Load symbol information
syminfo = SymInfo.load_toml(Path("data.toml"))

# Execute script
with OHLCVReader(Path("data.ohlcv")) as reader:
    runner = ScriptRunner(
        script_path=Path("strategy.py"),
        ohlcv_iter=reader,
        syminfo=syminfo
    )
    runner.run()
```

## Automatic Data Conversion

Convert CSV/JSON/TXT files automatically, just like the CLI:

```python
from pynecore.core.data_converter import DataConverter
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.syminfo import SymInfo
from pynecore.core.script_runner import ScriptRunner
from pathlib import Path

def run_with_auto_conversion(script_path: Path, data_path: Path):
    """Run script with automatic data conversion if needed"""
    
    # Convert non-OHLCV files automatically
    if data_path.suffix.lower() in ['.csv', '.json', '.txt']:
        converter = DataConverter()
        
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

## Save Strategy Results

Export results to CSV files for analysis:

```python
with OHLCVReader(data_path) as reader:
    runner = ScriptRunner(
        script_path, reader, syminfo,
        plot_path=Path("plots.csv"),      # Chart indicators
        trade_path=Path("trades.csv"),     # Trade details
        strat_path=Path("statistics.csv")  # Performance metrics
    )
    runner.run()
```

## Process Time Ranges

Run scripts on specific date ranges:

```python
from datetime import datetime, UTC

start = datetime(2024, 1, 1, tzinfo=UTC)
end = datetime(2024, 6, 30, tzinfo=UTC)

with OHLCVReader(data_path) as reader:
    # Get data subset
    data_iter = reader.read_from(
        int(start.timestamp()),
        int(end.timestamp())
    )
    
    runner = ScriptRunner(script_path, data_iter, syminfo)
    runner.run()
```

## Download Market Data

Download data programmatically:

```python
from pynecore.providers.ccxt import CCXTProvider
from pynecore.core.ohlcv_file import OHLCVWriter
from datetime import datetime, UTC, timedelta

# Setup provider
provider = CCXTProvider(
    symbol="BYBIT:BTC/USDT:USDT",
    config_dir=Path("./config")
)

# Download last 30 days
end_date = datetime.now(UTC)
start_date = end_date - timedelta(days=30)

# Save to OHLCV file
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

## Convert Data Formats

Convert between CSV, JSON, TXT, and OHLCV:

```python
from pynecore.core.data_converter import DataConverter

converter = DataConverter()

# Auto-detect symbol from filename
# BTCUSDT_1h.csv → Symbol: BTC/USDT
converter.convert_to_ohlcv(
    file_path=Path("BTCUSDT_1h.csv"),
    force=True  # Overwrite existing
)

# Manual conversion with OHLCVWriter
from pynecore.core.ohlcv_file import OHLCVWriter

with OHLCVWriter(Path("output.ohlcv"), truncate=True) as writer:
    writer.load_from_csv(Path("data.csv"), tz="UTC")
    # Access analysis results
    print(f"Detected tick size: {writer.analyzed_tick_size}")
```

## Compile Pine Script

Convert Pine Script to Python:

```python
from pynecore.pynesys.compiler import PyneComp

compiler = PyneComp(api_key="your-api-key")

if compiler.needs_compilation(Path("script.pine"), Path("script.py")):
    compiler.compile(Path("script.pine"), Path("script.py"))
```

## Progress Monitoring

Track execution progress:

```python
def show_progress(current_dt):
    print(f"Processing: {current_dt}", end='\r')

with OHLCVReader(data_path) as reader:
    runner = ScriptRunner(
        script_path, reader, syminfo,
        last_bar_index=reader.size - 1
    )
    runner.run(on_progress=show_progress)
```

## Error Handling

Handle common errors:

```python
try:
    syminfo = SymInfo.load_toml(Path("data.toml"))
    
    with OHLCVReader(Path("data.ohlcv")) as reader:
        runner = ScriptRunner(script_path, reader, syminfo)
        runner.run()
        
except FileNotFoundError as e:
    print(f"File not found: {e}")
except ImportError as e:
    print(f"Script error - check @pyne decorator: {e}")
except ValueError as e:
    print(f"Data format error: {e}")
```

## Next Steps

- See [API Reference](./api-reference) for complete class documentation
- Check [Practical Examples](./practical-examples) for real-world use cases
- Review [PyneCore documentation](https://docs.pynecore.com) for more details