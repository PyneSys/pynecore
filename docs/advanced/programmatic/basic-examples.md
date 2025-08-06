<!--
---
weight: 10002
title: "Basic Examples"
description: "Simple examples of using PyneCore programmatically"
icon: "play_circle"
date: "2025-08-06"
lastmod: "2025-08-06"
draft: false
toc: true
categories: ["Advanced", "API", "Examples"]
tags: ["examples", "tutorial", "basic", "getting-started"]
---
-->

# Basic Examples

This page provides simple, focused examples showing how to use PyneCore's programmatic API. Each example is self-contained and demonstrates a specific use case.

## Reading OHLCV Data

### Example 1: Basic Data Reading

```python
from pathlib import Path
from pynecore.core.ohlcv_file import OHLCVReader

def read_ohlcv_data(file_path: str):
    """Read and display basic information about OHLCV data."""
    data_file = Path(file_path)
    
    with OHLCVReader.open(data_file) as reader:
        print(f"File: {data_file.name}")
        print(f"Total bars: {reader.get_size()}")
        print(f"Date range: {reader.start_datetime} to {reader.end_datetime}")
        print(f"Interval: {reader.interval} seconds")
        
        # Show first 5 bars
        print("\nFirst 5 bars:")
        for i, bar in enumerate(reader):
            if i >= 5:
                break
            print(f"  {bar.timestamp}: O={bar.open} H={bar.high} L={bar.low} C={bar.close} V={bar.volume}")

# Usage
read_ohlcv_data("data/BTCUSD_1D.ohlcv")
```

### Example 2: Data Filtering

```python
from datetime import datetime
from pynecore.core.ohlcv_file import OHLCVReader

def find_high_volume_bars(file_path: str, volume_threshold: float):
    """Find bars with volume above threshold."""
    with OHLCVReader.open(Path(file_path)) as reader:
        high_volume_bars = []
        
        for bar in reader:
            if bar.volume > volume_threshold:
                high_volume_bars.append({
                    'timestamp': bar.timestamp,
                    'close': bar.close,
                    'volume': bar.volume
                })
        
        print(f"Found {len(high_volume_bars)} bars with volume > {volume_threshold}")
        
        # Show top 5 by volume
        top_bars = sorted(high_volume_bars, key=lambda x: x['volume'], reverse=True)[:5]
        for bar in top_bars:
            print(f"  {bar['timestamp']}: Close={bar['close']}, Volume={bar['volume']}")

# Usage
find_high_volume_bars("data/BTCUSD_1D.ohlcv", 1000000)
```

## Working with Symbol Information

### Example 3: Creating Symbol Info

```python
from pynecore.core.syminfo import SymInfo

def create_symbol_info_examples():
    """Examples of creating SymInfo objects."""
    
    # Basic symbol info
    btc_info = SymInfo(
        symbol="BTCUSD",
        exchange="BINANCE",
        timeframe="1D"
    )
    print(f"BTC: {btc_info.symbol} on {btc_info.exchange} ({btc_info.timeframe})")
    
    # Stock symbol with additional info
    aapl_info = SymInfo(
        symbol="AAPL",
        exchange="NASDAQ",
        timeframe="1H",
        description="Apple Inc.",
        currency="USD",
        sector="Technology"
    )
    print(f"AAPL: {aapl_info.description} - {aapl_info.sector}")
    
    # Forex pair
    eur_info = SymInfo(
        symbol="EURUSD",
        exchange="FOREX",
        timeframe="5m",
        base_currency="EUR",
        quote_currency="USD"
    )
    print(f"EUR/USD: {eur_info.base_currency}/{eur_info.quote_currency}")

# Usage
create_symbol_info_examples()
```

### Example 4: Loading Symbol Info from File

```python
from pathlib import Path
from pynecore.core.syminfo import SymInfo

def load_symbol_from_file(syminfo_path: str):
    """Load symbol information from a TOML file."""
    try:
        syminfo = SymInfo.load_from_file(Path(syminfo_path))
        
        print(f"Loaded symbol: {syminfo.symbol}")
        print(f"Exchange: {syminfo.exchange}")
        print(f"Timeframe: {syminfo.timeframe}")
        
        # Access additional fields if they exist
        if hasattr(syminfo, 'description'):
            print(f"Description: {syminfo.description}")
        if hasattr(syminfo, 'currency'):
            print(f"Currency: {syminfo.currency}")
            
        return syminfo
        
    except FileNotFoundError:
        print(f"Symbol info file not found: {syminfo_path}")
        return None
    except Exception as e:
        print(f"Error loading symbol info: {e}")
        return None

# Usage
syminfo = load_symbol_from_file("symbols/BTCUSD.toml")
```

## Running Scripts

### Example 5: Simple Indicator Script

```python
from pathlib import Path
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.syminfo import SymInfo
from pynecore.core.script_runner import ScriptRunner

def run_simple_indicator(data_path: str, script_path: str):
    """Run a simple indicator script."""
    # Create symbol info
    syminfo = SymInfo(
        symbol="BTCUSD",
        exchange="BINANCE",
        timeframe="1D"
    )
    
    # Load data and run script
    with OHLCVReader.open(Path(data_path)) as reader:
        print(f"Running {script_path} on {reader.get_size()} bars")
        
        runner = ScriptRunner(
            script_path=script_path,
            ohlcv_iter=reader,
            syminfo=syminfo
        )
        
        success = runner.run()
        
        if success:
            print("✅ Script executed successfully")
        else:
            print("❌ Script execution failed")
        
        return success

# Usage
run_simple_indicator(
    data_path="data/BTCUSD_1D.ohlcv",
    script_path="scripts/simple_ma.py"
)
```

### Example 6: Indicator with Plot Output

```python
def run_indicator_with_output(data_path: str, script_path: str, output_dir: str):
    """Run an indicator and save plot data."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    syminfo = SymInfo(symbol="BTCUSD", exchange="BINANCE", timeframe="1D")
    
    with OHLCVReader.open(Path(data_path)) as reader:
        runner = ScriptRunner(
            script_path=script_path,
            ohlcv_iter=reader,
            syminfo=syminfo,
            plot_path=str(output_path / "indicator_data.csv")
        )
        
        success = runner.run()
        
        if success:
            plot_file = output_path / "indicator_data.csv"
            if plot_file.exists():
                print(f"✅ Plot data saved to: {plot_file}")
                
                # Show file size
                size_kb = plot_file.stat().st_size / 1024
                print(f"   File size: {size_kb:.1f} KB")
            else:
                print("⚠️  Script ran but no plot data was generated")
        else:
            print("❌ Script execution failed")

# Usage
run_indicator_with_output(
    data_path="data/BTCUSD_1D.ohlcv",
    script_path="scripts/bollinger_bands.py",
    output_dir="output/bollinger"
)
```

### Example 7: Strategy Script

```python
def run_simple_strategy(data_path: str, strategy_path: str, output_dir: str):
    """Run a Pine strategy script and save outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    syminfo = SymInfo(symbol="BTCUSD", exchange="BINANCE", timeframe="1D")
    
    with OHLCVReader.open(Path(data_path)) as reader:
        runner = ScriptRunner(
            script_path=strategy_path,
            ohlcv_iter=reader,
            syminfo=syminfo,
            plot_path=str(output_path / "strategy_plot.csv"),
            trade_path=str(output_path / "trades.csv"),
            strat_path=str(output_path / "strategy_stats.csv")
        )
        
        success = runner.run()
        return success

# Usage
run_simple_strategy(
    data_path="data/BTCUSD_1D.ohlcv",
    strategy_path="scripts/ma_crossover_strategy.py",
    output_dir="output/ma_strategy"
)
```

## Progress Tracking

### Example 8: Script with Progress Callback

```python
from datetime import datetime

def run_with_progress(data_path: str, script_path: str):
    """Run a script with progress tracking."""
    syminfo = SymInfo(symbol="BTCUSD", exchange="BINANCE", timeframe="1D")
    
    def progress_callback(current_time: datetime) -> None:
        """Called for each processed bar."""
        if current_time != datetime.max:
            print(f"\rProcessing: {current_time.strftime('%Y-%m-%d %H:%M:%S')}", end="")
        else:
            print("\n✅ Processing complete!")
    
    with OHLCVReader.open(Path(data_path)) as reader:
        print(f"Starting analysis of {reader.get_size()} bars...")
        
        runner = ScriptRunner(
            script_path=script_path,
            ohlcv_iter=reader,
            syminfo=syminfo
        )
        
        # Run with progress tracking
        for _ in runner.run_iter():
            pass  # Progress is handled by callback
        
        print("Analysis completed!")

# Usage
run_with_progress(
    data_path="data/BTCUSD_1D.ohlcv",
    script_path="scripts/rsi_indicator.py"
)
```

## Partial Data Processing

### Example 9: Process Last N Bars

```python
def analyze_recent_data(data_path: str, script_path: str, last_n_bars: int = 100):
    """Analyze only the most recent N bars."""
    syminfo = SymInfo(symbol="BTCUSD", exchange="BINANCE", timeframe="1D")
    
    with OHLCVReader.open(Path(data_path)) as reader:
        total_bars = reader.get_size()
        start_index = max(0, total_bars - last_n_bars)
        
        print(f"Analyzing last {last_n_bars} bars (from index {start_index})")
        
        runner = ScriptRunner(
            script_path=script_path,
            ohlcv_iter=reader.read_from(start_index),
            syminfo=syminfo,
            last_bar_index=total_bars - 1
        )
        
        success = runner.run()
        
        if success:
            print(f"✅ Analyzed {last_n_bars} recent bars successfully")
        else:
            print("❌ Analysis failed")

# Usage
analyze_recent_data(
    data_path="data/BTCUSD_1D.ohlcv",
    script_path="scripts/momentum_indicator.py",
    last_n_bars=50
)
```

## Multiple Symbols

### Example 10: Process Multiple Symbols

```python
def analyze_multiple_symbols(symbols_data: list[dict], script_path: str):
    """Analyze multiple symbols with the same script."""
    results = []
    
    for symbol_data in symbols_data:
        try:
            syminfo = SymInfo(
                symbol=symbol_data['symbol'],
                exchange=symbol_data['exchange'],
                timeframe=symbol_data['timeframe']
            )
            
            with OHLCVReader.open(Path(symbol_data['data_path'])) as reader:
                runner = ScriptRunner(
                    script_path=script_path,
                    ohlcv_iter=reader,
                    syminfo=syminfo,
                    plot_path=f"output/{symbol_data['symbol']}_analysis.csv"
                )
                
                success = runner.run()
                results.append({
                    'symbol': symbol_data['symbol'],
                    'success': success
                })
                
        except Exception as e:
            results.append({
                'symbol': symbol_data['symbol'],
                'success': False,
                'error': str(e)
            })
    
    return results

# Usage
symbols = [
    {
        'symbol': 'BTCUSD',
        'exchange': 'BINANCE',
        'timeframe': '1D',
        'data_path': 'data/BTCUSD_1D.ohlcv'
    },
    {
        'symbol': 'ETHUSD',
        'exchange': 'BINANCE',
        'timeframe': '1D',
        'data_path': 'data/ETHUSD_1D.ohlcv'
    },
    {
        'symbol': 'ADAUSD',
        'exchange': 'BINANCE',
        'timeframe': '1D',
        'data_path': 'data/ADAUSD_1D.ohlcv'
    }
]

results = analyze_multiple_symbols(symbols, "scripts/rsi_divergence.py")
```

## Next Steps

These basic examples show the fundamental patterns for using PyneCore programmatically. For more complex scenarios, see:

- [Advanced Examples](advanced-examples.md) - Error handling, parallel processing, and production patterns
- [Best Practices](best-practices.md) - Performance optimization and common pitfalls
- [Core Components](core-components.md) - Detailed API reference