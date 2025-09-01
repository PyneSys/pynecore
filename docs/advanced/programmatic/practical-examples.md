<!--
---
weight: 10004
title: "Practical Examples"
description: "Real-world examples showing advanced PyneCore usage"
icon: "integration_instructions"
date: "2025-08-19"  
lastmod: "2025-09-01"
draft: false
toc: true
categories: ["Advanced", "API"]
tags: ["examples", "practical", "workflows", "automation"]
---
-->

# Practical Examples

Real-world applications of PyneCore's programmatic API for automation systems, batch processing, and trading applications.

## Batch Strategy Testing

Test multiple strategies across different time periods and symbols:

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
    """Test multiple strategies across different data and time periods"""
    results = []
    output_dir.mkdir(exist_ok=True)
    
    for strategy_path in strategies:
        for data_path in data_files:
            syminfo = SymInfo.load_toml(data_path.with_suffix(".toml"))
            
            for start_time, end_time in time_ranges:
                with OHLCVReader(data_path) as reader:
                    data_iter = reader.read_from(
                        int(start_time.timestamp()),
                        int(end_time.timestamp())
                    )
                    
                    test_id = f"{strategy_path.stem}_{data_path.stem}_{start_time.strftime('%Y%m%d')}"
                    
                    try:
                        runner = ScriptRunner(
                            strategy_path,
                            data_iter,
                            syminfo,
                            trade_path=output_dir / f"{test_id}_trades.csv",
                            strat_path=output_dir / f"{test_id}_stats.csv"
                        )
                        runner.run()
                        
                        results.append({
                            'strategy': strategy_path.stem,
                            'symbol': data_path.stem,
                            'period': f"{start_time.date()} to {end_time.date()}",
                            'status': 'success'
                        })
                    except Exception as e:
                        results.append({
                            'strategy': strategy_path.stem,
                            'symbol': data_path.stem,
                            'period': f"{start_time.date()} to {end_time.date()}",
                            'status': f'error: {e}'
                        })
    
    # Save summary
    pd.DataFrame(results).to_csv(output_dir / "test_summary.csv", index=False)
    return results
```

## Multi-Timeframe Analysis

Process the same data on multiple timeframes:

```python
from pynecore.core.ohlcv_file import OHLCVReader, OHLCVWriter
from pynecore.core.syminfo import SymInfo
from pynecore.types.ohlcv import OHLCV
from pathlib import Path
from collections import defaultdict

def resample_to_higher_timeframe(source_path: Path, target_minutes: int) -> Path:
    """Resample OHLCV data to higher timeframe"""
    output_path = source_path.parent / f"{source_path.stem}_{target_minutes}m.ohlcv"
    
    with OHLCVReader(source_path) as reader:
        interval = reader.interval
        if interval >= target_minutes * 60:
            raise ValueError(f"Source interval {interval}s >= target {target_minutes * 60}s")
        
        bars_per_candle = (target_minutes * 60) // interval
        buffer = []
        
        with OHLCVWriter(output_path, truncate=True) as writer:
            for candle in reader:
                buffer.append(candle)
                
                if len(buffer) == bars_per_candle:
                    # Aggregate buffer into one candle
                    resampled = OHLCV(
                        timestamp=buffer[0].timestamp,
                        open=buffer[0].open,
                        high=max(c.high for c in buffer),
                        low=min(c.low for c in buffer),
                        close=buffer[-1].close,
                        volume=sum(c.volume for c in buffer)
                    )
                    writer.write(resampled)
                    buffer = []
    
    # Copy and update syminfo
    syminfo = SymInfo.load_toml(source_path.with_suffix('.toml'))
    syminfo.period = f"{target_minutes}m"
    syminfo.save_toml(output_path.with_suffix('.toml'))
    
    return output_path

# Usage
timeframes = [5, 15, 60, 240]  # minutes
source = Path("BTCUSD_1m.ohlcv")

for tf in timeframes:
    resampled = resample_to_higher_timeframe(source, tf)
    print(f"Created {tf}m timeframe: {resampled}")
```

## Real-time Progress Monitoring

Monitor large-scale backtests with detailed progress:

```python
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.syminfo import SymInfo
import time
from datetime import datetime

class ProgressMonitor:
    def __init__(self, total_bars: int):
        self.total_bars = total_bars
        self.processed = 0
        self.start_time = time.time()
        
    def update(self, current_dt: datetime):
        self.processed += 1
        elapsed = time.time() - self.start_time
        rate = self.processed / elapsed if elapsed > 0 else 0
        eta = (self.total_bars - self.processed) / rate if rate > 0 else 0
        
        if self.processed % 100 == 0:  # Update every 100 bars
            print(f"[{self.processed}/{self.total_bars}] "
                  f"Rate: {rate:.1f} bars/s | "
                  f"ETA: {eta:.0f}s | "
                  f"Current: {current_dt.strftime('%Y-%m-%d')}")

def run_with_monitoring(script_path: Path, data_path: Path):
    syminfo = SymInfo.load_toml(data_path.with_suffix(".toml"))
    
    with OHLCVReader(data_path) as reader:
        monitor = ProgressMonitor(reader.size)
        
        runner = ScriptRunner(
            script_path, reader, syminfo,
            last_bar_index=reader.size - 1
        )
        runner.run(on_progress=monitor.update)
        
        elapsed = time.time() - monitor.start_time
        print(f"Completed in {elapsed:.1f}s ({reader.size/elapsed:.0f} bars/s)")
```

## Data Quality Validation

Validate and clean OHLCV data:

```python
from pynecore.core.ohlcv_file import OHLCVReader, OHLCVWriter
from pynecore.types.ohlcv import OHLCV
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ValidationReport:
    total_bars: int = 0
    gaps_found: int = 0
    invalid_ohlc: int = 0
    zero_volume: int = 0
    outliers: int = 0

def validate_and_clean(input_path: Path, fix_issues: bool = False) -> ValidationReport:
    """Validate OHLCV data and optionally fix issues"""
    report = ValidationReport()
    output_path = input_path.with_name(f"{input_path.stem}_cleaned.ohlcv") if fix_issues else None
    
    with OHLCVReader(input_path) as reader:
        prev_candle = None
        expected_interval = reader.interval
        cleaned_data = []
        
        for candle in reader:
            report.total_bars += 1
            is_valid = True
            
            # Check for gaps
            if prev_candle:
                gap = candle.timestamp - prev_candle.timestamp
                if gap > expected_interval * 1.5:
                    report.gaps_found += 1
            
            # Validate OHLC relationships
            if not (candle.low <= candle.open <= candle.high and
                    candle.low <= candle.close <= candle.high):
                report.invalid_ohlc += 1
                if fix_issues:
                    # Fix by adjusting high/low
                    candle = OHLCV(
                        timestamp=candle.timestamp,
                        open=candle.open,
                        high=max(candle.high, candle.open, candle.close),
                        low=min(candle.low, candle.open, candle.close),
                        close=candle.close,
                        volume=candle.volume
                    )
            
            # Check for zero/negative volume
            if candle.volume <= 0:
                report.zero_volume += 1
            
            # Detect price outliers (>10% move in single bar)
            if prev_candle:
                price_change = abs(candle.close - prev_candle.close) / prev_candle.close
                if price_change > 0.1:
                    report.outliers += 1
            
            if fix_issues and is_valid:
                cleaned_data.append(candle)
            
            prev_candle = candle
    
    # Write cleaned data if requested
    if fix_issues and cleaned_data:
        with OHLCVWriter(output_path, truncate=True) as writer:
            for candle in cleaned_data:
                writer.write(candle)
        print(f"Cleaned data saved to {output_path}")
    
    return report

# Usage
report = validate_and_clean(Path("BTCUSD_1D.ohlcv"), fix_issues=True)
print(f"Validation Report: {report}")
```

## Automated Trading System

Complete automated trading system with data updates and strategy execution:

```python
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.ohlcv_file import OHLCVReader, OHLCVWriter
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
        
    def update_data(self, symbol: str, timeframe: str = "1D") -> Path:
        """Update market data for a symbol"""
        provider = CCXTProvider(symbol=symbol, config_dir=self.config_dir)
        
        safe_symbol = symbol.replace(":", "_").replace("/", "_")
        data_file = self.data_dir / f"{safe_symbol}_{timeframe}.ohlcv"
        
        # Get latest data
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=7)
        
        with OHLCVWriter(data_file, truncate=False) as writer:
            if data_file.exists():
                writer.seek_to_timestamp(int(start_date.timestamp()))
            
            for candle in provider.download_ohlcv(
                timeframe=timeframe,
                since=start_date,
                until=end_date
            ):
                writer.write(candle)
        
        # Update syminfo if needed
        syminfo_file = data_file.with_suffix(".toml")
        if not syminfo_file.exists():
            syminfo = provider.get_syminfo(timeframe=timeframe)
            syminfo.save_toml(syminfo_file)
        
        return data_file
    
    def run_strategy(self, strategy_name: str, symbol: str, timeframe: str = "1D") -> dict:
        """Run a strategy and return signals"""
        # Update data
        data_file = self.update_data(symbol, timeframe)
        
        # Load strategy
        strategy_path = self.strategies_dir / f"{strategy_name}.py"
        if not strategy_path.exists():
            raise FileNotFoundError(f"Strategy not found: {strategy_path}")
        
        syminfo = SymInfo.load_toml(data_file.with_suffix(".toml"))
        
        # Run on recent data only (last 100 bars for efficiency)
        with OHLCVReader(data_file) as reader:
            all_data = list(reader)
            recent_data = all_data[-100:] if len(all_data) > 100 else all_data
            
            output_dir = Path("./signals") / datetime.now().strftime("%Y%m%d")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            runner = ScriptRunner(
                strategy_path,
                recent_data,
                syminfo,
                trade_path=output_dir / f"{strategy_name}_{symbol.replace(':', '_')}.csv"
            )
            runner.run()
            
            # Parse latest signal from trade file
            return self._get_latest_signal(output_dir / f"{strategy_name}_{symbol.replace(':', '_')}.csv")
    
    def _get_latest_signal(self, trade_file: Path) -> dict:
        """Extract latest trading signal"""
        if not trade_file.exists():
            return {"action": "hold"}
        
        # Parse CSV and get last trade signal
        import pandas as pd
        trades = pd.read_csv(trade_file)
        if not trades.empty:
            last_trade = trades.iloc[-1]
            return {
                "action": last_trade.get("action", "hold"),
                "timestamp": last_trade.get("timestamp", ""),
                "price": last_trade.get("price", 0)
            }
        return {"action": "hold"}

# Usage
trading_system = TradingSystem(
    config_dir=Path("./config"),
    strategies_dir=Path("./strategies"),
    data_dir=Path("./data")
)

# Schedule runs
schedule.every().hour.do(
    trading_system.run_strategy,
    "momentum_strategy",
    "BYBIT:BTC/USDT:USDT",
    "1H"
)

# Run scheduler
print("Trading system started. Press Ctrl+C to stop.")
while True:
    schedule.run_pending()
    time.sleep(60)
```

## Parallel Script Execution

Run multiple scripts on the same data efficiently:

```python
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.syminfo import SymInfo
from pathlib import Path
import concurrent.futures

def run_script(args: tuple) -> dict:
    """Run a single script (for parallel execution)"""
    script_path, data_path, output_dir = args
    
    try:
        syminfo = SymInfo.load_toml(data_path.with_suffix(".toml"))
        
        with OHLCVReader(data_path) as reader:
            output_file = output_dir / f"{script_path.stem}_results.csv"
            
            runner = ScriptRunner(
                script_path, reader, syminfo,
                plot_path=output_file
            )
            runner.run()
            
        return {"script": script_path.stem, "status": "success", "output": output_file}
    except Exception as e:
        return {"script": script_path.stem, "status": "error", "error": str(e)}

def parallel_script_execution(scripts: list[Path], data_path: Path, max_workers: int = 4):
    """Execute multiple scripts in parallel"""
    output_dir = Path("./parallel_results")
    output_dir.mkdir(exist_ok=True)
    
    # Prepare arguments for each script
    args_list = [(script, data_path, output_dir) for script in scripts]
    
    # Execute in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_script, args_list))
    
    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    print(f"Executed {len(scripts)} scripts: {successful} successful")
    
    return results

# Usage
scripts = list(Path("./indicators").glob("*.py"))
results = parallel_script_execution(scripts, Path("BTCUSD_1D.ohlcv"))
```

## Advanced Data Analysis

Analyze market data patterns and statistics:

```python
from pynecore.core.ohlcv_file import OHLCVReader
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class MarketStats:
    volatility: float
    average_volume: float
    price_range: tuple[float, float]
    trend_direction: str
    gaps_count: int

def analyze_market_data(data_path: Path, window: int = 20) -> MarketStats:
    """Perform advanced market analysis"""
    
    with OHLCVReader(data_path) as reader:
        prices = []
        volumes = []
        gaps = 0
        prev_close = None
        
        for candle in reader:
            prices.append(candle.close)
            volumes.append(candle.volume)
            
            # Detect gaps
            if prev_close and abs(candle.open - prev_close) / prev_close > 0.02:
                gaps += 1
            prev_close = candle.close
        
        # Calculate statistics
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Determine trend
        sma_short = np.mean(prices[-window:])
        sma_long = np.mean(prices[-window*3:])
        trend = "bullish" if sma_short > sma_long else "bearish"
        
        return MarketStats(
            volatility=volatility,
            average_volume=np.mean(volumes),
            price_range=(min(prices), max(prices)),
            trend_direction=trend,
            gaps_count=gaps
        )

# Usage
stats = analyze_market_data(Path("BTCUSD_1D.ohlcv"))
print(f"Market Analysis: {stats}")
```

These examples demonstrate advanced PyneCore usage for real-world trading applications, automation, and analysis tasks.