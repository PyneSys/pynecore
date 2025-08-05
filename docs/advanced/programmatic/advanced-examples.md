<!--
---
weight: 10003
title: "Advanced Examples"
description: "Complex examples with error handling, parallel processing, and production patterns"
icon: "settings"
date: "2025-03-31"
lastmod: "2025-03-31"
draft: false
toc: true
categories: ["Advanced", "API", "Examples"]
tags: ["advanced", "error-handling", "parallel", "production"]
---
-->

# Advanced Examples

This page provides comprehensive examples for production use cases, including robust error handling, parallel processing, and real-world scenarios.

## Robust Script Execution

### Example 1: Production-Ready Analysis Function

```python
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.syminfo import SymInfo
from pynecore.core.script_runner import ScriptRunner

def run_indicator_analysis(
    data_path: Path,
    script_path: Path,
    syminfo_path: Path,
    output_dir: Path,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Run a PyneCore indicator script with comprehensive error handling.
    
    :param data_path: Path to OHLCV data file
    :param script_path: Path to Pine script file
    :param syminfo_path: Path to symbol info file
    :param output_dir: Directory for output files
    :param progress_callback: Optional callback for progress updates
    :return: Dictionary with execution results and metadata
    :raises FileNotFoundError: If required files don't exist
    :raises ValueError: If data is invalid
    :raises RuntimeError: If script execution fails
    """
    # Validate input files
    missing_files = []
    if not data_path.exists():
        missing_files.append(f"OHLCV data: {data_path}")
    if not script_path.exists():
        missing_files.append(f"Script: {script_path}")
    if not syminfo_path.exists():
        missing_files.append(f"Symbol info: {syminfo_path}")
    
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output paths
    plot_output = output_dir / "plot_data.csv"
    
    # Execution metadata
    start_time = datetime.now()
    result = {
        'success': False,
        'start_time': start_time,
        'end_time': None,
        'duration_seconds': None,
        'bars_processed': 0,
        'output_files': [],
        'errors': []
    }
    
    try:
        # Load symbol information
        if progress_callback:
            progress_callback("Loading symbol information...")
        
        try:
            syminfo = SymInfo.load_from_file(syminfo_path)
        except Exception as e:
            raise ValueError(f"Failed to load symbol info from {syminfo_path}: {e}")
        
        # Validate symbol info
        required_fields = ['symbol', 'exchange', 'timeframe']
        missing_fields = [field for field in required_fields if not hasattr(syminfo, field)]
        if missing_fields:
            raise ValueError(f"Symbol info missing required fields: {missing_fields}")
        
        # Create OHLCV reader and validate data
        if progress_callback:
            progress_callback(f"Opening OHLCV data from {data_path}...")
        
        with OHLCVReader.open(data_path) as reader:
            # Check data availability
            data_size = reader.get_size()
            if data_size == 0:
                raise ValueError(f"No data found in {data_path}")
            
            if data_size < 10:  # Minimum bars for meaningful analysis
                raise ValueError(f"Insufficient data: only {data_size} bars (minimum 10 required)")
            
            result['bars_processed'] = data_size
            
            if progress_callback:
                progress_callback(f"Found {data_size} bars from {reader.start_datetime} to {reader.end_datetime}")
            
            # Create script runner with progress tracking
            if progress_callback:
                progress_callback(f"Initializing script runner for {script_path.name}...")
            
            def on_progress(current_time: datetime) -> None:
                if progress_callback and current_time != datetime.max:
                    progress_callback(f"Processing: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            runner = ScriptRunner(
                script_path=str(script_path),
                ohlcv_iter=reader,
                syminfo=syminfo,
                plot_path=str(plot_output),
                last_bar_index=data_size - 1
            )
            
            # Run the script
            if progress_callback:
                progress_callback("Running script analysis...")
            
            try:
                # Use run_iter for progress tracking
                for _ in runner.run_iter():
                    pass  # Progress handled by callback
                
                result['success'] = True
                
            except ImportError as e:
                if "@pyne" in str(e):
                    raise RuntimeError(f"Script missing @pyne decorator: {script_path}")
                elif "main" in str(e):
                    raise RuntimeError(f"Script missing main function: {script_path}")
                else:
                    raise RuntimeError(f"Script import failed: {e}")
            
            except Exception as e:
                raise RuntimeError(f"Script execution failed: {e}")
        
        # Verify and catalog output files
        if plot_output.exists():
            result['output_files'].append({
                'type': 'plot_data',
                'path': str(plot_output),
                'size_bytes': plot_output.stat().st_size
            })
        else:
            result['errors'].append("Script completed but no plot data was generated")
        
        if progress_callback:
            progress_callback("Analysis completed successfully!")
        
    except Exception as e:
        result['errors'].append(str(e))
        # Clean up partial outputs on failure
        if plot_output.exists():
            try:
                plot_output.unlink()
            except Exception:
                pass  # Ignore cleanup errors
        raise
    
    finally:
        result['end_time'] = datetime.now()
        result['duration_seconds'] = (result['end_time'] - start_time).total_seconds()
    
    return result

# Usage example
if __name__ == "__main__":
    def progress_printer(message: str):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    try:
        result = run_indicator_analysis(
            data_path=Path("data/EURUSD_1h.ohlcv"),
            script_path=Path("scripts/bollinger_bands.py"),
            syminfo_path=Path("data/EURUSD.syminfo"),
            output_dir=Path("output/indicator_analysis"),
            progress_callback=progress_printer
        )
        
        print(f"\n✅ Analysis completed in {result['duration_seconds']:.2f} seconds")
        print(f"   Processed {result['bars_processed']} bars")
        print(f"   Generated {len(result['output_files'])} output files")
        
        for file_info in result['output_files']:
            size_kb = file_info['size_bytes'] / 1024
            print(f"   - {file_info['type']}: {file_info['path']} ({size_kb:.1f} KB)")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
```

### Example 2: Strategy Backtest with Comprehensive Output

```python
def run_strategy_backtest(
    data_path: Path,
    strategy_path: Path,
    syminfo_path: Path,
    output_dir: Path,
    initial_capital: float = 10000.0
) -> Dict[str, Any]:
    """
    Run a comprehensive strategy backtest with all outputs.
    
    :param data_path: Path to OHLCV data file
    :param strategy_path: Path to Pine strategy script
    :param syminfo_path: Path to symbol info file
    :param output_dir: Directory for output files
    :param initial_capital: Starting capital for backtest
    :return: Dictionary with backtest results and performance metrics
    """
    # Validate inputs
    if not data_path.exists():
        raise FileNotFoundError(f"OHLCV data file not found: {data_path}")
    if not strategy_path.exists():
        raise FileNotFoundError(f"Strategy file not found: {strategy_path}")
    if not syminfo_path.exists():
        raise FileNotFoundError(f"Symbol info file not found: {syminfo_path}")
    
    if initial_capital <= 0:
        raise ValueError(f"Initial capital must be positive, got: {initial_capital}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output paths for strategy
    plot_output = output_dir / "plot_data.csv"
    trades_output = output_dir / "trades.csv"
    stats_output = output_dir / "strategy_stats.csv"
    
    start_time = datetime.now()
    result = {
        'success': False,
        'start_time': start_time,
        'end_time': None,
        'duration_seconds': None,
        'initial_capital': initial_capital,
        'bars_processed': 0,
        'trades_count': 0,
        'output_files': [],
        'performance_metrics': {},
        'errors': []
    }
    
    try:
        # Load symbol information
        syminfo = SymInfo.load_from_file(syminfo_path)
        
        with OHLCVReader.open(data_path) as reader:
            data_size = reader.get_size()
            if data_size == 0:
                raise ValueError(f"No data found in {data_path}")
            
            result['bars_processed'] = data_size
            
            print(f"Running strategy backtest on {data_size} bars...")
            print(f"Period: {reader.start_datetime} to {reader.end_datetime}")
            print(f"Initial capital: ${initial_capital:,.2f}")
            
            # Create strategy runner with all outputs
            runner = ScriptRunner(
                script_path=str(strategy_path),
                ohlcv_iter=reader,
                syminfo=syminfo,
                plot_path=str(plot_output),
                trade_path=str(trades_output),  # Note: trade_path, not equity_path
                strat_path=str(stats_output),
                last_bar_index=data_size - 1
            )
            
            # Execute strategy
            success = runner.run()
            
            if not success:
                raise RuntimeError("Strategy execution failed")
            
            result['success'] = True
            
        # Analyze generated outputs
        output_files = []
        
        # Plot data
        if plot_output.exists():
            output_files.append({
                'type': 'plot_data',
                'path': str(plot_output),
                'size_bytes': plot_output.stat().st_size
            })
        
        # Trades data
        if trades_output.exists():
            import pandas as pd
            try:
                trades_df = pd.read_csv(trades_output)
                result['trades_count'] = len(trades_df)
                
                # Calculate basic performance metrics
                if not trades_df.empty and 'pnl' in trades_df.columns:
                    total_pnl = trades_df['pnl'].sum()
                    winning_trades = len(trades_df[trades_df['pnl'] > 0])
                    losing_trades = len(trades_df[trades_df['pnl'] < 0])
                    
                    result['performance_metrics'] = {
                        'total_pnl': total_pnl,
                        'final_capital': initial_capital + total_pnl,
                        'total_return_pct': (total_pnl / initial_capital) * 100,
                        'winning_trades': winning_trades,
                        'losing_trades': losing_trades,
                        'win_rate_pct': (winning_trades / len(trades_df)) * 100 if len(trades_df) > 0 else 0
                    }
                
                output_files.append({
                    'type': 'trades',
                    'path': str(trades_output),
                    'size_bytes': trades_output.stat().st_size,
                    'records_count': len(trades_df)
                })
                
            except Exception as e:
                result['errors'].append(f"Failed to analyze trades data: {e}")
        
        # Strategy statistics
        if stats_output.exists():
            output_files.append({
                'type': 'strategy_stats',
                'path': str(stats_output),
                'size_bytes': stats_output.stat().st_size
            })
        
        result['output_files'] = output_files
        
        # Print summary
        print(f"\n✅ Strategy backtest completed!")
        print(f"   Trades executed: {result['trades_count']}")
        
        if result['performance_metrics']:
            metrics = result['performance_metrics']
            print(f"   Total P&L: ${metrics['total_pnl']:,.2f}")
            print(f"   Final capital: ${metrics['final_capital']:,.2f}")
            print(f"   Total return: {metrics['total_return_pct']:.2f}%")
            print(f"   Win rate: {metrics['win_rate_pct']:.1f}%")
        
        return result
        
    except Exception as e:
        result['errors'].append(str(e))
        # Clean up on failure
        for output_file in [plot_output, trades_output, stats_output]:
            if output_file.exists():
                try:
                    output_file.unlink()
                except Exception:
                    pass
        raise
    
    finally:
        result['end_time'] = datetime.now()
        result['duration_seconds'] = (result['end_time'] - start_time).total_seconds()

# Usage example
if __name__ == "__main__":
    try:
        result = run_strategy_backtest(
            data_path=Path("data/BTCUSD_1D.ohlcv"),
            strategy_path=Path("scripts/ma_crossover_strategy.py"),
            syminfo_path=Path("data/BTCUSD.syminfo"),
            output_dir=Path("output/strategy_backtest"),
            initial_capital=50000.0
        )
        
        print(f"\nBacktest completed in {result['duration_seconds']:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
```

## Parallel Processing

### Example 3: Multi-Symbol Analysis

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, NamedTuple
import multiprocessing as mp

class AnalysisTask(NamedTuple):
    symbol: str
    data_path: Path
    script_path: Path
    syminfo_path: Path
    output_dir: Path

def run_single_symbol_analysis(task: AnalysisTask) -> Dict[str, Any]:
    """
    Run analysis for a single symbol (designed for parallel execution).
    """
    try:
        # Important: Set update_syminfo_every_run=True for parallel execution
        syminfo = SymInfo.load_from_file(task.syminfo_path)
        
        # Create symbol-specific output directory
        symbol_output_dir = task.output_dir / task.symbol
        symbol_output_dir.mkdir(parents=True, exist_ok=True)
        
        with OHLCVReader.open(task.data_path) as reader:
            runner = ScriptRunner(
                script_path=str(task.script_path),
                ohlcv_iter=reader,
                syminfo=syminfo,
                plot_path=str(symbol_output_dir / "plot_data.csv"),
                update_syminfo_every_run=True  # Critical for parallel execution
            )
            
            start_time = datetime.now()
            success = runner.run()
            end_time = datetime.now()
            
            return {
                "symbol": task.symbol,
                "status": "success" if success else "failed",
                "bars_processed": reader.get_size(),
                "duration_seconds": (end_time - start_time).total_seconds(),
                "output_dir": str(symbol_output_dir),
                "data_period": f"{reader.start_datetime} to {reader.end_datetime}"
            }
            
    except Exception as e:
        return {
            "symbol": task.symbol,
            "status": "error",
            "error": str(e),
            "bars_processed": 0,
            "duration_seconds": 0
        }

def run_parallel_analysis(
    tasks: List[AnalysisTask], 
    max_workers: Optional[int] = None,
    progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Run analysis on multiple symbols in parallel.
    
    :param tasks: List of analysis tasks
    :param max_workers: Maximum number of parallel workers (default: CPU count)
    :param progress_callback: Optional callback for progress updates
    :return: List of results for each symbol
    """
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(tasks))
    
    print(f"Starting parallel analysis of {len(tasks)} symbols using {max_workers} workers...")
    
    results = []
    completed_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(run_single_symbol_analysis, task): task for task in tasks}
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            completed_count += 1
            
            try:
                result = future.result()
                results.append(result)
                
                if progress_callback:
                    progress_callback(completed_count, len(tasks), result)
                
                # Print progress
                status_icon = "✅" if result["status"] == "success" else "❌"
                print(f"{status_icon} [{completed_count}/{len(tasks)}] {result['symbol']}: {result['status']}")
                
                if result["status"] == "success":
                    print(f"    Processed {result['bars_processed']} bars in {result['duration_seconds']:.2f}s")
                elif result["status"] == "error":
                    print(f"    Error: {result['error']}")
                
            except Exception as e:
                error_result = {
                    "symbol": task.symbol,
                    "status": "error",
                    "error": f"Future execution failed: {e}",
                    "bars_processed": 0,
                    "duration_seconds": 0
                }
                results.append(error_result)
                print(f"❌ [{completed_count}/{len(tasks)}] {task.symbol}: Future failed - {e}")
    
    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    total_bars = sum(r["bars_processed"] for r in results)
    total_time = sum(r["duration_seconds"] for r in results)
    
    print(f"\n📊 Parallel Analysis Summary:")
    print(f"   Successful: {successful}/{len(tasks)} symbols")
    print(f"   Failed: {failed}/{len(tasks)} symbols")
    print(f"   Total bars processed: {total_bars:,}")
    print(f"   Total processing time: {total_time:.2f} seconds")
    print(f"   Average time per symbol: {total_time/len(tasks):.2f} seconds")
    
    return results

# Usage example
if __name__ == "__main__":
    # Define analysis tasks
    crypto_symbols = [
        AnalysisTask(
            symbol="BTCUSD",
            data_path=Path("data/BTCUSD_1D.ohlcv"),
            script_path=Path("scripts/rsi_divergence.py"),
            syminfo_path=Path("data/BTCUSD.syminfo"),
            output_dir=Path("output/parallel_analysis")
        ),
        AnalysisTask(
            symbol="ETHUSD",
            data_path=Path("data/ETHUSD_1D.ohlcv"),
            script_path=Path("scripts/rsi_divergence.py"),
            syminfo_path=Path("data/ETHUSD.syminfo"),
            output_dir=Path("output/parallel_analysis")
        ),
        AnalysisTask(
            symbol="ADAUSD",
            data_path=Path("data/ADAUSD_1D.ohlcv"),
            script_path=Path("scripts/rsi_divergence.py"),
            syminfo_path=Path("data/ADAUSD.syminfo"),
            output_dir=Path("output/parallel_analysis")
        )
    ]
    
    def progress_update(completed: int, total: int, result: Dict[str, Any]):
        progress_pct = (completed / total) * 100
        print(f"    Progress: {progress_pct:.1f}% complete")
    
    # Run parallel analysis
    results = run_parallel_analysis(
        tasks=crypto_symbols,
        max_workers=3,
        progress_callback=progress_update
    )
    
    # Save results summary
    import json
    summary_file = Path("output/parallel_analysis/summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults summary saved to: {summary_file}")
```

### Example 4: Multi-Timeframe Analysis

```python
def run_multi_timeframe_analysis(
    symbol: str,
    base_data_dir: Path,
    script_path: Path,
    timeframes: List[str],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run the same analysis across multiple timeframes for a single symbol.
    
    :param symbol: Trading symbol (e.g., "BTCUSD")
    :param base_data_dir: Directory containing OHLCV files
    :param script_path: Path to analysis script
    :param timeframes: List of timeframes to analyze
    :param output_dir: Output directory
    :return: Analysis results for all timeframes
    """
    print(f"Running multi-timeframe analysis for {symbol}...")
    
    results = {
        'symbol': symbol,
        'timeframes': {},
        'summary': {
            'successful_timeframes': 0,
            'failed_timeframes': 0,
            'total_bars_processed': 0
        }
    }
    
    for timeframe in timeframes:
        print(f"\nAnalyzing {symbol} on {timeframe} timeframe...")
        
        try:
            # Construct file paths
            data_file = base_data_dir / f"{symbol}_{timeframe}.ohlcv"
            syminfo_file = base_data_dir / f"{symbol}_{timeframe}.syminfo"
            
            if not data_file.exists():
                raise FileNotFoundError(f"Data file not found: {data_file}")
            
            # Create syminfo if file doesn't exist
            if not syminfo_file.exists():
                syminfo = SymInfo(
                    symbol=symbol,
                    exchange="BINANCE",  # Default exchange
                    timeframe=timeframe
                )
            else:
                syminfo = SymInfo.load_from_file(syminfo_file)
            
            # Create timeframe-specific output directory
            tf_output_dir = output_dir / symbol / timeframe
            tf_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run analysis
            with OHLCVReader.open(data_file) as reader:
                runner = ScriptRunner(
                    script_path=str(script_path),
                    ohlcv_iter=reader,
                    syminfo=syminfo,
                    plot_path=str(tf_output_dir / "analysis.csv")
                )
                
                start_time = datetime.now()
                success = runner.run()
                end_time = datetime.now()
                
                bars_processed = reader.get_size()
                duration = (end_time - start_time).total_seconds()
                
                results['timeframes'][timeframe] = {
                    'success': success,
                    'bars_processed': bars_processed,
                    'duration_seconds': duration,
                    'data_period': f"{reader.start_datetime} to {reader.end_datetime}",
                    'output_dir': str(tf_output_dir)
                }
                
                if success:
                    results['summary']['successful_timeframes'] += 1
                    results['summary']['total_bars_processed'] += bars_processed
                    print(f"  ✅ {timeframe}: {bars_processed} bars processed in {duration:.2f}s")
                else:
                    results['summary']['failed_timeframes'] += 1
                    print(f"  ❌ {timeframe}: Analysis failed")
                
        except Exception as e:
            results['timeframes'][timeframe] = {
                'success': False,
                'error': str(e),
                'bars_processed': 0,
                'duration_seconds': 0
            }
            results['summary']['failed_timeframes'] += 1
            print(f"  ❌ {timeframe}: Error - {e}")
    
    # Print summary
    summary = results['summary']
    print(f"\n📊 Multi-timeframe Analysis Summary for {symbol}:")
    print(f"   Successful: {summary['successful_timeframes']}/{len(timeframes)} timeframes")
    print(f"   Total bars processed: {summary['total_bars_processed']:,}")
    
    return results

# Usage example
if __name__ == "__main__":
    timeframes_to_analyze = ["1m", "5m", "15m", "1h", "4h", "1D"]
    
    results = run_multi_timeframe_analysis(
        symbol="BTCUSD",
        base_data_dir=Path("data"),
        script_path=Path("scripts/momentum_oscillator.py"),
        timeframes=timeframes_to_analyze,
        output_dir=Path("output/multi_timeframe")
    )
    
    # Save detailed results
    import json
    results_file = Path("output/multi_timeframe/BTCUSD/results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {results_file}")
```

## Memory and Resource Management

### Example 5: Large Dataset Processing

```python
import psutil
import os
from typing import Generator

def monitor_memory_usage(func):
    """Decorator to monitor memory usage during execution."""
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_diff = final_memory - initial_memory
            
            print(f"Final memory usage: {final_memory:.1f} MB")
            print(f"Memory change: {memory_diff:+.1f} MB")
    
    return wrapper

@monitor_memory_usage
def process_large_dataset(
    data_path: Path,
    script_path: Path,
    chunk_size: int = 10000,
    output_dir: Path = None
) -> Dict[str, Any]:
    """
    Process a large dataset in chunks to manage memory usage.
    
    :param data_path: Path to large OHLCV file
    :param script_path: Path to analysis script
    :param chunk_size: Number of bars to process in each chunk
    :param output_dir: Output directory for results
    :return: Processing results
    """
    if output_dir is None:
        output_dir = Path("output/large_dataset")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create symbol info
    syminfo = SymInfo(
        symbol="LARGE_DATASET",
        exchange="UNKNOWN",
        timeframe="1D"
    )
    
    results = {
        'total_bars': 0,
        'chunks_processed': 0,
        'chunks_failed': 0,
        'processing_times': [],
        'memory_snapshots': []
    }
    
    try:
        with OHLCVReader.open(data_path) as reader:
            total_bars = reader.get_size()
            results['total_bars'] = total_bars
            
            print(f"Processing {total_bars:,} bars in chunks of {chunk_size:,}...")
            
            # Process in chunks
            for chunk_start in range(0, total_bars, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_bars)
                chunk_num = (chunk_start // chunk_size) + 1
                total_chunks = (total_bars + chunk_size - 1) // chunk_size
                
                print(f"\nProcessing chunk {chunk_num}/{total_chunks} (bars {chunk_start:,}-{chunk_end:,})...")
                
                try:
                    # Create chunk-specific output
                    chunk_output = output_dir / f"chunk_{chunk_num:04d}.csv"
                    
                    # Create iterator for this chunk
                    chunk_iter = reader.read_from(chunk_start)
                    
                    # Limit the iterator to chunk size
                    def limited_iter(iterator, limit):
                        count = 0
                        for item in iterator:
                            if count >= limit:
                                break
                            yield item
                            count += 1
                    
                    chunk_data = limited_iter(chunk_iter, chunk_end - chunk_start)
                    
                    # Process chunk
                    start_time = datetime.now()
                    
                    runner = ScriptRunner(
                        script_path=str(script_path),
                        ohlcv_iter=chunk_data,
                        syminfo=syminfo,
                        plot_path=str(chunk_output),
                        update_syminfo_every_run=True
                    )
                    
                    success = runner.run()
                    
                    end_time = datetime.now()
                    chunk_duration = (end_time - start_time).total_seconds()
                    
                    if success:
                        results['chunks_processed'] += 1
                        results['processing_times'].append(chunk_duration)
                        
                        # Memory snapshot
                        process = psutil.Process(os.getpid())
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        results['memory_snapshots'].append(memory_mb)
                        
                        print(f"  ✅ Chunk {chunk_num} completed in {chunk_duration:.2f}s (Memory: {memory_mb:.1f} MB)")
                    else:
                        results['chunks_failed'] += 1
                        print(f"  ❌ Chunk {chunk_num} failed")
                        
                        # Clean up failed chunk output
                        if chunk_output.exists():
                            chunk_output.unlink()
                    
                except Exception as e:
                    results['chunks_failed'] += 1
                    print(f"  ❌ Chunk {chunk_num} error: {e}")
            
            # Combine chunk results if needed
            if results['chunks_processed'] > 1:
                print(f"\nCombining {results['chunks_processed']} chunk results...")
                combine_chunk_results(output_dir, results['chunks_processed'])
            
    except Exception as e:
        print(f"❌ Large dataset processing failed: {e}")
        raise
    
    # Summary
    avg_time = sum(results['processing_times']) / len(results['processing_times']) if results['processing_times'] else 0
    max_memory = max(results['memory_snapshots']) if results['memory_snapshots'] else 0
    
    print(f"\n📊 Large Dataset Processing Summary:")
    print(f"   Total bars: {results['total_bars']:,}")
    print(f"   Successful chunks: {results['chunks_processed']}")
    print(f"   Failed chunks: {results['chunks_failed']}")
    print(f"   Average chunk time: {avg_time:.2f} seconds")
    print(f"   Peak memory usage: {max_memory:.1f} MB")
    
    return results

def combine_chunk_results(output_dir: Path, num_chunks: int) -> None:
    """Combine individual chunk results into a single file."""
    import pandas as pd
    
    combined_file = output_dir / "combined_results.csv"
    
    try:
        # Read and combine all chunk files
        chunk_dfs = []
        for chunk_num in range(1, num_chunks + 1):
            chunk_file = output_dir / f"chunk_{chunk_num:04d}.csv"
            if chunk_file.exists():
                df = pd.read_csv(chunk_file)
                chunk_dfs.append(df)
        
        if chunk_dfs:
            combined_df = pd.concat(chunk_dfs, ignore_index=True)
            combined_df.to_csv(combined_file, index=False)
            
            print(f"  ✅ Combined results saved to: {combined_file}")
            print(f"     Total records: {len(combined_df):,}")
            
            # Clean up individual chunk files
            for chunk_num in range(1, num_chunks + 1):
                chunk_file = output_dir / f"chunk_{chunk_num:04d}.csv"
                if chunk_file.exists():
                    chunk_file.unlink()
        
    except Exception as e:
        print(f"  ⚠️  Failed to combine chunk results: {e}")

# Usage example
if __name__ == "__main__":
    # Process a large dataset
    results = process_large_dataset(
        data_path=Path("data/BTCUSD_1m_large.ohlcv"),  # Large 1-minute data file
        script_path=Path("scripts/volume_profile.py"),
        chunk_size=5000,  # Process 5000 bars at a time
        output_dir=Path("output/large_dataset_analysis")
    )
```

## Next Steps

These advanced examples demonstrate production-ready patterns for using PyneCore programmatically. For additional guidance, see:

- [Best Practices](best-practices.md) - Performance optimization and common pitfalls
- [Core Components](core-components.md) - Detailed API reference
- [Basic Examples](basic-examples.md) - Simple usage patterns