<!--
---
weight: 10004
title: "Best Practices"
description: "Performance optimization, error handling, and common pitfalls for PyneCore programmatic usage"
icon: "tips_and_updates"
date: "2025-08-06"
lastmod: "2025-08-06"
draft: false
toc: true
categories: ["Advanced", "API", "Best Practices"]
tags: ["performance", "optimization", "error-handling", "pitfalls"]
---
-->

# Best Practices

This guide covers performance optimization, error handling strategies, and common pitfalls when using PyneCore programmatically.

## Performance Optimization

### Memory Management

#### Use Context Managers
Always use context managers to ensure proper resource cleanup:

```python
# ✅ Good: Automatic resource cleanup
with OHLCVReader.open(data_path) as reader:
    runner = ScriptRunner(script_path, reader, syminfo)
    runner.run()
# Reader is automatically closed

# ❌ Bad: Manual resource management
reader = OHLCVReader.open(data_path)
try:
    runner = ScriptRunner(script_path, reader, syminfo)
    runner.run()
finally:
    reader.close()  # Easy to forget
```

#### Monitor Memory Usage
For large datasets, monitor memory consumption:

```python
import gc

def memory_efficient_analysis(data_paths: list[Path], script_path: Path):
    """Process multiple files with memory monitoring."""
    for data_path in data_paths:
        with OHLCVReader.open(data_path) as reader:
            runner = ScriptRunner(script_path, reader, syminfo)
            runner.run()
        
        # Force garbage collection after each file
        gc.collect()
```

#### Process Large Datasets in Chunks
For very large datasets, process data in chunks:

```python
def process_large_dataset_efficiently(
    data_path: Path,
    script_path: Path,
    chunk_size: int = 10000
):
    """Process large datasets in memory-efficient chunks."""
    with OHLCVReader.open(data_path) as reader:
        total_bars = reader.get_size()
        
        for start_idx in range(0, total_bars, chunk_size):
            end_idx = min(start_idx + chunk_size, total_bars)
            chunk_iter = reader.read_from(start_idx)
            
            runner = ScriptRunner(
                script_path=script_path,
                ohlcv_iter=chunk_iter,
                syminfo=syminfo
            )
            runner.run()
```

### Parallel Processing

#### Use ProcessPoolExecutor for CPU-Bound Tasks
```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def parallel_symbol_analysis(symbol_tasks: list, max_workers: int = None):
    """Process multiple symbols in parallel."""
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(symbol_tasks))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_symbol, task) for task in symbol_tasks]
        return [future.result() for future in futures]

def process_single_symbol(task):
    """Process a single symbol (designed for parallel execution)."""
    with OHLCVReader.open(task.data_path) as reader:
        runner = ScriptRunner(
            script_path=task.script_path,
            ohlcv_iter=reader,
            syminfo=task.syminfo,
            update_syminfo_every_run=True  # Required for parallel execution
        )
        return runner.run()
```



### Data Access Optimization

#### Minimize File I/O
```python
# ✅ Good: Single file open, multiple operations
with OHLCVReader.open(data_path) as reader:
    for script_path in script_list:
        runner = ScriptRunner(script_path, reader, syminfo)
        runner.run()

# ❌ Bad: Multiple file opens
for script_path in script_list:
    with OHLCVReader.open(data_path) as reader:  # Reopening file each time
        runner = ScriptRunner(script_path, reader, syminfo)
        runner.run()
```

#### Cache Symbol Information
```python
class SymInfoCache:
    def __init__(self):
        self._cache = {}
    
    def get_syminfo(self, syminfo_path: Path) -> SymInfo:
        cache_key = str(syminfo_path)
        if cache_key not in self._cache:
            self._cache[cache_key] = SymInfo.load_from_file(syminfo_path)
        return self._cache[cache_key]
```

## Error Handling Strategies

### Comprehensive Error Classification

```python
class PyneCoreProgrammaticError(Exception):
    """Base exception for programmatic PyneCore errors."""
    pass

class DataError(PyneCoreProgrammaticError):
    """Data-related errors (missing files, invalid data)."""
    pass

class ScriptError(PyneCoreProgrammaticError):
    """Script-related errors (compilation, execution)."""
    pass

class ConfigurationError(PyneCoreProgrammaticError):
    """Configuration-related errors (invalid parameters)."""
    pass

def safe_script_execution(
    data_path: Path,
    script_path: Path,
    syminfo_path: Path,
    output_dir: Path
) -> dict:
    """Execute script with comprehensive error handling."""
    result = {
        'success': False,
        'error_type': None,
        'error_message': None,
        'output_files': []
    }
    
    try:
        # Validate inputs
        validate_inputs(data_path, script_path, syminfo_path)
        
        # Load and validate data
        with OHLCVReader.open(data_path) as reader:
            if reader.get_size() == 0:
                raise DataError(f"No data found in {data_path}")
            
            # Load symbol info
            try:
                syminfo = SymInfo.load_from_file(syminfo_path)
            except Exception as e:
                raise DataError(f"Failed to load symbol info: {e}")
            
            # Execute script
            try:
                runner = ScriptRunner(
                    script_path=str(script_path),
                    ohlcv_iter=reader,
                    syminfo=syminfo,
                    plot_path=str(output_dir / "plot.csv")
                )
                
                success = runner.run()
                if not success:
                    raise ScriptError("Script execution returned failure status")
                
                result['success'] = True
                
            except ImportError as e:
                if "@pyne" in str(e):
                    raise ScriptError(f"Script missing @pyne decorator: {script_path}")
                elif "main" in str(e):
                    raise ScriptError(f"Script missing main function: {script_path}")
                else:
                    raise ScriptError(f"Script import failed: {e}")
    
    except DataError as e:
        result['error_type'] = 'data'
        result['error_message'] = str(e)
    except ScriptError as e:
        result['error_type'] = 'script'
        result['error_message'] = str(e)
    except ConfigurationError as e:
        result['error_type'] = 'configuration'
        result['error_message'] = str(e)
    except Exception as e:
        result['error_type'] = 'unknown'
        result['error_message'] = str(e)
    
    return result

def validate_inputs(data_path: Path, script_path: Path, syminfo_path: Path):
    """Validate all input files exist and are accessible."""
    missing_files = []
    
    if not data_path.exists():
        missing_files.append(f"Data file: {data_path}")
    if not script_path.exists():
        missing_files.append(f"Script file: {script_path}")
    if not syminfo_path.exists():
        missing_files.append(f"Symbol info file: {syminfo_path}")
    
    if missing_files:
        raise DataError(f"Missing required files: {', '.join(missing_files)}")
    
    # Check file permissions
    for file_path in [data_path, script_path, syminfo_path]:
        if not os.access(file_path, os.R_OK):
            raise DataError(f"Cannot read file: {file_path}")
```

### Retry Logic for Transient Failures

```python
import time
import random
from functools import wraps

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry function calls on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        # Final attempt failed
                        break
                    
                    # Calculate delay with jitter
                    retry_delay = delay * (backoff ** attempt)
                    jitter = random.uniform(0.1, 0.3) * retry_delay
                    total_delay = retry_delay + jitter
                    
                    print(f"Attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {total_delay:.2f} seconds...")
                    time.sleep(total_delay)
            
            # All retries exhausted
            raise last_exception
        
        return wrapper
    return decorator

@retry_on_failure(max_retries=2, delay=1.0)
def robust_script_execution(data_path: Path, script_path: Path, syminfo: SymInfo):
    """Execute script with automatic retry on failure."""
    with OHLCVReader.open(data_path) as reader:
        runner = ScriptRunner(
            script_path=str(script_path),
            ohlcv_iter=reader,
            syminfo=syminfo
        )
        
        success = runner.run()
        if not success:
            raise RuntimeError("Script execution failed")
        
        return success
```

### Graceful Degradation

```python
def analyze_with_fallbacks(
    primary_script: Path,
    fallback_scripts: list[Path],
    data_path: Path,
    syminfo: SymInfo
) -> dict:
    """Try primary script, fall back to alternatives on failure."""
    scripts_to_try = [primary_script] + fallback_scripts
    
    for i, script_path in enumerate(scripts_to_try):
        try:
            print(f"Trying script {i+1}/{len(scripts_to_try)}: {script_path.name}")
            
            with OHLCVReader.open(data_path) as reader:
                runner = ScriptRunner(
                    script_path=str(script_path),
                    ohlcv_iter=reader,
                    syminfo=syminfo
                )
                
                success = runner.run()
                
                if success:
                    return {
                        'success': True,
                        'script_used': str(script_path),
                        'attempt_number': i + 1
                    }
                else:
                    print(f"  Script {script_path.name} failed, trying next...")
        
        except Exception as e:
            print(f"  Script {script_path.name} error: {e}")
            if i == len(scripts_to_try) - 1:  # Last script
                raise
    
    return {
        'success': False,
        'error': 'All scripts failed'
    }
```

## Common Pitfalls and Solutions

### 1. Parameter Name Confusion

**Problem**: Using incorrect parameter names in ScriptRunner.

```python
# ❌ Common mistake: Using 'equity_path' instead of 'trade_path'
runner = ScriptRunner(
    script_path="strategy.py",
    ohlcv_iter=reader,
    syminfo=syminfo,
    equity_path="trades.csv"  # This parameter doesn't exist!
)

# ✅ Correct: Use 'trade_path'
runner = ScriptRunner(
    script_path="strategy.py",
    ohlcv_iter=reader,
    syminfo=syminfo,
    trade_path="trades.csv"  # Correct parameter name
)
```

**Solution**: Always refer to the [Core Components](core-components.md) documentation for correct parameter names.

### 2. Script Format Requirements

**Problem**: Scripts missing required decorators or functions.

```python
# ❌ Bad: Script without @pyne decorator
def main():
    # Script logic here
    pass

# ✅ Good: Proper script format
from pynecore import pyne

@pyne
def main():
    # Script logic here
    pass
```

**Solution**: Ensure all scripts have the `@pyne` decorator and a `main()` function.

### 3. Resource Management Issues

**Problem**: Not properly closing file handles.

```python
# ❌ Bad: Manual resource management
reader = OHLCVReader.open(data_path)
runner = ScriptRunner(script_path, reader, syminfo)
runner.run()
# Forgot to close reader!

# ✅ Good: Use context managers
with OHLCVReader.open(data_path) as reader:
    runner = ScriptRunner(script_path, reader, syminfo)
    runner.run()
# Reader automatically closed
```

**Solution**: Always use context managers (`with` statements) for file operations.

### 4. Parallel Execution Issues

**Problem**: Not setting `update_syminfo_every_run=True` for parallel execution.

```python
# ❌ Bad: Default setting for parallel execution
def process_symbol_parallel(task):
    with OHLCVReader.open(task.data_path) as reader:
        runner = ScriptRunner(
            script_path=task.script_path,
            ohlcv_iter=reader,
            syminfo=task.syminfo
            # Missing: update_syminfo_every_run=True
        )
        return runner.run()

# ✅ Good: Proper parallel execution setup
def process_symbol_parallel(task):
    with OHLCVReader.open(task.data_path) as reader:
        runner = ScriptRunner(
            script_path=task.script_path,
            ohlcv_iter=reader,
            syminfo=task.syminfo,
            update_syminfo_every_run=True  # Required for parallel execution
        )
        return runner.run()
```

**Solution**: Always set `update_syminfo_every_run=True` when running scripts in parallel.

### 5. Data Validation Oversights

**Problem**: Not validating data before processing.

```python
# ❌ Bad: No data validation
with OHLCVReader.open(data_path) as reader:
    runner = ScriptRunner(script_path, reader, syminfo)
    runner.run()  # May fail if data is empty or corrupted

# ✅ Good: Validate data first
with OHLCVReader.open(data_path) as reader:
    # Validate data
    if reader.get_size() == 0:
        raise ValueError(f"No data found in {data_path}")
    
    if reader.get_size() < 10:
        raise ValueError(f"Insufficient data: only {reader.get_size()} bars")
    
    # Check date range
    if reader.start_datetime == reader.end_datetime:
        raise ValueError("Data contains only one timestamp")
    
    runner = ScriptRunner(script_path, reader, syminfo)
    runner.run()
```

**Solution**: Always validate data size, date ranges, and data integrity before processing.

### 6. Error Handling Anti-patterns

**Problem**: Catching all exceptions without proper handling.

```python
# ❌ Bad: Silent failure
try:
    runner = ScriptRunner(script_path, reader, syminfo)
    runner.run()
except:
    pass  # Silent failure - very bad!

# ❌ Bad: Generic error handling
try:
    runner = ScriptRunner(script_path, reader, syminfo)
    runner.run()
except Exception as e:
    print(f"Error: {e}")  # Not helpful for debugging

# ✅ Good: Specific error handling
try:
    runner = ScriptRunner(script_path, reader, syminfo)
    success = runner.run()
    
    if not success:
        raise RuntimeError("Script execution failed")
        
except ImportError as e:
    if "@pyne" in str(e):
        raise RuntimeError(f"Script missing @pyne decorator: {script_path}")
    else:
        raise RuntimeError(f"Script import failed: {e}")
except FileNotFoundError as e:
    raise RuntimeError(f"Required file not found: {e}")
except Exception as e:
    raise RuntimeError(f"Unexpected error during script execution: {e}")
```

**Solution**: Use specific exception handling with meaningful error messages.

### 7. Path and File Issues

**Problem**: Using relative paths or not handling path separators correctly.

```python
# ❌ Bad: Relative paths and string concatenation
data_file = "data/" + symbol + "_1D.ohlcv"  # Platform-dependent
script_file = "../scripts/indicator.py"     # Relative path

# ✅ Good: Use pathlib for cross-platform compatibility
from pathlib import Path

data_dir = Path("data")
script_dir = Path("scripts")

data_file = data_dir / f"{symbol}_1D.ohlcv"
script_file = script_dir / "indicator.py"

# Convert to absolute paths
data_file = data_file.resolve()
script_file = script_file.resolve()
```

**Solution**: Use `pathlib.Path` for all file operations and convert to absolute paths when needed.

## Performance Monitoring

### Benchmarking Script Performance

```python
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class PerformanceMetrics:
    script_name: str
    bars_processed: int
    execution_time: float
    memory_peak: float
    success: bool
    error_message: Optional[str] = None
    
    @property
    def bars_per_second(self) -> float:
        return self.bars_processed / self.execution_time if self.execution_time > 0 else 0
    
    @property
    def memory_per_bar(self) -> float:
        return self.memory_peak / self.bars_processed if self.bars_processed > 0 else 0

def benchmark_script(
    script_path: Path,
    data_path: Path,
    syminfo: SymInfo
) -> PerformanceMetrics:
    """Benchmark script performance."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    start_time = time.time()
    bars_processed = 0
    success = False
    error_message = None
    
    try:
        with OHLCVReader.open(data_path) as reader:
            bars_processed = reader.get_size()
            
            runner = ScriptRunner(
                script_path=str(script_path),
                ohlcv_iter=reader,
                syminfo=syminfo
            )
            
            success = runner.run()
            
    except Exception as e:
        error_message = str(e)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_peak = final_memory - initial_memory
    
    return PerformanceMetrics(
        script_name=script_path.name,
        bars_processed=bars_processed,
        execution_time=execution_time,
        memory_peak=memory_peak,
        success=success,
        error_message=error_message
    )

def compare_script_performance(scripts: list[Path], data_path: Path, syminfo: SymInfo):
    """Compare performance of multiple scripts on the same data."""
    results = []
    
    for script_path in scripts:
        print(f"Benchmarking {script_path.name}...")
        metrics = benchmark_script(script_path, data_path, syminfo)
        results.append(metrics)
        
        if metrics.success:
            print(f"  ✅ {metrics.bars_per_second:.0f} bars/sec, {metrics.memory_peak:.1f} MB")
        else:
            print(f"  ❌ Failed: {metrics.error_message}")
    
    # Sort by performance
    successful_results = [r for r in results if r.success]
    successful_results.sort(key=lambda x: x.bars_per_second, reverse=True)
    
    print(f"\n📊 Performance Ranking:")
    for i, metrics in enumerate(successful_results, 1):
        print(f"  {i}. {metrics.script_name}: {metrics.bars_per_second:.0f} bars/sec")
    
    return results
```

## Next Steps

By following these best practices, you'll build robust, efficient, and maintainable PyneCore applications. For more information:

- [Core Components](core-components.md) - Detailed API reference
- [Basic Examples](basic-examples.md) - Simple usage patterns
- [Advanced Examples](advanced-examples.md) - Complex scenarios and patterns
- [CLI Documentation](../../cli/README.md) - Command-line interface usage