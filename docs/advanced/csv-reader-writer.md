<!--
---
weight: 1004
title: "Fast CSV Reader/Writer"
description: "High-performance CSV processing with multithreaded writing and memory mapping"
icon: "table_chart"
date: "2025-03-31"
lastmod: "2025-03-31"
draft: false
toc: true
categories: ["Advanced", "Data Handling"]
tags: ["csv", "data", "performance", "io", "multithreading", "memory-mapping"]
---
-->

# Fast CSV Reader/Writer

The PyneCore CSV Reader/Writer system provides a high-performance solution for handling CSV data files, with a particular focus on OHLCV (Open, High, Low, Close, Volume) market data. While not as specialized as the binary OHLCV format, this system offers optimized CSV processing with features like multithreaded writing and memory mapping for reading.

## Overview

CSV (Comma-Separated Values) is a universal format for tabular data exchange. While simple in concept, high-performance CSV processing presents several challenges:

1. I/O operations can be a significant bottleneck, especially when writing large datasets
2. String parsing and formatting can be computationally expensive
3. CSV files often require sequential processing, limiting random access capabilities

The PyneCore CSV system addresses these challenges through:

- **Multithreaded Writing**: Background thread for non-blocking I/O operations
- **Buffer Management**: Efficient buffer handling to minimize system calls
- **Memory Mapping**: Fast file access for reading operations
- **Format Auto-Detection**: Automatic detection of CSV dialect and headers
- **Flexible Data Types**: Support for various data formats including OHLCV structures

## CSVWriter: Multithreaded Performance

The `CSVWriter` class implements a high-performance CSV writer that leverages a background thread for I/O operations. This approach allows the main thread to continue processing while data is written asynchronously.

### Key Features

1. **Background Thread Processing**: All I/O operations run in a separate thread
2. **Command Queue**: Thread-safe queue for communication between threads
3. **Buffer Management**: Efficient buffer handling with configurable sizes
4. **Various Data Formats**: Support for tuple data, dictionaries, and OHLCV records
5. **Automatic Headers**: Header generation based on data structure
6. **Configurable Formatting**: Custom float formatting and timestamp conversion

### Architecture

The writer uses a producer-consumer pattern:

1. The main thread (producer) adds write commands to a thread-safe queue
2. A background worker thread (consumer) processes these commands
3. Data is accumulated in an internal buffer and flushed when:
   - The buffer reaches a threshold size
   - A timeout occurs with no new data
   - The writer is closed

This approach minimizes the impact of I/O operations on application performance, particularly important for real-time data processing.

## CSVReader: Memory Mapped Reading

The `CSVReader` class provides an efficient way to read CSV files, optimized for OHLCV data but flexible enough for general use. It leverages memory mapping for improved performance.

### Key Features

1. **Memory Mapping**: Fast access to file data through the OS's virtual memory system
2. **Format Auto-Detection**: Automatic detection of CSV dialect and headers
3. **Flexible Data Processing**: Support for various column mappings and data types
4. **Extra Fields Support**: Handling of additional columns beyond OHLCV data
5. **Timestamp Parsing**: Automatic conversion of various timestamp formats
6. **NA Value Support**: Special handling for NA/NaN values

## Usage Examples

### Basic CSV Writing

```python
from pynecore.core.csv_file import CSVWriter
from pathlib import Path

# Create a CSV writer
with CSVWriter(Path("example.csv"),
               headers=["timestamp", "value1", "value2"],
               timestamp_as_iso=True) as writer:

    # Write raw tuple data
    writer.write(1609459200, 42.5, 100.0)

    # Write dictionary data
    writer.write_dict({
        "timestamp": 1609459260,
        "value1": 43.2,
        "value2": 101.5
    })
```

### Writing OHLCV Data

```python
from pynecore.core.csv_file import CSVWriter
from pynecore.types.ohlcv import OHLCV
from pathlib import Path

# Create a CSV writer for OHLCV data
with CSVWriter(Path("market_data.csv"),
               timestamp_as_iso=True,
               float_fmt='.2f') as writer:

    # Write OHLCV records
    writer.write_ohlcv(OHLCV(
        timestamp=1609459200,
        open=100.0,
        high=110.0,
        low=90.0,
        close=105.0,
        volume=1000.0,
        extra_fields={"indicator1": 42.5, "signal": "buy"}
    ))
```

### Performance-Tuned CSV Writing

```python
from pynecore.core.csv_file import CSVWriter
from pathlib import Path

# Create a high-performance CSV writer
with CSVWriter(
    Path("large_dataset.csv"),
    buffer_size=65536,      # 64KB buffer
    queue_size=10000,       # Large command queue
    float_fmt='.4g',        # Compact float format
    idle_time=0.1           # Longer idle time before flush
) as writer:

    # Write large volumes of data
    for i in range(100000):
        writer.write(i, i * 2.5, i % 100)
```

### Reading CSV Data

```python
from pynecore.core.csv_file import CSVReader
from pathlib import Path

# Read a CSV file
with CSVReader(Path("market_data.csv")) as reader:
    # Iterate through all records
    for candle in reader:
        print(f"Time: {candle.timestamp}, Close: {candle.close}")

        # Access extra fields
        if candle.extra_fields and "indicator1" in candle.extra_fields:
            print(f"Indicator: {candle.extra_fields['indicator1']}")
```

### Reading Specific Time Ranges

```python
from pynecore.core.csv_file import CSVReader
from pathlib import Path

# Read a specific time range
with CSVReader(Path("market_data.csv")) as reader:
    start_time = 1609459200  # Unix timestamp
    end_time = 1609459800    # Unix timestamp

    for candle in reader.read_from(start_time, end_time):
        print(f"Time: {candle.timestamp}, Close: {candle.close}")
```

## Performance Optimization

### Writer Optimizations

The CSVWriter employs several optimization techniques:

1. **Threaded I/O Operations**: I/O is moved to a background thread to avoid blocking the main application
2. **Buffer Management**: Smart buffer management minimizes system calls
3. **Timeout-Based Flushing**: Buffers are flushed after an idle period, balancing throughput and latency
4. **Efficient String Formatting**: Custom float formatting for optimized string conversion
5. **Batched Operations**: Multiple records are processed together before flushing

### Reader Optimizations

The CSVReader is optimized through:

1. **Memory Mapping**: Data is accessed directly through the OS's virtual memory
2. **Format Auto-Detection**: The CSV dialect is automatically detected for optimal parsing
3. **Type Conversion**: Efficient parsing and conversion of values to appropriate types
4. **Sequential Access Patterns**: Optimized for the sequential nature of CSV files

## Technical Details

### CSVWriter Internals

The CSVWriter uses a command-based architecture:

1. Each write operation generates a command (tuple, dict, or OHLCV)
2. Commands are placed in a thread-safe queue
3. A worker thread processes commands from the queue
4. The worker accumulates data in a string buffer
5. The buffer is flushed to disk when it reaches a threshold size or after an idle period

This design provides several advantages:
- Non-blocking writes for the main application thread
- Batched I/O operations for improved throughput
- Graceful handling of high-volume data streams

### CSVReader Internals

The CSVReader leverages Python's built-in CSV parsing with additional optimizations:

1. Memory mapping provides efficient access to the file data
2. The CSV dialect (delimiter, quoting, etc.) is automatically detected
3. Headers are parsed and mapped to fields, with case-insensitive matching
4. A position system enables reading specific records or timestamp ranges
5. Values are converted to appropriate types (timestamps, floats, etc.)

### Thread Safety

The CSVWriter is designed with thread safety in mind:

1. A thread-safe queue manages communication between threads
2. Critical operations are protected by a lock
3. Error handling ensures that worker thread exceptions are propagated
4. Clean shutdown is guaranteed even in error conditions

## Choosing Between Binary OHLCV and CSV

When working with financial data in PyneCore, you have two main options:

### Binary OHLCV Format (ohlcv_file.py)
- **Pros**: Maximum performance, compact storage, direct random access
- **Cons**: Specialized format, less human-readable, fixed schema

### CSV Format (csv_file.py)
- **Pros**: Universal compatibility, human-readable, flexible schema
- **Cons**: Larger file size, slower access, primarily sequential

### Selection Guidelines

- Use the Binary OHLCV format for:
  - High-performance backtesting
  - Systems requiring frequent random access
  - Long-term data storage

- Use the CSV format for:
  - Data interchange with other systems
  - Human inspection and editing
  - Flexible schema requirements
  - When additional fields beyond OHLCV are needed (see [Extra Fields](./extra-fields.md))

Both systems are designed for performance while maintaining pure Python implementation, aligning with the PyneCore project vision.

## Conclusion

The PyneCore CSV Reader/Writer system provides a high-performance solution for CSV processing, optimized for financial data but flexible enough for general use. By leveraging multithreaded writing and memory-mapped reading, it achieves excellent performance while remaining a pure Python implementation.

The system demonstrates how thoughtful architecture and performance optimization techniques can overcome traditional bottlenecks in Python I/O processing. For applications where CSV compatibility is important but performance cannot be sacrificed, this system offers an ideal balance.