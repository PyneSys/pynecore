<!--
---
weight: 1003
title: "OHLCV Reader/Writer"
description: "Ultra-fast OHLCV data storage and access using memory mapping"
icon: "data_object"
date: "2025-03-31"
lastmod: "2025-03-31"
draft: false
toc: true
categories: ["Advanced", "Data Handling"]
tags: ["ohlcv", "data", "performance", "memory-mapping", "binary-format", "io"]
---
-->

# OHLCV Reader/Writer

The PyneCore OHLCV Reader/Writer system provides a blazing-fast, pure Python solution for storing and accessing financial market data in a highly optimized format. It leverages memory-mapped files to achieve near-native performance while maintaining the advantages of Python's simplicity and portability.

## Overview

Time series data is at the heart of any trading system. The OHLCV (Open, High, Low, Close, Volume) format is the industry standard for representing price action over a specific time interval. Efficient storage and retrieval of this data is critical for backtesting and algorithmic trading applications.

The PyneCore OHLCV system was designed with these principles in mind:

- **Maximum Speed**: Achieve the fastest possible read/write performance in pure Python
- **Memory Efficiency**: Access large datasets without loading them entirely into memory
- **Simple Format**: Use a straightforward binary format that's easy to understand
- **Automatic Gap Handling**: Intelligently handle missing data points
- **Conversion Support**: Import from and export to common formats like CSV and JSON

## Binary File Format

The `.ohlcv` file format uses a simple, fixed-size binary structure of 24 bytes per record:

| Field     | Type      | Size (bytes) | Description                     |
|-----------|-----------|-------------|---------------------------------|
| timestamp | uint32    | 4           | Unix timestamp (seconds)        |
| open      | float32   | 4           | Opening price                   |
| high      | float32   | 4           | Highest price                   |
| low       | float32   | 4           | Lowest price                    |
| close     | float32   | 4           | Closing price                   |
| volume    | float32   | 4           | Trading volume                  |

This format provides several benefits:

1. Fixed-size records allow direct position calculation and random access
2. 32-bit types strike an optimal balance between precision and space efficiency
3. Simple structure eliminates the need for complex parsing

The timestamp field uses a uint32 format which is good until the year 2106. While this might seem limiting, it's a deliberate choice to maintain the compact 24-byte record size. The implementation includes a humorous promise to "fix this then" when 2106 approaches.

## Memory Mapping for Maximum Performance

The OHLCV reader uses memory mapping (`mmap`) to access file data directly through the operating system's virtual memory, providing several key advantages:

1. **Zero-Copy Access**: Data is accessed directly from the file system cache without copying to Python memory
2. **Lazy Loading**: Only the portions of the file being accessed are actually loaded into memory
3. **OS-Level Optimization**: The operating system can optimize read-ahead and caching behavior
4. **Shared Resources**: Multiple processes can efficiently share the same memory-mapped file

This approach results in performance that rivals C/C++ implementations while maintaining the simplicity and portability of pure Python.

## Usage Examples

### Basic Reading and Writing

```python
from pynecore.core.ohlcv_file import OHLCVReader, OHLCVWriter
from pynecore.types.ohlcv import OHLCV
from pathlib import Path

# Writing OHLCV data
file_path = Path("example.ohlcv")
with OHLCVWriter(file_path) as writer:
    # Write individual candles
    writer.write(OHLCV(timestamp=1609459200, open=100.0, high=110.0, low=90.0, close=105.0, volume=1000.0))
    writer.write(OHLCV(timestamp=1609459260, open=105.0, high=115.0, low=95.0, close=110.0, volume=1200.0))

# Reading OHLCV data
with OHLCVReader(file_path) as reader:
    # Iterate through all candles
    for candle in reader:
        print(f"Time: {candle.timestamp}, Close: {candle.close}, Volume: {candle.volume}")

    # Get file metadata
    print(f"Interval: {reader.interval} seconds")
    print(f"Start time: {reader.start_datetime}")
    print(f"End time: {reader.end_datetime}")
```

### Reading Specific Time Ranges

```python
with OHLCVReader(file_path) as reader:
    # Read data from a specific time range
    start_time = 1609459200  # Unix timestamp
    end_time = 1609459800    # Unix timestamp

    for candle in reader.read_from(start_time, end_time):
        print(f"Time: {candle.timestamp}, Close: {candle.close}")
```

### Converting From/To Other Formats

```python
# Import from CSV
with OHLCVWriter(Path("from_csv.ohlcv")) as writer:
    writer.load_from_csv(Path("data.csv"))

# Export to CSV
with OHLCVReader(Path("example.ohlcv")) as reader:
    reader.save_to_csv("exported.csv", as_datetime=True)  # as_datetime=True uses ISO format dates

# Import from JSON
with OHLCVWriter(Path("from_json.ohlcv")) as writer:
    writer.load_from_json(
        Path("data.json"),
        mapping={"timestamp": "time", "volume": "vol"}  # Custom field mapping
    )

# Export to JSON
with OHLCVReader(Path("example.ohlcv")) as reader:
    reader.save_to_json("exported.json", as_datetime=True)
```

## Automatic Gap Handling

The OHLCV system automatically handles gaps in time series data, which is crucial for accurate backtesting. When writing data with missing intervals:

1. The system detects gaps based on the established interval between records
2. Gaps are filled with the previous candle's close price for OHLC values
3. Gap candles are marked with a special volume value of -1
4. When reading, gap candles can be automatically filtered with `skip_gaps=True`

```python
# Writing data with a gap
with OHLCVWriter(file_path) as writer:
    writer.write(OHLCV(timestamp=1609459200, open=100.0, high=110.0, low=90.0, close=105.0, volume=1000.0))
    writer.write(OHLCV(timestamp=1609459260, open=105.0, high=115.0, low=95.0, close=110.0, volume=1200.0))
    # Gap at 1609459320 will be automatically filled
    writer.write(OHLCV(timestamp=1609459380, open=115.0, high=125.0, low=105.0, close=120.0, volume=1300.0))

# Reading data while skipping gaps
with OHLCVReader(file_path) as reader:
    # Only includes actual candles, skips the gap at 1609459320
    for candle in reader.read_from(1609459200, skip_gaps=True):
        print(f"Time: {candle.timestamp}, Close: {candle.close}")
```

## Advanced Operations

### Seeking and Truncating

The OHLCV writer supports seeking to specific positions and truncating files:

```python
with OHLCVWriter(file_path) as writer:
    # Seek to a specific position (record number)
    writer.seek(5)  # Move to the 6th record

    # Seek to a specific timestamp
    writer.seek_to_timestamp(1609459500)

    # Truncate file at current position
    # All data after the current position will be deleted
    writer.truncate()
```

### Performance Considerations

The OHLCV reader/writer is designed for maximum performance:

1. Direct memory access provides near-native speed
2. Fixed-size records enable O(1) random access by position
3. Straightforward timestamp-to-position calculation allows fast time-range queries
4. No compression means no CPU overhead for decompression

For typical backtesting scenarios, the system can process millions of candles per second on modern hardware.

## Extra Fields Support

The OHLCV system supports storing additional data fields alongside the standard OHLCV values, providing a powerful way to include custom indicators, signals, and metadata in your market data files.

### What Are Extra Fields?

Extra fields allow you to store any additional data columns from CSV imports or programmatically created fields alongside your OHLCV data. These are custom fields that you define, not built-in PyneCore features. Examples include:

- Custom technical indicators you calculated (RSI, MACD, Bollinger Bands)
- Your own signals and trading conditions  
- Volume-based analysis you computed (VWAP, Volume averages)
- Boolean flags for your strategies (buy/sell signals, trend conditions)
- String metadata you want to track (market state, signal types)
- Any other numeric or text data relevant to your analysis

### Storage Architecture

Extra fields are stored separately from the main OHLCV binary file:

- **Main file**: `data.ohlcv` contains the standard 24-byte OHLCV records
- **Companion file**: `data.extra_fields.json` stores the additional fields as JSON

This architecture maintains the high-performance binary format for core OHLCV data while providing flexible storage for additional fields.

### Automatic CSV Import with Extra Fields

When importing CSV files, any columns beyond the standard OHLCV fields are automatically detected and stored as extra fields:

```python
# Example CSV with your custom fields: 
# timestamp,open,high,low,close,volume,my_rsi_14,vol_average,buy_signal
from pynecore.core.ohlcv_file import OHLCVWriter
from pathlib import Path

with OHLCVWriter(Path("data_with_extras.ohlcv")) as writer:
    writer.load_from_csv(Path("my_market_data.csv"))
    # Your custom fields (my_rsi_14, vol_average, buy_signal) automatically preserved
```

The system handles various data types intelligently:
- **Numeric values**: Stored as floats for calculations
- **Boolean values**: `true`/`false` strings converted to Python booleans
- **Text values**: Preserved as strings for metadata

### Programmatic Extra Fields Creation

You can also create extra fields programmatically when writing OHLCV data:

```python
from pynecore.types.ohlcv import OHLCV

with OHLCVWriter(Path("custom_data.ohlcv")) as writer:
    # Write candle with extra fields
    writer.write(OHLCV(
        timestamp=1609459200,
        open=29000.0,
        high=29500.0,
        low=28800.0,
        close=29200.0,
        volume=1000.5,
        extra_fields={
            "my_rsi_14": 65.4,
            "vol_average": 950.3,
            "buy_signal": True,
            "signal_type": "buy_entry",
            "trend_direction": "upward"
        }
    ))
```

### Using Extra Fields in PyneCore Scripts

Extra fields are accessible in your PyneCore trading scripts through the `extra_fields` dictionary:

```python
"""
@pyne
"""
from pynecore.lib import extra_fields, close, plot, color, script

@script.indicator(title="Custom Extra Fields Example")
def main():
    # Check what custom fields are available in your data
    available_fields = list(extra_fields.keys())
    print(f"My custom fields: {available_fields}")
    
    # Access your custom RSI indicator (that you imported from CSV)
    if 'my_rsi_14' in extra_fields:
        current_rsi = extra_fields['my_rsi_14'][0]  # Current bar
        plot(current_rsi, title="My RSI 14", color=color.purple)
    
    # Access your custom volume analysis
    if 'vol_average' in extra_fields:
        current_vol_avg = extra_fields['vol_average'][0]  # Current bar
        previous_vol_avg = extra_fields['vol_average'][1] if len(extra_fields['vol_average']) > 1 else None
        
        if previous_vol_avg:
            vol_change = current_vol_avg - previous_vol_avg
            plot(vol_change, title="Volume Average Change", color=color.orange)
    
    # Use your custom boolean signals
    if 'buy_signal' in extra_fields:
        buy_signal = extra_fields['buy_signal'][0]
        signal_value = 1.0 if buy_signal else 0.0
        plot(signal_value, title="My Buy Signal", color=color.green)
    
    # Combine multiple custom fields
    if 'my_rsi_14' in extra_fields and 'vol_average' in extra_fields:
        rsi = extra_fields['my_rsi_14'][0]
        vol_avg = extra_fields['vol_average'][0]
        
        # Create your own composite indicator
        composite = (rsi / 100.0) * (vol_avg / 1000.0)
        plot(composite, title="My RSI-Volume Composite", color=color.blue)
    
    # Reference: plot standard price data
    plot(close, title="Close Price", color=color.black)
```

### Extra Fields Data Types

The system supports various data types with automatic type detection and preservation:

```python
# Example showing different data types you can use
my_custom_fields_example = {
    # Numeric types
    "my_rsi_14": 65.42,           # Float
    "vol_average": 1250,          # Integer (stored as float)
    "signal_strength": 0.85,      # Decimal
    
    # Boolean types  
    "buy_signal": True,           # Boolean
    "above_ma": False,            # Boolean
    
    # String types
    "trend_state": "uptrend",     # Text
    "market_phase": "active",     # Status
    "entry_type": "breakout",     # Category
    
    # Edge cases
    "zero_threshold": 0.0,        # Zero
    "sentiment_score": -123.45,   # Negative
    "large_volume": 1e10,         # Scientific notation
}
```

### Performance Considerations

Extra fields are designed for efficiency:

1. **Lazy Loading**: Extra fields JSON is only loaded when the OHLCV file is opened
2. **Memory Efficient**: Extra fields don't impact the core OHLCV binary file size
3. **Fast Access**: JSON parsing is done once per file open, then cached in memory
4. **Graceful Degradation**: OHLCV files work normally even if extra fields JSON is missing or corrupted

### Data Integrity and Error Handling

The system handles various error conditions gracefully:

- **Missing JSON File**: OHLCV data remains accessible, extra fields return empty
- **Corrupted JSON**: Parsing errors are caught, core functionality preserved  
- **Type Mismatches**: Automatic type coercion where possible
- **Field Name Conflicts**: Non-standard field names are preserved (including Unicode)

### Best Practices

When working with extra fields:

1. **Consistent Field Names**: Use descriptive, consistent naming (e.g., `my_rsi_14`, `vol_sma_50`)
2. **Type Consistency**: Keep field types consistent across records when possible
3. **Field Documentation**: Document custom fields for team collaboration
4. **Performance Testing**: Test with realistic data sizes for your use case

```python
# Good field naming examples for your custom fields
my_extra_fields = {
    "my_rsi_14": 65.4,        # Clear custom indicator and period
    "bb_upper_20": 31250.5,   # Your Bollinger Band calculation
    "vol_sma_20": 1250.3,     # Your volume average with period
    "breakout_signal": True,  # Clear boolean condition you defined
    "trend_strength": 0.85,   # Your descriptive numeric analysis
}
```

## Import/Export Capabilities

The system provides flexible import and export options for various formats:

### CSV Import

```python
writer.load_from_csv(
    path=Path("data.csv"),
    timestamp_format="%Y-%m-%d %H:%M:%S",  # Optional custom datetime format
    timestamp_column="timestamp",          # Column name for the timestamp
    tz="UTC"                               # Timezone information
)
```

For split date/time fields:

```python
writer.load_from_csv(
    path=Path("data.csv"),
    date_column="date",
    time_column="time",
    tz="Europe/London"
)
```

### JSON Import

```python
writer.load_from_json(
    path=Path("data.json"),
    timestamp_field="t",            # Field name for timestamp
    mapping={                       # Custom field mapping
        "open": "o",
        "high": "h",
        "low": "l",
        "close": "c",
        "volume": "v"
    },
    tz="+0100"                      # Timezone as offset
)
```

## Technical Details

### Memory Efficiency

Unlike many data storage systems that load entire datasets into memory, the OHLCV reader uses memory mapping to access only the portions of data actually being used. This allows working with datasets that are larger than available RAM.

### File Structure Benefits

The simple, fixed-size record structure provides several advantages:

1. **Direct Position Calculation**: Record position = timestamp_diff / interval
2. **No Index Needed**: The file itself is inherently indexed by time
3. **Sequential Read Performance**: Adjacent records are stored contiguously for optimal read-ahead
4. **Atomic Updates**: Fixed-size records can be updated in-place without rewriting the entire file

### Chronological Validation

The writer enforces chronological order of timestamps to maintain data integrity:

1. Records must be written in ascending timestamp order
2. Duplicate timestamps are rejected
3. Intervals are automatically detected from the first two records
4. Timeframe consistency is maintained throughout the file

## Conclusion

The PyneCore OHLCV Reader/Writer system demonstrates that pure Python implementations can achieve exceptional performance when designed with careful attention to system-level optimizations like memory mapping. By combining a simple binary format with direct memory access, the system provides the speed of native code with the simplicity and portability of Python.

This design philosophy—prioritizing simplicity and speed while avoiding unnecessary dependencies—is central to the PyneCore project's vision. The OHLCV system exemplifies how thoughtful design choices and leveraging OS-level features like memory mapping can create a solution that is both fast and maintainable.