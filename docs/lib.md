<!--
---
weight: 500
title: "Library"
description: "PyneCore library reference"
icon: "library_books"
date: "2025-03-31"
lastmod: "2025-03-31"
draft: false
toc: true
categories: ["API Reference", "Library"]
tags: ["api", "library", "functions", "modules", "indicators", "technical-analysis"]
---
-->

# PyneCore Library

The PyneCore library is a complete Python implementation of the Pine Script API, providing compatibility with Pine Script functions and modules. This library enables easy porting of Pine Script code to Python while taking advantage of all the benefits of the Python language.

## Usage

To use the PyneCore library, simply import the required modules or functions from the `pynecore.lib` package:

```python
"""
@pyne
"""
from pynecore.lib import ta, math, array, close, high, low

# Import specific functions from a module
from pynecore.lib.ta import sma, ema, macd
```

## Global Properties

The global variables familiar from Pine Script (such as `open`, `high`, `low`, `close`, `hl2`, etc.) are available in PyneCore from the `pynecore.lib` package:

```python
"""
@pyne
"""
from pynecore.lib import open, high, low, close
from pynecore.lib import hl2, hlc3, ohlc4, hlcc4
from pynecore.lib import volume, bar_index
```

These variables are automatically managed by the system, always filled with the current bar data during execution.

## Extra Fields

PyneCore supports access to additional data fields beyond the standard OHLCV values through the `extra_fields` dictionary. This feature allows you to access custom indicators, signals, and metadata that you import from CSV files or create programmatically.

**Important**: Extra fields are not built into PyneCore - they are custom data columns that you provide through CSV imports or programmatic creation.

```python
"""
@pyne
"""
from pynecore.lib import extra_fields, close, plot, color

# Check what custom fields are available in your data
available_fields = list(extra_fields.keys())
print(f"My custom fields: {available_fields}")

# Example: Access a custom RSI indicator you imported from CSV
if 'my_rsi_14' in extra_fields:
    current_rsi = extra_fields['my_rsi_14'][0]  # Current bar
    plot(current_rsi, title="My Custom RSI", color=color.purple)

# Example: Access custom volume analysis you calculated
if 'vol_average' in extra_fields:
    current_vol_avg = extra_fields['vol_average'][0]     # Current bar
    previous_vol_avg = extra_fields['vol_average'][1]    # Previous bar (if available)

# Example: Use custom boolean signals you defined
if 'buy_signal' in extra_fields:
    buy_signal = extra_fields['buy_signal'][0]
    signal_value = 1.0 if buy_signal else 0.0
    plot(signal_value, title="My Buy Signal", color=color.green)
```

Extra fields come from your own data sources - either imported from CSV files with additional columns, or created programmatically when writing OHLCV data. See the [OHLCV Reader/Writer documentation](advanced/ohlcv-reader-writer.md#extra-fields-support) for detailed information on creating and using extra fields.

## Module System

The complete module system of the Pine Script API is available in PyneCore. The module names are almost identical to Pine Script names, with only one exception:

| Pine | PyneCore | Note |
|------|----------|------|
| str  | string   | `str` is a built-in Python type name, so it's available as `string` |

### Main Modules

The most important modules of the PyneCore library:

- **ta**: Technical indicators and analysis functions
- **math**: Mathematical functions
- **array**: Array operation functions
- **string**: Text handling functions
- **timeframe**: Timeframe handling functions
- **color**: Color definitions and manipulation functions
- **session**: Session handling functions
- **strategy**: Strategy functions and definitions

## Combining Pine Script and Python Styles

PyneCore allows both Pine Script functional and native Python styles. This flexibility lets you choose the most comfortable and efficient approach for a given task:

```python
"""
@pyne
"""
from pynecore.lib import array, map

# Pine Script-style array module usage
a = array.new_float(5)
array.set(a, 0, 1.0)
array.push(a, 2.0)

# Or use the same with native Python list operations
a[0] = 1.0
a.append(2.0)

# Pine Script-style map module usage
m = map.new()
map.put(m, "key", 42)

# Or use the same with native Python dict operations
m["key"] = 42
```

All functional Pine modules (array, map, etc.) in the PyneCore library work completely, but Python's native solutions can also be used, keeping code readability and personal preferences in mind.

## Pine Script Compatibility

The goal of the PyneCore library is to provide full compatibility with the Pine Script API while leveraging Python's strengths. Technical indicators and other calculations are performed with high precision (0.001% tolerance), just like Pine Script.

### Documentation

Since the library is fully compatible with the Pine Script API, the TradingView official Pine Script documentation is an excellent reference for function usage. You can find information about PyneCore-specific aspects in the documentation of the respective module.

The [Pine Script documentation](https://www.tradingview.com/pine-script-reference/) is available at:
- Technical indicators: [Pine Script ta module](https://www.tradingview.com/pine-script-reference/v5/#lib_ta)
- Mathematical functions: [Pine Script math module](https://www.tradingview.com/pine-script-reference/v5/#lib_math)
- Array operations: [Pine Script array module](https://www.tradingview.com/pine-script-reference/v5/#lib_array)

## Usage Examples

### Technical Indicators

```python
"""
@pyne
"""
from pynecore.lib import script, ta, close, plot

@script.indicator(title="SMA and EMA Comparison")
def main():
    sma_val = ta.sma(close, 20)
    ema_val = ta.ema(close, 20)

    plot(sma_val, "SMA 20")
    plot(ema_val, "EMA 20")
```

### Mathematical Functions

```python
"""
@pyne
"""
from pynecore.lib import script, math, close, plot

@script.indicator(title="Correlation Example")
def main():
    val1 = math.abs(close - close[1])
    result = math.log(val1) if val1 > 0 else 0

    plot(result, "Log of Abs Change")
```

### Strategy Example

```python
"""
@pyne
"""
from pynecore.lib import script, ta, close, high, low
from pynecore.lib import strategy

@script.strategy("Simple Crossover Strategy")
def main():
    fast_ma = ta.sma(close, 9)
    slow_ma = ta.sma(close, 21)

    if ta.crossover(fast_ma, slow_ma):
        strategy.entry("Long", strategy.long)

    if ta.crossunder(fast_ma, slow_ma):
        strategy.entry("Short", strategy.short)
```

## Python Advantages

While PyneCore provides full compatibility with the Pine Script API, you can also take advantage of Python language features. For more detailed differences, see the [Differences from Pine Script](/docs/overview/differences/) page.

### NA Handling

```python
"""
@pyne
"""
from pynecore.lib import na
from pynecore.types.na import NA

# Creating NA values
a = NA()  # or na()

# Typed NA values
f = NA(float)  # or na(float)

# Checking for NA
if isinstance(value, NA) or na(value):
    print("Value is NA")
```

## More Complex Modules

The PyneCore library also includes more complex modules, such as the `strategy` module for strategy development, or various display options. These are typically accessible in a submodule structure:

```python
"""
@pyne
"""
from pynecore.lib import plot, plot_style
from pynecore.lib.strategy import direction

# Using different style elements
plot(series, title="Line Type", style=plot_style.style_line)
plot(series, title="Column Type", style=plot_style.style_columns)
```