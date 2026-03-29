<!--
---
weight: 600
title: "Scripting with PyneCore"
description: "Writing effective and idiomatic Pyne scripts"
icon: "code"
date: "2025-03-31"
lastmod: "2025-03-31"
draft: false
toc: true
categories: ["Usage", "Scripting"]
tags: ["scripting", "python", "indicators", "strategies", "patterns", "practices"]
---
-->

# Scripting with PyneCore

This guide focuses on the PyneCore-specific aspects of writing trading scripts, particularly on the unique features that distinguish it from standard Python code. For a complete introduction to creating your first script, please start with the [First Script](/docs/getting-started/first-script/) tutorial.

## Script Structure

Every PyneCore script follows a consistent structure:

```python
"""
@pyne
"""
# Imports
from pynecore import Series, Persistent
from pynecore.lib import script, input, close, ta, plot, color

# Script declaration
@script.indicator("My Indicator", overlay=True)  # Or @script.strategy for strategies
def main(
    # Input parameters as function arguments
    src: Series[float] = input.source('close', title="Source"),
    length: int = input.int(14, minval=1, title="Length")
):
    # Script logic
    result = ta.sma(src, length)

    # Output/visualization
    plot(result, "SMA", color=color.blue)

    # Alternative: return a dictionary of plots
    return {
        "Result": result
    }
```

## PyneCore-Specific Elements

### 1. The Magic Comment

The `@pyne` comment at the top of your script is essential - it signals to PyneCore that this script should undergo AST transformations to enable Pine Script-like behavior.

```python
"""
@pyne
"""
```

### 2. Decorators for Script Types

PyneCore uses decorators to specify the script type:

```python
@script.indicator(title="My Indicator", overlay=True)
```

or

```python
@script.strategy(title="My Strategy", overlay=True)
```

These decorators accept numerous parameters for configuring your script, such as:

- `title`: The display name of the script
- `overlay`: Whether to display on the main chart (True) or in a separate pane (False)
- `format`: Formatting for displayed values
- And many others (see the documentation for each decorator for details)

### 3. Input Parameters

Unlike Pine Script where inputs are defined with `input.*()` functions in the global scope, PyneCore defines inputs as function arguments with default values:

```python
def main(
    length: int = input.int(14, title="Length", minval=1, maxval=100),
    source: Series[float] = input.source('close', title="Source"),
    show_bands: bool = input.bool(True, title="Show Bands")
):
```

Available input types:
- `input.int()` - Integer inputs
- `input.float()` - Float inputs
- `input.bool()` - Boolean inputs (checkboxes)
- `input.string()` - Text inputs or dropdown selections
- `input.color()` - Color picker
- `input.source()` - Data source selector

### 4. Series and Persistent Variables

Two special types unique to PyneCore:

- **Series[T]**: Time series data with historical values
  ```python
  price: Series[float] = close
  previous_price = price[1]  # Access previous bar's value
  ```

- **Persistent[T]**: Variables that maintain state between bars
  ```python
  counter: Persistent[int] = 0
  counter += 1  # Increments on each bar
  ```

These types are automatically transformed by PyneCore's AST transformers to implement Pine Script-like behavior. For more details, see [Core Concepts](/docs/overview/core-concepts/).

### 5. NA Handling

PyneCore implements Pine Script's NA (Not Available) concept for handling missing or undefined values:

```python
from pynecore.lib import na

# Checking if a value is NA
if na(value):
    # Handle NA case
    value = default_value

# Create NA values
from pynecore.types.na import NA
value = NA(float)  # Typed NA
```

## Output Methods

PyneCore provides two ways to generate output from your scripts:

### 1. Using `plot()` Function

```python
from pynecore.lib import plot, color

# Plot a value with title and color
plot(my_series, "My Indicator", color=color.blue)

# Additional plot styles are available
from pynecore.lib import plot_style
plot(my_series, "Columns", style=plot_style.style_columns)
```

### 2. Return Dictionary

PyneCore has a unique feature not found in Pine Script - you can return a dictionary of values to plot:

```python
def main():
    fast_ma = ta.sma(close, 10)
    slow_ma = ta.sma(close, 20)

    return {
        "Fast MA": fast_ma,
        "Slow MA": slow_ma
    }
```

Both approaches can be used simultaneously in the same script.

## Core Library Functions

PyneCore includes a comprehensive library of functions closely matching Pine Script's functionality:

### Technical Analysis

Technical indicators are available in the `ta` module:

```python
from pynecore.lib import ta

sma_value = ta.sma(close, 20)
rsi_value = ta.rsi(close, 14)
macd_line, signal, hist = ta.macd(close, 12, 26, 9)
```

### Mathematical Functions

Mathematical operations are available in the `math` module:

```python
from pynecore.lib import math

value = math.abs(close - close[1])
log_value = math.log(value)
```

### Bar Information

Information about the current bar:

```python
from pynecore.lib import bar_index, barstate

# Current bar index
current_bar = bar_index

# Bar state information
is_first_bar = barstate.isfirst
is_last_bar = barstate.islast
```

### Strategy Functions

For trading strategies, use the `strategy` module:

```python
from pynecore.lib import strategy

# Enter a long position
strategy.entry("Long", strategy.long)

# Exit all positions
strategy.close_all()
```

## Writing Functions

Functions in PyneCore can be defined in traditional Python style:

```python
def calculate_atr_bands(src, length=14, multiplier=2):
    atr_value = ta.atr(length)
    upper = src + atr_value * multiplier
    lower = src - atr_value * multiplier
    return upper, lower

# Using the function
upper_band, lower_band = calculate_atr_bands(close, length=20, multiplier=3)
```

One key difference from standard Python is that functions in PyneCore maintain isolated state for Series and Persistent variables, similar to Pine Script's behavior. This means each call instance maintains its own persistent state.

## Common Patterns

### Strategy Signal Generation

```python
# Generate signals with crossovers
buy_signal = ta.crossover(fast_ma, slow_ma)
sell_signal = ta.crossunder(fast_ma, slow_ma)

# Execute trades on signals
if buy_signal:
    strategy.entry("Long", strategy.long)
elif sell_signal:
    strategy.close("Long")
```

### Handling Multiple Timeframes

While PyneCore doesn't currently implement the `security()` function from Pine Script, you can work with data from different timeframes by using the `timeframe.change()` function:

```python
from pynecore.lib import timeframe

# Check if we're at the beginning of a new day
if timeframe.change("D"):
    # Execute logic at the start of a new daily candle
    daily_high: Persistent[float] = high
    daily_low: Persistent[float] = low
else:
    # Update daily high/low on intraday candles
    daily_high = max(daily_high, high)
    daily_low = min(daily_low, low)
```

## Debugging Techniques

Debug PyneCore scripts using:

```python
from pynecore.lib import log

# Debug logging
log.debug(f"Debug: close={close}, sma={sma_value}")
log.info("Information message")
log.warning("Warning message")
log.error("Error message")
```

## Further Resources

- [Core Concepts](/docs/overview/core-concepts/) - Detailed explanation of PyneCore's fundamental concepts
- [Differences from Pine Script](/docs/overview/differences/) - Key differences to be aware of
- [PyneCore Reference](/docs/reference/) - Complete reference for all library functions, types, and language features