<!--
---
weight: 202
title: "Your First PyneCore Script"
description: "Learn how to write and run your first PyneCore script"
icon: "code"
date: "2025-03-31"
lastmod: "2025-03-31"
draft: false
toc: true
categories: ["Getting Started"]
tags: ["examples", "tutorial", "scripting", "indicators"]
---
-->

# Your First PyneCore Script

This guide will walk you through creating and running your first PyneCore script. You'll learn the basic structure of a Pyne script and see how it differs from regular Python.

## The Magic Comment

Every PyneCore script must start with a special magic comment to identify it as a Pyne script:

```python
"""
@pyne
"""
```

This comment must be placed at the beginning of your file, before any import statements. It tells the PyneCore system to apply the necessary AST transformations to make your Python code behave like Pine Script.

## Basic Script Structure

A minimal PyneCore script has the following structure:

```python
"""
@pyne
"""
from pynecore.lib import script, close

@script.indicator("My First Indicator")
def main():
    # Your code goes here
    return {"close": close}
```

Let's break this down:

1. **Magic Comment**: Signals that this is a Pyne script
2. **Imports**: Import necessary modules from PyneCore
3. **Script Declaration**: Use a decorator to define the script type (indicator, strategy, etc.)
4. **Main Function**: The entry point of your script that will be executed for each bar

## Creating a Simple Moving Average Indicator

Let's create a simple indicator that calculates and plots a 20-period Simple Moving Average (SMA):

```python
"""
@pyne
"""
from pynecore import Series
from pynecore.lib import script, close, ta, plot, color

@script.indicator("Simple Moving Average", overlay=True)
def main():
    # Calculate 20-period SMA
    sma = ta.sma(close, 20)

    # Plot the SMA on the chart (PyneCore won't plot anything, it just saves the data)
    plot(sma, "SMA")
```

Save this code to a file named `simple_ma.py` in your working directory's scripts/` folder.

## Download some data

This will download the OHLCV data for the BTC/USDT pair from Bybit and save it as `ccxt_BYBIT_BTC_USDT_USDT_1D.ohlcv` in your working directory's data/` folder:

```bash
pyne data download ccxt --symbol "BYBIT:BTC/USDT:USDT"
```

## Running Your Script

To run your PyneCore script, use the command-line interface:

```bash
pyne run simple_ma ccxt_BYBIT_BTC_USDT_USDT_1D.ohlcv
```

This will run your script on every bar of the OHLCV data, and save the plots or returned values to the output folder of
your working directory. This mechanism is the heart of PyneCore. You actually write a script that will run on every candle.

### Direct Execution

If you have a compiled script (from [PyneSys](https://pynesys.io) or [converted from Pine Script](./converting-from-pine.md)), you can run it directly with Python — no CLI or workdir needed:

```bash
python simple_ma.py my_data.csv
```

This works because compiled scripts include a built-in bootstrap that:
- Accepts a CSV or OHLCV data file as argument
- Auto-detects the symbol and timeframe from the filename and data
- Outputs CSV files next to your script (plots, trades, strategy stats)

> **Note:** Direct execution requires PyneCore to be installed (`pip install pynecore`).
> For more control over execution (date ranges, custom output paths), use the `pyne run` CLI.

## Adding Parameters

Let's enhance our script by adding user-configurable parameters:

```python
"""
@pyne
"""
from pynecore import Series
from pynecore.lib import script, close, ta, plot, color, input

@script.indicator("Customizable Moving Average", overlay=True)
def main():
    # Input parameters
    length = input.int(20, "Period", minval=1)
    ma_type = input.string("SMA", "Type", options=["SMA", "EMA", "WMA"])
    line_color = input.color(color.blue, "Line Color")

    # Calculate the selected moving average
    ma: Series[float] = None
    if ma_type == "SMA":
        ma = ta.sma(close, length)
    elif ma_type == "EMA":
        ma = ta.ema(close, length)
    else:  # WMA
        ma = ta.wma(close, length)

    # Plot the result
    plot(ma, f"{ma_type} ({length})", color=line_color, linewidth=2)
```

PyneCore will save a [toml](https://toml.io/en/) file in the output folder which contains the parameters of your script. If you would like to change a parameter, you can do so by removing the comment from the line and changing the default value.

## Understanding Series and Persistence

Two key concepts in PyneCore are Series and Persistent variables:

### Series Variables

A Series variable holds a time series of values, with one value per bar:

```python
"""
@pyne
"""
from pynecore import Series
from pynecore.lib import script, close, high, low, ta, plot, color

@script.indicator("Series Example", overlay=True)
def main():
    # Series variables
    price: Series[float] = close
    highest: Series[float] = ta.highest(high, 10)
    lowest: Series[float] = ta.lowest(low, 10)

    # Series operations
    middle = (highest + lowest) / 2

    # Accessing historical values
    prev_price = price[1]  # Previous bar's price

    # Plot results
    plot(middle, "Middle", color=color.purple)
```

In Pyne code you can declare a Series by adding a `Series` type annotation, but you don't need to actually create a Series object. More about Series [here](../overview/core-concepts.md#3-series-variables).

### Persistent Variables

Persistent variables retain their values between bars:

```python
"""
@pyne
"""
from pynecore import Series, Persistent
from pynecore.lib import script, close, plot, color

@script.indicator("Persistent Example", overlay=True)
def main():
    # Persistent variable - retains its value from bar to bar
    counter: Persistent[int] = 0
    counter += 1

    # Persistent with series operations
    sum_price: Persistent[float] = 0
    sum_price += close

    avg_price = sum_price / counter

    # Plot
    plot(avg_price, "Average Price", color=color.orange)
```

More about Persistent variables [here](../overview/core-concepts.md#2-persistent-variables).

## Creating a Complete Strategy

Let's create a simple trading strategy using PyneCore:

```python
"""
@pyne
"""
from pynecore import Series
from pynecore.lib import script, close, ta, strategy, plot, color

@script.strategy("Simple Crossover Strategy", overlay=True)
def main():
    # Calculate fast and slow moving averages
    fast_ma: Series[float] = ta.ema(close, 9)
    slow_ma: Series[float] = ta.ema(close, 21)

    # Define entry conditions
    buy_signal = ta.crossover(fast_ma, slow_ma)
    sell_signal = ta.crossunder(fast_ma, slow_ma)

    # Execute the strategy
    if buy_signal:
        strategy.entry("Long", strategy.long)
    elif sell_signal:
        strategy.entry("Short", strategy.short)

    # Plot indicators
    plot(fast_ma, "Fast EMA", color=color.blue)
    plot(slow_ma, "Slow EMA", color=color.red)
```

<small>
Note: The `Series[float]` type annotation in the above code is not necessary, as it is not used as a series because it is not indexed. It is included for clarity. The `ta.ema` function creates its own Series internally, so the basic float type could have been used instead. This differs from Pine Script behavior.
</small>

## Working with Functions

You can define your own functions in PyneCore:

```python
"""
@pyne
"""
from pynecore import Series
from pynecore.lib import script, close, plot, color, math

def average(a, b):
    return (a + b) / 2

def custom_ma(src, length):
    """A simple moving average implementation"""
    sum = 0.0
    for i in range(length):
        sum += src[i]
    return sum / length

@script.indicator("Custom Functions", overlay=True)
def main():
    # Use custom functions
    avg = average(close, close[1])
    custom_average: Series[float] = custom_ma(close, 20)

    # Plot results
    plot(avg, "Bar Average", color=color.green)
    plot(custom_average, "Custom MA", color=color.purple)
```

You could also use inline functions, if you need variables from outer scopes. Learn more about how functions work in PyneCore in the [Function Isolation](../overview/core-concepts.md#4-function-isolation) and [Advanced Function Isolation](../advanced/function-isolation.md) documentation.

## Next Steps

Now that you've created your first PyneCore script, you can:

1. Learn how to [convert existing Pine Script code](./converting-from-pine.md) to PyneCore
2. Explore the [core concepts](../overview/core-concepts.md) of PyneCore in depth
3. Check out the [library documentation](../lib.md) for available functions and indicators
4. Dive into [advanced topics](../advanced/) for more technical details
