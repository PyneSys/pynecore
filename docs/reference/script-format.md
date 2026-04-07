<!--
---
weight: 405
title: "Script Format"
description: "Structure and anatomy of a PyneCore script file"
icon: "description"
date: "2026-04-05"
lastmod: "2026-04-05"
draft: false
toc: true
categories: ["Reference"]
tags: ["script", "pyne", "main", "decorator", "indicator", "strategy", "library"]
---
-->

# Script Format

A PyneCore script is a standard `.py` file with a specific structure. This page documents the
required elements and conventions.

## Minimal Example

```python
"""@pyne"""
from pynecore.lib import script, close


@script.indicator("My Indicator")
def main():
    return {"close": close}
```

## 1. Magic Comment

Every PyneCore script must begin with a docstring that starts with `@pyne`:

```python
"""@pyne"""
```

The docstring can also serve as the script's documentation — any text after `@pyne` is treated
as a description:

```python
"""
@pyne
RSI Mean Reversion Strategy

This strategy enters long positions when RSI crosses above the oversold level
and exits when RSI crosses below the overbought level.
"""
```

This marker tells the import hook to apply AST transformations (Series, Persistent, function
isolation, etc.) before executing the module. Without it, the file is treated as regular Python.

## 2. Imports

Import type annotations from `pynecore` and library functions from `pynecore.lib`:

```python
from pynecore.types import Series, Persistent, IBPersistent
from pynecore.lib import script, input, close, open, high, low, volume, ta, strategy, plot, color
```

`pynecore.types` provides the type annotations. `pynecore.lib` provides all runtime functions,
built-in variables, and namespaces (ta, math, strategy, color, etc.).

## 3. Script Decorator

The `main()` function must be decorated with one of:

### @script.indicator

For scripts that calculate and display values.

```python
@script.indicator("SMA Indicator", overlay=True)
def main():
    ...
```

Key parameters:

| Parameter          | Default | Description                                   |
|--------------------|---------|-----------------------------------------------|
| `title`            | `''`    | Display name                                  |
| `overlay`          | `False` | Show on price chart (`True`) or separate pane |
| `format`           | inherit | Number formatting                             |
| `precision`        | `None`  | Decimal digits                                |
| `max_bars_back`    | `0`     | History buffer length (0 = auto)              |
| `dynamic_requests` | `False` | Allow dynamic `request.*()` calls             |

### @script.strategy

For scripts that simulate trading.

```python
@script.strategy(
    "MA Crossover",
    overlay=True,
    initial_capital=10000,
    commission_type=strategy.commission.percent,
    commission_value=0.1,
)
def main():
    ...
```

Key parameters (in addition to indicator parameters):

| Parameter                 | Default              | Description                            |
|---------------------------|----------------------|----------------------------------------|
| `initial_capital`         | `1000000`            | Starting capital                       |
| `currency`                | `currency.NONE`      | Account currency                       |
| `pyramiding`              | `0`                  | Max entries in same direction          |
| `default_qty_type`        | `strategy.fixed`     | Position sizing method                 |
| `default_qty_value`       | `1`                  | Default quantity                       |
| `commission_type`         | `commission.percent` | Commission calculation method          |
| `commission_value`        | `0.0`                | Commission amount                      |
| `slippage`                | `0`                  | Slippage in ticks                      |
| `margin_long`             | `100.0`              | Long margin percentage                 |
| `margin_short`            | `100.0`              | Short margin percentage                |
| `calc_on_order_fills`     | `False`              | Re-execute on fills                    |
| `calc_on_every_tick`      | `False`              | Re-execute on every tick (live only)   |
| `use_bar_magnifier`       | `True`               | Use LTF data for fill accuracy         |
| `process_orders_on_close` | `False`              | Extra order processing after bar close |
| `close_entries_rule`      | `'FIFO'`             | Trade closing order                    |
| `risk_free_rate`          | `2.0`                | For Sharpe ratio calculation           |

### @script.library

For reusable library modules.

```python
@script.library("My Utils", overlay=True)
def main():
    ...
```

## 4. The main() Function

The `main()` function is called **once per bar** during execution. It receives input parameters
and contains all script logic.

### Input Parameters

Configurable parameters are declared as function arguments with `input.*()` defaults:

```python
@script.indicator("RSI", overlay=False)
def main(
        length: int = input.int(14, title="RSI Length", minval=1),
        overbought: int = input.int(70, title="Overbought"),
        oversold: int = input.int(30, title="Oversold"),
        src: Series[float] = input.source(close, title="Source"),
):
    rsi_value = ta.rsi(src, length)
    ...
```

See [Input Functions](inputs.md) for all available input types.

### Return Value

`main()` can optionally return a `dict` of named values to plot:

```python
def main():
    fast = ta.sma(close, 9)
    slow = ta.sma(close, 21)
    return {"Fast MA": fast, "Slow MA": slow}
```

Alternatively, use `plot()` calls within the function body. Both approaches can be combined.

## Complete Example

```python
"""@pyne"""
from pynecore.types import Series, Persistent
from pynecore.lib import script, input, close, ta, strategy, plot, color


@script.strategy(
    "RSI Mean Reversion",
    overlay=True,
    initial_capital=10000,
    commission_type=strategy.commission.percent,
    commission_value=0.1,
)
def main(
        rsi_length: int = input.int(14, title="RSI Length", minval=1),
        oversold: int = input.int(30, title="Oversold Level"),
        overbought: int = input.int(70, title="Overbought Level"),
):
    rsi: Series[float] = ta.rsi(close, rsi_length)

    trade_count: Persistent[int] = 0

    if ta.crossover(rsi, oversold):
        strategy.entry("Long", strategy.long)
        trade_count += 1

    if ta.crossunder(rsi, overbought):
        strategy.close("Long")

    plot(rsi, "RSI", color=color.purple)

    return {
        "Overbought": overbought,
        "Oversold": oversold,
    }
```
