<!--
---
weight: 700
title: "Strategy Development"
description: "Creating and testing trading strategies with PyneCore"
icon: "trending_up"
date: "2025-03-31"
lastmod: "2026-09-28"
draft: false
toc: true
categories: ["Usage", "Strategy"]
tags: ["strategies", "backtesting", "trading", "examples", "position-sizing", "best-practices"]
---
-->

# Strategy Development with PyneCore

This guide introduces you to creating and testing trading strategies using PyneCore. PyneCore's strategy API follows
Pine Script v6, and its broker emulator is validated trade by trade against TradingView (see
[Compatibility](/docs/overview/compatibility/)). This document covers the basics and links to the PyneCore reference for
details.

## Creating a Strategy

In PyneCore, creating a strategy starts with a script decorated with `@script.strategy`:

```python
"""
@pyne
"""
from pynecore import Series, Persistent
from pynecore.lib import script, close, ta, strategy


@script.strategy("Simple MA Crossover", overlay=True)
def main():
    # Strategy logic here
    fast_ma = ta.sma(close, 9)
    slow_ma = ta.sma(close, 21)

    # Generate signals
    if ta.crossover(fast_ma, slow_ma):
        strategy.entry("Long", strategy.long)

    if ta.crossunder(fast_ma, slow_ma):
        strategy.close("Long")
```

The `@script.strategy` decorator accepts numerous parameters to configure backtest settings:

```python
@script.strategy(
    "Advanced Strategy Settings",
    initial_capital=10000,  # Starting capital
    commission_type=strategy.commission.percent,  # Commission type
    commission_value=0.1,  # Commission value (0.1%)
    pyramiding=1,  # Max number of entries in same direction
    default_qty_type=strategy.percent_of_equity,  # Position sizing method
    default_qty_value=10,  # Size value (10% of equity)
)
```

### Configuration with TOML Files

One of the powerful features of PyneCore is that strategy settings can be modified without changing the code. When you
run a strategy, PyneCore automatically generates a `.toml` configuration file with the same base name as your strategy
script. For example, if your strategy is `mystrategy.py`, PyneCore will create `mystrategy.toml`.

This TOML file contains all strategy settings and input parameters, allowing you to:

1. Change strategy parameters (like `initial_capital` or `commission_value`)
2. Modify input values
3. Enable/disable specific settings

Example TOML file structure:

```toml
# Indicator / Strategy / Library Settings

[script]
#initial_capital = 10000
#commission_value = 0.1
#pyramiding = 1
# ... every other setting, commented out

# Input Settings

[inputs.fast_length]
# Input metadata, cannot be modified
#   input_type: "int"
#      defval: 9
#       title: "Fast MA Length"
#     minval: 1
# Change here to modify the input value
value = 12
```

Settings are written commented out; uncomment a line and change its value to override it. An input is overridden by
its `value` line. Simply edit this file and run your strategy again - PyneCore will use the updated values without requiring any code
changes. This is extremely useful for:

- Testing different parameter combinations
- Optimizing strategies
- Running the same strategy with different settings for various markets

## Strategy Entry and Exit

### Entering Positions

```python
# Long entry
strategy.entry("Long", strategy.long)

# Short entry
strategy.entry("Short", strategy.short)

# Entry with specific amount
strategy.entry("Long", strategy.long, qty=2)
```

### Exiting Positions

```python
# Close specific entry
strategy.close("Long")

# Close all positions
strategy.close_all()

# Exit with profit/loss targets
strategy.exit("Exit", "Long", profit=100, loss=50)
```

For more detailed information on entries and exits, see the
[strategy reference](/docs/reference/lib/strategy/).

## Position Sizing

PyneCore supports various position sizing methods:

```python
# Fixed size (specific quantity)
@script.strategy("Fixed Size", default_qty_type=strategy.fixed, default_qty_value=1)
# Percentage of equity
@script.strategy("Percent of Equity", default_qty_type=strategy.percent_of_equity, default_qty_value=10)
# Cash amount
@script.strategy("Cash Amount", default_qty_type=strategy.cash, default_qty_value=1000)
```

You can also dynamically size positions within your strategy:

```python
# Dynamic position sizing based on ATR
atr_value = ta.atr(14)
risk_amount = 500  # Risk $500 per trade
qty = risk_amount / atr_value

strategy.entry("Long", strategy.long, qty=qty)
```

## Strategy Properties

PyneCore provides access to various strategy properties through the `strategy` module:

```python
from pynecore.lib import strategy

# Current position info
current_position = strategy.position_size

# Account info
current_equity = strategy.equity

# Trade info
if strategy.opentrades > 0:
    entry_price = strategy.opentrades.entry_price(0)
    entry_time = strategy.opentrades.entry_time(0)
```

## Example Strategies

### Simple Moving Average Crossover

```python
"""
@pyne
"""
from pynecore import Series
from pynecore.lib import script, close, ta, strategy, plot, color, input


@script.strategy("MA Crossover Strategy", overlay=True)
def main(
        fast_length: int = input.int(9, title="Fast MA Length", minval=1),
        slow_length: int = input.int(21, title="Slow MA Length", minval=1)
):
    # Calculate indicators
    fast_ma: Series[float] = ta.sma(close, fast_length)
    slow_ma: Series[float] = ta.sma(close, slow_length)

    # Generate signals
    buy_signal = ta.crossover(fast_ma, slow_ma)
    sell_signal = ta.crossunder(fast_ma, slow_ma)

    # Execute strategy
    if buy_signal:
        strategy.entry("Long", strategy.long)
    elif sell_signal:
        strategy.close("Long")

    # Visualization
    plot(fast_ma, "Fast MA", color=color.blue)
    plot(slow_ma, "Slow MA", color=color.red)
```

### RSI Mean Reversion Strategy

```python
"""
@pyne
"""
from pynecore import Series
from pynecore.lib import script, close, ta, strategy, input


@script.strategy("RSI Mean Reversion", overlay=True)
def main(
        rsi_length: int = input.int(14, title="RSI Length", minval=1),
        overbought: int = input.int(70, title="Overbought Level"),
        oversold: int = input.int(30, title="Oversold Level")
):
    # Calculate RSI
    rsi_value: Series[float] = ta.rsi(close, rsi_length)

    # Generate signals
    buy_signal = ta.crossover(rsi_value, oversold)
    sell_signal = ta.crossunder(rsi_value, overbought)

    # Execute strategy
    if buy_signal:
        strategy.entry("Long", strategy.long)
    elif sell_signal:
        strategy.close("Long")

    # Return RSI values as plot outputs
    return {
        "RSI": rsi_value,
        "Overbought": overbought,
        "Oversold": oversold
    }
```

## Backtesting with PyneCore

PyneCore's strategy backtesting functionality mirrors Pine Script's, providing:

- Performance metrics
- Trade simulation
- Position management
- Commission modeling
- Equity curve calculation

To run a backtest using PyneCore's CLI:

```bash
pyne run mystrategy.py eurusd_daily.ohlcv
```

This executes your strategy on the provided price data and generates the following output files in the `workdir/output/` directory:

- **`<script_name>.csv`** - Plot data (values from `plot()` calls)
- **`<script_name>_strat.csv`** - Strategy statistics (net profit, Sharpe ratio, drawdown, etc.)
- **`<script_name>_trade.csv`** - Trade-by-trade data (entry/exit prices, P&L, cumulative metrics)

For more details on output files and CLI options, see [Running Scripts](./cli/run.md).

Plot and drawing data can also be exported as NDJSON with `--viz`, for your own visualization (see
[Visual Output](/docs/programmatic/visual-output/)).

## Best Practices

### 1. Mind the Look-Ahead Bias and Indexing

Ensure your strategy only uses data that would have been available at the time of the trading decision. In PyneCore,
similar to Pine Script, indices refer to historical data where larger indices indicate older bars:

```python
current = close[0]  # Current bar (or just `close`)
previous = close[1]  # One bar ago
older = close[10]  # Ten bars ago
```

Negative offsets never reach into the future: `close[-1]`, like any offset outside the available history, returns
`na`, as in Pine Script. Slices with negative bounds (`close[-1:5]`) raise an `IndexError`.

### 2. Test Strategy Robustness

- Test on multiple symbols and timeframes
- Vary strategy parameters to avoid over-optimization
- Include different market conditions in test data

### 3. Start Simple

Begin with simpler strategies before adding complexity:

1. Start with basic entry and exit conditions
2. Add position sizing after ensuring the base strategy works
3. Gradually incorporate filters and additional conditions
4. Add risk management logic

## Further Resources

- [PyneCore Core Concepts](/docs/overview/core-concepts/)
- [Strategy Reference](/docs/reference/lib/strategy/) - Complete strategy function reference
- [Strategy Risk Management](/docs/reference/lib/strategy_risk/) - Risk management functions

Beyond plain backtests, PyneCore supports `calc_on_order_fills`, the [bar magnifier](/docs/advanced/bar-magnifier/)
for intrabar fill accuracy, and [live and broker trading](/docs/advanced/live-mode/) with the same strategy code.