<!--
---
weight: 803
title: "Integration Patterns"
description: "Real-world patterns for integrating PyneCore into trading systems"
icon: "hub"
date: "2025-03-31"
lastmod: "2026-03-17"
draft: false
toc: true
categories: ["Programmatic", "Integration"]
tags: ["freqtrade", "live-trading", "integration", "pandas", "dataframe"]
---
-->

# Integration Patterns

PyneCore can be embedded into any Python application. This page covers common integration patterns
with real-world trading frameworks.

## DataFrame Bridge (FreqTrade, Backtrader, etc.)

Many trading frameworks work with pandas DataFrames. The bridge pattern converts between DataFrames
and PyneCore's OHLCV format:

```python
import pandas as pd
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.syminfo import SymInfo
from pynecore.types.ohlcv import OHLCV


def dataframe_to_ohlcv(df: pd.DataFrame) -> list[OHLCV]:
    """Convert a pandas DataFrame to a list of OHLCV objects."""
    return [
        OHLCV(
            timestamp=int(row.Index.timestamp()),
            open=float(row.open), high=float(row.high),
            low=float(row.low), close=float(row.close),
            volume=float(row.volume),
        )
        for row in df.itertuples()
    ]


def run_indicator(df, script_path, syminfo, inputs=None):
    """Run a PyneCore indicator, return results as dict of pd.Series."""
    runner = ScriptRunner(
        script_path=script_path,
        ohlcv_iter=dataframe_to_ohlcv(df),
        syminfo=syminfo,
        inputs=inputs,
    )

    results = {}
    for _candle, plot_data in runner.run_iter():
        for key, value in plot_data.items():
            results.setdefault(key, []).append(value)

    return {
        key: pd.Series(values, index=df.index[:len(values)])
        for key, values in results.items()
    }
```

Usage:

```python
rsi_data = run_indicator(dataframe, Path("rsi.py"), syminfo)
dataframe["rsi"] = rsi_data["RSI"]
```

## FreqTrade Integration

PyneCore integrates with FreqTrade in two ways:

### Pattern 1: Indicators as Data Sources

Use Pine Script indicators for calculations, write entry/exit logic in Python:

```python
class PyneIndicatorStrategy(IStrategy):
    def populate_indicators(self, dataframe, metadata):
        pair = metadata.get("pair", "BTC/USDT")
        rsi = run_indicator(dataframe, Path("scripts/rsi.py"), syminfo)
        dataframe["rsi"] = rsi.get("RSI")
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe.loc[dataframe["rsi"] < 30, "enter_long"] = 1
        return dataframe
```

### Pattern 2: Strategy Signals

Let a Pine Script strategy generate buy/sell signals, FreqTrade just executes them:

```python
class PyneStrategySignals(IStrategy):
    def populate_indicators(self, dataframe, metadata):
        _indicators, trades = run_strategy(dataframe, Path("scripts/sma_cross.py"), syminfo)

        dataframe["pyne_enter_long"] = 0
        for trade in trades:
            if trade.size > 0 and trade.entry_bar_index < len(dataframe):
                dataframe.iloc[trade.entry_bar_index,
                               dataframe.columns.get_loc("pyne_enter_long")] = 1
        return dataframe
```

> For complete, runnable FreqTrade examples, see
> [pynecore-examples/05-freqtrade-indicators](https://github.com/PyneSys/pynecore-examples/tree/main/05-freqtrade-indicators)
> and
> [pynecore-examples/06-freqtrade-strategy](https://github.com/PyneSys/pynecore-examples/tree/main/06-freqtrade-strategy).

## Live Data Feed

Process bars as they arrive from an exchange:

```python
import ccxt
import time
from pynecore.types.ohlcv import OHLCV

exchange = ccxt.binance({"enableRateLimit": True})


def live_candles(symbol, timeframe, warmup=200):
    """Yield OHLCV bars: historical warmup first, then poll for new bars."""
    # Warmup: fetch historical bars
    raw = exchange.fetch_ohlcv(symbol, timeframe, limit=warmup)
    for bar in raw:
        yield OHLCV(
            timestamp=bar[0] // 1000,
            open=bar[1], high=bar[2], low=bar[3], close=bar[4], volume=bar[5],
        )

    # Live: poll for new bars
    last_ts = raw[-1][0]
    while True:
        time.sleep(10)
        raw = exchange.fetch_ohlcv(symbol, timeframe, since=last_ts, limit=5)
        for bar in raw:
            if bar[0] > last_ts:
                last_ts = bar[0]
                yield OHLCV(
                    timestamp=bar[0] // 1000,
                    open=bar[1], high=bar[2], low=bar[3], close=bar[4], volume=bar[5],
                )


# Use with ScriptRunner
runner = ScriptRunner(
    script_path=Path("my_indicator.py"),
    ohlcv_iter=live_candles("BTC/USDT", "1h"),
    syminfo=syminfo,
)

for candle, plot_data in runner.run_iter():
    rsi = plot_data.get("RSI")
    if rsi < 30:  # NA values return False for all comparisons — no special handling needed
        print(f"RSI oversold: {rsi:.2f} at {candle.close}")
```

> For a complete live CCXT example, see
> [pynecore-examples/04-live-ccxt](https://github.com/PyneSys/pynecore-examples/tree/main/04-live-ccxt).

## Multiple Indicators on Same Data

Run several scripts on the same data efficiently:

```python
from pynecore.core.script_runner import ScriptRunner

data = list(ohlcv_source)  # Materialize once if it's a generator

# Run each indicator separately
rsi_runner = ScriptRunner(script_path=Path("rsi.py"), ohlcv_iter=data, syminfo=syminfo)
bb_runner = ScriptRunner(script_path=Path("bollinger.py"), ohlcv_iter=data, syminfo=syminfo)

rsi_values = [plot.get("RSI") for _, plot in rsi_runner.run_iter()]
bb_values = [(plot.get("Upper"), plot.get("Lower")) for _, plot in bb_runner.run_iter()]
```

## Parameter Optimization

Sweep over input combinations to find optimal parameters:

```python
from itertools import product

results = []
for length, confirm in product(range(5, 30, 5), range(1, 4)):
    runner = ScriptRunner(
        script_path=Path("sma_crossover.py"),
        ohlcv_iter=data,
        syminfo=syminfo,
        inputs={"Length": length, "Confirm bars": confirm},
    )

    trades = []
    for _, _, new_trades in runner.run_iter():
        trades.extend(new_trades)

    if trades:
        pnl = sum(t.profit for t in trades)
        results.append({"length": length, "confirm": confirm, "pnl": pnl, "trades": len(trades)})

# Find best combination
best = max(results, key=lambda r: r["pnl"])
print(f"Best: Length={best['length']}, Confirm={best['confirm']} → P&L={best['pnl']:+.2f}")
```

## Performance Tips

- **Materialize generators**: If running multiple scripts on the same data, convert the OHLCV
  iterator to a list first (`list(reader.read_from(...))`) to avoid re-reading from disk.

- **Cache in live systems**: When integrating with frameworks that re-call your indicator function
  on every new bar (like FreqTrade), cache computed values keyed by timestamp. Only run PyneCore
  on new bars — return cached values for bars you've already processed.

- **Batch processing**: PyneCore processes ~16,000 bars/second. For most use cases (hourly/daily
  data), re-running from scratch is fast enough that caching isn't necessary.

## Complete Examples

For full, runnable examples covering all these patterns, see the
[pynecore-examples](https://github.com/PyneSys/pynecore-examples) repository.
