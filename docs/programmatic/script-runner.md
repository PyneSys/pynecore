<!--
---
weight: 801
title: "ScriptRunner API"
description: "Running PyneCore scripts programmatically from Python"
icon: "play_circle"
date: "2025-03-31"
lastmod: "2026-03-17"
draft: false
toc: true
categories: ["Programmatic", "API"]
tags: ["script-runner", "run-iter", "indicators", "strategies", "trades"]
---
-->

# ScriptRunner API

`ScriptRunner` is the core class for running PyneCore scripts from Python code. It processes OHLCV
data bar-by-bar through a compiled Pine Script, yielding indicator values and trade results as they
happen.

## Quick Start

```python
from pathlib import Path
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.syminfo import SymInfo
from pynecore.types.ohlcv import OHLCV

# Create data (see Data & SymInfo page for more options)
syminfo = SymInfo(
    prefix="BINANCE", ticker="BTCUSD", currency="USD", basecurrency="BTC",
    description="Bitcoin", period="60", type="crypto",
    mintick=0.01, pricescale=100, minmove=1, pointvalue=1.0,
    timezone="UTC", volumetype="base",
    opening_hours=[], session_starts=[], session_ends=[],
)

candles = [
    OHLCV(timestamp=1704067200, open=42000, high=42500, low=41800, close=42300, volume=1000),
    OHLCV(timestamp=1704070800, open=42300, high=42800, low=42100, close=42600, volume=1200),
    # ... more bars
]

# Run an indicator
runner = ScriptRunner(
    script_path=Path("my_indicator.py"),
    ohlcv_iter=candles,
    syminfo=syminfo,
)

for candle, plot_data in runner.run_iter():
    rsi = plot_data.get("RSI")
    print(f"Close={candle.close:.2f}  RSI={rsi}")
```

## Constructor

```python
ScriptRunner(
    script_path: Path,
    ohlcv_iter: Iterable[OHLCV],
    syminfo: SymInfo,
    *,
    plot_path: Path | None = None,
    strat_path: Path | None = None,
    trade_path: Path | None = None,
    update_syminfo_every_run: bool = False,
    last_bar_index: int = 0,
    inputs: dict[str, Any] | None = None,
    security_data: dict[str, str | Path] | None = None,
    magnifier_iter: Iterable[OHLCV] | None = None,
)
```

### Parameters

| Parameter                  | Type              | Description                                                    |
|----------------------------|-------------------|----------------------------------------------------------------|
| `script_path`              | `Path`            | Path to a compiled PyneCore script (`.py` with `@pyne` marker) |
| `ohlcv_iter`               | `Iterable[OHLCV]` | Any iterable of OHLCV objects — list, generator, reader, etc.  |
| `syminfo`                  | `SymInfo`         | Symbol information (from TOML or manually created)             |
| `plot_path`                | `Path \| None`    | Save indicator plot data to CSV                                |
| `strat_path`               | `Path \| None`    | Save strategy statistics to CSV                                |
| `trade_path`               | `Path \| None`    | Save trade-by-trade data to CSV                                |
| `update_syminfo_every_run` | `bool`            | Re-apply syminfo before each bar (for parallel runners)        |
| `last_bar_index`           | `int`             | Override last bar index (for multi-script setups)              |
| `inputs`                   | `dict \| None`    | Override script `input()` defaults at runtime                  |
| `security_data`            | `dict \| None`    | OHLCV paths for `request.security()` contexts (see below)      |
| `magnifier_iter`           | `Iterable \| None` | LTF OHLCV bars for bar magnifier mode (intrabar simulation)    |

### Overriding Inputs

The `inputs` parameter lets you change script parameters without editing the script file:

```python
runner = ScriptRunner(
    script_path=Path("sma_crossover.py"),
    ohlcv_iter=candles,
    syminfo=syminfo,
    inputs={"Length": 20, "Confirm bars": 3},  # override input() defaults
)
```

Keys must match the `title` parameter of `input()` calls in the script. If a key doesn't match
any input, it's silently ignored.

### Providing Security Data

If your script uses `request.security()` to fetch data from other symbols or timeframes, you must
provide the OHLCV data files for each security context via the `security_data` parameter.

Keys can be in two formats:

- **`"TIMEFRAME"`** — matches any security context with that timeframe (e.g., `"1D"`, `"1W"`)
- **`"SYMBOL:TIMEFRAME"`** — matches a specific symbol and timeframe (e.g., `"AAPL:1H"`)

Values are paths to `.ohlcv` data files (with corresponding `.toml` syminfo files in the same
directory).

```python
# Script that uses request.security() for daily data
runner = ScriptRunner(
    script_path=Path("multi_tf_indicator.py"),
    ohlcv_iter=candles_5m,  # chart data: 5-minute bars
    syminfo=syminfo,
    security_data={
        "1D": "data/EURUSD_1D",  # daily bars for same symbol
    },
)

# Script that fetches data from multiple symbols
runner = ScriptRunner(
    script_path=Path("advance_decline.py"),
    ohlcv_iter=candles,
    syminfo=syminfo,
    security_data={
        "USI:ADVN.NY": "data/USI_ADVN_NY",  # advancing issues
        "USI:DECL.NY": "data/USI_DECL_NY",  # declining issues
    },
)
```

Each OHLCV path should point to a directory base name (without extension). The system expects
both `<path>.ohlcv` (binary data) and `<path>.toml` (symbol info) to exist.

> **Note:** Security contexts spawn separate OS processes. Each process re-imports the script,
> loads its own OHLCV data, and builds Series history from bar 0. For technical details, see
> [request.security() Internals](../advanced/request-security-internals.md).

## run_iter() — Processing Bars

The primary method. Returns an iterator that yields results for each bar processed.

### Indicators

Indicators yield a 2-tuple: `(candle, plot_data)`.

```python
for candle, plot_data in runner.run_iter():
    # candle: the OHLCV object for this bar
    # plot_data: dict of values from plot() calls in the script

    rsi = plot_data.get("RSI")  # float, or NA during warmup
    basis = plot_data.get("Basis")  # keys match plot() title parameter
```

### Strategies

Strategies yield a 3-tuple: `(candle, plot_data, new_trades)`.

```python
for candle, plot_data, new_trades in runner.run_iter():
    # candle: the OHLCV object for this bar
    # plot_data: dict of plotted values
    # new_trades: list of trades that CLOSED on this bar

    for trade in new_trades:
        direction = "LONG" if trade.size > 0 else "SHORT"
        print(f"{direction}  P&L={trade.profit:+.2f}")
```

> **Note:** `new_trades` contains only trades that **closed** on the current bar, not open
> positions. Each trade appears exactly once — on the bar where it exits.

### NA Values During Warmup

During the warmup period (first N bars where the indicator doesn't have enough data), plot values
are `NA` objects — PyneCore's equivalent of Pine Script's `na`.

`NA` works transparently — no special handling needed:

- **Comparisons** return `False`: `NA < 30`, `NA > 70`, `NA == x` → all `False`
- **Arithmetic** propagates: `NA + 1` → `NA`, `NA * 2.0` → `NA`
- **Format strings** work: `f"{na_value:.2f}"` → `"NaN"`

```python
for candle, plot_data in runner.run_iter():
    rsi = plot_data.get("RSI")

    if rsi > 70:  # False when rsi is NA — no crash, no special check needed
        print(f"Overbought: RSI={rsi:.2f}")

    # NA values print as "NaN" in f-strings
    print(f"RSI={rsi:.2f}")  # "RSI=NaN" during warmup, "RSI=65.32" after
```

## Trade Object

Trades returned by strategies have the following fields:

| Field                  | Type  | Description                              |
|------------------------|-------|------------------------------------------|
| `size`                 | float | Quantity (positive=long, negative=short) |
| `entry_id`             | str   | ID from `strategy.entry()` call          |
| `entry_bar_index`      | int   | Bar index where entry filled             |
| `entry_time`           | int   | Entry timestamp (milliseconds)           |
| `entry_price`          | float | Fill price for entry                     |
| `entry_comment`        | str   | Comment from `strategy.entry()`          |
| `exit_id`              | str   | ID from exit call                        |
| `exit_bar_index`       | int   | Bar index where exit filled              |
| `exit_time`            | int   | Exit timestamp (milliseconds)            |
| `exit_price`           | float | Fill price for exit                      |
| `profit`               | float | Absolute P&L in account currency         |
| `profit_percent`       | float | P&L as percentage                        |
| `cum_profit`           | float | Cumulative P&L up to this trade          |
| `cum_profit_percent`   | float | Cumulative P&L %                         |
| `max_runup`            | float | Max unrealized profit during trade       |
| `max_runup_percent`    | float | Max runup %                              |
| `max_drawdown`         | float | Max unrealized loss during trade         |
| `max_drawdown_percent` | float | Max drawdown %                           |
| `commission`           | float | Fees paid                                |

## Saving Output to CSV

You can write results to CSV files (same format as the CLI `pyne run` command):

```python
runner = ScriptRunner(
    script_path=Path("my_strategy.py"),
    ohlcv_iter=candles,
    syminfo=syminfo,
    plot_path=Path("output/plot.csv"),  # indicator values per bar
    strat_path=Path("output/stats.csv"),  # strategy statistics summary
    trade_path=Path("output/trades.csv"),  # trade-by-trade details
)

# Must exhaust the iterator for files to be written
for candle, plot_data, new_trades in runner.run_iter():
    pass  # files are written as bars are processed
```

## Complete Example: Strategy with Trade Analysis

```python
from pathlib import Path
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.data_converter import DataConverter
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.syminfo import SymInfo

# Convert CSV data to OHLCV format
csv_path = Path("data/EURUSD_1h.csv")
DataConverter().convert_to_ohlcv(csv_path)

# Load converted data
ohlcv_path = csv_path.with_suffix(".ohlcv")
toml_path = csv_path.with_suffix(".toml")
syminfo = SymInfo.load_toml(toml_path)

with OHLCVReader(ohlcv_path) as reader:
    runner = ScriptRunner(
        script_path=Path("sma_crossover.py"),
        ohlcv_iter=reader.read_from(reader.start_timestamp, reader.end_timestamp),
        syminfo=syminfo,
        inputs={"Length": 20, "Confirm bars": 2},
    )

    all_trades = []
    for candle, plot_data, new_trades in runner.run_iter():
        all_trades.extend(new_trades)

# Analyze results
if all_trades:
    wins = [t for t in all_trades if t.profit > 0]
    total_pnl = sum(t.profit for t in all_trades)
    print(f"Trades: {len(all_trades)}  Win rate: {len(wins) / len(all_trades) * 100:.1f}%  P&L: {total_pnl:+.2f}")
```
