<!--
---
weight: 506
title: "request.security()"
description: "Using request.security() for multi-symbol and multi-timeframe data in PyneCore"
icon: "security"
date: "2026-03-27"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Library", "API Reference"]
tags: ["request-security", "multi-symbol", "multi-timeframe"]
---
-->

# request.security()

`request.security()` lets you access data from other symbols or timeframes within your script.
PyneCore runs each security context as a separate OS process with its own Series history,
enabling true multi-symbol and multi-timeframe analysis.

## Quick Start

### 1. Prepare OHLCV Data

Each security context needs its own OHLCV data file. Convert your data sources to PyneCore's
binary format using the CLI:

```bash
# Chart data (5-minute EURUSD)
pyne data convert csv EURUSD_5m.csv

# Security data (daily EURUSD for HTF analysis)
pyne data convert csv EURUSD_1D.csv

# Or aggregate from existing data
pyne data aggregate EURUSD_5m -tf 1D
```

This creates `.ohlcv` + `.toml` file pairs in `workdir/data/`.

### 2. Write Your Script

Use `request.security()` like you would in Pine Script:

```python
"""@pyne"""

from pynecore.lib import *
from pynecore.types import *


@script.indicator("Multi-Timeframe SMA")
def main():
    # Fetch daily SMA(20) while running on 5-minute chart
    daily_sma: Series[float] = lib.request.security(
        lib.syminfo.tickerid, "1D", lib.ta.sma(lib.close, 20)
    )

    lib.plot.plot(daily_sma, "Daily SMA", color=lib.color.blue)
    lib.plot.plot(lib.close, "Close")
```

### 3. Run with Security Data

Use the `--security` flag to provide OHLCV data for each security context. The flag can be
repeated for multiple contexts:

```bash
# Single security context (daily data for same symbol)
pyne run multi_tf_sma.py EURUSD_5m --security "1D=EURUSD_1D"

# Multiple security contexts (different symbols)
pyne run advance_decline.py SPX_1D \
  --security "USI:ADVN.NY=USI_ADVN_NY" \
  --security "USI:DECL.NY=USI_DECL_NY"
```

The format is `"KEY=DATA_NAME"` where:

- **KEY** is `"TIMEFRAME"` or `"SYMBOL:TIMEFRAME"` (matching the `request.security()` call)
- **DATA_NAME** is the OHLCV data name in `workdir/data/` (without extension)

### Key Matching Rules

The `security_data` dict keys are matched against each `request.security()` call's symbol and
timeframe:

| Key format           | Example         | Matches                                  |
|----------------------|-----------------|------------------------------------------|
| `"TIMEFRAME"`        | `"1D"`          | Any security call with timeframe `"1D"`  |
| `"SYMBOL:TIMEFRAME"` | `"AAPL:1H"`     | Exact match on both symbol and timeframe |
| `"SYMBOL"`           | `"USI:ADVN.NY"` | Any security call with that symbol       |

Timeframe-only keys are convenient when all security calls use the same symbol (the chart symbol).

> For programmatic usage (ScriptRunner API), see
> [Providing Security Data](../programmatic/script-runner.md#providing-security-data).

## Examples

### Multi-Timeframe Indicator

```python
"""@pyne"""

from pynecore.lib import *
from pynecore.types import *


@script.indicator("MTF RSI")
def main():
    rsi_5m: Series[float] = lib.ta.rsi(lib.close, 14)

    # Get RSI from higher timeframes
    rsi_1h: Series[float] = lib.request.security(
        lib.syminfo.tickerid, "60", lib.ta.rsi(lib.close, 14)
    )
    rsi_daily: Series[float] = lib.request.security(
        lib.syminfo.tickerid, "1D", lib.ta.rsi(lib.close, 14)
    )

    lib.plot.plot(rsi_5m, "RSI 5m")
    lib.plot.plot(rsi_1h, "RSI 1H")
    lib.plot.plot(rsi_daily, "RSI Daily")
```

```python
security_data = {
    "60": "workdir/data/EURUSD_60",  # 1-hour bars
    "1D": "workdir/data/EURUSD_1D",  # daily bars
}
```

### Multi-Symbol Analysis (Advance/Decline Ratio)

```python
"""@pyne"""

from pynecore.lib import *
from pynecore.types import *


@script.indicator("Advance/Decline Ratio")
def main():
    advancing: Series[float] = lib.request.security("USI:ADVN.NY", "", lib.close)
    declining: Series[float] = lib.request.security("USI:DECL.NY", "", lib.close)

    ratio: Series[float] = lib.nz(advancing) / lib.nz(declining, 1.0)
    lib.plot.plot(ratio, "A/D Ratio")
```

```python
security_data = {
    "USI:ADVN.NY": "workdir/data/USI_ADVN_NY",
    "USI:DECL.NY": "workdir/data/USI_DECL_NY",
}
```

> **Note:** When the timeframe argument is `""` (empty string), the chart's own timeframe is used.

## Supported Features

| Feature                 | Status        | Notes                                             |
|-------------------------|---------------|---------------------------------------------------|
| Different timeframe     | supported     | HTF (1D, 1W, 1M, etc.) from lower TF chart        |
| Different symbol        | supported     | Any symbol with available OHLCV data              |
| Lower timeframe (LTF)   | supported     | `request.security_lower_tf()` returns arrays      |
| Multiple security calls | supported     | Each gets its own process                         |
| Conditional calls       | supported     | Inside `if`/`for`/`while` blocks                  |
| Nested security calls   | supported     | `security(... security(...) ...)`                 |
| `barmerge.gaps_off`     | supported     | Forward-fills last value (default)                |
| `barmerge.gaps_on`      | supported     | Returns `na` between periods                      |
| `lookahead_off`         | supported     | Confirmed previous period (default)               |
| `ignore_invalid_symbol` | supported     | Returns `na` for missing symbols                  |
| `lookahead_on`          | not supported | Deliberate safety-first decision                  |
| `currency` parameter    | supported     | Auto-converts result using `CurrencyRateProvider` |

## How It Works

Under the hood, each `request.security()` call spawns a separate OS process:

1. **AST transformation** rewrites the call into signal/write/read/wait protocol functions
2. **ScriptRunner** detects the transformed code, creates shared memory, and spawns processes
3. Each security **process** loads its own OHLCV data and runs the script independently
4. Processes communicate results via **shared memory**
5. The chart process **waits** for security results only when a new period is confirmed
6. **Pipeline parallelism**: security processes run on separate CPU cores concurrently

### HTF Period Confirmation

For higher-timeframe data, values are confirmed with **lookahead_off** semantics: a daily value
becomes available only when the next daily bar opens (i.e., when the period boundary is crossed).

```
Chart bars (5m):    10:00  10:05  10:10 ... 23:55  00:00  00:05
                                                    ^
                                              New daily period starts
                                              Yesterday's daily value is now confirmed
```

For same-timeframe contexts (different symbol), values are confirmed on every bar.

### gaps_on vs gaps_off

| Mode       | New period confirmed | Between periods                  |
|------------|----------------------|----------------------------------|
| `gaps_off` | Return new value     | Return last value (forward-fill) |
| `gaps_on`  | Return new value     | Return `na`                      |

## Limitations

- **lookahead_on** — not supported. Only `lookahead_off` (confirmed previous period) semantics are
  available. `lookahead_on` gives the script access to the *current* (unconfirmed) higher-timeframe
  value before the period closes. This has legitimate uses in **retrospective market analysis** —
  for example, examining how intraday price action related to the final daily close. However, in
  backtesting it effectively leaks future data into past decisions, producing inflated results that
  cannot be replicated in live trading. Since PyneCore is designed primarily for **backtesting and
  forward-looking strategy evaluation** rather than retrospective charting, supporting `lookahead_on`
  would undermine the reliability of its results.
- **Standalone mode** — `python script.py data.csv` does not support `--security` yet.
  Use `pyne run` or the ScriptRunner API.

## Known Differences from TradingView

On markets with **shortened trading sessions** (e.g., half-day sessions before holidays), minor
differences may occur when the chart symbol and the security symbol follow different session
calendars — one closes early while the other trades a full day. This can cause period boundary
alignment to differ slightly from TradingView. In practice, this is rare and only affects a handful
of bars on specific calendar dates. PyneCore's handling of these cases appears to produce more
consistent results, though TradingView's behavior on irregular sessions may follow internal calendar
rules that are not publicly documented.

> For technical implementation details (AST transformation, shared memory layout, process lifecycle),
> see the [request.security() Internals](../advanced/request-security-internals.md) page.
