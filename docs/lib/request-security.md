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
pyne data convert-from EURUSD_5m.csv

# Security data (daily EURUSD for HTF analysis)
pyne data convert-from EURUSD_1D.csv

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

| Feature                          | Status    | Notes                                                                            |
|----------------------------------|-----------|----------------------------------------------------------------------------------|
| Different timeframe              | supported | HTF (1D, 1W, 1M, etc.) from lower TF chart                                       |
| Different symbol                 | supported | Any symbol with available OHLCV data                                             |
| Lower timeframe (LTF)            | supported | `request.security_lower_tf()` returns arrays                                     |
| Multiple security calls          | supported | Each gets its own process                                                        |
| Conditional calls                | supported | Inside `if`/`for`/`while` blocks                                                 |
| Nested security calls            | supported | `security(... security(...) ...)`                                                |
| `barmerge.gaps_off`              | supported | Forward-fills last value (default)                                               |
| `barmerge.gaps_on`               | supported | Returns `na` between periods                                                     |
| `barmerge.lookahead_off`         | supported | Most recently closed bar — historical + live (same-symbol HTF in live)           |
| `barmerge.lookahead_last_closed` | supported | PyneSys-native synonym for "last closed"; identical transport to `lookahead_off` |
| `barmerge.lookahead_on`          | supported | Same-symbol HTF: TV-compatible (live steps into developing HTF bar). Cross-symbol HTF: `close[0]` is `na` inside an open HTF period, `close[1]` at boundary delivers just-closed bar |
| `ignore_invalid_symbol`          | supported | Returns `na` for missing symbols                                                 |
| `currency` parameter             | supported | Auto-converts result using `CurrencyRateProvider`                                |

## How It Works

Under the hood, each `request.security()` call spawns a separate OS process:

1. **AST transformation** rewrites the call into signal/write/read/wait protocol functions
2. **ScriptRunner** detects the transformed code, creates shared memory, and spawns processes
3. Each security **process** loads its own OHLCV data and runs the script independently
4. Processes communicate results via **shared memory**
5. The chart process **waits** for security results only when a new period is confirmed
6. **Pipeline parallelism**: security processes run on separate CPU cores concurrently

### HTF Period Confirmation

For higher-timeframe data, values are confirmed with **lookahead_off** semantics following
TradingView's historical merge rule: an HTF bar is confirmed on the chart bar whose **close
instant** reaches the HTF bar's close — the period's last chart bar already carries the
period's final value, not the next period's first bar.

```
Chart bars (5m):    10:00  10:05  10:10 ... 23:50  23:55  00:00
                                                    ^
                                              Closes at 00:00 — exactly when the
                                              daily bar closes, so the day's value
                                              is confirmed on THIS chart bar
```

For same-timeframe contexts (different symbol), values are confirmed on every bar.

### gaps_on vs gaps_off

| Mode       | New period confirmed | Between periods                  |
|------------|----------------------|----------------------------------|
| `gaps_off` | Return new value     | Return last value (forward-fill) |
| `gaps_on`  | Return new value     | Return `na`                      |

### Lookahead modes

`request.security()` accepts a `lookahead` argument. Three modes are recognized:

| Mode                               | Behavior (historical / backtest)                                                                  | Behavior (live mode, same-symbol HTF)                                                                                                                                                                                                                                                                                                 |
|------------------------------------|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `barmerge.lookahead_off` (default) | Most recently CLOSED security bar (TV-faithful)                                                   | Most recently CLOSED security bar; each HTF period close is shipped via the chart-side `HTFAggregator` (the static `.ohlcv` file cannot grow at runtime). No developing exposure.                                                                                                                                                     |
| `barmerge.lookahead_last_closed`   | Most recently CLOSED security bar (functionally equivalent to `lookahead_off`)                    | Most recently CLOSED security bar — uses the same closed-bar transport as `lookahead_off`; repaint-free.                                                                                                                                                                                                                              |
| `barmerge.lookahead_on`            | Most recently CLOSED security bar — historical falls back to `lookahead_off` to avoid future-leak | Developing (containing) HTF bar with `barstate.isconfirmed=False`; OHLCV is aggregated from the chart timeframe by `HTFAggregator`. On HTF period close the just-closed bar is delivered first (so the security `bar_index` advances), then the new developing bar. TV-compatible `close[1]` idiom returns the previously closed bar. |

**Why `lookahead_last_closed`** — in historical backtests it matches `lookahead_off`, and in
live mode it stays repaint-free (it never shows the developing security bar). Prefer it when
you want stable last-closed values without depending on the TV `close[1]` idiom.

**Why historical `lookahead_on` is degraded to `lookahead_off`** — TV's classical historical
`lookahead_on` behavior is to expose the containing HTF bar's close on every chart bar,
producing a future-leak that silently inflates backtest results. PyneCore deliberately does
not reproduce this leak: in historical mode the closed bar is the only bar the subprocess
ever runs. The TV idiom `request.security(..., lookahead_on)[1]` still returns the correct
last-closed value (since `close[1]` is always the previously closed bar). In live mode the
subprocess additionally steps into the developing bar so `close[0]` exposes the in-progress
close as TV does.

**Cross-symbol HTF** — the chart-side `HTFAggregator` aggregates the *chart* symbol's
OHLCV; it cannot produce OHLCV for a different security symbol. Cross-symbol HTF
contexts therefore keep no aggregator and read closed bars directly from the
security's own `.ohlcv` (historical) or live feed.

* `lookahead_off` / `lookahead_last_closed` — closed-bar semantics, identical to
  same-symbol behaviour. Repaint-free.
* `lookahead_on` — the containing developing bar is **unknown** (cannot be aggregated
  from the wrong instrument). Chart-side `__sec_read__` returns `na` for every chart
  bar inside an open HTF period; the subprocess still advances on closed cross-symbol
  HTF bars, so `close[1]` on the first chart bar of a fresh HTF period delivers the
  just-closed cross-symbol HTF close. The TV `lookahead_on + close[1]` idiom continues
  to work at the period boundary. Behaviour is identical in historical and live mode —
  the backtest never silently emits a value live could not produce.

```python
# Default — most recently closed bar
htf_close = lib.request.security(lib.syminfo.tickerid, "60", lib.close)

# PyneSys-native — repaint-free in both historical and live mode
htf_close = lib.request.security(
    lib.syminfo.tickerid, "60", lib.close,
    lookahead=lib.barmerge.lookahead_last_closed,
)

# TV-compatible — live mode exposes the developing bar; the close[1] idiom
# is the canonical TV way to read the most recently closed HTF bar.
htf_close_prev = lib.request.security(
    lib.syminfo.tickerid, "60", lib.close[1],
    lookahead=lib.barmerge.lookahead_on,
)
```

## Limitations

- **Cross-symbol HTF** — the live HTF transport aggregates chart OHLCV, so it
  works only when the security symbol matches the chart symbol. Cross-symbol HTF
  `lookahead_off` / `lookahead_last_closed` deliver closed bars from the security's own
  `.ohlcv` / live feed. Cross-symbol HTF `lookahead_on` returns `na` for the current
  chart bar inside any open HTF period (the developing bar cannot be aggregated
  cross-instrument); `close[1]` at the period boundary still delivers the just-closed
  cross-symbol HTF close, preserving the TV idiom. For continuous developing-bar
  coverage of a cross-symbol HTF, run that context as a separate same-symbol chart.
- **Standalone mode** — `python script.py data.csv` does not support `--security` yet.
  Use `pyne run` or the ScriptRunner API.

## Debugging Security Contexts

`log.info()`, `log.warning()`, and `log.error()` calls inside security processes are suppressed by
default (matching TradingView behavior). To enable logging for debugging, set the
`PYNE_SECURITY_LOG` environment variable:

```bash
PYNE_SECURITY_LOG=security.log pyne run my_script.py my_data --security "1D=my_data_1D"
```

Each line is prefixed with the security context identifier:

```
[AAPL 1D] [2025-07-05 14:30:00-0400] bar:    42 INFO    Daily SMA: 150.25
[EURUSD 1H] [2025-07-05 14:00:00+0000] bar:   100 INFO    RSI: 72.5
```

> See [Debugging](../debugging.md#debugging-security-contexts) for more details.

## Session anchoring of intraday higher timeframes

For intraday higher timeframes (minutes and hours), PyneCore anchors the HTF bar grid to the
**session open**, matching TradingView. On a market that opens at 09:30, a `1H` security therefore
produces bars at 09:30, 10:30, 11:30… rather than 09:00, 10:00, 11:00. The grid steps in real time
from the open, so it stays correct across daylight-saving transitions, and the alignment is derived
from the symbol's own session — a cross-symbol security in a different exchange session anchors to
*its* open, not the chart's. Markets that open on a whole-`tf` boundary (24/7 crypto, on-hour forex
and futures) are unaffected: the session-anchored grid is identical to a plain clock-floor there.

> The session open is read from the symbol's `session_starts` metadata. If a security's session
> information is missing, intraday bars fall back to the UTC clock-floor for that symbol.

## Known Differences from TradingView

On markets with **shortened trading sessions** (e.g., half-day sessions before holidays), minor
differences may occur when the chart symbol and the security symbol follow different session
calendars — one closes early while the other trades a full day. This can cause period boundary
alignment to differ slightly from TradingView. In practice, this is rare and only affects a handful
of bars on specific calendar dates.

Markets with an **intraday recess** (e.g. a lunch break) keep a single grid anchored to the day's
primary session open; the bar whose window spans the recess simply holds the data of the first
trade after it, as on TradingView for the common case.

> For technical implementation details (AST transformation, shared memory layout, process lifecycle),
> see the [request.security() Internals](../advanced/request-security-internals.md) page.
