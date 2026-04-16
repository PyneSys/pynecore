<!--
---
weight: 1070
title: "Live Mode"
description: "Real-time data streaming with intra-bar updates, varip support, and paper trading"
icon: "stream"
date: "2026-04-08"
lastmod: "2026-04-08"
draft: false
toc: true
categories: ["Advanced", "Strategy", "Live"]
tags: ["live", "streaming", "intra-bar", "varip", "paper-trading", "real-time"]
---
-->

# Live Mode

Live mode extends PyneCore beyond backtesting: after replaying historical data the script
seamlessly transitions to real-time streaming from a `LiveProviderPlugin`.  Indicators update
on every tick; strategies run in paper-trading mode with tick-level order fill accuracy.

## Quick Start

```bash
# Stream BTC/USDT on 1-minute bars from Bybit via CCXT, prefetching 500 historical bars
pyne run my_strategy.py ccxt:BYBIT:BTC/USDT:USDT@1 --live -f -500
```

The `--live` flag requires a **provider string** as the data source — it does not work with
local OHLCV files.

## Historical Bar Count

The `-f` / `--from` parameter controls how many historical bars the script processes before
going live.  **Default: 500 bars** — enough for most scripts out of the box.

Indicators need a warm-up period: `ta.sma(close, 200)` requires 200 bars before producing its
first value, and the initial values are still distorted by limited lookback.  A good rule of
thumb is **2× the largest `length` parameter** in your script:

```bash
# Script uses ta.ema(close, 50) and ta.atr(14) → largest length is 50 → -f -100 is enough
pyne run my_strategy.py ccxt:BYBIT:BTC/USDT:USDT@1 --live -f -100

# Script uses ta.sma(close, 200) → use -f -400
pyne run my_strategy.py ccxt:BYBIT:BTC/USDT:USDT@1 --live -f -400

# No -f specified → default 500 bars, sufficient for most scripts
pyne run my_strategy.py ccxt:BYBIT:BTC/USDT:USDT@1 --live
```

Too few bars → indicators produce `NaN` or unreliable values, leading to missed or false
signals.  Too many → longer startup, but no harm beyond that.  When in doubt, err on the side
of more.

## How It Works

A live session has two phases:

| Phase      | Data source        | `barstate.ishistory` | `barstate.isrealtime` | Strategy |
|------------|--------------------|----------------------|-----------------------|----------|
| Historical | Provider download   | `True`               | `False`               | Suppressed |
| Live       | WebSocket streaming | `False`              | `True`                | Active   |

### Historical Phase

The provider downloads OHLCV data (controlled by `-f` / `--from`).  The script runs on each
bar exactly like a normal backtest — indicators build up their series, `ta.sma()` warms up,
etc.  **Strategy functions are suppressed**: calls to `strategy.entry()`, `strategy.exit()`,
and friends are silently ignored.  This prevents phantom trades on historical bars that the
script sees for the first time.

### Transition

The ScriptRunner detects the transition automatically when the iterator yields its first
`BarUpdate` object (instead of a plain `OHLCV`).  At this point:

- `barstate.islastconfirmedhistory` becomes `True` on the final historical bar
- Output writers flush to disk (plot CSV, trade CSV)
- Strategy suppression is lifted — orders are now active

### Live Phase

The provider streams `BarUpdate` objects via WebSocket.  Each update carries an OHLCV snapshot
and an `is_closed` flag:

```
BarUpdate(ohlcv=OHLCV(...), is_closed=False)   # intra-bar tick
BarUpdate(ohlcv=OHLCV(...), is_closed=True)     # bar closed
```

The script executes on **every update** — both intra-bar ticks and bar closes.

## Intra-Bar Updates

On TradingView, a real-time script re-executes on every tick within a bar.  PyneCore replicates
this behavior in live mode.

### barstate Values

| Event              | `isconfirmed` | `isnew` | `islast` | `isrealtime` |
|--------------------|---------------|---------|----------|--------------|
| First tick of bar  | `False`       | `True`  | `True`   | `True`       |
| Later intra-bar    | `False`       | `False` | `True`   | `True`       |
| Bar close          | `True`        | `False` | `True`   | `True`       |

### var vs varip in Live Mode

The distinction between `Persistent` (Pine `var`) and `IBPersistent` (Pine `varip`) becomes
meaningful during intra-bar re-executions — the same mechanism used by
[calc_on_order_fills](./bar-magnifier.md#calc_on_order_fills):

- **`Persistent` (var)**: rolled back to the bar-open snapshot before each intra-bar tick.
  Every tick starts from the same baseline.
- **`IBPersistent` (varip)**: **not** rolled back — accumulates across all ticks within the bar.

```python
var_counter: Persistent[int] = 0
varip_counter: IBPersistent[int] = 0

var_counter += 1      # always == bar_index + 1 (rolled back each tick)
varip_counter += 1    # bar_index + 1 + total intra-bar ticks across all bars
```

This uses the same `VarSnapshot` mechanism as the bar magnifier's COOF loop.

## Order Processing

Strategies use **magnifier-style order processing** in live mode: intra-bar ticks are
accumulated as `sub_bars`.  When the bar closes, `process_orders_magnified(sub_bars, final_bar)`
runs — checking limit, stop, and trailing stop orders against each tick's OHLCV in chronological
order.  This gives tick-level fill accuracy even in paper trading.

If `calc_on_order_fills=True`, the COOF re-execution loop runs on bar close as well — exactly
as it does in backtesting with the bar magnifier.

### Strategy Suppression

During the historical phase, all 7 strategy functions (`entry`, `exit`, `close`, `close_all`,
`cancel`, `cancel_all`, `order`) are no-ops.  This is controlled by the internal
`lib._strategy_suppressed` flag — the same pattern as `lib._lib_semaphore`.

## Output

### Plot CSV

Written only on **closed bars**.  Intra-bar ticks do not produce plot output.  This matches
TradingView behavior where plot values are committed only at bar close.

### Strategy Stats CSV

In live mode, the strategy statistics file is **rewritten after every closed bar** — not
appended.  This means opening the file at any time shows the complete, up-to-date statistics
aggregated over the entire run (historical + live).

### Trade CSV

Trade entries and exits are recorded on the bar where the fill occurs, as in backtesting.

## Provider String Format

```
provider:EXCHANGE:SYMBOL:SETTLE@TIMEFRAME
```

| Part         | Example         | Description                        |
|--------------|----------------|------------------------------------|
| `provider`   | `ccxt`         | Plugin name (entry point)          |
| `EXCHANGE`   | `BYBIT`        | Exchange identifier                |
| `SYMBOL`     | `BTC/USDT`     | Trading pair                       |
| `SETTLE`     | `USDT`         | Settlement currency (optional)     |
| `TIMEFRAME`  | `1`            | TradingView timeframe format       |

The `-f` / `--from` option accepts a negative integer for relative bar count:

```bash
# Prefetch last 500 bars before going live
pyne run script.py ccxt:BYBIT:BTC/USDT:USDT@1 --live -f -500
```

## CLI Options

| Flag                  | Description                                          |
|-----------------------|------------------------------------------------------|
| `--live`, `-l`        | Enable live streaming after historical phase         |
| `--shutdown-timeout`  | Max seconds for graceful shutdown (default: 120)     |

Press `Ctrl+C` to stop live streaming.  The provider goes through a graceful shutdown sequence:
`can_shutdown()` is polled every second, then `disconnect()` is called.

## Architecture

```
┌─────────────┐    provider.download()    ┌──────────────┐
│  run.py     │ ───────────────────────── │  OHLCV file  │
│  (CLI)      │                           └──────┬───────┘
│             │                                  │ OHLCVReader
│             │    itertools.chain()             │
│             │ ◄────────────────────────────────┤
│             │                                  │
│             │    live_ohlcv_generator() ┌──────┴───────┐
│             │ ◄──── Queue ◄──── async ──│  WebSocket   │
└──────┬──────┘                           └──────────────┘
       │ Iterator[OHLCV | BarUpdate]
       │
┌──────▼───────┐
│ ScriptRunner │
│              │  isinstance() detects BarUpdate → live transition
│  historical  │  OHLCV bars → normal backtest loop
│  live loop   │  BarUpdate → intra-bar + bar close processing
└──────────────┘
```

The `live_ohlcv_generator` bridges the async WebSocket world to synchronous iteration via a
background thread and `queue.Queue`.  The ScriptRunner is completely data-source agnostic — it
only cares whether it receives `OHLCV` or `BarUpdate` objects.

## Limitations

- **Paper trading only** — no real order execution.  Live order routing is provided by
  dedicated per-exchange broker plugins (`pynecore-bybit`, `pynecore-binance`, etc.).
- **Single timeframe** — `request.security()` with live providers (multi-timeframe live) is not
  yet supported.
- **Provider required** — `--live` only works with provider strings, not local data files.
- **No replay** — there is no mechanism to replay missed ticks if the connection drops mid-bar.
  The provider reconnects and resumes from the next available update.