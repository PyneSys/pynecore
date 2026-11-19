<!--
---
weight: 1070
title: "Live Mode"
description: "Run PyneCore scripts on live provider data after historical replay. Configure tick updates, persistent intrabar state, paper trading and broker order routing."
icon: "stream"
date: "2026-04-08"
lastmod: "2026-11-18"
draft: false
toc: true
categories: ["Live Trading"]
tags: ["live", "streaming", "intrabar", "varip", "paper-trading", "real-time", "ibpersistent"]
---
-->

# Live Mode

Live mode extends PyneCore beyond backtesting: after replaying historical data the script
seamlessly transitions to real-time streaming from a `LiveProviderPlugin`.  Indicators update
on every tick; strategies run in paper-trading mode with tick-level order fill accuracy.
With `--broker` (which implies `--live`), orders are routed to the real exchange through a
`BrokerPlugin` instead of being simulated.

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
first value. Recursive indicators (`ta.ema`, `ta.rma`, `ta.rsi`, `ta.atr`, …) produce a value
early, but it still carries the influence of the starting state, which fades gradually and never
reaches exactly zero. How much history is enough therefore depends on how close to a fully warmed-up
run your values need to be. A common starting point is **2× the largest `length` parameter**; give
recursive indicators more when exact values matter, and compare a run against one with a longer
history to check:

```bash
# Script uses ta.ema(close, 50) and ta.atr(14) → largest length is 50 → start with -f -100
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

The data iterator marks the end of the history with the `LIVE_TRANSITION` sentinel, and the
ScriptRunner switches to the live phase when it reads it.  At this point:

- `barstate.islast` is `True` on the final historical bar: it is the dataset's last bar until
  the first live update arrives, so `if barstate.islast` blocks (tables, last-bar labels) show
  at once, even on a closed market
- `barstate.islastconfirmedhistory` becomes `True` on the final confirmed historical bar
- Output writers flush to disk (plot CSV, trade CSV)
- Strategy suppression is lifted — orders are now active

### Live Phase

The provider streams `OHLCV` updates via WebSocket.  The `is_closed` field tells a price update
of the forming bar from its close:

```
OHLCV(..., is_closed=False)   # intra-bar tick
OHLCV(..., is_closed=True)    # bar closed
```

An indicator executes on **every update** — both intra-bar ticks and bar closes.  A strategy
executes on the bar close, and on every tick only with `calc_on_every_tick=true`.

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

Arrays, maps, matrices and UDT instances held by a `Persistent` are rolled back as objects, not
replaced by copies: their content returns to the bar-open state, and two variables that name the
same object keep naming the same object.

```python
var_counter: Persistent[int] = 0
varip_counter: IBPersistent[int] = 0

var_counter += 1      # always == bar_index + 1 (rolled back each tick)
varip_counter += 1    # bar_index + 1 + total intra-bar ticks across all bars
```

This uses the same `VarSnapshot` mechanism as the bar magnifier's COOF loop.

## Order Processing

How paper trading fills orders on a live bar depends on when the strategy executes.

**With `calc_on_every_tick=true`** every update is a tick the simulator fills on, before the
script executes on it.  An order placed on one tick fills on the next one, and the script sees
the fill on the tick it happened on — `strategy.position_size` changes inside the bar, as on
TradingView's realtime bars.  A live update carries the forming bar so far (its open, the
extremes since the open, the latest price), so each tick is filled on its own part of the bar
only:

- The first update of a bar is filled whole: the orders queued at the previous bar's close fill
  at the bar's open.
- A later update is filled at its latest price.  A high or low it moved past the previous update
  was reached since the previous tick, so a price-based order at that level fills there, at its
  own price.
- A price-based order whose level the price stepped over between two ticks fills at the tick
  price, the same rule a gap between two bars follows.  Nothing before the order was placed —
  the bar's open, an earlier extreme — can fill it.

The script executing right after the fills of every tick plays the role of the
`calc_on_order_fills` re-execution, so no separate re-execution loop runs.

**Otherwise** the script executes on the bar close only, and the orders it places there are
filled on the next bar's ticks.  The ticks are accumulated as `sub_bars`, and when that bar
closes `process_orders_magnified(sub_bars, final_bar)` checks limit, stop and trailing stop
orders against them in chronological order.  If `calc_on_order_fills=True`, the COOF
re-execution loop runs on bar close as well — exactly as it does in backtesting with the bar
magnifier.

With `--broker` the exchange fills the orders: the order book is synced on every tick the script
executes on, and on every bar close.

### Strategy Suppression

During the historical phase, all 7 strategy functions (`entry`, `exit`, `close`, `close_all`,
`cancel`, `cancel_all`, `order`) are no-ops.  Live mode turns this on for the historical phase
and lifts it at the transition.

With `--history-trades` (paper trading only, not with `--broker`) the historical phase trades
too: it runs as a backtest, and the live phase continues it with the position it left.  A broker
run never trades on history, so its record holds only what it really traded.  The last historical
bar places no orders while its period was still open at the download, because the live feed goes
on forming that bar.

## Security Contexts in Live Mode

In a live run, every `request.security()` context except the chart's own symbol at the
chart's timeframe runs in a process of its own and downloads a warmup from the history of
the chart's provider.  A context of another symbol, and the chart's own symbol at a lower
timeframe, then follow the provider's live stream.  The chart's own symbol at a higher
timeframe gets its closed live bars from the chart's own bars, aggregated to the context's
timeframe.

- **Symbol** — the chart's own symbol at another timeframe is read from the symbol of the
  chart's provider string.  Any other symbol is translated to the provider's own with the
  `symbol_map` of the provider's config (`config/plugins/<provider>.toml`), then with the
  workdir's `config/symbol_map.toml`; a symbol neither maps is passed on as written.  A
  symbol the provider cannot serve stops the run with
  [R0055](../reference/codes/runtime-errors.md#R0055), or gives `na` with
  `ignore_invalid_symbol = true`.
- **Warmup** — the history starts at the earlier of the chart's own history start and 500
  bars of the context's timeframe, so a higher-timeframe context gets a depth of its own,
  not just the few bars the chart window holds.  A warmup that fails stops the run with
  [R0049](../reference/codes/runtime-errors.md#R0049).
- **Outage recovery** — after a reconnect the context's stream recovers the bars that closed
  during the outage, the same way the chart's stream does (see
  [Limitations](#limitations)).

### A close that comes late

A chart bar waits for the closed bar of its period of each context that follows the
provider's stream.  The stream usually delivers it within seconds, but a venue may publish
a closed bar late.  The chart bar first waits a grace for the stream: half the context's
period, kept between 15 and 30 seconds.  Past the grace it goes on waiting and queries the
provider's history for the bar at once, then again after each query that found nothing,
with a pause of 2 seconds that doubles up to 30 seconds.  The wait is bounded by a budget
of a quarter of the context's period, kept between 2 and 10 minutes and counted from the
start of the chart bar's wait:

| What arrives                                   | Result                                                          |
|------------------------------------------------|-----------------------------------------------------------------|
| The bar, from the stream or from the history   | The chart bar gets its real value                               |
| A later bar from the history, without this one | The period had no bar: the same as any period without trades    |
| Nothing within the budget                      | The chart bar completes with `na` for it, with one warning      |

While the context's feed stays silent, the next chart bars wait only the grace, not the
budget again, and give no new warning; the next closed bar that arrives re-arms the full
budget.  A value once given to a chart bar is never revised, even if the missing bar
arrives later: a chart bar never looks into the future.  A backtest run later, on a history
that holds the bar the live wait missed, can therefore differ from the live run at that
chart bar.  The final intrabar of a `request.security_lower_tf()` window is waited for the
same way.

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

| Flag                  | Description                                                            |
|-----------------------|------------------------------------------------------------------------|
| `--live`, `-l`        | Enable live streaming after historical phase                           |
| `--broker`            | Route orders to the exchange (implies `--live`; needs a `BrokerPlugin`) |
| `--history-trades`    | Paper trading: trade on the historical bars too and continue live      |
| `--shutdown-timeout`  | Max seconds for graceful shutdown (default: 120)                       |

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
       │ Iterator[OHLCV]: history, LIVE_TRANSITION, live updates
       │
┌──────▼───────┐
│ ScriptRunner │  live=True
│              │  LIVE_TRANSITION → live transition
│  historical  │  OHLCV bars → normal backtest loop
│  live loop   │  OHLCV(is_closed=...) → intra-bar + bar close processing
└──────────────┘
```

The `live_ohlcv_generator` bridges the async WebSocket world to synchronous iteration via a
background thread and `queue.Queue`.  The ScriptRunner is completely data-source agnostic — it
only needs the history, the `LIVE_TRANSITION` sentinel and the live updates, in this order.  Any
feed can drive it from Python: see [Live Mode](../programmatic/script-runner.md#live-mode) in the
ScriptRunner API.

`live_ohlcv_generator(..., on_feed_status=callback)` reports the state of the feed as a
`FeedStatus`: `connecting`, `live` (data flows, again after an outage or a closed market),
`reconnecting` (every attempt, with its number, the time offline and the error),
`market_closed` and `stopped`.  A front end can show it without parsing the log.  The callback
runs on the provider's background thread, so it must return quickly; an exception it raises is
logged and does not stop the feed.

## Limitations

- **Paper trading by default** — with `--live` alone, order fills are simulated.  Real order
  execution requires a `BrokerPlugin` data source run with `pyne run --broker` (implies
  `--live`); shipped broker plugins are `pynesys-pynecore-bybit`,
  `pynesys-pynecore-capitalcom` and `pynesys-pynecore-ctrader`.
- **Provider required** — `--live` only works with provider strings, not local data files.
- **Outage recovery covers closed bars, not ticks** — after a reconnect, the shipped plugins fetch
  every bar that closed during the outage from the venue's REST history and feed it to the script
  as a closed bar before the live stream resumes, so the series continues without a hole. At
  startup PyneCore likewise recovers the bars that closed while the subscription was being
  established. What is not replayed is the stream of ticks inside a bar that was still forming
  during the outage.
