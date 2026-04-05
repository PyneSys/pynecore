<!--
---
weight: 1060
title: "Bar Magnifier"
description: "Accurate intrabar order fills using lower-timeframe data"
icon: "zoom_in"
date: "2025-07-24"
lastmod: "2025-07-24"
draft: false
toc: true
categories: ["Advanced", "Strategy"]
tags: ["bar-magnifier", "backtesting", "order-fills", "intrabar", "timeframe"]
---
-->

# Bar Magnifier

The bar magnifier improves backtesting accuracy by processing order fills against lower-timeframe
(LTF) data instead of relying on a direction heuristic applied to a single aggregated bar.

## The Problem

When backtesting a strategy on e.g. 1-hour bars, each bar has four prices: Open, High, Low, Close.
But the order in which High and Low were reached within the bar is unknown. PyneCore (like
TradingView) uses a heuristic to guess the intrabar direction:

```
ohlc = (high - open) < (open - low)
```

- If `True`: assumes the path was Open → High → Low → Close
- If `False`: assumes the path was Open → Low → High → Close

The intuition: if the open is closer to the high, reaching the high required less price movement, so
it likely happened first — then the larger move to the low followed. This is a solid Bayesian
inference, and over many trades the errors tend to cancel out. However, when both a take-profit and
a stop-loss could fill within the same bar, the heuristic has a 50/50 chance of picking the wrong
one for that specific bar — which is where the magnifier adds value.

### Example

Consider a long position with TP at 110 and SL at 90. The 1-hour bar shows:

```
Open=100, High=112, Low=88, Close=100
```

Both TP and SL are within the bar's range. Which filled first? The heuristic guesses based on
whether price moved up or down first — but it has no real information about this. With the bar
magnifier, the actual 5-minute (or 1-minute) candles within this hour reveal the true price path.

## How It Works

The bar magnifier gives the **broker emulator** access to lower-timeframe OHLCV data within each
chart bar. The key principle:

- **The script still runs once per chart bar** — it sees aggregated OHLCV values
- **Order fills are checked against each sub-bar** — sequentially, in chronological order
- **No script re-execution** — only the broker emulator uses the sub-bar data

### Three-Phase Processing

Order processing is split into three phases:

| Phase | Runs                     | Purpose                                           |
|-------|--------------------------|---------------------------------------------------|
| 1     | Once (first sub-bar)     | Gap detection, market order fills, margin at open |
| 2     | Once per sub-bar         | Limit, stop, and trailing stop fills              |
| 3     | Once (aggregated values) | P&L calculation, drawdown/runup stats             |

Phase 2 is where the magnifier provides its value: instead of checking limit/stop orders once
against a guessed OHLC direction, it checks them against each sub-bar's actual OHLC — in order.

## Usage

The bar magnifier is **enabled by default** in PyneCore strategies (`use_bar_magnifier=True`). It
only activates when lower-timeframe data is actually provided via `--timeframe` — otherwise it has
no effect. You can explicitly disable it with `use_bar_magnifier=False` if you want `--timeframe`
to only aggregate on-the-fly without magnified fills.

Run with the `--timeframe` option, providing lower-timeframe data:

```bash
# Data file contains 10-minute candles, strategy runs on 1-hour chart
pyne run my_strategy.py EURUSD_10m.ohlcv --timeframe 60

# Data file contains 1-minute candles, strategy runs on 5-minute chart
pyne run my_strategy.py BTCUSDT_1m.ohlcv --timeframe 5
```

The `--timeframe` value is the **chart timeframe** (what the script sees). The data file contains
the **lower timeframe** (what the broker emulator uses for fills).

### What the Script Sees

| Aspect                 | Without magnifier        | With magnifier                  |
|------------------------|--------------------------|---------------------------------|
| `open/high/low/close`  | Chart TF OHLCV           | Aggregated from LTF (identical) |
| `close[1]`             | Previous chart bar close | Previous chart bar close        |
| `bar_index`            | Increments per chart bar | Increments per chart bar        |
| `barstate.isconfirmed` | `True`                   | `True`                          |
| `syminfo.period`       | Chart TF                 | Chart TF                        |
| Order fill check       | 1x against guessed OHLC  | Nx against each sub-bar OHLC    |

From the script's perspective, nothing changes. The improvement is entirely in fill accuracy.

## TradingView Comparison

TradingView's bar magnifier uses a fixed mapping table — the user cannot choose the LTF:

| Chart TF | TV Magnifier TF |
|----------|-----------------|
| 1S       | 1S              |
| 30S      | 5S              |
| 1        | 10S             |
| 5        | 30S             |
| 10       | 1               |
| 15       | 2               |
| 30       | 5               |
| 60       | 10              |
| 240      | 30              |
| 1D       | 60              |
| 3D       | 240             |
| 1W       | 1D              |

### PyneCore Advantage

In PyneCore, **you choose the LTF resolution**. This means:

- **Higher accuracy**: if you have 1-minute data for a 1D chart, use it — TV is limited to 60m
- **TV-compatible results**: use the same LTF as TV's mapping table for identical results
- **Flexible trade-offs**: use coarser LTF data when fine-grained data isn't available — even 30m
  sub-bars on a 1D chart are better than the OHLC heuristic

### Performance vs. Accuracy

The magnifier checks orders against every sub-bar, so finer LTF data means more iterations per
chart bar. For a 1D chart:

| LTF   | Sub-bars per day | Relative speed |
|-------|------------------|----------------|
| 60m   | ~24              | Fast           |
| 10m   | ~144             | Moderate       |
| 1m    | ~1440            | Slow           |

Choose the LTF that balances accuracy with acceptable runtime. TradingView's mapping (60m for 1D)
is a reasonable middle ground — but if you need maximum precision and can afford the runtime, finer
data is always better.

### Matching TradingView Results

To reproduce TradingView's bar magnifier results exactly, use the LTF from the mapping table above.
For example, for a 1-hour strategy:

```bash
# Use 10-minute data (same as TV's magnifier TF for 60m charts)
pyne run my_strategy.py EURUSD_10m.ohlcv --timeframe 60
```

## On-the-Fly Aggregation

When `--timeframe` is used with a strategy that does **not** have `use_bar_magnifier=True`, the
data is aggregated on-the-fly to the chart timeframe. This is equivalent to running
`pyne data aggregate` but without creating an intermediate file:

```bash
# Aggregates 1-minute data to 1-hour bars, no magnifier
pyne run my_indicator.py EURUSD_1m.ohlcv --timeframe 60
```

This is useful for quick testing without pre-aggregating data files.

## calc_on_order_fills

When `calc_on_order_fills=True` is set on a strategy, the script **re-executes** after each order
fill within a bar. This works with both the standard path and the bar magnifier path.

### Execution Model

```
for each bar:
    committed = snapshot(var globals)
    process_orders()                        # fills from previous bar's orders

    while new fills detected:
        restore var globals to committed    # rollback (varip excluded)
        main()                              # re-execution
        process_orders()                    # process new orders from re-execution

    restore var globals to committed        # final rollback
    main()                                  # definitive bar-close execution
```

### Persistent vs IBPersistent

On historical bars, a script normally executes once per bar — no rollback occurs, so `Persistent`
and `IBPersistent` behave identically. With `calc_on_order_fills`, the script can execute
**multiple times on the same bar**, and the difference appears:

- `Persistent` (TradingView `var`): **rolled back** to the previous bar's committed state before
  each re-execution. Every execution starts from the same baseline.
- `IBPersistent` (TradingView `varip`): **not rolled back** — retains its value across all
  executions on the same bar.

```python
var_count: Persistent[int] = 0       # rolled back before each re-execution
varip_count: IBPersistent[int] = 0   # persists across re-executions

var_count += 1    # always bar_index+1 (rollback ensures no extra increments)
varip_count += 1  # bar_index+1 + number of re-executions on fill bars
```

## Limitations

- **Sub-bar data must align**: the LTF data must divide evenly into the chart TF (e.g., 10m into
  60m, not 7m into 60m). PyneCore validates this at startup.
- **Data availability**: you need to source and store the LTF data yourself — PyneCore does not
  download it automatically.
