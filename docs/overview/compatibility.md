<!--
---
weight: 103
title: "Pine Script Compatibility"
description: "Implementation status of Pine Script v6 features in PyneCore"
icon: "checklist"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Overview", "Compatibility"]
tags: ["pine-script", "compatibility", "features", "status"]
---
-->

# Pine Script Compatibility

PyneCore aims for high compatibility with TradingView's Pine Script v6. This page tracks the
implementation status of all major Pine Script features.

> **Note:** PyneCore only supports Pine Script **v6**. Scripts written in v5 or earlier must be
> updated to v6 syntax (PyneComp handles this automatically during compilation).

## Status Legend

| Symbol      | Meaning                                                                |
|-------------|------------------------------------------------------------------------|
| full        | Fully implemented and tested against TradingView                       |
| partial     | Partially implemented — see notes for limitations                      |
| stub        | Function exists but returns `na` or raises `NotImplementedError`       |
| stub/plugin | API accepted (scripts don't break), but no real output — future plugin |
| plugin      | Not in core — designed as future plugin extension point                |
| —           | Not applicable to offline execution                                    |

## Technical Analysis (ta)

| Feature                  | Status | Notes                                        |
|--------------------------|--------|----------------------------------------------|
| Moving averages          | full   | SMA, EMA, WMA, HMA, ALMA, RMA, KAMA, etc.    |
| Momentum indicators      | full   | RSI, MACD, CCI, CMO, MFI, ROC, TSI, etc.     |
| Volatility indicators    | full   | ATR, BB, KC, STDev, Supertrend, etc.         |
| Volume indicators        | full   | OBV, ACCDIST, PVT, WAD, WVAD, etc.           |
| Pivot points             | full   | All 7 pivot types                            |
| Statistical functions    | full   | Correlation, percentile, variance, etc.      |
| Pattern detection        | full   | Crossover, crossunder, rising, falling, etc. |
| **Total: 74 indicators** | full   | Precision-tested against TradingView         |

## Strategy Simulator

| Feature                 | Status | Notes                               |
|-------------------------|--------|-------------------------------------|
| Entry/exit orders       | full   | Market, limit, stop orders          |
| Position management     | full   | Long, short, pyramiding             |
| Take-profit / stop-loss | full   | Price, ticks, percent-based         |
| Trailing stops          | full   | Offset-based trailing               |
| OCA groups              | full   | One-Cancels-All order groups        |
| Commission models       | full   | Fixed, percent, per-contract        |
| Margin calls            | full   | TradingView-exact 10-step algorithm |
| Slippage                | full   | Configurable tick-based slippage    |
| Equity tracking         | full   | Equity curve, drawdown, P&L         |
| Trade logging           | full   | CSV export with all trade fields    |
| `strategy.close_all()`  | full   |                                     |
| `strategy.cancel_all()` | full   |                                     |
| Risk management         | full   | `strategy.risk.*` functions         |

## Request Module

| Feature                       | Status | Notes                                          |
|-------------------------------|--------|------------------------------------------------|
| `request.security()`          | full   | Multiprocessing with shared memory             |
| `request.security_lower_tf()` | full   | Returns arrays of intrabar values              |
| `request.currency_rate()`     | full   | TOML-based currency pair auto-detection        |
| `request.dividends()`         | stub   | Returns `na` with `ignore_invalid_symbol=True` |
| `request.splits()`            | stub   | Returns `na` with `ignore_invalid_symbol=True` |
| `request.earnings()`          | stub   | Returns `na` with `ignore_invalid_symbol=True` |
| `request.financial()`         | plugin | FactSet data — external data source            |
| `request.economic()`          | plugin | Macro data — external data source              |
| `request.quandl()`            | plugin | Nasdaq Data Link — paid external API           |
| `request.seed()`              | plugin | GitHub repository data                         |
| `request.footprint()`         | plugin | Requires tick-level data                       |

### request.security() Details

| Feature                 | Status | Notes                                  |
|-------------------------|--------|----------------------------------------|
| Higher timeframe        | full   | 1D, 1W, 1M, etc. from lower TF chart   |
| Different symbol        | full   | Any symbol with OHLCV data             |
| Lower timeframe (LTF)   | full   | Via `request.security_lower_tf()`      |
| Multiple calls          | full   | Each gets its own OS process           |
| Conditional calls       | full   | Inside if/for/while blocks             |
| Nested calls            | full   | security(... security(...) ...)        |
| `barmerge.gaps_off`     | full   | Forward-fills last value (default)     |
| `barmerge.gaps_on`      | full   | Returns `na` between periods           |
| `lookahead_off`         | full   | Confirmed previous period (default)    |
| `ignore_invalid_symbol` | full   | Returns `na` for missing symbols       |
| `lookahead_on`          | —      | Deliberate safety-first exclusion      |
| `currency` parameter    | full   | Auto-converts via `CurrencyRateProvider` |

## Drawing Objects

Scripts using drawing objects will **not error out** — all API calls are accepted and object
state is tracked in memory. However, PyneCore currently has no built-in visualization. A future
plot plugin will consume the stored drawing data to render charts.

| Feature      | Status      | Notes                                                   |
|--------------|-------------|---------------------------------------------------------|
| `label.*`    | stub/plugin | API functional, data stored — visualization via plugin  |
| `line.*`     | stub/plugin | API functional, data stored — visualization via plugin  |
| `box.*`      | stub/plugin | API functional, data stored — visualization via plugin  |
| `table.*`    | stub/plugin | API functional, data stored — visualization via plugin  |
| `polyline.*` | stub/plugin | API functional, data stored — visualization via plugin  |
| `linefill.*` | stub/plugin | API functional, data stored — visualization via plugin  |

## Core Modules

| Module                    | Status | Functions  | Notes                               |
|---------------------------|--------|------------|-------------------------------------|
| `math`                    | full   | 24         | + constants (e, pi, phi, rphi)      |
| `array`                   | full   | 57         | Wraps Python `list`                 |
| `map`                     | full   | 11         | Wraps Python `dict`                 |
| `matrix`                  | full   | 51         | Full 2D array operations            |
| `string`                  | full   | 21         | Named `string` (not `str`)          |
| `color`                   | full   | 50+ colors | RGB creation, constants             |
| `timeframe`               | full   | 19         | Conversion, validation              |
| `session`                 | full   | 8          | Market session handling             |
| `barstate`                | full   | 5          | isfirst, islast, isnew, etc.        |
| `syminfo`                 | full   | 20+ props  | From TOML metadata                  |
| `chart`                   | full   |            | Chart type flags, colors            |
| `log`                     | full   | 3          | info, warning, error                |
| `alert`                   | full   | 1 + 3      | Alert with frequency constants      |
| `runtime`                 | full   | 1          | `error()` for script termination    |
| `input`                   | full   |            | All input types via function params |
| `plot` / `hline` / `fill` | full   |            | Output to CSV                       |

## Type System

| Type                          | Status | Notes                                  |
|-------------------------------|--------|----------------------------------------|
| `int`, `float`, `bool`, `str` | full   | Native Python types                    |
| `Series[T]`                   | full   | Circular buffer, historical access     |
| `Persistent[T]`               | full   | Cross-bar state via AST transformation |
| `na` / `NA[T]`                | full   | Full NA propagation                    |
| `Color`                       | full   | RGBA with transparency                 |
| `label`, `line`, `box`        | full   | Dataclass-based drawing objects        |
| `table`, `polyline`           | full   |                                        |
| `chart.point`                 | full   | `ChartPoint` in PyneCore               |
| `array` / `matrix` / `map`    | full   | Python `list` / custom / `dict`        |
| `footprint` / `volume_row`    | stub   | Types defined, no data source          |

## Enum Constants

All Pine Script v6 enum constants are implemented:

| Module          | Constants                                                 |
|-----------------|-----------------------------------------------------------|
| `currency`      | 21 currency codes (USD, EUR, BTC, etc.)                   |
| `barmerge`      | gaps_on/off, lookahead_on/off                             |
| `display`       | none, all, data_window, pane, price_scale, status_line    |
| `color`         | 50+ named colors                                          |
| `extend`        | none, left, right, both                                   |
| `format`        | inherit, price, volume, percent, mintick                  |
| `location`      | abovebar, belowbar, top, bottom, absolute                 |
| `position`      | 9 table positions (top/middle/bottom x left/center/right) |
| `shape`         | 10+ marker shapes                                         |
| `size`          | auto, tiny, small, normal, large, huge                    |
| `xloc` / `yloc` | bar_index/bar_time, price/abovebar/belowbar               |
| `dayofweek`     | monday through sunday                                     |
| `dividends`     | gross, net                                                |
| `earnings`      | actual, estimate, standardized                            |
| `adjustment`    | none, dividends, splits                                   |
| `scale`         | right, left, none                                         |

## Pine Script Syntax Features

| Feature                    | Status | Notes                                 |
|----------------------------|--------|---------------------------------------|
| Functions                  | full   | Including nested/inline functions     |
| `if`/`else`/`switch`       | full   | Via PyneComp compilation              |
| `for`/`while` loops        | full   |                                       |
| `var` (persistent)         | full   | `Persistent[T]` annotation            |
| `varip` (intrabar persist) | —      | Not applicable in offline mode        |
| Methods on types           | full   | `.get()`, `.set()`, `.size()`, etc.   |
| User-defined types (UDT)   | full   | Via PyneComp compilation              |
| Enums                      | full   | Via PyneComp compilation              |
| Libraries                  | full   | Import and use Pine libraries         |
| Operator overloading       | full   | `+`, `-`, `*`, `/`, `%`, comparisons  |
| Ternary operator           | full   | Python conditional expression         |
| Type casting               | full   | `int()`, `float()`, `str()`, `bool()` |
| `na` propagation           | full   | Arithmetic, comparison, logical       |
| Multi-return (tuples)      | full   |                                       |
| Default parameters         | full   |                                       |
| `export` (libraries)       | full   | Via PyneComp compilation              |

## Not Applicable to PyneCore

These Pine Script features exist only in TradingView's live charting environment and are not
applicable to offline backtesting:

| Feature              | Reason                                            |
|----------------------|---------------------------------------------------|
| `varip`              | Intrabar persistence — offline bars are confirmed |
| Realtime bar updates | All bars are historical in offline mode           |
| `alert()` triggers   | No broker/notification integration                |
| Chart rendering      | No visual chart — output is CSV                   |
| `input()` UI widgets | Inputs are function parameters or TOML config     |
| Order execution      | Strategy simulator, not live trading              |

## Precision

PyneCore is precision-tested against TradingView:

- **Indicator values**: 0.001% relative tolerance, 0.00000001 absolute tolerance
- **Strategy trades**: Exact trade-by-trade matching (245+ trades verified)
- **OHLCV data**: float32 storage with 6-significant-digit rounding
