<!--
---
weight: 105
title: "Pine Script Compatibility"
description: "PyneCore vs TradingView: validated on 809 published Pine Script v6 scripts, 99.714% of 99 million values bit-identical, 289,074 trades matched. Feature status of every Pine Script v6 module."
icon: "checklist"
date: "2026-03-28"
lastmod: "2026-09-26"
draft: false
toc: true
categories: ["Overview", "Compatibility"]
tags: ["pine-script", "compatibility", "features", "status", "tradingview", "validation"]
---
-->

# Pine Script Compatibility

PyneCore runs Pine Script v6 with results that match TradingView. The match is measured, not
assumed: PyneCore is run on a public corpus of real, published TradingView scripts and every
comparable output is compared with TradingView's own.

> **Validation status** — snapshot 2026-09-23, [Pyne in the Wild](https://wild.pynesys.io/)
>
> - **809** published open-source Pine Script v6 scripts (407 indicators, 402 strategies), all run
> - **1,090 / 1,090** outputs comparable with TradingView verified (712 plot outputs, 378
>   strategy trade lists)
> - **99.714%** of 99,018,068 plotted values identical to TradingView to the bit; the largest
>   relative gap anywhere is 1.2 × 10⁻¹⁰ (about 9 significant figures)
> - **289,074** strategy trades in 378 strategies: entry and exit timing and trade counts match
> - Chart: BINANCE:BTCUSDT, 30-minute bars; per-script results and methodology are public

The rest of this page lists the status of every Pine Script v6 feature area, the places where
PyneCore differs from TradingView on purpose, and where the remaining 0.3% of values come from.

> **Note:** PyneCore only supports Pine Script **v6**. Scripts written in v5 or earlier must be
> updated to v6 syntax (PyneComp handles this automatically during compilation).

## Status Legend

| Status       | Meaning                                                                                                           |
|--------------|-------------------------------------------------------------------------------------------------------------------|
| full         | Implemented; results match TradingView                                                                            |
| no-lookahead | Implemented; differs only where TradingView would leak future data — see [No lookahead, ever](#no-lookahead-ever) |
| no data feed | API present, but the data has no offline source: the call returns `na` or raises — see notes                      |
| no renderer  | Full API and state; exported with `pyne run --viz`, but PyneCore draws no chart                                   |
| —            | Not applicable to offline execution                                                                               |

## Technical Analysis (ta)

| Feature                 | Status | Notes                                        |
|-------------------------|--------|----------------------------------------------|
| Moving averages         | full   | SMA, EMA, WMA, HMA, ALMA, RMA, VWMA, SWMA    |
| Momentum indicators     | full   | RSI, MACD, CCI, CMO, MFI, ROC, TSI, etc.     |
| Volatility indicators   | full   | ATR, BB, KC, STDev, Supertrend, etc.         |
| Volume indicators       | full   | OBV, ACCDIST, PVT, WAD, WVAD, NVI, PVI, etc. |
| Pivot points            | full   | All 6 pivot types                            |
| Statistical functions   | full   | Correlation, percentile, variance, etc.      |
| Pattern detection       | full   | Crossover, crossunder, rising, falling, etc. |
| **Total: 67 functions** | full   | Every `ta.*` function of Pine Script v6      |

## Strategy Simulator

| Feature                      | Status       | Notes                                                                                                                                       |
|------------------------------|--------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| Entry/exit orders            | full         | Market, limit, stop orders                                                                                                                  |
| Position management          | full         | Long, short, pyramiding                                                                                                                     |
| Take-profit / stop-loss      | full         | Price, ticks, percent-based                                                                                                                 |
| Trailing stops               | full         | Offset-based trailing                                                                                                                       |
| OCA groups                   | full         | One-Cancels-All order groups                                                                                                                |
| Commission models            | full         | Fixed, percent, per-contract                                                                                                                |
| Margin calls                 | full         | TradingView-exact 10-step algorithm                                                                                                         |
| Slippage                     | full         | Configurable tick-based slippage                                                                                                            |
| Equity tracking              | full         | Equity curve, drawdown, P&L                                                                                                                 |
| Trade logging                | full         | CSV export with all trade fields                                                                                                            |
| `strategy.close_all()`       | full         |                                                                                                                                             |
| `strategy.cancel_all()`      | full         |                                                                                                                                             |
| Risk management              | full         | `strategy.risk.*` functions                                                                                                                 |
| `calc_on_order_fills`        | full         | Re-execution after fills, var rollback / varip persist                                                                                      |
| `calc_on_every_tick`         | full         | Live mode only — no effect on historical bars                                                                                               |
| `calc_on_every_history_tick` | no-lookahead | Four passes per bar (per sub-bar with the magnifier); each pass sees the bar as built so far — see [No lookahead, ever](#no-lookahead-ever) |

## Request Module

| Feature                       | Status       | Notes                                                                                       |
|-------------------------------|--------------|---------------------------------------------------------------------------------------------|
| `request.security()`          | full         | Multiprocessing with shared memory                                                          |
| `request.security_lower_tf()` | full         | Returns arrays of intrabar values                                                           |
| `request.currency_rate()`     | full         | TOML-based currency pair auto-detection                                                     |
| `request.dividends()`         | no data feed | `na` with `ignore_invalid_symbol=True` or for a crypto chart's own ticker; raises otherwise |
| `request.splits()`            | no data feed | `na` with `ignore_invalid_symbol=True` or for a crypto chart's own ticker; raises otherwise |
| `request.earnings()`          | no data feed | `na` with `ignore_invalid_symbol=True` or for a crypto chart's own ticker; raises otherwise |
| `request.financial()`         | no data feed | FactSet fundamentals; `na` with `ignore_invalid_symbol=True`, raises otherwise              |
| `request.economic()`          | no data feed | Macro data; raises `NotImplementedError`                                                    |
| `request.quandl()`            | no data feed | Nasdaq Data Link; raises `NotImplementedError`                                              |
| `request.seed()`              | no data feed | GitHub repository data; returns `na` (a tuple of `na` for tuple expressions)                |
| `request.footprint()`         | no data feed | Returns `na` — no tick order-flow source                                                    |

### request.security() Details

| Feature                 | Status       | Notes                                                                                                                                                                                           |
|-------------------------|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Higher timeframe        | full         | 1D, 1W, 1M, etc. from lower TF chart                                                                                                                                                            |
| Different symbol        | full         | Any symbol with OHLCV data                                                                                                                                                                      |
| Lower timeframe (LTF)   | full         | Via `request.security_lower_tf()`                                                                                                                                                               |
| Multiple calls          | full         | Each gets its own OS process                                                                                                                                                                    |
| Conditional calls       | full         | Inside if/for/while blocks                                                                                                                                                                      |
| Nested calls            | full         | security(... security(...) ...)                                                                                                                                                                 |
| `barmerge.gaps_off`     | full         | Forward-fills last value (default)                                                                                                                                                              |
| `barmerge.gaps_on`      | full         | Returns `na` between periods                                                                                                                                                                    |
| `lookahead_off`         | full         | Confirmed previous period (default)                                                                                                                                                             |
| `ignore_invalid_symbol` | full         | Returns `na` for missing symbols                                                                                                                                                                |
| `lookahead_on`          | no-lookahead | Steps into the containing period, so the `close[1]` idiom matches TV; a bare `close` reads the period as built so far instead of its final value — see [No lookahead, ever](#no-lookahead-ever) |
| `currency` parameter    | full         | Auto-converts via `CurrencyRateProvider`                                                                                                                                                        |

## Drawing Objects

Every drawing API is implemented: objects are created, updated, deleted and read back exactly as
in Pine Script, and their state rolls back with the bar on intra-bar re-execution.
`pyne run --viz` exports plots and drawings as NDJSON (`--viz-journal` adds a per-bar
create/update/delete event log) for any charting front end — see
[Visual Output](../programmatic/visual-output.md). PyneCore itself draws no chart.

| Feature      | Status      | Notes                                     |
|--------------|-------------|-------------------------------------------|
| `label.*`    | no renderer | Full API and state; exported with `--viz` |
| `line.*`     | no renderer | Full API and state; exported with `--viz` |
| `box.*`      | no renderer | Full API and state; exported with `--viz` |
| `table.*`    | no renderer | Full API and state; exported with `--viz` |
| `polyline.*` | no renderer | Full API and state; exported with `--viz` |
| `linefill.*` | no renderer | Full API and state; exported with `--viz` |

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

| Type                          | Status       | Notes                                                                           |
|-------------------------------|--------------|---------------------------------------------------------------------------------|
| `int`, `float`, `bool`, `str` | full         | Native Python types; an `int` travels as an integral `float`, as on TradingView |
| `Series[T]`                   | full         | Circular buffer, historical access                                              |
| `Persistent[T]`               | full         | Cross-bar state via AST transformation                                          |
| `na` / `NA[T]`                | full         | Full NA propagation                                                             |
| `Color`                       | full         | RGBA with transparency                                                          |
| `label`, `line`, `box`        | full         | Dataclass-based drawing objects                                                 |
| `table`, `polyline`           | full         |                                                                                 |
| `chart.point`                 | full         | `ChartPoint` in PyneCore                                                        |
| `array` / `matrix` / `map`    | full         | Python `list` / custom / `dict`                                                 |
| `footprint` / `volume_row`    | no data feed | Types defined, no order-flow source                                             |

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

| Feature                    | Status | Notes                                              |
|----------------------------|--------|----------------------------------------------------|
| Functions                  | full   | Including nested/inline functions                  |
| `if`/`else`/`switch`       | full   | Via PyneComp compilation                           |
| `for`/`while` loops        | full   |                                                    |
| `var` (persistent)         | full   | `Persistent[T]` annotation                         |
| `varip` (intrabar persist) | full   | Persists across re-executions (COOF and live mode) |
| Methods on types           | full   | `.get()`, `.set()`, `.size()`, etc.                |
| User-defined types (UDT)   | full   | Via PyneComp compilation                           |
| Enums                      | full   | Via PyneComp compilation                           |
| Libraries                  | full   | Import and use Pine libraries                      |
| Operator overloading       | full   | `+`, `-`, `*`, `/`, `%`, comparisons               |
| Ternary operator           | full   | Python conditional expression                      |
| Type casting               | full   | `int()`, `float()`, `str()`, `bool()`              |
| `na` propagation           | full   | Arithmetic, comparison, logical                    |
| Multi-return (tuples)      | full   |                                                    |
| Default parameters         | full   |                                                    |
| `export` (libraries)       | full   | Via PyneComp compilation                           |

## Not Applicable to PyneCore

These Pine Script features exist only in TradingView's live charting environment and have no
equivalent in PyneCore:

| Feature              | Reason                                                     |
|----------------------|------------------------------------------------------------|
| Chart rendering      | No built-in chart — output is CSV, and NDJSON with `--viz` |
| `input()` UI widgets | Inputs are function parameters or TOML config              |

## No lookahead, ever

PyneCore is a backtest and live-trading runtime, not a chart-analysis tool. Where TradingView
hands a script data the bar could not have known yet, **PyneCore diverges from TradingView on
purpose** and returns what was actually knowable at that moment.

The reasoning is asymmetric. On a chart, repaint is visible and recoverable — you watch it
happen and re-read the chart. In a backtest it silently inflates results, and in a live bot it
produces decisions the market never supported. A loud, safe divergence beats a quiet, dangerous
match, so this rule outranks TV parity everywhere it applies.

| Situation                                                               | TradingView                                    | PyneCore                                                   |
|-------------------------------------------------------------------------|------------------------------------------------|------------------------------------------------------------|
| `lookahead_on`, bare `close`, inside an open HTF period                 | the period's FINAL close and high              | the period as built up to the current chart bar            |
| `lookahead_on` with the `close[1]` idiom                                | the just-closed prior period                   | identical — no divergence                                  |
| `lookahead_off` / `lookahead_last_closed`                               | the last CLOSED period                         | identical — no divergence                                  |
| Cross-symbol HTF `lookahead_on` inside an open period                   | the developing bar                             | `na` (nothing can be aggregated from the wrong instrument) |
| An intra-bar pass (`calc_on_order_fills`, `calc_on_every_history_tick`) | the bar's COMPLETED open/high/low/close/volume | the bar as built up to that pass's point in it             |

Note what is *not* affected: the canonical daily-pivot idiom
`request.security(sym, "D", close[1], lookahead_on)` lives entirely in `close[1]`, reads a period
that has genuinely closed, and matches TradingView exactly. Only the bare form — which is future
data on every chart bar except a period's last — differs.

`barmerge.lookahead_last_closed` is a PyneSys-native mode for stating "last closed" intent
explicitly, without relying on the TV `close[1]` idiom at all.

The same rule governs every body execution that stands mid-bar. `calc_on_order_fills` re-runs
the body when an order fills, and `calc_on_every_history_tick` runs it at every point of the
bar the broker emulator walks: without the magnifier the four assumed nodes — open, the extreme
nearest the open, the other extreme, close — and with it, the end of every sub-bar. TradingView
hands all of those passes the bar's finished OHLCV, so a pass standing at the open can already
read the high that only happens later; PyneCore gives each pass the bar truncated to its own
point, with `volume` accrued to match and `hl2` / `hlc3` / `ohlc4` / `hlcc4` recomputed from the
truncated values. Only the bar's last, definitive execution sees the completed bar — which is
also the only execution a strategy without these flags ever gets, so default strategies are
unaffected.

## Known TradingView quirks not reproduced

Behaviour that looks like a TradingView defect rather than a rule is documented here instead
of being copied. Each entry says where to look if a script ever turns out to depend on it.

### Session mask on the day before a fall DST change

On the day BEFORE a timezone's fall clock change — where no offset shifts at all — TradingView
appends one extra hour to the intraday close of every session run that has an endpoint on the
changing wall-clock hour (02:00 in `America/New_York` and `Europe/London`, 03:00 in
`Australia/Sydney`). On the first chart bar of that extra hour `time(tf, session, tz)` returns
`na` while `time_close(tf, session, tz)` returns a value, so the same bar is out of and in the
session at once. PyneCore returns `na` from both.

Measured on a 60-minute chart, 2024-01 to 2026-09, identically on BINANCE:BTCUSDT,
COINBASE:BTCUSD, BITSTAMP:BTCUSD, KRAKEN:XBTUSD and CAPITALCOM:BTCUSD, so it does not depend on
the data source:

| Session, timezone              | Bars where `time()` is `na` and `time_close()` is not (UTC) |
|--------------------------------|-------------------------------------------------------------|
| `"1700-0200"` America/New_York | 2024-11-02 06:00, 2025-11-01 06:00                          |
| `"0200-1000"` America/New_York | 2024-11-02 14:00, 2025-11-01 14:00                          |
| `"0900-1600"` America/New_York | none — no endpoint on the changing hour                     |
| `"1700-0200"` Europe/London    | 2024-10-26 01:00, 2025-10-25 01:00                          |
| `"0200-1000"` Europe/London    | 2024-10-26 09:00, 2025-10-25 09:00                          |
| `"1700-0300"` Australia/Sydney | 2024-04-05 16:00, 2025-04-04 16:00, 2026-04-03 16:00        |

It can only show on an instrument that trades on that day (the day before the change is a
Saturday, or a Friday evening UTC for Sydney), which rules out exchange-traded and FX symbols,
and only in a script that reads `time_close()` with such a session: the usual
`na(time(tf, session, tz))` test agrees with TradingView on that bar. It is one bar per
affected session per year.

Where to look: `_session_occurrences_opening_on` and `_intraday_session_bounds` in
`pynecore/lib/__init__.py`. Both functions read one session occurrence per bar, so reproducing
the quirk needs a rule of its own for `time_close()` on that single bar.

## Precision

What "matches TradingView" means, measured on the validation corpus (snapshot 2026-09-23):

- **Plotted values**: 99.714% of 99,018,068 values are identical to TradingView to the bit. In 687
  of the 712 compared plot outputs every single value is bit-identical. The largest relative gap
  anywhere is 1.2 × 10⁻¹⁰ (about 9 significant figures).
- **Strategy trades**: 289,074 trades in 378 strategies; entry and exit timing and trade counts
  match TradingView in every one of them.
- **OHLCV data**: the v2 `.ohlcv` format stores int64 millisecond timestamps, float64 open and
  volume, and high/low/close as float32 deltas from the open — promoted to float64 whenever a
  delta cannot hold the symbol's tick grid exactly — and snaps prices back to the tick grid on
  read.

### Where the remaining 0.3% comes from

IEEE-754 leaves implementations free to differ on operation ordering and internal precision, so
two correct implementations of the same formula can produce results that differ by a few ULPs.
The values that are not bit-identical differ in their last bits: the largest relative gap in the
whole corpus is 1.2 × 10⁻¹⁰, and typical disagreements are 1e-15 to 1e-12 absolute, adding up to
at most a fraction of a percent equity drift over thousands of trades.

The disagreement is invisible in arithmetic but matters at **exact-equality comparisons**, where
one side may see `a == b` while the other sees `a` slightly above or below `b`. At those
sub-tick scales the difference is numerical noise, not a real trading signal. The functions
most exposed to it are `ta.crossover` and `ta.crossunder`, which follow Pine's strict-comparison
spec (`>` / `<=`) and therefore inherit any boundary disagreement TradingView and PyneCore have
on the same bar. Any user code that compares two computed series with `==`, `!=`, `>`, `<` etc.
without an explicit tolerance is subject to the same kind of ULP-level disagreement; if the
result matters, round the operands to a meaningful number of decimal places or use
`math.isclose`.

## Resolved limitations

Limitations that older pages, issues and articles may still mention, with the release that
closed them:

- **v6.4.1** — bar magnifier and `calc_on_order_fills`.
- **v6.5.1** — multi-period D/W/M aggregation (`2D`, `3W`, ...) is trading-day and holiday aware
  ([#65](https://github.com/PyneSys/pynecore/issues/65)).
- **v6.5.2** — multi-day `request.security` exposes the last confirmed period, not the developing
  one ([#70](https://github.com/PyneSys/pynecore/issues/70)).
- **v6.5.3** — nD/nW/nM aggregation follows exchange holidays and session schedules, e.g. on CME
  futures ([#71](https://github.com/PyneSys/pynecore/issues/71)).
- **v6.6.0** — live data (`pyne run --live`), broker trading (`pyne run --broker`) and realtime
  `barstate` flags; plot and drawing export (`pyne run --viz`).
