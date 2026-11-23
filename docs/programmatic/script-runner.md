<!--
---
weight: 801
title: "ScriptRunner API"
description: "Run PyneCore scripts from Python with ScriptRunner. Configure OHLCV data, symbol metadata and script inputs, then access indicator values and strategy results."
icon: "play_circle"
date: "2025-03-31"
lastmod: "2026-11-14"
draft: false
toc: true
categories: ["Programmatic"]
tags: ["script-runner", "run-iter", "indicators", "strategies", "trades", "inputs", "settings"]
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
    mintick=0.01, pricescale=100, minmove=1, pointvalue=1.0, mincontract=0.00001,
    timezone="UTC", volumetype="base",
    opening_hours=[], session_starts=[], session_ends=[],
)

candles = [
    OHLCV(timestamp=1704067200000, open=42000, high=42500, low=41800, close=42300, volume=1000),
    OHLCV(timestamp=1704070800000, open=42300, high=42800, low=42100, close=42600, volume=1200),
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
    viz_path: Path | None = None,
    viz_journal: bool = False,
    update_syminfo_every_run: bool = False,
    last_bar_index: int | None = None,
    last_bar_time: int | None = None,
    inputs: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    security_data: dict[str, str | Path] | None = None,
    magnifier_iter: Iterable[OHLCV] | None = None,
    magnifier_source_tf: str | None = None,
    config_dir: Path | None = None,
    profile_path: Path | None = None,
    # ... plus the live / broker parameters, see below
)
```

### Parameters

| Parameter                  | Type               | Description                                                    |
|----------------------------|--------------------|----------------------------------------------------------------|
| `script_path`              | `Path`             | Path to a compiled PyneCore script (`.py` with `@pyne` marker) |
| `ohlcv_iter`               | `Iterable[OHLCV]`  | Any iterable of OHLCV objects — list, generator, reader, etc.  |
| `syminfo`                  | `SymInfo`          | Symbol information (from TOML or manually created)             |
| `plot_path`                | `Path \| None`     | Save indicator plot data to CSV                                |
| `strat_path`               | `Path \| None`     | Save strategy statistics to CSV                                |
| `trade_path`               | `Path \| None`     | Save trade-by-trade data to CSV                                |
| `viz_path`                 | `Path \| None`     | Write plot-style + drawing visual data as NDJSON (see below)   |
| `viz_journal`              | `bool`             | Emit per-bar drawing create/update/delete events               |
| `update_syminfo_every_run` | `bool`             | Re-apply syminfo before each bar (for parallel runners)        |
| `last_bar_index`           | `int \| None`      | Index of the last historical bar, for `last_bar_index`         |
| `last_bar_time`            | `int \| None`      | Time (ms) of the last historical bar, for `last_bar_time`      |
| `live`                     | `bool`             | Feed is history, then live updates (see Live Mode)             |
| `live_history_trades`      | `bool`             | Live paper trading: trade on the warmup bars too (Live Mode)   |
| `history_as_of`            | `int \| None`      | Time (ms) the warmup history was downloaded, at the latest     |
| `inputs`                   | `dict \| None`     | Override `input()` values at runtime (see below)               |
| `settings`                 | `dict \| None`     | Override script settings at runtime (see below)                |
| `security_data`            | `dict \| None`     | OHLCV paths for `request.security()` contexts (see below)      |
| `magnifier_iter`           | `Iterable \| None` | LTF OHLCV bars for bar magnifier mode (intrabar simulation)    |
| `magnifier_source_tf`      | `str \| None`      | Timeframe of the `magnifier_iter` bars                         |
| `config_dir`               | `Path \| None`     | Workdir `config/` folder, for its `symbol_map.toml`            |
| `profile_path`             | `Path \| None`     | Profile the run and write the profile JSON here (see below)    |

On historical bars Pine fixes `last_bar_index` and `last_bar_time` to the chart's final bar. With a
list or tuple `ohlcv_iter` outside live mode the runner takes both from its last bar, so they need
to be passed only for any other iterable (a generator, a reader). Without them they track the
current bar, as on a realtime bar, and `calc_bars_count` has no end to count back from, so every
bar is calculated.

The remaining keyword parameters (`broker_plugin`, `broker_event_loop`, `broker_store_ctx`,
`chart_provider_name`, `chart_provider_instance`, `chart_data_path`, `time_from`, `log_ohlcv`,
`chart_bar_window`) wire up live data providers and broker
trading for `pyne run`. A plain backtest from Python needs none of them; see
[Live Mode](../advanced/live-mode.md) for how live runs work.

### Profiling

`profile_path` profiles a backtest and writes the time and execution count of every line, and
the run's time by category, to that JSON file once the run completes (see
[Profiling](../profiling.md)). It needs Python 3.13 or newer and the compiled profiler
extension: without them, or while another profiler holds Python's profiler slot, the
constructor raises `pynecore.core.profiler.ProfilerUnavailable`. A live or broker run with
`profile_path` raises `ValueError`. A run that fails or is closed early writes no profile.

```python
runner = ScriptRunner(script_path, ohlcv, syminfo, profile_path=Path("output/my_profile.json"))
runner.run()
```

### Overriding Inputs

The `inputs` parameter lets you change script parameters without editing the script file:

```python
# In the script:
#   def main(length=input.int(14, "Length"), confirm=input.int(2, "Confirm bars")):

runner = ScriptRunner(
    script_path=Path("sma_crossover.py"),
    ohlcv_iter=candles,
    syminfo=syminfo,
    inputs={"length": 20, "confirm": 3},  # override input() values
)
```

Keys are the **parameter names** of `main()` (`length`), not the `title` shown in the settings
dialog (`"Length"`). This is the same name the input's `[inputs.<name>]` section uses in the
script's `.toml`. A key that matches no input raises `ValueError`, which lists the script's
input names, so a typo or a title used as a key fails loudly.

### Overriding Script Settings

The `settings` parameter overrides the script's own settings: the arguments of its
`@script.indicator(...)` / `@script.strategy(...)` decorator, the same fields the `[script]`
section of its `.toml` holds. Use it to backtest a strategy with a different capital, commission
or sizing without touching the script:

```python
from pynecore.lib import strategy

runner = ScriptRunner(
    script_path=Path("sma_crossover.py"),
    ohlcv_iter=candles,
    syminfo=syminfo,
    settings={
        "initial_capital": 50_000,
        "commission_type": strategy.commission.cash_per_order,
        "commission_value": 2,
        "default_qty_type": strategy.fixed,
        "default_qty_value": 1,
        "pyramiding": 3,
    },
)
```

- Keys are the decorator argument names: `initial_capital`, `currency`, `default_qty_type`,
  `default_qty_value`, `pyramiding`, `commission_type`, `commission_value`, `slippage`,
  `margin_long`, `margin_short`, `process_orders_on_close`, `close_entries_rule`,
  `calc_on_order_fills`, `calc_on_every_tick`, `use_bar_magnifier`, `max_bars_back`,
  `timeframe`, and so on. `runner.script.settable_fields()` lists every one of them.
- Constant-valued settings take the `strategy.*` constants or their string values
  (`strategy.fixed` or `"fixed"`). `commission_type` also takes `strategy.cash`, the same as in
  the decorator.
- `pyramiding` is at least 1: `0` means 1, as in the decorator.
- A key that is not a setting raises `ValueError`, so a typo such as `"initial_capitol"` fails
  loudly instead of being ignored.
- Title, script type and the script's inputs are not settings; inputs go in `inputs`.

### Precedence and the Script's `.toml` File

A script's settings and inputs come from three places, each overriding the one before:

1. the script itself: the decorator arguments and the `input()` defaults,
2. the `.toml` file next to the script (`my_strategy.toml` for `my_strategy.py`),
3. the `inputs` and `settings` you pass to `ScriptRunner`.

Programmatic overrides apply to **that run only**. They are never written into the `.toml`, so
a parameter sweep leaves the file as it was, and a later run without overrides gets the `.toml`
values again.

Importing a script (re)writes its `.toml`, though, with every available setting and input
listed (unchanged ones commented out). That keeps the file a complete, self-documenting
template, and it is what you get when you run a script for the first time. To keep a read-only
script folder untouched, set the `PYNE_SAVE_SCRIPT_TOML=0` environment variable before the
runner is created. Under `pytest` the file is never written.

Overrides are applied to an indicator or strategy script, never to a library it imports. When
the script uses `request.security()`, every security context runs with the same `inputs` and
`settings` as the chart.

### Reading the Effective Settings

`runner.script` is the script's settings object, after the `.toml` and your overrides:

```python
script = runner.script
print(script.initial_capital)            # the value this run uses
print(script.default("initial_capital"))  # what the decorator declares
print(script.settable_fields())          # every setting name settings= accepts
```

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

## Importing a Script Without Running It

`import_script()` loads a script the way `ScriptRunner` does, without processing any bars. Tools
that only need the script's declarations (a settings form, a validator, a `.toml` generator) use
it directly:

```python
from pathlib import Path
from pynecore.core.script_runner import import_script

module = import_script(Path("sma_crossover.py"),
                       inputs={"length": 20}, settings={"initial_capital": 50_000})
script = module.main.script

for name, data in script.inputs.items():   # every input(): type, defval, title, ...
    print(name, data.input_type, data.defval, data.title)
for key in script.settable_fields():
    print(key, script.default(key), getattr(script, key))
```

`inputs` and `settings` work as in `ScriptRunner`. With `save_overrides=True` they are also
written into the script's `.toml` (a value equal to the script's own declaration is written
commented out), which is how a settings editor saves what the user entered. The file is only
written when `PYNE_SAVE_SCRIPT_TOML` is not `0`.

> **Note:** each call executes the script file from its first statement, as a fresh module.

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

### Keeping Results vs. the Fast Path

Every bar yields its own `plot_data` dict and `new_trades` list, so the results can be kept as
they are, for example to collect a whole run at once:

```python
results = list(runner.run_iter())  # each element keeps its own bar's values
```

Internally the runner fills one dict and one list, and `run_iter()` hands out a copy of them on
every bar. When the loop body reads each bar's values before asking for the next one, the copy
is not needed, and `copy_results=False` skips it:

```python
closes = []
for candle, plot_data in runner.run_iter(copy_results=False):
    closes.append(plot_data["Close"])  # read now: plot_data is refilled for the next bar
```

With `copy_results=False` the yielded `plot_data` and `new_trades` are the runner's own
containers, emptied and refilled on every bar. Keeping them (rather than the values read out of
them) leaves you with empty containers. The `Trade` objects themselves stay valid either way.
The copy costs a fraction of a microsecond per bar, which only shows on very light scripts.

### NA Values During Warmup

During the warmup period (first N bars where the indicator doesn't have enough data), numeric
plot values are `na`. A numeric `na` is a native `float('nan')`, so your code receives an ordinary
Python float and Python's rules apply to it, not Pine's:

- **Ordered comparisons** are `False`: `nan < 30` and `nan > 70` are both `False`
- **`!=` is `True`**: `nan != 30` (and even `nan != nan`) is `True`
- **Truthiness**: `bool(nan)` is `True`, so `if rsi:` does **not** skip a missing value
- **Arithmetic** propagates: `nan + 1` is `nan`
- **Formatting** prints `nan`: `f"{nan:.2f}"` is `"nan"`

Test for a missing value explicitly with `math.isnan()`:

```python
import math

for candle, plot_data in runner.run_iter():
    rsi = plot_data.get("RSI")

    if math.isnan(rsi):  # warmup: the RSI has no value yet
        continue

    if rsi > 70:
        print(f"Overbought: RSI={rsi:.2f}")
```

## Live Mode

By default every bar is history. The runner reads one item ahead of the bar it executes, because
`barstate.islast` and the confirmation of higher-timeframe `request.security()` values depend on
whether another bar follows. A list or a file costs nothing extra this way. A generator that
waits for the market does: each bar's result only comes out when the next candle arrives, one
candle late.

With `live=True` the iterator yields three phases:

1. The historical (warmup) bars, closed, as in a backtest.
2. `LIVE_TRANSITION`, once, when the history is done.
3. The live updates. An `OHLCV` with `is_closed=False` is a price update of the forming bar, one
   with `is_closed=True` closes it. A new timestamp opens a new bar.

```python
from pynecore.core.script_runner import ScriptRunner, LIVE_TRANSITION
from pynecore.types.ohlcv import OHLCV


def feed():
    yield from history          # closed bars, oldest first
    yield LIVE_TRANSITION
    while True:
        bar = wait_for_closed_candle()  # your exchange / websocket code
        yield OHLCV(timestamp=bar.ts, open=bar.o, high=bar.h, low=bar.l, close=bar.c,
                    volume=bar.v, is_closed=True)


runner = ScriptRunner(Path("rsi.py"), feed(), syminfo, live=True)
for candle, plot_data in runner.run_iter():
    print(candle.timestamp, plot_data["RSI"])  # printed as soon as the candle closed
```

What live mode changes:

- **No delay.** A closed live bar is executed and yielded as soon as it arrives. Only the warmup
  bars are read one ahead, and the last of them runs as soon as `LIVE_TRANSITION` comes.
- **`barstate`** reports `ishistory` on the warmup bars, `islastconfirmedhistory` on the last of
  them, and `isrealtime` from the transition on.
- **Intra-bar updates.** An indicator runs on every update of the forming bar, a strategy only
  with `calc_on_every_tick=true`. `var` variables roll back between these runs and `varip`
  variables keep their value, as in Pine. `run_iter()` yields on bar close only; pass
  `on_tick=` to see every update, or `yield_ticks=True` to get the forming bar's results too
  (see below).
- **Strategy orders start at the transition.** The warmup bars build the script's state, but
  their `strategy.*` orders are ignored, so the strategy does not start with a position it took
  in the past. Orders are simulated (paper trading) unless a broker plugin is attached.
  `live_history_trades=True` lets a paper-trading strategy trade on the warmup bars too, as a
  backtest the live phase continues, position and all (see below).
- **Feed errors are dropped with a warning.** A bar that would move time backwards (a closed bar
  older than the last one, or an update of a bar that is already closed) is not executed.

A feed that only knows closed candles skips the `is_closed=False` updates. It still has to drop
the candle that is still forming: most REST APIs return it as the last row.

### From a Provider String

`pynecore.core.provider_download` runs a provider the way `pyne run` does, without any terminal
output: `download_provider_history()` downloads the history of a provider string into its OHLCV
file (fresh symbol information, `-N` means N real bars), and `open_live_stream()` connects the
same provider instance and returns the live iterator to chain after the history bars. Progress and
warnings go to a `DownloadListener` you may subclass; failures are raised (`ProviderError`,
`ProviderSetupError`, `ValueError` for a bad start value). `download_provider_history_resilient()`
also rides out transient provider outages, as a long-running live run needs.

The OHLCV file is the one a `pyne data download` of the same symbol and timeframe writes, so a
download replaces it. With `in_memory=True` the history stays in `history.bars` instead, and
nothing is written into `data_dir`, not even the symbol information. The provider still gets
`data_dir`: it finds the workdir's config next to it.

```python
import itertools
from pynecore.core.provider_download import download_provider_history, open_live_stream

history = download_provider_history("ccxt:BYBIT:BTC/USDT:USDT@1", "-500",
                                    data_dir=workdir / "data", config_dir=workdir / "config",
                                    in_memory=True)
bars = [bar for bar in history.bars if bar.timestamp >= (history.time_from_ts or 0)]
live = open_live_stream(history, last_historical_timestamp=bars[-1].timestamp,
                        on_feed_status=show_status)
runner = ScriptRunner(Path("rsi.py"), itertools.chain(bars, live), history.syminfo,
                      live=True)
for candle, plot_data in runner.run_iter(yield_ticks=True):
    ...
```

The connection opens as soon as `open_live_stream()` is called, so no bar is lost while the
history runs: the bars that close meanwhile come right after the history, as closed bars.

### Results of the Forming Bar

`run_iter(yield_ticks=True)` also yields an item for every intra-bar update, so the forming
candle can be drawn on each one. The item carries the update itself, so `candle.is_closed` is
`False`. It is provisional: the next update of the same bar replaces it, and the bar's close is
yielded once more, as without the switch.

When the body executed on the update (every update for an indicator, every update with
`calc_on_every_tick=true` for a strategy), the item carries its results, and a strategy's trade
list holds the trades closed so far on the bar. When it did not, `plot_data` and the trade list
are empty: only the candle is new, and the plots of the bar come with its close. The plot, trade
and viz files are written for closed bars only either way.

```python
for candle, plot_data in runner.run_iter(yield_ticks=True):
    if candle.is_closed:
        commit_bar(candle, plot_data)        # final values of the bar
    else:
        update_forming_bar(candle, plot_data)  # replaced by the next update
```

### Trading on the Warmup Bars

With `live_history_trades=True` a paper-trading strategy runs its `strategy.*` calls on the warmup
bars too: the history is a backtest, and the live phase continues it with whatever position it
left. It is off by default, so a live run starts flat at the transition, and it is not available
with a broker plugin, where only what the broker really trades may enter the record (the
constructor raises `ValueError` for a broker or a non-live run). The last warmup bar places no
orders while its period had not ended yet when the history was downloaded: the live feed goes on
forming that bar, and an order placed on its partial data could fill at a price from before it was
placed. Pass the download time as `history_as_of` (UNIX ms, no later than the download); without
it the runner takes the time it was created.

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

## Visual Output (Plot Styles & Drawings)

The `plot_path` CSV holds only numeric plot values. To also capture plot **styles** (colors, widths,
shapes) and **drawing objects** (lines, labels, boxes, tables, polylines, linefills), enable the viz
output with `viz_path` (and optionally `viz_journal`):

```python
runner = ScriptRunner(
    script_path=Path("my_indicator.py"),
    ohlcv_iter=candles,
    syminfo=syminfo,
    viz_path=Path("output/viz.ndjson"),  # opt-in NDJSON stream
    viz_journal=True,                     # per-bar drawing events
)
```

The runner also exposes the same state programmatically:

| Accessor            | Description                                                          |
|---------------------|----------------------------------------------------------------------|
| `runner.plot_meta`  | `{id: PlotMeta}` — registered plot-family metadata (kept after the run) |
| `runner.drawings()` | Full snapshot of the live drawing objects                            |
| `runner.viz_events` | Optional callback receiving each bar's journal events                |

The plots CSV is unchanged and viz output is entirely opt-in. See
[Visual Output (Viz)](./visual-output.md) for the NDJSON format, the only-on-change color encoding,
ordinal-id semantics, journal mode, and the live `lib._plot_meta` / `lib._viz_dyn` read patterns.

## Complete Example: Strategy with Trade Analysis

```python
from pathlib import Path
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.data_converter import DataConverter
from pynecore.core.ohlcv import OHLCVReader
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
        inputs={"length": 20, "confirm": 2},
        settings={"initial_capital": 10_000, "commission_value": 0.05},
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
