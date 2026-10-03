<!--
---
weight: 801
title: "ScriptRunner API"
description: "Running PyneCore scripts programmatically from Python"
icon: "play_circle"
date: "2025-03-31"
lastmod: "2026-10-03"
draft: false
toc: true
categories: ["Programmatic", "API"]
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
    last_bar_index: int = 0,
    last_bar_time: int | None = None,
    inputs: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    security_data: dict[str, str | Path] | None = None,
    magnifier_iter: Iterable[OHLCV] | None = None,
    magnifier_source_tf: str | None = None,
    config_dir: Path | None = None,
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
| `last_bar_index`           | `int`              | Override last bar index (for multi-script setups)              |
| `last_bar_time`            | `int \| None`      | Time (ms) of the last historical bar, for `last_bar_time`      |
| `inputs`                   | `dict \| None`     | Override `input()` values at runtime (see below)               |
| `settings`                 | `dict \| None`     | Override script settings at runtime (see below)                |
| `security_data`            | `dict \| None`     | OHLCV paths for `request.security()` contexts (see below)      |
| `magnifier_iter`           | `Iterable \| None` | LTF OHLCV bars for bar magnifier mode (intrabar simulation)    |
| `magnifier_source_tf`      | `str \| None`      | Timeframe of the `magnifier_iter` bars                         |
| `config_dir`               | `Path \| None`     | Workdir `config/` folder, for its `symbol_map.toml`            |

`last_bar_time` matters for scripts that read `last_bar_time` on historical bars: Pine fixes it to
the chart's final bar, while `None` makes it track the current bar (live semantics).

The remaining keyword parameters (`broker_plugin`, `broker_event_loop`, `broker_store_ctx`,
`chart_provider_name`, `chart_provider_instance`, `chart_data_path`, `time_from`, `log_ohlcv`,
`lossless_volume`, `lossless_prices`, `chart_bar_window`) wire up live data providers and broker
trading for `pyne run`. A plain backtest from Python needs none of them; see
[Live Mode](../advanced/live-mode.md) for how live runs work.

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
script's `.toml`. A key that matches no input is silently ignored, so check the spelling when a
value does not seem to apply.

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
