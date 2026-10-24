<!--
---
weight: 1200
title: "FAQ"
description: "Frequently asked questions about PyneCore"
icon: "psychology"
date: "2025-03-31"
lastmod: "2026-10-18"
draft: false
toc: true
categories: ["Support"]
tags: ["faq", "troubleshooting", "installation", "configuration", "pyne-code", "dsl", "pyne-edge", "ide", "viz"]
---
-->

# FAQ

This section contains frequently asked questions about PyneCore.

## General Questions

### What is PyneCore?

PyneCore is the open-source runtime for **Pyne code**, a domain-specific language for
market data, indicators, and trading strategies. It executes programs bar by bar and
provides the time-series, persistent-state, technical-analysis, and strategy machinery.
Its current implementation runs in Python.

### What is Pyne code? Is it a separate language?

Yes. Pyne is a domain-specific language (DSL) with **Python syntax** and its own
execution rules. You use familiar Python notation to describe calculations for the
current candle, while PyneCore manages series history, retained state, and stateful
function calls. A Pyne file uses the `.py` extension and begins with an `@pyne` module
docstring; running the same statements as ordinary Python does not provide Pyne's
behavior automatically.

The [Pyne code overview](./overview/pyne-code.md) explains the language and includes
a complete example.

### What is the difference between Pyne code, PyneCore, PyneIDE, and PyneComp?

- **Pyne code** is the language you write.
- **PyneCore** prepares and executes it.
- **PyneIDE** is the Visual Studio Code extension, currently in preview, for editing,
  running, charting, and debugging programs.
- **PyneComp** converts existing Pine Script into Pyne code through a separate service.

You can write Pyne directly without using Pine Script or PyneComp.

### What does `@pyne edge` mean?

It selects a restricted Pyne language profile with defined syntax, imports, and calls.
PyneIDE reports code outside that profile. The beginner's book uses `@pyne edge`;
it does not allow arbitrary Python packages or every Python language feature.
Plain `@pyne` allows broader Python integration in PyneCore. See
[Pyne code — edge](./overview/pyne-code.md#what-does-edge-mean).

### Do I need to know Python or Pine Script first?

No. The [beginner's book](https://pynecore.org/learn/en/) teaches Pyne code from the beginning through
PyneIDE, including the Python syntax used by its examples. If you already have a
Pine Script, you can instead follow the
[conversion guide](./getting-started/converting-from-pine.md).

### How does PyneCore relate to TradingView?

PyneCore is not affiliated with or endorsed by TradingView. It is an independent project that
implements Pine Script's functionality and execution model in Python. PyneCore is validated against TradingView on <!--wild:scripts_total-->1,000<!--/wild--> published TradingView scripts (Pine Script v4–v6, converted to Pyne code with [PyneComp](https://pynesys.io)): all <!--wild:tv_verified-->1,353<!--/wild--> comparable outputs match, <!--wild:bars_exact_pct-->99.770<!--/wild-->% of <!--wild:bars_compared_millions-->132<!--/wild--> million plotted values are bit-identical, and <!--wild:strategy_trades-->353,577<!--/wild--> strategy trades match TradingView's timing (snapshot <!--wild:generated_at-->2026-10-23<!--/wild-->, [Pyne in the Wild](https://wild.pynesys.io/)). See [Compatibility](./overview/compatibility.md) for details.

### Is PyneCore free to use?

Yes. PyneCore is open-source under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).
Writing and running Pyne code locally does not require a PyneSys API key. Converting
Pine Script through PyneComp is a separate service and requires a PyneSys API key;
see [Compiling Pine Scripts](./cli/compile.md) for configuration.

### What are the system requirements for PyneCore?

PyneCore requires Python 3.11 or newer. It is designed to work on Windows, macOS, and Linux operating systems. The core system has no external dependencies, though some additional features may require specific packages (e.g. command line tools).

### How do I report bugs or request features?

You can report bugs and request features through the project's [GitHub repository issue tracker](https://github.com/PyneSys/pynecore/issues). Please include detailed information about the issue, steps to reproduce it, and your system environment.

## Installation & Setup

### How do I install PyneCore?

The recommended way to install PyneCore is using pip:

```bash
pip install "pynesys-pynecore[cli]"  # With user-friendly CLI
```

For additional features, you can specify optional dependencies:

```bash
pip install "pynesys-pynecore[cli,providers]"  # With data providers
pip install "pynesys-pynecore[all]"  # With all features
```

For detailed installation instructions, see the [Installation Guide](./getting-started/installation.md).

### How do I set up a working directory?

PyneCore uses a "workdir" directory structure to organize scripts, data, and configuration. When you run PyneCore, it automatically searches for a "workdir" directory in the current or parent directories. The basic structure is:

```
workdir/
├── scripts/     # Your Pyne code
├── data/        # OHLCV data files
├── output/      # Output files
└── config/      # Configuration files
```

The PyneCore CLI will create this structure automatically if it doesn't exist.

### How do I use PyneCore with my existing data?

PyneCore supports importing data from various formats:

```bash
pyne data convert-from path/to/your/data.csv --symbol "BTCUSDT"
```

You can also download data from supported providers:

```bash
pyne data download ccxt:BYBIT:BTC/USDT:USDT@1D
```

The built-in CCXT provider requires the `providers` or `ccxt` extra. Additional data
providers and brokers are available as [plugins](https://pynecore.org/plugins/). See
[Data Management](./cli/data.md) for supported formats and symbol metadata.

## Writing Scripts

### How do I start a PyneCore script?

Every PyneCore script must begin with a module docstring whose first non-whitespace token is `@pyne`. This marker tells the import hook to treat the file as Pyne code and apply the AST transformations:

```python
"""
@pyne
"""
```

After this docstring, import from PyneCore and define the script's main function:

```python
from pynecore.lib import script, close

@script.indicator("My Indicator")
def main():
    return {"Close": close}
```

### What's the difference between Pyne code and regular Python code?

PyneCore transforms marked Pyne modules before execution. Python supplies the syntax
and runtime, while Pyne supplies additional semantics:

- Series variables that track historical values
- Persistent variables that maintain state across bars
- Bar-by-bar execution model
- Function isolation (stateful call sites maintain separate state)
- Special handling for NA (Not Available) values

### How do I access historical values in a series?

You can access historical values using standard array indexing syntax:

```python
current_price = close
previous_price = close[1]  # Previous bar's close
two_bars_ago = close[2]    # Close from two bars ago
```

### How do I define user-configurable inputs?

In PyneCore, you can define inputs as arguments to your main function:

```python
from pynecore.lib import script, input, close, ta, plot, color

@script.indicator("Simple Moving Average")
def main(
    length: int = input.int(20, "Period", minval=1),
    line_color = input.color(color.blue, "Line Color")
):
    sma = ta.sma(close, length)
    plot(sma, "SMA", color=line_color)
```

### How do I plot indicators?

PyneCore produces calculation results and visual output; **PyneIDE displays them on
a chart**, including supported plot styles and drawing objects. CLI users can save
plot values as CSV and enable `--viz` for visual metadata and drawing data in NDJSON.
See [Visual Output](./programmatic/visual-output.md).

You can emit numeric plot values in two ways:

1. Using the plot function (similar to Pine Script):
```python
plot(sma, "SMA", color=color.blue)
```

2. Using the return value of the main function (more Pythonic):
```python
return {
    "SMA": sma,  # Title: value
}
```

You can use either or both methods (even in the same script) based on your preference.

## Compatibility and Technical Details

### How reliable is PyneCore compared to TradingView?

PyneCore's results are measured against TradingView on a public validation corpus,
[Pyne in the Wild](https://wild.pynesys.io/). The snapshot of <!--wild:generated_at-->2026-10-23<!--/wild--> covers
<!--wild:scripts_total-->1,000<!--/wild--> published open-source TradingView scripts (<!--wild:indicators-->502<!--/wild--> indicators,
<!--wild:strategies-->498<!--/wild--> strategies), written in Pine Script v4 (<!--wild:pine_v4-->329<!--/wild-->), v5 (<!--wild:pine_v5-->364<!--/wild-->) and v6
(<!--wild:pine_v6-->307<!--/wild-->), converted to Pyne code with [PyneComp](https://pynesys.io) and tested on
<!--wild:symbol-->BINANCE:BTCUSDT<!--/wild--> <!--wild:timeframe_min-->30<!--/wild-->-minute bars:

- All <!--wild:scripts_total-->1,000<!--/wild--> scripts run.
- All <!--wild:tv_verified-->1,353<!--/wild--> outputs that can be compared with TradingView are verified
  (<!--wild:plot_outputs-->885<!--/wild--> plot outputs and <!--wild:trade_outputs-->468<!--/wild--> strategy trade lists).
- Of <!--wild:bars_compared-->131,743,106<!--/wild--> plotted values, <!--wild:bars_exact_pct-->99.770<!--/wild-->% are identical to
  TradingView to the bit; the rest are within the published band. The largest relative gap
  anywhere is <!--wild:max_rel-->6.3 × 10⁻¹²<!--/wild--> (about <!--wild:max_rel_sig_figures-->11<!--/wild--> significant figures).
- In all <!--wild:strategies_compared-->468<!--/wild--> strategies with trades, the <!--wild:strategy_trades-->353,577<!--/wild--> trades
  match TradingView's entry and exit timing, and the trade counts are identical.

See [Compatibility](./overview/compatibility.md) for details.

### Is PyneCore 100% compatible with Pine Script?

The measured outputs match on the published validation corpus above. This does not
mean every Pine Script feature, external data feed, or future script is supported.
There are also intentional differences in the language and execution model, documented in the
[Differences from Pine Script](./overview/differences.md) page.
See [Compatibility](./overview/compatibility.md) for the feature-by-feature status.

### How does PyneCore handle NA values?

PyneCore implements a custom NA (Not Available) system similar to Pine Script's. You can check if a value is NA using:

```python
from pynecore.lib import na

if na(value):
    # Handle NA case
```

Use `na(value)` to test availability rather than equality or formatting. Floating-point
missing values use `nan`; Pyne's predicate also treats positive and negative infinity
as unavailable. NaN propagates through arithmetic and its ordered comparisons are
false, but infinity follows normal floating-point comparison rules. Do not assume
all values recognized by `na()` behave identically in a comparison.

Use `nz(value, replacement)` when a calculation needs a replacement value. See
[Core Concepts](./overview/core-concepts.md#5-na-not-available-system) for details.

### How do Series variables work in PyneCore?

Series variables retain previous calculation values in circular buffers. A declaration
such as `average: Series[float] = ta.sma(close, 20)` enables history access through
`average[1]`. PyneCore manages that history in the relevant execution context and
call-site state. See [Series variables](./overview/core-concepts.md#3-series-variables).

### What's the performance like compared to Pine Script?

PyneCore is designed for high performance while maintaining the intuitive bar-by-bar execution model. While vectorized operations in pandas or numpy might be faster for certain calculations, PyneCore offers a good balance between performance and the familiarity of Pine Script's execution model.

### Can I use external Python libraries with PyneCore?

Plain `@pyne` programs can use external Python libraries in PyneCore's Python runtime.
You are responsible for their dependencies and how their operations interact with
bar-by-bar execution. **`@pyne edge` restricts imports and calls**, so arbitrary Python
libraries are outside that profile. You can also keep general Python integration
outside a script and embed it through [ScriptRunner](./programmatic/script-runner.md).

### Why doesn't division by zero raise an exception in PyneCore?

PyneCore automatically handles division by zero to match Pine Script behavior. When you write:

```python
result = numerator / denominator
```

PyneCore transforms numeric division to follow Pine-style floating-point behavior.
A nonzero numerator divided by zero produces signed infinity; `0 / 0` produces `nan`.
The `na()` predicate recognizes these non-finite results, so you can test them without
catching a Python `ZeroDivisionError`. A missing input propagates a missing result.
The generated implementation can inline the common case or use a runtime helper.

## Programmatic Usage

### Can I use PyneCore from Python code, not just the CLI?

Yes! PyneCore scripts can be run programmatically using the `ScriptRunner` class. This lets you
embed indicators and strategies into trading bots, custom backtesting frameworks, data pipelines,
or web services.

```python
from pathlib import Path
from pynecore.core.script_runner import ScriptRunner

runner = ScriptRunner(
    script_path=Path("my_indicator.py"),
    ohlcv_iter=candles,     # any iterable of OHLCV objects
    syminfo=syminfo,        # symbol metadata
)

for candle, plot_data in runner.run_iter():
    rsi = plot_data.get("RSI")
    if rsi < 30:
        print("Oversold!")
```

See the [Programmatic Usage](./programmatic/README.md) guide for full documentation, and the
[pynecore-examples](https://github.com/PyneSys/pynecore-examples) repository for runnable examples
covering CSV data, custom data sources, live exchange feeds, and FreqTrade integration.

### Can PyneCore run scripts on live market data?

Yes. `pyne run --live` streams real-time data from a provider plugin after the historical phase;
strategies run in paper-trading mode with simulated fills. `pyne run --broker` (which implies
`--live`) sends real orders to the exchange through a broker plugin. See
[Live Mode](./advanced/live-mode.md).

## Troubleshooting

### My script isn't being recognized as a PyneCore script

Make sure:
1. Your file's first statement is a module docstring (`"""…"""`)
2. The docstring's first non-whitespace token is `@pyne`, followed by whitespace or the end of the docstring (so `@pynex` won't match, and `@pyne` cannot appear after a description line — it must come first)
3. You have a main() function defined
4. You've imported necessary modules from pynecore.lib

### I'm getting unexpected NA values in my calculations

NA values can propagate through calculations just like in Pine Script. If any input to an operation is NA, the result will typically be NA. Check for NA values using the `na()` function and provide appropriate default values if needed.

### My script runs differently than the Pine Script version

PyneCore's results match TradingView on the public validation corpus (see
[How reliable is PyneCore compared to TradingView?](#how-reliable-is-pynecore-compared-to-tradingview)).
When a result differs, check:

1. Different input data: TradingView and your data source may have different bars, and a
   comparison is only meaningful on identical OHLCV data
2. Data feeds PyneCore has no source for: `request.dividends()`, `request.splits()`,
   `request.earnings()` have no corporate-event feed and return `na` with
   `ignore_invalid_symbol=True`; they also return `na` when requesting the current
   crypto symbol. Other calls raise an error. `request.financial()` returns `na`
   only with `ignore_invalid_symbol=True`; `request.economic()` and
   `request.quandl()` raise an error. See [Request data](./lib/request-data.md).
3. An intentional difference from Pine Script (see
   [Differences from Pine Script](./overview/differences.md))

If you find a discrepancy on identical data, please report it as an issue in the GitHub
repository.

### How do I debug my PyneCore script?

PyneCore provides several debugging approaches. See the [Debugging](./debugging.md) guide for detailed information.
