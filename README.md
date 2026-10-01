<div align="center">

<img src="https://raw.githubusercontent.com/PyneSys/pynecore/refs/heads/main/docs/logo.svg" alt="PyneCore Logo">
<h1>PyneCore™</h1>
<strong>Pine Script in Python - Without Limitations</strong>

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%2B-blue" alt="Python"></a>
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>

**Have Pine Script code? [Convert it to Python](#converting-pine-script)**

</div>

## What is PyneCore?

PyneCore runs Pine Script-style trading code in Python. You write ordinary Python, and PyneCore rewrites it at import time with AST transformations, so it executes bar by bar with Pine Script's semantics: series with history, variables that keep their value between bars, and `na` for missing data. The rest of the Python ecosystem stays available to the same code.

PyneCore is tested against TradingView on <!--wild:scripts_total-->920<!--/wild--> published TradingView scripts (Pine Script v4 to v6, converted to Pyne code with [PyneComp](https://pynesys.io)). All <!--wild:tv_verified-->1,241<!--/wild--> comparable outputs match, <!--wild:bars_exact_pct-->99.768<!--/wild-->% of <!--wild:bars_compared_millions-->118<!--/wild--> million plotted values are bit-identical, and <!--wild:strategy_trades-->318,982<!--/wild--> strategy trades match TradingView's timing (snapshot <!--wild:generated_at-->2026-10-01<!--/wild-->). Every script and every result is public at [Pyne in the Wild](https://wild.pynesys.io/), and the [Compatibility](https://pynecore.org/docs/overview/compatibility/) page lists the status of each Pine Script feature.

## Key features

- Pine Script's bar-by-bar execution model in plain Python
- Scripts marked with `@pyne` are transformed when they are imported
- No mandatory dependencies outside the Python standard library
- `Series` variables with bar history and `Persistent` variables that keep state between bars
- Function isolation: every call of a function keeps its own persistent state
- `na` handling that follows Pine Script
- Every `ta.*` function of Pine Script v6
- Pine Script-compatible strategy backtesting

## Quick example

```python
"""
@pyne
"""
from pynecore.lib import script, close, ta, plot, color, input

@script.indicator(title="Bollinger Bands")
def main(
    length=input.int(20, "Length", minval=1),
    mult=input.float(2.0, "Multiplier", minval=0.1, step=0.1),
    src=input.source(close, "Source")
):
    # Calculate Bollinger Bands
    basis = ta.sma(src, length)
    dev = mult * ta.stdev(src, length)

    upper = basis + dev
    lower = basis - dev

    # Output to chart
    plot(basis, "Basis", color=color.orange)
    plot(upper, "Upper", color=color.blue)
    plot(lower, "Lower", color=color.blue)
```

## Core concepts

### The `@pyne` magic comment

A script is marked with a magic comment in its docstring:

```python
"""
@pyne
"""
```

PyneCore's import hook recognizes the marker and applies its AST transformations to the module when it is imported.

### Series variables

A series variable keeps its history across bars, as in Pine Script:

```python
from pynecore import Series

price: Series[float] = close
previous_price = price[1]  # Access previous bar's price
```

### Persistent variables

A `Persistent` annotation keeps a variable's value from one bar to the next:

```python
from pynecore import Persistent

counter: Persistent[int] = 0
counter += 1  # Increments with each bar
```

### Function isolation

Every call of a function keeps its own persistent state:

```python
def my_indicator(src, length):
    # Each call gets its own instance of total
    total: Persistent[float] = 0
    total += src
    return total / length
```

## Installation

```bash
# Basic installation
pip install pynesys-pynecore

# With CLI tools (recommended)
pip install pynesys-pynecore[cli]

# With all features including data providers
pip install pynesys-pynecore[all]
```

> **Windows:** PyneCore needs timezone data that Windows does not ship. The `[cli]` and `[all]` extras install the `tzdata` package. With the basic installation, run `pip install tzdata` if you get timezone errors.

## Getting started

### Create a simple script

1. Create a file with the `@pyne` annotation:

```python
"""
@pyne
"""
from pynecore.lib import script, close, plot

@script.indicator("My First Indicator")
def main():
    # Calculate a simple moving average
    sma_value = (close + close[1] + close[2]) / 3

    # Plot the result
    plot(sma_value, "Simple Moving Average")
```

2. Run it with the PyneCore CLI:

```bash
# First, download some price data
pyne data download ccxt --symbol "BYBIT:BTC/USDT:USDT"

# Then run your script on the data
pyne run my_script.py ccxt_BYBIT_BTC_USDT_USDT_1D.ohlcv
```

### Running Pine Script files

PyneCore runs Pyne code. Existing Pine Script (v4, v5 and v6, and v1 to v3 on a best-effort basis) is converted to Pyne code by [PyneComp](https://pynesys.io), a separate PyneSys service that needs an API key. With the key, the `pyne` CLI compiles `.pine` files for you:

```bash
# Run a Pine Script file directly (requires PyneSys API key)
pyne run my_indicator.pine ccxt_BYBIT_BTC_USDT_USDT_1D.ohlcv --api-key YOUR_API_KEY

# Or compile Pine Script to Python first
pyne compile my_indicator.pine --api-key YOUR_API_KEY

# Then run the compiled Python file
pyne run my_indicator.py ccxt_BYBIT_BTC_USDT_USDT_1D.ohlcv
```

You can get an API key at [pynesys.io](https://pynesys.io).

## Why PyneCore?

- Your scripts run outside TradingView, without its platform restrictions or code size limits.
- Python's data science, machine learning and analysis libraries work in the same code as the trading logic.
- The match with TradingView is measured on [real published scripts](https://wild.pynesys.io/): <!--wild:bars_exact_pct-->99.768<!--/wild-->% of <!--wild:bars_compared_millions-->118<!--/wild--> million plotted values are identical to the bit.
- The runtime and its library are open source under the Apache 2.0 license.

## Converting Pine Script

If you have Pine Script code you want to run in Python, PyneComp converts it to Pyne code. You can use it on the [web](https://pynesys.io), from the `pyne` CLI with an API key, or through the [PyneSys Discord bot](https://discord.pynesys.io): `/pyne-help` shows how the bot works, `/pyne-convert` converts a script, and every Discord user gets 3 free conversions.

The subscription plans on [pynesys.io](https://pynesys.io) raise the daily conversion limit and the maximum script size. They also fund the development of PyneCore, which stays free and open source.

## Documentation and support

- Documentation: [pynecore.org](https://pynecore.org/docs)
- Validation report: [Pyne in the Wild](https://wild.pynesys.io/), where every corpus script is compared with TradingView

### Community

- Discussions: [GitHub Discussions](https://github.com/pynesys/pynecore/discussions)
- Discord: [discord.pynesys.io](https://discord.pynesys.io)
- X: [x.com/pynesys](https://x.com/pynesys)
- Website: [pynecore.org](https://pynecore.org)

## License

PyneCore is licensed under the [Apache License 2.0](LICENSE).

## Disclaimer

Pine Script™ is a trademark of TradingView, Inc. PyneCore is not affiliated with, endorsed by, or sponsored by TradingView. This project is an independent implementation of the Pine Script language concept in Python; its results are measured against TradingView on [Pyne in the Wild](https://wild.pynesys.io/).

### Risk warning

Trading involves significant risk of loss and is not suitable for all investors. The use of PyneCore does not guarantee any specific results. Past performance is not indicative of future results.

- PyneCore is provided "as is" without any warranty of any kind
- PyneCore is not a trading advisor and does not provide trading advice
- Scripts created with PyneCore should be thoroughly tested before using with real funds
- Users are responsible for their own trading decisions
- You should consult with a licensed financial advisor before making any financial decisions

By using PyneCore, you acknowledge that you are using the software at your own risk. The creators and contributors of PyneCore shall not be held liable for any financial loss or damage resulting from the use of this software.

## Commercial support

PyneCore is part of the PyneSys ecosystem. For commercial support, custom development or enterprise solutions, contact us at [pynesys.com/contact](https://pynesys.com/contact).
