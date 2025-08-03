<div align="center">

<img src="https://raw.githubusercontent.com/PyneSys/pynecore/refs/heads/main/docs/logo.svg" alt="PyneCore Logo">
<h1>PyneCore™</h1>
<strong>Pine Script in Python - Without Limitations</strong>

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%2B-blue" alt="Python"></a>
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>

**🚀 Have Pine Script code? [Get automatic Python translation →](#pine-script-migration-made-easy)**

</div>

## What is PyneCore?

PyneCore brings TradingView's Pine Script capabilities to Python through a revolutionary approach - it transforms regular Python code to behave like Pine Script through AST transformations, while maintaining the full power of the Python ecosystem.

Instead of creating another object-oriented wrapper or a new language, PyneCore modifies your Python code at import time, giving you the intuitive bar-by-bar execution model of Pine Script without leaving Python's rich environment.

## Key Features

- **Native Pine Script Semantics in Python**: Write familiar Python code that runs with Pine Script's bar-by-bar execution model
- **AST Transformation Magic**: Your code is transformed at import time to implement Pine Script behavior
- **High Performance**: Zero mandatory external dependencies with highly optimized implementation
- **Series & Persistent Variables**: Full support for Pine Script's time series and state persistence
- **Function Isolation**: Each function call gets its own isolated persistent state
- **NA Handling**: Graceful handling of missing data with Pine Script's NA system
- **Technical Analysis Library**: Comprehensive set of Pine Script-compatible indicators and functions
- **Strategy Backtesting**: Pine Script compatible framework for developing and testing trading strategies

## Quick Example

```python
"""
@pyne
"""
from pynecore import Series
from pynecore.lib import script, close, ta, plot, color, input

@script.indicator(title="Bollinger Bands")
def main(
    length=input.int("Length", 20, minval=1),
    mult=input.float("Multiplier", 2.0, minval=0.1, step=0.1),
    src=input.source("Source", close)
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

## Innovative Concepts

PyneCore introduces several revolutionary concepts:

### 1. Magic Comment & Import Hook

Identify your scripts with a simple magic comment:

```python
"""
@pyne
"""
```

This activates PyneCore's import hook system which intercepts Python imports and applies AST transformations to recognized scripts.

### 2. Series Variables

Track historical data across bars, just like in Pine Script:

```python
from pynecore import Series

price: Series[float] = close
previous_price = price[1]  # Access previous bar's price
```

### 3. Persistent Variables

Maintain state between bars with simple type annotations:

```python
from pynecore import Persistent

counter: Persistent[int] = 0
counter += 1  # Increments with each bar
```

### 4. Function Isolation

Each call to a function maintains its own isolated state:

```python
def my_indicator(src, length):
    # Each call gets its own instance of sum
    sum: Persistent[float] = 0
    sum += src
    return sum / length
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

> **Note for Windows users**: PyneCore requires timezone data that is not included in Windows by default. The `[cli]` and `[all]` installations automatically include the `tzdata` package. If you're using the basic installation and encounter timezone errors, install it manually with `pip install tzdata`.

## Getting Started

### Create a Simple Script

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

2. Run your script with the PyneCore CLI:

```bash
# First, download some price data
pyne data download ccxt --symbol "BYBIT:BTC/USDT:USDT"

# Then run your script on the data
pyne run my_script.py ccxt_BYBIT_BTC_USDT_USDT_1D.ohlcv
```

### Working with Pine Script Files

PyneCore also supports running Pine Script files directly with automatic compilation:

```bash
# Run a Pine Script file directly (requires PyneSys API key)
pyne run my_indicator.pine ccxt_BYBIT_BTC_USDT_USDT_1D.ohlcv --api-key YOUR_API_KEY

# Or compile Pine Script to Python first
pyne compile my_indicator.pine --api-key YOUR_API_KEY

# Then run the compiled Python file
pyne run my_indicator.py ccxt_BYBIT_BTC_USDT_USDT_1D.ohlcv
```

> **Note**: Pine Script compilation requires a PyneSys API key. Get yours at [pynesys.io](https://pynesys.io).

## Why Choose PyneCore?

- **Beyond TradingView Limitations**: No more platform restrictions, code size limits, or subscription fees
- **Python Ecosystem Access**: Use Python's data science, ML, and analysis libraries alongside trading logic
- **Performance & Precision**: Designed for speed and precision, the same results as Pine Script
- **Open Source Foundation**: The core library and runtime is open source under Apache 2.0 license
- **Professional Trading Tools**: Build institutional-grade systems with Pine Script simplicity
- **Advanced Backtesting**: Run sophisticated strategy tests outside platform constraints

## Pine Script Migration Made Easy

Have existing Pine Script code you want to run in Python? **PyneSys now offers automatic Pine 
Script to PyneCore translation** through our [web platform](https://pynesys.io) and [Discord bot](https://discord.com/invite/jegnhtq6gy).

### 🚀 Get Started Instantly
- **Try for free on Discord**: Use `/pine-help` in our [Discord](https://discord.com/invite/jegnhtq6gy) for instant conversion - 3 free translations!
- **Full service**: Visit [pynesys.io](https://pynesys.io) for subscriptions and higher limits

### 💡 Support the Ecosystem
Love PyneCore and want to see it grow? Consider a **Seed subscription or higher** ($5+/mo) to support continued development:
- **Daily translations** of your Pine Script files (5+ per day)
- **Larger script support** for complex indicators and strategies
- **Direct contribution** to keeping PyneCore open source and advancing

*Every subscription helps maintain this free, open-source runtime and drives innovation in algorithmic trading tools.*

## Documentation & Support

- **Documentation**: [pynecore.org](https://pynecore.org/docs)

### Community

- **Discussions**: [GitHub Discussions](https://github.com/pynesys/pynecore/discussions)
- **Discord**: [discord.com/invite/jegnhtq6gy](https://discord.com/invite/jegnhtq6gy)
- **X**: [x.com/pynesys](https://x.com/pynesys)
- **Website**: [pynecore.org](https://pynecore.org)

## License

PyneCore is licensed under the [Apache License 2.0](LICENSE.txt).

## Disclaimer

Pine Script™ is a trademark of TradingView, Inc. PyneCore is not affiliated with, endorsed by, or sponsored by TradingView. This project is an independent implementation that aims to provide compatibility with the Pine Script language concept in the Python ecosystem.

### Risk Warning

Trading involves significant risk of loss and is not suitable for all investors. The use of PyneCore does not guarantee any specific results. Past performance is not indicative of future results.

- PyneCore is provided "as is" without any warranty of any kind
- PyneCore is not a trading advisor and does not provide trading advice
- Scripts created with PyneCore should be thoroughly tested before using with real funds
- Users are responsible for their own trading decisions
- You should consult with a licensed financial advisor before making any financial decisions

By using PyneCore, you acknowledge that you are using the software at your own risk. The creators and contributors of PyneCore shall not be held liable for any financial loss or damage resulting from the use of this software.

## Commercial Support

PyneCore is part of the PyneSys ecosystem. For commercial support, custom development, or enterprise solutions:

- **Website**: [pynesys.com/contact](https://pynesys.com/contact)

---
<strong>Elevate Your Trading with the Power of Python & Pine Script</strong>