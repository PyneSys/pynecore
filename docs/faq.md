<!--
---
weight: 1200
title: "FAQ"
description: "Frequently asked questions about PyneCore"
icon: "psychology"
date: "2025-03-31"
lastmod: "2025-03-31"
draft: false
toc: true
categories: ["Support"]
tags: ["faq", "troubleshooting", "installation", "configuration"]
---
-->

# FAQ

This section contains frequently asked questions about PyneCore.

## General Questions

### What is PyneCore?

PyneCore is an open-source framework that implements TradingView's Pine Script paradigm in Python. It brings the intuitive bar-by-bar execution model of Pine Script into Python while leveraging the vast Python ecosystem. PyneCore is not a tool that simply runs Pine Script code, but a complete reimagining of the Pine Script concept natively in Python.

### How does PyneCore relate to TradingView?

PyneCore is not affiliated with or endorsed by TradingView. It is an independent project that aims to be compatible with Pine Script functionality while offering the advantages of Python. PyneCore strives to match TradingView's Pine Script calculations with precision testing with tolerances of 0.001% (relative) and 0.00000001 (absolute).

### Is PyneCore free to use?

Yes, the core PyneCore functionality is open-source and free to use under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0). The PyneComp compiler (which converts Pine Script to Python) is a separate closed-source component that's available as a (affordable) SaaS offering.

### What are the system requirements for PyneCore?

PyneCore requires Python 3.11 or newer. It is designed to work on Windows, macOS, and Linux operating systems. The core system has no external dependencies, though some additional features may require specific packages (e.g. command line tools).

### How do I report bugs or request features?

You can report bugs and request features through the project's [GitHub repository issue tracker](https://github.com/PyneSys/pynecore/issues). Please include detailed information about the issue, steps to reproduce it, and your system environment.

## Installation & Setup

### How do I install PyneCore?

The recommended way to install PyneCore is using pip:

```bash
pip install pynesys-pynecore[cli]  # With user-friendly CLI
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
├── scripts/     # Your Pyne scripts
├── data/        # OHLCV data files
├── output/      # Output files
└── config/      # Configuration files
```

The PyneCore CLI will create this structure automatically if it doesn't exist.

### How do I use PyneCore with my existing data?

PyneCore supports importing data from various formats:

```bash
pyne data convert-from path/to/your/data.csv --symbol "BTCUSDT" --timeframe 1D
```

You can also download data from supported providers:

```bash
pyne data download ccxt --symbol "BYBIT:BTC/USDT:USDT" --timeframe 1D
```

More providers could be added in the future by the community as plugins.

## Writing Scripts

### How do I start a PyneCore script?

Every PyneCore script must start with a special magic comment to identify it as a Pyne script:

```python
"""
@pyne
"""
```

After this comment, you can import from PyneCore libraries and define your main function:

```python
from pynecore.lib import script, close

@script.indicator("My Indicator")
def main():
    # Your code here
```

### What's the difference between PyneCore and regular Python code?

PyneCore applies AST transformations to your Python code, enabling it to behave like Pine Script while maintaining Python syntax. Key differences include:

- Series variables that track historical values
- Persistent variables that maintain state across bars
- Bar-by-bar execution model
- Function isolation (each function call gets its own state)
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

Actually, PyneCore will never plot anything, it just gives you the data to plot. By default it saves the data to the `output/` folder in the working directory as CSV files. In the future we'll develop a separate PynePlot plotting library.

Though the syntax is there. PyneCore offers two ways to "plot" indicators:

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

### Is PyneCore 100% compatible with Pine Script?

PyneCore aims for high compatibility with Pine Script functionality and calculations, with high precision (0.001% tolerance). However, there are some intentional differences to make the experience more Pythonic and to leverage Python's strengths. These differences are documented in the [Differences from Pine Script](./overview/differences.md) page.

### How does PyneCore handle NA values?

PyneCore implements a custom NA (Not Available) system similar to Pine Script's. You can check if a value is NA using:

```python
from pynecore.lib import na

if na(value):
    # Handle NA case
```

NA values behave safely in all contexts — no special handling needed in most cases:

- **Comparisons** return `False`: `NA < 30`, `NA > 70`, `NA == x`
- **Arithmetic** propagates: `NA + 1` → `NA`, `NA * 2.0` → `NA`
- **Format strings** work: `f"{na_value:.2f}"` → `"NaN"`
- **bool()** returns `False`

```python
rsi = plot_data.get("RSI")
if rsi > 70:                    # False when rsi is NA — no crash
    print(f"Overbought: {rsi:.2f}")  # NA prints as "NaN"
```

### How do Series variables work in PyneCore?

Series variables in PyneCore store historical data points. Behind the scenes, they are implemented as global circular buffers that maintain historical values. When you declare a variable as a Series, PyneCore's AST transformations handle the proper creation and access to these buffers automatically.

### What's the performance like compared to Pine Script?

PyneCore is designed for high performance while maintaining the intuitive bar-by-bar execution model. While vectorized operations in pandas or numpy might be faster for certain calculations, PyneCore offers a good balance between performance and the familiarity of Pine Script's execution model.

### Can I use external Python libraries with PyneCore?

Yes! One of the main advantages of PyneCore is the ability to leverage the vast Python ecosystem. You can import and use any Python library in your PyneCore scripts. This allows you to combine technical analysis with data science, machine learning, and more.

### Why doesn't division by zero raise an exception in PyneCore?

PyneCore automatically handles division by zero to match Pine Script behavior. When you write:

```python
result = numerator / denominator
```

PyneCore's AST transformation system converts this to:

```python
result = safe_convert.safe_div(numerator, denominator)
```

This function returns `NA(float)` instead of raising a `ZeroDivisionError` when the denominator is zero or NA, which matches how Pine Script handles division by zero. This transformation only applies to dynamic divisions (not literal values like `1/2`), ensuring both Pine Script compatibility and optimal performance.

This is part of PyneCore's "it just works" philosophy - your Pine Script logic will behave exactly as expected without requiring explicit error handling for common edge cases.

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

## Troubleshooting

### My script isn't being recognized as a PyneCore script

Make sure:
1. Your file starts with the `@pyne` magic comment
2. The comment is at the very beginning of the file (before imports)
3. You have a main() function defined
4. You've imported necessary modules from pynecore.lib

### I'm getting unexpected NA values in my calculations

NA values can propagate through calculations just like in Pine Script. If any input to an operation is NA, the result will typically be NA. Check for NA values using the `na()` function and provide appropriate default values if needed.

### My script runs differently than the Pine Script version

While PyneCore aims for high compatibility, there might be subtle differences in behavior. Common issues include:

1. Different handling of edge cases
2. Differences in function implementation details
3. Missing Pine Script features (check the documentation for supported features)

If you encounter significant discrepancies, please report them as issues in the GitHub repository.

### How do I debug my PyneCore script?

PyneCore provides several debugging approaches. See the [Debugging](./debugging.md) guide for detailed information.
