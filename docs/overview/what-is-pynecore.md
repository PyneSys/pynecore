<!--
---
weight: 101
title: "What is PyneCore"
description: "Introduction to PyneCore and its core concepts"
icon: "code"
date: "2025-03-31"
lastmod: "2026-10-18"
draft: false
toc: true
categories: ["Overview"]
tags: ["introduction", "pine-script", "python"]
---
-->

# What is PyneCore?

## The Python-Powered Trading Framework

PyneCore is the open-source runtime for [Pyne code](./pyne-code.md), a
domain-specific language for market data, indicators, and trading strategies.
Pyne uses Python syntax and defines bar-by-bar execution, series history, and
persistent state. PyneCore implements those language rules in Python and provides
a Pine Script-compatible calculation and strategy library.

PyneCore is validated against TradingView on <!--wild:scripts_total-->1,000<!--/wild--> published
TradingView scripts (Pine Script v4–v6, converted to Pyne code with [PyneComp](https://pynesys.io/compiler)):
all <!--wild:tv_verified-->1,353<!--/wild--> comparable outputs match,
<!--wild:bars_exact_pct-->99.771<!--/wild-->% of
<!--wild:bars_compared_millions-->132<!--/wild--> million plotted values are bit-identical, and
<!--wild:strategy_trades-->353,577<!--/wild--> strategy trades match TradingView's timing
(snapshot <!--wild:generated_at-->2026-10-24<!--/wild-->, [Pyne in the
Wild](https://wild.pynesys.io/)). See [Compatibility](./compatibility.md) for details.

## The PyneCore Vision

Trading strategies and technical indicators are usually implemented in one of two ways:

1. Using specialized proprietary languages like TradingView's Pine Script, which are powerful but limited to their platforms
2. Using general-purpose languages like Python or JavaScript, which offer flexibility but lack the bar-by-bar execution model that makes Pine Script so intuitive for traders (they usually use vectorized operations, which is faster, but hard to understand)

**PyneCore bridges this gap** by bringing Pine Script's intuitive bar-by-bar execution model into Python — a language with vast libraries, rich ecosystem, and unlimited extensibility.

## Project Goals and Design Principles

PyneCore was built with several ambitious goals in mind:

- **Results that match TradingView**: Indicator values and strategy trades match TradingView,
  measured on a public corpus of <!--wild:scripts_total-->1,000<!--/wild--> published TradingView scripts
- **Zero mandatory dependencies**: The core system operates without external libraries, ensuring portability and reliability
- **Maximum performance**: Designed from the ground up for speed and efficiency
- **Clean, well-documented code**: The source code prioritizes readability and proper documentation
- **Pythonic approach**: While maintaining Pine Script compatibility, the system embraces Python's strengths and conventions
- **Logical improvements**: In some areas, PyneCore improves upon Pine Script's design where beneficial
- **Developer experience**: The system is designed to make writing trading algorithms enjoyable and productive

## Part of a Larger Ecosystem

PyneCore is the foundation of the PyneSys project, which consists of these components:

- **PyneCore**: The open-source Pine Script-like system in Python (what you're reading about)
- **PyneComp**: A Pine Script to Pyne Code compiler (closed source, accessible through the Pyne API)
- More is coming soon!

More about the PyneSys ecosystem can be found in the [Ecosystem](./ecosystem.md) page.

## Technical Innovation

What makes PyneCore truly groundbreaking is its approach to implementing Pine Script's execution model in Python:

### Unique Architecture

PyneCore parses marked Pyne modules with Python's parser, transforms their abstract
syntax trees, and executes the result in Python. The syntax is Python-compatible;
the transformations implement Pyne's time-series and state semantics. The
`@pyne edge` profile restricts the language features a program can use, while plain
`@pyne` supports broader Python integration.

### AST Transformation Magic

Rather than creating yet another object-oriented framework, PyneCore uses Python's Abstract Syntax Tree (AST) transformation capabilities to modify Python code before execution:

1. Your Python code is parsed into an AST representation
2. The AST is transformed to implement Pine Script-like behavior
3. The transformed AST is then executed in the Python environment

This approach means:
- You write clean, pythonic code with minimal boilerplate
- The transformations handle all the complexity behind the scenes
- Your code runs with Pyne's execution rules, with Python integration available outside the edge profile

More about this in the [AST Transformations](../advanced/ast-transformations.md) documentation.

## Key Concepts and Innovations

### 1. Series Variables

Series variables in PyneCore store historical data points, just like in Pine Script:

```python
s: Series[float] = close  # Create a Series of closing prices
prev_close = close[1]     # Access the previous bar's close
```

Behind the scenes, series are implemented as global circular buffers that maintain historical values.

More about Series variables in the [Core Concepts](./core-concepts.md#3-series-variables) documentation.

### 2. Persistent Variables

Persistent variables maintain their values across bars, allowing state to be preserved as your script processes each candle:

```python
p: Persistent[int] = 0  # Initialize a persistent counter
p += 1                 # Increments with each bar
```

Learn more about Persistent variables in the [Core Concepts](./core-concepts.md#2-persistent-variables) documentation.

### 3. Function Isolation

Each call to a function gets its own isolated state for persistent and series variables:

```python
def my_indicator(input_series, length):
    # Each call to this function has its own state
    sum: Persistent[float] = 0

    sum += input_series - input_series[length]
    return sum / length
```

More details about Function Isolation in the [Core Concepts](./core-concepts.md#4-function-isolation) and [Function Isolation](../advanced/function-isolation.md) documentation.

### 4. NA (Not Available) System

Pine Script's NA concept is fully implemented, allowing graceful handling of missing or undefined values:

```python
if na(value):  # Check if a value is NA
    value = default
```

Learn more about the NA system in the [Core Concepts](./core-concepts.md#5-na-not-available-system) documentation.

## Why Choose PyneCore?

- **Performance**: We designed PyneCore to be as fast as possible in pure Python, while still being easy to understand.
- **Pythonic**: Write code that feels natural to both Python and Pine Script developers
- **Extensible**: Leverage Python's vast ecosystem alongside PyneCore's capabilities
- **Open Source**: Core functionality is open source and free to use (Apache 2.0 license)
- **Modern**: Type hints, error handling, and comprehensive documentation (always improving)
- **Precise**: All <!--wild:tv_verified-->1,353<!--/wild--> comparable outputs of <!--wild:scripts_total-->1,000<!--/wild--> published TradingView scripts
  match TradingView; <!--wild:bars_exact_pct-->99.771<!--/wild-->% of <!--wild:bars_compared_millions-->132<!--/wild--> million plotted values are
  bit-identical, and the largest relative gap anywhere is <!--wild:max_rel-->6.3 × 10⁻¹²<!--/wild-->

## Who Is PyneCore For?

- **Algorithmic Traders** who want to move beyond TradingView's limitations
- **Python Developers** entering the trading space who prefer Python to Pine Script
- **Quant Researchers** who need to combine technical analysis with data science
- **Fintech Companies** building trading tooling and infrastructure
- **Everyone** who loves Pine Script and/or Python will love PyneCore more!

## Getting Started

Ready to explore PyneCore? Continue to the [Getting Started](../getting-started/README.md) section to install PyneCore and write your first script!
