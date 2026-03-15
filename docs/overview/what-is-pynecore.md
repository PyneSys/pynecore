<!--
---
weight: 101
title: "What is PyneCore"
description: "Introduction to PyneCore and its core concepts"
icon: "code"
date: "2025-03-31"
lastmod: "2025-03-31"
draft: false
toc: true
categories: ["Overview"]
tags: ["introduction", "pine-script", "python"]
---
-->

# What is PyneCore?

## The Python-Powered Trading Framework

PyneCore is an innovative, open-source framework that brings the power of TradingView's Pine Script paradigm to the Python ecosystem. It's not just a tool that runs Pine Script code — it's a complete reimagining of the Pine Script concept, natively implemented in Python, while trying to be as compatible with Pine Script as possible.

## The PyneCore Vision

Trading strategies and technical indicators are usually implemented in one of two ways:

1. Using specialized proprietary languages like TradingView's Pine Script, which are powerful but limited to their platforms
2. Using general-purpose languages like Python or JavaScript, which offer flexibility but lack the bar-by-bar execution model that makes Pine Script so intuitive for traders (they usually use vectorized operations, which is faster, but hard to understand)

**PyneCore bridges this gap** by bringing Pine Script's intuitive bar-by-bar execution model into Python — a language with vast libraries, rich ecosystem, and unlimited extensibility.

## Project Goals and Design Principles

PyneCore was built with several ambitious goals in mind:

- **100% compatibility with TradingView Pine Script**: The system aims to replicate Pine Script functionality with high precision (0.001% tolerance)
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

More about the PyneSys ecosystem can be found in the [Ecosystem](/docs/overview/ecosystem/) page.

## Technical Innovation

What makes PyneCore truly groundbreaking is its approach to implementing Pine Script's execution model in Python:

### Unique Architecture

Unlike other systems that might attempt to replicate Pine Script's functionality through object-oriented wrappers or by creating a new language, PyneCore takes a fundamentally different approach. It transforms regular Python code to behave like Pine Script while maintaining Python's syntax, tools, and ecosystem.

### AST Transformation Magic

Rather than creating yet another object-oriented framework, PyneCore uses Python's Abstract Syntax Tree (AST) transformation capabilities to modify Python code before execution:

1. Your Python code is parsed into an AST representation
2. The AST is transformed to implement Pine Script-like behavior
3. The transformed AST is then executed in the Python environment

This approach means:
- You write clean, pythonic code with minimal boilerplate
- The transformations handle all the complexity behind the scenes
- Your code runs with Pine Script semantics but with Python's full power

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
- **Precise**: Calculations match TradingView's results with high precision (the goal is to be 100% compatible)

## Who Is PyneCore For?

- **Algorithmic Traders** who want to move beyond TradingView's limitations
- **Python Developers** entering the trading space who prefer Python to Pine Script
- **Quant Researchers** who need to combine technical analysis with data science
- **Fintech Companies** building trading tooling and infrastructure
- **Everyone** who loves Pine Script and/or Python will love PyneCore more!

## Getting Started

Ready to explore PyneCore? Continue to the [Getting Started](/docs/getting-started/) section to install PyneCore and write your first script!
