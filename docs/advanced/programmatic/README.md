<!--
---
weight: 10000
title: "Programmatic Usage"
description: "Using PyneCore programmatically in Python applications"
icon: "code"
date: "2025-03-31"
lastmod: "2025-03-31"
draft: false
toc: true
categories: ["Advanced", "API"]
tags: ["programmatic", "api", "python", "integration", "scripting"]
---
-->

# Programmatic Usage

PyneCore provides a powerful programmatic API that allows you to integrate Pine Script execution directly into your Python applications. This approach gives you full control over data loading, script execution, and output handling without relying on the command-line interface.

## Overview

The programmatic API is built around three core components:

- **OHLCVReader**: Handles loading and iteration of OHLCV market data
- **SymInfo**: Manages symbol information and metadata
- **ScriptRunner**: Executes Pine scripts with the provided data and configuration

This approach is ideal when you need to:

- Integrate PyneCore into larger Python applications
- Process multiple symbols or timeframes programmatically
- Customize data sources beyond standard file formats
- Handle script outputs programmatically
- Build automated trading or analysis systems

## Documentation Structure

This directory contains comprehensive documentation for PyneCore's programmatic API:

- **[Core Components](core-components.md)** - Detailed reference for `OHLCVReader`, `SymInfo`, and `ScriptRunner`
- **[Basic Examples](basic-examples.md)** - Simple usage patterns and common tasks
- **[Advanced Examples](advanced-examples.md)** - Complex scenarios and production patterns
- **[Best Practices](best-practices.md)** - Performance optimization, error handling, and common pitfalls

## Quick Start

Here's a minimal example to get you started:

```python
from pathlib import Path
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.syminfo import SymInfo
from pynecore.core.script_runner import ScriptRunner

# Load data and run script
with OHLCVReader.open(Path("data/BTCUSD_1D.ohlcv")) as reader:
    syminfo = SymInfo(symbol="BTCUSD", exchange="BINANCE", timeframe="1D")
    runner = ScriptRunner(
        script_path="scripts/my_indicator.py",
        ohlcv_iter=reader,
        syminfo=syminfo
    )
    runner.run()
```

## Next Steps

1. Start with [Core Components](core-components.md) to understand the API
2. Try the [Basic Examples](basic-examples.md) to see it in action
3. Explore [Advanced Examples](advanced-examples.md) for production use cases
4. Review [Best Practices](best-practices.md) for optimal performance

For CLI usage, see the [CLI documentation](../../cli/README.md).