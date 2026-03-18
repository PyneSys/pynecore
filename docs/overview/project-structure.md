<!--
---
weight: 106
title: "Project Structure"
description: "Overview of the PyneCore project structure and architecture"
icon: "account_tree"
date: "2025-04-03"
lastmod: "2026-03-18"
draft: false
toc: true
categories: ["Overview", "Architecture"]
tags: ["project-structure", "architecture", "organization", "components", "modules"]
---
-->

# Project Structure

PyneCore is organized in a modular structure that allows for clean separation of concerns and extension points. This page provides a high-level overview of the project's architecture and main components.

## Repository Organization

The PyneCore repository is organized as follows:

```
pynecore/
├── src/                    # Source code
│   └── pynecore/           # Main package
│       ├── core/           # Core functionality and runtime components
│       ├── lib/            # Pine Script compatible function library (134 indicators)
│       │   └── strategy/   # Strategy backtesting subsystem
│       ├── transformers/   # AST transformers for Pine Script syntax
│       ├── types/          # Type definitions and interfaces
│       ├── utils/          # Utility functions and helpers
│       ├── cli/            # Command-line interface (pyne command)
│       │   └── commands/   # CLI subcommands (run, compile, data, benchmark, tv)
│       ├── providers/      # OHLCV data providers (TradingView, CCXT, Capital.com)
│       ├── pynesys/        # PyneSys API client and cloud compilation service
│       └── standalone.py   # Standalone script execution (python script.py data.csv)
├── tests/                  # Test suite
├── docs/                   # Component-specific documentation
└── scripts/                # Utility scripts
```

## Core Components

### Core Module (`core/`)

The heart of the PyneCore system, implementing the fundamental data structures and execution model:

- **Series Implementation** (`series.py`): Circular buffer-based data structure that emulates Pine Script's series behavior (~490k ops/sec)
- **Script Runner** (`script_runner.py`): Executes Pyne scripts bar-by-bar, handles strategy simulation lifecycle
- **Import Hook System** (`import_hook.py`): Allows Python to process `@pyne` scripts with AST transformations
- **Data Handling** (`ohlcv_file.py`, `csv_file.py`, `data_converter.py`): OHLCV reading/writing, CSV conversion with extra fields support
- **Aggregator** (`aggregator.py`): Timeframe aggregation (e.g. 1m → 1H → 1D)
- **Resampler** (`resampler.py`): Bar time calculation with timezone-aware calendar alignment
- **Overload Dispatch** (`overload.py`): Pine Script-compatible function overloading system
- **Pine Script Constructs** (`pine_udt.py`, `pine_method.py`, `pine_export.py`): User-defined types, methods, and export support
- **SymInfo** (`syminfo.py`): Symbol information (mintick, currency, session, etc.)
- **Strategy Stats** (`strategy_stats.py`): Backtesting performance metrics calculation

### Transformers Module (`transformers/`)

Python Abstract Syntax Tree (AST) transformers that modify Python code to behave like Pine Script:

- **Series Transformations** (`series.py`): Convert Python operations to Series-aware operations
- **Persistent Variables** (`persistent.py`, `persistent_series.py`): Implement bar-persistent variables with automatic Kahan summation for `+=`
- **Function Isolation** (`function_isolation.py`): Create isolated function scopes with per-call state
- **Import Management** (`import_normalizer.py`, `import_lifter.py`): Organize and normalize imports
- **Library Series** (`lib_series.py`): Convert library attribute access to Series-aware operations
- **Input Transformer** (`input_transformer.py`): Transform `input.*()` calls to runtime configuration
- **Safe Division** (`safe_division_transformer.py`): Prevent division-by-zero (Pine Script returns `na`)
- **Safe Convert** (`safe_convert_transformer.py`): Type-safe conversions matching Pine Script behavior
- **Module Properties** (`module_property.py`): Transform module-level property access
- **Unused Series Detector** (`unused_series_detector.py`): Optimization pass to skip unnecessary Series allocations
- **Closure Arguments** (`closure_arguments_transformer.py`): Handle closure variable capture

### Library Module (`lib/`)

Full Pine Script v6 function library implemented in Python — 134 TradingView-compatible indicators and functions:

- **Technical Analysis** (`ta.py`): SMA, EMA, RSI, MACD, Bollinger Bands, and 130+ other indicators
- **Strategy** (`strategy/`): Complete TradingView-compatible backtesting subsystem
    - Order processing, margin calls, position management
    - Closed/open trade tracking (`closedtrades.py`, `opentrades.py`)
    - Commission models, OCA groups, risk management
- **Math Functions** (`math.py`): Mathematical operations
- **Data Structures** (`array.py`, `map.py`, `matrix.py`): Collection types
- **String Operations** (`string.py`): String manipulation functions
- **Request Functions** (`request.py`): Multi-timeframe data requests
- **Chart Elements** (`label.py`, `line.py`, `box.py`, `table.py`, `polyline.py`, `linefill.py`): Chart objects
- **Timeframe** (`timeframe.py`): Timeframe utilities
- **Runtime** (`runtime.py`, `barstate.py`, `syminfo.py`): Bar state and symbol information
- **Enum Constants**: 20+ modules for Pine Script constant types (display, extend, format, color, etc.)

### Types Module (`types/`)

Type definitions for Pine Script compatible constructs (40+ type modules):

- **Series Types** (`series.py`): Type definitions for Series objects
- **NA Value** (`na.py`): Not Available value handling
- **Persistent Types** (`persistent.py`): Persistent variable type definitions
- **OHLCV** (`ohlcv.py`): Bar data record types with extra fields support
- **Script Type** (`script_type.py`): Indicator/strategy/library script type definitions
- **Strategy** (`strategy.py`): Strategy-related type definitions
- **Color** (`color.py`): Color handling and constants
- **Chart Objects** (`label.py`, `line.py`, `box.py`, `table.py`, `polyline.py`): Chart element types
- **Source** (`source.py`): Price source type (open, high, low, close, etc.)
- **Enum Types**: Matching modules for all Pine Script constant enums

### CLI Module (`cli/`)

Command-line interface providing the `pyne` command:

- **run**: Execute Pyne scripts with OHLCV data
- **compile**: Compile Pine Script via the PyneSys API
- **data**: Download, convert, and aggregate OHLCV data
- **benchmark**: Performance benchmarking
- **tv**: Private TradingView integration tools

### Providers Module (`providers/`)

OHLCV data download providers:

- **TradingView** (`tv.py`): Chart data via TradingView WebSocket
- **CCXT** (`ccxt.py`): Cryptocurrency exchanges via CCXT library
- **Capital.com** (`capitalcom.py`): Capital.com broker API

### PyneSys Module (`pynesys/`)

Cloud compilation service client:

- **API Client** (`api.py`): HTTP client for the PyneSys compilation API
- **Compiler** (`compiler.py`): Programmatic interface for Pine Script → Python compilation

## Script Execution Flow

PyneCore processes and executes scripts through several stages:

1. **Script Recognition**: Scripts with the `@pyne` magic comment are recognized by the import hook
2. **AST Transformation**: Python code is transformed through 11 AST transformer passes
3. **Initialization**: Script runner loads OHLCV data and initializes Series buffers
4. **Bar-by-Bar Execution**: The `main()` function is called once per bar with updated OHLCV values
5. **Strategy Simulation** (strategies only): After each `main()` call, pending orders are processed through a multi-phase pipeline (gap detection → market fills → OHLC simulation → margin calls)
6. **Output Generation**: Plot values, trade logs, and strategy statistics are written to CSV

Compiled scripts also support standalone execution via `python script.py data.csv` without needing a workdir or the `pyne` CLI.

## Key Features

- **Pine Script v6 Compatibility**: Familiar syntax and behavior for TradingView users
- **134 Technical Indicators**: Full TradingView indicator library
- **Strategy Backtesting**: Complete TradingView-compatible strategy simulation with margin, slippage, and commission
- **AST Transformations**: 11 transformer passes implement Pine Script semantics in Python
- **High Performance**: Optimized circular buffer implementation (~490k ops/sec)
- **Standalone Execution**: Compiled scripts are runnable as regular Python files
- **Timeframe Aggregation**: Aggregate OHLCV data across timeframes
- **Multiple Data Providers**: TradingView, CCXT, Capital.com
- **Comprehensive Testing**: 130+ test files with precision validation against TradingView

For more detailed information about specific components, see the relevant sections in the documentation.