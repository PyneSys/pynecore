<!--
---
weight: 1300
title: "Debugging"
description: "Debugging techniques for PyneCore scripts"
icon: "bug_report"
date: "2025-07-05"
lastmod: "2025-07-05"
draft: false
toc: true
---
-->

# Debugging PyneCore Scripts

This guide covers debugging techniques for PyneCore scripts, including inspection of Series variables and execution flow.

## Overview

PyneCore provides several debugging approaches to help you understand and troubleshoot your Pine Script translations:

- **Print statements**: Traditional Python debugging
- **Pine Script logging**: Native Pine Script debug functions
- **IDE debugging**: Step-through debugging with considerations for AST transformations

## Print Statements

The simplest debugging approach is using Python's built-in `print` function:

```python
# Basic print debugging
print("Current bar_index:", bar_index)
print("Close price:", close)
print("Variable value:", my_variable)
```

## Pine Script Logging (Recommended)

PyneCore supports Pine Script's native logging functions, which provide colorized output and better formatting:

### log.info()

Use for general information and variable inspection:

```python
log.info("Current bar index: {0}", bar_index)
log.info("Close price: {0}", close)
log.info("SMA value: {0}", ta.sma(close, 20))
```

### log.warning()

Use for warnings and potential issues:

```python
if volume == 0:
    log.warning("Zero volume detected at bar {0}", bar_index)
```

### log.error()

Use for error conditions:

```python
if close < 0:
    log.error("Invalid close price: {0}", close)
```

### Formatting

Pine Script logging supports formatting with curly braces:

```python
log.info("Bar {0}: Open={1}, High={2}, Low={3}, Close={4}",
         bar_index, open, high, low, close)
```

## Limiting Debug Output

To avoid overwhelming output, use `bar_index` to limit debug messages:

```python
# Only log first 10 bars
if bar_index < 10:
    log.info("Bar {0}: Close={1}", bar_index, close)

# Log every 100 bars
if bar_index % 100 == 0:
    log.info("Processing bar {0}", bar_index)

# Log specific conditions
if ta.crossover(close, ta.sma(close, 20)):
    log.info("Golden cross at bar {0}", bar_index)
```

## IDE Debugging

You can use your IDE's debugger with PyneCore scripts, but be aware of AST transformations:

### AST Transformations

Before execution, PyneCore applies AST transformations (see [AST Transformations](./advanced/ast-transformations.md) for
details). This affects debugging:

- **Series variables**: May be stored in `globals()` dictionary
- **Persistent variables**: Renamed with special prefixes
- **Function isolation**: Functions may have additional parameters

### Debugging with Transformations

When using the debugger:

1. **Series variables**: Look for them in `globals()` if not found in local scope
2. **Persistent variables**: Check for renamed versions with `__persistent_` prefix
3. **Function parameters**: May include additional isolation parameters

## Debugging Security Contexts

By default, `log.info()`, `log.warning()`, and `log.error()` calls inside `request.security()`
contexts are **suppressed** (matching TradingView behavior, where security context logs don't appear
in the log panel).

To debug security processes, set the `PYNE_SECURITY_LOG` environment variable to a file path:

```bash
PYNE_SECURITY_LOG=security.log pyne run my_script.py my_data --security "1D=my_data_1D"
```

All security process log output is redirected to the specified file. Each line is prefixed with
the security context identifier (symbol + timeframe) for easy identification:

```
[AAPL 1D] [2025-07-05 14:30:00-0400] bar:    42 INFO    SMA value: 150.25
[AAPL 1D] [2025-07-05 14:30:00-0400] bar:    42 WARNING Unusual volume spike
[EURUSD 1H] [2025-07-05 14:00:00+0000] bar:   100 INFO    RSI: 72.5
```

When multiple security processes run simultaneously, all write to the same file with their context
labels, making it easy to trace cross-context behavior.

## Best Practices

1. **Use Pine Script logging**: Prefer `log.info()`, `log.warning()`, and `log.error()` over `print()`
2. **Limit output**: Always use `bar_index` checks to prevent excessive logging
3. **Structured logging**: Use consistent formatting for easier analysis
4. **Remove debug code**: Clean up debug statements before production
5. **Test incrementally**: Add debug statements progressively to isolate issues
