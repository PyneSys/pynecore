<!--
---
weight: 450
title: "log"
description: "Logging functions for debugging"
icon: "terminal"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["log", "library", "reference"]
---
-->

# log

Logging functions for debugging and troubleshooting. The `log` namespace provides functions to print messages at different severity levels (info, warning, error) to the console. Messages include timestamps and the current bar index for easy reference during backtesting.

## Quick Example

```python
from pynecore.lib import log, close, high, ta, bar_index, script

@script.indicator(title="Debug Indicator", overlay=True)
def main():
    sma20: float = ta.sma(close, 20)
    
    if bar_index % 10 == 0:
        log.info("Bar {0}: close={1}, sma20={2}", bar_index, close, sma20)
    
    if close > high:
        log.warning("Invalid data at bar {0}", bar_index)
    
    if sma20 < 0:
        log.error("Unexpected negative SMA: {0}", sma20)
```

## Functions

### info()

Logs an informational message to the console.

| Parameter | Type | Description |
|-----------|------|-------------|
| `formatString` | str | Format string with `{0}`, `{1}`, etc. placeholders |
| `*args` | Any | Values to substitute into the format string |

**Returns:** None

**Example:**
```python
log.info("Signal at bar {0}: price={1}", bar_index, close)
```

### warning()

Logs a warning message to the console.

| Parameter | Type | Description |
|-----------|------|-------------|
| `formatString` | str | Format string with `{0}`, `{1}`, etc. placeholders |
| `*args` | Any | Values to substitute into the format string |

**Returns:** None

**Example:**
```python
log.warning("Price spike detected: {0}", high)
```

### error()

Logs an error message to the console.

| Parameter | Type | Description |
|-----------|------|-------------|
| `formatString` | str | Format string with `{0}`, `{1}`, etc. placeholders |
| `*args` | Any | Values to substitute into the format string |

**Returns:** None

**Example:**
```python
log.error("Calculation error: {0}", error_value)
```

## Behavior in Security Contexts

Log calls inside `request.security()` processes are **suppressed by default**, matching TradingView
behavior. To redirect security process logs to a file, set the `PYNE_SECURITY_LOG` environment
variable:

```bash
PYNE_SECURITY_LOG=security.log pyne run script.py data --security "1D=data_1D"
```

Each log line in the file is prefixed with the context identifier (`[SYMBOL TIMEFRAME]`):

```
[AAPL 1D] [2025-07-05 14:30:00-0400] bar:    42 INFO    SMA value: 150.25
```

See [Debugging Security Contexts](../../debugging.md#debugging-security-contexts) for details.

## Compatibility

All logging functions are fully implemented. Format strings use 0-indexed placeholders (`{0}`, `{1}`, etc.) similar to Python's `str.format()` method. Timestamps are adjusted to match the script's timezone, and the current bar index is automatically included in log output for easier debugging during backtests.