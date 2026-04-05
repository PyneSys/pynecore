<!--
---
weight: 444
title: "session"
description: "Trading session detection"
icon: "access_time"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["session", "library", "reference"]
---
-->

# session

Trading session detection and classification. The `session` namespace provides boolean flags for determining when the current bar falls within different trading session phases or session boundaries (first/last bar of the day). Use these flags to trigger actions at specific times during the trading day.

## Quick Example

```python
from pynecore.lib import close, session, bar_index, label, strategy

@script.indicator(title="Session Detector", overlay=True)
def main():
    # Mark the first bar of each day's session
    if session.isfirstbar_regular:
        label.new(bar_index, close, "Day start", textcolor="color.green")
    
    # Mark the last bar of regular trading hours
    if session.islastbar_regular:
        label.new(bar_index, close, "Regular close", textcolor="color.red")
    
    # Trade only during market hours
    if session.ismarket:
        strategy.entry("Long", strategy.long)
```

## Variables

All session variables are read-only module properties (accessed without parentheses).

### isfirstbar_regular

Returns `True` if the current bar is the first bar of the day's regular trading session, `False` otherwise.

**Type:** `bool`

**Example:**
```python
is_first: bool = session.isfirstbar_regular  # True on first regular bar of day
```

### isfirstbar

Returns `True` if the current bar is the first bar of the trading day, `False` otherwise. When extended session data is enabled, only returns `True` for the first bar of pre-market hours.

**Type:** `bool`

**Note:** Extended session support is not yet fully implemented; behaves like `isfirstbar_regular`.

**Example:**
```python
is_session_start: bool = session.isfirstbar  # True at session open
```

### islastbar_regular

Returns `True` if the current bar is the last bar of the day's regular trading session, `False` otherwise.

**Type:** `bool`

**Example:**
```python
is_last: bool = session.islastbar_regular  # True on last regular bar of day
```

### islastbar

Returns `True` if the current bar is the last bar of the trading day, `False` otherwise. When extended session data is enabled, only returns `True` for the last bar of post-market hours.

**Type:** `bool`

**Note:** Extended session support is not yet fully implemented; behaves like `islastbar_regular`.

**Example:**
```python
is_session_end: bool = session.islastbar  # True at session close
```

### ismarket

Returns `True` if the current bar is within regular market hours, `False` otherwise. On daily or longer timeframes, the result depends on whether the bar's time range overlaps with session hours — typically `True` for trading days.

**Type:** `bool`

**Example:**
```python
trading_hours: bool = session.ismarket  # True during market hours
```

### ispremarket

Returns `True` if the current bar is within pre-market hours, `False` otherwise. Always `False` on non-intraday charts.

**Type:** `bool`

**Note:** Not yet implemented. Always returns `False`.

**Example:**
```python
early_hours: bool = session.ispremarket  # Always False
```

### ispostmarket

Returns `True` if the current bar is within post-market hours, `False` otherwise. Always `False` on non-intraday charts.

**Type:** `bool`

**Note:** Not yet implemented. Always returns `False`.

**Example:**
```python
after_hours: bool = session.ispostmarket  # Always False
```

## Constants

| Name | Type | Description |
|------|------|-------------|
| `session.regular` | `Session` | Session type for regular trading hours only (no extended hours). |
| `session.extended` | `Session` | Session type including extended hours (pre-market and post-market). |

## Compatibility Notes

- **Extended session modes**: The `session.extended` constant is defined but extended session detection is not fully implemented. `isfirstbar` and `islastbar` currently behave identically to `isfirstbar_regular` and `islastbar_regular`.
- **Pre/post-market detection**: `ispremarket` and `ispostmarket` are not yet implemented and always return `False`.
- **Daily+ charts**: On daily or longer timeframes, session variables still evaluate using normal session overlap logic. Results depend on whether the bar's time range overlaps with configured session hours. This may differ from TradingView, which returns `False` for all session variables on daily+ charts.