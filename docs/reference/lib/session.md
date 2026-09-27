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

Returns `True` if the current bar is the first bar of the trading day, `False` otherwise. On bars of the extended hours (`syminfo.session == session.extended`) only the first bar of the pre-market is the first bar.

**Type:** `bool`

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

Returns `True` if the current bar is the last bar of the trading day, `False` otherwise. On bars of the extended hours only the last bar of the post-market is the last bar.

**Type:** `bool`

**Example:**
```python
is_session_end: bool = session.islastbar  # True at session close
```

### ismarket

Returns `True` if the current bar is within regular market hours, `False` otherwise. Every bar of a regular-hours chart is; on bars of the extended hours the bar has to open inside the regular hours.

**Type:** `bool`

**Example:**
```python
trading_hours: bool = session.ismarket  # True during market hours
```

### ispremarket

Returns `True` if the current bar is within pre-market hours, `False` otherwise: a bar of the extended hours opening before the day's regular open. Always `False` on a regular-hours chart.

**Type:** `bool`

**Example:**
```python
early_hours: bool = session.ispremarket  # True on the 04:00-09:00 bars of a US stock
```

### ispostmarket

Returns `True` if the current bar is within post-market hours, `False` otherwise: a bar of the extended hours opening at or after the day's regular close (13:00 on an early-close day). Always `False` on a regular-hours chart.

**Type:** `bool`

**Example:**
```python
after_hours: bool = session.ispostmarket  # True on the 16:00-19:00 bars of a US stock
```

## Constants

| Name | Type | Description |
|------|------|-------------|
| `session.regular` | `Session` | Session type for regular trading hours only (no extended hours). |
| `session.extended` | `Session` | Session type including extended hours (pre-market and post-market). |

## Compatibility Notes

- **Extended hours**: the properties follow the symbol's `extended_hours` template and the `session` flag of its TOML (see [Extended Trading Hours](../../programmatic/data-and-syminfo.md#extended-trading-hours)). The bars of an extended-hours chart are cut from the extended open, so the regular open never falls on a bar edge: the regular hours are read by the bar's open (the first regular bar of a 60-minute US stock chart is the 10:00 bar, not the 09:00 bar that contains 09:30), while the chart's own open and close are the bars that contain them. Measured on the NASDAQ:AAPL 60-minute chart on both hours.
- **Daily+ charts**: On daily or longer timeframes, session variables still evaluate using normal session overlap logic. Results depend on whether the bar's time range overlaps with configured session hours. This may differ from TradingView, which returns `False` for all session variables on daily+ charts.