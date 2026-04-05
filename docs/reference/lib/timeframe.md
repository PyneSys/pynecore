<!--
---
weight: 441
title: "timeframe"
description: "Timeframe detection and conversion utilities"
icon: "schedule"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["timeframe", "library", "reference"]
---
-->

# timeframe

The timeframe namespace provides utilities for detecting timeframe changes and converting between timeframe strings and seconds. It also includes query functions to determine the type of the current timeframe (daily, weekly, monthly, intraday, etc.) and properties for accessing the current timeframe information.

## Quick Example

```python
from pynecore.lib import timeframe, close, high, low, label, bar_index, script

@script.indicator(title="Timeframe Analysis", overlay=True)
def main():
    # Detect when entering a new daily timeframe
    if timeframe.change("D"):
        label.new(bar_index, high, "New Day")
    
    # Get current timeframe in seconds
    current_seconds: int = timeframe.in_seconds(timeframe.period)
    
    # Check if running on daily or higher timeframe
    if timeframe.isdwm():
        multiplier_value: int = timeframe.multiplier
    
    # Verify we're on an intraday timeframe before using minute-level logic
    if timeframe.isintraday():
        one_hour_str: str = timeframe.from_seconds(3600)
```

## Functions

### change()

Detects if the current bar is the first bar of a new timeframe.

| Parameter | Type | Description |
|-----------|------|-------------|
| timeframe | str | The timeframe to check (e.g., "D", "W", "60", "240") |

**Returns:** `bool`

Returns `True` on the first bar of a new period in the specified timeframe, `False` otherwise. The specified timeframe must be equal to or larger than the current chart timeframe. For example, on a 5-minute chart, `change("D")` returns `True` once per day when the first 5-minute candle of that day opens.

```python
if timeframe.change("D"):  # True at start of each new day
    label.new(bar_index, high, "New Day")
```

### from_seconds()

Converts a number of seconds into a valid timeframe string.

| Parameter | Type | Description |
|-----------|------|-------------|
| seconds | int | Number of seconds to convert |

**Returns:** `str`

Converts seconds to timeframe format: 3600 becomes "60" (60 minutes), 86400 becomes "D" (one day), 604800 becomes "W" (one week), etc.

```python
tf_str: str = timeframe.from_seconds(3600)  # "60"
```

### in_seconds()

Converts a timeframe string into the equivalent number of seconds.

| Parameter | Type | Description |
|-----------|------|-------------|
| timeframe | str | Timeframe string (e.g., "D", "W", "60", "240") |

**Returns:** `int`

Parses timeframe strings and returns the total seconds. Handles minute values ("5", "60", "240"), seconds ("5S"), days ("D"), weeks ("W"), and months ("M"). There is no "H" suffix — hours are expressed in minutes (e.g., "240" for 4 hours).

```python
seconds: int = timeframe.in_seconds("D")  # 86400
```

### isdaily()

Returns `True` if the current chart timeframe is daily, `False` otherwise.

**Returns:** `bool`

```python
is_daily: bool = timeframe.isdaily()
```

### isdwm()

Returns `True` if the current chart timeframe is daily, weekly, or monthly, `False` otherwise.

**Returns:** `bool`

Identifies "calendar-aligned" timeframes that respect weeks and months.

### isintraday()

Returns `True` if the current chart timeframe is intraday (minutes or seconds), `False` otherwise.

**Returns:** `bool`

```python
is_intraday: bool = timeframe.isintraday()
```

### isminutes()

Returns `True` if the current chart timeframe is in minutes, `False` otherwise.

**Returns:** `bool`

### ismonthly()

Returns `True` if the current chart timeframe is monthly, `False` otherwise.

**Returns:** `bool`

### isseconds()

Returns `True` if the current chart timeframe is in seconds, `False` otherwise.

**Returns:** `bool`

### isticks()

Returns `True` if the current chart timeframe is tick-based, `False` otherwise.

**Returns:** `bool`

Note: Tick-based charts are not yet fully supported in PyneCore.

### isweekly()

Returns `True` if the current chart timeframe is weekly, `False` otherwise.

**Returns:** `bool`

## Variables

### main_period

The string representation of the script's main timeframe, as specified in the script's declaration.

**Type:** `str`

Set once when the script starts and remains constant. Example values: "5", "60", "D", "W", "M".

### multiplier

The multiplier component of the current timeframe resolution.

**Type:** `int`

Represents the numeric multiplier of the timeframe unit. For example: 60 on a "60" (60-minute) chart, 240 on a "240" (4-hour) chart, 1 on a "D" (daily) chart.

```python
mult: int = timeframe.multiplier
```

### period

The string representation of the current timeframe.

**Type:** `str`

Reflects the active timeframe. Example values: "5", "60", "D", "W", "M".

## Compatibility

All functions and variables in the timeframe namespace are fully implemented. The `isticks()` function exists but tick-based charts are not yet fully supported in PyneCore.