<!--
---
weight: 472
title: "dayofweek"
description: "Day of week constants (sunday through saturday)"
icon: "calendar_today"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["dayofweek", "library", "reference"]
---
-->

# dayofweek

The `dayofweek` namespace provides integer-like constants representing days of the week. These constants are used to compare against the built-in `dayofweek` variable, which holds the day of the current bar's timestamp.

## Quick Example

```python
from pynecore.lib import script, dayofweek, dayofweek as dow_var, close, bar_index, label

@script.indicator(title="Weekend Marker", overlay=True)
def main():
    if dow_var == dayofweek.monday:
        label.new(bar_index, close, "Monday open")
```

> **Note:** `dayofweek` serves dual purpose — import it both as the namespace (for constants) and as the built-in variable (for the current bar's day). In practice, use an alias to avoid shadowing: `from pynecore.lib import dayofweek as dow_var` for the variable, and `dayofweek.monday` etc. for constants.

---

## Constants

All constants are of type `DayOfWeek`, which behaves like an integer in comparisons.

| Constant               | Description          |
|------------------------|----------------------|
| `dayofweek.sunday`     | Sunday (1)           |
| `dayofweek.monday`     | Monday (2)           |
| `dayofweek.tuesday`    | Tuesday (3)          |
| `dayofweek.wednesday`  | Wednesday (4)        |
| `dayofweek.thursday`   | Thursday (5)         |
| `dayofweek.friday`     | Friday (6)           |
| `dayofweek.saturday`   | Saturday (7)         |

### Usage

Compare the built-in `dayofweek` variable against these constants:

```python
is_friday: bool = dayofweek == dayofweek.friday
is_weekend: bool = dayofweek in (dayofweek.saturday, dayofweek.sunday)
```