<!--
---
weight: 486
title: "earnings"
description: "Earnings data field constants"
icon: "trending_up"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["earnings", "library", "reference"]
---
-->

# earnings

The `earnings` namespace provides access to earnings report data for the current instrument. It contains properties for upcoming earnings estimates and named constants used with `request.earnings()` to specify which earnings value type to retrieve.

## Quick Example

```python
from pynecore.lib import script, earnings, close, bar_index, label, color

@script.indicator(title="Earnings Monitor", overlay=True)
def main():
    eps: float = earnings.future_eps
    rev: float = earnings.future_revenue
    t: int = earnings.future_time

    if eps != None:
        label.new(bar_index, close, f"EPS est: {eps:.2f}", color=color.blue)
```

---

## Variables

### earnings.future_eps

Returns the estimated Earnings per Share for the next earnings report, in the instrument's currency. Returns `na` if the data is unavailable.

| Property | Detail        |
|----------|---------------|
| Type     | `float`       |
| Returns  | `float \| na` |

---

### earnings.future_revenue

Returns the estimated Revenue for the next earnings report, in the instrument's currency. Returns `na` if the data is unavailable.

| Property | Detail        |
|----------|---------------|
| Type     | `float`       |
| Returns  | `float \| na` |

---

### earnings.future_time

Returns a UNIX timestamp (milliseconds) indicating the expected time of the next earnings report. Returns `na` if unavailable.

| Property | Detail      |
|----------|-------------|
| Type     | `int`       |
| Returns  | `int \| na` |

---

### earnings.future_period_end_time

Returns a UNIX timestamp (milliseconds) for the last day of the financial period covered by the next earnings report. Returns `na` if unavailable.

| Property | Detail      |
|----------|-------------|
| Type     | `int`       |
| Returns  | `int \| na` |

---

## Constants

These constants are passed to `request.earnings()` to specify which earnings value to retrieve.

| Constant                   | Description                                               |
|----------------------------|-----------------------------------------------------------|
| `earnings.actual`          | The earnings value exactly as reported                    |
| `earnings.estimate`        | The estimated (consensus) earnings value                  |
| `earnings.standardized`    | The standardized earnings value for cross-instrument comparison |

**Example usage with `request.earnings()`:**

```python
eps_actual: float = request.earnings("AAPL", earnings.actual)
eps_estimate: float = request.earnings("AAPL", earnings.estimate)
```