<!--
---
weight: 494
title: "pivotpointtype"
description: "Pivot point calculation type constants"
icon: "pivot_table_chart"
date: "2026-04-05"
lastmod: "2026-10-25"
draft: false
toc: true
categories: ["Reference"]
tags: ["pivotpointtype"]
---
-->

# pivotpointtype

Constants for selecting the pivot point calculation method used by `ta.pivot_point_levels()`.

## Constants

| Constant                       | Description                          |
|--------------------------------|--------------------------------------|
| `pivotpointtype.traditional`   | Traditional (Standard) pivot points. |
| `pivotpointtype.fibonacci`     | Fibonacci-based pivot points.        |
| `pivotpointtype.woodie`        | Woodie pivot points.                 |
| `pivotpointtype.classic`       | Classic pivot points.                |
| `pivotpointtype.dm`            | DeMark pivot points.                 |
| `pivotpointtype.camarilla`     | Camarilla pivot points.              |

## Example

```python
from pynecore.lib import ta, pivotpointtype, timeframe, script, plot, array

@script.indicator(title="Pivot Points", overlay=True)
def main():
    # Start a new pivot calculation when the trading day changes
    new_day: bool = timeframe.change("D")
    levels = ta.pivot_point_levels(pivotpointtype.traditional, new_day)

    # levels = [P, R1, S1, R2, S2, R3, S3, R4, S4, R5, S5]
    plot(array.get(levels, 0), "P")
    plot(array.get(levels, 1), "R1")
    plot(array.get(levels, 2), "S1")
```

`ta.pivot_point_levels(type, anchor, developing=False)` takes the calculation type, an `anchor`
condition and an optional `developing` flag. A bar where `anchor` is true closes the current period
and starts a new one. The levels are computed from the period that just closed and stay constant
until the next anchor; with `developing=True` they follow the current period as it builds instead.
All levels are `na` until the first anchor, and levels a type does not define (R4–S5 for
`traditional`, everything after S1 for `dm`) are always `na`.

## Compatibility

All constants are fully supported.
