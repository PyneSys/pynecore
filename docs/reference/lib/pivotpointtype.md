<!--
---
weight: 494
title: "pivotpointtype"
description: "Pivot point calculation type constants"
icon: "pivot_table_chart"
date: "2026-04-05"
lastmod: "2026-04-05"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["pivotpointtype", "library", "reference"]
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
from pynecore.lib import ta, pivotpointtype, script

@script.indicator(title="Pivot Points")
def main():
    levels = ta.pivot_point_levels(pivotpointtype.traditional)
```

## Compatibility

All constants are fully supported.
