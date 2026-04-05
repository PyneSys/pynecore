<!--
---
weight: 477
title: "location"
description: "Plot location constants (abovebar, belowbar, etc.)"
icon: "place"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["location", "library", "reference"]
---
-->

# location

The `location` namespace provides constants that specify where shapes are plotted on the chart when using `plotshape` and `plotchar`. These constants control vertical positioning relative to bars or chart borders.

## Quick Example

```python
from pynecore.lib import script, ta, close, high, low, plotshape, location, color

@script.indicator(title="Signal Markers", overlay=True)
def main():
    is_cross_up = ta.crossover(close, ta.sma(close, 20))
    is_cross_down = ta.crossunder(close, ta.sma(close, 20))

    plotshape(is_cross_up, style="triangleup", location=location.belowbar, color=color.green)
    plotshape(is_cross_down, style="triangledown", location=location.abovebar, color=color.red)
```

---

## Constants

| Constant             | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `location.abovebar`  | Shape is plotted above each bar of the main series.                         |
| `location.belowbar`  | Shape is plotted below each bar of the main series.                         |
| `location.top`       | Shape is plotted near the top border of the chart pane.                     |
| `location.bottom`    | Shape is plotted near the bottom border of the chart pane.                  |
| `location.absolute`  | Shape is positioned using the indicator's value as an absolute price coordinate. |

---

## Compatibility

All five `location` constants are fully supported in PyneCore.