<!--
---
weight: 481
title: "shape"
description: "plotshape marker shape constants"
icon: "category"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["shape", "library", "reference"]
---
-->

# shape

The `shape` namespace provides marker shape constants for use with `plotshape()`. Each constant specifies the visual style of the marker rendered on the chart.

## Quick Example

```python
from pynecore.lib import script, plotshape, ta, close, high, low, shape, color

@script.indicator(title="Shape Demo", overlay=True)
def main():
    cross_up = ta.crossover(close, ta.sma(close, 20))
    cross_down = ta.crossunder(close, ta.sma(close, 20))

    plotshape(cross_up, style=shape.triangleup, location="belowbar",
              color=color.green, size="small")
    plotshape(cross_down, style=shape.triangledown, location="abovebar",
              color=color.red, size="small")
```

## Constants

All `shape` constants are of type `Shape` and are passed to the `style` parameter of `plotshape()`.

| Constant              | Description                                      |
|-----------------------|--------------------------------------------------|
| `shape.arrowdown`     | Downward-pointing arrow                          |
| `shape.arrowup`       | Upward-pointing arrow                            |
| `shape.circle`        | Filled circle                                    |
| `shape.cross`         | Cross (plus sign)                                |
| `shape.diamond`       | Diamond shape                                    |
| `shape.flag`          | Flag marker                                      |
| `shape.labeldown`     | Label with downward pointer                      |
| `shape.labelup`       | Label with upward pointer                        |
| `shape.square`        | Filled square                                    |
| `shape.triangledown`  | Downward-pointing triangle                       |
| `shape.triangleup`    | Upward-pointing triangle                         |
| `shape.xcross`        | X-shaped cross                                   |

## Compatibility

All 12 constants are fully implemented. No stubs or missing items.