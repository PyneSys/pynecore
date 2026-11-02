<!--
---
weight: 473
title: "display"
description: "Plot display mode constants"
icon: "visibility"
date: "2026-03-28"
lastmod: "2026-11-02"
draft: false
toc: true
categories: ["Reference"]
tags: ["display"]
---
-->

# display

The `display` namespace provides constants that control where plotted values and inputs appear in the TradingView chart interface. These constants are used as arguments to the `display` parameter of `plot*()` and `input*()` functions.

## Quick Example

```python
from pynecore.lib import script, close, ta, plot, display

@script.indicator(title="Display Example", overlay=False)
def main():
    sma: float = ta.sma(close, 20)
    # Show only in the chart pane, hide from price scale and status line
    plot(sma, title="SMA20", display=display.pane)
```

## Constants

| Constant                | Value        | Description                                                                              |
|-------------------------|--------------|------------------------------------------------------------------------------------------|
| `display.all`           | `0xFFFFFFFF` | Show in every location.                                                                  |
| `display.none`          | `0`          | Hide entirely: the value is not shown anywhere in the UI.                                |
| `display.pane`          | `1`          | Show in the chart pane where the script is rendered.                                     |
| `display.data_window`   | `2`          | Show the value in the Data Window panel.                                                 |
| `display.price_scale`   | `4`          | Show the plot's label and value on the price scale (if the chart settings allow it).     |
| `display.status_line`   | `8`          | Show the value in the status line next to the script title.                              |
| `display.pine_screener` | `16`         | Show the value in the Pine Screener.                                                     |

Each location is one bit, the same values TradingView uses. `display.all` has every bit set, so it
also covers locations that have no constant of their own.

## Combining Locations

A `display` value is a set of locations. `+` adds locations to the set and `-` removes them:

```python
# Only the status line and the Data Window
plot(close, display=display.status_line + display.data_window)

# Everywhere except the chart pane
plot(close, display=display.all - display.pane)
```

Both work as set operations, not as integer arithmetic: adding a location that is already in the
set, or removing one that is not, leaves the set unchanged. `display.pane + display.pane` is
`display.pane`, and `display.pane - display.data_window` is still `display.pane`. The result is a
`display` value again, so it can be combined further.

`plot()`, `plotshape()`, `plotchar()`, `plotarrow()`, `plotbar()`, `plotcandle()` and the `input*()`
functions accept any combination. `hline()`, `fill()`, `bgcolor()` and `barcolor()` accept only
`display.all` and `display.none`.

## Visual Output

The `--viz` stream writes a `display` value as the list of locations it covers, in bit order, for
example `["data_window", "status_line"]`. `display.all` lists every location and `display.none` is
an empty list. See [Visual Output](../../programmatic/visual-output.md).
