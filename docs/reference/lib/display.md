<!--
---
weight: 473
title: "display"
description: "Plot display mode constants"
icon: "visibility"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["display", "library", "reference"]
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

| Constant               | Description                                                                                         |
|------------------------|-----------------------------------------------------------------------------------------------------|
| `display.all`          | Show in all available locations (chart pane, price scale, status line, and Data Window).            |
| `display.none`         | Hide entirely — the value is not shown anywhere in the UI.                                          |
| `display.pane`         | Show only in the chart pane where the script is rendered.                                           |
| `display.price_scale`  | Show the plot's label and value on the price scale (if the chart settings allow it).                |
| `display.status_line`  | Show the value in the status line next to the script title.                                         |
| `display.data_window`  | Show the value in the Data Window panel.                                                            |

Constants can be combined using the `+` operator:

```python
plot(close, display=display.pane + display.status_line)
```