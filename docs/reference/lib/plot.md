<!--
---
weight: 461
title: "plot"
description: "Plot functions for displaying data on charts"
icon: "show_chart"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["plot", "library", "reference"]
---
-->

# plot

The `plot` namespace provides chart plotting functionality. The namespace itself is callable — `plot(...)` records a value series for output. In PyneCore, plots are written to CSV output by default, but the mechanism can be extended. The `plot.*` constants control visual style and line appearance when used in compatible renderers.

## Quick Example

```python
from pynecore.lib import script, plot, close, ta

@script.indicator(title="SMA Plot Example", overlay=True)
def main():
    sma20: float = ta.sma(close, 20)
    plot(sma20, title="SMA 20")
    plot(close, title="Close")
```

---

## Functions

### plot()

Records a series value on every bar for output. In PyneCore, this writes the value to CSV by default.

| Parameter | Type        | Default | Description                                                              |
|-----------|-------------|---------|--------------------------------------------------------------------------|
| `series`  | any         | —       | The value to record each bar.                                            |
| `title`   | str \| None | `None`  | Label for the plot. Defaults to `"Plot"`. Duplicate titles are suffixed. |

**Returns:** `Plot` — a plot reference object.

> **Note:** `plot()` must be called from the script's `main()` function. Calling it from a helper function raises `RuntimeError`.

```python
p = plot(close, title="Close Price")
```

---

## Constants

Style constants are passed to the `style` parameter of `plot()` in compatible renderers.

| Constant                      | Description                                       |
|-------------------------------|---------------------------------------------------|
| `plot.style_area`             | Area fill style.                                  |
| `plot.style_areabr`           | Area fill style with breaks on `na` values.       |
| `plot.style_circles`          | Circles at each bar.                              |
| `plot.style_columns`          | Vertical columns (bar chart style).               |
| `plot.style_cross`            | Cross markers at each bar.                        |
| `plot.style_histogram`        | Histogram bars relative to zero.                  |
| `plot.style_line`             | Continuous line (default).                        |
| `plot.style_linebr`           | Line with breaks on `na` values.                  |
| `plot.style_stepline`         | Stepped line.                                     |
| `plot.style_steplinebr`       | Stepped line with breaks on `na` values.          |
| `plot.style_stepline_diamond` | Stepped line with diamond markers at each vertex. |

---

## Compatibility Notes

The following Pine Script `plot` namespace members are **not available** in PyneCore:

| Name                    | Status                  |
|-------------------------|-------------------------|
| `plot.linestyle_dashed` | Not available in PyneCore |
| `plot.linestyle_dotted` | Not available in PyneCore |
| `plot.linestyle_solid`  | Not available in PyneCore |

The `plot()` function accepts but silently ignores Pine Script visual parameters (`color`, `linewidth`, `style`, `trackprice`, etc.) for source-level compatibility. These have no effect in the current PyneCore output backend.