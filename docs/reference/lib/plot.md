<!--
---
weight: 461
title: "plot"
description: "Plot functions for displaying data on charts"
icon: "show_chart"
date: "2026-03-28"
lastmod: "2026-10-25"
draft: false
toc: true
categories: ["Reference"]
tags: ["plot"]
---
-->

# plot

The `plot` namespace provides chart plotting functionality. The namespace itself is callable — `plot(...)` records a value series for output. The values go to the plot CSV; the visual parameters (color, line width, style, …) are recorded as plot metadata and written to the opt-in [visual output](../../programmatic/visual-output.md) (`pyne run --viz`, `ScriptRunner(viz_path=...)`) for renderers to use.

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

Records a series value on every bar for output. The value is written to the plot CSV; the other
parameters are recorded as the plot's visual metadata.

| Parameter       | Type          | Default | Description                                                                                  |
|-----------------|---------------|---------|----------------------------------------------------------------------------------------------|
| `series`        | any           | —       | The value to record each bar.                                                                |
| `title`         | str \| None   | `None`  | Label for the plot. Defaults to `"Plot"`. Duplicate titles get a number appended.            |
| `color`         | color \| None | `None`  | Plot color. A color that changes from bar to bar is recorded as a per-bar color channel.     |
| `linewidth`     | int           | `1`     | Line width in pixels.                                                                        |
| `style`         | plot style    | `None`  | One of the `plot.style_*` constants; `None` means `plot.style_line`.                         |
| `trackprice`    | bool          | `False` | Show a horizontal price line at the last value.                                              |
| `histbase`      | float         | `0.0`   | Reference level for `style_histogram`, `style_columns` and `style_area`.                     |
| `offset`        | int           | `0`     | Horizontal shift of the plot, in bars.                                                       |
| `join`          | bool          | `False` | Join `style_circles` / `style_cross` points with lines.                                      |
| `editable`      | bool          | `True`  | Whether the plot style is editable in a settings dialog.                                     |
| `show_last`     | int \| None   | `None`  | Plot only the last `show_last` bars.                                                         |
| `display`       | display       | `None`  | Where the plot is displayed (`display.*` constants); recorded as `all` when `None`.          |
| `format`        | str \| None   | `None`  | Value formatting (`format.price`, `format.volume`, …).                                       |
| `precision`     | int \| None   | `None`  | Number of decimal places for the plotted values.                                             |
| `force_overlay` | bool          | `False` | Display the plot on the main chart pane.                                                     |

**Returns:** `Plot` — a plot reference object.

> **Note:** Plot declarations must execute unconditionally in the direct body of the module-level `main()` function, before any possible early return. Calls inside helpers, branches, loops, or conditional expressions raise a source-located `SyntaxError` during AST transformation, before the script runs. This rule also applies to `plotshape`, `plotchar`, `plotarrow`, `plotcandle`, `plotbar`, `hline`, `fill`, `bgcolor`, `barcolor`, and `alertcondition`.

To show a plot conditionally, keep the call unconditional and pass `na` when it should be hidden. The CSV retains its column and records `NaN` on those bars:

```python
plot(close if bar_index >= 3 else na, title="Close Price")
```

Import aliases and single static function aliases are supported. Passing plot functions as callbacks or storing them in containers is rejected because their call scope cannot be verified. Plot references returned by `plot()` can still be stored and passed to `fill()`.

```python
p = plot(close, title="Close Price")
```

---

## Constants

Style constants are passed to the `style` parameter of `plot()` and recorded in the visual output.

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

Line style constants:

| Constant                | Description   |
|-------------------------|---------------|
| `plot.linestyle_solid`  | Solid line.   |
| `plot.linestyle_dashed` | Dashed line.  |
| `plot.linestyle_dotted` | Dotted line.  |

---

## Compatibility Notes

All `plot.style_*` and `plot.linestyle_*` constants exist. `plot()` accepts a `linestyle` argument
so that Pine sources using it run, but the line style is not recorded in the visual output yet;
every other visual parameter in the table above is.

The CSV output holds only the plotted values. Visual parameters reach a renderer only through the
[visual output](../../programmatic/visual-output.md).
