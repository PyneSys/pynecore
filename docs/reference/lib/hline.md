<!--
---
weight: 462
title: "hline"
description: "Horizontal line drawing functions and style constants"
icon: "horizontal_rule"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["hline", "library", "reference"]
---
-->

# hline

`hline` draws a horizontal line at a fixed price level on the chart. Unlike `line.new()`, it renders across the entire chart history and future bars. The module itself is callable — `hline(...)` creates a line, while `hline.style_*` constants control its appearance.

## Quick Example

```python
from pynecore.lib import script, hline, color

@script.indicator(title="Support/Resistance Levels", overlay=True)
def main():
    hline(100.0, title="Resistance", color=color.red, linestyle=hline.style_dashed, linewidth=2)
    hline(50.0, title="Support", color=color.green, linestyle=hline.style_solid)
```

## Functions

### hline()

Renders a horizontal line at a fixed price level that spans the entire chart.

| Parameter   | Type             | Default             | Description                                                                          |
|-------------|------------------|---------------------|--------------------------------------------------------------------------------------|
| `price`     | float            | required            | Price level at which the line is drawn                                               |
| `title`     | str              | `""`                | Label for the line (shown in Format dialog)                                          |
| `color`     | color.Color      | `color.blue`        | Line color — must be a constant, not a dynamic expression                            |
| `linestyle` | HLineEnum        | `hline.style_solid` | Line style — one of `hline.style_solid`, `hline.style_dotted`, `hline.style_dashed` |
| `linewidth` | int              | `1`                 | Line thickness in pixels                                                             |
| `editable`  | bool             | `True`              | Whether the style is editable in the Format dialog                                   |
| `display`   | display.Display  | `display.all`       | Where to display the line — `display.all` or `display.none`                         |

**Returns:** `HLine` — an hline object that can be used with `fill()`

```python
mid: HLine = hline(200.0, title="Midpoint", color=color.gray, linestyle=hline.style_dotted)
```

## Constants

| Constant              | Description                    |
|-----------------------|--------------------------------|
| `hline.style_solid`   | Solid line style (default)     |
| `hline.style_dotted`  | Dotted line style              |
| `hline.style_dashed`  | Dashed line style              |

## Compatibility

All three style constants and the `hline()` function are fully supported. Note that the `color` parameter must be a constant value — dynamic (per-bar) color expressions are not supported.