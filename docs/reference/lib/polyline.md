<!--
---
weight: 437
title: "polyline"
description: "Multi-segment line drawings"
icon: "polyline"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["polyline", "library", "reference"]
---
-->

# polyline

The `polyline` namespace provides functions to create multi-segment line drawings on the chart. Each polyline connects an array of points sequentially, optionally with curved segments and fill color. Polylines are useful for drawing trend channels, patterns, or custom technical structures.

## Quick Example

```python
from pynecore.lib import (
    polyline, chart, close, high, low, bar_index,
    color, script
)

@script.indicator(title="Polyline Pattern", overlay=True)
def main():
    # Create a simple trend channel using two points
    if bar_index == 50:
        point1: chart.point = chart.point(50, low[0])
        point2: chart.point = chart.point(100, high[0])
        points: list[chart.point] = [point1, point2]
        
        # Draw a curved, closed polyline with blue line and light fill
        pl: polyline = polyline.new(
            points=points,
            curved=True,
            closed=False,
            line_color=color.blue,
            fill_color=color.new(color.blue, 80),
            line_width=2
        )
```

## Functions

### polyline.new()

Creates a new polyline by connecting points sequentially with line segments.

| Parameter | Type | Description |
|-----------|------|-------------|
| `points` | `list[chart.point]` | Array of points to connect |
| `curved` | `bool` | Use curved line segments instead of straight (default: `False`) |
| `closed` | `bool` | Connect the last point back to the first (default: `False`) |
| `xloc` | `xloc` | Use `xloc.bar_index` (default) or `xloc.bar_time` for x-coordinate |
| `line_color` | `color` | Line segment color (default: `color.blue`) |
| `fill_color` | `color \| None` | Fill color inside closed polylines; `None` for no fill (default: `None`) |
| `line_style` | `line_style` | Line style: `polyline.style_solid`, `style_dashed`, etc. (default: solid) |
| `line_width` | `int` | Line width in pixels (default: `1`) |
| `force_overlay` | `bool` | Draw on main chart even if indicator is in separate pane (default: `False`) |

**Returns:** `polyline` object, or `na` if points array is empty or contains `na` values.

**Example:**
```python
pt1: chart.point = chart.point(0, 100.0)
pt2: chart.point = chart.point(10, 105.0)
pl: polyline = polyline.new([pt1, pt2], line_color=color.green, line_width=2)
```

### polyline.delete()

Removes a polyline from the chart. Has no effect if the polyline ID does not exist.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `polyline` | Polyline object to delete |

**Returns:** `None`

**Example:**
```python
polyline.delete(pl)  # Removes the polyline
```

## Variables

### polyline.all

Returns an array of all polyline objects currently drawn by the script.

**Type:** `list[polyline]`

**Example:**
```python
all_polylines: list[polyline] = polyline.all
for pl in all_polylines:
    polyline.delete(pl)  # Delete all polylines
```

## Constants

Line style constants for the `line_style` parameter:

| Constant | Usage |
|----------|-------|
| `polyline.style_solid` | Solid line (default) |
| `polyline.style_dashed` | Dashed pattern |
| `polyline.style_dotted` | Dotted pattern |
| `polyline.style_arrow_left` | Arrow pointing left |
| `polyline.style_arrow_right` | Arrow pointing right |
| `polyline.style_arrow_both` | Arrows at both ends |

## Compatibility

All polyline functions are fully implemented. Polylines support curved and closed modes, custom line styles, and optional fill colors. Note that `xloc.bar_time` requires valid bar timestamps in the `chart.point` objects.