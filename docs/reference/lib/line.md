<!--
---
weight: 434
title: "line"
description: "Chart lines — create and modify line drawings"
icon: "timeline"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["line", "library", "reference"]
---
-->

# line

Chart lines — create and modify line drawings. The `line` namespace provides functions to draw lines on a chart with customizable colors, styles, and extending behavior.

## Quick Example

```python
from pynecore.lib import close, bar_index, line, color, script

@script.indicator(title="Support Lines", overlay=True)
def main():
    # Draw a line connecting two price points
    support = line.new(
        x1=bar_index - 20,
        y1=close[20],
        x2=bar_index,
        y2=close,
        color=color.red,
        width=2,
        style=line.style_dashed
    )
    
    # Modify the line's appearance
    line.set_color(support, color.blue)
    line.set_width(support, 3)
```

## Creation and Management

### line.new()

Creates a new line object on the chart.

**Overload 1: Using coordinates**

| Parameter | Type | Description |
|-----------|------|-------------|
| x1 | int | Bar index or UNIX time of the first point |
| y1 | float | Price of the first point |
| x2 | int | Bar index or UNIX time of the second point |
| y2 | float | Price of the second point |
| xloc | xloc | Reference type: `xloc.bar_index` (default) or `xloc.bar_time` |
| extend | extend | Extending behavior (default: `extend.none`) |
| color | color | Line color (default: `color.blue`) |
| style | line.LineEnum | Line style (default: `line.style_solid`) |
| width | int | Line width in pixels (default: 1) |
| force_overlay | bool | Force display on main chart pane (default: False) |

**Returns:** `line.Line`

**Overload 2: Using chart.point objects**

| Parameter | Type | Description |
|-----------|------|-------------|
| first_point | chart.point | Starting point of the line |
| second_point | chart.point | Ending point of the line |
| xloc | xloc | Reference type (default: `xloc.bar_index`) |
| extend | extend | Extending behavior (default: `extend.none`) |
| color | color | Line color (default: `color.blue`) |
| style | line.LineEnum | Line style (default: `line.style_solid`) |
| width | int | Line width in pixels (default: 1) |
| force_overlay | bool | Force display on main chart pane (default: False) |

**Returns:** `line.Line`

### line.copy()

Clones an existing line object.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | line.Line | Line object to copy |

**Returns:** `line.Line`

```python
duplicate: line.Line = line.copy(original)
```

### line.delete()

Deletes a line object from the chart. Does nothing if already deleted.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | line.Line | Line object to delete |

**Returns:** None

## Getters

### line.get_x1()

Returns the bar index or UNIX time of the line's first point.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | line.Line | Line object |

**Returns:** `int`

```python
x: int = line.get_x1(my_line)
```

### line.get_x2()

Returns the bar index or UNIX time of the line's second point.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | line.Line | Line object |

**Returns:** `int`

### line.get_y1()

Returns the price of the line's first point.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | line.Line | Line object |

**Returns:** `float`

### line.get_y2()

Returns the price of the line's second point.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | line.Line | Line object |

**Returns:** `float`

### line.get_price()

Returns the interpolated price level of a line at a specified bar index.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | line.Line | Line object |
| x | int | Bar index |

**Returns:** `float`

```python
price: float = line.get_price(my_line, bar_index - 5)
```

## Coordinate and Point Setters

### line.set_x1()

Sets the bar index or UNIX time of the line's first point.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | line.Line | Line object |
| x | int | New bar index or UNIX time |

**Returns:** None

### line.set_x2()

Sets the bar index or UNIX time of the line's second point.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | line.Line | Line object |
| x | int | New bar index or UNIX time |

**Returns:** None

### line.set_y1()

Sets the price of the line's first point.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | line.Line | Line object |
| y | float | New price |

**Returns:** None

### line.set_y2()

Sets the price of the line's second point.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | line.Line | Line object |
| y | float | New price |

**Returns:** None

### line.set_xy1()

Sets both the bar index/time and price of the line's first point.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | line.Line | Line object |
| x | int | New bar index or UNIX time |
| y | float | New price |

**Returns:** None

### line.set_xy2()

Sets both the bar index/time and price of the line's second point.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | line.Line | Line object |
| x | int | New bar index or UNIX time |
| y | float | New price |

**Returns:** None

### line.set_xloc()

Sets the x-location mode and updates both x-coordinates of the line.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | line.Line | Line object |
| x1 | int | New first x-coordinate |
| x2 | int | New second x-coordinate |
| xloc | xloc | Reference mode: `xloc.bar_index` or `xloc.bar_time` |

**Returns:** None

### line.set_first_point()

Sets the first point of the line using a chart.point object.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | line.Line | Line object |
| point | chart.point | New first point |

**Returns:** None

### line.set_second_point()

Sets the second point of the line using a chart.point object.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | line.Line | Line object |
| point | chart.point | New second point |

**Returns:** None

## Appearance Setters

### line.set_color()

Sets the color of the line.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | line.Line | Line object |
| color | color | New line color |

**Returns:** None

```python
line.set_color(my_line, color.red)
```

### line.set_style()

Sets the line style (solid, dotted, dashed, or arrow variants).

| Parameter | Type | Description |
|-----------|------|-------------|
| id | line.Line | Line object |
| style | line.LineEnum | New line style |

**Returns:** None

```python
line.set_style(my_line, line.style_dashed)
```

### line.set_width()

Sets the line width in pixels.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | line.Line | Line object |
| width | int | New width in pixels |

**Returns:** None

```python
line.set_width(my_line, 3)
```

### line.set_extend()

Sets the extending behavior of the line.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | line.Line | Line object |
| extend | extend | Extending type: `extend.none`, `extend.left`, `extend.right`, or `extend.both` |

**Returns:** None

```python
line.set_extend(my_line, extend.both)
```

## Collections

### line.all

Returns an array of all line objects currently on the chart.

**Type:** `list[line.Line]`

```python
all_lines: list[line.Line] = line.all
```

## Style Constants

| Constant | Description |
|----------|-------------|
| `line.style_solid` | Solid line |
| `line.style_dotted` | Dotted line |
| `line.style_dashed` | Dashed line |
| `line.style_arrow_left` | Solid line with arrow at first point |
| `line.style_arrow_right` | Solid line with arrow at second point |
| `line.style_arrow_both` | Solid line with arrows at both points |

## Compatibility

All functions and constants in the `line` namespace are fully implemented. `NA` values are handled gracefully — operations on `NA` line objects return `NA` values or have no effect for setters.