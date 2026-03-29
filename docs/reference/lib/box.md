<!--
---
weight: 435
title: "box"
description: "Chart boxes — create and modify rectangular drawings"
icon: "check_box_outline_blank"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["box", "library", "reference"]
---
-->

# box

Chart boxes are rectangular drawing objects that can display text, have customizable borders and backgrounds, and be positioned either by bar index or time coordinates. Use boxes to highlight price ranges, mark trade entry/exit zones, or annotate significant chart events.

## Quick Example

```python
from pynecore.lib import (
    close, high, low, bar_index, ta, color, extend, xloc, 
    size, text, line, script
)
from pynecore.types import Persistent

@script.indicator(title="Box Drawer", overlay=True)
def main():
    # Create a box when a crossover occurs
    if ta.crossover(close, ta.sma(close, 20)):
        top_box: Persistent = None
        
        # Draw box from 5 bars ago to current bar
        box_obj = box.new(
            left=bar_index - 5,
            top=high,
            right=bar_index,
            bottom=low,
            border_color=color.green,
            border_width=2,
            border_style=line.style_solid,
            bgcolor=color.new(color.green, 80),
            text="Signal",
            text_color=color.white,
            text_size=size.small
        )
        top_box = box_obj
```

## Functions

### box.new()

Creates a new box object on the chart with specified dimensions, styling, and optional text label.

**Overload 1: Using Chart Points**

| Parameter | Type | Description |
|-----------|------|-------------|
| `top_left` | `chart.point` | Chart point specifying the top-left corner |
| `bottom_right` | `chart.point` | Chart point specifying the bottom-right corner |
| `border_color` | `color.Color` | Border color (default: `color.blue`) |
| `border_width` | `int` | Border width in pixels (default: 1) |
| `border_style` | `line.LineEnum` | Border style: `line.style_solid`, `line.style_dotted`, `line.style_dashed` |
| `extend` | `extend.Extend` | Extend type: `extend.none`, `extend.left`, `extend.right`, `extend.both` |
| `xloc` | `xloc.XLoc` | Coordinate system: `xloc.bar_index` (default) or `xloc.bar_time` |
| `bgcolor` | `color.Color` | Background color (default: `color.blue`) |
| `text` | `str` | Text label to display inside box |
| `text_size` | `size.Size` | Text size (default: `size.auto`): `size.auto`, `size.tiny`, `size.small`, `size.normal`, `size.large`, `size.huge` |
| `text_color` | `color.Color` | Text color (default: `color.black`) |
| `text_halign` | `text.AlignEnum` | Horizontal alignment: `text.align_left`, `text.align_center`, `text.align_right` |
| `text_valign` | `text.AlignEnum` | Vertical alignment: `text.align_top`, `text.align_center`, `text.align_bottom` |
| `text_wrap` | `text.WrapEnum` | Text wrapping: `text.wrap_none` or `text.wrap_word` |
| `text_font_family` | `font.FontFamilyEnum` | Font family: `font.family_default`, `font.family_monospace` |
| `force_overlay` | `bool` | If `True`, draw on main pane regardless of indicator placement |
| `text_formatting` | `text.FormatEnum` | Text formatting: `text.format_none`, `text.format_bold`, `text.format_italic` |

**Return type:** `box`

```python
pt1: chart.point = chart.point(bar_index - 10, high)
pt2: chart.point = chart.point(bar_index, low)
b: box = box.new(top_left=pt1, bottom_right=pt2, bgcolor=color.new(color.green, 80))
```

**Overload 2: Using Raw Coordinates**

| Parameter | Type | Description |
|-----------|------|-------------|
| `left` | `int` | Left edge bar index (if `xloc=bar_index`) or UNIX time in ms (if `xloc=bar_time`) |
| `top` | `float` | Top edge price level |
| `right` | `int` | Right edge bar index or UNIX time |
| `bottom` | `float` | Bottom edge price level |
| Other parameters | — | Same as Overload 1 |

**Return type:** `box`

```python
b: box = box.new(
    left=bar_index - 5, 
    top=100.5, 
    right=bar_index, 
    bottom=99.0,
    border_color=color.red
)
```

### box.copy()

Creates a clone of an existing box object with all properties copied.

| Parameter | Type |
|-----------|------|
| `id` | `box` |

**Return type:** `box`

```python
original: box = box.new(bar_index - 10, 105.0, bar_index, 100.0)
clone: box = box.copy(original)
```

### box.delete()

Removes a box from the chart. If the box has already been deleted, does nothing.

| Parameter | Type |
|-----------|------|
| `id` | `box` |

```python
box.delete(my_box)
```

### box.get_top()

Returns the price value of the box's top border.

| Parameter | Type |
|-----------|------|
| `id` | `box` |

**Return type:** `float`

```python
b: box = box.new(bar_index - 10, 105.0, bar_index, 100.0)
top_price: float = box.get_top(b)  # 105.0
```

### box.get_bottom()

Returns the price value of the box's bottom border.

| Parameter | Type |
|-----------|------|
| `id` | `box` |

**Return type:** `float`

```python
b: box = box.new(bar_index - 10, 105.0, bar_index, 100.0)
bottom_price: float = box.get_bottom(b)  # 100.0
```

### box.get_left()

Returns the left edge position: bar index (if `xloc=bar_index`) or UNIX time in milliseconds (if `xloc=bar_time`).

| Parameter | Type |
|-----------|------|
| `id` | `box` |

**Return type:** `int`

```python
b: box = box.new(bar_index - 10, 105.0, bar_index, 100.0)
left_idx: int = box.get_left(b)  # bar_index - 10
```

### box.get_right()

Returns the right edge position: bar index or UNIX time in milliseconds.

| Parameter | Type |
|-----------|------|
| `id` | `box` |

**Return type:** `int`

```python
b: box = box.new(bar_index - 10, 105.0, bar_index, 100.0)
right_idx: int = box.get_right(b)  # bar_index
```

### box.set_top()

Moves the top border to a new price level.

| Parameter | Type |
|-----------|------|
| `id` | `box` |
| `top` | `float` |

```python
box.set_top(my_box, 110.0)
```

### box.set_bottom()

Moves the bottom border to a new price level.

| Parameter | Type |
|-----------|------|
| `id` | `box` |
| `bottom` | `float` |

```python
box.set_bottom(my_box, 99.0)
```

### box.set_left()

Moves the left border to a new position (bar index or time depending on `xloc`).

| Parameter | Type |
|-----------|------|
| `id` | `box` |
| `left` | `int` |

```python
box.set_left(my_box, bar_index - 20)
```

### box.set_right()

Moves the right border to a new position.

| Parameter | Type |
|-----------|------|
| `id` | `box` |
| `right` | `int` |

```python
box.set_right(my_box, bar_index + 5)
```

### box.set_lefttop()

Sets both left and top coordinates in a single call.

| Parameter | Type |
|-----------|------|
| `id` | `box` |
| `left` | `int` |
| `top` | `float` |

```python
box.set_lefttop(my_box, bar_index - 15, 108.5)
```

### box.set_rightbottom()

Sets both right and bottom coordinates in a single call.

| Parameter | Type |
|-----------|------|
| `id` | `box` |
| `right` | `int` |
| `bottom` | `float` |

```python
box.set_rightbottom(my_box, bar_index, 101.0)
```

### box.set_top_left_point()

Moves the top-left corner to a new `chart.point` position.

| Parameter | Type |
|-----------|------|
| `id` | `box` |
| `point` | `chart.point` |

```python
pt: chart.point = chart.point(bar_index - 10, 110.0)
box.set_top_left_point(my_box, pt)
```

### box.set_bottom_right_point()

Moves the bottom-right corner to a new `chart.point` position.

| Parameter | Type |
|-----------|------|
| `id` | `box` |
| `point` | `chart.point` |

```python
pt: chart.point = chart.point(bar_index, 100.0)
box.set_bottom_right_point(my_box, pt)
```

### box.set_xloc()

Changes the coordinate system and updates both left and right edges.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `box` | |
| `left` | `int` | New left position |
| `right` | `int` | New right position |
| `xloc` | `xloc.XLoc` | `xloc.bar_index` or `xloc.bar_time` |

```python
box.set_xloc(my_box, bar_index - 10, bar_index, xloc.bar_index)
```

### box.set_border_color()

Changes the border color of all four sides.

| Parameter | Type |
|-----------|------|
| `id` | `box` |
| `color` | `color.Color` |

```python
box.set_border_color(my_box, color.red)
```

### box.set_border_width()

Changes the border width in pixels.

| Parameter | Type |
|-----------|------|
| `id` | `box` |
| `width` | `int` |

```python
box.set_border_width(my_box, 3)
```

### box.set_border_style()

Changes the border style (solid, dotted, or dashed).

| Parameter | Type |
|-----------|------|
| `id` | `box` |
| `style` | `line.LineEnum` |

```python
box.set_border_style(my_box, line.style_dotted)
```

### box.set_bgcolor()

Changes the box's background color.

| Parameter | Type |
|-----------|------|
| `id` | `box` |
| `color` | `color.Color` |

```python
box.set_bgcolor(my_box, color.new(color.yellow, 50))
```

### box.set_extend()

Sets how the box's horizontal borders extend beyond left and right edges.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `box` | |
| `extend` | `extend.Extend` | `extend.none` (no extension), `extend.left` (extend left only), `extend.right` (extend right only), `extend.both` (extend both directions) |

```python
box.set_extend(my_box, extend.both)
```

### box.set_text()

Sets the text displayed inside the box.

| Parameter | Type |
|-----------|------|
| `id` | `box` |
| `text` | `str` |

```python
box.set_text(my_box, "Entry Zone")
```

### box.set_text_color()

Changes the color of the text inside the box.

| Parameter | Type |
|-----------|------|
| `id` | `box` |
| `text_color` | `color.Color` |

```python
box.set_text_color(my_box, color.white)
```

### box.set_text_size()

Changes the text size.

| Parameter | Type |
|-----------|------|
| `id` | `box` |
| `text_size` | `size.Size` |

```python
box.set_text_size(my_box, size.large)
```

### box.set_text_halign()

Changes the horizontal alignment of text inside the box.

| Parameter | Type |
|-----------|------|
| `id` | `box` |
| `text_halign` | `text.AlignEnum` |

```python
box.set_text_halign(my_box, text.align_left)
```

### box.set_text_valign()

Changes the vertical alignment of text inside the box.

| Parameter | Type |
|-----------|------|
| `id` | `box` |
| `text_valign` | `text.AlignEnum` |

```python
box.set_text_valign(my_box, text.align_bottom)
```

### box.set_text_wrap()

Sets the text wrapping mode.

| Parameter | Type |
|-----------|------|
| `id` | `box` |
| `text_wrap` | `text.WrapEnum` |

```python
box.set_text_wrap(my_box, text.wrap_word)
```

### box.set_text_font_family()

Changes the font family used for the text.

| Parameter | Type |
|-----------|------|
| `id` | `box` |
| `text_font_family` | `font.FontFamilyEnum` |

```python
box.set_text_font_family(my_box, font.family_monospace)
```

### box.set_text_formatting()

Sets text formatting attributes like bold or italic.

| Parameter | Type |
|-----------|------|
| `id` | `box` |
| `text_formatting` | `text.FormatEnum` |

```python
box.set_text_formatting(my_box, text.format_bold)
```

## Variables

### box.all

Returns an array containing all box objects currently drawn by the script.

**Type:** `list[box]`

```python
all_boxes: list[box] = box.all
num_boxes: int = len(all_boxes)
```

## Compatibility

All `box` namespace functions are fully implemented in PyneCore. The namespace supports both bar index and time-based coordinate systems, comprehensive styling options, and text formatting.