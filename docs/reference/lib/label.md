<!--
---
weight: 433
title: "label"
description: "Chart labels — create, modify, and style text labels"
icon: "label"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["label", "library", "reference"]
---
-->

# label

Chart labels — create, modify, and style text labels on the chart.

## Quick Example

```python
from pynecore.lib import (
    close, high, low, bar_index, label, color, xloc, yloc
)
from pynecore.types import Persistent

@script.indicator(title="Label Example", overlay=True)
def main():
    # Create a label at current bar
    lbl: Persistent[label] = label.new(
        bar_index,
        high,
        text="Peak",
        color=color.red,
        style=label.style_label_up,
        textcolor=color.white
    )
    
    # Modify label properties
    label.set_text(lbl, "New Text")
    label.set_color(lbl, color.blue)
    label.set_y(lbl, low)
```

## Functions

### label.new()

Creates a new label object on the chart.

**Overload 1: With coordinates**

```python
lbl: label = label.new(
    x: int,
    y: int | float,
    text: str = "",
    xloc: xloc = xloc.bar_index,
    yloc: yloc = yloc.price,
    color: color = color.blue,
    style: label_style = label.style_label_down,
    textcolor: color = color.white,
    size: size = size.normal,
    textalign: text_align = text_align.center,
    tooltip: str = "",
    force_overlay: bool = False
) → label
```

| Parameter | Type | Description |
|-----------|------|-------------|
| x | int | Bar index (if xloc=xloc.bar_index) or UNIX timestamp (if xloc=xloc.bar_time) |
| y | int \| float | Price level (used only if yloc=yloc.price) |
| text | str | Label text (default: "") |
| xloc | xloc | Coordinate mode: xloc.bar_index or xloc.bar_time |
| yloc | yloc | Vertical positioning: yloc.price, yloc.abovebar, or yloc.belowbar |
| color | color | Border and arrow color (default: blue) |
| style | label_style | Visual style (default: label.style_label_down) |
| textcolor | color | Text color (default: white) |
| size | size | Text/arrow size: size.tiny, size.small, size.normal, size.large (default: normal) |
| textalign | text_align | Alignment: text_align.left, text_align.center, text_align.right |
| tooltip | str | Hover tooltip text (default: "") |
| force_overlay | bool | If true, display on main chart pane (default: false) |

**Overload 2: With chart.point**

```python
lbl: label = label.new(
    point: chart.point,
    text: str = "",
    xloc: xloc = xloc.bar_index,
    yloc: yloc = yloc.price,
    color: color = color.blue,
    style: label_style = label.style_label_down,
    textcolor: color = color.white,
    size: size = size.normal,
    textalign: text_align = text_align.center,
    tooltip: str = "",
    force_overlay: bool = False
) → label
```

Uses a chart.point object to specify position.

### label.delete()

Deletes a label from the chart. If already deleted, does nothing.

```python
label.delete(id: label) → None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| id | label | Label object to delete |

### label.copy()

Clones a label object with all current properties.

```python
lbl_copy: label = label.copy(id: label) → label
```

| Parameter | Type | Description |
|-----------|------|-------------|
| id | label | Label to clone |

Returns: New label with same properties as the original.

### label.get_text()

Returns the text of a label.

```python
txt: str = label.get_text(id: label) → str
```

| Parameter | Type | Description |
|-----------|------|-------------|
| id | label | Label object |

Returns: The label's text string.

### label.get_x()

Returns the x-coordinate (bar index or UNIX time) of the label position.

```python
x: int = label.get_x(id: label) → int
```

| Parameter | Type | Description |
|-----------|------|-------------|
| id | label | Label object |

Returns: Bar index (if xloc=xloc.bar_index) or UNIX timestamp in milliseconds (if xloc=xloc.bar_time).

### label.get_y()

Returns the price level of the label position.

```python
y: float = label.get_y(id: label) → float
```

| Parameter | Type | Description |
|-----------|------|-------------|
| id | label | Label object |

Returns: Price as floating-point value.

### label.set_text()

Sets the text of a label.

```python
label.set_text(id: label, text: str) → None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| id | label | Label object |
| text | str | New text value |

### label.set_x()

Sets the x-coordinate (bar index or time) of the label position.

```python
label.set_x(id: label, x: int) → None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| id | label | Label object |
| x | int | New bar index or UNIX timestamp (milliseconds) |

### label.set_y()

Sets the price level of the label position.

```python
label.set_y(id: label, y: int | float) → None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| id | label | Label object |
| y | int \| float | New price level |

### label.set_xy()

Sets both x and y coordinates of the label position.

```python
label.set_xy(id: label, x: int, y: int | float) → None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| id | label | Label object |
| x | int | New bar index or UNIX timestamp (milliseconds) |
| y | int \| float | New price level |

Example:
```python
label.set_xy(lbl, bar_index + 1, high + 10)  # Move label
```

### label.set_color()

Sets the border and arrow color of a label.

```python
label.set_color(id: label, color: color) → None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| id | label | Label object |
| color | color | New color value |

### label.set_textcolor()

Sets the text color of a label.

```python
label.set_textcolor(id: label, textcolor: color) → None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| id | label | Label object |
| textcolor | color | New text color |

### label.set_size()

Sets the size of the label text and arrow symbol.

```python
label.set_size(id: label, size: size) → None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| id | label | Label object |
| size | size | New size: size.tiny, size.small, size.normal, or size.large |

### label.set_style()

Sets the visual style of the label.

```python
label.set_style(id: label, style: label_style) → None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| id | label | Label object |
| style | label_style | New style constant (e.g., label.style_label_up) |

### label.set_textalign()

Sets the text alignment within the label.

```python
label.set_textalign(id: label, textalign: text_align) → None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| id | label | Label object |
| textalign | text_align | Alignment: text_align.left, text_align.center, or text_align.right |

### label.set_tooltip()

Sets the hover tooltip text for a label.

```python
label.set_tooltip(id: label, tooltip: str) → None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| id | label | Label object |
| tooltip | str | Tooltip text to display on hover |

### label.set_yloc()

Sets the y-location calculation mode for the label position.

```python
label.set_yloc(id: label, yloc: yloc) → None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| id | label | Label object |
| yloc | yloc | Mode: yloc.price, yloc.abovebar, or yloc.belowbar |

## Variables

### label.all

Returns an array of all label objects currently drawn by the script.

```python
all_labels: list[label] = label.all  # list[label]
```

## Constants

All label style constants for use with `label.new()` and `label.set_style()`:

| Constant | Style |
|----------|-------|
| label.style_none | No visible shape |
| label.style_xcross | X-shaped cross |
| label.style_cross | Plus-shaped cross |
| label.style_triangleup | Upward-pointing triangle |
| label.style_triangledown | Downward-pointing triangle |
| label.style_flag | Flag marker |
| label.style_circle | Circle |
| label.style_arrowup | Upward-pointing arrow |
| label.style_arrowdown | Downward-pointing arrow |
| label.style_label_up | Text label above the bar |
| label.style_label_down | Text label below the bar |
| label.style_label_left | Text label to the left |
| label.style_label_right | Text label to the right |
| label.style_label_upper_left | Text label in upper-left corner |
| label.style_label_upper_right | Text label in upper-right corner |
| label.style_label_lower_left | Text label in lower-left corner |
| label.style_label_lower_right | Text label in lower-right corner |
| label.style_label_center | Text label centered |
| label.style_square | Square shape |
| label.style_diamond | Diamond shape |
| label.style_text_outline | Text with outline |

## Compatibility

**Not yet implemented:**
- `label.set_point()` — Not available
- `label.set_text_font_family()` — Font family control not yet supported
- `label.set_text_formatting()` — Text formatting (bold, italic) not yet supported
- `label.set_xloc()` — Direct xloc changes not supported; use `label.new()` or `label.set_x()` instead