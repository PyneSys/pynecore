<!--
---
weight: 438
title: "linefill"
description: "Fill areas between two lines"
icon: "format_color_fill"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["linefill", "library", "reference"]
---
-->

# linefill

The `linefill` namespace provides functions to create and manage linefill objects that fill the area between two lines on the chart. Linefills are useful for visual representation of price channels, support/resistance zones, or any area bounded by two lines.

## Quick Example

```python
from pynecore.lib import line, linefill, color, script, bar_index, high, low

@script.indicator(title="Line Fill Example", overlay=True)
def main():
    # Create two lines
    line1 = line.new(bar_index - 5, high, bar_index, high)
    line2 = line.new(bar_index - 5, low, bar_index, low)
    
    # Fill the space between them
    fill = linefill.new(line1, line2, color.new(color.blue, 80))
    
    # Retrieve the first line and update color
    first_line = linefill.get_line1(fill)
    linefill.set_color(fill, color.new(color.green, 50))
```

## Functions

### linefill.new()

Creates a new linefill object that fills the space between two lines and displays it on the chart.

| Parameter | Type | Description |
|-----------|------|-------------|
| `line1` | line | The first line object |
| `line2` | line | The second line object |
| `color` | color | The color used to fill the space between the lines |

**Returns:** linefill

**Example:**
```python
fill = linefill.new(line1, line2, color.new(color.blue, 80))
```

### linefill.delete()

Deletes the specified linefill object. If it has already been deleted, does nothing.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | linefill | A linefill object |

**Returns:** None

**Example:**
```python
linefill.delete(fill)
```

### linefill.get_line1()

Returns the first line object used in the linefill.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | linefill | A linefill object |

**Returns:** line

**Example:**
```python
first_line = linefill.get_line1(fill)
```

### linefill.get_line2()

Returns the second line object used in the linefill.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | linefill | A linefill object |

**Returns:** line

**Example:**
```python
second_line = linefill.get_line2(fill)
```

### linefill.set_color()

Sets the color of the linefill object.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | linefill | A linefill object |
| `color` | color | The new color for the linefill |

**Returns:** None

**Example:**
```python
linefill.set_color(fill, color.new(color.red, 60))
```

## Compatibility

- **Not available:** `linefill.all()` — There is no function to retrieve all linefill objects created by the script.