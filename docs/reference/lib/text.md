<!--
---
weight: 483
title: "text"
description: "Text alignment and wrapping constants"
icon: "text_format"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["text", "library", "reference"]
---
-->

# text

The `text` namespace provides constants for controlling text alignment, formatting, and wrapping in visual drawing objects such as labels, boxes, and table cells.

## Quick Example

```python
from pynecore.lib import script, label, text, bar_index, high, low, close, ta

@script.indicator(title="Signal Labels", overlay=True)
def main():
    is_cross: bool = ta.crossover(close, ta.sma(close, 20))
    if is_cross:
        label.new(
            bar_index, high,
            text="BUY",
            textalign=text.align_center,
            style=label.style_label_up,
            text_formatting=text.format_bold
        )
```

## Constants

### Alignment

| Constant              | Description                                                       |
|-----------------------|-------------------------------------------------------------------|
| `text.align_left`     | Aligns text to the left horizontally                             |
| `text.align_center`   | Centers text horizontally or vertically                          |
| `text.align_right`    | Aligns text to the right horizontally                            |
| `text.align_top`      | Aligns text to the top vertically                                |
| `text.align_bottom`   | Aligns text to the bottom vertically                             |

Horizontal alignment constants (`align_left`, `align_center`, `align_right`) are used with the `textalign` parameter of `label.new()` and `label.set_textalign()`, and the `text_halign` parameter of `box.new()` and `box.set_text_halign()`.

Vertical alignment constants (`align_top`, `align_center`, `align_bottom`) are used with the `text_valign` parameter of `box.new()`, `box.set_text_valign()`, `table.cell()`, and `table.cell_set_text_valign()`.

### Formatting

| Constant              | Description                                          |
|-----------------------|------------------------------------------------------|
| `text.format_bold`    | Renders text in bold                                 |
| `text.format_italic`  | Renders text in italic                               |
| `text.format_none`    | No special formatting (default)                      |

Used with the `text_formatting` parameter of `label.new()`, `box.new()`, `table.cell()`, and their corresponding `*set_text_formatting()` functions.

### Wrapping

| Constant           | Description                                                  |
|--------------------|--------------------------------------------------------------|
| `text.wrap_auto`   | Text wraps automatically to fit within the object's bounds   |
| `text.wrap_none`   | Text wrapping is disabled; text overflows if too long        |

Used with the `text_wrap` parameter of `box.new()` and `box.set_text_wrap()`.