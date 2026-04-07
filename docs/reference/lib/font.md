<!--
---
weight: 475
title: "font"
description: "Font family constants"
icon: "font_download"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["font", "library", "reference"]
---
-->

# font

The `font` namespace provides font family constants for use with text-rendering functions such as `label.new`, `box.new`, and `table.cell`. These constants control whether text is displayed in the default proportional font or a fixed-width monospace font.

## Quick Example

```python
from pynecore.lib import script, label, font, bar_index, high, close, ta

@script.indicator(title="Font Example", overlay=True)
def main():
    if ta.crossover(close, ta.sma(close, 20)):
        label.new(
            bar_index, high,
            text="Cross",
            text_font_family=font.family_monospace
        )
```

## Constants

| Constant               | Type           | Description                                                                                      |
|------------------------|----------------|--------------------------------------------------------------------------------------------------|
| `font.family_default`  | `FontFamilyEnum` | Default proportional font. Used with `label.new`, `label.set_text_font_family`, `box.new`, `box.set_text_font_family`, `table.cell`, and `table.cell_set_text_font_family`. |
| `font.family_monospace` | `FontFamilyEnum` | Fixed-width monospace font. Used with the same set of text-rendering functions as `family_default`. |