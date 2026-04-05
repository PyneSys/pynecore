<!--
---
weight: 476
title: "format"
description: "Number format constants for labels and tooltips"
icon: "format_list_numbered"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["format", "library", "reference"]
---
-->

# format

The `format` namespace provides named constants that control how script output values are displayed in the indicator pane legend and how numbers are rendered by `str.tostring()`. These constants are passed to the `format` parameter of `script.indicator()` or to `str.tostring()`.

## Quick Example

```python
from pynecore.lib import script, close, str, format

@script.indicator(title="Price Display", format=format.price, overlay=False)
def main():
    price_str: str = str.tostring(close, format.mintick)
```

---

## Constants

| Constant          | Type     | Description                                                                                                  |
|-------------------|----------|--------------------------------------------------------------------------------------------------------------|
| `format.inherit`  | `Format` | Inherits the display format from the parent series. Use with `script.indicator()`.                           |
| `format.mintick`  | `Format` | Rounds a number to the nearest mintick value when passed to `str.tostring()`.                                |
| `format.percent`  | `Format` | Formats output values as a percentage (appends `%`). Use with `script.indicator()`.                          |
| `format.price`    | `Format` | Formats output values as prices. Use with `script.indicator()`.                                              |
| `format.volume`   | `Format` | Formats output values as volume (e.g., `5183` → `5.183K`). Use with `script.indicator()`.                   |

---

## Compatibility

All five constants are fully supported. This namespace contains no functions or variables.