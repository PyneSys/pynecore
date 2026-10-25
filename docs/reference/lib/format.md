<!--
---
weight: 476
title: "format"
description: "Number format constants for labels and tooltips"
icon: "format_list_numbered"
date: "2026-03-28"
lastmod: "2026-10-25"
draft: false
toc: true
categories: ["Reference"]
tags: ["format"]
---
-->

# format

The `format` namespace provides named constants that control how script output values are displayed in the indicator pane legend and how numbers are rendered by `str.tostring()`. These constants are passed to the `format` parameter of `script.indicator()` or to `str.tostring()`.

## Quick Example

```python
from pynecore.lib import script, close, string as pstr, format

@script.indicator(title="Price Display", format=format.price, overlay=False)
def main():
    price_str: str = pstr.tostring(close, format.mintick)
```

---

## Constants

| Constant          | Type     | Description                                                                                                  |
|-------------------|----------|--------------------------------------------------------------------------------------------------------------|
| `format.inherit`  | `Format` | Inherits the display format from the parent series. Use with `script.indicator()`.                           |
| `format.mintick`  | `Format` | Rounds to whole ticks and prints the tick's decimals in `str.tostring()` (`1.1` → `1.10000`).                |
| `format.percent`  | `Format` | Formats output values as a percentage; `str.tostring()` prints two decimals and `%` (`0.5` → `0.50%`).       |
| `format.price`    | `Format` | Formats output values as prices. Use with `script.indicator()`.                                              |
| `format.volume`   | `Format` | Formats output values as volume (e.g., `5183` → `5.183K`). Use with `script.indicator()`.                   |

---

## Compatibility

All five constants are fully supported. This namespace contains no functions or variables.
