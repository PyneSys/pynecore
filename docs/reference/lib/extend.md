<!--
---
weight: 474
title: "extend"
description: "Line/box extend direction constants"
icon: "open_in_full"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["extend", "library", "reference"]
---
-->

# extend

The `extend` namespace provides constants that control how lines are extended beyond their start and end points. These constants are used with `line.new()` and `line.set_extend()` to specify the extension direction.

## Quick Example

```python
from pynecore.lib import script, line, extend, close, bar_index, ta

@script.indicator(title="Trend Line with Extension", overlay=True)
def main():
    if ta.crossover(close, ta.sma(close, 20)):
        ln = line.new(bar_index, close, bar_index + 1, close, extend=extend.right)
```

## Constants

| Constant       | Description                                              |
|----------------|----------------------------------------------------------|
| `extend.both`  | Extend the line in both directions (left and right)      |
| `extend.left`  | Extend the line to the left beyond the start point       |
| `extend.none`  | Do not extend the line (default behavior)                |
| `extend.right` | Extend the line to the right beyond the end point        |