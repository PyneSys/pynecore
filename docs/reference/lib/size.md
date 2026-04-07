<!--
---
weight: 482
title: "size"
description: "Label and shape size constants"
icon: "format_size"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["size", "library", "reference"]
---
-->

# size

The `size` namespace provides constants for specifying the visual size of chart objects such as labels, shapes, boxes, and table cells. These constants are passed to the `size` parameter of drawing functions like `label.new()`, `box.new()`, `plotchar()`, and `plotshape()`.

## Quick Example

```python
from pynecore.lib import close, ta, label, size, bar_index, script

@script.indicator(title="Size Demo", overlay=True)
def main():
    if ta.crossover(close, ta.sma(close, 20)):
        label.new(bar_index, close, "Cross", size=size.normal)
```

## Constants

| Constant      | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `size.auto`   | Automatically adjusts size based on context. Supported by `plotchar`, `plotshape`, `label.new`, and `box.new`. |
| `size.huge`   | Largest fixed size. Supported by `plotchar`, `plotshape`, `label.new`, `box.new`, and `table.cell`. |
| `size.large`  | Large fixed size. Supported by `plotchar`, `plotshape`, `label.new`, `box.new`, and `table.cell`. |
| `size.normal` | Standard size. Supported by `plotchar`, `plotshape`, `label.new`, `box.new`, and `table.cell`. |
| `size.small`  | Small fixed size. Supported by `plotchar`, `plotshape`, `label.new`, `box.new`, and `table.cell`. |
| `size.tiny`   | Smallest fixed size. Supported by `plotchar`, `plotshape`, `label.new`, `box.new`, and `table.cell`. |