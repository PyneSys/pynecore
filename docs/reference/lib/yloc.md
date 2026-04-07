<!--
---
weight: 493
title: "yloc"
description: "Y-axis location constants for labels"
icon: "swap_vert"
date: "2026-04-05"
lastmod: "2026-04-05"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["yloc", "library", "reference"]
---
-->

# yloc

Constants that control the y-axis location mode for `label.new()`. Determines whether a label is placed at a specific price or relative to the bar.

## Constants

| Constant          | Description                                              |
|-------------------|----------------------------------------------------------|
| `yloc.price`      | Label is placed at the y coordinate price (default).     |
| `yloc.abovebar`   | Label is placed above the bar's high, y value is ignored. |
| `yloc.belowbar`   | Label is placed below the bar's low, y value is ignored.  |

## Example

```python
from pynecore.lib import label, bar_index, high, yloc, script

@script.indicator(title="YLoc Demo", overlay=True)
def main():
    label.new(bar_index, high, "Above", yloc=yloc.abovebar)
```

## Compatibility

All constants are fully supported.
