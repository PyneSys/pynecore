<!--
---
weight: 492
title: "xloc"
description: "X-axis location constants for labels and lines"
icon: "swap_horiz"
date: "2026-04-05"
lastmod: "2026-04-05"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["xloc", "library", "reference"]
---
-->

# xloc

Constants that control the x-axis location mode for `label.new()` and `line.new()`. Determines whether x coordinates are interpreted as bar indices or timestamps.

## Constants

| Constant          | Description                                                        |
|-------------------|--------------------------------------------------------------------|
| `xloc.bar_index`  | X coordinates are bar indices (default). Use with `bar_index`.     |
| `xloc.bar_time`   | X coordinates are UNIX timestamps in milliseconds. Use with `time`. |

## Example

```python
from pynecore.lib import label, bar_index, time, close, xloc, script

@script.indicator(title="XLoc Demo", overlay=True)
def main():
    label.new(bar_index, close, "By index", xloc=xloc.bar_index)
    label.new(time, close, "By time", xloc=xloc.bar_time)
```

## Compatibility

Both constants are fully supported.
