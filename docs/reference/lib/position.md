<!--
---
weight: 479
title: "position"
description: "Table/label position constants"
icon: "dashboard"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["position", "library", "reference"]
---
-->

# position

The `position` namespace provides constants for anchoring tables to fixed screen locations. These values are passed to `table.new()` and `table.cell()` to control where a table appears on the chart.

## Quick Example

```python
from pynecore.lib import script, table, position, close, bar_index, barstate

@script.indicator(title="Price Display", overlay=True)
def main():
    t = table.new(position.top_right, 1, 2)
    if barstate.islast:
        table.cell(t, 0, 0, "Close")
        table.cell(t, 0, 1, str(close))
```

## Constants

All constants are of type `Position` and represent a fixed anchor point on the chart canvas.

| Constant                   | Description                              |
|----------------------------|------------------------------------------|
| `position.top_left`        | Upper-left corner of the chart           |
| `position.top_center`      | Top edge, horizontally centered          |
| `position.top_right`       | Upper-right corner of the chart          |
| `position.middle_left`     | Left edge, vertically centered           |
| `position.middle_center`   | Center of the chart                      |
| `position.middle_right`    | Right edge, vertically centered          |
| `position.bottom_left`     | Lower-left corner of the chart           |
| `position.bottom_center`   | Bottom edge, horizontally centered       |
| `position.bottom_right`    | Lower-right corner of the chart          |