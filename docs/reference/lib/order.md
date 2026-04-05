<!--
---
weight: 478
title: "order"
description: "Sort order constants (ascending, descending)"
icon: "sort"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["order", "library", "reference"]
---
-->

# order

The `order` namespace provides sort order constants used with array sorting functions. Import `order` from `pynecore.lib` to specify whether to sort ascending or descending.

## Quick Example

```python
from pynecore.lib import script, ta, close, order
from pynecore.lib import array
from pynecore.types import Persistent

@script.indicator(title="Sorted Closes Demo", overlay=False)
def main():
    closes: Persistent[list] = array.new_float(0)
    array.push(closes, close)
    if array.size(closes) > 10:
        array.shift(closes)

    sorted_asc: list = array.copy(closes)
    array.sort(sorted_asc, order.ascending)
```

---

## Constants

| Constant            | Description                                                                |
|---------------------|----------------------------------------------------------------------------|
| `order.ascending`   | Sort from smallest to largest. Used as the `sort_order` argument in array sorting functions. |
| `order.descending`  | Sort from largest to smallest. Used as the `sort_order` argument in array sorting functions. |

---

## Compatibility

All entries in this namespace are fully supported.