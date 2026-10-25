<!--
---
weight: 488
title: "volume_row"
description: "Accessors of one price row of a volume footprint"
icon: "table_rows"
date: "2026-10-25"
lastmod: "2026-10-25"
draft: false
toc: true
categories: ["Reference"]
tags: ["volume_row", "footprint", "order-flow"]
---
-->

# volume_row

The `volume_row` namespace reads one price row of a volume footprint: its price boundaries and the
buy and sell volume traded inside them. A row comes from [`footprint.poc()`](footprint.md#poc),
[`footprint.vah()`](footprint.md#vah) or [`footprint.val()`](footprint.md#val).

PyneCore has no footprint data: [`request.footprint()`](request.md#footprint) always returns
`na`, so every row derived from it is `na` too. Guard the footprint with `na()` before reading its
rows.

## Quick Example

```python
from pynecore.lib import script, request, footprint, volume_row, na, high, low, plot


@script.indicator(title="Value Area", overlay=True)
def main():
    fp = request.footprint(100, 70)

    if na(fp):
        # No order-flow data: fall back to the bar's range
        va_high: float = high
        va_low: float = low
    else:
        va_high = volume_row.up_price(footprint.vah(fp))
        va_low = volume_row.down_price(footprint.val(fp))

    plot(va_high, title="VA High")
    plot(va_low, title="VA Low")
```

## Functions

Every function takes the row object as its only argument.

| Parameter | Type        | Description                                                          |
|-----------|-------------|----------------------------------------------------------------------|
| `id`      | `VolumeRow` | A row from `footprint.poc()`, `footprint.vah()` or `footprint.val()` |

### up_price()

Upper price boundary of the row.

**Returns:** `float`

### down_price()

Lower price boundary of the row.

**Returns:** `float`

### buy_volume()

Buy volume traded in the row.

**Returns:** `float`

### sell_volume()

Sell volume traded in the row.

**Returns:** `float`

### delta()

Volume delta of the row: buy volume minus sell volume.

**Returns:** `float`

### total_volume()

Total volume traded in the row.

**Returns:** `float`

```python
poc = footprint.poc(fp)
poc_delta: float = volume_row.delta(poc)   # na when fp is na
```

## Missing data

- Called with an `na` row, every `volume_row.*` function returns `na` instead of raising. Since
  `request.footprint()` always returns `na`, so do `footprint.poc()`, `footprint.vah()` and
  `footprint.val()`, and every value read through this namespace is `na`.
- Calling a method on an actual `VolumeRow` object raises `NotImplementedError`; PyneCore never
  creates one.

## Compatibility

The `volume_row` functions and the `VolumeRow` type have the Pine v6 signatures, so scripts that
use them compile and run. There is no order-flow data source behind them; see
[footprint](footprint.md#missing-data).
