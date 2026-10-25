<!--
---
weight: 487
title: "footprint"
description: "Volume footprint accessors — buy/sell volume, delta, POC, VAH and VAL of a bar"
icon: "stacked_bar_chart"
date: "2026-10-25"
lastmod: "2026-10-25"
draft: false
toc: true
categories: ["Reference"]
tags: ["footprint", "volume_row", "request", "order-flow"]
---
-->

# footprint

The `footprint` namespace reads a bar's volume footprint: the buy and sell volume split by
aggressor side, the delta between them, and the Point of Control, Value Area High and Value Area
Low rows. A footprint object comes from [`request.footprint()`](request.md#footprint), and the
three row functions return [`volume_row`](volume_row.md) objects.

PyneCore has no footprint data. A footprint is built from tick-level bid/ask order flow, and
PyneCore works from OHLCV bars only, so `request.footprint()` always returns `na`. Guard every
footprint read with `na()` and give the script a fallback path, as in the example below.

## Quick Example

```python
from pynecore.lib import script, request, footprint, volume_row, na, close, open, volume, plot


@script.indicator(title="Footprint Delta")
def main():
    fp = request.footprint(100, 70)

    if na(fp):
        # No order-flow data: approximate the delta from the bar's direction
        bar_delta: float = volume if close >= open else -volume
        poc_price: float = (close + open) / 2
    else:
        bar_delta = footprint.delta(fp)
        poc = footprint.poc(fp)
        poc_price = (volume_row.up_price(poc) + volume_row.down_price(poc)) / 2

    plot(bar_delta, title="Delta")
    plot(poc_price, title="POC")
```

## Getting a footprint

`request.footprint(ticks_per_row, va_percent=70, imbalance_percent=300)` requests the footprint of
the current bar. In PyneCore it returns `na` of type `Footprint` on every bar. See
[request](request.md#footprint).

## Functions

Every function takes the footprint object as its only argument.

| Parameter | Type        | Description                                   |
|-----------|-------------|-----------------------------------------------|
| `id`      | `Footprint` | A footprint object from `request.footprint()` |

### buy_volume()

Total buy volume of the bar.

**Returns:** `float`

### sell_volume()

Total sell volume of the bar.

**Returns:** `float`

### delta()

Volume delta of the bar: buy volume minus sell volume.

**Returns:** `float`

### total_volume()

Total volume of the bar: buy volume plus sell volume.

**Returns:** `float`

### poc()

The Point of Control row: the price row with the highest volume.

**Returns:** [`VolumeRow`](volume_row.md)

### vah()

The Value Area High row: the upper boundary row of the value area.

**Returns:** [`VolumeRow`](volume_row.md)

### val()

The Value Area Low row: the lower boundary row of the value area.

**Returns:** [`VolumeRow`](volume_row.md)

```python
fp = request.footprint(100, 70)
buy: float = footprint.buy_volume(fp) if not na(fp) else 0.0
```

## Missing data

- `request.footprint()` returns `na` on every bar, so `na(fp)` is always true and the guarded
  branch of a script is the one that runs.
- Every `footprint.*` function called with an `na` footprint returns `na` instead of raising, and
  so does a `volume_row.*` function called with the `na` row that `poc()`, `vah()` or `val()`
  returns. An unguarded read therefore gives `na`, which propagates through arithmetic.
- The `Footprint` and `VolumeRow` types exist so that scripts using them load and run. They hold
  no data, and calling one of their methods on an actual object raises `NotImplementedError`.

## Compatibility

The `footprint` functions and the `Footprint` type have the Pine v6 signatures, so footprint
scripts compile and run. There is no order-flow data source: `request.footprint()` returns `na`,
and scripts produce output only through their `na()` fallback paths.
