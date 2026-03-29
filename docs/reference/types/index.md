<!--
---
weight: 411
title: "Type System"
description: "PyneCore type system — primitives, collections, drawing types, and Series"
icon: "category"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Types"]
tags: ["types", "series", "na", "int", "float", "bool", "string", "color", "array", "matrix", "map"]
---
-->

# Types

PyneCore uses Python's native type system instead of Pine Script's qualifier-based types.
This results in simpler, more Pythonic code while maintaining full compatibility.

## Fundamental Difference from Pine Script

Pine Script has a complex type system with four qualifiers: `const`, `input`, `simple`, and
`series`. Each value carries both a base type and a qualifier, and the compiler enforces
strict qualifier compatibility rules.

**PyneCore eliminates this complexity.** A value is simply a value. Series behavior is
created automatically when a value is used in a time-series context (e.g., history reference
with `[n]`). There is no need to annotate variables with `series float` or `simple int`.

| Pine Script            | PyneCore    | Notes                              |
|------------------------|-------------|------------------------------------|
| `const int`            | `int`       | Compile-time constant              |
| `input float`          | `float`     | From `input.*()`, resolved at compile time |
| `simple string`        | `str`       | Does not change after bar 0        |
| `series float`         | `float`     | Automatically becomes Series when needed |
| `series bool`          | `bool`      | Same — Series is implicit          |

## Primitive Types

### int

Integer values. Maps directly to Python's `int`.

```python
length: int = 14
```

### float

Floating-point values. Maps directly to Python's `float`.

```python
price: float = 100.50
```

### bool

Boolean values (`true`/`false`). Maps to Python's `bool`.

```python
is_bull: bool = close > open
```

### string

Text values. Maps to Python's `str`.

```python
title: str = "My Script"
```

### color

Color values with RGBA components. Implemented as `pynecore.types.color.Color`.

```python
my_color = color.rgb(255, 0, 0, 50)  # Red with 50% transparency
```

See [Color](color.md) for details.

### na

Represents "not available" — the absence of a value. Implemented as `pynecore.types.na.NA`.

In PyneCore, `na` serves dual purpose:
- As a value: `x = na` assigns an NA value
- As a check: `na(x)` returns `true` if x is NA

```python
x = na          # x has no value
if na(x):       # check if x is NA
    x = 0.0
y = nz(x, 0)   # replace NA with 0
```

## Collection Types

### array

Dynamic arrays. Implemented as Python lists with Pine Script-compatible wrapper methods.

```python
arr = array.new_float(10, 0.0)  # array of 10 floats, initialized to 0.0
array.push(arr, close)
val = array.get(arr, 0)
```

### matrix

Two-dimensional matrices. Implemented as `pynecore.types.matrix.Matrix`.

```python
m = matrix.new<float>(3, 3, 0.0)
matrix.set(m, 0, 0, 1.0)
```

### map

Key-value dictionaries. Implemented as Python dicts with Pine Script-compatible methods.

```python
m = map.new<string, float>()
map.put(m, "AAPL", 150.0)
val = map.get(m, "AAPL")
```

## Drawing Types

Drawing types create visual elements on the chart. Each type has a corresponding namespace
with constructor and manipulation functions.

| Type       | Constructor      | Description                    |
|-----------|------------------|--------------------------------|
| `label`   | `label.new()`    | Text labels on the chart       |
| `line`    | `line.new()`     | Lines between two points       |
| `box`     | `box.new()`      | Rectangular boxes              |
| `table`   | `table.new()`    | Data tables                    |
| `polyline`| `polyline.new()` | Multi-segment lines            |
| `linefill`| `linefill.new()` | Filled area between two lines  |

All drawing types support the `.all` property to access all active instances.

### chart.point

A point on the chart defined by bar index/time and price. Implemented as
`pynecore.types.chart.ChartPoint`.

```python
p = chart.point.from_index(bar_index, close)
p = chart.point.from_time(time, high)
p = chart.point.now(close)
```

## Special Types

### Series

Series is not a type you declare — it is an implicit behavior. When a value needs historical
access (e.g., `close[1]`), PyneCore automatically manages the history buffer.

In Pine Script, you write `series float`. In PyneCore, you write `float` — the Series
behavior is added by the runtime when the value participates in history references or
indicator calculations.

### Source

The `source` type represents plottable values that can be selected in `input.source()`.
In PyneCore, this is `pynecore.types.source.Source`, but in practice it behaves as `float`.

## User-Defined Types (UDT)

Custom types declared with the `type` keyword compile to Python dataclasses.

```
type OrderInfo
    string id
    float price
    int qty = 0
```

Compiles to:

```python
@dataclass
class OrderInfo:
    id: str
    price: float
    qty: int = 0
```

## Enums

Enumerations declared with the `enum` keyword compile to Python enum classes.

```
enum Signal
    Buy
    Sell
    Hold
```
