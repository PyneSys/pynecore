<!--
---
weight: 410
title: "Type System"
description: "PyneCore type system — primitives, annotations, collections, and special types"
icon: "category"
date: "2026-03-28"
lastmod: "2026-04-05"
draft: false
toc: true
categories: ["Reference", "Types"]
tags: ["types", "series", "persistent", "ibpersistent", "na", "int", "float", "bool", "string", "color", "array", "matrix", "map", "udt", "enum"]
---
-->

# Types

PyneCore uses Python's native type system extended with a set of special type annotations. These
annotations control how variables behave across bar executions. Everything else is standard Python.

## Primitive Types

### int

Integer values. Python's `int`.

```python
length: int = 14
```

### float

Floating-point values. Python's `float`.

```python
price: float = 100.50
```

### bool

Boolean values. Python's `bool`.

```python
is_bull: bool = close > open
```

### string

Text values. Python's `str`.

```python
title: str = "My Script"
```

### color

Color values with RGBA components. Implemented as `pynecore.types.color.Color`.

```python
from pynecore.lib import script, close, color, plot

@script.indicator("Color Example", overlay=True)
def main():
    my_color = color.rgb(255, 0, 0, 50)  # Red with 50% transparency
    bull_color = color.green if close > close[1] else color.red
    plot(close, "Close", color=bull_color)
```

See [Color](lib/color.md) for details.

### na

Represents "not available" — the absence of a value. Implemented as `pynecore.types.na.NA`.

```python
from pynecore.types import Series
from pynecore.lib import script, close, ta, na, nz

@script.indicator("NA Example")
def main():
    sma10: Series[float] = ta.sma(close, 10)

    # sma10 is na for the first 9 bars
    if na(sma10):
        safe_value = 0.0
    else:
        safe_value = sma10

    # or use nz() for the same thing
    safe_value = nz(sma10, 0.0)
```

Operations involving `na` propagate — `na + 1` results in `na`. In boolean contexts, `na`
evaluates to `False`.

## Type Annotations

These annotations are recognized by the AST transformer at import time and control how variables
behave during bar-by-bar execution.

### Series[T]

A time series variable that stores a history of values, one per bar. Supports indexing to access
past values.

```python
from pynecore.types import Series
from pynecore.lib import close

current_close: Series[float] = close
previous_close = close[1]    # one bar ago
older_close = close[5]       # five bars ago
```

The assignment runs every bar, pushing the new value onto a circular buffer. Historical values
are accessible via `[n]` indexing where `n` is the number of bars back. The buffer size defaults
to 500 and is configurable via `max_bars_back`.

PyneCore also supports slicing (`close[1:5]`), which returns a read-only view of historical
values. This is a PyneCore extension — not available in Pine Script.

When used without indexing, a Series value behaves like a plain value — arithmetic, comparisons,
and function calls all work on the current bar's value transparently.

Corresponds to TradingView's `series` type.

### Persistent[T]

A variable that **persists its value across bars**. Without this annotation, variables are
re-initialized on every bar execution.

```python
from pynecore.types import Persistent

bar_count: Persistent[int] = 0
bar_count += 1  # increments every bar, value carries over
```

The initialization (`= 0`) runs only on the first bar. On subsequent bars, the variable retains
its value from the end of the previous bar.

Corresponds to TradingView's `var` keyword.

When a script executes **multiple times on the same bar** (via
[calc_on_order_fills](../advanced/bar-magnifier.md#calc_on_order_fills) or on live ticks),
`Persistent` variables are **rolled back** to the previous bar's committed state before each
re-execution.

### IBPersistent[T]

Like `Persistent`, persists across bars — but **eliminates the rollback** that normally occurs
before each successive execution on the same bar.

```python
from pynecore.types import Persistent, IBPersistent

var_count: Persistent[int] = 0       # rolled back on re-execution
varip_count: IBPersistent[int] = 0   # NOT rolled back

var_count += 1    # always increments by 1 per bar
varip_count += 1  # increments by 1 per execution (including re-executions)
```

Corresponds to TradingView's `varip` keyword.

On historical bars, a script typically executes only once per bar, so there is no rollback and
`IBPersistent` behaves identically to `Persistent`. The distinction appears when a script runs
**multiple times on the same bar**:

- **Strategies with `calc_on_order_fills=True`**: the script re-executes after each order fill.
  `Persistent` variables are rolled back; `IBPersistent` variables retain their accumulated value.
- **Real-time / live bars**: the script runs on every new tick. `Persistent` variables are rolled
  back to the bar's opening state; `IBPersistent` variables keep their value across ticks.

See [Bar Magnifier — calc_on_order_fills](../advanced/bar-magnifier.md#calc_on_order_fills)
for the execution model details.

## Collection Types

### array

Dynamic arrays. Python lists with Pine Script-compatible wrapper methods.

```python
from pynecore.types import Persistent
from pynecore.lib import script, close, array

@script.indicator("Array Example")
def main():
    prices: Persistent[list] = array.new_float(0)
    array.push(prices, close)

    if array.size(prices) > 20:
        array.shift(prices)  # keep last 20

    avg = array.avg(prices)
```

### matrix

Two-dimensional matrices. Implemented as `pynecore.types.matrix.Matrix`.

```python
from pynecore.types import Persistent
from pynecore.lib import script, matrix

@script.indicator("Matrix Example")
def main():
    m: Persistent[matrix] = matrix.new(float, 3, 3, 0.0)
    matrix.set(m, 0, 0, 1.0)
    val = matrix.get(m, 0, 0)
```

### map

Key-value dictionaries with Pine Script-compatible methods.

```python
from pynecore.types import Persistent
from pynecore.lib import script, close, map

@script.indicator("Map Example")
def main():
    prices: Persistent[map] = map.new(str, float)
    map.put(prices, "last_close", close)
    val = map.get(prices, "last_close")
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

A point on the chart defined by bar index/time and price.

```python
from pynecore.lib import script, close, high, bar_index, time, chart, label

@script.indicator("Chart Point Example", overlay=True)
def main():
    p1 = chart.point.from_index(bar_index, close)
    p2 = chart.point.from_time(time, high)
    p3 = chart.point.now(close)

    label.new(p1, "Here")
```

## Special Types

### User-Defined Types (@udt)

Custom structured types using the `@udt` decorator.

```python
from pynecore.core.pine_udt import udt

@udt
class OrderInfo:
    id: str
    price: float
    qty: int = 0

    def get_value(self) -> float:
        return self.price * self.qty
```

Corresponds to TradingView's `type` keyword. The `@udt` decorator provides Pine
Script-compatible semantics (NA-aware field defaults, copy behavior, etc.).

### Enumerations

Enumerations use Python's `StrEnum`. The value of each field is its display title.

```python
from enum import StrEnum as Enum

class Direction(Enum):
    Up = "Up"
    Down = "Down"
    Sideways = "Sideways"
```

Corresponds to TradingView's `enum` keyword.

### Source

The `source` type represents plottable values selectable via `input.source()`. In practice it
behaves as `float`. Implemented as `pynecore.types.source.Source`.
