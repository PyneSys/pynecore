<!--
---
weight: 403
title: "Keywords"
description: "Pine Script keywords and their PyneCore equivalents"
icon: "key"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Language"]
tags: ["keywords", "var", "varip", "if", "for", "while", "switch", "type", "enum"]
---
-->

# Keywords

PyneCore supports all Pine Script v6 keywords. The PyneComp compiler translates them to their
Python equivalents.

## Variable Declaration

### var

Declares a variable that persists its value across bars. Without `var`, variables are
re-initialized on every bar.

```
var float mySum = 0.0
mySum += close
```

In PyneCore, `var` compiles to a persistent variable managed by the AST transformer. The value
is stored and restored between bar executions automatically.

### varip

Similar to `var`, but also persists between real-time bar updates (intrabar ticks). In
historical bars, it behaves identically to `var`.

```
varip int tickCount = 0
tickCount += 1
```

PyneCore supports `varip` in the same way as `var` for offline/historical execution. The
distinction only matters in live real-time mode.

## Control Flow

### if

Conditional execution. Supports `else if` and `else` branches.

```
if close > open
    label.new(bar_index, high, "Bull")
else if close < open
    label.new(bar_index, low, "Bear")
```

Compiles directly to Python's `if`/`elif`/`else`.

### switch

Pattern matching on values or conditions.

```
switch syminfo.type
    "stock"  => strategy.entry("Long", strategy.long)
    "forex"  => strategy.entry("Short", strategy.short)
    =>          runtime.error("Unsupported")
```

Compiles to Python's `match`/`case` (Python 3.10+) or `if`/`elif` chains.

### for

Count-controlled loop.

```
for i = 0 to 9
    sum += array.get(arr, i)
```

Compiles to Python's `for i in range(...)`.

### for...in

Iterates through array or matrix elements.

```
for [i, val] in arr
    sum += val
```

Compiles to Python's `for i, val in enumerate(arr)`.

### while

Condition-controlled loop.

```
while i > 0
    sum += i
    i -= 1
```

Compiles directly to Python's `while`.

The loop must eventually terminate. Use `break` to exit early.

## Modules and Types

### import

Loads an external Pine Script library.

```
import TradingView/ta/7
```

PyneComp resolves library imports at compile time.

### export

Makes functions or type definitions available to other scripts when used in a library.

```
export myFunction(float x) => x * 2
```

### type

Declares a user-defined type (UDT).

```
type OrderInfo
    string id
    float price
    int qty
```

Compiles to a Python dataclass.

### enum

Declares an enumeration with predefined constants.

```
enum Direction
    Up
    Down
    Sideways
```

Compiles to a Python enum class.

### method

Enables dot-notation function calls on a type.

```
method getRange(OrderInfo this) =>
    this.price * this.qty
```

Compiles to a method on the corresponding Python class.

## Logical Operators (Keywords)

### and / or / not

See [Operators](operators.md#logical-operators).
