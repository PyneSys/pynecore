<!--
---
weight: 102
title: "Differences from Pine Script"
description: "Key differences between PyneCore and TradingView Pine Script"
icon: "compare"
date: "2025-03-31"
lastmod: "2025-03-31"
draft: false
toc: true
categories: ["Overview", "Comparisons"]
tags: ["pine-script", "differences", "comparison", "python", "syntax", "types"]
---
-->

# Differences from Pine Script

1st and most important, it is Python ;)

## Structure differences

### Pyne magic comment

Pyne codes must start with a magic doc-comment `@pyne` to be recognized as Pyne code.

```python
"""
@pyne
"""
```

Under the hood, this tells the AST transformers that this is a Pyne code and should be transformed.
Also you, the programmer can clearly see, this code will have extra Pine like features.

### Main function

In Pyne all runnable scripts must have a `main()` function. This function is the entry point of the script.

```python
def main():
    ...
    # Your code here
```

Imported modules don't need to have a `main()` function, but they can e.g. for testing purposes.

### Indicator, strategy or library

In Pine Script, you should have an `indicator()`, `strategy()` or `library()` function to define your script type.
In contrast, Pyne uses decorators to define the script type.

```python

from pynecore.lib import script


@script.indicator("My Indicator", shorttitle="MI")
def main():
    ...
```

Looks much more cleaner and pythonic, right?

### Inputs

In Pine Script, you define your inputs with the `input()` and `input.int()`, `input.X()` like functions.
In Pyne, you define your inputs as the main function arguments.

```python
from pynecore import Series
from pynecore.lib import script, input


@script.indicator("My Indicator", shorttitle="MI")
def main(
        src: Series[float] = input.source('close', title="Source"),
        length: int = input.int(5, title="Length"),
):
    ...
```

Still better, huh?

### Plotting

In Pine Script, you plot your data with the `plot()` function. You can use this in Pyne too, but Pyne
has a more pythonic way to "plot" your data:

```python
from pynecore import Series
from pynecore.lib import script, input, plot, ta


@script.indicator("My Indicator", shorttitle="MI")
def main(
        src: Series[float] = input.source('close', title="Source"),
        length: int = input.int(5, title="Length"),
):
    sma = ta.sma(src, length)
    ema = ta.ema(src, length)

    plot(sma, title="SMA")  # This works the same way as in Pine

    # This is the Pyne way:
    return {
        "EMA": ema,  # Title: value
    }
```

You can use both way according to your taste.

### Variable Scope and Lifecycle

Both Pine Script and Python (Pyne) use lexical scoping, but with important differences:

#### Scope Model Differences

Pine Script has stricter scoping rules than Python:

**Pine Script**:

- Uses block-level scoping
- Every code block (if statements, for loops, functions) has its own scope
- Variables defined within blocks are not accessible outside that block

```javascript
if (condition)
    blockVar = 10  // Only accessible within this if block
```

**Python/PyneCore**:

- Uses function-level scoping
- Only functions create new scopes
- Variables from blocks (if statements, for loops) remain accessible outside those blocks

```python
if condition:
    block_var = 10  # Still accessible outside the if block!

print(block_var)  # Works in Python, would fail in Pine Script
```

#### Variable Lifecycle Differences

**Pine Script**:

- Variables reinitialize on each bar by default
- Use the `var` keyword to make a variable persist between bars

```javascript
counter = 0  // Resets to 0 on each bar
var persistentCounter = 0  // Keeps its value across bars
```

**PyneCore**:

- Regular variables follow normal Python behavior
- Use `Persistent[T]` type annotation to indicate persistence between bars

```python
counter = 0  # Normal Python variable
persistentCounter: Persistent[int] = 0  # Persists across bars
```

#### Series Behavior

**Pine Script**:

- Every variable is a Series by default
- Can access historical values with index operator: `close[1]`

**PyneCore**:

- Series variables must be explicitly marked with `Series[T]` type annotation
- Historical values accessed the same way: `close[1]`

```python
regular_var = 5  # Not a Series
series_var: Series[float] = close  # Explicitly marked as Series
previous_value = series_var[1]  # Access previous bar's value
```

These differences require special attention when converting code from Pine Script to PyneCore, particularly regarding
block-level variables and explicit type annotations.

### Inline functions

In Pyne code you can use inline functions. That is very similar to Pine Script's functions, where you can access
variables from outer scopes. Though in Pyne you can also modify outer scope variables by using the `nonlocal` keyword.

```javascript
// Pine Script
x = 1
inner()
=>
x += 1  // This raises an error in Pine Script
x
```

```python
def main():
    x = 1

    def inner():
        nonlocal x
        x += 1  # This works in Pyne (Python)
        return x

    y = inner()

    return {
        "x": x,
        "y": y,
    }
```

Both `x` and `y` will be `2` in this case.

## Library differences from Pine

### Different module names

| Pine | PyneCore | Why?                                                 |
|------|----------|------------------------------------------------------|
| str  | string   | `str` is a frequently used builtin keyword in python |

### Different function names

| Pine         | PyneCore           | Why?                                  |
|--------------|--------------------|---------------------------------------|
| array.from() | array.from_items() | Because `from` is a keyword in Python |

## Type differences

### Basic types

| Pine Script type | PyneCore type |
|------------------|---------------|
| int              | int           |
| float            | float         |
| bool             | bool          |
| string           | str           |
| color            | Color         |

### Other builtin types

In Pine Script types are sometimes inconsistent. E.g. the `line` is also a type and also a package.
In Python it is not easily possible and it would be confusing. So in PyneCore we use capitalized names
and you need to import them explicitly:

| Pine Script type | PyneCore type | PyneCore import                               |
|------------------|---------------|-----------------------------------------------|
| line             | Line          | `from pynecore.types import Line`             |
| label            | Label         | `from pynecore.types import Label`            |
| box              | Box           | `from pynecore.types import Box`              |
| table            | Table         | `from pynecore.types import Table`            |
| linefill         | LineFill      | `from pynecore.types import LineFill`         |
| polyline         | Polyline      | `from pynecore.types import Polyline`         |
| chart.point      | ChartPoint    | `from pynecore.types.chart import ChartPoint` |

### na

In Pine Script every basic types can be `na` (not available). In Pyne there is an `NA` class that works
the same way. Though in Pine if you write the following code:

```javascript
int
i = na  // Note that you must specify the type here
float
f = na
```

`i` will be an `int(na)` and `f` will be a `float(na)`. So `na` will have the type of the variable.
In Python it is not possible, so in Pyne you can add a type to `na` like this:

```python
from pynecore.types.na import NA

i = NA(int)
f = NA(float)
```

or

```python
from pynecore.lib import na

i = na(int)
f = na(float)
```

Most of the time it is not necessary to specify the type, it is needed only when you want to use method
overloading. If you do not specify the type, it will be `NA[int]` by default.

### Array

Although we implemented all the array functions in array module, we decided to not implement Pine's `array` class and
its methods. Instead, we use Python's builtin `list`, because it is almost the same
and Pyne is just a Python framework after all.

This means if you see something like this in Pine:

```javascript
a = array.new_float(5)
a.set(0, 1.0)
a.push(2.0)
```

You can do the same in Pyne like this:

```python
from pynecore.types.na import NA
from pynecore.lib import array, na

# (Note, type hints are optional here)
a: list[float | NA[float]] = array.new_float(5)  # Functional Pine Script way, `a` is just a python list
b: list[float | NA[float]] = [na(float)] * 5  # Pythonic way, for the same result

array.set(a, 0, 1.0)  # Functional Pine Script way
b[0] = 1.0  # Pythonic way for the same

array.push(a, 2.0)  # Functional Pine Script way
b.append(2.0)  # Pythonic way for the same
```

### Map

The same goes for the `map` class. We use the builtin `dict` class which is almost the same. Though
you can use the `map` module too, like in Pine, but it is just a wrapper around the builtin `dict` class.

### Series

Pine Script uses dedicated series objects. Which is created on declaration. In Pyne we designed the system
to use the "primitive" type everywhere. If you mark a variable as a series, it will create a series
object, but every usage of them will be that basic type. This way the implementation could be much simpler.
We don't need to handle everything differently because it is a series or not.