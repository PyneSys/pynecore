<!--
---
weight: 103
title: "Core Concepts"
description: "Fundamental concepts and mechanisms of PyneCore"
icon: "psychology_alt"
date: "2025-03-31"
lastmod: "2025-03-31"
draft: false
toc: true
categories: ["Overview", "Fundamentals"]
tags: ["ast", "series", "persistent", "function-isolation", "na-values", "core-concepts"]
---
-->

# Core Concepts of PyneCore

Understanding the following core concepts is essential to effectively use PyneCore and leverage its unique approach to bringing Pine Script's power into the Python world.

## 1. AST Transformation

PyneCore's magic lies in its use of Python's **Abstract Syntax Tree (AST) transformation**. Instead of relying on complex object-oriented wrappers, PyneCore modifies your standard Python code *before* it runs.

Here's how it works:
1. Your Python script (marked with `@pyne`) is parsed into an AST.
2. A chain of AST transformers modify the tree, including:
   - `PersistentTransformer`: Converts `Persistent[type]` annotated variables to specially named globals, this is the fastest possible way to maintain state between bars.
   - `SeriesTransformer`: Converts `Series[type]` annotated variables to special Series operations, on assignment it stores the value in a circular buffer, on subscription it retrieves the value from the circular buffer. Every other operation uses the pure value. Makes it really fast.
   - `FunctionIsolationTransformer`: Ensures functions have their own isolated contexts
   - Lot of other optimizations, like library normalization, etc.
3. The transformed AST is then compiled and executed by the standard Python interpreter.
4. The compiled code is cached into the Python bytecode cache for future runs, so you can enjoy the performance of PyneCore without the overhead (very fast anyway) of the first run. The AST transformations are run only after modifications of the source code.

**Why is this important?**
- You write clean, idiomatic Python code.
- The complexity of mimicking Pine Script's execution model is handled automatically in the background.
- You get Pine Script like semantics combined with Python's full power and ecosystem.

More details in the [AST Transformations](../advanced/ast-transformations.md) page.

## 2. Persistent Variables

`Persistent` corresponds to TradingView's `var` keyword. These variables maintain their state
across different bar executions. This is crucial for calculations that accumulate values or track
conditions over time.

- **Declaration**: Use the `Persistent[type]` type hint, but use a simple value for initialization.
- **Behavior**: The variable retains its value from the end of the previous bar's execution to the start of the current bar's execution. When a script executes multiple times on the same bar (e.g. with `calc_on_order_fills`), `Persistent` variables are **rolled back** to the previous bar's committed state before each re-execution.

```python
from pynecore import Persistent

# Initialize a persistent counter (runs only on the first bar)
bar_count: Persistent[int] = 0

# Increment the counter on every bar
bar_count += 1

# The value of bar_count persists across bars
print(f"Current bar index (persistent): {bar_count}")
```

The implementation of Persistent variables in PyneCore is especially clever:

1. The `PersistentTransformer` identifies variables with `Persistent[type]` annotations at compile time.
2. It transforms these into specially named global variables (e.g., `__persistent_main_bar_count__`).
3. Every function that uses persistent variables gets global statements for those variables.

This approach allows PyneCore to maintain state between bar executions without relying on complex object-oriented designs. Makes it really fast.

### IBPersistent Variables

`IBPersistent` (IntraBar Persistent) corresponds to TradingView's `varip` keyword. Like
`Persistent`, it maintains its value across bars. The difference: when a script executes
**multiple times on the same bar**, `Persistent` variables are rolled back to the bar's opening
state before each re-execution, while `IBPersistent` variables are not — they retain their
accumulated value.

On historical bars, a script normally executes only once per bar, so there is no rollback and
the two types behave identically. Multiple executions per bar occur in two cases:

- **Strategies with `calc_on_order_fills=True`**: re-execution after each order fill
- **Live / real-time bars**: execution on every new tick

```python
from pynecore.types import Persistent, IBPersistent

var_count: Persistent[int] = 0       # rolled back on re-execution
varip_count: IBPersistent[int] = 0   # NOT rolled back

var_count += 1    # always increments by 1 per bar
varip_count += 1  # increments by 1 per execution (including re-executions)
```

## 3. Series Variables

Just like in Pine Script, `Series` variables in PyneCore represent time series data, holding a history of values for each bar. Also the syntax is the same.

- **Declaration**: Use the `Series[type]` type hint, but use a simple value for initialization.
- **Accessing History**: Use array-like indexing (e.g., `close[1]` for the previous close).

```python
from pynecore import Series
from pynecore.lib import close

# Declare a Series variable holding closing prices
current_close: Series[float] = close

# Access the closing price from the previous bar
previous_close = close[1]

# Access the closing price from 5 bars ago
close_5_bars_ago = close[5]
```

Behind the scenes, PyneCore implements Series variables as dynamic circular buffers (`SeriesImpl` class) that efficiently manage historical values:
1. Each Series has a configurable `max_bars_back` parameter (default: 500) that determines how many historical values are stored.
2. The Series includes two key operations:
   - `add()`: Adds a new value for the current bar
   - `set()`: Updates the value for the current bar
3. Accessing values with array syntax (`series[n]`) retrieves the value from n bars ago.
4. Series support slicing (`series[1:5]`) to access ranges of historical values.

The AST transformers automatically convert normal variable assignments to the appropriate Series operations. So you never need to worry about it, it works like in Pine Script. Series variables acts like normal values, just you can index them to get the value from the past.

## 4. Function Isolation

In Pine Script, each call to a function maintains its own independent state for internal series or persistent variables. PyneCore replicates this behavior through **function isolation**.

When you call a function within your Pyne script:
- It gets its own unique context.
- Any `Persistent` or `Series` variables defined *inside* that function maintain their state *independently* for that specific call context.

```python
from pynecore import Persistent, Series
from pynecore.lib import close

def calculate_running_sum(input_series: Series[float]) -> Series[float]:
    # This 'running_sum' is persistent *within each call context* of this function
    running_sum: Persistent[float] = 0.0
    running_sum += input_series
    return running_sum

# Example usage in main()
def main():
    sum1 = calculate_running_sum(close)  # Instance 1 of running_sum
    sum2 = calculate_running_sum(close * 2) # Instance 2 of running_sum, independent of sum1
    # sum1 and sum2 will have different persistent states
```

The function isolation system uses a sophisticated runtime mechanism:

1. The `isolate_function` utility creates separate instances of functions for each call context.
2. Each function instance gets its own copy of persistent variables and fresh Series instances.
3. A function cache (`_function_cache`) keeps track of these isolated function instances.
4. Call IDs and parent scopes are used to uniquely identify each call context.
5. This ensures that when a function is called from different contexts, it maintains separate state for each caller.

This approach allows for true isolation while preserving the functional programming style that Pine Script developers are accustomed to.

More details in the [Function Isolation](../advanced/function-isolation.md) page.

## 5. NA (Not Available) System

PyneCore fully implements Pine Script's `na` (Not Available) concept. `na` represents missing or undefined data points, often occurring at the beginning of calculations that require a certain lookback period (like moving averages).

- **Propagation**: Operations involving `na` always result in `na`.
- **Checking**: Use the `na()` function (from `pynecore.lib`) to check if a value is `na`.
- **Handling**: Use functions like `nz()` (replace `na` with zero or another value) or `fixnan()` (fill `na` with the last non-`na` value) for graceful handling.

```python
from pynecore import Series
from pynecore.lib import ta, close, na, nz

def main():
    sma10: Series[float] = ta.sma(close, 10)

    # sma10 will be 'na' for the first 9 bars
    if na(sma10):
        print("SMA10 is not yet available.")
    else:
        print(f"SMA10: {sma10}")

    # Use nz() to replace na with 0.0 for calculations
    safe_sma10 = nz(sma10, 0.0)
    print(f"Safe SMA10 (na replaced with 0): {safe_sma10}")
```

The `NA` class in PyneCore is a sophisticated implementation that:

1. Maintains type information using Python's generics system (`NA[T]`).
2. Overrides all operators and methods to return `NA` when operations involve `NA` values.
3. Always evaluates to `False` in boolean contexts (just like in Pine Script).
4. Caches NA instances by type to optimize memory usage.
5. Enables built-in functions like `na()`, `nz()`, and `fixnan()` to work just like in Pine Script.

This system allows graceful handling of missing data in calculations, eliminating common errors like `IndexError` when accessing data that isn't available yet.

## How These Concepts Work Together

These core concepts don't operate in isolation—they work together to create a cohesive system:

1. **AST Transformation** is the foundation that enables everything else, modifying Python code to behave like Pine Script.
2. **Series Variables** track historical data points across bars.
3. **Persistent Variables** maintain state from bar to bar.
4. **Function Isolation** ensures each function call has its own independent context.
5. **NA Handling** provides graceful handling of missing or undefined data.

Through the combination of these concepts, PyneCore achieves a unique blend of Python's flexibility and Pine Script's trading-specific semantics, without forcing you to learn a new language or complex object-oriented patterns.

---

These core concepts, powered by AST transformation, allow PyneCore to provide a Pine Script-like experience within the familiar and powerful Python environment.