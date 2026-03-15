<!--
---
weight: 1002
title: "Function Isolation"
description: "How function isolation works in PyneCore and why it's essential for Pine Script compatibility"
icon: "privacy_tip"
date: "2025-03-31"
lastmod: "2025-03-31"
draft: false
toc: true
categories: ["Advanced"]
tags: ["function-isolation", "ast", "persistence", "series"]
---
-->

# Function Isolation

Function isolation is a crucial feature of PyneCore that enables the precise replication of Pine Script's unique execution model. This document explains how function isolation works, why it's necessary, and how it's implemented in PyneCore.

## What is Function Isolation?

In Pine Script, every function call creates an isolated environment with its own copies of variables. This is different from traditional programming languages where functions share access to the same variables. Function isolation means:

1. Every function call gets its own copies of *Series* and *Persistent* variables
2. State changes made within a function remain isolated to that specific function call
3. Multiple calls to the same function maintain separate states

This behavior is essential for correctly implementing indicators and strategies in a bar-by-bar execution model.

## Why Function Isolation is Necessary

Function isolation solves several critical requirements for Pine Script compatibility:

### 1. Per-Call Variable State

Consider the following Pine Script example:

```pine
//@version=6
indicator("Function Call State Example")

myFunc() =>
    var count = 0
    count := count + 1
    count

plot(myFunc())
plot(myFunc())
```

In Pine Script, this would plot two different lines because each call to `myFunc()` has its own isolated state for the `count` variable. Without function isolation, both calls would share the same variable, resulting in identical values.

### 2. Correct Historical Behavior

For indicators like moving averages, each function call needs to maintain its own buffer of historical values. Multiple moving averages with different periods need to maintain separate data buffers even though they use the same underlying function.

## Implementation in PyneCore

PyneCore implements function isolation through two main components:

### 1. AST Transformation (transformers/function_isolation.py)

The `FunctionIsolationTransformer` class is an Abstract Syntax Tree (AST) transformer that:

1. Scans Python code at import time
2. Identifies function calls that need isolation
3. Wraps each applicable function call with `isolate_function()`
4. Maintains a hierarchy of scope IDs to track the call chain

Here's a simplified view of the transformation process:

```python
# Original Python code
result = some_function(argument)

# Transformed code
result = isolate_function(some_function, 'unique_call_id', __scope_id__)(argument)
```

Key aspects of the transformer:

- Creates unique call IDs that include the full function call path and position
- Skips standard library functions and other non-transformable functions
- Supports nested function calls and maintains proper scope hierarchy
- Adds global `__scope_id__` declaration to functions that use isolation

### 2. Runtime Isolation (core/function_isolation.py)

The `isolate_function()` function performs the actual isolation at runtime:

1. Creates or retrieves a function instance based on the call ID
2. Copies persistent and Series variables for this specific function call
3. Maintains the scope chain through `__scope_id__` variable
4. Caches function instances for performance

```python
def isolate_function(func, call_id, parent_scope):
    """Create a new isolated function instance with its own variable state"""
    ...
    # Generate the full call ID with parent scope
    full_call_id = f"{parent_scope}->{call_id}#{counter}"
    ...
    # Create new function with isolated globals
    isolated_function = FunctionType(
        func.__code__,
        new_globals_with_copies,
        func.__name__,
        func.__defaults__,
        func.__closure__
    )
    ...
    return isolated_function
```

The isolation process handles:
- Copying of Persistent variables
- Creating new instances of Series variables
- Maintaining the scope chain for nested calls
- Tracking call counters to distinguish between multiple calls at the same bar

## Call ID Generation and Scope Chain

Each function call gets a unique identifier constructed from:

1. The parent scope ID (`__scope_id__` from the calling context)
2. The function's path (e.g., `lib.ta.sma`)
3. A counter to distinguish between multiple calls to the same function
4. The complete call hierarchy for nested function calls

For example, a call ID might look like:
`module_scope->main->lib.ta.sma#1`

This ID system ensures that:
- Each function call gets a unique ID
- The ID reflects the full call hierarchy
- Multiple calls to the same function get distinct IDs
- Function calls in different bars are properly tracked

## Example: Transformed Code

Here's a simple example showing the transformation process:

**Original code:**
```python
"""
@pyne
"""
from pynecore import Series

def main():
    def calculate():
        a: Series[float] = 1
        a += 1
        return a[1]

    result1 = calculate()
    result2 = calculate()
```

**Transformed code:**
```python
"""
@pyne
"""
from pynecore.core.series import SeriesImpl
from pynecore.core.function_isolation import isolate_function
__series_main_calculate_a__ = SeriesImpl()
__series_function_vars__ = {'main.calculate': ['__series_main_calculate_a__']}
__scope_id__ = 'module_hash_filename'

def main():
    global __scope_id__

    def calculate():
        a = __series_main_calculate_a__.add(1)
        a = __series_main_calculate_a__.set(a + 1)
        return __series_main_calculate_a__[1]

    result1 = isolate_function(calculate, 'main|calculate|0', __scope_id__)()
    result2 = isolate_function(calculate, 'main|calculate|1', __scope_id__)()
```

## Performance Considerations

The function isolation mechanism is designed for performance:

1. Function instances are cached to avoid recreating them on every bar
2. Only Series and persistent variables are isolated, other variables are shared
3. Standard library and non-transformable functions are excluded from isolation
4. Optimizations for specific function patterns and classes

## Summary

Function isolation is a fundamental PyneCore feature that enables accurate Pine Script compatibility. By creating isolated function instances with their own variable state, PyneCore ensures that indicators, strategies, and other scripts behave correctly in the bar-by-bar execution model. The combination of AST transformation and runtime instance creation provides an elegant solution to this complex problem.
