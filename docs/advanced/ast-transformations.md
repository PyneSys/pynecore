<!--
---
weight: 1001
title: "AST Transformation"
description: "How PyneCore uses AST transformation to implement Pine Script behavior"
icon: "code"
date: "2025-03-31"
lastmod: "2026-03-18"
draft: false
toc: true
categories: ["Advanced", "Technical Implementation"]
tags: ["ast", "python", "transformations", "compiler", "internals"]
---
-->

# AST Transformation

## Import Hook System

The system's entry point is the import hook, which transforms Python files marked with the `@pyne` magic comment:

```python
# Import hook through importlib meta_path system
sys.meta_path.insert(0, PyneImportHook())
```

The `PyneLoader` class performs code transformation in multiple steps, applying the AST transformation chain.

## Transformation Chain

PyneCore applies several key transformations to Python code to make it behave like Pine Script:

1. **Import Lifter** - Moves function-level imports to module level
2. **Import Normalizer** - Standardizes import statements
3. **Security Transformer** - Rewrites `request.security()` calls into signal/write/read/wait pattern
   (see [request.security()](./request-security.md))
4. **PersistentSeries Transformer** - Manages the hybrid PersistentSeries type
5. **Library Series Transformer** - Prepares library Series variables
6. **Module Property Transformer** - Handles module properties
7. **Closure Arguments Transformer** - Converts closure variables to function arguments
8. **Function Isolation Transformer** - Ensures separate state for each function call
9. **Unused Series Detector** - Removes unnecessary Series annotations for performance
10. **Series Transformer** - Handles Series variables
11. **Persistent Transformer** - Manages persistent variables (with automatic Kahan summation for `+=`)
12. **Input Transformer** - Processes input parameters
13. **Safe Convert Transformer** - Converts float()/int() calls to safe versions
14. **Safe Division Transformer** - Protects against division by zero

This order ensures that dependencies between transformations are properly handled. For example, PersistentSeries transformation must happen before both Persistent and Series transformations.

Each transformation step modifies the Python AST to implement Pine Script behavior while maintaining Python syntax and readability.

## Detailed Transformation Process

### Import Lifter

The Import Lifter moves function-level imports to module level.

**Original code:**
```python
def main():
    from pynecore.lib.ta import sma
    result = sma(close, 14)
```

**Transformed code:**
```python
from pynecore.lib.ta import sma

def main():
    result = sma(close, 14)
```

Key aspects:
- Lifts all pynecore.lib related imports to module level
- Ensures imports are accessible throughout the module
- Prevents duplicate imports

### Import Normalizer

The Import Normalizer transforms all PyneCore imports to use a consistent format.

**Original code:**
```python
from pynecore.lib.ta import sma, ema
from pynecore.lib import plot, close

def main():
    plot(close)
    plot(sma(close, 14))
    plot(ema(close, 14))
```

**Transformed code:**
```python
from pynecore import lib
import pynecore.lib.ta

def main():
    lib.plot(lib.close)
    lib.plot(lib.ta.sma(lib.close, 14))
    lib.plot(lib.ta.ema(lib.close, 14))
```

Key aspects:
- Converts all lib-related imports to 'from pynecore import lib'
- Transforms variable references to use fully qualified names (lib.ta.sma)
- Maintains compatibility with wildcard imports
- Ensures consistent import style across the codebase

This is very important to make lib level properties work like `close`, `open`, `high`, `low`, `volume`, etc.
If you would use this kind of import:
```python
a = close
```
That would not work, because the value would never be updated in the next bar.
However, after using the import normalizer, it will work:
```python
a = lib.close
```
Because the module level variable changed, and we access through the lib module object.

### PersistentSeries Transformer

The PersistentSeries transformer converts the combined PersistentSeries type into separate Persistent and Series declarations.

**Original code:**
```python
ps: PersistentSeries[float] = 1
ps += 1
```

**Transformed code:**
```python
p: Persistent[float] = 1
s: Series[float] = p
s += 1
```

Key aspects:
- Splits PersistentSeries declarations into two separate declarations
- Must be applied before both Persistent and Series transformers

This makes easier to declare variables are both persistent and series.

### Library Series Transformer

The Library Series transformer prepares library Series variables (like close, open, high, etc.) for proper handling by the Series transformer.

**Original code:**
```python
def main():
    def f():
        return lib.high[1]
    h1 = lib.high[1]
```

**Transformed code:**
```python
def main():
    __lib·high: Series = lib.high

    def f():
        __lib·high: Series = lib.high
        return __lib·high[1]
    h1 = __lib·high[1]
```

Key aspects:
- Creates local Series variables for library Series in each scope
- Uses Unicode middle dot (·) as separator to prevent name collisions
- Maintains proper function context across nested functions
- Prepares variables for Series transformer processing

**Collision Prevention**: The transformer uses `__lib·` prefix with Unicode middle dot separators to prevent naming conflicts. For example:
- `mylib.bar.foo` becomes `__lib·mylib·bar·foo`  
- `mylib.bar_foo` becomes `__lib·mylib·bar_foo`

This ensures that hierarchical module names cannot collide with underscore-separated names.

If you import a variable from a library, it does not know if it is a series or not. But if you use indexing (subscription) on it, it should initialize it as a series. This is needed, because the AST transformer does not know anything about the other files just the one it is currently transforming.

### Module Property Transformer

The Module Property transformer handles attributes that should be called as functions based on configuration.

**Original code:**
```python
bar_index = lib.bar_index
time = lib.time
```

**Transformed code:**
```python
bar_index = lib.bar_index()
time = lib.time
```

Key aspects:
- Uses configuration to determine which attributes are properties
- Automatically adds parentheses for property calls
- Preserves normal attributes as is
- Handles dynamic cases with runtime checks

### Closure Arguments Transformer

The Closure Arguments transformer converts closure variables in inner functions to explicit function arguments, enabling proper function isolation.

**Original code:**
```python
@lib.script.indicator("Test")
def main():
    length = 14
    multiplier = 2.0
    
    def calculate(offset=0):
        return lib.ta.sma(lib.close, length) * multiplier + offset
    
    return calculate() + calculate(10)
```

**Transformed code:**
```python
@lib.script.indicator("Test")
def main():
    length = 14
    multiplier = 2.0
    
    def calculate(length: int, multiplier: float, offset=0):
        return lib.ta.sma(lib.close, length) * multiplier + offset
    
    return calculate(length, multiplier) + calculate(length, multiplier, 10)
```

Key aspects:
- Adds closure variables as function parameters at the beginning of parameter list
- Preserves type annotations from original variable declarations
- Updates all function calls to pass closure variables as arguments
- Only processes functions inside @lib.script.indicator or @lib.script.strategy decorated main functions
- Maintains proper scope isolation for nested functions
- Prepares functions for the Function Isolation transformer

### Function Isolation Transformer

The Function Isolation transformer ensures each function call gets its own isolated scope by wrapping functions with the isolate_function decorator.

**Original code:**
```python
def compute_avg(source):
    return (source + source[1]) / 2

result = compute_avg(close)
```

**Transformed code:**
```python
from pynecore.core.function_isolation import isolate_function
__scope_id__ = "8af7c21e_example.py"

def compute_avg(source):
    global __scope_id__
    return (source + source[1]) / 2

result = isolate_function(compute_avg, "main|compute_avg|0", __scope_id__)(close)
```

Key aspects:
- Wraps each function call with isolate_function
- Generates a unique call ID for each invocation
- Maintains scope hierarchy information
- Adds scope ID handling to each function
- Excludes standard library and non-transformable functions

### Unused Series Detector

The Unused Series Detector optimizes performance by removing Series annotations from variables that are never indexed with the subscript operator.

**Original code:**
```python
def main():
    # This variable is never indexed - can be optimized
    s: Series[float] = close
    
    def f(source: Series[float], m = 1.0):
        # This parameter IS indexed - must keep Series annotation
        return source * m + s[1]
    
    r = f(s, 2.0)
    plot(s)
```

**Transformed code:**
```python
def main():
    # Series annotation removed since s is never indexed in main scope
    s: float = close
    
    def f(source: float, m = 1.0):
        # Series annotation removed since source is never indexed in f scope
        # Note: s[1] refers to the closure variable, not the parameter
        return source * m + s[1]
    
    r = f(s, 2.0)
    plot(s)
```

Key aspects:
- Uses scope-aware analysis to track variable usage independently in each function scope
- Distinguishes between variables with the same name in different scopes (e.g., closure vs parameter)
- Only removes Series annotations from variables that are never used with subscript syntax `[index]`
- Runs before SeriesTransformer to prevent unnecessary SeriesImpl creation
- Significantly improves performance by avoiding Series overhead for simple variables
- Preserves type annotations for variables that are actually indexed

**Performance Impact**: This optimization can dramatically reduce memory usage and improve execution speed by eliminating unnecessary Series object creation for variables that are only used for simple arithmetic operations.

### Series Transformer

The Series transformer converts Series annotated variables in Python code into a global SeriesImpl instance with add() and set() operations.

**Original code:**
```python
s: Series[float] = close
s += 1
previous = s[1]
```

**Transformed code:**
```python
from pynecore.core.series import SeriesImpl

__series_main·s__ = SeriesImpl()
__series_function_vars__ = {'main': ['__series_main·s__']}

def main():
    s = __series_main·s__.add(close)
    s = __series_main·s__.set(s + 1)
    previous = __series_main·s__[1]
```

Key aspects:
- Creates a global SeriesImpl instance for each Series variable
- Converts assignments to add() and set() operations
- Redirects indexing operations to the global instance
- Maintains a registry of all Series variables per function scope
- Uses Unicode middle dot (·) as scope separator to prevent conflicts with underscores in function names

### Persistent Transformer

The Persistent transformer converts variables with Persistent type annotation to global variables that maintain their values across function calls.

**Original code:**
```python
p: Persistent[float] = 0
p += 1
```

**Transformed code:**
```python
__persistent_main·p__ = 0
__persistent_function_vars__ = {'main': ['__persistent_main·p__']}

def main():
    global __persistent_main·p__
    __persistent_main·p__ += 1
```

Key aspects:
- Creates a global variable for each Persistent variable
- Adds global declarations in functions
- Handles initialization for non-literal values
- Maintains a registry of all Persistent variables by scope
- Uses `·` (middle dot, U+00B7) as scope separator in variable names to avoid conflicts with underscores in function names

This is the fastest possible way to implement persistent variables.

**Kahan Summation**: The `+=` operator on Persistent float variables is automatically transformed into Kahan summation by the AST transformer. This eliminates accumulated floating-point errors in running sums. To bypass Kahan summation, use `x = x + val` instead of `x += val`.

**Important Note**: The Persistent and Series transformers use the Unicode character `·` (middle dot, U+00B7) as the internal scope separator. This prevents conflicts when function names contain underscores. For example:
- Function `f_f` in scope `main` creates variables like `__persistent_main·f_f·a__`
- This ensures proper isolation between functions with similar names

Avoid using the `·` character in function or variable names to prevent conflicts with the internal scoping system.

### Input Transformer

The Input transformer processes input parameters and adds necessary ID information.

**Original code:**
```python
@script.indicator
def main(source=lib.input.source(lib.close, "Source")):
    result = source * 2
```

**Transformed code:**
```python
@script.indicator
def main(source=lib.input.source(lib.close, "Source", _id="source")):
    source = getattr(lib, source, lib.na)
    result = source * 2
```

Key aspects:
- Adds _id parameter to input calls
- Adds getattr for source inputs at the start of functions
- Enables proper input parameter resolution
- Handles source inputs specially

### Safe Convert Transformer

The Safe Convert transformer replaces float() and int() calls with safe versions that handle NA values properly.

**Original code:**
```python
value = float(some_value)
number = int(another_value)
```

**Transformed code:**
```python
from pynecore.core import safe_convert

value = safe_convert.safe_float(some_value)
number = safe_convert.safe_int(another_value)
```

Key aspects:
- Converts float() and int() to safe_float() and safe_int()
- Returns NA(float) or NA(int) when TypeError occurs (e.g., from NA inputs)
- Maintains Pine Script semantics for type conversions
- Only adds import if conversion functions are actually used

### Safe Division Transformer

The Safe Division transformer converts division operations to safe alternatives that handle division by zero like Pine Script.

**Original code:**
```python
result = (close - open_) / (high - low)
ratio = value / divisor
constant = 1 / 2  # Literal division remains unchanged
```

**Transformed code:**
```python
from pynecore.core import safe_convert

result = safe_convert.safe_div(close - open_, high - low)
ratio = safe_convert.safe_div(value, divisor)
constant = 1 / 2  # Literal divisions are not transformed
```

Key aspects:
- Converts division operations (/) to safe_div() calls
- Returns NA(float) instead of raising ZeroDivisionError
- Literal divisions (e.g., 1/2) remain unchanged for performance
- Matches Pine Script behavior where division by zero returns NA
- Only adds import if division operations are actually transformed


## Example of Complete Transformation

Let's see a full example of how a simple Pyne code is transformed:

**Original Pyne Code:**
```python
"""
@pyne
"""
from pynecore import Series, Persistent
from pynecore.lib.ta import sma
from pynecore.lib import close, open_, high, low, plot

def main():
    # Persistent counter
    count: Persistent[int] = 0
    count += 1

    # Moving average calculation
    ma: Series[float] = sma(close, 14)
    
    # Safe division that could cause division by zero
    range_ratio = (close - open_) / (high - low)

    # Plot results
    plot(ma, "MA", color=lib.color.blue)
    plot(count, "Count", color=lib.color.red)
    plot(range_ratio, "Range Ratio", color=lib.color.green)
```

**Transformed Code:**
```python
"""
@pyne
"""
from pynecore import lib
import pynecore.lib.ta
from pynecore.core.series import SeriesImpl
from pynecore.core.function_isolation import isolate_function
from pynecore.core import safe_convert

# Global variables and scope ID
__scope_id__ = "8af7c21e_example.py"
__persistent_main·count__ = 0
__series_main·ma__ = SeriesImpl()
__series_main·range_ratio__ = SeriesImpl()

# Function and variable registries
__persistent_function_vars__ = {'main': ['__persistent_main·count__']}
__series_function_vars__ = {'main': ['__series_main·ma__', '__series_main·range_ratio__']}

def main():
    global __scope_id__
    global __persistent_main·count__
    
    # Library Series declarations
    __lib·close: Series = lib.close
    __lib·open_: Series = lib.open_
    __lib·high: Series = lib.high
    __lib·low: Series = lib.low

    # Persistent counter
    __persistent_main·count__ += 1

    # Moving average calculation
    ma = __series_main·ma__.add(isolate_function(lib.ta.sma, "main|lib.ta.sma|0", __scope_id__)(__lib·close, 14))
    
    # Safe division that could cause division by zero
    range_ratio = __series_main·range_ratio__.add(safe_convert.safe_div(__lib·close - __lib·open_, __lib·high - __lib·low))

    # Plot results
    lib.plot(ma, "MA", color=lib.color.blue)
    lib.plot(__persistent_main·count__, "Count", color=lib.color.red)
    lib.plot(range_ratio, "Range Ratio", color=lib.color.green)
```

This example demonstrates how the different transformers work together to convert a simple Pyne script into equivalent Python code that provides Pine Script-like behavior through PyneCore's runtime system.