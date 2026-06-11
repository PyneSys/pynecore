<!--
---
weight: 1001
title: "AST Transformation"
description: "How PyneCore uses AST transformation to implement Pine Script behavior"
icon: "code"
date: "2025-03-31"
lastmod: "2026-06-10"
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

### Marker Recognition Rules

The `@pyne` marker is recognized strictly to avoid accidentally transforming ordinary library modules that merely mention the token in prose:

1. The module's **first statement** must be a string-literal expression (a module docstring).
2. After `lstrip()`, the docstring's content must **start with `@pyne`**.
3. `@pyne` must be followed by whitespace or the end of the docstring (so `@pynex` does not match, and a docstring like `"""Some description.\n@pyne\n"""` does **not** trigger transformation — the marker must come first, not somewhere inside).

A cheap regex prefilter (`@pyne(\s|$)`) skips full AST parsing for files that obviously cannot match. Files that pass the prefilter still go through the strict AST check above before any transformer runs.

## Transformation Chain

PyneCore applies several key transformations to Python code to make it behave like Pine Script, in this order:

1. **Import Lifter** - Moves function-level imports to module level
2. **TYPE_CHECKING Stripper** - Removes `if TYPE_CHECKING:` blocks (IDE-only hints)
3. **Import Normalizer** - Standardizes import statements
4. **Security Transformer** - Rewrites `request.security()` calls into signal/write/read/wait pattern
   (see [request.security() Internals](./request-security-internals.md))
5. **PersistentSeries Transformer** - Splits the hybrid PersistentSeries type
6. **Library Series Transformer** - Prepares library Series variables
7. **Module Property Transformer** - Handles module properties
8. **Closure Arguments Transformer** - Converts closure variables to function arguments
9. **Unused Series Detector** - Removes unnecessary Series annotations for performance
10. **Series Transformer** - Handles Series variables
11. **Persistent Transformer** - Manages persistent variables (with automatic Kahan summation for `+=`)
12. **Function Isolation Transformer** - Ensures separate state for each function call
13. **Input Transformer** - Processes input parameters
14. **Safe Convert Transformer** - Converts float()/int() calls to safe versions
15. **Safe Division Transformer** - Protects against division by zero

This order ensures that dependencies between transformations are properly handled. For example, PersistentSeries transformation must happen before both Persistent and Series transformations, and Function Isolation must run after them because it routes calls based on the state slots they allocated.

Each transformation step modifies the Python AST to implement Pine Script behavior while maintaining Python syntax and readability.

## The Slot Layout

The state-related transformers (Series, Persistent, Function Isolation) share one **module layout**: a table that assigns a **slot index** to every piece of per-instance state — persistent variables, series buffers, Kahan compensation values, and the state of isolated call sites. At the end of the chain this table is emitted into the module as a plain dict constant, and every state-carrying function gets:

- a hidden first parameter (`__state__`) that receives its **state vector** — a plain Python list whose slots are addressed with literal int indexes, and
- a `__pyne_layout__` attribute describing how to build such a vector (initial values, series slots, `varip` slots, child call sites).

```python
__pyne_slot_layout__ = {'main': {'init': (0,), 'series': (), 'varip': (), 'children': (), 'names': ('p',)}}

def main(__state__):
    __state__[0] += 1
main.__pyne_layout__ = __pyne_slot_layout__['main']
```

The runtime side of this scheme (who creates the state vectors and when) is described on the [Function Isolation](./function-isolation.md) page.

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

### TYPE_CHECKING Stripper

Removes `if TYPE_CHECKING:` blocks and the `TYPE_CHECKING` import itself. These blocks carry IDE-only type hints (casts, re-annotations) that have no runtime role, so stripping them keeps the transformed module free of dead code.

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

The Library Series transformer prepares library Series variables (like close, open, high, etc.) for proper handling by the Series transformer: every scope that indexes a library value gets a local Series anchor for it.

**Original code:**
```python
def main():
    a = lib.close[1]

    def nested():
        return lib.high[1]
    result = nested()
    print(a, result)
```

**Transformed code (after the Series transformer has assigned the slots):**
```python
__pyne_slot_layout__ = {'main': {'init': (None, None), 'series': ((0, None), (1, None)), 'varip': (), 'children': (), 'names': ('__lib·close', '__lib·high')}}

def main(__state·main__):
    __lib·close = __state·main__[0].add(lib.close)
    __lib·high = __state·main__[1].add(lib.high)
    a = __state·main__[0][1]

    def nested():
        return __state·main__[1][1]
    result = nested()
    print(a, result)
main.__pyne_layout__ = __pyne_slot_layout__['main']
```

Key aspects:
- Creates local Series variables for library Series in each scope that needs them
- Uses Unicode middle dot (·) as separator to prevent name collisions
- The buffers anchor in the outermost function that uses them; nested functions reach them through the parent's state vector
- Prepares variables for Series transformer processing

**Collision Prevention**: The transformer uses `__lib·` prefix with Unicode middle dot separators to prevent naming conflicts. For example:
- `mylib.bar.foo` becomes `__lib·mylib·bar·foo`
- `mylib.bar_foo` becomes `__lib·mylib·bar_foo`

This ensures that hierarchical module names cannot collide with underscore-separated names.

If you import a variable from a library, it does not know if it is a series or not. But if you use indexing (subscription) on it, it should initialize it as a series. This is needed, because the AST transformer does not know anything about the other files just the one it is currently transforming.

### Module Property Transformer

The Module Property transformer handles attributes that should be called as functions based on
the generated `module_properties.json` registry.

**Original code:**
```python
t = lib.time
bar_index = lib.bar_index
plot(close, "Close")
d = dayofweek
```

**Transformed code:**
```python
t = lib.time()
bar_index = lib.bar_index
lib.plot.plot(lib.close, "Close")
d = lib.dayofweek.dayofweek()
```

Key aspects:
- The registry (`module_properties.json`, generated from the lib source by
  `scripts/module_property_collector.py`) determines which attributes are properties
- Automatically adds parentheses for property calls; explicit calls are left untouched
- Normal attributes (variables, constants, function references) stay plain attribute reads
- Calls and promoted bare reads of function-and-namespace modules (`plot`, `hline`, `alert`,
  `dayofweek`, `strategy.opentrades`, `strategy.closedtrades`) are routed to the module's
  self-named function
- Unknown names on known `pynecore.lib` modules raise at transform time — this catches typos
  and a stale registry early (the test suite keeps the committed registry current)
- Unknown module paths (user `lib.*` workdir libraries) and `_`-prefixed names are plain reads

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

The Series transformer converts Series annotated variables into operations on a `SeriesImpl` instance (a circular buffer) living in a slot of the function's state vector.

**Original code:**
```python
from pynecore import Series
from pynecore.lib import close

def main():
    s: Series[float] = close
    s += 1
    previous = s[1]
    print(previous)
```

**Transformed code:**
```python
from pynecore import lib
__pyne_slot_layout__ = {'main': {'init': (None,), 'series': ((0, None),), 'varip': (), 'children': (), 'names': ('s',)}}

def main(__state__):
    s = __state__[0].add(lib.close)
    s = __state__[0].set(s + 1)
    previous = __state__[0][1]
    print(previous)
main.__pyne_layout__ = __pyne_slot_layout__['main']
```

Key aspects:
- Allocates a series slot in the function's state vector for each Series variable; the runtime puts a fresh `SeriesImpl` into these slots when an instance is created
- Converts the declaration to an `add()` (push the bar's value) and assignments to `set()` operations
- Redirects indexing operations to the slot (`s[1]` becomes `__state__[0][1]`)
- Statement-position `lib.max_bars_back(s, n)` calls become assignments to the slot's `max_bars_back` attribute
- Each function instance gets its own buffers, because each instance has its own state vector

### Persistent Transformer

The Persistent transformer converts variables with Persistent type annotation to slots of the function's state vector, so they maintain their values across function calls.

**Original code:**
```python
p: Persistent[float] = 0
p += 1
```

**Transformed code:**
```python
__pyne_slot_layout__ = {'main': {'init': (0,), 'series': (), 'varip': (), 'children': (), 'names': ('p',)}}

def main(__state__):
    __state__[0] += 1
main.__pyne_layout__ = __pyne_slot_layout__['main']
```

Key aspects:
- Allocates a slot with the initial value in the layout's `init` tuple; literal initializers are baked in, non-literal initializers get a lazy init-flag companion slot that triggers the assignment on the instance's first call
- Rewrites every read and write of the variable to the slot (`__state__[0]`)
- `IBPersistent` (varip) variables get their slot listed in the layout's `varip` tuple, which excludes them from the `var` rollback on intra-bar re-execution
- Slot reads/writes are plain list indexing with literal indexes — the fastest state access Python offers

**Kahan Summation**: The `+=` operator on Persistent float variables is automatically transformed into Kahan summation. This eliminates accumulated floating-point errors in running sums. The compensation value lives in a companion slot (`p·kahan`), which also follows the variable's `varip` flag so a rollback can never desynchronize the pair. To bypass Kahan summation, use `x = x + val` instead of `x += val`.

```python
__pyne_slot_layout__ = {'main': {'init': (0.0, 0.0), 'series': (), 'varip': (), 'children': (), 'names': ('cumulative', 'cumulative·kahan')}}

def main(__state__):
    __kahan_corrected__ = some_value - __state__[1]
    __kahan_new_sum__ = __state__[0] + __kahan_corrected__
    __state__[1] = __kahan_new_sum__ - __state__[0] - __kahan_corrected__
    __state__[0] = __kahan_new_sum__
```

**Important Note**: The state-related transformers use the Unicode character `·` (middle dot, U+00B7) as the internal scope separator in slot names and call-site identifiers (e.g. `main·t·0`). This prevents conflicts when function names contain underscores. Avoid using the `·` character in function or variable names to prevent conflicts with the internal scoping system.

### Function Isolation Transformer

The Function Isolation transformer ensures each function call site gets its own isolated state. The state of a callee instance lives in a dedicated **child slot of the caller's state vector**, assigned at transform time.

**Original code:**
```python
from pynecore.lib import ta, close

def main():
    print(ta.sma(close, 12))
```

**Transformed code:**
```python
from pynecore import lib
import pynecore.lib.ta
from pynecore.core.instance_state import __resolve_slot__
__pyne_slot_layout__ = {'main': {'init': (None,), 'series': (), 'varip': (), 'children': ((0, 'main·lib.ta.sma·0', False),), 'names': ('main·lib.ta.sma·0',)}}

def main(__state__):
    print(lib.ta.sma(__st__ if (__st__ := __state__[0]) is not None else __resolve_slot__(__state__, 0, lib.ta.sma), lib.close, 12))
main.__pyne_layout__ = __pyne_slot_layout__['main']
```

Key aspects:
- The callee receives its own state vector as hidden first argument; after the first call it is a single list-index read
- Callees the transformer can prove stateful get this fast path; callees it cannot resolve at transform time go through a uniform binding path; stateless callees are called directly; builtins, types and module properties are left untouched
- Call sites in loops get one child state per iteration

The full routing logic, the loop emission and the runtime side are described on the [Function Isolation](./function-isolation.md) page.

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
from pynecore.lib import script, ta, close, open, high, low, plot, color


@script.indicator("Example")
def main():
    # Persistent counter
    count: Persistent[int] = 0
    count += 1

    # Moving average calculation
    ma: Series[float] = ta.sma(close, 14)

    # Safe division that could cause division by zero
    range_ratio = (close - open) / (high - low)

    # Plot results
    plot(ma, "MA", color=color.blue)
    plot(count, "Count", color=color.red)
    plot(range_ratio, "Range Ratio", color=color.green)
```

**Transformed Code:**
```python
"""
@pyne
"""
from pynecore import lib
import pynecore.lib.color
import pynecore.lib.ta
from pynecore.core.instance_state import __bind_any__, __resolve_slot__
from pynecore.core import safe_convert
from pynecore.core.instance_state import __attach_layout__
__pyne_slot_layout__ = {'main': {'init': (0, None), 'series': (), 'varip': (), 'children': ((1, 'main·lib.ta.sma·0', False),), 'names': ('count', 'main·lib.ta.sma·0')}}

@lib.script.indicator('Example')
@__attach_layout__(__pyne_slot_layout__['main'])
def main(__state__):
    __state__[0] += 1
    ma: float = lib.ta.sma(__st__ if (__st__ := __state__[1]) is not None else __resolve_slot__(__state__, 1, lib.ta.sma), lib.close, 14)
    range_ratio = safe_convert.safe_div(lib.close - lib.open, lib.high - lib.low)
    lib.plot.plot(ma, 'MA', color=lib.color.blue)
    lib.plot.plot(__state__[0], 'Count', color=lib.color.red)
    lib.plot.plot(range_ratio, 'Range Ratio', color=lib.color.green)
```

Worth noting in the output:

- `count` became slot 0 of main's state vector (`init` starts with its initial value `0`).
- `ma` lost its Series annotation (never indexed — Unused Series Detector), so no series slot was allocated for it.
- The `ta.sma` call got child slot 1: the first call creates the callee's state vector there, subsequent calls reuse it.
- Since `main` is decorated, the layout attach uses the `@__attach_layout__` decorator form (innermost position, so it tags the raw function before other decorators wrap it).
- The `plot(...)` calls were routed to the module's self-named function (`lib.plot.plot`) by the Module Property transformer and stay direct calls — `plot` is a function-and-namespace module.

This example demonstrates how the different transformers work together to convert a simple Pyne code into equivalent Python code that provides Pine Script-like behavior through PyneCore's runtime system.

## Debugging the Transformation

To see the transformed code of a script, use:

```bash
pyne debug ast my_script.py
```

or the `PYNE_AST_DEBUG` family of environment variables — see [Debugging](../debugging.md#inspecting-the-transformed-code) for details.
