<!--
---
weight: 1002
title: "Function Isolation"
description: "How function isolation works in PyneCore and why it's essential for Pine Script compatibility"
icon: "privacy_tip"
date: "2025-03-31"
lastmod: "2026-06-10"
draft: false
toc: true
categories: ["Advanced"]
tags: ["function-isolation", "ast", "persistence", "series", "state-vector"]
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

PyneCore implements function isolation with **per-instance state vectors**: every function instance's state (persistent variables, series buffers, the state of its own callees) lives in a plain Python list whose slots are assigned at transform time. Two components cooperate:

### 1. State Vectors and Layouts

The state-related AST transformers (see [AST Transformation](./ast-transformations.md)) give every state-carrying function:

- a hidden first parameter (`__state__`) — its state vector, and
- a `__pyne_layout__` attribute that describes how to build one.

A layout is a plain dict with these keys:

- `init` — template value for every slot; instantiation is essentially `list(init)`
- `series` — `(slot, max_bars_back)` pairs; instantiation puts a fresh `SeriesImpl` circular buffer into these slots
- `varip` — slot indexes excluded from the `var` rollback on intra-bar re-execution
- `children` — `(slot, call_id, in_loop)` triples describing the function's isolated call sites
- `names` — per-slot debug names (used only by debug tooling)

The key idea is the **parent-slot scheme**: the state of a callee instance occupies a dedicated *child slot of the caller's state vector*, assigned at transform time. All live state therefore forms a **tree** hanging off a few root vectors. There is no global instance cache and no key lookups — reaching an instance's state is a list index away, and dropping a parent releases its entire subtree through normal garbage collection.

### 2. Root Vectors (core/instance_state.py)

The runner creates **root state vectors** for the entry points it drives directly — the script's `main()`, imported library mains and `request.security()` processes — via `create_root(key, layout)`. Everything else lives in the tree below them. The same module also provides:

- `reset()` — clears the child slots of all roots (drops every function instance) between runs
- `RootVarSnapshot` — snapshot/restore of the roots' `var` slots for the `calc_on_order_fills` rollback; `varip` slots and child instances are excluded
- `explain_state(func, state)` — renders a state vector as a readable name -> value dict for debugging

## Call Routing

At transform time every call site is classified, because the best possible emission depends on what the callee is:

- **fast** — the callee is provably state-carrying (a same-module def, or an imported `@pyne` function with a layout). Emission: child slot with the `__resolve_slot__` cold path.
- **direct** — the callee is provably stateless (e.g. a transformed function with no state slots). Emission: plain call, untouched.
- **uniform** — the callee cannot be resolved at transform time (function-valued variables, parameters, conditionally rebound names, overload dispatchers). Emission: anchor slot with `__bind_any__` and an identity check.
- **skip** — builtins, stdlib, types/constructors, module properties, plot/log-style display functions, and the synthetic series-slot method calls emitted by the Series transformer. Untouched.

### Fast path

The child slot is filled on first call and reused afterwards — the hot path is a single list read:

```python
lib.ta.sma(__st__ if (__st__ := __state__[1]) is not None
           else __resolve_slot__(__state__, 1, lib.ta.sma), lib.close, 14)
```

### Fast path in a loop

A call site inside a loop needs one instance per iteration. The loop's call counter indexes a child list, grown on demand:

```python
def main(__state__):
    __cnt_0__ = 0
    __chl_0__ = __state__[0]
    total = 0
    for length in (5, 10, 20):
        total += counter(__chl_0__[__i__] if (__i__ := ((__cnt_0__ := (__cnt_0__ + 1)) - 1)) < len(__chl_0__)
                         else __grow__(__chl_0__, counter))
```

### Uniform path

When the callee is only known at runtime, the call site gets an **anchor slot** holding a `(callee, bound)` pair. The emission checks the callee's identity: on a hit it calls the cached binding, on a miss (first call, or the callee changed) `__bind_any__` rebinds with fresh state:

```python
(__b__[1] if (__b__ := __state__[7]) is not None and __b__[0] is f
 else __bind_any__(__state__, 7, f))(x)
```

`__bind_any__` handles what the callee turns out to be at runtime: state-carrying functions get a fresh state vector baked into a partial, exported library functions are unwrapped, overload dispatchers are bound through their `__pyne_bind__` factory, and plain callables pass through as-is. Loop-shaped uniform sites keep a list of pairs indexed by the call counter, so each iteration keeps its own instance.

Note: when the callee at a uniform site changes (`g = a if cond else b; g(x)`), the site is rebound with fresh state — state does not survive an `a -> b -> a` swap.

## Call Site Identifiers

Each isolated call site gets an identifier built from the scope path, the callee path and an ordinal — for example `main·lib.ta.sma·0`. These appear in the layout's `children` tuple and in the debug `names`, so a state tree can be navigated by eye. The Unicode middle dot (`·`) separator prevents collisions with underscores in function names.

## Example: Transformed Code

**Original code:**
```python
"""
@pyne
"""
from pynecore import Series

def main():
    def t():
        a: Series[float] = 1
        a += 1
        return a[1]

    a = t()
    print(a)
    b = t()
    print(b)
```

**Transformed code:**
```python
"""
@pyne
"""
from pynecore.core.instance_state import __resolve_slot__
__pyne_slot_layout__ = {'main': {'init': (None, None), 'series': (), 'varip': (), 'children': ((0, 'main·t·0', False), (1, 'main·t·1', False)), 'names': ('main·t·0', 'main·t·1')}, 'main·t': {'init': (None,), 'series': ((0, None),), 'varip': (), 'children': (), 'names': ('a',)}}

def main(__state·main__):

    def t(__state__):
        a = __state__[0].add(1)
        a = __state__[0].set(a + 1)
        return __state__[0][1]
    t.__pyne_layout__ = __pyne_slot_layout__['main·t']
    a = t(__st__ if (__st__ := __state·main__[0]) is not None else __resolve_slot__(__state·main__, 0, t))
    print(a)
    b = t(__st__ if (__st__ := __state·main__[1]) is not None else __resolve_slot__(__state·main__, 1, t))
    print(b)
main.__pyne_layout__ = __pyne_slot_layout__['main']
```

The two call sites of `t` got child slots 0 and 1 of main's state vector, so the two calls keep fully independent series buffers — exactly the Pine Script behavior. Note that `main` itself takes a state vector too (`__state·main__`, qualified because it contains a nested definition); the runner provides it as a root vector.

For decorated definitions the layout attach uses a decorator form (`@__attach_layout__(...)` in the innermost position) so the layout lands on the raw function before any other decorator wraps it.

## Performance Considerations

The slot scheme was designed for the bar-by-bar hot loop:

1. After the first call, reaching a callee's state is a single list index read — no dictionaries, no key construction, no cache lookups
2. Function objects are never recreated or wrapped at runtime; the emitted code calls them directly with the state vector as hidden argument
3. Only provably stateful call sites pay the slot check; stateless calls run untouched
4. State slots are read and written with literal int indexes — the fastest state access Python offers
5. Dropping an instance (e.g. on `reset()`) releases its whole subtree through normal garbage collection — no global registry to invalidate

## Summary

Function isolation is a fundamental PyneCore feature that enables accurate Pine Script compatibility. By giving every function instance its own state vector, parked in a slot of its caller's state vector, PyneCore ensures that indicators, strategies, and other scripts behave correctly in the bar-by-bar execution model — with hot-path costs reduced to plain list indexing.
