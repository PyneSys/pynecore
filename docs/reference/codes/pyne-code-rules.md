<!--
---
weight: 432
title: "Pyne Code Rules"
description: "Look up the P codes PyneCore reports when a module breaks a rule of Pyne code, before it runs: the message, the place it points at and the fix."
icon: "code"
date: "2026-11-15"
lastmod: "2026-11-15"
draft: false
toc: true
categories: ["Reference"]
tags: ["error-codes", "pyne-code", "transformations"]
---
-->

# Pyne Code Rules

<a id="P0001"></a>

## P0001: Identifier with PyneCore's reserved character

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{name}` contains `·`, which is reserved for PyneCore's internal names in Pyne code; rename the identifier.
```

PyneCore names the state it adds to a script with a `·` in the name, so no
identifier of Pyne code may hold one. Python folds a few other characters
into `·` (NFKC normalization), so an identifier spelled with them is rejected
too.

<a id="P0002"></a>

## P0002: Identifier reserved by the transform

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{name}` is a name PyneCore's transform generates, reserved in Pyne code; rename the identifier.
```

The transform adds functions and variables of its own to every script. An
identifier of the script spelled like one of them would be overwritten by
it, so the name is reserved.

<a id="P0003"></a>

## P0003: Script decorator not on main

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
The script decorator belongs on `main`, not on `{function}`.
```

A script has one entry point, the function the runner calls: `main`. The
`@script.indicator`, `@script.strategy` or `@script.library` decorator marks
it, so it stands on `main` and on no other function.

<a id="P0004"></a>

## P0004: Second script decorator

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
A module has one script decorator, and `main` is defined twice.
```

A module has one entry point. Keep one decorated `main` and remove the other.

<a id="P0005"></a>

## P0005: na_bool is not a literal

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`na_bool` must be a literal `True` or `False`: the script's bool semantics are fixed before its module body runs.
```

`na_bool` decides how the script's bools are compiled, so PyneCore reads it
from the source before the module runs. Write it as `True` or `False`, not as
a name or an expression.

<a id="P0006"></a>

## P0006: Series variable outside a function

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
A `Series` variable must be declared inside a function.
```

A series keeps its history per call of the function that declares it. At
module level there is no such call, so declare the series inside `main` or
another function.

<a id="P0007"></a>

## P0007: Persistent variable outside a function

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
A `Persistent` variable must be declared inside a function.
```

A persistent variable keeps its value from bar to bar per call of the
function that declares it, as Pine's `var` does. Declare it inside `main` or
another function.

<a id="P0008"></a>

## P0008: Stateful call at module level

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
A function that keeps state from bar to bar cannot be called at module level.
```

A `ta.*` call, or any function that keeps a series or a persistent, stores
its state per call site and per bar. The module body runs once, not on every
bar, so call it from `main` or a function `main` calls.

<a id="P0009"></a>

## P0009: Input outside main's parameters

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{function}()` is not a `main()` parameter default; inputs must be declared as `main()` parameter defaults: `def main(length=input.int(14, ...))`.
```

PyneCore reads a script's inputs from the defaults of `main`'s parameters, so
that a run can set them before the script starts. An input called anywhere
else cannot be set.

<a id="P0010"></a>

## P0010: lib imported under an alias

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`lib` must be imported as itself, not as an alias.
```

The transform recognizes the Pine builtins by the `lib.` prefix. Write
`from pynecore import lib`.

<a id="P0011"></a>

## P0011: Wildcard import of a module without __all__

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
Cannot use a wildcard import: the module `{module}` has no `__all__` defined.
```

A wildcard import brings in exactly the names of the module's `__all__`.
Without one, import the names you use one by one.

<a id="P0012"></a>

## P0012: Unknown lib attribute

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{module}` has no attribute `{name}`.
```

The name is not part of this module of `pynecore.lib`. Check its spelling.
If it is a new `pynecore.lib` name, the module property registry is stale:
regenerate `module_properties.json` with `scripts/module_property_collector.py`.

<a id="P0013"></a>

## P0013: Module-level object modified inside a function

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{name}` {how} inside a function: an object created outside a function cannot be modified inside one.
```

Module-level state is not restored when a bar is re-executed, so a change
made to it on one execution of a bar would stay when the bar runs again.
Declare the value as `Persistent` or `IBPersistent` inside the function
instead.

<a id="P0014"></a>

## P0014: Plot called conditionally

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{function}()` must be called unconditionally in the direct body of module-level `main`, before any possible early return.
```

A plot is one series of the script's output, and its place in the output is
fixed by its call. A call inside a branch, a loop, another function or after
an early return would leave the series without a value on some bars. Plot
`na` instead where nothing should be drawn.

<a id="P0015"></a>

## P0015: Plot used as a value

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{function}()` cannot be passed or stored as a function value; use a direct call or a single static alias so its scope can be checked.
```

PyneCore checks where each plot is called. A plot function passed around as
a value could be called from anywhere, so call it directly.

<a id="P0016"></a>

## P0016: Stateful method of the wrong kind

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
The method `{method}` keeps state (a `ta` call, a series or a persistent), which only a plain or a `@staticmethod` method can do.
```

A method's state lives with the call site of the method, which PyneCore can
tie only to a plain method or a static method. Remove the other decorator,
or move the stateful code into a function.

<a id="P0017"></a>

## P0017: Method receiver from an unpacked argument

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{function}()` uses variables of `main()`, so its receiver cannot come from an unpacked argument; pass the receiver explicitly: `{function}(receiver, *args)`.
```

A method that reads variables of `main()` gets them passed in after its
receiver, so PyneCore has to see which argument the receiver is.

<a id="P0018"></a>

## P0018: Export captures a library global

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
The exported function `{function}` cannot capture the non-constant library global `{name}`.
```

An exported function runs in the importing script, where the library's
module-level variables do not change with the library's bars. Pass the value
in as a parameter, or make the global a constant.

<a id="P0019"></a>

## P0019: Strategy state in a security expression

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`strategy.{name}` cannot be used as the expression argument of `{function}()`: strategy state is only available in the chart context, not in a security context.
```

A `request.security` expression runs on another symbol or timeframe, where
the chart's strategy has no position, orders or trades.

<a id="P0020"></a>

## P0020: Drawing in a security expression

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{name}()` cannot be used in the expression of `{function}()`.
```

A drawing belongs to the chart. A `request.security` expression runs in
another context, so it cannot create or change one.

<a id="P0021"></a>

## P0021: Too many requests in one function

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
The function `{function}` reaches {count} `request.security` calls through the functions it calls; a function can use at most {limit} request values.
```

Each `request.security` call a function reaches, directly or through the
functions it calls, takes one request value of that function. Split the
work across functions, or request a tuple of values in one call.
