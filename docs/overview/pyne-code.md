<!--
---
weight: 102
title: "What is Pyne code?"
description: "Pyne's domain-specific language for market data, indicators, and trading strategies: Python syntax, bar-by-bar semantics, and the edge subset."
icon: "code"
date: "2026-10-18"
lastmod: "2026-10-25"
draft: false
toc: true
categories: ["Overview"]
tags: ["pyne-code", "dsl", "python", "scripting", "series", "persistent", "pyne-edge"]
---
-->

# What is Pyne code?

**Pyne code** is the language used to describe calculations on market data, technical
indicators, and trading strategies. Pyne is a **domain-specific language (DSL)**: its
execution rules and library are designed for this particular kind of work.

Pyne uses **Python syntax**. Indentation, function definitions, expressions, and type
annotations use Python's notation. Its time-series and state rules come from Pyne, so
understanding ordinary Python syntax alone does not describe how a Pyne program runs.

You can learn Pyne directly. Prior knowledge of Python or Pine Script is not required;
the [beginner's book](https://pynecore.org/learn/en/) introduces the language through PyneIDE examples.

## The language, runtime, and editor

| Name      | Role                                                                                |
| --------- | ----------------------------------------------------------------------------------- |
| Pyne code | The language in which you write indicators, strategies, and reusable libraries.     |
| PyneCore  | The open-source runtime that prepares and executes Pyne code.                       |
| PyneIDE   | The Visual Studio Code extension (currently in preview) for editing and running it. |
| PyneComp  | The separate compiler that converts existing Pine Script into Pyne code.            |

Writing Pyne code does not require converting a Pine Script first. You can create your
own program in PyneIDE or a text editor and run it with PyneCore.

## Python syntax, Pyne execution rules

PyneCore processes market candles in order and calls a script's `main()` function for
each historical bar. The program describes what to calculate for the current bar;
the runtime advances through the data and manages the calculation state.

Pyne adds behavior that ordinary Python does not provide automatically:

- **Series history:** `close[1]` reads the previous bar's close. A variable annotated
  with `Series[T]` retains its own calculated history.
- **Persistent state:** `Persistent[T]` retains a variable's value between bars.
  `IBPersistent[T]` supports state that survives intrabar rollback during live updates.
- **Stateful calculations:** functions such as `ta.sma()` maintain calculation state.
  Stateful function call sites have separate state, so independent calls do not share
  an indicator's history accidentally.
- **Missing values:** the `na()` predicate and the library's missing-value rules let
  calculations represent unavailable data, including indicator warm-up periods.

Live execution can recalculate an open bar as updates arrive. See
[Core Concepts](./core-concepts.md) and [Live Mode](../advanced/live-mode.md)
for the execution and rollback rules.

## A small Pyne program

Save this as `moving_average.py`:

```python
"""@pyne edge"""

from pynecore.lib import close, plot, script, ta


@script.indicator("Moving Average", overlay=True)
def main():
    average = ta.sma(close, 20)
    plot(average, title="SMA 20")
```

On each bar, the program calculates a 20-bar moving average and emits a plot value.
PyneIDE can display those values on its chart. The average is unavailable until
enough closing prices have been processed.

The file uses the `.py` extension because its syntax is Python-compatible. The module
docstring must be the file's first statement, and `@pyne` must be its first
non-whitespace token. This tells PyneCore's loader to apply Pyne's language
transformations before execution. The marker alone does not launch the program;
use PyneIDE, the [CLI](../cli/run.md), or the
[ScriptRunner API](../programmatic/script-runner.md).

## What does `edge` mean?

`@pyne edge` selects a defined, restricted subset of Pyne code. PyneIDE checks this
profile and reports constructs outside it. The beginner's book uses this subset
throughout.

The profile limits available syntax, imports, and calls. It supports the Pyne
calculation model without requiring arbitrary Python facilities. For example,
general Python packages, comprehensions, and exception handling are outside the
profile. This gives tooling a defined language contract without requiring a
general-purpose Python environment. A class is a user-defined type: its body
declares fields and methods, with no inheritance (see
[Types and methods in the edge profile](../reference/script-format.md#types-and-methods-in-the-edge-profile)).

Plain `@pyne` supports broader Python integration in the Python-based PyneCore
runtime, including external libraries. Those programs still use Pyne's execution
rules, but they no longer promise compatibility with the edge subset. Removing
`edge` therefore changes the program's language-profile contract, not just its label.

See [Script Format](../reference/script-format.md) for the file structure and
marker rules.

## How is Pyne related to Pine Script?

Pyne shares Pine Script's bar-by-bar approach and provides a compatible technical
analysis and strategy API. It has its own syntax conventions and explicit state
annotations. Existing Pine Script can be converted with PyneComp; new Pyne programs
can be written directly.

Pine compatibility is measured against TradingView and has documented boundaries.
See [Compatibility](./compatibility.md) and
[Differences from Pine Script](./differences.md) for details.
