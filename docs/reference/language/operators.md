<!--
---
weight: 402
title: "Operators"
description: "Arithmetic, comparison, logical, and special operators in PyneCore"
icon: "calculate"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Language"]
tags: ["operators", "arithmetic", "comparison", "assignment", "ternary"]
---
-->

# Operators

PyneCore supports all Pine Script v6 operators. Since PyneCore scripts are compiled Python,
operators behave identically to their Pine Script counterparts.

## Arithmetic Operators

| Operator | Description              | Example          | Result Type   |
|----------|--------------------------|------------------|---------------|
| `+`      | Addition / string concat | `a + b`          | int/float/str |
| `-`      | Subtraction / negation   | `a - b` or `-a`  | int/float     |
| `*`      | Multiplication           | `a * b`          | int/float     |
| `/`      | Division                 | `a / b`          | float         |
| `%`      | Modulo                   | `a % b`          | int/float     |

When applied to Series values, operators work element-wise on the current bar's value.

## Comparison Operators

| Operator | Description            | Example   | Result Type |
|----------|------------------------|-----------|-------------|
| `==`     | Equal                  | `a == b`  | bool        |
| `!=`     | Not equal              | `a != b`  | bool        |
| `<`      | Less than              | `a < b`   | bool        |
| `<=`     | Less than or equal     | `a <= b`  | bool        |
| `>`      | Greater than           | `a > b`   | bool        |
| `>=`     | Greater than or equal  | `a >= b`  | bool        |

Comparison with `na` always returns `na`.

## Logical Operators

| Operator | Description  | Example        |
|----------|-------------|----------------|
| `and`    | Logical AND | `cond1 and cond2` |
| `or`     | Logical OR  | `cond1 or cond2`  |
| `not`    | Logical NOT | `not cond`        |

## Assignment Operators

| Operator | Description              | Example    |
|----------|--------------------------|------------|
| `=`      | Assignment               | `a = 1`    |
| `:=`     | Reassignment             | `a := 2`   |
| `+=`     | Addition assignment      | `a += 1`   |
| `-=`     | Subtraction assignment   | `a -= 1`   |
| `*=`     | Multiplication assignment| `a *= 2`   |
| `/=`     | Division assignment      | `a /= 2`   |
| `%=`     | Modulo assignment        | `a %= 3`   |

### PyneCore note

In Pine Script, `=` declares a new variable and `:=` reassigns an existing one. In compiled
PyneCore code, both map to Python's `=` assignment. The PyneComp compiler enforces the
declaration/reassignment distinction at compile time.

## Special Operators

| Operator | Description           | Example                 |
|----------|-----------------------|-------------------------|
| `?:`     | Ternary conditional   | `cond ? a : b`          |
| `[]`     | History reference     | `close[1]`              |
| `=>`     | Function body / switch| `f(x) => x + 1`         |

### History reference operator `[]`

The `[]` operator accesses historical values of a Series. `close[1]` returns the previous bar's
close price, `close[2]` the one before that, and so on. `close[0]` is equivalent to `close`.

In PyneCore, this compiles to Python's subscript operator on Series objects.

### Ternary operator `?:`

Pine Script's `cond ? valueA : valueB` compiles to Python's `valueA if cond else valueB`.
