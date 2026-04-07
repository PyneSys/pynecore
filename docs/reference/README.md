<!--
---
weight: 1400
title: "Reference"
description: "PyneCore type system and library reference documentation"
icon: "menu_book"
date: "2026-03-28"
lastmod: "2026-04-05"
draft: false
toc: true
categories: ["Reference"]
tags: ["reference", "api", "functions", "variables", "constants", "types"]
---
-->

# Reference

Complete reference documentation for PyneCore.

## Sections

### [Script Format](script-format.md)

Anatomy of a `@pyne` script: magic comment, imports, decorator, `main()` function, and input
parameters.

### [Types](types.md)

PyneCore's type system: primitives, type annotations (`Series`, `Persistent`, `IBPersistent`),
collections, drawing types, and special types (`@udt`, `StrEnum`).

### [Input Functions](inputs.md)

User-configurable script parameters (`input.int()`, `input.float()`, `input.string()`, etc.)
and TOML-based configuration.

### [Library](lib/)

Reference for all PyneCore library functions, variables, and constants. Organized by namespace
(ta, math, strategy, etc.) with compatibility status for each entry.
