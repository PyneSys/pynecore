<!--
---
weight: 1400
title: "Reference"
description: "PyneCore language and library reference documentation"
icon: "menu_book"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference"]
tags: ["reference", "api", "functions", "variables", "constants", "types"]
---
-->

# Reference

Complete reference documentation for the PyneCore framework. PyneCore is compatible with
TradingView's Pine Script v6, but uses Python's native type system and conventions.

## Sections

### [Language](language/)

Pine Script language elements supported by PyneCore: operators, keywords, and input functions.

### [Types](types/)

PyneCore's type system, including primitive types, collections, drawing types, and
user-defined types. Explains how PyneCore simplifies Pine Script's qualifier-based type
system.

### [Library](lib/)

Auto-generated reference for all PyneCore library functions, variables, and constants.
Organized by namespace (ta, math, strategy, etc.) with compatibility status for each entry.

## Compatibility

PyneCore aims for full compatibility with Pine Script v6. The auto-generated library
reference includes a compatibility status for each entry:

| Status    | Description                                        |
|-----------|----------------------------------------------------|
| **full**  | Fully compatible with TradingView                  |
| **partial** | Works but with limitations (documented per entry) |
| **stub**  | Exists but not yet implemented                     |
| **missing** | Not available in PyneCore                        |
