<!--
---
weight: 401
title: "Language"
description: "Pine Script language elements supported by PyneCore"
icon: "translate"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Language"]
tags: ["operators", "keywords", "inputs", "language", "syntax"]
---
-->

# Language

Pine Script language elements and how they map to PyneCore (compiled Python).

## [Operators](operators.md)

Arithmetic, comparison, logical, assignment, and special operators.

## [Keywords](keywords.md)

Variable declarations (`var`, `varip`), control flow (`if`, `for`, `while`, `switch`),
and module system (`import`, `export`, `type`, `enum`, `method`).

## [Input Functions](inputs.md)

User-configurable script parameters (`input.int()`, `input.float()`, `input.string()`, etc.).
In PyneCore, these are resolved at compile time to their default values.
