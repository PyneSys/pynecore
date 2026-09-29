<!--
---
weight: 102
title: "Pyne Ecosystem"
description: "The Pyne ecosystem: PyneCore, the PyneComp Pine Script compiler, Pyne Edge and the Pyne in the Wild validation report, and how they work together"
icon: "lan"
date: "2025-03-31"
lastmod: "2026-09-28"
draft: false
toc: true
categories: ["Overview", "Ecosystem"]
tags: ["ecosystem", "pynecomp", "pynecore", "pyne-edge", "validation", "community"]
---
-->

# Pyne Ecosystem

The Pyne ecosystem lets you run TradingView-style indicators and strategies in Python. PyneCore is the open-source
runtime; the other components convert existing Pine Script to it and measure how closely its results match
TradingView.

## How the Components Fit Together

1. You write **Pyne code** directly, or convert an existing Pine Script with **PyneComp**.
2. **PyneCore** runs the Pyne code bar by bar, with Pine Script's semantics.
3. **Pyne in the Wild** runs the same pipeline on hundreds of published TradingView scripts and compares every result
   with TradingView's own.

## PyneCore (Open Source)

PyneCore is the foundation of the ecosystem: a Python runtime whose API follows TradingView's Pine Script v6. It
provides:

- Import-time AST transformations that give ordinary Python Pine Script's bar-by-bar execution model
- Series variables with bar history and persistent variables that keep their value from one bar to the next
- `na` handling that follows Pine Script
- The Pine Script v6 library, including every `ta.*` function
- Strategy backtesting that follows TradingView's strategy engine

PyneCore is free and open source under the Apache 2.0 license. Learn more on the
[What is PyneCore](/docs/overview/what-is-pynecore/) page.

## PyneComp (Pine Script Compiler)

PyneComp converts Pine Script to Pyne code. It compiles Pine Script v4, v5 and v6, and converts v1 to v3 sources on
a best-effort basis. It is a separate PyneSys service that needs an API key, available through:

- The [pynesys.io](https://pynesys.io) web interface
- The PyneCore CLI (`pyne compile`, or `pyne run` on a `.pine` file) with an API key
- The [PyneSys Discord bot](https://discord.pynesys.io): `/pyne-help` shows how it works, `/pyne-convert` converts a
  script, and every Discord user gets 3 free conversions
- The PyneSys API

See [Compiling Pine Scripts](/docs/cli/compile/) for the CLI workflow.

## Pyne Edge

PyneComp always generates **Pyne Edge** code, marked with `"""@pyne edge"""`: a strict, Pine-equivalent subset of
Pyne code. PyneCore runs it exactly like any other Pyne code; the marker guarantees that the script stays within the
subset, so tooling can rely on it. See [The `edge` Variant](/docs/reference/script-format/#the-edge-variant) for details.

## Pyne in the Wild (Validation Report)

[Pyne in the Wild](https://wild.pynesys.io/) is a public, reproducible comparison with TradingView. It converts
<!--wild:scripts_total-->830<!--/wild--> published TradingView scripts with PyneComp, runs them with PyneCore and
compares every comparable output with TradingView's: all <!--wild:tv_verified-->1,115<!--/wild--> comparable outputs
match, <!--wild:bars_exact_pct-->99.725<!--/wild-->% of <!--wild:bars_compared_millions-->103<!--/wild--> million plotted
values are bit-identical, and <!--wild:strategy_trades-->298,674<!--/wild--> strategy trades match TradingView's timing
(snapshot <!--wild:generated_at-->2026-09-29<!--/wild-->). Each script's source is pinned by a SHA-256 hash, and its
result is published individually. See [Compatibility](/docs/overview/compatibility/) for the feature status.

## What Comes Next

As of September 2026, development is focused on a hosted bot platform for running PyneCore strategies; a strategy
marketplace is planned later.

## Community and Support

Join our community to get help and share your experiences:

- [Discord Server](https://discord.pynesys.io)
- [Reddit](https://www.reddit.com/r/PyneSys)
- [GitHub Discussions](https://github.com/PyneSys/pynecore/discussions)
