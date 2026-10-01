<!--
---
weight: 100
title: "PyneCore Documentation"
description: "PyneCore documentation: run Pine Script-style indicators and strategies in Python, with results validated against TradingView"
icon: "code"
date: "2025-03-31"
lastmod: "2026-09-28"
draft: false
toc: true
categories: ["Documentation"]
tags: ["overview", "introduction", "documentation"]
---
-->

# PyneCore Documentation

PyneCore runs Pine Script-style indicators and strategies in Python. Its API follows TradingView's Pine Script v6,
and existing Pine Script is converted to Pyne code with [PyneComp](https://pynesys.io), a separate service used with an
API key.

The results are validated against TradingView on <!--wild:scripts_total-->900<!--/wild--> published TradingView scripts:
all <!--wild:tv_verified-->1,213<!--/wild--> comparable outputs match, <!--wild:bars_exact_pct-->99.762<!--/wild-->% of
<!--wild:bars_compared_millions-->115<!--/wild--> million plotted values are bit-identical, and
<!--wild:strategy_trades-->314,210<!--/wild--> strategy trades match TradingView's timing (snapshot
<!--wild:generated_at-->2026-10-01<!--/wild-->, [Pyne in the Wild](https://wild.pynesys.io/)). See
[Compatibility](./overview/compatibility.md) for the status of each feature.

## Documentation Sections

- [Overview](./overview/README.md) - PyneCore overview and main concepts
- [Getting Started](./getting-started/README.md) - Learn how to install, configure and write your first PyneCore script
- [Command Line Interface](./cli/README.md) - PyneCore Command Line Interface (CLI) overview and usage
- [Library](./lib/README.md) - PyneCore library reference
- [Scripting with PyneCore](./scripting.md) - Writing effective and idiomatic Pyne code
- [Strategy Development](./strategy.md) - Creating and testing trading strategies with PyneCore
- [Programmatic Usage](./programmatic/README.md) - Using PyneCore from Python code (ScriptRunner, integrations)
- [Debugging](./debugging.md) - Debugging techniques for PyneCore scripts
- [Advanced](./advanced/README.md) - Advanced topics and features of PyneCore
- [Development](./development/README.md) - Documentation for PyneCore developers
- [FAQ](./faq.md) - Frequently asked questions about PyneCore
