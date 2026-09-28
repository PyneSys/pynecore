<!--
---
weight: 1080
title: "How Broker Plugins Are Tested"
description: "How the PyneCore broker plugins are tested before live trading: an offline conformance lab, continuous live runs on venue demo accounts, fault injection, and an end-of-cycle reconciliation with the venue"
icon: "verified"
date: "2026-09-28"
lastmod: "2026-09-28"
draft: false
toc: true
categories: ["Advanced", "Live Trading"]
tags: ["broker", "plugins", "testing", "live-trading", "reliability", "reconciliation"]
---
-->

# How Broker Plugins Are Tested

A broker plugin turns strategy decisions into real orders, so a bug costs money rather than a
wrong plot. The Bybit, Capital.com and cTrader plugins are therefore tested on three levels,
each catching failures the others cannot:

1. **Unit tests** of each plugin, fully offline, against a mocked venue.
2. **An offline conformance lab** that drives the real plugin through restarts, duplicated and
   reordered venue events, and partial fills, and checks invariants after every step.
3. **Continuous live runs** on venue demo accounts: real venues, real order flow, real
   reconnects, for days at a time, with injected network faults and an automatic comparison
   with the venue after every cycle.

## Offline conformance lab

The lab runs the real plugin code against the real PyneCore broker machinery (order sync engine,
persistent broker store, position model), with only the lowest transport layer replaced by a
modelled venue. No socket is opened.

Scenarios combine steps such as entries, exits, fills, partial fills, restarts, delayed,
duplicated, reordered or missing venue acknowledgements, and cancellations. After every step the
lab checks invariants, among them:

- the position PyneCore holds equals the modelled venue position;
- every client order id is unique, and no fill is applied twice;
- a restarted run adopts its own open orders and never another run's;
- every open position keeps its protective take-profit and stop-loss coverage;
- quantities stay on the venue's price and quantity grid.

The scenario corpus is deterministic: the same seed produces the same scenarios, and a failure
is reported with a minimized step sequence that reproduces it. Deliberately broken control
profiles make sure the checks actually detect the defects they target. A machine-checked
coverage map links each known failure family to the scenarios that cover it. Plugin authors can
use the same lab; see [Plugin System](../development/plugin-system.md).

An offline model can only prove what it encodes. Authentication, rate limits, undocumented venue
responses, matching behaviour and real network timing need the live runs.

## Live runs on demo accounts

Each plugin runs a trading bot with `pyne run --broker` on a demo account of its venue, around
the clock, including weekends. The instruments are ones that also trade on weekends, so a run
never idles for days.

### Cycles and restarts

A run is split into cycles of several hours. Each cycle ends with a regular shutdown, and the
next cycle starts with the same run identity, so it must adopt the open position and orders of
the previous one and restore their brackets. The restart path, the most fragile part of any
trading bot, is therefore exercised several times a day, usually with a position open.

A cycle that dies on its own is a failure: its log is kept, and after repeated failures the run
stops until the cause is fixed.

### The strategy matrix

Every cycle runs the next strategy of a fixed rotation, so the cycle boundary is both a restart
test and a switch of broker features:

| Strategy   | What it exercises                                                         |
|------------|---------------------------------------------------------------------------|
| `trend`    | market entries, reversals, a full take-profit / stop-loss bracket         |
| `partial`  | an entry in two slices, two partial exit levels and a stop                |
| `resting`  | limit and stop entries away from the price, cancellation of unfilled ones |
| `trailing` | a trailing stop together with a fixed stop                                |
| `pyramid`  | `pyramiding=3`, several entries, each with its own bracket (`from_entry`) |
| `flat`     | OCA-cancelled entry pairs, `strategy.close`, `cancel_all` and `close_all` |

The signals are deterministic (moving-average crossovers and ATR-based distances on one-minute
bars), so every cycle's log can be read back and explained afterwards.

### Reconciliation with the venue

At the end of every cycle the venue's position and open orders are compared with PyneCore's own
book. Any difference fails the cycle. This is the check that proves correctness rather than mere
survival: a bot that keeps running while its book has drifted from the venue is exactly the
failure that loses money.

### Fault injection

A healthy venue rarely produces network failures on demand, so one strategy of the rotation runs
behind a fault-injecting proxy. Each cycle follows a seeded fault plan of refused and
black-holed connections, cut and stalled streams, `5xx` and `429` responses, garbage responses
and slow responses. A fault cycle passes only if the bot survives every fault window, logs no
error outside them, and still matches the venue at the end.

### What is tracked

For every cycle the run records filled orders, bars that reached the strategy late or not at
all, bars the plugin recovered from the venue's REST history after a dropped stream, bars
PyneCore had to synthesize because a feed went silent, and every error. A bar gap that reaches
the strategy counts as a failure.

## Release criteria

A plugin is declared production-grade only after it passes all of these on its own venue:

- 72 hours of continuous running in which the plugin reports a manual intervention only when
  someone really acted on the account outside the bot;
- no data gap in the live phase, and every recovered or synthesized bar traced to a concrete
  feed event;
- the venue reconciliation after every cycle;
- at least 20 clean cycles of every strategy in the matrix;
- at least 20 restarts with an open position, each adopting it correctly and keeping its bracket;
- a flat account with no leftover order at the end of the campaign;
- at least 10 clean fault-injection cycles.

Every failure found in live runs is fixed at its root cause and, where the venue behaviour can
be modelled offline, added to the conformance lab as a regression scenario.
