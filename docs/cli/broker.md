<!--
---
weight: 305
title: "Broker Run Management"
description: "Resuming a quarantined live broker run with the PyneCore CLI"
icon: "shield"
date: "2026-10-14"
lastmod: "2026-10-14"
draft: false
toc: true
categories: ["Usage", "CLI", "Live Trading"]
tags: ["broker", "live-trading", "quarantine", "resume"]
---
-->

# Broker Run Management

`pyne broker` manages live runs started with `pyne run --broker`.

## Quarantine

A live bot never stops its process on its own: a stopped process would be
restarted by its supervisor (systemd, a container runtime) and carry on as
if nothing happened. When the bot must stop trading, it enters a
**quarantine** instead: the process stays alive, keeps ingesting fills and
logs, and refuses to trade.

There are two kinds:

| Kind                    | Cause                                                                                   | What the bot still does on the venue                       |
|-------------------------|-----------------------------------------------------------------------------------------|------------------------------------------------------------|
| `external_intervention` | Someone else changed a bot-owned order: the operator or another bot cancelled it        | Nothing. No order, amend, cancel, close or SL/TP update    |
| `entry_block`           | The bot's own venue trouble: a protective exit or a close the venue keeps refusing      | Protection keeps working; only new exposure is blocked     |

An `external_intervention` quarantine leaves the venue exactly as the
operator left it. While it holds, the bot undoes nothing you change: move
the stop, delete the take-profit to ride a move, or close by hand.

The quarantine is stored in the broker store
(`workdir/output/logs/broker.sqlite`). An `external_intervention`
quarantine survives every restart: a crash, a deploy, a server reboot or a
move to another node starts the bot in quarantine again. An `entry_block`
quarantine ends when the run restarts.

When the bot enters quarantine, it logs an ERROR line that ends with the
exact command that resumes it:

```
sync engine quarantined: Bot-owned order cancelled unexpectedly on the venue: ...
Resume with: pyne broker resume 'my_strategy@bybit-demo-123:BTCUSDT.P:60' 9f3a01bc
```

## `pyne broker resume`

```bash
pyne broker resume RUN_ID QUARANTINE_ID
```

Resumes a quarantined run. Run it on the machine (or in the container)
where the bot runs, with the same workdir.

- A running bot picks the request up within seconds, stops, and restarts
  itself in place: same process, same command line, no supervisor
  involved. It re-syncs with the venue through the regular startup path
  and restores its own orders, including stop-loss and take-profit levels
  you moved by hand.
- A bot that is not running resumes at its next start.
- The request only resumes the quarantine it names. A stale or mistyped id
  never lifts a newer quarantine: the command fails and lists the open
  quarantines with their run ids and ids.

Before you resume, settle the position on the venue the way you want the
bot to continue from it.
