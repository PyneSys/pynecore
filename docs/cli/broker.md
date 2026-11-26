<!--
---
weight: 305
title: "Broker Run Management"
description: "Manage quarantined live broker runs with the PyneCore CLI. Understand quarantine states, inspect what a bot continues to do and resume a run when appropriate."
icon: "shield"
date: "2026-10-14"
lastmod: "2026-11-25"
draft: false
toc: true
categories: ["CLI"]
tags: ["broker", "live-trading", "quarantine", "resume", "cli"]
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
| `external_intervention` | Someone else changed the bot's own venue state: a bot-owned order, its SL/TP, its share | Nothing. No order, amend, cancel, close or SL/TP update    |
| `entry_block`           | The bot's own venue trouble: a protective exit or a close the venue keeps refusing      | Protection keeps working; only new exposure is blocked     |

What counts as an intervention:

- A change to the bot's own venue state that neither the bot nor the venue's
  own mechanics (a fill, a triggered stop, a liquidation the venue reports)
  explains: an order of the bot cancelled or amended, its stop-loss or
  take-profit moved or removed, its position reduced below the bot's own
  share.
- A separate position or deal of another party on the same account is not
  an intervention, and neither is a later edit of it.
- When the evidence is inconclusive, the bot logs an ERROR, writes a durable
  audit event to the broker store and keeps trading. For example, the venue
  holds less than the run's journal owns at startup, with no fill report
  that explains it: the bot was not running, so it cannot tell a missed
  stop fill from a manual close.

An `external_intervention` quarantine leaves the venue exactly as the
operator left it. While it holds, the bot undoes nothing you change: move
the stop, delete the take-profit to ride a move, or close by hand.

The quarantine is stored in the broker store
(`workdir/output/logs/broker.sqlite`). An `external_intervention`
quarantine survives every restart: a crash, a deploy, a server reboot or a
move to another node starts the bot in quarantine again. An `entry_block`
quarantine ends when the run restarts.

### Log records

A quarantine is logged as two records, each on one physical line in every
log sink (a long record is never folded onto continuation lines), so a
supervisor can read the log line by line. The first carries the kind and
the quarantine id ahead of the free text, the second the exact command that
resumes it:

```
sync engine quarantined: kind=external_intervention quarantine_id=9f3a01bc — Bot-owned order cancelled unexpectedly on the venue ... — trading stopped, process stays alive; the bot leaves the venue untouched (no order, amend, cancel or close); ingestion keeps running
sync engine quarantine 9f3a01bc resume command: pyne broker resume 'my_strategy@bybit-demo-123:BTCUSDT.P:60' 9f3a01bc
```

The same pair is repeated, with a note after `quarantined`, when:

| Note                                         | Level   | When                                                                     |
|----------------------------------------------|---------|--------------------------------------------------------------------------|
| `(escalated from entry_block)`               | ERROR   | An intervention is detected while an `entry_block` quarantine holds      |
| `(restored at startup, entered <UTC time>)`  | ERROR   | A restarted bot finds an intervention that no operator resumed           |
| `(reminder, entered <UTC time>)`             | WARNING | Every 30 minutes while the quarantine holds                              |

A further intervention detected while the quarantine already holds (the
operator also moves another stop) is logged once, as a WARNING, and audited
in the broker store. It does not open a new quarantine, and a restart does
not log it again:

```
sync engine quarantine 9f3a01bc further intervention: kind=external_intervention — ...
```

## `pyne broker resume`

```bash
pyne broker resume RUN_ID QUARANTINE_ID
```

Resumes a quarantined run. Run it on the machine (or in the container)
where the bot runs, with the same workdir.

- A running bot picks the request up within seconds, logs
  `sync engine quarantine <id> resume requested by the operator`, stops,
  and restarts itself in place: same process, same command line, no
  supervisor involved. The quarantine stays in force until the restart:
  the stopping process writes nothing to the venue.
- The restarted process applies the resume, logs
  `sync engine quarantine resumed: kind=... quarantine_id=...` and
  re-syncs with the venue through the regular startup path.
- The quarantine is closed in the broker store once the restarted bot ran
  its first sync and handled the re-assert of its stop-loss and
  take-profit described below (logged as
  `sync engine quarantine <id> resume completed`). If the bot stops before
  that, for example because it cannot connect to the venue, its next start
  applies the resume again and logs
  `sync engine quarantine resumed (again: ...)`. What the stopped process
  already took over from the venue stands: a change someone made after it
  saw the venue, while no bot ran, is judged as a new intervention, not
  accepted by the old resume.
- A bot that is not running resumes at its next start, the same way.
- The request only resumes the quarantine it names. A stale or mistyped id
  never lifts a newer quarantine: the command fails and lists the open
  quarantines with their run ids and ids.
- Resuming an `entry_block` quarantine only restarts the bot. It accepts no
  venue change: the restarted bot judges the venue as after any restart.
  If an intervention escalates the quarantine to `external_intervention`
  before the restart, your request is dropped: the bot keeps (or restores)
  the escalated quarantine with an ERROR, and you resume it on its own.
  The rest of this section applies to an `external_intervention`
  quarantine.

A resume accepts the venue as it stands:

- **Position.** The bot takes its position from the venue. If you reduced
  or closed it, this is logged as a WARNING and audited. A later restart
  that finds the same reduction does not report it again, also after the
  bot's own later fills, and does not take exposure someone added on top
  of the reduced position as the bot's. A further reduction is reported,
  and so is the reduction of a later position, even when it has the same
  size.
- **Orders.** An order of the bot that you cancelled before the resume,
  while the bot ran or while it was down, is accepted as cancelled and not
  reported as a new intervention. A cancelled stop-loss or take-profit
  comes back through the re-assert below. Some venues only show which
  working orders are gone from the order history; there, a cancel is
  accepted when the restarted bot finds it within 5 minutes of the resume,
  unless the restarted bot already saw the order on the venue. A cancel
  found later is judged as a new intervention.
- **Stop-loss and take-profit.** The bot puts its own levels back once,
  from the exits the strategy still declares: a position-attached bracket
  is amended in place, an order-based bracket is cancelled and re-armed on
  new orders. Each exit is logged as `re-asserted` once its write went
  out. An amend that cannot go out yet (a connection loss, a market that
  is closed) is retried on the next sync, and the quarantine is closed
  only after it went out. When the venue's answer is unknown, a WARNING
  says `not applied (amend_unconfirmed)` (or `rearm_unconfirmed` for a
  re-armed order-based bracket); the bot settles the write like any other
  one whose answer was lost. When the levels could not go out yet (the
  old orders' cancel has not settled, the plugin skipped the amend), a
  WARNING says `not applied`, and the regular dispatch of a later sync
  puts the levels back. When the venue refuses the amend, the bot
  dispatches the exit fresh at once; if the venue refuses that too (or
  refuses a re-armed bracket twice), the bot closes the position and
  blocks new entries, as for any exit the venue refuses to place. When
  the market already trades through the bot's own level, the bot does
  what the strategy's exit does at a crossed level: it closes the
  position at market (a WARNING, no new quarantine). A broker-native
  fail-safe stop you moved is accepted as the new baseline and is not
  reported as a further intervention.
- If the venue refuses to put a bracket back at startup (for example, the
  market already trades through its level), the bot rebuilds that exit
  through its regular dispatch. A refusal there is handled like any other
  refused exit: a level the market already crossed closes the position at
  market.

Before you resume, settle the position on the venue the way you want the
bot to continue from it.

## A held partial-bracket close

A software partial bracket (a partial take-profit, stop-loss or trailing exit
the bot watches itself) closes its share of the position with a market close
when its level is reached. The bot settles that close only from what the
venue reports: until the close executes, ends without a fill, or ends after
executing part of it, the exit's leg is **held**. While it is held, the
exit's other legs (the stop-loss of a take-profit, and so on) stay in place
but do not trigger, and no second close is sent.

- An executed close retires the exit's other legs.
- A close that ended with nothing executed re-arms the leg; the next cross
  sends a new close under a new client order id.
- A close that ended after executing part of the exit re-arms the leg for
  the rest, and the exit's other legs protect only the rest.
- A close the venue still works on stays held. This is not an error.

When no venue record shows whether a close executed (its answer was lost and
the venue keeps no record the plugin can read), the bot cannot decide. It
keeps the leg held and enters an `entry_block` quarantine, with an ERROR that
names the close's client order id and the commands that settle it:

```
partial-bracket leg ... held: its close my_tag-...-c1 may have executed and is not sent again; no venue record shows whether it executed, and none will. ... To settle it, check the venue and record the outcome: "pyne broker resume 'RUN_ID' 9f3a01bc --close-executed my_tag-...-c1" if it executed, "pyne broker resume 'RUN_ID' 9f3a01bc --close-not-executed my_tag-...-c1" if it did not
```

Look the close up on the venue (its client order id is the order's link id
or label), then resume with the outcome:

```bash
pyne broker resume RUN_ID QUARANTINE_ID --close-executed CLOSE_COID
pyne broker resume RUN_ID QUARANTINE_ID --close-not-executed CLOSE_COID
```

Both options can be repeated for several held closes. The command refuses a
client order id that names no held close of the run and lists the held ones.
The decision is stored with the resume request; the restarted bot settles
the leg with it, as if the venue had reported that outcome, and journals an
`operator_close_decision_applied` event. A plain resume without a decision
only restarts the bot, and the same close quarantines it again.

On a hedging account, a close the bot fanned out over several position legs
may have a leg close whose send is not settled yet. The ERROR then says so
and offers only the plain resume: the restarted bot first settles that send
itself, then decides the close from the venue.
