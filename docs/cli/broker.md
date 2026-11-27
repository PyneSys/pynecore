<!--
---
weight: 305
title: "Broker Run Management"
description: "Manage quarantined live broker runs with the PyneCore CLI. Understand quarantine states, inspect what a bot continues to do and resume a run when appropriate."
icon: "shield"
date: "2026-10-14"
lastmod: "2026-11-26"
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

| Kind                    | Cause                                                                                     | What the bot still does on the venue                        |
|-------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| `external_intervention` | Someone else changed the bot's own venue state: a bot-owned order, its SL/TP, its share   | Nothing. No order, amend, cancel, close or SL/TP update     |
| `entry_block`           | The bot's own trouble: a protective exit or a close the venue keeps refusing, a lost fill | Protection and closes keep working; new exposure is blocked |

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
quarantine ends when the run restarts, and the running bot ends it by itself
(see below).

### The bot ends an `entry_block` quarantine by itself

A bot is meant to run without anyone watching it. An `entry_block`
quarantine is the bot's own trouble, so the bot gets out of it without a
command: it keeps reading what it is missing from the venue, with a delay
that grows up to five minutes and no deadline, and ends the quarantine once
every cause it was entered for is resolved. The process keeps running and
the strategy's next entry is dispatched.

| Cause                                                | Ends when                                                                                       |
|------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| The broker store failed on consecutive syncs         | A sync completes with every store write landed                                                  |
| An entry's executed quantity or fill never arrived   | The bot read the order's execution from the venue and booked it, and the venue matches the book |
| A partial-bracket close no record settles            | The bot read the close from the venue's order history and settled the leg                       |
| A hedging-fan close leg whose send is unsettled      | The bot settled the send itself (persisted first, never sent blindly), then the close           |
| A netting close fill no record identifies            | No netting is owed and the venue holds what the book holds                                      |
| A protective exit the venue refused                  | The defensive close of the position settled and nothing of the entry is open                    |
| A close the venue rejected repeatedly                | The close landed, or the strategy withdrew it                                                   |
| A defensive close that did not close its position    | The close, sent again from a fresh venue read, settled and nothing of the entry is open         |
| A partial-bracket cancel whose outcome stays unknown | The venue answered the cancel probe                                                             |
| Spot inventory: a failed read or a negative ledger   | The bot measured the balance and the execution history again and they settled                   |
| Spot inventory: another live run held the asset      | The other run released the asset; the bot starts on the measured balance, owning nothing        |
| Spot inventory: a fill another run booked too        | The measurement settled; the log names both runs, two runs trade one account and base           |
| A plugin report, or a write fault it left raw        | A reconcile passed without another one and the venue holds what the book holds                  |
| A broker plugin port that breaks its contract        | A port probe passes (the bot sends nothing until then; fix the plugin and restart it)           |

While it holds, a close or a stop-and-reverse the strategy asks for still
closes the position (a reversal's new side stays blocked). A defensive close
that did not go out is kept in the broker store, so a restart sends it again
from a fresh venue read instead of forgetting it. A broker plugin that lets an
unclassified error escape an order write is taken as a write of unknown
outcome: the bot verifies it against the venue, never sends it again blindly,
and keeps running. The bot never sends an order again whose outcome it does
not know: it reads the venue first, and a close it sends again is sized from
that read so it cannot over-close. The end is logged as one WARNING line:

```
sync engine quarantine 9f3a01bc ended by the bot: kind=entry_block — the execution of entry L is booked and the venue holds what the book holds; entries are dispatched again
```

No command is needed, and no statement of yours settles anything: the bot
learns every outcome from the venue. `pyne broker resume` is only needed
after an external intervention. On an `entry_block` quarantine it only
restarts the bot, which judges the venue as after any restart.

One state is not an `entry_block` the bot ends: a spot bot whose asset lease a
replacement instance of the same run took over (a zombie process that came
back after the restart). That process stops writing to the venue for good,
no order, amend, cancel or close, and logs:

```
spot inventory lease lost (...): a replacement instance of the run owns the BTC lease now; this process writes nothing to the venue any more (no order, amend, cancel or close) and never resumes, the replacement runs the bot
```

Stop that process; the replacement keeps running the bot.

A broker-native fail-safe stop the bot keeps on the venue blocks new entries
while its state is degraded (its write failed or stayed unconfirmed after the
retries). When the bot itself wrote the bracket last, it re-arms it by
itself, first after 5 seconds and then with a delay that doubles up to five
minutes, and new entries resume once the venue confirms the bracket. A
bracket someone else edited on the venue is never re-armed by the bot.

### Log records

A quarantine is logged as two records, each on one physical line in every
log sink (a long record is never folded onto continuation lines), so a
supervisor can read the log line by line. The first carries the kind and
the quarantine id ahead of the free text, the second what ends it. For an
`external_intervention` that is the exact command that resumes it:

```
sync engine quarantined: kind=external_intervention quarantine_id=9f3a01bc — Bot-owned order cancelled unexpectedly on the venue ... — trading stopped, process stays alive; the bot leaves the venue untouched (no order, amend, cancel or close); ingestion keeps running
sync engine quarantine 9f3a01bc resume command: pyne broker resume 'my_strategy@bybit-demo-123:BTCUSDT.P:60' 9f3a01bc
```

An `entry_block` is the bot's own trouble and needs no command: the second
record names its open causes, which the bot ends by itself:

```
sync engine quarantined: kind=entry_block quarantine_id=4c1d77e0 — entry_execution_evidence_missing (intent_key=L, context={...}) — trading stopped, process stays alive; entry dispatch is blocked while ingestion/exits/cancels/closes keep running
sync engine quarantine 4c1d77e0 ends by itself: open causes: fill_evidence:L — each ends once its condition is gone, and new entries are dispatched again with the last; no command is needed
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

When the plugin's journal holds no record of whether a close executed (its
answer was lost), the bot keeps the leg held and enters an `entry_block`
quarantine. It then reads the close from the venue's order history by its
client order id, again and again with a growing delay, and settles the leg
with what the venue shows; the quarantine ends with it. Nothing is needed
from you: every broker plugin reads an order's execution from the venue.

On a hedging account, a close the bot fanned out over several position legs
may have a leg close whose send is not settled yet. The bot settles that
send itself in the running process once the send's window has passed: it
records what it is about to do before it writes to the venue, and it never
sends a leg close again without first reading whether the earlier one
landed. Then it decides the close from the venue as above.

## An entry whose execution never arrived

When the venue ends an entry's order without saying how much of it executed,
or reports an execution whose fill never reaches the bot, the bot holds the
entry: nothing is sent under its id, and a close or reversal the strategy
asks for waits for the true position. After the fill-report grace the bot
enters an `entry_block` quarantine and reads the order from the venue by
itself:

- An execution is booked with the venue's figures: its own fills when the
  venue keeps them, otherwise its average price and cumulative commission.
  A commission the venue does not report is booked as none and logged as
  unreported. The bot never invents a price: an order it holds part of and
  whose venue keeps only the totals waits for its fills.
- An order the venue shows unexecuted, or does not know at all, releases the
  entry; the strategy may send it again.
- A close or reversal the strategy asks for meanwhile closes the venue
  position first, in a form that cannot over-close or flip.

The quarantine ends by itself once the execution is booked and the venue
holds what the book holds. An execution record that contradicts itself (its
fills do not add up to the executed quantity, it belongs to another order,
it executed more than the order's size) books nothing: the bot logs an ERROR
naming the broken rule and reads the order again later.

The same holds when the broker plugin proves that an entry's order reached
the venue and is closed since, so no fill of it can arrive any more: a send
whose answer was lost (or that an earlier process sent before it stopped),
or a resting entry order that executed while the bot was not running. The bot
does not send the entry again: it logs

```
entry L: its order <client order id> (venue order <id>) ended at the venue and the stream will not report its execution — the order is not sent again; new entries wait while the bot reads the order's execution from the venue and books it
```

and books the order's execution from the venue as above, whether or not the
strategy still asks for the entry. The plugin then reports the close of the
position the order opened, and the round trip is booked with the venue's
prices. When no such report arrives within the fill-report grace, the bot
closes the position out by itself at its entry price (the realized P&L is the
commission only, approximate) and logs it as one WARNING:

```
the venue holds 0.0 and the book 1.0: the position the handed-over execution(s) ['<id>'] opened closed at the venue, and no close fill reported it within 120 s — closed out as sell 1.0 at the entry price 100.0 (...)
```

The quarantine ends with the book flat like the venue. A close report that
arrives later re-prices that close at the venue's price.
