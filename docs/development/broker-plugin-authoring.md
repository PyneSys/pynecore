<!--
---
weight: 1105
title: "Broker Plugin Authoring Guide"
description: "Implement a PyneCore BrokerPlugin using the documented contract for live order handling. Review persistence requirements and the behavior broker authors must test."
icon: "account_balance"
date: "2026-07-12"
lastmod: "2026-11-26"
draft: false
toc: true
categories: ["Development"]
tags: ["plugins", "broker", "live-trading", "contract", "capabilities"]
---
-->

# Broker Plugin Authoring Guide

This guide is the **contract checklist** for writing a `BrokerPlugin` — the
requirements that do not show up in the abstract method surface but will
break live trading (sometimes silently, sometimes with real money) when
missed. Read [plugin-system.md](plugin-system.md) first for the scaffolding:
entry points, config TOML generation, the `ProviderPlugin` basics and the
`BrokerStore` persistence surface. This document assumes all of that and
covers what a new broker author cannot see from the type signatures alone.

Three reference implementations exist, deliberately covering the common
venue shapes:

- **`pynecore-capitalcom`** — REST/poll CFD venue: per-deal positions,
  position-attribute brackets, no exchange-side client ids, software order
  stream fused from snapshot polls.
- **`pynecore-ctrader`** — push venue: protobuf WebSocket wire, native
  execution events, centi-unit volume grid, hedging-mode accounts.
- **`pynecore-bybit`** — push crypto exchange: JSON REST + WebSocket
  order/execution/position streams, client-id echo, native in-place
  amends, and category-based model selection (spot inventory / linear
  position / inverse contract mapping) inside one plugin.

All three follow the same file layout (`plugin.py` / `config.py` /
`provider.py` / `execution.py` / `reconcile.py` / `recovery.py` /
`models.py` / `exceptions.py` + a `_base.py` tying the mix-ins together) —
copy that structure rather than inventing a new one. When in doubt about
how to implement something, find the corresponding code in whichever
reference is closest to your venue's shape.

## The plugin stack and the division of labour

```
Plugin                     entry point, Config dataclass, TOML self-healing
└── ProviderPlugin         offline OHLCV download, SymInfo, timeframe mapping
    └── LiveProviderPlugin streaming bars, connection lifecycle, reconnect
        └── BrokerPlugin   order execution, order events, broker state
```

The Order Sync Engine (core) owns: the Pine-intent diff, dispatch
idempotency, retries, cancel-tentative state, OCA cascade fallback,
persistence, restart replay and startup adoption. The plugin owns: the
transport, translating `execute_*` intents to exchange calls, mapping
exchange events back to Pine identity, and **detecting orders that
disappear behind the engine's back** (see the checklist below — this last
one is the most commonly missed responsibility).

## Data layer requirements (`ProviderPlugin`)

Beyond the five abstract methods documented in `plugin-system.md`, live
trading depends on `update_symbol_info()` upholding these invariants:

- **`mintick` must equal `minmove / pricescale`.** Pine scripts and the
  broker sizing path both derive price grids from these; an inconsistent
  triple produces subtle price-rounding divergence.
- **`mincontract` must be positive** — order quantities are truncated to
  this grid before dispatch.
- **`session_starts` / `session_ends` must be populated even for 24/7
  instruments** (a full-week schedule). Empty session lists make
  session-driven built-ins like `timeframe.change("D")` hang waiting for a
  session boundary that never comes.
- **`session_schedules`** (effective-dated session history) should be
  populated for venues whose UTC session times drift with DST. The cTrader
  plugin's schedule rendering is the reference: it detects the venue/UTC
  offset changes and opens one variant per change. Leave it empty when the
  symbol's hours never changed.
- **`symbol_map`** comes for free from `LiveProviderConfig` — do not invent
  a plugin-specific translation knob. Symbols not in the map fall back to
  `normalize_symbol()`.

## Live layer requirements (`LiveProviderPlugin`)

- `watch_ohlcv()` blocks until the next update and returns an `OHLCV` with
  `is_closed=True` for a final bar, `False` for an intra-bar update. The
  framework calls it in a loop.
- **`connect()` is called again on every reconnect**, not just once. The
  framework drives a full `disconnect()` → `connect()` cycle after a
  connection error or a stale feed, so `connect()` must re-initialize
  every piece of connection-scoped state (auth, server-side subscriptions,
  partially accumulated quote state). Never assume a first-call-only
  environment.
- Reconnection is retried **indefinitely** with exponential backoff
  (`reconnect_delay` doubling up to `max_reconnect_delay`) — a live bot
  must survive arbitrarily long outages. Do not add your own retry limit.
- **Feed-liveness watchdog:** when the transport looks alive but
  `watch_ohlcv` stays silent for `feed_timeout_bars` timeframe periods
  during an open session, the framework assumes a half-open socket / lost
  subscription and forces a reconnect cycle. Only set it to `None` when
  the venue's own liveness machinery genuinely covers the dead-feed case.
- **Closed-bar recovery belongs to the framework, the history query to
  you.** The live stream itself recovers a close the socket missed: at
  startup (the bars that closed while the subscription came up), after
  every reconnect, and for a single close that did not arrive. It does so
  through `backfill_closed_bars(symbol, timeframe, since_ms)`, which you
  must implement as a **pure query**: any number of calls in any order
  leave the plugin's state as it was. Do not recover missed closes inside
  `watch_ohlcv()` or your reconnect handling — one slot must not be
  recovered twice. The answer has exact meaning:
  - return only closed bars, strictly after `since_ms`, ascending, in the
    venue's own order of publishing;
  - an **empty list** is the venue's answer "no such bar"; a query the
    venue **did not answer** must raise, and the stream then waits for
    the gap instead of going on over it;
  - `since_ms` is a bar open on the timeframe grid — never shift it by a
    millisecond to make a comparison inclusive.
- **`history_publish_lag`** — the longest delay after a bar's close until
  the history returns that bar, in seconds. It is the line between "not
  yet" and "none": an empty answer counts as "the venue has no bar for
  this slot" only when asked at least this long after the slot's close.
  Declare the longest delay you measured, not the typical one; a lag set
  too short turns a late-published bar into a hole.
- **`history_has_idle_bars`** — whether the venue's history has a
  zero-volume bar for a slot without trades. `True`: the stream emits the
  same filler bar. `False`: the stream emits none and `bar_index` does
  not step — the series a restart would replay from the history. Measure
  it on the venue; do not assume it.
- **`history_revises_before_lag`** — `True` when the history returns a
  fresh bar right at its close and changes its values afterwards, within
  `history_publish_lag`. The stream then takes no bar from an answer asked
  earlier than the lag past the bar's close, and asks again later.
  Default `False`. Poll the history every second after a few closes and
  watch whether the newest bar changes.
- **`stream_closes_match_history`** — `False` when the socket's closed
  bar is built from another price feed than the history (a throttled
  quote stream, for example), so its values can differ from the bar a
  restart replays. The stream then takes every closed bar from the
  history once `history_publish_lag` has passed, and the socket's close
  stands in only when the history has not answered by the close's
  deadline. Default `True`. Compare a few hundred socket closes with the
  history bars of the same slots before declaring `True`.
- **Error taxonomy:** subclass your plugin's expected failures from
  `ProviderError`; mark transient ones (connectivity drops, broker
  maintenance, socket timeouts) as `TransientProviderError` or set
  `retryable` from an error code. Long-running `--broker` / `--live` runs
  wait out retryable faults; a misclassified permanent fault (bad
  credentials marked retryable) loops forever, and a misclassified
  transient one stops a healthy bot from starting.

## Broker layer — the hidden contract

The abstract surface (`execute_entry` / `execute_exit` / `execute_close` /
`execute_cancel`, the state queries, `get_capabilities`) is small. The
following six requirements are what actually make a plugin safe to trade
with. `validate_plugin_contract()` (see below) enforces the
machine-checkable subset at startup; the rest can only be upheld by you.

### 1. Bot-owned-order disappearance detection is YOUR job

The engine deliberately does **not** diff its in-memory order mapping
against `get_open_orders()`, because resource namespaces are
broker-specific: a Pine entry may live as a working order, an open
position, or a position-attached bracket depending on the venue, and
`get_open_orders()` only covers one namespace. Without plugin-side
detection, a manual close on the broker UI, a broker-side liquidation or a
silent cancel is **never noticed** — the bot keeps trading against a
position that no longer exists.

The venue-agnostic core of the job — the persisted stamp/clear/grace
state machine, the grace-expiry confirmation protocol, the dual signal
and the `on_unexpected_cancel` policy — lives in
`pynecore.core.broker.disappearance.DisappearanceTracker`. Build one per
plugin instance and feed it per-namespace presence sets; your venue
knowledge enters through its hooks (see the Capital.com and cTrader
plugins' `reconcile.py` for the two reference wirings — the simple
snapshot-is-authoritative venue and the deal-history-re-verification
venue respectively):

1. Per poll (or per push-gap), snapshot **all** namespaces your venue
   stores bot orders in (positions, working orders, activity history)
   and feed them to the tracker (`observe` / `observe_presence`). A
   namespace whose fetch failed is reported as `None` — an incomplete
   snapshot must never look like a complete absence.
2. `tracked_refs` maps each live row to its `(namespace, ref)` set;
   the tracker stamps a row whose refs all vanished and clears the
   stamp the moment one reappears.
3. Only a stamp older than the **grace window** triggers anything, and
   even then your `confirm_missing` hook re-verifies first (deal
   history, order-status probe) — never conclude a cancel from missing
   evidence.
4. The tracker reports through `watch_orders()` with a dual signal: a
   synthesised `cancelled` `OrderEvent` (the engine's router cleans its
   tracking) plus the configured `on_unexpected_cancel` policy — the
   default `stop` latches the engine's **external-intervention
   quarantine** (wire the runner-injected `quarantine_sink` into the
   tracker's `request_quarantine`); no policy ends the process. A tracker
   whose sink is missing or raised raises `UnexpectedCancelError` from
   `observe()` instead, and the engine latches the same quarantine when it
   reaches it. The sink returns `False`
   when the held latch already noted the same intervention (the same
   order, protection key or reason); the tracker then skips its audit
   event, and a plugin that calls the sink itself can do the same.

   The sink returning says the latch holds in this process, not that it
   survives a restart: the engine latches in memory first and retries a
   failed store write on its own, and an `ENTRY_BLOCK` latch never survives
   a restart. When your plugin retires state on the strength of an
   external-intervention latch alone (a row whose retirement only the latch
   guards), request the latch with
   `BrokerPlugin.request_durable_intervention_quarantine(reason, context)`
   and retire only when it returns `True`: the store then holds an open
   external-intervention latch whose resume nobody requested yet
   (`RunContext.intervention_latch_stored()`, the one query for it). On
   `False` keep the state and ask again on a later pass; the engine's own
   retry stores the latch meanwhile. Call it outside your store
   transactions, so the latch write commits on its own. Without a wired
   sink it returns `False` (nothing latched).

   The tracker follows the same rule for its own unexpected cancels: under
   `stop` it requests the latch before it retires the row, and retires it
   only once `intervention_latch_stored()` answers `True`; until then the row
   keeps its stamp and is confirmed again on the next pass. A sink you wire
   into the tracker (and every test double of one paired with a real store)
   must therefore store the latch in the run's store, as the engine's sink
   does.
5. When the evidence shows the **venue** terminated the order itself
   (a reduce-only squeeze, an OCA sibling trigger, a cleared position),
   return `CANCELLED` with `cancel_reason` set to one of
   `VENUE_DRIVEN_CANCEL_REASONS`. The push of such a cancel can be lost
   too, and the history is then the only thing that tells it apart from
   an operator's cancel: the tracker retires the row and yields the
   `cancelled` event with that reason, and no policy fires. Stamp the
   same reason on a `cancelled` event you push from the private stream:
   the engine honours it for every bot order — an exit leg is trimmed
   (whatever its `reduce_only` flag, so a spot leg counts too), an
   unfilled entry or a close order is retired like a reject, and a
   partially filled entry is consumed at its filled quantity (the
   remainder is not placed again) — with a `venue_driven_cancel` audit
   and no policy. Repeated venue cancels of the same entry count towards
   the entry reject-retry cap, so the engine stops re-placing it until
   the script changes or re-issues it. Leave `cancel_reason` at `None`
   when the venue does not name the cause.

The tracker's `sibling_coids` hook (the live rows that share a `CLOSED`
row's position) runs **inside** the store transaction that retires the
row, after the row's terminal close and its audit event. This is a
guarantee you may build on: a journal write you make in it (a redelivery
record of the close that ended the position, for example) commits with
the retirement and rolls back with it, and an exception you raise there
rolls the whole retirement back and propagates. Never call the venue from
it.

The engine's stop-coverage guard depends on these verdicts. A missing
stop leg whose journal row is still live is not re-armed while its fate
is open: an external cancel must quarantine the run, not be overridden.
Only a closed row (the bot's own cancel, a venue-driven cancel, a
dispatch that never landed) lets the re-arm go ahead. If no verdict
arrives within `STOP_FATE_VERDICT_CEILING_S` (120 s), the engine re-arms
the stop anyway and reports it with an ERROR and a
`ProtectionRearmedUnverifiedEvent`, so retire every row whose fate you
know instead of leaving it live.

### 1a. An external intervention freezes every venue write — including yours

A bot-owned order that someone else cancelled or changed means the
operator (or another bot) knows something the bot does not. The engine
then latches a `QuarantineKind.EXTERNAL_INTERVENTION` quarantine: the
process stays alive and keeps ingesting events, but nothing is written
to the venue — no entry, exit, close, cancel, amend, SL/TP update or
protection re-arm — until the operator runs
`pyne broker resume <run_id> <quarantine_id>`. The latch is stored in
the broker store and survives every restart. A resume restarts the run in
place, so the bot re-syncs through the regular startup path. The latch
holds in the stopping process until it exits; the restarted process ends
it before your plugin is started.

The engine stops its own writes. Your plugin must stop the writes it
starts on its own (sibling cancels, trailing-stop activation, deferred
cleanup, retries): call `self.ensure_venue_writes_allowed(...)` at the
single transport chokepoint every order-mutating request passes through,
before anything leaves the client. It raises `VenueWritesQuarantinedError`,
an `ExchangeConnectionError`, so every caller keeps its state and retries
after the resume. Never treat it as "order not found" or as a settled
outcome, and never retire a journal row because of it. A protection write
it stopped never left the client: settle its guard expectation as refused
(`guard.settle(key, None, rejected=levels, write=write_id)`, see 1b), not
as an unknown outcome, or the unsent levels stay acceptable on the venue.

A quarantine your plugin requests for its own venue trouble (for example
a spot inventory conflict) keeps the default `QuarantineKind.ENTRY_BLOCK`:
new exposure is blocked, protection keeps working.

### 1b. Report a moved SL/TP and a manual close — and nothing else

Only the plugin can see who changed a bot-owned order or position, so it
must report it: call `self.report_external_intervention(reason, context)`.
It applies the `on_unexpected_cancel` policy exactly like an unexpected
cancel (`stop` latches the external-intervention quarantine).

**Moved or removed SL/TP.** Use a
`pynecore.core.broker.protection_guard.ProtectionLevelGuard` with
`report=self.report_external_intervention`:

- Before every request that sets protection — attach, amend, trailing
  activation, fail-safe stop, clear — call `guard.expect(key, levels,
  row_coid=...)`; when the venue answered, `guard.settle(key, confirmed,
  write=write_id)` with the id `expect()` returned, so overlapping writes of
  the same levels are settled exactly. A write the venue definitively refused
  settles as `guard.settle(key, None, rejected=levels, write=write_id)` and
  stops being acceptable at once; an unknown outcome settles as
  `guard.settle(key, None, write=write_id)` and stays
  acceptable until the venue is seen carrying it, or two consecutive fresh
  snapshots show another own level. Writes the plugin makes on its own count
  too: any write missing here becomes a false quarantine.
- Feed every venue observation of the protection (an order push, a
  position push, a polled snapshot) to `guard.observe(key, levels,
  tick=..., read_started_at=..., immediate=...)`. A polled mismatch needs
  two consecutive observations; a push for this exact order may pass
  `immediate=True`. A snapshot read that started before the last own write
  is skipped through `read_started_at`.
- Under a venue-side trailing stop only the trailing flag, the distance and
  the take-profit are compared.
- `guard.forget(key)` when the protection or its position is gone.
- The levels persist on the named journal row, so a restarted process
  judges the venue against the levels the previous process wrote.
- An operator resume voids every persisted write: the first observation of
  each key after the resume becomes its baseline, unless it shows a level
  the restarted bot wrote itself. The engine then re-asserts the
  strategy's own exits once through your regular amend path; expect and
  settle those writes like any other. A re-assert the venue confirms
  becomes the new baseline, a refused one leaves the operator's levels as
  the baseline, and neither is reported as an intervention. Once the venue
  confirmed the re-assert, a snapshot read after it that shows other levels
  is an edit; a push that does not match it is not judged until a view
  shows the re-assert, since it may still carry the operator's levels. When the
  restarted process stops before it completed the resume, the next one
  applies the resume again without voiding the baselines the stopped one
  adopted: only keys it never observed accept their next observation.

**After an operator resume.** The operator accepted the venue as it stood,
so nothing that changed before the resume may be reported. The guard and
the disappearance tracker handle this themselves. A row your plugin stamps
`missing_pending_since` on its own (outside the tracker's presence feed)
counts as cancelled before the resume when the stamp is written within
`RESUME_FIRST_ABSENCE_WINDOW_S` (5 minutes) of it; the tracker settles the
`missing_accepted_by_resume` mark on its next pass, so read that mark only
after the pass. A `cancelled` event your plugin reports itself for a mapped
order (an activity catch-up replaying a deletion after the restart) gets
the same verdict from the same marks: the engine audits it as
`cancel_accepted_by_resume` and applies no `on_unexpected_cancel` policy
when the row's absence stamp is the one the resume accepted, or when the
row is still marked for its first absence and the event comes within the
window. A row the engine's first open-orders read after the restart
(`get_open_orders`) lists loses that mark: the order was still on the venue
after the resume, so its later cancel is judged as usual. Return your
working orders there with their client order ids. Every other judge of
your plugin must handle it too: a quantity check of a resting order, a
close found at startup, an activity or deal-history catch-up. The runner
sets `self.operator_resume_ts_ms` (wall-clock ms, `None` without a
resume) before `connect()`. Pass the change's venue time to
`report_external_intervention(..., changed_at_ms=...)`: a change dated at
or before the resume is logged, audited as
`external_intervention_accepted_by_resume` and not quarantined. It is the
time the resume was first applied: a process that applies the same resume
again passes the same time, so a change made while no process ran is
judged. A judge that has no change time takes its first view after the
resume as its baseline, the way the guard does; keep that baseline on a
journal row, as the guard does, or a process that applies the resume again
accepts a change made while no process ran.

**Manual close.** Report a close only when the venue proves it was neither
the bot's nor the venue's: not a fill of the bot's own close order, not a
TP/SL trigger, not a stop-out, liquidation, auto-deleverage, settlement or
dealing-desk close. Book the fill as usual so the book matches the venue,
then report. A position change no fill explains is ambiguous; the engine
adopts it with an ERROR and an `UnattributedPositionChangeEvent`, never a
quarantine. A late fill of an order that was live when such a reduction
was adopted (your stream lost or delayed it) is held out of the book until
the venue position shows whether it was that reduction (dropped, never
booked twice) or a new one (booked); new entries wait meanwhile. An order
your `get_open_orders()` showed resting at the adoption, with no fill
beyond what its journal row records, cannot be in the reduction: its later
fill is booked at once, whatever other runs' exposure does to the venue
net. Keep the order's venue id on its `ExchangeOrder` and its cumulative
`filled_qty` current, so the engine can match it.

### 2. `OrderEvent.fill_qty` is INCREMENTAL; `fill_id` must be stable

`fill_qty` is the quantity of *this* fill event, not a running total —
`BrokerPosition.record_fill` **adds** it. Reporting cumulatively corrupts
the position on the second partial fill. The cumulative total belongs on
`ExchangeOrder.filled_qty`.

`fill_id` is the engine's duplicate-fill gate: the same real broker
execution must carry the same `fill_id` on **every** path that can surface
it (push event, dispatch response, poll snapshot, reconnect replay). Use
the broker-native execution/deal id. A `None` `fill_id` bypasses the gate,
so it is only allowed on paths where your own persisted `filled_qty`
cursor guarantees no other path re-emits the same execution.

A `cancelled` event of an order that executed part of its quantity first
must carry that executed quantity in `ExchangeOrder.filled_qty`, whichever
of the cancellation and the execution the venue delivers first. The core
keeps the entry's protection for the filled part on such a cancel (it never
retires the partial-bracket legs as `parent_never_arrived`), and the
execution, when it comes after the cancel, arms them. A `cancelled` or
`rejected` event of an entry reporting more executed than the fills you
delivered of that order so far puts the entry on hold until those fills are
booked: nothing is dispatched on its key, the hold is persisted and listed
in `store_ctx.fill_evidence_holds()`, and it falls under the same
`fill_report_grace_s` expiry as the hold described below. Deliver the
missing fills as usual.

**A zero proves "nothing executed" only when the venue said so.** On a
`cancelled` or `rejected` event the core reads `filled_qty == 0.0` as proof
that the entry never filled: it aborts the entry's pending legs, retires its
native fail-safe, confirms a cancel-tentative parent as cancelled and lets
the strategy place the entry again. When the venue did not report the
executed quantity for that event (the cumulative field is missing from the
push, or unparsable), set `ExchangeOrder.filled_qty_known=False`; never
substitute `0.0` as a known value. `filled_qty` is then a lower bound (what
you have seen executed, usually `0.0`).

| Terminal event of an entry                           | Core behaviour                                                                                    |
|------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `filled_qty > 0` (known or not)                      | execution proven: legs stay for the filled part, a tentative parent resolves as filled            |
| `filled_qty == 0`, `filled_qty_known=True` (default) | proven unfilled: legs aborted, mapping and intent retired, tentative cancel confirmed             |
| `filled_qty == 0`, `filled_qty_known=False`          | nothing applied: order stays mapped, legs pending, tentative parent tentative, no policy           |

In the last case you owe the core the evidence: deliver the same terminal
event again with the venue's quantity once you know it (an order-history
read), or keep the order's journal row live so your `DisappearanceTracker`
resolves it (its `confirm_missing` returns `INCONCLUSIVE` until the executed
quantity is known, and `CANCELLED` / `FILLED` with it). A cancel-tentative
parent is also resolved by the core's cancel-retry probe,
`execute_cancel_with_outcome`, which must itself answer `UNKNOWN` while the
executed quantity is unreadable. Executions reported in between are booked
and arm the legs as usual.

Until that evidence arrives the core holds the entry: no strategy cancel,
amend, OCA cascade or position cleanup retires its mapping, envelope or
pending legs, and nothing new is dispatched on its key. The hold is
persisted, so a restart re-arms it together with the order mapping your
later events route through. Deliver the evidence within your
`fill_report_grace_s`: a hold older than that puts the bot into an
`ENTRY_BLOCK` quarantine with one warning naming the entry and its orders.
While the hold stands, a reversal of the other side or a `strategy.close_all`
over the entry waits: sized from the book, it would close only the booked part
(see 2b for what happens once the hold expired). Once the wait ends, the
dispatch is sized from the true book: a price-based reversal's flip and a
`strategy.close_all` close the position the book holds at dispatch.

Once the hold expired the bot recovers the execution itself; nobody is
expected to look. It calls `fetch_order_execution(order_id, None)` for every
order of the hold, with a backoff that doubles up to five minutes and no
deadline (see 2c), and feeds the answer through the same event path as your
stream:

| Your answer                                                    | What the core does                                                                 |
|----------------------------------------------------------------|------------------------------------------------------------------------------------|
| the order's executions (`fills`)                               | books the fills it has not taken (by `fill_id`), which settles the hold            |
| totals only, nothing of the order booked                       | books one fill at `average_fill_price` with the order's `fee`                      |
| totals only, part of the order booked                          | nothing: the rest's price is not exact; asks again (return the executions)         |
| `filled_qty == 0` (known), or `order=None` (no such order)     | releases the entry as unexecuted                                                   |
| order not terminal, quantity unknown, `None`, or a raise       | asks again later                                                                   |

A fill booked from your totals (`fill_id` `<order id>:venue-total`) is kept
with the book: a later stream fill of the same order is taken up by it
instead of being booked twice. The quarantine cause of the hold ends by
itself once the hold ended and the venue position (`get_position()`, or the
net of `fetch_raw_positions`) equals the book. No operator statement
releases a hold: `fetch_order_execution` is abstract, so every plugin
answers it, and an answer that breaks its contract (see 2c) books nothing
and is asked again.

A plugin that must count the held order's execution as the run's own (in its
`get_position()`, its owned legs, its startup adoption of unjournaled legs)
reads the order ids from `RunContext.held_execution_order_ids()`: the orders
of every hold and of the venue close the core sent over an expired one. Read
it fresh on every pass; never read the hold events' payloads yourself. A leg
or deal such an order opened is the run's own execution, never a foreign or
an adopted one: adopt it into the book only through the hold's recovery,
or it is booked twice.

The core stores the evidence the moment it receives it: a terminal event
with the venue's quantity for an order a hold waits on is written to the
store before it is queued, so a process that stops before applying it
settles the hold at its restart. The stored hold and that record are
deleted only after the event's consequences were written, so a process
that dies while applying it replays it too. A terminal event reporting a
positive executed quantity for an entry held across a restart waits, with
the hold, until the script issues the entry again, and is then applied
through it.

Read the holds with `store_ctx.fill_evidence_holds()`: intent key -> the
venue order ids the hold names, scoped to the logical run. A hold listed
there names an order whose terminal event the core has not applied yet;
once the hold is gone, the core applied it or booked a fill of the order,
which is evidence of its own. A terminal event reporting a
positive executed quantity keeps the hold listed until a fill event of the
order showing that quantity is routed, so deliver that fill as usual. Do
not read the underlying store event kind directly.

After a restart you do not need to deliver an owed terminal event again:
once you delivered it, the core holds it, and replays it itself. Doing so
anyway is harmless. A copy that arrives while the replayed event is still
queued is applied after it and finds the order mapped to no intent any
more, so it is only logged; a copy that arrives after the hold was settled
does the same. An event the earlier process never yielded is still owed:
deliver it.

The default is `filled_qty_known=True`, so a plugin that always reads the
executed quantity from the venue (or from its own execution ledger) needs no
change; every constructor of a terminal `ExchangeOrder` that does not have
that quantity must set the flag.

The gate is durable. Every fill the engine takes gets a **fill receipt**
(`FillReceipt`, table `fill_receipts`, keyed by `fill_id` on the logical
run), written in the same store transaction as the binding-book write that
follows the fill's routing. A receipt therefore exists exactly when the
store holds the engine's books with the fill in them. A fill whose receipt
exists is dropped in every later process of the run too. Read it with
`store_ctx.fill_delivered(fill_id)` (`True` once the engine took the fill)
or `store_ctx.fill_receipt(fill_id)` (the receipt, with the commission the
engine accounted the fill id with in `fee`). The receipt lands after the
engine drained the event (the next sync at the latest, later only while a
storage failure delays the book write), and a fill held for the venue's
verdict (an exit fill that does not reduce the book, or one that may be an
adopted reduction reported late) gets it when the verdict settles it.

How to deliver a fill again safely:

- **Inside one process** re-yielding a fill with the same `fill_id` is
  always safe: the engine drops the second copy.
- **After a restart** the engine starts from the venue position and your
  journal. A fill whose journal write an earlier process made (its row's
  `filled_qty` raised, or the exposure it closed netted off your rows) is
  already in that position, whether or not the dead process's engine
  booked it. Keep your own durable record of every fill you journaled and
  yielded (written in the same transaction as the journal write). At
  startup, for each one: if `fill_delivered(fill_id)` is `True`, drop the
  record (yielding it again is ignored); otherwise yield it again with
  `OrderEvent.journaled_before_restart=True`, and drop the record once
  `fill_delivered` turns `True`. The engine gives such a fill its receipt
  and books nothing, because the startup already holds it, except the fee
  of an entry fill (see 2a).
- A fill your journal does **not** hold yet (the dead process never wrote
  its row or its netting, e.g. a close whose retirement it never
  committed) is a new fill to this process: journal it and yield it as an
  ordinary event, without the flag.
- Never set `journaled_before_restart` on a fill this process journaled:
  the engine would not book it.

Retention: a receipt lives as long as a journal row of the run carries the
fill's order (matched by the event's `ExchangeOrder.client_order_id` or its
venue order id `ExchangeOrder.id`), and at least `RETENTION_DAYS` after it
was written; the retention purge removes it together with that row. Keep
your redelivery records no longer than the row they belong to: once the
row is purged, so is the receipt, and a fill delivered again after that is
booked as a new one.

### 2a. A fee without its fill: `event_type='fee'`

When a fill's commission arrives without the fill, or after you already
delivered the fill without it (a fill booked from a snapshot that carried
no fee), deliver an `OrderEvent` with `event_type='fee'`:

- `fill_id`: the fill the fee belongs to (mandatory; an event without one
  is not booked).
- `fee`: that fill's **total** commission as now known, not an increment;
  `fee_currency` as on a fill.
- `fill_qty` / `fill_price`: `None`.
- `order`, `pine_id`, `from_entry`, `leg_type`: as the fill event would
  carry them.

The engine books the difference between `fee` and the commission it has
already accounted the fill id with (the receipt's `fee`, `0` without a
receipt) onto the trades the fill opened or closed, in the shares the
fill's own fee would have taken: an open trade carries it into its profit
at close, a closed trade takes it at once (net profit included). When this
process did not book the fill itself, it goes to the open trades of the
entry (`pine_id` of an entry leg), or to the latest closed trade of the
entry (`from_entry` of an exit leg). The fee event creates no trade and
moves no position. Delivering the same total again, before or after a
restart, books nothing; a fill event that later arrives with the same
`fill_id` books only the part of its own fee the fee event did not; one
whose `fee` is `0.0` counts as carrying no commission data, and the total
the fee event reported stands. To
know whether the fee is durably booked, compare `fill_receipt(fill_id).fee`
with your total. A fee of a fill the engine holds for the venue's verdict
is not booked (logged at WARNING); deliver it again once
`fill_delivered(fill_id)` is `True`.

The fee of a fill delivered again with `journaled_before_restart` takes
the same channel. The trades a restart adopts carry no commission of the
earlier process, so the fee of such an **entry** fill (its `fee`, less
what a fee event of it already accounted) is booked onto the open trades
of its entry by size, and the receipt records it, so delivering the fill
or its fee again books nothing. Two limits follow from what a restart can
recover. The fee of an **exit-leg** fill (a close, TP, SL or trailing
stop) is not booked: it belonged to the trade that fill closed, which the
earlier process held, and a restart does not rebuild that trade. Nor is
the fee of an entry fill whose entry has no open trade any more (the
position was closed meanwhile). In both cases the commission is missing
from this process's trade history; the receipt still counts it as
accounted.

`journal_fill_qty` is the same slice in the unit your journal counts the
order's `filled_qty` in: the amount this execution raises that row's
`filled_qty` by. Set it on **every** fill event (`fill_qty` set), whether or
not the order has a journal row. It differs from `fill_qty` wherever the
position moves by another figure than the journal records: a spot buy whose
fee is charged in the base coin (`fill_qty` fee-net, the row gross), an
inverse contract fill (`fill_qty` in base units, the row in contracts).
Where the two units agree, pass `fill_qty`; when you split or shrink a fill
event, split its journal quantity the same way. The engine sums it per order
(the binding book's fill cursor), and a restart resumes the stored book only
when no live or recently closed journal row of the run shows more
`filled_qty` than that sum; a fill without it is booked, reported at ERROR,
and leaves the cursor behind, so the restart adopts the position FIFO from
the journal instead.

### 2b. A close over an entry whose evidence never came: `held_execution_close`

A quarantine never leaves open a position the strategy asked to close. When an
entry's hold expired and the strategy asks to close or reverse its exposure
(`strategy.close` of the entry, `strategy.close_all`, a market
stop-and-reverse), the core reads the venue position and closes it, then the
quarantine stands until the core recovered the entry's execution (above); a
reversal's new side does not open meanwhile. An `EXTERNAL_INTERVENTION` latch still forbids every
venue write, and a `get_position()` (or `fetch_raw_positions`) that fails sends
nothing.

The close is a `CloseIntent` with `synthetic_kind='held_execution_close'`:
symbol-wide, no targets, `qty` the venue position the core read right before.
It includes an execution your journal may own no row of (the entry's order
ended without the venue's executed quantity), so it is sized against the
venue, not the journal.

- **Hedging account** (`position_port`): nothing to do. The core's one-way
  emulator closes the venue legs (`close_leg`), the unbooked execution's leg
  included.
- **Netting account**: set `supports_held_execution_close = True` on the
  plugin once `execute_close` reduces the symbol's venue position by
  `intent.qty` in a form that cannot open or grow opposite exposure (a native
  reduce-only market order, a close of the venue position by its own
  quantity), whatever your journal rows hold; journal it under
  `envelope.client_order_id(KIND_CLOSE)` like any close and report its fills
  as `LegType.CLOSE` fills of that order, labelled with the intent's
  `pine_id`. Without the flag (the default) the core sends the book's own
  symbol-wide `reversal_close` for what the book holds and tells the operator
  what is left on the venue.

The core persists the close before it sends it, never sends another one
before it read the venue again, and books its fill only up to what the book
holds; the rest closes the entry's late fill when you deliver it (or when
the core recovers it), so keep delivering that fill as usual.

**A second close after a fresh read must never over-close.** The core waits
at least `max(execute_timeout, 60 s)` after a close, and for as long as the
close's dispatch row (its `intent_key`) is still open, before it reads the
venue again; a close whose outcome is unknown is never re-sent blindly. A
plugin that journals the close without that `intent_key`, or ends the row at
once on an unknown outcome, leaves only the 60 s window: then a first close
that executes after the read can meet a second one. That is safe only
because both are bounded by the venue position at the time they execute (a
reduce-only order, a close of a position id by at most its volume, a per-leg
close): the contract above is what makes the re-send harmless, so never
send this close as a plain market order.

### 2c. The bot gets out of its own trouble: `fetch_order_execution`, causes

The bot runs unattended. A quarantine that waits for a human is right only
where a human acted on the account (`EXTERNAL_INTERVENTION`, section 1a). An
`ENTRY_BLOCK` quarantine is the bot's own trouble, and the bot must end it by
itself:

- **Read the venue's order history.** `fetch_order_execution(order_id,
  client_order_id)` is abstract: every plugin implements it (see its
  docstring for the exact contract). Return the order's venue record
  (`OrderExecution`) with its cumulative `filled_qty`, `average_fill_price`,
  `fee`, and its executions as the fill events your stream would deliver,
  with the same `fill_id`s. The core uses it for a held entry's lost
  execution (section 2) and for a partial-bracket close no journal record
  settles (looked up by `client_order_id` only). Set `fee_reported=False`
  when the venue reports no commission; return `order=None` only when the
  venue proves it has no such order. Read-only.
- **Answer in the journal's units.** The record is compared with the book
  and the journal, which count in the unit your journal rows use
  (`OrderRow.qty` and `OrderRow.filled_qty`), so:
  - `order.qty` and `order.filled_qty` are in the journal unit, as the
    order's journal row holds them;
  - every execution carries `journal_fill_qty` in the journal unit, and
    `fill_qty` in the position unit, exactly as the stream's fill event of it
    (they differ for a fee-net spot fill or a contract converted to base
    units);
  - `fee` is the commission as the stream's fill events carry it, in their
    `fee_currency`: cumulative on `order`, per execution on the fills. A
    venue that keeps only the order's totals and counts in another unit than
    the journal returns the executions instead.
- **A record that contradicts itself books nothing.** The core checks the
  answer before it books anything; a violation is logged as an ERROR that
  names the broken rule (`... breaks the BrokerPlugin.fetch_order_execution
  contract`), and the order is read again later with the same backoff. It is
  never a crash. The checks:

  | Rule                                                                | Violation example                                          |
  |---------------------------------------------------------------------|------------------------------------------------------------|
  | `order=None` comes without executions                               | executions of an order the venue does not hold             |
  | the record is of the order asked for (venue id, else client id)     | the record is of order 77, not 42                          |
  | `filled_qty` is finite, not negative, at most a set `qty`           | executed 1.5 of an order of 1.0                            |
  | every execution is a `partial` / `filled` event with a `fill_id`    | execution 'x' is not a priced fill with an id              |
  | every execution has positive `journal_fill_qty`, `fill_qty`, price  | execution 'x' is not a priced fill with an id              |
  | every execution is of the same order                                | execution 'x' is of order 77                               |
  | with `filled_qty_known`, the `journal_fill_qty` sum is `filled_qty` | the executions add up to 0.2, the order shows 0.4 executed |

- **Name the cause of every `ENTRY_BLOCK` you latch.** Pass
  `cause='<unique key>'` to `quarantine_sink`, and call
  `quarantine_release_sink(cause, resolution)` once the condition is gone
  (you closed the deal you had to close, the venue's records proved the
  outcome, the foreign deal that blocked your write is gone). The latch ends
  with its last cause, in the running process. The cause is mandatory: a
  request without one latches nothing and raises `ValueError`.
- **A cause that starts with `FENCE_CAUSE_PREFIX` (`'fence:'`, from
  `pynecore.core.broker.models`) blocks every venue write of the process**,
  exits, cancels and closes included, not only new entries. Use it only for
  a process that must not write at all any more because another instance
  owns the position (the spot manager's lost asset lease). The fence lives in
  the process memory only: it is never persisted, so the replacement
  instance that owns the position is not caught by it.

### 3. `ExchangeOrder.client_order_id` echo is mandatory

Every `ExchangeOrder` you return or embed in an event must carry the
`client_order_id` the envelope allocated. This is how post-restart
adoption maps broker-side survivors (open TP/SL legs, resting entries)
back to Pine identity. A venue that does not echo client ids (Capital.com
generates its own `dealReference`) must persist the alias instead:
`store_ctx.add_ref(coid, 'exchange_order_id', broker_id)` at dispatch,
`find_by_ref` on the way back. Without one of the two, restart adoption
silently breaks and the bot can double-open.

### 4. Override pairs and amend overrides

- A plugin that raises `BracketAttachAfterFillRejectedError` and whose
  venue can leave residual cancellable orders (unfilled partial-fill
  parent remainder, separate TP/SL entities) must override **both**
  `get_residual_orders_after_bracket_attach_reject()` **and**
  `cancel_broker_order_ref()` — the defensive-close recovery loop feeds
  the first's refs into the second. `cancel_broker_order_ref` must
  normalise "not found / already cancelled / already filled" to a benign
  no-op, raise connection faults as retryable, and propagate genuine
  rejections.
- Override `execute_cancel_with_outcome()` whenever the venue's
  post-cancel disposition is readable. The inherited default collapses
  everything to `UNKNOWN`, which keeps cancel-tentative orders resolvable
  only through broker-pushed events.
  An order that filled before the cancel answers `ALREADY_FILLED` only
  while its position is still open: the engine restores the bracket
  protection onto it. When the plugin's journal proves both the fill and
  the close of that position (booked earlier, e.g. before a restart),
  answer `FILLED_AND_CLOSED` instead: the engine resolves the cancel
  without restoring anything, booking a fill or re-arming it on restart.
  `UNKNOWN` for that case is retried; past the stale grace new entries are
  blocked and the engine keeps asking, with backoff, until you answer.
- `execute_cancel()` may answer `True` once the venue order is gone, even
  when it executed part of its quantity first: deliver that execution's
  event (and a terminal event with the executed quantity) to the engine
  before returning. The engine does not tear an entry down on a landed
  cancel while an execution of its orders is still queued: it keeps the
  mapping and the pending protection, routes the execution with the next
  drain and retires what is left afterwards. The same holds for a
  confirming `execute_cancel_with_outcome()` answer. Gating `True` on the
  engine's receipt of the fill (`store_ctx.fill_delivered`) is not needed,
  and stays correct.
- Override `modify_entry()` / `modify_exit()` with the venue's in-place
  amend whenever one exists. The inherited defaults are cancel+recreate;
  for `modify_exit` that opens a window with **no protection live**.

### 5. Lifecycle: populate `_account_id` during authentication

`connect()` (or the first authenticating call) must set
`self._account_id` to a plugin-qualified string, e.g.
`"foo-demo-1234567"`, **before** the CLI opens the broker storage run —
the run identity (`run_id`, `run_tag`, and with them the whole
idempotency chain) derives from it. Left unset, `account_id` returns the
`"default"` sentinel and every run of every account collides on one
identity. The startup probe rejects this.

### 6. Exit legs never open exposure; late fills are settled by the venue

Every exit leg (`strategy.exit` take-profit / stop-loss / trailing,
`strategy.close`) must be delivered **reduce-only end-to-end**: whatever
the venue does with it, it can only reduce the position it closes, never
open or grow one. This is what `reduce_only` declares (`NATIVE`: the
venue's reduce-only flag; `SOFTWARE`: position-attached brackets or
netting that cannot flip), and the validator refuses scripts with exits
on a plugin that does not declare it.

The engine builds on that guarantee, because `watch_orders()` events may
arrive late or out of order (a recovery backfill after an outage, a fill
surfaced by a slower namespace). An exit-leg fill that would not reduce
the book — the book is flat, or holds the fill's own side — is never
booked as a new position. The engine holds it and settles it against
`get_position()` once no fill has arrived for the fill report grace
(see below), nothing of its own is in flight and the venue size has been
stable for that long:

- venue = book: the fill was already accounted for and is dropped;
- venue = book + the fill: an earlier fill was booked out of order, and the
  book takes the venue size (parent trades rebuilt from the order ledger)
  unless another run trades the same account and symbol;
- anything else: the fill is dropped and the book stays as it is.

Until it settles, new entries are deferred (exits and closes still
dispatch), so no exposure is sized against a book that may be off by the
held fill.

So the plugin must: honour the declared reduce-only semantics on every
exit path; label a fill `LegType.ENTRY` whenever it opened exposure, never
as an exit leg; and report the venue's net size with its side from
`get_position()`.

### 6a. Journal how a close ended

The software partial-bracket engine (`partial_qty_bracket_exit =
SOFTWARE`) closes a leg's share with an `execute_close` (a `CloseIntent`
with `synthetic_kind='partial_trigger'`). It does not retire the exit's
other legs when the venue acknowledges that close: an acknowledged market
order can still end unfilled. The leg stays held until your journal shows
how the close ended, and it is settled only from that journal, under the
close's client order id (`envelope.client_order_id(KIND_CLOSE)`), across
every process of the run:

| Journal under the close's client order id                                   | Engine verdict                                                |
|-----------------------------------------------------------------------------|---------------------------------------------------------------|
| `execute_close` returns the order `FILLED` (or with its whole `qty` filled) | executed at once                                              |
| a `confirmed` event, or a closed row with `filled_qty` equal to its `qty`   | executed: the other legs cancel                               |
| `CLOSE_ENDED_PARTIALLY_EVENT` with `CLOSE_EXECUTED_QTY_KEY`                 | partly executed: the leg re-arms for the rest                 |
| `CLOSE_ENDED_UNFILLED_EVENT`                                                | nothing executed: the leg re-arms                             |
| a closed `rejected` row with nothing filled (`is_proven_refusal`)           | refused: the leg re-arms                                      |
| a live row with the venue's order id                                        | working: held, no alarm                                       |
| a live row without it                                                       | the answer is open: held                                      |
| anything else, or no row                                                    | undecidable: held, `entry_block` quarantine, operator decides |

The constants live in `pynecore.core.broker.store_helpers`:

- `CLOSE_ENDED_UNFILLED_EVENT` (`'close_ended_unfilled'`): the close's venue
  order ended (cancelled, expired, deactivated, or rejected after the venue
  accepted it) and the **venue's own** final cumulative executed quantity
  is zero. This is the uniform "ended with zero fill" mark. It is not a
  venue refusal: leave the row's state as it is (do not bend it to
  `rejected`) and close the row. Never write it from the absence of a fill
  in your own records: a fill your stream missed would then be closed a
  second time.
- `CLOSE_ENDED_PARTIALLY_EVENT` (`'close_ended_partially'`): the order ended
  with a final executed quantity above zero and below the row's `qty`. The
  payload carries it under `CLOSE_EXECUTED_QTY_KEY` (`'executed_qty'`), in
  the unit of the row's `qty` (your wire unit; the engine only uses the
  share). Write a `confirmed` event instead when the whole close executed.
- A `rejected` row proves a refusal only when the venue answered it. The
  parked-dispatch resolver also rejects a dispatch it did not find on the
  venue on enough lookups; it flags that row with
  `EXTRAS_KEY_REJECTED_ON_ABSENCE`, and `is_proven_refusal(row)` is `False`
  for it, because an order that did land can still show up executed. Use
  `is_proven_refusal` wherever your own code reads a `rejected` row as
  "nothing of it executed" (a cancel probe, a close-leg settlement).

Write each mark once, in the transaction that records the ending on the
row. A plugin without a journal row per close leaves every acknowledged
close undecidable once its process stops. On a hedging account the engine
settles the close from the close-leg rows and `close_leg_executed` instead
(see "Hedging-mode accounts").

**Declare `fill_report_grace_s`.** The same window decides when a venue
position that differs from the book counts as an external flatten or
reduction. It is the longest time a position change can show in
`get_position()` before the fill event that explains it arrives through
`watch_orders()` with a healthy stream: the venue's publication lag plus
the plugin's poll or backfill cadence for that fill. Left at `None`, the
engine uses its conservative default (120 s), sized for polled venues
whose activity feed publishes a deal up to a minute late. A push-stream
venue that delivers fills in seconds declares its own worst case, so the
bot reacts in seconds instead of minutes. Too short a value is the
dangerous direction: the engine would book a change that the late fill
then books a second time. The startup contract probe rejects anything but
`None` or a positive, finite number.

**Flag a refusal of a crossed level.** When the venue refuses an exit, an
exit amend (`modify_exit`) or a bracket because the market already trades
through its level (a long's stop at or above the bid, a take-profit
already reached), raise the refusal with `level_crossed=True`
(`ExchangeOrderRejectedError(message, level_crossed=True)`, or pass it
through from a subclass). The engine then closes the parent at market,
which is what Pine's exit would have done, instead of treating the
refusal as venue trouble (a defensive close plus an `entry_block`
quarantine after the retry, or, for a refused amend, keeping the old
levels armed). The engine also checks the last price it saw, so an
unflagged refusal is still recognised when that price shows the crossing.
A `PositionPort.amend_bracket` refusal carries the flag the same way.

**Attribute reductions in your journal.** When the venue position shrinks
and no fill of the bot explains it, the engine books the venue size into
the book and reports it, but it leaves the journal alone. Only the plugin
knows which rows the reduction hit, so it attributes the reduction to
them (lower the owned exposure, or raise the row's
`journal_exposure_retired` counter). The engine reads the journal's owned
net at startup: a venue position below it with no fill in flight is
logged as an ERROR and audited once. After an operator resume it is a
WARNING, unless a process that already applied the same resume saw a
larger position: a reduction beyond that view happened while no bot ran
and stays an ERROR. An exit fill of the bot the startup drain holds
unbooked (a TP/SL fill replayed after the restart) delays that comparison
until it settles, so net your journal for it by then. While no plugin
attributed a reported reduction, a restart does not adopt a venue position
above the one the report saw: the excess is someone else's.

**Mark a journal in wire units.** A row whose `qty` / `filled_qty` are in
the venue's contract units instead of the book's units (inverse
contracts) records the conversion price under the `anchor` extra
(`pynecore.types.strategy.JOURNAL_WIRE_QTY_ANCHOR_EXTRA_KEY`). The engine
then skips that startup comparison, since the journal and the venue are
in different units, and adopts the venue position at startup as one
parent trade instead of splitting it per entry. Startup adoption still
takes only the run's own share: the run's owned wire quantity converts to
base at the venue position's entry price (`base = wire / price`), so
`get_position()` must report such a position in base at that price.

## Capabilities: declare what you deliver end-to-end

Every `ExchangeCapabilities` field is a `CapabilityLevel`
(`UNSUPPORTED` / `SOFTWARE` / `PARTIAL_NATIVE` / `NATIVE`) — **never a
bool**. Declare what the plugin delivers *end-to-end for the script*, not
raw exchange support:

- `UNSUPPORTED` rejects scripts that need the capability at startup
  (fail fast beats failing on the first live bar).
- `NATIVE` tells the sync engine "the exchange is authoritative — suppress
  my software fallback". The engine reads this for `oca_cancel` and
  `tp_sl_bracket`; a wrong `NATIVE` there removes a safety net.
- When unsure, **under-declare** (`SOFTWARE` instead of `NATIVE`) — it
  never lies to the validator, it only leaves an engine fallback active.
- `idempotency=UNSUPPORTED` is refused for live trading: without client-id
  echo or dedup, restart/timeout retries can double-fill. If the exchange
  offers nothing, declare `SOFTWARE` and dedup locally through the store
  (the Capital.com plugin is the reference).
- `cancel_all=SOFTWARE` does **not** require overriding
  `execute_cancel_all()` — Pine `strategy.cancel_all()` is delivered by
  the engine's diff loop as per-intent cancels.
- `short_selling=NATIVE` on margin/CFD venues that hold signed positions;
  leave it `UNSUPPORTED` on spot (see "No shorting" below — it arms the
  startup reject and the engine's projected-position runtime gate).
- `sized_leg_bracket` is read only for a plugin with a `position_port`:
  `NATIVE` when `amend_bracket` can limit a leg's protection to a quantity
  below the venue position it is written on, `SOFTWARE` when the venue
  bracket always covers the whole position (see "Protecting less than the
  whole leg" below). Leaving it `UNSUPPORTED` behaves like `SOFTWARE`.

## Idempotency: the client-order-id chain

Every `execute_*` method receives a `DispatchEnvelope`; allocate exchange
client ids exclusively via `envelope.client_order_id(KIND_*)`. The formula
(`idempotency.py`) is pure and deterministic —
`{run}-{pid}-{bar}-{kind}{retry}`, up to 30 characters — so retries,
reconnects and full process restarts converge on the same id. Never
generate your own random ids: the determinism *is* the crash safety.

The engine persists first and dispatches second; on an ambiguous write
(`OrderDispositionUnknownError`) it parks the envelope and verifies
against `get_open_orders()` by client id. Your job is only to (a) send the
allocated id to the exchange, (b) echo it back (or persist the alias — see
checklist point 3), and (c) never reuse one id for two different broker
entities (distinct `KIND_*` codes exist precisely so a stop-fired market
never shares an id with the limit leg it replaces).

## Transient faults: reads park, writes park for verification

Classify connectivity faults into the broker taxonomy instead of letting
SDK exceptions escape raw:

- **Recoverable drop** → `ExchangeConnectionError`: the engine reconnects
  and retries next bar.
- **Write whose acknowledgement was lost** →
  `OrderDispositionUnknownError`: the engine parks the dispatch and
  verifies instead of blindly retrying (a blind retry risks a duplicate
  fill).

The engine has a safety net for escaped raw transients, and it is
asymmetric by design: on a per-bar **read** it parks the cycle and reads
again (reads are idempotent); on an order **write** it cannot tell a
never-sent request from a landed one, so it takes the write as one of
unknown outcome — parked and verified like an `OrderDispositionUnknownError`,
never sent again blindly — and blocks new entries (an `entry_block`
quarantine) until a reconcile passes without another such fault and the
venue holds what the book holds. A plugin that classifies explicitly keeps
the recoverable paths recovering without the entry block — the net only
exists to make contract violations safe, not to be the plan. Centralise SDK-specific
classification in `_map_exception()`.

The classified outcomes of a write are exactly these:
`ExchangeConnectionError` and its subclasses (not sent),
`ExchangeOrderRejectedError` and its subclasses and `OrderSkippedByPlugin`
(refused), `OrderDispositionUnknownError` (unknown outcome),
`BrokerManualInterventionError` (an intervention report) and a failed store
write (`sqlite3.Error`). Every other exception escaping an entry, exit,
close, cancel or amend — a raw SDK error, a `ProviderError`, an
`ExchangeRateLimitError`, an `AuthenticationError`, an `OrderSyncError`, a
bare `ValueError` — is taken as a write of unknown outcome on every dispatch
path, including the engine's own recovery writes. It never ends the run
loop and is never sent again blindly. A rate limit on a write is therefore
best reported as `ExchangeConnectionError` when the venue refused the request
before acting on it.

### A parked dispatch must be released, not only promoted

After `OrderDispositionUnknownError` the engine parks the dispatch. Its
own verification promotes an order that `get_open_orders()` echoes back,
and a complete `filled` ENTRY event for the dispatch's client order id
unparks it, whether the park is this process's or one an earlier process
left (a `partially_filled` event keeps the park, so verification can still
promote the rest). A dispatch that **never
reached the venue** matches neither, so without your help it stays parked
for the rest of the run: the entry never goes out again and every exit
that depends on it is skipped.

The rule that releases it lives in the core —
`pynecore.core.broker.parked_dispatch.ParkedDispatchResolver` — and you
supply only the venue question. Run `resolve(store_ctx, lookup)` at your
reconcile cadence; `lookup(row)` answers for one parked row with a
`ParkedVerdict`:

| Outcome        | Meaning                                                    | What the core does                              |
|----------------|------------------------------------------------------------|-------------------------------------------------|
| `LANDED`       | The venue has the order working, or it executed            | Nothing; verification / the fill path unpark it |
| `REFUSED`      | The venue has the order, dead, with nothing executed       | Rejects the row at once                         |
| `ABSENT`       | A **complete** read of the venue does not know the order   | Rejects after enough consecutive absences       |
| `INCONCLUSIVE` | The venue could not be read, or the read was truncated     | Nothing; the miss streak stays as it was        |

Never report `ABSENT` from a failed or partial read: a wrong rejection
makes the engine dispatch the intent again, and an order that did land
then fills twice. If the venue indexes its order history late, pass
`absent_after_s` so an absence only counts once the row is older than
that lag. A rejected row is closed through the journal with a
`'rejected'` resolution; the engine drops the park on its next sync and
re-dispatches the intent.

The same rule covers a dispatch an **earlier process** left pending. Your
startup recovery may not be able to resolve such a row yet (the venue
indexes the order late, or the read failed), and then the row has to stay
pending — but it may be `submitted` rather than `disposition_unknown`, and
it may have no park at all, because the crash came before the venue's
answer. The store marks every row adopted pending at startup as inherited
(`store_ctx.is_inherited_dispatch(coid)`; your own persist-first write
under the same id, `create_entry_order_row`, hands it back to the running
process), and the resolver looks inherited rows up in every pending state.
A `submitted` row of the running process is never looked up: it is
mid-flight, or the engine retries it itself.

While such a row is unresolved, the script emitting the same entry again
makes the engine rebuild the same client order id from the persisted
envelope anchor. What happens next depends on your `idempotency`
capability:

| `idempotency`             | Second send under the same id                      | What the engine does                                          |
|---------------------------|----------------------------------------------------|---------------------------------------------------------------|
| `NATIVE`                  | The venue refuses it and you adopt the first order | Sends it at once                                              |
| `PARTIAL_NATIVE`          | The venue executes it as a second order            | Parks the inherited entry rows at startup and holds the entry |
| `SOFTWARE`, `UNSUPPORTED` | No id on the venue: a second, independent order    | Parks the inherited entry rows at startup and holds the entry |

The hold is a skip with reason `inherited_dispatch_unresolved`: one WARNING
when it starts, then a reminder every 10 minutes while it stands, so a held
entry never goes unnoticed for a whole run.

Your resolver pass releases it. `LANDED` lets the engine's verification
adopt a working order from its park (record the venue refs on the row, so
your reconcile can recover a fill), and a rejection drops the anchor, so the
entry goes out under a fresh client order id. **Without a resolver pass the
entry stays held for the whole run.** So every plugin whose `idempotency` is
not `NATIVE` must provide:

- a `resolve(store_ctx, lookup)` pass at its reconcile cadence that also
  reaches inherited rows (the core selects them; you do not filter on
  `disposition_unknown` yourself — use
  `parked_dispatch.awaits_venue_answer(store_ctx, row)` if you walk rows of
  your own);
- a `lookup(row)` that answers from evidence the venue keeps of every
  accepted write. A venue without a client order id cannot answer by id:
  answer from a history that records every write causally (an activity or
  deal log), report `ABSENT` only when one complete read covers the row's
  whole life, and set `absent_after_s` to the longest publication lag you
  measured for that history;
- for `LANDED`, the same take-over your startup recovery uses, so the row
  gets its venue id and a MARKET entry's fill reaches the engine as an
  `OrderEvent` — never only a row update.
- for `LANDED` on an order whose execution your stream can **no longer
  deliver** (the order, or the position it opened, is closed since, and the
  venue does not list it any more), hand the order to the engine with
  `self.request_execution_hold(row.client_order_id, order_id)`, described
  below. The same call hands over a taken-over working order that executed
  while the bot was not running.

#### An execution the stream cannot report: `request_execution_hold`

```python
def request_execution_hold(self, client_order_id: str, order_id: str) -> bool
```

For a live entry row (`store_ctx.execution_hold_applies(row)`) whose order you
proved reached the venue as `order_id` and ended there, while the stream will
never report what it executed: the order, or the position it opened, is
closed since, and the venue does not list it any more. Two rows qualify:

- a dispatch that waits for the venue's answer
  (`store_ctx.awaits_venue_answer(row)`: parked with an unknown disposition,
  or left pending by an earlier process). The engine's verification promotes
  only an order the venue still lists open, and the fill path needs an event
  you cannot produce, so without this call the row stays pending and the
  intent's entry stays held for the rest of the run;
- a taken-over working order (the row has its venue order id) that executed
  while no process of the run was watching. After a restart nothing maps
  that order to its entry, so a terminal event of it reaches no entry, and a
  re-declared entry would go out again next to the execution. Call it from
  your startup recovery, **before** the engine's first sync; a terminal
  event without the executed quantity (`filled_qty_known=False`) is no
  substitute here, because the engine holds such an event only for an order
  it has mapped to an active entry.

When your venue keeps the order's executions, deliver them as fill events as
usual; this call is for the order whose execution you cannot deliver.

The call stores the request (`RunContext.request_execution_hold`, one keyed
event per client order id; asking again replaces it) and returns `True` once
it is stored. It returns `False`, with a warning in the log, when the run has
no store, when the row is no live entry dispatch waiting for the venue's
answer (it was settled meanwhile), or when the store write failed: then
nothing changed, and you ask again on a later pass. Call it outside every
store transaction.

The engine takes the request at its next sync, in one transaction: a pending
row gets `order_id` and the `confirmed` state (the send is not repeated, and
`inherited_dispatch_in_flight` no longer holds the entry), a park of the row
is retired, and the entry gets the fill-evidence hold over `order_id`,
already expired, with or without its Pine intent active; a re-declared entry
adopts the held order instead of being sent afresh. A restart before or after
that take-over holds the entry the same way. From then on the held-execution
rules apply unchanged:

- new entries are blocked under the cause `fill_evidence:<intent key>`, and
  the engine reads the order's execution with
  `fetch_order_execution(order_id, None)` (backoff up to 5 minutes, no
  deadline) until it can book it; a venue that holds no such order releases
  the entry as unexecuted;
- the execution is booked with the venue's figures, and the block ends once
  the book holds what the order executed and the venue holds what the book
  holds (see below for the position that closed since);
- while the hold stands, the strategy cannot cancel or re-send the entry,
  and a close or reversal it asks for over it closes the venue position as
  read from the venue, never more. An entry the strategy no longer emits is
  booked all the same: its execution is the venue's, not the script's.

`store_ctx.held_execution_order_ids()` lists `order_id` from the take-over
until the execution is booked; your reconcile must not take the order's
absence from the venue for an intervention while it is listed.

**Your duty: deliver the close of the position that closed since.** The
execution you hand over opened a position, and the venue holds less of it
now (the deal was closed by a protective leg, a stop-out, or by anyone).
Deliver that close as the stream would have: `CLOSE` fill events labelled
with the entry (`pine_id`, or `from_entry`), with the venue's close price,
quantity, `journal_fill_qty`, commission and its own `fill_id`. Send them
whenever you know them: a close fill that arrives while the entry's
execution is still being read from the venue waits in the engine and closes
the entry once it is booked, so the order of the two does not matter. The
round trip is then booked with both real prices.

When no close fill reaches the engine, the engine still ends in a
consistent state by itself, in the running process: once the fill-report
grace passed with the venue steady below the book, nothing of the bot in
flight and no fill routed, it closes the handed-over exposure the venue no
longer holds at the entries' average price, through the same provisional
close it books an unexplained reduction with. The trades are closed, not
wiped; the realized P&L is the commission only and approximate; one WARNING
is logged and the close-out is audited (`held_execution_close_adopted`),
with no `UnattributedPositionChangeEvent` and no "no fill explains it"
ERROR, since the reduction is the one the hand-over announced. A close fill
you deliver later is not booked a second time: once the venue confirms it
was that close, it re-prices the close-out at its own price and fee. A
process that restarts before the close-out resumes the booked position from
the journal, and its startup reconcile adopts the venue as it does for every
resumed position.

If the venue cannot prove either outcome for a row (its history no longer
reaches back to the row's dispatch), answer `INCONCLUSIVE` and log why: the
entry stays held, which is the safe side. `pynecore-capitalcom` is the
reference: rows with a `dealReference` are answered from the deal
confirmation, all of them from `/history/activity` matched on epic,
direction, size and a ±3 s window around the row's creation, with
`absent_after_s=fill_report_grace_s` (the venue has published an activity
61 s after its deal).

The rule is the same for an entry whose stop fired while a both-set
`strategy.entry` waited on its limit: the engine sends the stop-triggered
MARKET under a client order id it persisted first. On a venue that is not
`NATIVE`, a restart, or an unknown disposition inside the run, does not
send that MARKET again until the journal row of its earlier send settles:
a row still waiting for the venue holds it (with the same WARNING and
reminders), a landed one settles the watch on that order without a second
send, and only a send that provably did not land (no row, a refused row with
nothing filled, or a `submitted` row of the running process) goes out
again.

A venue without a client order id may also resolve from evidence of its
own outside the resolver and write the resolution itself with
`store_ctx.record_resolution(coid, 'attached' | 'rejected')`
(`pynecore-capitalcom` does this from the deal confirmation).

### A fill you recover must reach the engine

When your reconcile proves a fill the stream missed, emit it as an
`OrderEvent` — never only correct your own rows. While the recovery is
still running, override `fill_recovery_pending(entry_ids)` to say so if
the default (the disappearance tracker's stamps) does not cover your
case: the engine then holds its "someone else changed the position"
verdict, for a bounded time, so your own exit is not booked as an
external close.

The general operational rule: **the bot runs unattended and never stops
itself.** Prefer keep-running behaviours (retry, observe, degrade) on every
recoverable path. Where continuing could lose money, raise
`BrokerManualInterventionError` or latch an `entry_block` with a `cause`
you end yourself: the engine never ends the process on it. It latches an
`entry_block` that ends once a reconcile passes without another report and
the venue holds what the book holds (an `UnexpectedCancelError` latches the
external-intervention quarantine instead), keeps exits, closes and cancels
running, and parks a write the report interrupted as one of unknown outcome.
Only an external intervention waits for the operator.

## Optional hooks

These are not abstract, so nothing forces you to notice them. Each one
matters only for the venue shape it names.

- **`publish_native_failsafe_sl(snapshot)`** — for a venue whose protection
  is a position attribute (`tp_sl_bracket=NATIVE`) running engine-driven
  partial brackets (`partial_qty_bracket_exit=SOFTWARE`). Those brackets
  are only as alive as the bot, so the engine keeps one worst-case stop
  for the whole position on the venue and calls this coroutine to write
  it. Set the position's protection to the snapshot (full replacement)
  and return, or raise when the venue did not take it; the engine records
  a return as a successful write and any exception as a failed one. Left
  undefined, the fail-safe is state-only and nothing protects the
  position on the venue while the bot is down. The write changes
  protection levels, so it follows the protection guard's rules (1b).
- **`native_cancel_all_expected_sink`** — if `execute_cancel_all()` calls
  one native bulk-cancel endpoint, call this sink **before** the venue
  request. The bulk cancel bypasses the engine's per-order cancel
  intents, so the `CANCELLED` pushes that follow would otherwise read as
  external cancels and quarantine the bot. It is `None` on state-only
  test paths; guard for that.
- **`least_tradable_qty(price)`** — the smallest quantity one venue order
  can trade. The position book never leaves a partially closed trade a
  remainder below it while another trade is open, because no order could
  close that remainder on its own. Override it when the venue has a
  minimum order size and you support pyramiding; it is called while a
  fill is booked, so it must not do I/O.
- **`operator_resume_ts_ms`** — after an operator resume, a change the
  venue made before this time is no intervention any more. The core
  applies the boundary to the protection guard and the disappearance
  tracker. A judge of your own must apply it too: pass the change's venue
  time as `changed_at_ms` to `report_external_intervention()` (an
  execution, a closing deal and an order amend all carry one), or, for a
  judge without a change time, take its first view after the resume as
  the new baseline.

## Hedging-mode accounts: `PositionPort`

Pine strategy semantics are one-way; a hedging-mode account holds multiple
broker legs per symbol. Do not emulate netting inside the plugin — opt
into the core `OneWayEmulator` by setting `self.position_port = self` once
you know the account is hedging-mode (leave it `None` for netting
accounts), and implement the eight `PositionPort` primitives
(`fetch_raw_positions`, `get_volume_quantizer`, `close_leg`,
`close_leg_executed`, `reject_out_of_range`, `place_leg`, `amend_bracket`,
`bracket_matches`).
Each primitive touches exactly ONE broker entity; the emulator owns all
FIFO/netting and crash-replay logic.

On a hedging account `fetch_raw_positions()` returns the legs this run
opened, and `get_position()` their net. Another party's leg on the same
symbol is not the bot's position and not an intervention; counted in, its
close would read as a reduction of the bot's share.

Fill `PositionLeg.entry_id` with the Pine entry id whose order opened
the leg whenever your own records tie the leg to that one entry (the
entry's dispatch matched to the leg through its fill or the position id
the venue returned). Leave it `None` for a leg that aggregates several
entries (one leg per side), for a leg the bot did not open itself
(manual, adopted, or opened by a run whose records are gone), and
whenever the mapping is uncertain. The emulator replicates a
`strategy.exit(from_entry=...)` bracket onto that entry's legs plus
every unattributed leg, so each pyramided entry keeps its own levels on a
venue with one leg per entry. A leg owned by several exits gets the merge
of their levels in one `amend_bracket` call (a field only one exit sets
is kept; a field several exits set differently gets the level the price
reaches first), and retiring an exit re-amends its legs to the merge of
the exits that still own them, clearing a leg only when none is left.
Treat every `amend_bracket` call as the leg's complete protection. A
fixed `sl_price` and a `trail_offset` in one call may come from two
different exits; write the pair the way your venue defines it (the core
logs that combination as a warning). The `coid` of an `amend_bracket`
call is a correlation id only: a retry, the per-sync drain or a restart
replay can send different levels under a coid used before, so never
dedup or skip an amend by its coid, always write the levels you receive.
A re-amend with the levels the leg already holds must be harmless. The
optional
`supports_partial_leg_close = False` attribute makes the emulator close
the whole leg wherever a plan would slice one, on venues whose position
close is full-row only (Capital.com). The script asked to reduce risk, so
the position ends up smaller than asked, never larger; each widened close
is logged at WARNING with the asked and the closed quantity.

A MARKET reversal on a hedging account closes the opposing legs and opens
only the residual in the order's own direction. `close_leg` only
dispatches a close, and a close can still execute in part (a reduce-only
IOC market close cancelled after a partial fill), so the core opens the
residual only once a fresh `fetch_raw_positions()` read shows no opposing
leg. If the read right after the closes still lists one, the residual is
not opened in that dispatch: on each following sync the core asks
`close_leg_executed` what the close executed, re-sends only the part that
provably did not, and calls `place_leg` for the residual (under the
reversal's own entry coid) on the first sync whose read shows
the opposing legs gone. So `fetch_raw_positions()` must report a leg as
long as the venue still holds any of it: a leg dropped from the read
before its close executed opens the residual beside it.

A resting LIMIT or STOP entry is placed with `place_leg` as it is, with no
close: it rests as a leg of its own. When it fills against opposing legs,
the core nets the account afterwards: it closes the opposing legs (oldest
first) and the new leg by the quantity the fill closed in the book, with
`close_leg` calls under the same persist-first close-leg rows and
`close_leg_executed` settlement as a reversal. The net position stays as
it is, so the core books the fills of these closes as no position move
(only their fee reaches the books): deliver them like any other close-leg
fill (`LegType.CLOSE`, the close's side and quantity, a `fill_id`).

Put the `coid` your `close_leg` received on the close fill's
`ExchangeOrder.client_order_id`, in the core's form (map a wire form such
as a sanitized or shortened venue label back to it). The core tells a
netting close's fill from a strategy close's fill on the same side by that
id, also when the fill comes late. A close fill whose id names no close of
the run is taken by its side; when another close of the run went out on
that side while the netting was in flight, the core cannot tell the two
apart: it books the fill as an ordinary close and quarantines the run
(`ENTRY_BLOCK`, reason `netting_close_fill_unidentified`) so the operator
checks the books against the venue.

A fill on a position port that carries no `fill_id` (allowed where your
persisted `filled_qty` cursor guarantees single delivery) is keyed by the
core from its leg type, side, order reference (`client_order_id`, else
`id`) and the order's cumulative `ExchangeOrder.filled_qty` after the
slice: report that cumulative quantity on every such fill, never the
slice. A resting entry fill the core cannot key that way (no `fill_id` and
no order reference) cannot be netted: the core quarantines the run
(`ENTRY_BLOCK`, reason `netting_fill_without_id`) and the operator nets the
sides. On a port without partial
leg closes the netting closes whole legs; where that cannot keep the net
position on its side and no larger, nothing is sent and both sides stay
open (logged at WARNING). A restart while a netting is in flight adopts
the journal's net, not the venue's (which is off until both closes have
landed), so journal a close fill on your rows the same way you journal
any other close of the run.

### Tell the restart replay what a leg close executed

A crash between a `close_leg` call and its return, a timeout, or a guard
re-firing on a close already sent leaves the close's outcome unknown: it
may have executed in full, in part, or not at all. The core never re-sends
such a close under a `coid` it already used, and it never assumes your
venue dedups a re-send. Before it sends anything it asks
`close_leg_executed(symbol, leg_id, coids, since_ms)`: `coids` are every
client order id the core recorded for that close, oldest first, and
`since_ms` is when the first one was about to be sent (local clock). The
core also compares the leg's size with the size it recorded at dispatch,
and re-sends only the part neither measure accounts for, under a new
`coid`.

There are exactly three answers, and each must come from venue evidence:

- **A positive volume**: the broker-grid volume (the unit of
  `close_leg`'s `volume`) those closes provably executed, once none of
  them can execute any more.
- **`0`**: the venue proves none of them executed and none can execute any
  more.
- **`None`** (cannot tell yet): a close still working, a read that failed,
  timed out or came back short, or an empty read made before the venue's
  own indexing or publish delay has passed. The core sends nothing, keeps
  the leg's protection, and asks again on the next sync.

Obligations that follow from these answers:

- The core records each `coid` before it calls `close_leg`, so a crash can
  leave a `coid` that never reached your plugin or the venue. Once the
  venue's records provably cover `since_ms` and hold nothing for it,
  answer `0` for it. Answering `None` forever leaves the close
  undecidable and the leg's close stuck.
- Judge every `coid` by what the venue holds, never only by your own local
  records. The send may come from an earlier run instance of the bot, or
  your local row for it may be gone to retention; neither proves the venue
  never received it.
- An empty venue read is evidence of absence only after the venue's own
  indexing or publish delay for `since_ms` has passed, and only when the
  read covers `since_ms`. Before that, answer `None`. A history that ages
  records out cannot prove absence for a send older than its window.
- A failed, timed-out or partial read is always `None`, never `0`.
- If your records cannot tell these closes from another reduction of the
  same leg after `since_ms` (the venue's close request carries no client
  id), count that reduction too. Counting too much leaves the leg less
  reduced, which the reconcile settles; counting too little closes it
  twice.
- On a venue whose position close is full-row only, a row still open
  proves its close did not execute, once no close of it can still be
  working.
- A re-send's `coid` carries a suffix after the leg close's own id; map it
  back with `close_leg_row_client_order_id` when you look up the core's
  close-leg row in `close_leg`.

The same answer settles a software partial-bracket close fanned out over
the legs: the engine holds the leg until `close_leg_executed` answers for
every close-leg row of the fan (all executed: done; none: the leg re-arms;
anything between: the leg re-arms for the rest). Answer `None` while a send
can still execute, never a guess.

### Report the outcome of `amend_bracket` truthfully

The core writes a leg again only when it knows what became of the leg's
previous write. A write that timed out can still reach the venue later;
if a newer write went out in between, the late one replaces it and the
position is left on an older stop. So the outcome of every
`amend_bracket` call is part of the contract:

| What happened                                                  | Raise                                                                    |
|----------------------------------------------------------------|--------------------------------------------------------------------------|
| The venue confirmed the write                                  | nothing (return)                                                         |
| The venue answered and did not apply it                        | `ExchangeOrderRejectedError` (or a subclass), `ExchangeRateLimitError`   |
| The request never left the client                              | `ExchangeConnectionError`                                                |
| The request may have reached the venue and no answer came back | `OrderDispositionUnknownError`                                           |

Never report the last case as a rejection, not even on an attach: the
core treats a rejection as a verdict and may send the leg something else
at once.

After an unknown outcome (and while a call the engine stopped waiting for
is still running) the emulator holds the leg: no different write goes
out. Only the identical write may be repeated, because whichever of the
two arrives last leaves the same protection. On each following sync the
emulator asks `bracket_matches(symbol, leg_id, side=, tp_price=,
sl_price=, trail_offset=, qty=)` whether the leg holds exactly the
protection that write carries:

- Read the leg's protection from the venue. Do not answer from a cache
  your own writes feed.
- Compare after your own conversion to the broker grid (tick rounding).
- All-`None` levels ask whether the leg carries no protection at all.
- A trailing stop the venue has moved still matches: compare what
  identifies the write (the trailing flag or distance, the take-profit),
  not the stop price the venue trails.
- A leg that is no longer open matches nothing: return `False`.
- When the venue does not report what tells the write from the leg's
  earlier protection, return `None` ("cannot tell"). The typical case is a
  trailing stop whose distance the venue does not return: an amend from
  one trailing stop to another with the take-profit kept looks the same
  whether it landed or not. Return `None` only when every level the read
  does report agrees; a level that differs is a `False`.
- A failed read raises like `fetch_raw_positions`. Never guess.

`True` settles the write as landed. With `False` the leg stays held for
`UNSETTLED_WRITE_WINDOW_S` (30 s, counted from the first such read),
after which the write counts as lost and the leg is written again. That
window is a core assumption, not a measured property of any venue.

`None` neither confirms the write nor counts it as lost. The leg stays
held for the same window and nothing is repeated in it. After the window
the leg is written once more with the levels its exits ask for (the same
write when they did not change), and that write's own outcome settles the
leg. If an identical repeat the venue confirmed went out inside the
window, that repeat already re-asserted the levels and nothing more is
sent. A re-sent trailing write carries the stop and distance the exits ask
for, like any amend: on a venue that takes the stop from the amend, a
trailing stop the venue has moved since is written back to that level.

The same rule holds off the emulator: after an `execute_exit` or
`modify_exit` that raised `OrderDispositionUnknownError`, the engine
sends no further `modify_exit` for that exit until the parked dispatch
is resolved or the same window has passed.

### Protecting less than the whole leg

A bracket on a hedging venue is usually an attribute of the whole
position: when it fires, the whole position closes. Two cases need less:

- **A partly offset leg.** A long leg of 10 beside an opposing leg of 4 is
  a net long 6. A bracket on the whole leg closes 10 and leaves the book
  short 4.
- **A shared venue position.** On a venue with one position per side, the
  position can hold exposure another run or a manual trade opened. Report
  the run's own share as `PositionLeg.qty` and the venue position's size
  as `PositionLeg.venue_qty` (leave it `None` when they are equal). A
  bracket on the whole position would close the other party's exposure
  too.

Either case can start or end while a bracket is already on the leg. While
any leg carries protection, the core re-reads the legs with
`fetch_raw_positions` once per sync and re-decides the protection of a
leg whose share changed: a whole-position bracket is cleared (after the
software guard below is armed) once the leg is shared, and written again
(before the guard is dropped) once it is not. A switch whose write did
not go through is tried again on the next sync. Report `venue_qty` on every
read, not only on the first one after a fill.

Every `amend_bracket` call therefore carries `qty`, the quantity the
protection may close. What the core does with a leg that needs less than
its venue position protected depends on `sized_leg_bracket`:

- `NATIVE`: the call carries the smaller `qty`, and the protection you
  write must close at most that much.
- anything else: the core never writes a venue bracket on such a leg
  (and clears one it wrote there before). It guards the leg in software:
  it watches the price and, when a level is reached, closes exactly the
  protected quantity with `close_leg`. `qty` in your `amend_bracket` is
  then always the whole venue position, and you may ignore it.

The software guard runs only while the bot runs. Its trailing stop is
persisted on the leg's ownership rows each time it moves, so a restart
resumes it from the last persisted extreme, never looser; it trails on from
there with the next prices. The core logs each guarded leg once at WARNING. On a port with
`supports_partial_leg_close = False` the guard cannot close a part of a
leg either, so a partly offset leg there carries no protection until the
opposing leg is closed (logged once at ERROR). Declare `NATIVE` only for
behaviour you measured on the venue.

## Spot venues: opt into the core inventory layer

Spot venues expose **no position object** — no position row, no entry
price, no unrealized P&L, no position id, no position-attached bracket.
The position itself is not missing: the base-asset inventory relative to
the quote asset *is* the long exposure. The core
`pynecore.core.broker.spot_inventory` module owns this bookkeeping; a
spot plugin opts in by setting `self.spot_inventory_port = self` and
implementing the `SpotInventoryPort` surface, then drives one
`SpotInventoryManager` per run (the Bybit plugin's spot category is the
reference wiring):

- **Construction** passes both quarantine sinks:
  `request_quarantine=self.quarantine_sink` and
  `release_quarantine=self.quarantine_release_sink`. The manager latches its
  own trouble (a failed or inconclusive measurement, a negative ledger, a
  fill id another run booked, the asset lease another live run holds) as an
  `ENTRY_BLOCK` under the cause `spot_inventory:<product id>`, measures the
  venue again on every `reconcile` while it holds (paced from 5 s up to
  5 min, no deadline), and releases the cause once a measurement settles.
  Without the release sink the engine's entry block holds until a restart.
  Someone taking the bot's own base is the `EXTERNAL_INTERVENTION` latch,
  which only the operator resumes. A lease a replacement instance took over
  fences the old (zombie) process under the cause
  `fence:spot_inventory:<product id>`: it writes nothing to the venue any
  more, no order, amend, cancel or close, and never resumes; the replacement
  runs the bot. A fill id another run booked is a configuration conflict (two
  runs trade the same account and base): the manager blocks new entries and
  keeps measuring, exits and closes keep running, and the log and the audit
  name both runs and the shared fill id.
- **`startup()`** once after authentication, before the engine's startup
  reconcile — asset-ownership lease claim, execution catch-up from the
  durable cursor, epoch/invariant validation, adoption watermark. A failed
  or inconclusive read latches the entry block before a single dispatch; a
  drift is left to the runtime verdict.
- **`record_live_fill()`** for every fill observed on the live stream,
  BEFORE emitting the corresponding `OrderEvent` — the return value is
  the emit gate (a replayed fill id returns `False` and must not be
  re-emitted). The ledger insert and the outbox flip commit in one
  transaction; a crash between commit and emit loses the event, not the
  fill (the next startup folds the row into the adopted position —
  exactly-once, adoption path).
- **`reconcile(now_ms)`** per poll cycle — lease heartbeat plus the
  balance invariant with a persisted settlement-grace state machine. It
  **returns** the ledger rows a runtime catch-up recovered this cycle (a
  stream-gap fill), already flipped to `delivered`; emit an `OrderEvent`
  for each. The periodic engine reconcile ignores position *increases*,
  so a recovered fill left un-emitted would stay invisible to the
  strategy until the next restart's adoption. The rows an in-process
  recovery books come back the same way. The lease heartbeat is
  fenced by the physical run instance: if a replacement instance took
  the lease over while this process was silent, `reconcile` latches the
  write fence instead of trading on a lease it no longer holds.
- **`consume_pending_halt()`** after each `reconcile`: a conflict whose
  quarantine sink was missing or raised arms this signal once; raise it on
  the engine's channel (the event stream) and the engine latches the
  quarantine it carries. Nothing halts: the process stays alive.
- **`synthesize_position(mark)`** from `get_position()`.

### The port surface

`SpotInventoryPort` carries the venue facts (`product_id`, `base_asset`,
`quote_asset`, `cursor_scope`, `base_tolerance`, `settlement_grace_s`)
and two reads:

- `fetch_executions(cursor)` — the **bot's** execution history from the
  durable cursor, paged (`SpotExecutionBatch`). Return only fills
  attributable to this bot: every `SpotExecution` must carry the bot's
  own `client_order_id` (the core rejects an execution without one; a raw
  `exchange_order_id` is not proof of bot ownership — a manual/web trade
  carries one too). Map the fill's venue order id to the bot's
  client_order_id via the plugin's own order records when the venue does
  not echo it. The ledger tracks the bot's inventory, not the account's —
  a manual or foreign trade folded in would move the balance *and* the
  ledger together, so the invariant would never fire. A venue whose
  account-history endpoint cannot be filtered/attributed to bot orders
  MUST return `conclusive=False` (which fails closed) rather than dump
  raw account trades. `cursor=None` means "no
  history belongs to the bot yet": return an EMPTY batch anchored at the
  venue's current watermark — the account's prior history is foreign
  inventory and belongs to the epoch baseline, not the ledger. A
  time-scoped API must serve an overlapping window (the ledger dedups on
  fill id) rather than assume a strict cursor. Return `conclusive=False`
  when the venue answered but cannot be trusted as complete.
- `fetch_base_balance()` — the account's **TOTAL owned** base amount:
  available PLUS locked in open sell orders PLUS pending settlement. An
  available-only read false-fires the invariant the moment a sell order
  rests.

Executions are exact `Decimal` arithmetic end to end (the ledger stores
canonical decimal strings — float accumulation over crypto atoms is not
acceptable). The canonical delta equations: `base_delta` = signed
executed base minus any BASE-currency fee; `quote_delta` =
opposite-signed notional minus any QUOTE-currency fee; a third-currency
fee is recorded but touches neither delta. Set `SpotExecution.venue_seq`
to the venue's monotonic execution-sequence number when the venue
exposes one; the fold orders by `(ts_ms, venue_seq, fill_id)`, so a
venue whose fills can share a millisecond MUST provide it or a same-ms
buy/sell pair may replay reversed into a false oversell. `settlement_grace_s`
must be a finite, non-negative real number (a NaN/inf grace would let a
confirmed conflict stay pending forever).

### `get_position()` must never return `None` while holding inventory

The engine treats `None` as an authoritative flat: the startup reconcile
adopts it unconditionally and the periodic reconcile reads a
held-position → `None` transition as an external flatten. A spot plugin
that returns `None` because "the venue has no positions" silently breaks
restart adoption — after a restart the engine believes it is flat and
**double-opens** on the first entry signal. Return
`synthesize_position(mark)` instead: `size` = net base inventory from
the ledger, `entry_price` = ledger VWAP (proportional basis reduction on
partial sells), `unrealized_pnl` = (mark − VWAP) × size, `leverage=1.0`,
`liquidation_price=None`, `margin_mode="cash"`. It returns `None` only
when the bot's net inventory is genuinely flat.

### The ledger is retention-exempt

The core `orders` table keeps only a cumulative `filled_qty` (no
per-fill price, fee, fee currency or execution id), and the `events`
table is purged by the retention cleanup — neither can reconstruct a
position that has been open longer than the retention window. The
`spot_executions` ledger is exempt from `cleanup_old_data` by design: an
open position stays reconstructible for as long as it is open, however
old its fills are.

### The balance is pooled — the bot's own share decides

The venue balance does not separate the bot's inventory from
pre-existing holdings, manual trades or deposits. The ledger is
authoritative for the bot's inventory; the venue balance is checked
against `expected_total = foreign_baseline + bot_inventory(ledger)`,
where the baseline was frozen at epoch creation. The numeric tolerance
(`base_tolerance`) covers ONLY asset quantization; settlement lag is a
*temporal* state (a persisted pending-conflict grace re-checked with
fresh executions and balance), never a wider tolerance. While the bot's
own live orders cover a drift, the verdict waits for their fills.

A drift still unexplained past the grace is judged against the bot's
**own** inventory, because base is fungible:

- **The balance still holds at least the bot's inventory** (a deposit, a
  foreign buy, a sale out of the foreign holdings): someone else's
  separate activity left the bot's share intact. The manager logs an
  ERROR, writes a durable `spot_foreign_holdings_reanchored` audit event,
  re-anchors the foreign baseline on the balance (a new epoch, the bot's
  ledger untouched) and the bot keeps trading. No quarantine.
- **The balance is below the bot's inventory and the history read was
  complete**: someone sold or withdrew what the bot holds. The manager
  latches the `EXTERNAL_INTERVENTION` quarantine; there is no adoption
  path. After the operator resume
  the first start keeps the part of the bot's inventory the account still
  holds.
- **The balance is below the bot's inventory but the history read was
  incomplete** (`conclusive=False`): a fill of the bot's own cannot be
  ruled out. The manager logs an ERROR and writes a durable
  `spot_inventory_shortfall_unverified` audit event once, books nothing,
  latches nothing, and keeps the conflict pending until a complete read
  decides it — booking the shortfall now would take an own fill the
  history returns later from the bot's inventory a second time.

The startup does not judge a drift. The venue can index a fill of the
bot's own (an order that rested while the bot was down) behind its
balance, and the startup cannot wait for it, so it adopts the ledger,
keeps the execution cursor where its catch-up began (the next read
returns such a fill), writes a durable `spot_startup_drift_pending` event,
and leaves the verdict to the runtime reconcile: the same grace and the
same three outcomes as above. Its history read is settled against the
balance read: the history is read again after the balance, so an own fill
landing between the two is in both.

### One position asset, one bot

Every asset serves exactly one role per account. As a **position asset**
it is the exclusive base of exactly one bot — no other bot may touch it
as base *or* quote (a BTCUSDT bot and an ETHBTC bot conflict: ETHBTC
moves BTC as its quote). As a **cash asset** it is a shared quote pool
for any number of bots (BTCUSDT + ETHUSDT is fine); cash exhaustion is a
normal recoverable order reject, not bookkeeping corruption. The
`spot_asset_owner` lease enforces this within one broker store: the
claim rejects a base-vs-quote overlap with a live run up front (so the
conflict is caught before either bot trades, not surfaced later as a
spurious drift quarantine), and a concurrent second run on the same base
starts quarantined. The lease is keyed on the physical run instance, so
a resumed zombie sharing the reused `run_id` cannot steal its lease back
from the replacement instance — its next heartbeat reports the loss and
the process stops writing to the venue for good. An instance on another workdir or machine is only
*detected* by the balance invariant, not prevented — a dedicated
(sub)account per bot per base asset remains the recommended mode. It
makes drift conflicts rare, and every conflict that still occurs is
real.

### No shorting

Spot cannot go short — leave `short_selling` at its `UNSUPPORTED`
default (a spot inventory port combined with a supported `short_selling`
declaration is a contract error). The core enforces the rest on two
levels:

- **Startup**: a script that passes a syntactically constant
  `strategy.short` direction to `strategy.entry` / `strategy.order` is
  rejected by `validate_at_startup` (`ScriptRequirements.may_go_short`,
  detected at compile time). A dynamic direction cannot be proven and
  passes this static gate.
- **Runtime**: the sync engine preflights every sell-side entry dispatch
  (and every entry amend) against the projected aggregate position —
  current position minus every active sell-side entry intent's quantity,
  judged after the market stop-and-reverse fold. A `strategy.order` sell
  that only reduces an existing long flows through; a dispatch that
  would project negative is refused like a venue reject (nothing is sent,
  the Pine order sees no fill, an identical re-emission is silenced after
  a few refusals), and a stop-and-reverse still runs its close leg. Exits
  and closes are reduce-only by engine contract and are not gated.

Margin/CFD venues that hold signed positions natively declare
`short_selling=NATIVE` (both reference plugins do), which disables both
gates.

## Runtime configuration

- Plugin credentials/tunables live in the plugin's own `Config` dataclass
  (self-healing TOML at `workdir/config/plugins/<name>.toml`).
- **Broker-agnostic policies do not belong in plugin configs.** The
  cross-broker `workdir/config/brokers.toml` (`BrokerDefaults`) owns them;
  today that is `on_unexpected_cancel` (`stop` / `re_place` / `ignore`)
  and `on_inventory_conflict`
  (`quarantine` only, spot venues only — deliberately narrower:
  `re_place` would buy back an operator's withdrawal and `ignore` would
  trade on corrupt books), injected onto the plugin instance by the CLI
  before the runner starts.
- Do not expose internal cadence/backoff constants as user config
  ("no internal tunables" policy) — keep them module constants.

## The `--broker` startup sequence

What runs, in order, when the user starts `pyne run --broker`:

1. Provider plugin is loaded and instantiated from the provider string;
   historical data download drives **authentication** (this is where
   `_account_id` must get populated).
2. The instance is gated: it must be a `BrokerPlugin`.
3. `BrokerDefaults` (`brokers.toml`) values are injected.
4. **`validate_plugin_contract()`** probes the plugin (see below) —
   errors abort startup, warnings go to the broker log.
5. The broker event loop thread starts.
6. `BrokerStore.open_run()` registers the run — the `RunIdentity` derives
   from `account_id` here.
7. The script runner starts: `get_capabilities()` is validated against
   the script's `ScriptRequirements` (`validate_at_startup`) — scripts
   needing unsupported capabilities are refused.
8. A one-shot `get_balance()` auth probe fails fast on bad credentials.
9. The engine starts broker I/O: the `watch_orders()` stream task and the
   startup reconcile (restart adoption of surviving broker state).

## `validate_plugin_contract()` — what is enforced at startup

The probe (`pynecore.core.broker.validation.validate_plugin_contract`)
turns the machine-checkable slice of this guide into fail-fast errors:

| Check                                                                   | Severity |
|-------------------------------------------------------------------------|----------|
| Residual-enumerator override without `cancel_broker_order_ref`          | error    |
| Any capability field that is not a `CapabilityLevel`                    | error    |
| `idempotency=UNSUPPORTED`                                               | error    |
| Supported `watch_orders` capability without method override             | error    |
| `NATIVE` / `PARTIAL_NATIVE` `amend_order` without a `modify_*` override | error    |
| Non-`None` `position_port` missing port methods                         | error    |
| Non-`None` `spot_inventory_port` missing port members                   | error    |
| Spot port set with an invalid `on_inventory_conflict` policy            | error    |
| Spot port set with a supported `short_selling` capability               | error    |
| `account_id` still `"default"` after authentication                     | error    |
| Inherited `backfill_closed_bars()` (every missed close reads "no bar")  | error    |
| `history_publish_lag` not a positive finite number                      | error    |
| `history_has_idle_bars` not a `bool`                                    | error    |
| `history_revises_before_lag` not a `bool`                               | error    |
| `stream_closes_match_history` not a `bool`                              | error    |
| `feed_timeout_bars` neither `None` nor a positive `int`                 | error    |
| No `watch_orders()` stream (poll-only, no disappearance channel)        | warning  |
| Inherited `execute_cancel_with_outcome` (all cancels `UNKNOWN`)         | warning  |

A plugin usually learns its account's mode only in `connect()` or on its
first broker call, so it sets `position_port` / `spot_inventory_port`
after the startup probe ran. The order sync engine therefore runs the
port checks again (`validate_plugin_ports`) the first time it sees a
port. A port that fails them cannot be driven: the bot stays alive, books
what the venue reports and sends nothing (an `entry_block` quarantine with
every venue write refused) until a probe passes; fix the plugin and restart
it.

A conforming plugin produces zero findings — all three reference plugins
do, and each asserts it in its own test suite (`test_*_contract.py`).
Everything the probe cannot check (incremental `fill_qty`, stable
`fill_id`, client-id echo, actual disappearance detection quality) is on
this guide's checklist instead.

## Known venue-fit constraints

Be aware of these before targeting an unusual venue class; none of them
is hit by mainstream CFD/crypto venues:

- **Client-id budget:** the deterministic id needs up to 30 characters. A
  venue with a shorter client-id limit (some FIX gateways: 20) does not
  fit the current fixed-width scheme; parameterising the scheme is
  required work, not a plugin-side workaround.
- **The software engines are venue-shaped:** the partial-bracket engine
  models per-deal CFD venues, the one-way emulator models grid-volume
  hedging venues, and the spot inventory manager models pooled-balance
  spot venues (see the spot section above). A venue class outside these
  three inherits the full disappearance-detection and state-synthesis
  burden itself.
- **Hedge-native strategies are out of scope by design:** `ExitIntent` /
  `CloseIntent` require `reduce_only=True` at the model level. Hedging
  *accounts* are supported (via `PositionPort`); hedge *semantics* in the
  strategy are reserved for a future `HedgeBrokerPlugin`.

## Testing conventions

Use the opt-in [Offline Broker Conformance Lab](plugin-system.md#offline-broker-conformance-lab)
for lifecycle, restart, concurrency, event-ordering, and modeled venue-semantic
coverage. Keep its `broker_lab/` suite outside the normal `tests/` tree so it is
never collected by the default plugin test command. The central plugin-system
chapter contains the full profile API, transport replacement patterns, commands,
and reproduction workflow.

- The plugin's `tests/` directory must be a package (`__init__.py`),
  otherwise pytest collection silently skips it.
- Test functions follow the `__test_*__` naming pattern.
- Contract-probe tests need real `BrokerPlugin` subclasses — the probe
  compares method identity against the base class, so duck-typed mocks
  do not register as overrides.
- Assert the contract probe on the real plugin class: construct the
  plugin offline and check that `validate_plugin_contract(plugin)` returns
  no error, then wire each port the plugin can opt into and check
  `validate_plugin_ports(plugin)` the same way.
- Keep the set of functions that write a stop loss or a take profit
  closed: a source-level test (`test_*_protection_write_sites.py` in the
  reference plugins) fails when a new write site does not go through the
  protection guard's entry points. An unguarded write is not caught by any
  functional test — it shows up as a false quarantine on a live account.
- Keep the plugin feature-complete: no `NotImplementedError` stubs on
  intent paths. If the venue cannot deliver a capability, declare it
  `UNSUPPORTED` and let startup validation refuse incompatible scripts.
