<!--
---
weight: 1105
title: "Broker Plugin Authoring Guide"
description: "The full contract a live broker plugin must uphold"
icon: "account_balance"
date: "2026-07-12"
lastmod: "2026-10-15"
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
- **Error taxonomy:** subclass your plugin's expected failures from
  `ProviderError`; mark transient ones (connectivity drops, broker
  maintenance, socket timeouts) as `TransientProviderError` or set
  `retryable` from an error code. Long-running `--broker` / `--live` runs
  wait out retryable faults; a misclassified permanent fault (bad
  credentials marked retryable) loops forever, and a misclassified
  transient one halts a healthy bot.

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
   tracker's `request_quarantine`), while `halt` raises
   `UnexpectedCancelError` for a process exit. The sink returns `False`
   when the held latch already noted the same intervention (the same
   order, protection key or reason); the tracker then skips its audit
   event, and a plugin that calls the sink itself can do the same.
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
cancel (`stop` latches the external-intervention quarantine, `halt` raises).

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

## Transient faults: reads park, writes halt

Classify connectivity faults into the broker taxonomy instead of letting
SDK exceptions escape raw:

- **Recoverable drop** → `ExchangeConnectionError`: the engine reconnects
  and retries next bar.
- **Write whose acknowledgement was lost** →
  `OrderDispositionUnknownError`: the engine parks the dispatch and
  verifies instead of blindly retrying (a blind retry risks a duplicate
  fill).

The engine has a safety net for escaped raw transients, and it is
asymmetric by design: on a per-bar **read** it parks (reads are
idempotent), on an order **write** it halts
(`BrokerManualInterventionError`) because it cannot tell a never-sent
request from a landed one. A plugin that classifies explicitly keeps the
recoverable paths recovering — the net only exists to make contract
violations safe, not to be the plan. Centralise SDK-specific
classification in `_map_exception()`.

The general operational rule: **a thrown error stops the bot and strands
the user.** Prefer keep-running behaviours (retry, observe, degrade) on
every recoverable path; halt only when continuing could lose money
(ambiguous writes, failed defensive recovery).

## Hedging-mode accounts: `PositionPort`

Pine strategy semantics are one-way; a hedging-mode account holds multiple
broker legs per symbol. Do not emulate netting inside the plugin — opt
into the core `OneWayEmulator` by setting `self.position_port = self` once
you know the account is hedging-mode (leave it `None` for netting
accounts), and implement the six `PositionPort` primitives
(`fetch_raw_positions`, `get_volume_quantizer`, `close_leg`,
`reject_out_of_range`, `place_leg`, `amend_bracket`). Each primitive
touches exactly ONE broker entity; the emulator owns all FIFO/netting and
crash-replay logic.

On a hedging account `fetch_raw_positions()` returns the legs this run
opened, and `get_position()` their net. Another party's leg on the same
symbol is not the bot's position and not an intervention; counted in, its
close would read as a reduction of the bot's share. The optional
`supports_partial_leg_close = False` attribute makes the emulator
atomically skip plans containing partial leg slices on venues whose
position close is full-row only (Capital.com).

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

- **`startup()`** once after authentication, before the engine's startup
  reconcile — asset-ownership lease claim, execution catch-up from the
  durable cursor, epoch/invariant validation, adoption watermark. Any
  inconclusive read or unexplainable drift quarantines before a single
  dispatch.
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
  strategy until the next restart's adoption. The lease heartbeat is
  fenced by the physical run instance: if a replacement instance took
  the lease over while this process was silent, `reconcile` quarantines
  instead of trading on a lease it no longer holds.
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
  latches the `EXTERNAL_INTERVENTION` quarantine (or exits under the
  `halt` policy); there is no adoption path. After the operator resume
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
it quarantines. An instance on another workdir or machine is only
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
  would project negative records a graceful halt
  (`BrokerManualInterventionError`) — never a silent skip, which would
  leave the script believing it is short. Exits and closes are
  reduce-only by engine contract and are not gated.

Margin/CFD venues that hold signed positions natively declare
`short_selling=NATIVE` (both reference plugins do), which disables both
gates.

## Runtime configuration

- Plugin credentials/tunables live in the plugin's own `Config` dataclass
  (self-healing TOML at `workdir/config/plugins/<name>.toml`).
- **Broker-agnostic policies do not belong in plugin configs.** The
  cross-broker `workdir/config/brokers.toml` (`BrokerDefaults`) owns them;
  today that is `on_unexpected_cancel` (`stop` / `re_place` / `ignore` /
  `halt`) and `on_inventory_conflict`
  (`quarantine` / `halt`, spot venues only — deliberately narrower:
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
| No `watch_orders()` stream (poll-only, no disappearance channel)        | warning  |
| Inherited `execute_cancel_with_outcome` (all cancels `UNKNOWN`)         | warning  |

A conforming plugin produces zero findings — both reference plugins do.
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
- Keep the plugin feature-complete: no `NotImplementedError` stubs on
  intent paths. If the venue cannot deliver a capability, declare it
  `UNSUPPORTED` and let startup validation refuse incompatible scripts.
