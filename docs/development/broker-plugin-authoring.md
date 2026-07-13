<!--
---
weight: 1105
title: "Broker Plugin Authoring Guide"
description: "The full contract a live broker plugin must uphold"
icon: "account_balance"
date: "2026-07-12"
lastmod: "2026-07-13"
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

Two reference implementations exist, deliberately covering the two common
venue shapes:

- **`pynecore-capitalcom`** — REST/poll CFD venue: per-deal positions,
  position-attribute brackets, no exchange-side client ids, software order
  stream fused from snapshot polls.
- **`pynecore-ctrader`** — push venue: protobuf WebSocket wire, native
  execution events, centi-unit volume grid, hedging-mode accounts.

Both follow the same file layout (`plugin.py` / `config.py` / `provider.py`
/ `execution.py` / `reconcile.py` / `recovery.py` / `models.py` /
`exceptions.py` + a `_base.py` tying the mix-ins together) — copy that
structure rather than inventing a new one. When in doubt about how to
implement something, find the corresponding code in whichever of the two is
closer to your venue's shape.

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
following five requirements are what actually make a plugin safe to trade
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
   default `stop` / `stop_and_cancel` latch the engine's **quarantine**
   (trading stops, the process stays alive; wire the runner-injected
   `quarantine_sink` into the tracker's `request_quarantine`), while
   `halt` raises `UnexpectedCancelError` for a process exit.

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
crash-replay logic. The optional `supports_partial_leg_close = False`
attribute makes the emulator atomically skip plans containing partial leg
slices on venues whose position close is full-row only (Capital.com).

## Spot venues: opt into the core inventory layer

Spot venues expose **no position object** — no position row, no entry
price, no unrealized P&L, no position id, no position-attached bracket.
The position itself is not missing: the base-asset inventory relative to
the quote asset *is* the long exposure. The core
`pynecore.core.broker.spot_inventory` module owns this bookkeeping; a
spot plugin opts in by setting `self.spot_inventory_port = self` and
implementing the `SpotInventoryPort` surface, then drives one
`SpotInventoryManager` per run:

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

### The balance is pooled — the invariant fails closed

The venue balance does not separate the bot's inventory from
pre-existing holdings, manual trades or deposits. The ledger is
authoritative for the bot's inventory; the venue balance is checked
against `expected_total = foreign_baseline + bot_inventory(ledger)`,
where the baseline was frozen at epoch creation. Any unexplainable drift
in **either** direction is an attribution conflict — a positive drift
can mask an external sale netted against a deposit, so warn-and-continue
is not an option. The numeric tolerance (`base_tolerance`) covers ONLY
asset quantization; settlement lag is a *temporal* state (a persisted
pending-conflict grace re-checked with fresh executions and balance),
never a wider tolerance. External intervention in the bot's inventory is
not supported: the manager detects it and stops trading (quarantine, or
process exit under the `halt` policy); there is no adoption path.
Recovery is an operator `rebaseline()` — a new epoch generation frozen
in one transaction, refused while unresolved dispatches remain (and
refused when the account balance is below the ledger's bot inventory,
which would freeze a negative baseline and launder the very withdrawal
that triggered the conflict) — plus a restart.

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
  today that is `on_unexpected_cancel` (`stop` / `stop_and_cancel` /
  `re_place` / `ignore` / `halt`) and `on_inventory_conflict`
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

- The plugin's `tests/` directory must be a package (`__init__.py`),
  otherwise pytest collection silently skips it.
- Test functions follow the `__test_*__` naming pattern.
- Contract-probe tests need real `BrokerPlugin` subclasses — the probe
  compares method identity against the base class, so duck-typed mocks
  do not register as overrides.
- Keep the plugin feature-complete: no `NotImplementedError` stubs on
  intent paths. If the venue cannot deliver a capability, declare it
  `UNSUPPORTED` and let startup validation refuse incompatible scripts.
