# DNSE Broker Plugin v2 — native conditional orders on openapi-sdk

## ⚡ PIVOT (2026-08-06) — READ FIRST; supersedes the software-watch design below

The entire "software-watch" design in the sections that follow is **SUPERSEDED**.
It was built on the belief that DNSE's native conditional (STOP/OCO) orders were
unusable. That belief was **wrong** — it was caused by a stale `version` header.

**What actually happened (all proven live 2026-08-06):**
- DNSE's native STOP/OCO conditional orders are **fully API-manageable**: place ✅,
  list ✅ (dedicated conditional book, by `orderCategory`), cancel ✅ — gated behind
  **API `version >= 2026-07-23`** (older/absent version → 500 or silently returns the
  NORMAL book). Sending `2026-01-01`/`2026-05-07` is what made native look "broken".
- Native STOP/OCO are **server-side**, so stops fire even if the plugin is offline —
  strictly safer than a plugin-owned watch. This **deletes** watch.py, the armed-leg
  engine, intrabar OCA teardown (H1), armed-leg persistence/re-arm (H3), the
  DisappearanceTracker work, and the WS intrabar trigger. v2 is **REST-only**.
- Status lifecycle: `New` (armed, cancellable) → `Activated` (triggered → becomes a
  NORMAL order) → `Cancelled`/`Expired`/`Rejected`/`Failed`. Cancel only accepts `New`.
- OCO = a NORMAL take-profit leg (cancellable via NORMAL DELETE) + a conditional SL leg.

**SDK: official `openapi-sdk` 2.0.0**, not `dnse-py`. Vendored at `_vendor/dnse`
(tag v2.0.0 / 1532e33). Its `DNSEClient` drives the account-scoped
`/accounts/{accountNo}/orders` endpoints with `orderCategory=STOP/OCO`, pagination
(`pageIndex/pageSize`), string `orderId`, and defaults to `version=2026-07-23`.

**Done so far:** v1 + dnse-py removed (`b8483e9`); openapi-sdk vendored + verified
(`3530126`); DNSE docs mirrored + sync/verify script (`09d4048`,
`docs/dnse-openapi-documentation/`).

**v2 build plan (native conditional):**
1. **Provider** (`provider.py`) — thin, on the vendored `DNSEClient`: contract
   resolution (`/instruments`), OHLC bars (`/price/ohlc`), secdef, `market_type`.
2. **Broker execution** — map Pine intents to native orders:
   `entry(stop)` → STOP; `entry(limit)` → NORMAL LO; `exit(stop=SL)` → STOP (reduce);
   `exit(limit=TP)` → NORMAL LO; `exit(limit,stop)` → OCO; `close`/market → marketable LO.
   Cancel/replace while `New`; on `strategy.cancel`/OCA, cancel the conditional.
3. **Broker state** — poll the conditional book (`New`→`Activated`) + NORMAL book
   (fills) via REST; position self-tracked from the fill ledger (no undocumented
   `/deals`; `/positions` used read-only for reconciliation only); capabilities
   declare `stop_order/tp_sl_bracket/oca_cancel = NATIVE`.
4. **Trading-token lifecycle** — daily OTP refresh (unchanged from the plan section
   below; mechanism proven live).
5. **Verify** — backtest oracle (`testing/strategies/t3_long_stop` … `t6`) vs live;
   `validate_plugin_contract`; a live smoke test (far-from-market STOP place→cancel).

Everything below this section is the OLD plan, kept for history / salvageable detail
(the DNSE API quirks, the token lifecycle, the Pine intent semantics). Ignore all
`software watch` / `armed leg` / `DisappearanceTracker` / `WS intrabar` machinery.

---

## Context (v1 — historical)

v1 (`plugins/dnse/pynecore_dnse/`) declares `stop_order = SOFTWARE` but does **not**
actually emulate stops — its `execute_exit` places DNSE **CONDITIONAL** orders
(the conditional orderbook) for TP/SL, and `execute_entry` prices a stop entry as
a plain `LO` at the stop level (`intent.limit or intent.stop`, broker.py:472),
which is not a stop at all. v2 has two goals:

1. **Migrate transport to DNSE's official Python SDK** (`dnse` package from
   github.com/dnse-tech/openapi-sdk), replacing the hand-rolled `client.py`.
2. **Make `SOFTWARE` stops real** — emulate every stop/trigger with a
   plugin-owned price-watch that fires a marketable `LO` market order on a
   bar-close cross, and **never** touch DNSE's conditional book.

Decision (confirmed with user): the watch lives **inside the DNSE plugin** (no
pynecore core changes — lowest rebase burden on the rebased fork), and un-fired
watches are **persisted and re-armed** on restart.

## Revisions after approval (2026-08-06)

Decisions/findings that supersede parts of the approved text below:

- **SDK source = VENDOR `dnse-py`, not pip, not `openapi-sdk`.** The real SDK is
  github.com/dnse-tech/**dnse-py** (PyPI `dnse` 0.5.0) — a typed client
  (`DnseClient`/`AsyncDnseClient`, `DnseMarketStream`/`DnseTradingStream`,
  pydantic models, `resources/` = orders/accounts/deals/market/registration).
  User said **"don't trust pip"** → vendored main @ `d0d5c8c` into
  `_vendor/dnse`, imported via `_sdk.py` (`sys.path` shim). The older
  `dnse-tech/openapi-sdk` repo is NOT it (no PyPI, identical WS channels) — ignore.
  Deps: httpx, websockets, pydantic. **SDK has no OHLC resource** → authoritative
  bars go through the thin signed `client.get("/price/ohlc", …)`. Order CRUD =
  `client.orders.place/list/cancel/update`; OTP = `registration.send_otp()` →
  `verify_otp(otp)` → `client.set_trading_token()`.
- **Watch trigger = REST bar high/low (authoritative) + WS intrabar as an optional
  speed layer.** Live probe (40s, session open): quotes `top_price` **740**, ticks
  `tick` **80**, `ohlc`/`ohlc_closed` WS **0 (silent)**, `order` WS **0 (inconclusive)**.
  BUT the WS `tick` feed is a **~50%-partial view** of the matching engine (v1: volume
  ~50% low — the same reason synthetic candles drifted from official OHLC). So WS ticks
  can MISS a real cross → they are a best-effort **speed** signal, NOT authoritative. The
  authoritative trigger is the **REST closed-bar high/low** from `watch_ohlcv` (matches
  the backtest oracle `_check_high_stop`/`_check_low_stop`, never misses); WS ticks/quotes
  only fire *earlier* when the partial feed catches the cross. Fire on whichever crosses
  first, deduped. (Bars stay on REST `/price/ohlc`; the WS `ohlc` channel is silent.)
- **Order/fill feed: TEST the WS order channel live** (place one far-from-market
  order this session, watch `order.DERIVATIVE`) before choosing WS vs REST-poll.

**STATUS:** branch `dnse-broker-v2` created. SDK vendored + `_sdk.py` shim
**verified live** (accounts, contract resolve `VN30F1M→41I1G8000`, `security_info`
ceiling/floor, OHLC via `client.get`). Next: W1 rewrite of `provider.py`/`broker.py`
onto the SDK, then W2 watch + WS intrabar trigger + persistence.

## Review must-fixes (2026-08-06 adversarial review — verdict: SOUND WITH REVISIONS)

The transport work (W1) is unaffected. ALL of the following are **W2 (the watch)**;
they must be designed in before/while building it. The review confirmed no core
changes are required and the identity-routing/threading assumptions hold.

MUST-FIX before shipping W2:
- **H1 — immediate intrabar OCA teardown (plugin-owned).** The engine only tears
  down the sibling leg at the *next bar close* (`_cleanup_position_tracking`,
  `sync_engine.py:6220-6230`, drained via `apply_async_events` at
  `script_runner.py:2141`). Because we fire INTRABAR and DNSE has no reduce_only, a
  resting TP `LO` can fill against a now-flat book and **open an opposite position**
  in that up-to-one-bar window. On fire, the plugin MUST itself cancel the sibling
  `LO` / disarm sibling watches for that `intent_key` *in the same action*, before
  the next tick — never rely on the engine's next-bar cascade.
- **H2 — clamp fire qty to the live net position (software reduce_only).** `_place`
  sends `quantity=int(qty)` with no position read (`broker.py:423-432`); if the row
  was partly reduced, the close overshoots and flips. Read `get_position` at fire
  time; fire `min(leg.qty, abs(net_on_side))`; skip if already flat.
- **H3 — one restart authority; idempotent arm.** The engine ALREADY re-dispatches
  `execute_exit` on restart via `_reconstruct_pine_bracket_state`
  (`settle_restart_state`, `sync_engine.py:1772`), which re-arms exit watches. A
  plugin `connect()` JSON re-arm on top → **double-armed SL → double close → flip.**
  Make arming strictly **idempotent on `intent_key` (replace, never append**, unlike
  v1 `broker.py:448`). Verify what the engine re-dispatches: exits are covered by
  the engine → the plugin JSON is redundant for them; only *un-fired stop-ENTRY*
  watches may need plugin persistence (engine believes the synthetic entry rests and
  won't re-dispatch it). Prefer dropping the JSON where the engine already recovers.
- **M1 — `execute_entry` dispatches strictly on `order_type`.** both-set entry keeps
  `order_type=LIMIT` but still carries `stop` (`intent_builder.py:62-77`) and the
  engine owns its stop watch (`sync_engine.py:12021`). Arm the plugin watch ONLY for
  `order_type==STOP`; `LIMIT` → rest `LO` (ignore `stop`); `MARKET` → marketable
  `LO`. Dispatching on `stop is not None` (v1's shape) double-arms the both-set case.
- **M2 — session-closed fire policy + stream hardening.** A closed-session place
  returns `CANNOT_PLACE_ORDER_IN_THE_CLOSED_SESSION` (`broker.py:581`) and `_place`
  raises; an unhandled raise inside the WS reader kills the feed (v1 `_read_loop`
  collapse, `broker.py:292`). Wrap the fire so it never propagates into the stream
  loop, queue the close for session reopen, and emit a loud protection-degraded log.
- **M6 — resolve trailing.** `trailing_stop=SOFTWARE` passes startup validation but
  `execute_exit` skips a trail-only exit (`broker.py:487`) → position runs
  unprotected. Implement a trailing watch OR downgrade to `UNSUPPORTED` (reject at
  startup). Do not ship declared-but-unimplemented.

SHOULD-FIX:
- **M3 — re-tag, don't "swap the id in."** `_order_mapping` is engine-private; the
  plugin can't swap ids. Cancel/reject route by `order.id`
  (`_find_key_for_order_id`, `sync_engine.py:5039`), so emit cancel/reject
  `OrderEvent`s for a fired order under the **placeholder id** (keep a
  `real_id→placeholder` map); fills can keep the real id (they route by identity).
- **M4 — RESOLVED: authoritative REST bar high/low + optional WS speed layer.** The
  WS `tick` feed is ~50% partial (can miss a cross), so it cannot be the authoritative
  trigger — the REST closed-bar high/low is (it matches the oracle exactly), with WS as
  a best-effort early-fire. See the Revisions "Watch trigger" bullet and the two-tier
  Evaluate. Still add a **near-market** oracle test (far-from-market avoids the region
  where WS/REST divergence would bite) and document the slippage envelope.
- **M5 — arm-bar guard.** Record the arm bar's close ts on each `ArmedLeg`; ignore
  WS ticks at/before it so a straggler same-bar tick can't fire on the arm bar
  (Pine "next bar earliest").

NON-BLOCKING: L1 (the two stop paths fire at different times — engine both-set entry
at bar close vs plugin intrabar; document), L2 (band-edge `LO` partial-fill residual),
L3 (`get_position` returns uPnL=0 — confirm risk hooks don't need it).

Persistence note: **non-blocking writes only** — persistence must not block the
broker event loop (arm + fire + WS all share that one loop), so `store_ctx` writes
stay off the hot path.

## Hidden-contract + doc reconciliation (2026-08-06, 3-reviewer pass)

Reconciled against `broker-plugin-authoring.md`, `plugin-system.md`, `testing-system.md`,
and the real code (storage.py, disappearance.py, validation.py, models.py,
broker_lab/). Common root cause of the restart gaps: **v1 keeps restart-critical state
in memory**. But the sanctioned fix is more constrained than my first edit assumed —
corrections below.

**NEW BUGS in v1 the plan must fix (found in this pass, fix in the W2 order-path rewrite):**
- **B1 — `envelope.client_order_id()` called with NO arg** (`broker.py:441`) but it
  requires `kind` (`models.py:1213`) — and it sits in the dropped-socket `status==0`
  branch, so a write whose socket drops raises `TypeError` instead of parking. Pass a
  per-leg `KIND_*` (`idempotency.py:109-134`: `KIND_ENTRY`, `KIND_ENTRY_STOP`,
  `KIND_EXIT_TP`, `KIND_EXIT_SL`, `KIND_CLOSE`).
- **B2 — `get_position` returns `None` on any fault** (`broker.py:682`). `None` is the
  engine's *authoritative-flat* signal (`sync_engine.py:3059,7991`); the read safety net
  parks only when the read RAISES (`sync_engine.py:15927`). So a transient position-read
  fault → false external-close → double-open. Must **raise `ExchangeConnectionError`**
  like `get_open_orders` already does (`broker.py:660`). Highest-severity read bug.
- **B3 — `_to_exchange_order` never sets `client_order_id`** (`broker.py:373`). Even
  after P1 persists the alias, park-then-verify + restart adoption match survivors **by
  `client_order_id`**, so the read path must **backfill** it via
  `store_ctx.find_by_ref('exchange_order_id', order.id)`.

**P1 — Persist via `store_ctx`, but CORRECTED (there is no arbitrary-row table):**
- `add_ref`/`find_by_ref`/`upsert_order`/`set_filled`/`iter_live_orders` are real
  (`storage.py:1776,1825,1560,1681,1856`). BUT the schema is fixed — a plugin can persist
  only **`orders` rows** (+ `extras` JSON), `order_refs`, `events`. Model an armed watch
  as an `orders` row (`state='armed'`, level in `sl_level`/`tp_level`, the rest —
  `trigger_dir`, `leg_type`, `placeholder_id`, `real_order_id`, arm-bar ts — in `extras`),
  re-hydrated via `iter_live_orders()`.
- `find_by_ref` is an INNER JOIN `order_refs × orders` (`storage.py:1834`) → `add_ref`
  alone returns `None`; the coid must ALSO exist as an `orders` row via `upsert_order`.
- **CORRECTION to my "non-blocking store_ctx writes":** `store_ctx` has **no async API** —
  every write is synchronous lock-serialized SQLite/WAL on the calling thread
  (`storage.py:652`). "Persist-first from the broker event loop" therefore IS a blocking
  write on that loop; there is no free "off the hot path." Decide explicitly: accept the
  short blocking write (simplest, keeps persist-first) OR `asyncio.to_thread` it (breaks
  persist-first ordering). Default: **accept the blocking write** (WAL is fast; correctness
  > a sub-ms stall).

**P2 — CORRECTED: the persisted `filled_qty` cursor is REQUIRED, not optional.** The
engine's `fill_id` dedup set `_seen_fill_ids` is **in-memory, reset every process start**
(`sync_engine.py:929`); cross-restart double-count is prevented ONLY by the persisted
`OrderRow.filled_qty` cursor (`set_filled`), not by `fill_id`. So: emit a stable `fill_id`
built from the **real fired-order venue id** (not the placeholder) + cumulative — the
reference broker does exactly `f"{order.id}:{filled}"` (`reference.py:372`) — AND persist
the fill cursor via `store_ctx`. A real DNSE execution/deal id is preferable (canonical
across REST-poll and any WS redelivery).

**P3 — `DisappearanceTracker` (hidden-contract #1), but its API is bigger than I wrote.**
Real ctor: `DisappearanceTracker(store_ctx, *, grace_s, policy, tracked_refs,
confirm_missing, is_exempt=None, request_quarantine=None, cancelled_event_factory=None,
fill_event_factory=None, …)` (`disappearance.py:326`). Concretely:
- Presence is **per-NAMESPACE**: `observe(present: Mapping[str, set|None], now_ts)` with
  `tracked_refs(row) -> set[(namespace, ref)]`. DNSE uses **`{'orders': {ids}|None}` only** (no
  `positions` namespace — undocumented endpoint; failed fetch = `None`).
- Exclude un-fired watches via `tracked_refs(row) -> ∅` (or `is_exempt(row)`), keyed on the
  `extras` marker — NOT by pre-filtering the snapshot (the tracker walks `iter_live_orders`).
- **`confirm_missing(row) -> MissingConfirmation` is REQUIRED** (my P3 omitted it) — the
  tracker classifies nothing; at grace expiry the plugin re-verifies (`/orders/{id}` /
  deal history) and returns STILL_PRESENT/INCONCLUSIVE/FILLED/CLOSED/CANCELLED.
- Default event builders are ENTRY-shaped → SL/TP exit legs need custom
  `cancelled_event_factory`/`fill_event_factory`.
- `request_quarantine` = the runner's `quarantine_sink` — **but the conformance lab does
  NOT wire it** (`runner._open_run` never sets it); that holds in PRODUCTION only, so a lab
  quarantine test must inject it manually or accept the halt fallback.

**P4 — CORRECTED (I overstated it AND the prescription was wrong):** v1 does NOT collide on
`"default"` — the overridden `account_id` property resolves a real account, so
`validate_plugin_contract` passes. And `connect()` runs **after** `open_run`
(`run.py:1461` probe → `1506` open_run → ~`1698` connect), so "set it in `connect()`" is
too late. Real fixes: resolve account id **eagerly on the auth path the probe already
forces** (`run.py:1461`), make it **plugin-qualified** (`"dnse-<acct>"`, else `run_id`
won't carry the plugin), align on the base `_account_id` field (not private `_account_no`),
and avoid a lazy network call inside a property (a transient fault would abort run
identity). Severity: cleanliness/timing, not a double-open.

**Capability relabels — declare what is delivered end-to-end (guide §Capabilities):**
- `watch_orders` NATIVE → **SOFTWARE** (it's a REST poll, not a live WS channel).
- `reduce_only` / `stop_order` / `trailing_stop` are only honest AFTER H1+H2 (+M6) land —
  sequence the declaration behind the implementation (or under-declare meanwhile).
- `short_selling` NATIVE — add a **derivative-vs-stock guard** (VN stocks are long-only;
  NATIVE would wrongly admit a short-stock script).
- `amend_order` NATIVE → consider **PARTIAL_NATIVE** (for software-watched legs "amend" is
  an in-memory level update, not a venue amend).
- **Idempotency-SOFTWARE obligation:** add **store-keyed dispatch dedup keyed on
  `client_order_id`** (`models.py:493`), not just the alias — else a restart/timeout retry
  double-submits. The coid (per-leg `KIND_*`) is both the dispatch id AND the `add_ref` key.

**Testing (supersedes the ad-hoc `fake_dnse`) — governed by authoring §Testing +
`plugin-system.md#offline-broker-conformance-lab` (NOT `testing-system.md`, which is the
Pyne-code dogfooding system):**
- Conformance lab: a `VenueProfile` whose `create_broker(run_name, store_ctx)` returns a
  `DNSEBroker` subclass with an **injected fake `DNSEClient`** (`broker._client =
  FakeClient` returning recorded `(status, body)` — this intercepts the ENTIRE REST
  surface, since every method funnels through `DNSEClient.request()`), **plus** a WS message
  source fed from an in-memory queue for the intrabar-fire scenarios. (Not "override
  `request()`.") The profile MUST supply its own `handle_step` (WS-tick injection,
  order-book snapshots) and `check_invariants` (H1/H3/coverage oracle) — none are inherited.
- **No in-fork sibling to copy** — `plugins/` has only `dnse`; bybit/capitalcom/cTrader
  suites + their `reconcile.py` are documented *external* references, not in this tree.
  Copy the skeleton from top-level `broker_lab/suite.py` (which uses `ReferenceVenueProfile`).
  DNSE is the FIRST plugin-subclassing `VenueProfile` and first `DisappearanceTracker`
  consumer here. Run: `python -m pynecore.testing.broker_lab run
  plugins/dnse/broker_lab/suite.py --mode smoke|extended`.
- `validate_plugin_contract()` = step-0 gate; keep a proper `plugins/dnse/tests/` package
  (`__init__.py`) for it, DISTINCT from `broker_lab/` (which lives outside `tests/`).
  Conventions: `__test_*__` naming, real `BrokerPlugin` subclasses, one-line-summary
  docstrings, `expected_violation` control profiles for the H1 flip + H3 double-dispatch.

**Packaging / deps corrections:** declare the framework dependency **`pynesys-pynecore>=6.x`**
in the plugin's `pyproject` (currently absent — only works via the editable fork install).
The vendored `dnse` needs `__init__.py` throughout to ship in a non-editable wheel, and its
top-level `sys.path` `dnse` shadows any pip `dnse` process-wide — both flagged-open, keep on
the risk list.

## Reference-grounding (capitalcom + bybit read, 2026-08-06)

Ground-checked against the real `pynecore-plugin-{capitalcom,bybit}` (cloned as workspace
siblings). Both confirm the core design; concrete reference patterns to COPY, plus a few
corrections to my earlier edits.

**ADOPT these real patterns (file:line in the sibling repos):**
- **H2 clamp = bybit `_inverse_reduce_contracts`** (`bybit/execution.py:525-535`):
  `min(requested, abs(net_on_side))`, and **snap a full close to the exact net** (`:529`) so it
  leaves zero residue; treat a reduce-only-zero-position reject as proven-flat (`:975-979`).
  DNSE applies this on EVERY reduce path (it has no `reduceOnly` flag to lean on).
- **P1 persist = persist-first leg row**: `upsert_order(coid, state='armed'|'submitted',
  sl_level/tp_level=…, intent_key=…, extras={leg_kind,trigger_dir,placeholder_id,…})` BEFORE
  the wire call, then `add_ref(coid,'exchange_order_id',id)` AFTER
  (`bybit/execution.py:800-816,595-603`; `capitalcom/execution.py:943-993` adds reopen-on-retry).
  Re-hydrate via `iter_live_orders()`+`find_by_ref` (`capitalcom/recovery.py:249-333`;
  `bybit/recovery.py:94-219`). Synchronous persist-first WAL write on the loop is the accepted
  norm in both — confirms my "accept the blocking write."
- **P2 = persisted `filled_qty` cursor (REQUIRED) + synthetic `fill_id` from `/orders`**: bybit
  `set_filled(coid, cumulative)` (`events.py:536`) + a stable id. **CONSTRAINT (user): NO `/deals`
  endpoint.** So `fill_id = f"{real_order_id}:{cumulative_fillQuantity}"` derived from the
  `/orders` snapshot (exactly the reference default `reference.py:372`), NOT a deal id. Seed
  `previous`/`cumulative` from the persisted `row.filled_qty` on restart (fixes v1's in-memory
  `_last_seen` double-count). The persisted cursor — not the id — is what guarantees restart dedup.
- **B2 raise-on-fault = bybit `_classify_read_error`** (`state.py:501-514`) — `get_position` must
  RAISE `ExchangeConnectionError`, never return `None`.
- **P3 `sibling_coids` (netting-native, was MISSING from the plan)** = bybit
  `_closed_position_siblings` (`reconcile.py:423-440`): on a confirmed net-close, retire ALL
  co-symbol sibling rows, else a sibling strands live against a flat venue.
- **Capabilities per-category split** = bybit returning caps by market type in one
  `get_capabilities` (`bybit/state.py:215-246`): DNSE → `short_selling` NATIVE (deriv) /
  **UNSUPPORTED (stock)**; `watch_orders` **SOFTWARE** (REST-poll = `models.py:476`);
  `trailing_stop` **UNSUPPORTED** (bybit does this deliberately, `state.py:234`, to avoid DNSE's
  declared-but-unimplemented trap); `amend_order` **PARTIAL_NATIVE**; `reduce_only` SOFTWARE
  backed by the H2 clamp.
- **Module split** (both plugins): `broker.py` facade + `execution.py` + `activity.py`
  (watch_orders poll + order→event map + fill_id/cursor) + `reconcile.py` (DisappearanceTracker) +
  `recovery.py` (restart re-hydrate) + `watch.py`. Adopt over one `broker.py`.
- **Lab** (`capitalcom/broker_lab/suite.py`): subclass `ReferenceVenueProfile`, inject the fake
  transport at the single seam (`broker.client = FakeClient`), and drive the plugin's coroutines
  via `asyncio.run(broker._poll_once())` / `broker._evaluate_watch(tick)` inside `handle_step`
  (`suite.py:457-467`) — **no live WS transport needed**. Express H1/H3 defects as
  `expected_violation` **control profiles** (model: `BrokenBracketClearProfile`, `suite.py:1421`).

**CORRECTIONS to my earlier plan edits:**
- `confirm_missing` is a required ctor arg but **a one-liner returning `CANCELLED` suffices**
  when the poll snapshot is authoritative (capitalcom `reconcile.py:501-513`); the FILLED-vs-
  CANCELLED re-verify (bybit `reconcile.py:269-356`) is an OPTIONAL upgrade, not required. (I
  overstated it.)
- `check_invariants` **IS inherited** from `ReferenceVenueProfile` — keep its flip/terminal/
  coid-alias oracle; add defects as control profiles. (I wrote "none are inherited" — wrong.)
- `tracked_refs`: working order → `{('orders',id)}`; armed watch / filled / terminal → **∅**
  (no `positions` namespace — that endpoint is undocumented). NOT capitalcom's migrating `dealId`.
- Drop the "(Not 'override request()')" aside — capitalcom overrides its `_call` seam, DNSE
  injects `broker.client`; both are single-seam interception, same idea.

**CONFIRMED (no change):** H1 plugin-owned intrabar teardown is **mandatory** — bybit defers to
the engine's OCA cascade ONLY because it has native `reduceOnly` (`execution.py:1038`) + bar-close
fills; DNSE has neither, so a resting TP filling intrabar flips the book. The software-watch
architecture, fills-by-identity, synthetic placeholder ids, the `store_ctx` orders-row model, and
dropping the conditional book all match the references.

## Key facts established (grounding)

- **Watch price = `lib.close`.** In `--broker` mode the runner uses ONE plugin
  instance as both the data feed and the broker (`run.py:1445`
  `broker_plugin = provider_data.provider_instance`; requires it to be a
  `BrokerPlugin`). `DNSEBroker` already inherits `DNSEProvider`, so its
  `watch_ohlcv` (authoritative DNSE `/price/ohlc`) is the feed that becomes
  `lib.close` → the engine's `last_price` (`script_runner.py:936`). **The watch
  MUST evaluate on those same closed bars, not a second price poll**, or it would
  trigger on a price the strategy never saw. There is no separate data-source
  plugin in a broker run.
- **Engine already fires MARKET on cross for the paths it owns** — both-set
  entry stop → `execute_entry(market)` after cancelling the resting limit;
  partial-qty exit → `execute_close(market)`. Full-row `exit(limit,stop)`,
  SL-only exit, and stop-only entry are NOT engine-driven; they route to
  `execute_*` where the plugin is expected to rest an order. That gap is what
  the plugin-owned watch fills.
- **Engine↔plugin contract for an armed watch (verified viable):**
  - Returned `ExchangeOrder.id` is engine-internal bookkeeping, never sent to the
    venue; synthetic placeholder ids are already used by the engine
    (`bracket:{leg_id}`). So `execute_*` may return a synthetic OPEN order.
  - **Fills route by Pine identity** (`pine_id`/`from_entry`/`leg_type`), not by
    order.id (`sync_engine._filled_intent_key`). So a synthesized fill just needs
    the right identity — which v1's `watch_orders` already tags via `_identity`.
  - Engine does **not** diff `_order_mapping` against `get_open_orders`, so a
    never-resting watch will not trip a false disappearance — but the plugin's
    own disappearance/reconcile pass must exclude un-fired watches.
  - `execute_cancel` is called by the engine for `strategy.cancel`, `modify_exit`
    (default cancel+recreate), and the SOFTWARE OCA cascade — so it is the hook
    that must **disarm** a watch by Pine identity.
  - Gap to mind: if a fired market order is later CANCELLED/REJECTED by the venue,
    `_find_key_for_order_id` looks up by order.id; keep `_order_mapping` coherent
    by emitting those events under the placeholder id (or by swapping the id in).

## Phase 0 — Verify the DNSE data source (GATING, do first)

**STATUS: PASSED (2026-08-06, live).** `client.get_ohlc("DERIVATIVE",
{symbol:"VN30F1M", resolution:"5"})` → HTTP 200, 146 bars/3 days, fresh (last
closed 13:10 at 13:17 VN), correctly 5m-spaced with the 11:25→13:00 lunch gap
handled; api_key/api_secret signing only (no trading token for market data). The
authoritative REST feed is trustworthy — proceed on it, drop tick synthesis.

The entire software-stop design triggers on `lib.close` from `watch_ohlcv`. If
that feed is wrong or unavailable, every stop fires on a bad price. Yesterday's
check synthesized 5m candles from **tick data** (v1's `_TickAggregator`), which
v1 itself found **drifts** from DNSE's official bars (open off ~0.6, volume ~50%
low — broker.py:85-99, 334-345). So before building anything:

- **Using a working token**, hit the authoritative OHLC endpoint via the SDK
  (`DNSEClient.get_ohlc(bar_type, {symbol, resolution, from, to})`) for the target
  symbol/timeframe (VN30F1M, 5m) and confirm it returns correct, fully-closed
  candles — cross-check a handful of bars against a known-good reference
  (TradingView / DNSE web). Confirm the token scope actually required for market
  data (does OHLC need the trading token, or only api_key/api_secret signing?).
- **Do NOT rely on tick-synthesized candles.** The authoritative REST `get_ohlc`
  is the only sanctioned feed for the watch; remove/ignore the vestigial
  `_TickAggregator` (v1 already bypasses it in `watch_ohlcv`).
- This is a hard gate: if `get_ohlc` is not reliably available for live closed
  bars, the plugin-owned watch is not safe to ship and the approach must be
  revisited before W2. Requires a working token from the user (live call — not
  runnable in plan mode).

## Architecture (chosen)

Only **stop/trigger** legs become software watches. **Limit** legs stay real
resting `LO` orders (a take-profit or a limit entry is a genuine limit the venue
supports — resting it is more faithful and needs no conditional book):

| Pine construct | v2 handling |
|---|---|
| `entry(stop=S)` (stop-only) | arm entry-stop watch → marketable `LO` on cross |
| `entry(limit=L)` | real resting `LO` (unchanged) |
| `entry(limit,stop)` both-set | engine owns the stop watch; plugin only rests the LIMIT leg (unchanged) |
| `exit(limit=TP)` | real resting `LO` reduce-side (unchanged) |
| `exit(stop=SL)` | arm SL watch → marketable `LO` on cross |
| `exit(limit=TP, stop=SL)` | rest TP as `LO` + arm SL watch |
| `close` / market | marketable `LO` (unchanged) |

### The watch (new module `watch.py` in the plugin)

An in-plugin table: `intent_key → [ArmedLeg]` where `ArmedLeg =
{level, trigger_dir (up/down), side, qty, leg_type, pine_id, from_entry,
placeholder_id, real_order_id|None}`.

- **Arm**: `execute_entry`(stop) / `execute_exit`(sl) insert legs and return a
  synthetic `OPEN` `ExchangeOrder` (placeholder id) so the engine records a
  handle. No venue call.
- **Evaluate — two-tier (authoritative REST bar + optional WS speed):**
  - **Authoritative:** at each closed bar from `watch_ohlcv`, test each armed leg's
    `trigger_dir` against that bar's **high/low** — matches the backtest oracle
    (`_check_high_stop`/`_check_low_stop`) and NEVER misses a cross. This is the
    correctness guarantee (worst-case latency: one bar).
  - **Optional speed layer:** also test each `DnseMarketStream` tick/quote intrabar
    (740 quote + 80 tick frames/40s) and fire *earlier* when it catches the cross.
    The WS `tick` feed is ~50% partial, so it is best-effort ONLY — it can miss a
    cross, which the REST bar backstop then catches. Fire on whichever crosses first,
    **deduped** so a leg fires once.
  - Arm→evaluate still respects Pine "next bar" (arming happens in `execute_*` after
    the bar body; add the arm-bar guard, M5). REST bars remain the strategy feed; the
    WS `ohlc` channel is silent, so bars are NOT sourced from WS.
- **Fire**: on cross, place a marketable `LO` via the existing `_place`,
  record the real venue id, and map placeholder→real so subsequent cancel/reject
  events route. The resulting fill is surfaced by the existing REST-poll
  `watch_orders`, tagged with the leg's Pine identity.
- **Disarm**: `execute_cancel(intent_key)` removes armed legs by Pine identity
  (drives OCA sibling-cancel, `strategy.cancel`, and `modify_exit`). `modify_exit`
  /`modify_entry` on an un-fired watch updates the level in place — no venue call.
- **Persist & re-arm via `store_ctx`** (NOT a JSON file — see corrected P1): persist
  each armed watch as an **`orders` row** (`state='armed'`, level in `sl_level`/`tp_level`,
  the rest in `extras` JSON) + the `coid↔broker_id` alias (`add_ref` **plus** the
  `upsert_order` row it joins to), persist-first on arm/disarm/fire. On restart the engine
  re-dispatches `execute_exit` (re-arming exits); the plugin re-hydrates via
  `iter_live_orders()` for what the engine doesn't (un-fired stop-entries). Idempotent on
  `intent_key` (H3). Writes are **synchronous SQLite/WAL on the loop** — accept the sub-ms
  stall (no async store API; see P1).
- **Disappearance**: via `DisappearanceTracker` (P3) over the **`/orders` namespace only**
  (no `positions` — undocumented): presence `{'orders': {ids}|None}`, `tracked_refs` = working
  order → `{('orders',id)}`, armed watch / filled / terminal → ∅. A required `confirm_missing`
  (one-liner `CANCELLED` suffices — poll authoritative), and custom SL/TP event factories.
  Detects a rested-TP `LO` vanishing; a manual close appears as an untracked `/orders` row.

## Workstreams & files

### W1 — SDK migration (vendored `dnse`) — DONE + VERIFIED
- `_sdk.py` shim (`sys.path`→`_vendor/dnse`) exposes the vendored SDK; deps
  (httpx/websockets/pydantic + aiohttp for now) declared in `pyproject.toml`.
- `client.py` REWRITTEN as a thin **adapter** over the SDK's signed transport
  (`DnseClient._request_headers` + `_send`), preserving v1's exact `(status, body)`
  surface (+ status-0-on-network-error) — so `provider.py`/`broker.py` stay UNCHANGED.
  All 8 read call-sites verified live (200s, correct shapes); import chain clean.
  Order writes (`post_order`/`cancel_order`) use the same proven `request()` path;
  the live POST/DELETE+token round-trip is deferred to the next open session.
- Authoritative bars via `client.get_ohlc` → `/price/ohlc` (SDK has no OHLC resource).
  Trading token: caller's `trading-token` header routes through `set_trading_token`.
- **Remaining W1 (folds into W2's WS rewrite):** swap the WS to `DnseMarketStream`
  (intrabar watch feed) + `DnseTradingStream`, and remove the vestigial
  `_TickAggregator`. W2 may adopt the SDK's typed order resources
  (`client.orders.place(PlaceOrderRequest)`) when rewriting the order path, or keep
  the adapter's `post_order`/`cancel_order`.

### W2 — Software stops (the core of v2)
- New `watch.py`: `ArmedLeg`, the watch table, evaluate/fire — persisted through
  `store_ctx` (Hidden-contract P1 — NOT a JSON file).
- `broker.py`:
  - `execute_entry`: dispatch strictly on `order_type` (M1) — `STOP` → arm entry-stop
    watch (fixes v1 LO-at-stop bug); `LIMIT` → rest `LO` (ignore `stop`, engine owns
    it); `MARKET` → marketable `LO`.
  - `execute_exit`: **drop `category="CONDITIONAL"`**. TP (`tp_price`) → real resting
    `LO`; SL (`sl_price`) → arm SL watch. Return synthetic OPEN orders for watched
    legs + real orders for rested legs. On fire: **immediate sibling teardown (H1)**
    and **fire-qty clamp to live net position (H2)**.
  - `execute_cancel` / `execute_cancel_with_outcome`: disarm armed legs by Pine
    identity first, then cancel real venue ids.
  - `modify_entry` / `modify_exit`: update an un-fired watch level in place; real
    amend for rested legs.
  - **State-query fixes (B1–B3):** pass a per-leg `KIND_*` to
    `envelope.client_order_id(kind)` (B1 — currently crashes on the dropped-socket path);
    **`get_position` = self-tracked `store_ctx` ledger net** (no `/positions` — undocumented),
    returning `None` only when genuinely flat, never on a spurious read (B2 mooted — the ledger
    is local); **backfill `ExchangeOrder.client_order_id`** in `_to_exchange_order` via
    `find_by_ref` (B3).
  - `watch_orders`: emit `fill_id = f"{real_order_id}:{cumulative}"` from the `/orders`
    snapshot (P2 — **NO `/deals`**) AND persist the `filled_qty` cursor via `store_ctx`
    (required — engine dedup is in-memory); re-tag cancel/reject of a fired order under the
    placeholder id (M3). Wire **`DisappearanceTracker` (P3)** — but see the "no `/positions`"
    constraint below: its `positions` namespace is dropped and position is self-tracked.
  - Two-tier evaluate (M4): **authoritative REST bar high/low** from `watch_ohlcv`
    (never misses) **+ optional `DnseMarketStream` tick/quote early-fire** (~50% partial →
    best-effort), fire-first-wins deduped; **arm-bar guard (M5)**; **session-closed fire
    policy + stream hardening (M2)**.
  - **Account identity (P4, corrected):** resolve a **plugin-qualified** `_account_id`
    (`"dnse-<acct>"`) eagerly on the auth path the probe already forces (NOT in `connect()`
    — it runs after `open_run`); re-hydrate the watch table + alias from `store_ctx`.
  - `get_capabilities` — **relabel to what is delivered**: `watch_orders` → `SOFTWARE`
    (REST poll); `stop_order/tp_sl_bracket/oca_cancel = SOFTWARE`; `reduce_only = SOFTWARE`
    honest only after H1+H2; add a stock-vs-derivative guard for `short_selling`; consider
    `amend_order = PARTIAL_NATIVE`; **resolve `trailing_stop` (M6)** (implement or
    `UNSUPPORTED`). Add store-keyed dispatch dedup on `client_order_id` (idempotency).

### Trading-token lifecycle (daily OTP refresh) — hard prerequisite for the live order path

The `trading_token` is short-lived + self-invalidating; **placement/cancel/amend need a valid
one, reads do not**. Chosen path: **email OTP read from Gmail** (not `smart_otp`/TOTP — no seed).

**Architecture — separate cron minter + shared token file; the plugin is a pure consumer.**
Keeps the fragile, credential-heavy Gmail/OTP dance OUT of the broker event loop and out of the
conformance lab (the plugin only ever reads a token string; the fake venue supplies a fake one).
Same producer/consumer split as `pyne data download` → `.ohlcv` → `pyne run`.

- **Two legs, ONE source of truth.** Both the automatic (cron) and manual legs mint via the
  **same script**, which writes the **single** `workdir/state/dnse_trading_token.json` =
  `{"trading_token": "...", "minted_at": <unix>}` via an **atomic write** (temp + `os.replace`).
  The plugin reads only that file — it never arbitrates between two token sources, so a manual
  refresh simply overwrites the cron's file and the plugin picks it up on its next read.
  `config.trading_token` is a bootstrap fallback used ONLY when the state file is absent. Do NOT
  mutate `dnse_broker.toml` (config = static creds; state file = runtime token).
- **Minter** (`plugins/dnse/tools/refresh_token.py`, reuses the plugin's `DNSEClient` for
  signing) with two modes:
  - **auto (cron):** `send_email_otp()` → **poll Gmail for the NEWEST DNSE OTP that arrived
    *after* the send timestamp** (self-invalidation: a prior code is dead; delivery lags
    ~30–120s) → `create_trading_token("email_otp", code)` → atomic-write. Retry-send once + alert
    if no OTP arrives. Gmail creds (IMAP app-password / OAuth) live ONLY with the minter.
  - **manual (operator):** `refresh_token.py --otp <code>` — skips the Gmail scrape; the operator
    reads the code themselves and passes it (optionally `--send` to trigger the OTP email first),
    then it writes the same state file. This is the "something went wrong" leg.
- **Schedule:** ONE daily cron at **08:00 GMT+7 (ICT)**. Token **TTL = 8h**, so an 08:00 mint
  (expires ~16:00) covers the full session across the lunch break (morning 09:00–11:30 + afternoon
  13:00–14:45) — no mid-day refresh needed. The manual leg covers any morning where the cron fails.
- **Expiry coordination:** on a token-expired placement reject, the plugin **re-reads the state
  file** (cron may have refreshed since startup); if still stale → loud operator-attention error /
  quarantine. The plugin **never mints** (no Gmail access on the order path).
- **Security:** the token file is order-placement authority — perms + gitignored under
  `workdir/state/`, treated like `api_secret`; never logged.

### Open implementation decisions (recommend, confirm while building)
- **Trigger price** → RESOLVED: WS intrabar ticks/quotes (see Revisions). Residual
  software-stop gap remains: firing sends a MARKET(-able `LO`) that fills at the
  next available price, not exactly the trigger level — WS intrabar just narrows it.
- **Intra-bracket TP↔SL sibling cancel ownership** at `tp_sl_bracket=SOFTWARE`:
  H1 makes this plugin-owned at fire time; verify with the conformance-lab
  reduce-only sibling-cancel invariant + an H1-flip control profile.

## Verification (uses the sanctioned harness — see Hidden-contract §Testing)

0. **Data-source gate (Phase 0):** PASSED live (above).
1. **Contract probe (step-0 gate):** `validate_plugin_contract()` on the v2 plugin →
   zero findings, in a proper `plugins/dnse/tests/` package (distinct from `broker_lab/`).
2. **Offline Broker Conformance Lab** — a NEW `plugins/dnse/broker_lab/` `VenueProfile`
   subclassing `ReferenceVenueProfile`, whose `create_broker` returns a `DNSEBroker` subclass
   with an **injected fake `DNSEClient`** (`broker._client`, recorded `(status, body)` —
   intercepts the whole REST surface). **KEEP the inherited `check_invariants`** oracle
   (flip/terminal/coid-alias) and write only `handle_step`, which **drives the broker's
   coroutines via `asyncio.run`** (`broker._evaluate_watch(tick)`, `broker._poll_once()`) and
   pumps events into the engine — **no live WS transport needed** (pattern:
   `capitalcom/broker_lab/suite.py:457`). Skeleton to copy: the cloned
   `pynecore-plugin-{capitalcom,bybit}` (now workspace siblings). Scenarios: SL cross →
   exactly one marketable `LO` + one
   identity-tagged fill; TP fill disarms SL and vice-versa; **no CONDITIONAL ever sent**;
   restart re-arms with **no duplicate dispatch (H3)**; **fill-replay dedup (P2)**;
   **rested-TP disappearance detected (P3, `/orders`-only)**; **position ledger correct across
   restart (self-tracked from `/orders`, no `/positions`)**. Run:
   `python -m pynecore.testing.broker_lab run plugins/dnse/broker_lab/suite.py
   --mode smoke|extended`.
3. **Control profiles** (`expected_violation`): inject the **H1 flip** and the **H3
   double-dispatch**; each passes only when the oracle catches the injected defect. Note:
   the lab does NOT wire `quarantine_sink` — a P3 quarantine assertion must inject it in
   the profile or accept the halt fallback.
4. **Backtest as oracle**: `t3_long_stop` / `t4_short_stop` (+ a stop-limit exit)
   through the backtest and the lab; compare fills/positions. Include a
   **near-market** case (M4), not only far-from-market, to exercise WS/REST drift.
5. **Live smoke (session-gated, next session):** the deferred WS-order-channel test +
   one far-from-market SL on VN30F1M — nothing rests in the conditional book, the
   watch fires a market order on cross, the fill routes back to the engine.

## Non-goals
- No pynecore core (`sync_engine.py`) changes.
- No reliance on DNSE's conditional/native STOP orderbook. **PROVEN non-viable via the
  documented endpoint (2026-08-06 live probe, fresh token):** `POST /accounts/orders` with
  `orderCategory=STOP` + `stopPrice` returns `200 PendingNew` but the readback shows
  `orderCategory=NORMAL`, the `stopPrice` silently DROPPED, and `metadata.orderSession=ATO` — i.e.
  it degrades to a plain NORMAL LO. `orderType=STO` → `ORDER_TYPE_INVALID`; `MTL` →
  `INVALID_ORDER_TYPE_FOR_THIS_SESSION`. So a "native stop" here is a dangerous silent no-op →
  the plugin-owned software watch is the only safe path. (Also confirmed: LO orders placed
  outside session queue as ATO; market orders are session-blocked.)
  **UPDATE (2026-08-06): the NEW account-scoped endpoint `POST /accounts/{accountNo}/orders`
  DOES place a real STOP (201, rests `New`) — but it lands in DNSE's separate CONDITIONAL ORDER
  BOOK** (user confirmed seeing it there in the app), i.e. exactly the book v2 excludes. And it is
  **place-only through the gateway**: `GET /accounts/{acct}/orders` returns only the NORMAL book
  and silently ignores the `orderCategory` filter, so GET-detail / DELETE-by-id on the STOP id both
  `500 REMOTE_SERVER_ERROR` (header-independent — the `X-Aux-Date` idea was a red herring). An
  order that cannot be read or cancelled cannot be lifecycle-managed. **Decision stays: software
  watch.** Full new-backend order docs mirrored at `docs/dnse-openapi-documentation/`.
- **No DNSE `/deals` or `/positions` endpoints — they are UNDOCUMENTED** (unknown response
  shape/behaviour; don't build on them). Everything comes from the DOCUMENTED `/orders`:
  - **Fills:** `/orders` cumulative `fillQuantity` diff → `fill_id = order_id:cumulative` (no deal
    id) + persisted `store_ctx` cursor. Same as capitalcom's snapshot-diff (no per-fill delta,
    `models.py:148`), on `/orders` instead of `/positions`.
  - **Position (`get_position`) = self-tracked net of the bot's filled `/orders`, persisted in
    `store_ctx`** — the docs' "synthesize position from your own fill ledger" pattern (authoring
    §Spot venues). Durable across restarts AND days, so **carry needs no venue adoption**. On a
    first-ever run with no ledger, assume flat.
  - **H2 clamp** sizes the SL fire against that ledger net; **reconcile/P3** run over the `/orders`
    namespace only (no `positions` namespace). External change is still partly visible: `/orders`
    is the WHOLE account, so a manual close *placed as an order* can be flagged; only a truly
    out-of-band position change (never an order) is invisible — unavoidable given the endpoint is
    undocumented.
- No SDK from pip: the `dnse` package is vendored from source (don't trust pip).
- WS **bars** stay out (channel silent); REST `/price/ohlc` is the strategy feed.
  WS **ticks/quotes** ARE used (intrabar watch trigger). WS **order** channel is
  under live test; until it proves out, REST `/orders` poll is the fill feed.
