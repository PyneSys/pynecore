## Item 6 review — netting accounting (round 2)

Scope: plan `docs/plan/dnse_v2_fix_plan_V1_UNDER_REVIEW.md` lines 44, 143-148,
293-298, 339 — "sibling retirement in one transaction + flat-sweep freshness
gate + FIFO-pinned shared-handle alias" (bybit + ctrader, Phase B).

**Verdict up front: the item partially dissolves.** One of the three named
sub-mechanisms is inert today, one is redundant with machinery the engine
already ships, and one has no DNSE mechanism to attach to. What survives is a
**different and more urgent defect the item walks past**: a truncated
`/positions` read is byte-identical to a flat account, and the mandated L0 gate
reports `FLAT (exit 0)` on it.

### Premises verified

1. **The DNSE `/positions` row carries a venue handle and a venue clock.**
   `docs/dnse-openapi-documentation/dnse-get-positions.md:146-202`: every row has
   `id` (int64, "ID vị thế"), `status` (OPEN / PENDING_CLOSE / CLOSED / ODD_LOT),
   `accumulateQuantity` / `tradeQuantity` / `closedQuantity` / `openQuantity` /
   `overNightQuantity`, `costPrice` / `marketPrice` / `breakEvenPrice`, and
   `createdDate` / `modifiedDate` (ISO-8601 Z, microsecond precision). The
   envelope carries `pageIndex` / `pageSize` / `pageNumber` / `total`, documented
   as derivative-only. Dedicated endpoints address the handle:
   `GET /positions/{positionId}`, `POST /positions/{positionId}/close`,
   `/positions/{positionId}/pnl-configs`.

2. **`get_position` discards all of it but four fields.**
   `plugins/dnse/pynecore_dnse/broker.py:1289-1317` reads only `symbol`,
   `side`, `openQuantity` (fallback `quantity`) and `costPrice` /
   `averagePrice` / `price`. Probe P2: `total`, `pageSize`, `pageNumber`,
   `pageIndex`, `modifiedDate`, `createdDate`, `id`, `accumulateQuantity`,
   `closedQuantity`, `overNightQuantity`, `marketPrice` — none referenced.
   (P2 reports `status` as present only because the local HTTP variable
   `status, body = ...` shares the name; P4 proves `row["status"]` is never
   read — a row with `status: CLOSED`, `openQuantity: 0` and `quantity: 7`
   produced a size-7 LONG.) `ExchangePosition`
   (`src/pynecore/core/broker/models.py:308-328`) has no field for a handle
   anyway — only the hedging-only `PositionLeg.leg_id` (models.py:346) does,
   and DNSE does not use one-way emulation.

3. **Nothing in the plugin references a DNSE position id.** grep over
   `broker.py`: zero hits for `positionId` / `position_id` / `positions/`. The
   vendored SDK's `get_position_by_id` (`_vendor/dnse/api/client.py:63-70`) and
   the position-close endpoint are never called; closes go out as offsetting
   orders.

4. **DNSE fills carry no positionId.** Zero hits for `positionId` in
   `dnse-get-orders.md`, `dnse-get-order-detail.md`, `dnse-get-executions.md`.
   The order/report schema (`dnse-get-executions.md:438-477`) is order-scoped
   only: `id`, `side`, `symbol`, `fillQuantity`, `lastPrice`, `averagePrice`,
   `transDate`, `createdDate`, `modifiedDate`.

5. **DNSE has zero journal rows today.** `grep -c store_ctx
   plugins/dnse/pynecore_dnse/broker.py` → `0`. The only local order state is
   in-memory `_identity` / `_order_category` / `_last_seen`
   (broker.py:165-169), keyed by **order** id, never by position.

6. **The engine already performs netting-sibling retirement AND a flat
   freshness gate.** `src/pynecore/core/broker/sync_engine.py`:
   - `EXTERNAL_FLATTEN_CONFIRM_GRACE_S = 120.0` (line 197) — a venue-flat
     reading against a non-flat book is not acted on until it survives the
     grace.
   - `_last_position_fill_monotonic` (lines 1362-1369) + the `recent_fill`
     clause (4560-4568) — "a venue snapshot read within the grace of a
     just-booked fill can predate that fill". This is bybit's flat-sweep
     freshness gate, in local-monotonic-elapsed form.
   - lines 4578-4610: on a confirmed external flatten it clears `open_trades`
     wholesale and retires **every** entry id's tracking in one pass — an
     all-siblings retirement of the engine's own rows.
   - lines 4432-4456 + `_durable_owned_signed_size` (4133-4180): startup
     adoption is already clamped to run-owned exposure on netting venues.

7. **bybit's "siblings" are plugin store rows, scoped to one run.**
   `reconcile.py:423-440` (`_closed_position_siblings`) iterates
   `store_ctx.iter_live_orders(symbol=...)`; `iter_live_orders`
   (`src/pynecore/core/broker/storage.py:1856-1876`) filters
   `WHERE run_instance_id = ?`. Retirement is the pair `close_order` +
   `record_complete` (`events.py:397-441`). Those rows exist because bybit
   keeps a filled entry row live for the **position's** lifetime
   (events.py:397-400: "Entry rows live as long as the position they opened").

8. **bybit's freshness gate compares venue clock to venue clock.**
   `positions.py:141-176`: `_deriv_snapshot_ms` = max position-row
   `updatedTime`; `events.py:509-519`: `_last_own_fill_ms` = max execution
   `execTime`; `positions.py:194-199` compares them. Local clock is used only
   as a documented fallback for "a degenerate feed" that ships no timestamp
   (positions.py:168-176).

9. **DNSE fill events carry only a LOCAL clock.** `broker.py:1184` and
   `broker.py:1261` emit `timestamp=int(time.time())`; no order-row
   `modifiedDate` / `transDate` is ever parsed. `OrderEvent.timestamp` is
   documented unix **seconds** (models.py:223, 268) — whole-second resolution.

10. **The vendored SDK already models a WS position push with handle + venue
    clock.** `_vendor/dnse/websocket/models.py:428-473`: `Position(id, status,
    openQuantity, closedQuantity, createdDate, modifiedDate, _receivedAt)`;
    the Order model carries `transDate` / `createdDate` / `modifiedDate`
    (lines 421-424). The plugin never opens a socket (plan's own
    lines 369-378), and delivery is UNPROVEN by the plan's own text.

11. **ctrader's FIFO pin fixes a fill→position reverse-map.**
    `pynecore-plugin-ctrader/src/pynecore_ctrader/_base.py:1160-1195`: the
    `position_id` alias holds exactly ONE client-order-id, and the pin exists
    because "a closing fill — which carries its own `orderId`, not an entry's —
    would reverse-map to the NEWEST entry". It presupposes the venue stamping a
    positionId on fills.

### False-positive challenge

**6a — "a proven close retires all settled sibling rows in one transaction".**
Verdict: **NOT A FIX TODAY; TRUE ONLY AS A PHASE-A DESIGN CONSTRAINT, and it is
mis-scheduled.** There are no rows to retire (premise 5). Item 1's round-2
verdict puts journal rows in Phase A, so the rows this mechanism operates on do
not exist until then — and whether *settled sibling* rows exist at all is a
Phase-A choice DNSE has not made: bybit only has them because it keeps a filled
entry row alive for the position's lifetime (premise 7). If DNSE's Phase A
journals an **order** lifecycle (row closed on terminal order state), zero
settled sibling rows are ever created and this mechanism is permanently inert.
Listing it as independent Phase-B work mis-schedules it: by Phase B the
lifecycle is already fixed. It belongs in Phase A as a constraint on the
row-lifecycle decision.

The plan's stated harm — "one-row closes and last-write-wins silently corrupt
position state" (line 44) — is also overstated for DNSE. The engine already
wipes all its own trade rows on a confirmed flatten (premise 6), so the
*engine's* position state is not what a plugin-side sibling gap would corrupt.
What it would corrupt is the **journal**, which then corrupts
`_durable_owned_signed_size` and therefore the startup ownership clamp on the
**next** run. Real, but a different and later-firing harm than stated.

**6b — "flat-sweep freshness gate: never trust a flat snapshot older than this
run's newest fill".** Verdict: **the named mechanism is REDUNDANT and
un-portable as written; the item SURVIVES only after being reframed.**
- Redundant: the engine already refuses to conclude flat inside a grace of its
  own last booked fill (premise 6), and DNSE has no plugin-side flat sweep for a
  plugin-side gate to protect (premise 5).
- Un-portable: bybit's gate needs venue clock on both sides (premise 8). DNSE
  has a venue clock on the position side (`modifiedDate`) but **not** on the
  fill side (premise 9). Porting it as written compares a venue timestamp
  against a local `time.time()` stamp.
- **What actually bites, and what the item walks past:** a truncated
  `/positions` read is indistinguishable from a flat account. Probe P1:
  `{"positions": [...20 settled rows...], "total": 34}` → `None`, identical to
  a genuine flat `{"positions": [], "total": 0}` → `None`, identical to `{}` →
  `None`. The parser never reads `total` (P2). The vendored client sends no
  page size at all — probe P6 shows the outgoing call is
  `('get_positions', ('ACC1','DERIVATIVE'), {})` and
  `_vendor/dnse/api/client.py:55-62` has no page parameter — so the read rides
  the venue default (the docs example shows `pageSize: 20`), while the sibling
  read `_iter_orders` explicitly asks for `page_size=100`
  (broker.py:1132-1137). That `None` is fed straight into the external-flatten
  detector: `new_size = exch_pos.size if exch_pos is not None else 0.0`
  (sync_engine.py:4362). The 120 s grace only *delays* it — a persistently
  truncated read stays flat past the grace and the engine wipes the whole book
  (4578-4610) against a live venue position.
- **And the MANDATED L0 gate inherits it.** Probe L0: `venue.py flat` returns
  `FLAT (exit 0)` on the truncated body. `read_state` (tools/venue.py:88-92)
  raises only on a failed read; a 200 with a truncated body is not a failed
  read. This is precisely the false-affirmative class the toolkit rule in
  CLAUDE.md was written to eliminate (the 2026-08-19 "account is FLAT" report
  against a real +2). The method-name variant was fixed; the truncation variant
  is open.
- So 6b survives **generalised**: "never conclude flat from a read you cannot
  prove complete", of which "not older than my newest fill" is one clause — and
  not the clause that bites here.
- **Load-bearing UNVERIFIED premise:** whether `/positions` returns CLOSED rows,
  so that >20 rows can accumulate on an active day. The docs make it likely
  (`status` enum includes CLOSED and PENDING_CLOSE, `closedQuantity` is a row
  field, and paging is documented at all — for derivatives specifically), but
  no captured live `/positions` body exists anywhere in the repo (grep for
  `accumulateQuantity` outside the docs mirror hits only the WS model). Probe
  named under "Probes run".

**6c — "ctrader's FIFO-pinned shared-handle alias".** Verdict: **FALSE for DNSE
as stated — the mechanism has nothing to attach to.** ctrader's pin fixes a
fill→position reverse-map (premise 11). DNSE fills carry no positionId
(premise 4); nothing in the plugin reverse-maps by position (premise 3); and
the engine contract has no field to carry a handle (premise 2). The plan's own
DNSE mapping — "our triggered-conditional→child spawn is the same
one-handle-many-rows problem" (line 298) — is a category error: conditional→child
is ONE parent to ONE child, a chain resolved through `externalOrderId`, not one
handle shared by many rows. Two conditionals spawn two distinct children.

The genuine shared handle at DNSE is the `/positions` row `id`, and it is shared
not among *our* rows but among *everyone's*: the venue merges the operator's and
the bot's exposure onto one accumulating row (`accumulateQuantity` /
`closedQuantity`, docs:156-159). A FIFO pin over local rows would not change
that.

What survives is smaller and different, and it is worth keeping: **the position
`id` is DNSE's only per-position identity and the plugin throws it away.**
Consequence today — the operator closing our +2 and immediately opening their
own +2 is byte-identical to our position simply persisting; `id` +
`createdDate` would separate them. That is an operator-shared-account
observability fix, not an alias-pinning fix, and it needs a place to live
(Phase A refs) before it is worth doing.

**6d — `external_activity_ignored` audit "MANDATORY for us" (plan:147-148).**
Out of Item 6's scope: it is attribution machinery, belonging with
run-ownership isolation (consolidated item 4) and depending on Phase A. Not
adjudicated here.

### Path 1

**Completeness-gated parser — make `get_position` refuse to answer "flat" from
a read it cannot prove complete.** Derived as the cheapest possible intervention
at the exact site where the ambiguity is created.

Mechanism: `get_position` distinguishes three outcomes instead of collapsing all
of them to `None` — (i) rows present, not proven truncated → answer normally;
(ii) rows key absent from a 200 dict body → raise `ExchangeConnectionError`;
(iii) `len(rows) < total` with both present → raise. This reuses the function's
existing failure vocabulary (broker.py:1292-1296 already raises for non-200 and
for a non-dict 200 body). Request an explicit page size by defining
`get_positions` on the wrapper `client.py` — whose documented job is exactly
"normalize the vendored SDK's sharp edges … all fixes live here, not in the SDK"
(client.py:1-19) — leaving `_vendor/` pristine. Filter rows on
`status ∈ {OPEN, PENDING_CLOSE}`.

Excludes: any dependency on Phase A, on any clock, or on the WS track.

**Bad-fix check.**
- *Operator partial close mid-run*: the venue mutates the SAME row —
  `openQuantity` down, `closedQuantity` up, `id` unchanged (docs:182, 191-192).
  Completeness is unaffected (still one complete page); the reduced net is
  returned; the engine's #48 drift detector warns. No new failure, no
  improvement. **Neutral — acceptable.**
- *Same-second bot+operator fills*: Path 1 has no time dimension. **Neutral —
  acceptable.**
- *Clock source*: **none used.** This is the path's central safety property — it
  cannot be got wrong by clock skew, and it is the only one of the three for
  which that is true.
- **Bad-fix risk found (1): turning flat-ambiguity into an exception changes the
  failure mode of a path that currently "works".** On the L0 gate this degrades
  correctly — `cmd_status` catches it into "COULD NOT READ" and the toolkit
  contract already reserves exit 2 for could-not-determine (tools/venue.py:14-16,
  128-130). On the engine side it does not: a read that stays unresolved past
  its bridge timeout stops dispatch (sync_engine.py:200-207), so a *chronic*
  paging quirk would turn a live run into a permanent no-trade. **Mandatory
  mitigation: raise only on PROVEN truncation** (`len(rows) < total`, both
  present) **and on rows-key-absent** — never merely because `total` is missing.
  The docs mark the paging envelope derivative-only, so a STOCK read
  legitimately ships none.
- **Bad-fix risk found (2): `len(rows) >= total` is a one-sided test.** It is
  sound only if `total` counts the same population as `positions`. UNVERIFIED —
  if `total` ever counts only OPEN rows while `positions` includes CLOSED ones,
  the inequality passes on a truncated read. **Mitigation: treat
  `len(rows) >= total` as "not proven truncated", never as "proven complete",
  and pair it with the explicit `pageSize` request so the common case has
  headroom rather than resting on the inequality.**
- **Verdict: VIABLE with both mitigations. Lowest risk of the three, and the
  only one whose red-first evidence already exists (P1, P5, L0).**

### Path 2

**Trust-tiered snapshot with a venue-timestamp freshness clause, held
plugin-side.** Derived by exclusion from Path 1: Path 1 makes an unprovable read
loud but gives the caller nothing while it lasts — every truncated poll becomes
an exception. Path 2 keeps answering, with a graded verdict backed by remembered
state.

Mechanism: parse `id`, `status`, `openQuantity`, `modifiedDate` per row into a
plugin-held last-known-good snapshot. A provably complete read replaces it. A
read that is incomplete, or whose newest `modifiedDate` regresses below the
last-known-good, is refused and the previous snapshot is re-served carrying an
explicit staleness verdict. A **flat** verdict is emitted only from a provably
complete read whose newest `modifiedDate` is at or past the newest we have seen.
A position-`id` transition (old id CLOSED, new id OPEN) is emitted as an audit
event — the operator-reopen signal that is invisible today.

Excludes: Phase A (state is in-memory, exactly like `_last_seen`); excludes WS.

**Bad-fix check.**
- *Operator partial close mid-run*: **this is where the path bites itself.** The
  re-serve rule means that if a truncation coincides with the operator's partial
  close, the plugin keeps serving the PRE-close size — it manufactures a
  confident wrong answer in exactly the situation Path 1 would have raised in.
  A cache that hides the failure it was built to survive is the classic bad fix.
  **Mandatory mitigation: a re-served snapshot must carry a distinct stale
  verdict the caller can act on, and must expire under a bounded age — never
  re-serve indefinitely.**
- *Same-second bot+operator fills*: the position side can separate them
  (`modifiedDate` is microsecond-precision, docs example
  `2026-03-23T04:07:45.692156Z`), but our own fill events stamp
  `int(time.time())` — whole seconds (premise 9). Any gate of the form "snapshot
  must be ≥ my newest fill" therefore has a ±1 s blind window in which an
  operator fill can be attributed to us, or ours dropped. Bounded, not fatal —
  but it must never be documented as exact.
- *Clock source*: **the most dangerous point, and the plan's as-written bybit
  port is already wrong here.** bybit compares venue `updatedTime` to venue
  `execTime` (premise 8). The same shape on today's DNSE fields compares venue
  `modifiedDate` against a LOCAL `time.time()` fill stamp: any dev-machine↔venue
  skew silently either jams the gate shut (sweep never fires — looks safe, is
  dead) or opens it permanently (gate is decorative). **Constraint: venue
  timestamp may only ever be compared with venue timestamp.** Honouring it
  requires ALSO parsing the order row's `modifiedDate` / `transDate`
  (available — executions doc:461-462), which is scope Item 6 does not name.
  The skew-free clause available *today* is `modifiedDate`-monotonicity across
  successive position reads, which needs no fill-side timestamp at all — prefer
  that clause and drop the fill-comparison clause until an order-side venue
  timestamp lands.
- **Verdict: VIABLE ONLY WITH the stale-verdict-plus-expiry guard and the
  venue-clock-only constraint. Materially more machinery than Path 1 for a
  continuity benefit the engine's 120 s grace already largely provides.**

### Path 3

**Venue-push position feed — subscribe the vendored `TradingClient` position
channel; demote the REST list to startup baseline + reconcile cross-check.**
Derived by exclusion from 1 and 2: both still rest on a paged list poll whose
completeness the venue only *hints* at via `total`, and neither sees a change
between polls. The WS pushes one `Position` per change carrying `id`, `status`,
`openQuantity`, `createdDate`, `modifiedDate` (premise 10) — not paged, so
truncation cannot arise — and it supplies a venue-clock event stream on the
position side to pair with the WS Order model's `modifiedDate`, reproducing
bybit's gate in its exact shape.

**Bad-fix check.**
- *Operator partial close mid-run*: **best of the three.** A push arrives with
  the new `openQuantity` under the same `id`, immediately — no poll interval, no
  paging. Genuine improvement.
- *Same-second bot+operator fills*: **best of the three for resolution** —
  separate frames, distinct microsecond `modifiedDate`s, plus the SDK's
  `_receivedAt`, ordered per socket. Caveat: neither frame is *attributed* to a
  run, so telling an operator fill from ours still needs run-ownership (a
  different plan item). Path 3 improves resolution, not attribution — it must
  not be sold as solving the operator-shared-account problem.
- *Clock source*: **correct by construction** — venue `modifiedDate` on both the
  Position and the Order model. The only path that ports bybit's gate faithfully.
- **Bad-fix risk (a): the delivery premise is UNPROVEN by the plan's own text**
  (lines 376-378: three channels subscribe ACTIVE, "delivery proof pends ONE
  account event during a capture window … empty is not conclusive"). Building a
  flat verdict on a channel that might never deliver converts today's
  ambiguous-flat into a silent-never-updates — strictly worse, because a cache
  designed to "never fire on ignorance" degrades into permanent ignorance.
- **Bad-fix risk (b): a push feed has no completeness guarantee across a
  disconnect.** The reference plugins answer that with
  replay-subscription-then-backfill (plan:301-307) — i.e. Path 3 does not remove
  the REST snapshot, it adds a second consumer of it. Path 1's completeness
  problem must be solved anyway.
- **Bad-fix risk (c):** it drags Item 6 behind the whole #50 track.
- **Verdict: RIGHT LONG-TERM SHAPE, NOT VIABLE AS THIS ITEM'S FIX.** It
  presupposes Path 1 and an unproven premise.

### Selection

**Path 1**, with Path 2's `modifiedDate`-monotonicity clause and the
position-`id` capture folded in as a follow-on once Phase A gives them a durable
home, and Path 3 recorded as the #50-track target.

Citing all three verdicts: **Path 3** is excluded because it presupposes an
unproven delivery premise *and* still needs Path 1 (its own reconnect backfill
re-reads the same paged REST list). **Path 2** is excluded as the first move
because its central mechanism — re-serving a remembered snapshot — manufactures
a confident wrong answer in exactly the operator-partial-close case it was
reached for, and its freshness clause cannot be built correctly until an
order-side venue timestamp exists. **Path 1** is the only one whose correctness
depends on neither a clock, nor Phase A, nor an unproven channel, and the only
one that repairs the mandated L0 gate today.

Guards, top first:

1. **Raise only on PROVEN truncation, never on a missing paging envelope.**
   `len(rows) < total` with both present ⇒ raise; rows-key absent ⇒ raise; rows
   present with no `total` ⇒ answer normally. Otherwise a STOCK read (paging is
   documented derivative-only) or a venue that drops the envelope turns every
   reconcile into a halt — sync_engine.py:200-207 stops dispatch on unresolved
   reads.
2. Treat `len(rows) >= total` as *not proven truncated*, never *proven
   complete*; request an explicit `pageSize` on the wrapper so the common case
   has headroom instead of resting on the inequality.
3. Filter on `row["status"]` (OPEN / PENDING_CLOSE) — probe P4 shows a CLOSED
   row is currently countable whenever `openQuantity` is falsy and any
   `quantity` field is present.
4. Do not touch `_vendor/`. Add `get_positions(account_no, market_type,
   page_size=...)` to `plugins/dnse/pynecore_dnse/client.py`, the documented
   normalization seam (client.py:1-19).
5. Ship red first: the truncated body returns `None` today (P1) and
   `venue.py flat` exits 0 on it today (L0). Both tests must go red against
   unmodified code before the parser changes.

**First step:** add the two failing tests to
`plugins/dnse/tests/test_broker_state.py` — "a truncated body must not read as
flat" and "a CLOSED row must not be counted" — which fail on today's code as the
probes demonstrate; then make them pass in `get_position` plus the `client.py`
page-size wrapper. Zero dependency on Phase A; lands alongside #54.

**Re-scope recommendation for the plan:** demote 6a from a Phase-B work item to
a one-line Phase-A constraint on the journal row-lifecycle decision ("if entry
rows live for the position's lifetime, a proven close must retire all settled
siblings of the symbol in one transaction"); strike 6c's stated DNSE mapping
(line 298) as a category error and replace it with "capture the `/positions`
row `id` as a journal ref once Phase A exists"; retitle 6b from "flat-sweep
freshness gate" to "read-completeness gate for `/positions`" and move it out of
Phase B into the can-land-now group.

### Probes run

All probes are read-only, offline, against unmodified working-tree code; scripts
are in the session scratchpad.

- **P1 — truncated vs flat (RED).** `get_position` over four constructed bodies
  shaped per docs:146-172:
  `genuine flat (total=0) -> None`;
  `TRUNCATED (20 rows, total=34) -> None`;
  `empty {} body -> None`;
  `missing 'positions' key -> None`;
  `one real OPEN row -> ExchangePosition(side='long', size=2.0, ...)`.
  Four distinct venue situations, one indistinguishable answer.
- **P2 — discarded fields.** Source-level check of the `get_position` body:
  `total`, `pageSize`, `pageNumber`, `pageIndex`, `modifiedDate`, `createdDate`,
  `"id"`, `accumulateQuantity`, `closedQuantity`, `overNightQuantity`,
  `marketPrice` all absent. (`status` reports present only via the local HTTP
  variable of that name — P4 is the behavioural proof it is unread.)
- **P3 — handle not carried.** The returned `ExchangePosition` exposes
  `entry_price, leverage, liquidation_price, margin_mode, side, size, symbol,
  unrealized_pnl`; no field holds the row's `id`.
- **P4 — status unread (RED).** A row with `status: "CLOSED"`,
  `openQuantity: 0`, `quantity: 7` yields `size=7.0, side='long'`.
- **P5 — red-first control, the check that HAS teeth.** A `len(rows) < total`
  completeness test over the same bodies:
  `genuine flat -> COMPLETE`; `TRUNCATED -> TRUNCATED`;
  `empty {} -> UNDETERMINED (no rows key)`; `one OPEN row -> COMPLETE`.
  It separates exactly the cases the parser conflates, and it does not fire on
  the healthy ones — so it is a check with teeth, not a blanket alarm.
- **P6 — outgoing request.** The client call recorded is
  `('get_positions', ('ACC1', 'DERIVATIVE'), {})` — no page size, matching
  `_vendor/dnse/api/client.py:55-62`.
- **L0 — the mandated gate inherits the blindness (RED).** Driving
  `tools/venue.py`'s own `read_state` with the same bodies:
  `genuine flat -> FLAT (exit 0)`; `TRUNCATED -> FLAT (exit 0)` while the
  account holds live exposure.
- **NOT RUN — the one live probe this item needs.** Whether `/positions`
  returns CLOSED rows (hence whether >20 rows accumulate) is UNVERIFIED offline;
  no captured live body exists in the repo. One command settles it during any
  session with a token, and it is cheap:
  `GET /accounts/{accountNo}/positions?marketType=DERIVATIVE` — record `total`,
  `len(positions)`, and the distinct `status` values. If `total` can exceed the
  default page size on a normal trading day, Path 1 is urgent rather than
  prophylactic.

### Dependencies

- **Item 1 / Phase A — hard prerequisite for sub-claim 6a and for 6c's
  residue.** Sibling retirement operates on journal rows that do not exist
  (premise 5: zero `store_ctx` references), and the position-`id`-as-ref capture
  needs the refs table. 6a should additionally be *moved into* Phase A, because
  whether settled sibling rows exist at all is a Phase-A row-lifecycle decision
  (premise 7) that Phase B cannot revisit.
- **Item 8 — overlapping surface, but the fix belongs here.** `get_position` is
  NOT one of the empty-result read paths for a non-200 (it raises,
  broker.py:1292-1296) but IS one for a 200-with-truncated-body. Fixing it under
  Item 6 avoids a double edit of the same function. The sibling blindness in
  `_iter_orders` (broker.py:1132-1144: hardcoded `page_index=0, page_size=100`,
  `total` never read, read failure yields nothing) is Item 8 / #54 territory and
  should be fixed there, with the same one-sided completeness rule.
- **Consolidated item 4 (run-ownership) owns `external_activity_ignored`**
  (plan:147-148), not this item.
- **#50 (WS track) owns Path 3.** Item 6 should not wait on it.
- **#48 drift detector** (sync_engine.py:4612-4630) is the existing
  warn-only consumer of `get_position`'s output; a completeness gate makes its
  warnings trustworthy, and the two should be read together.
- No dependency on #51, #41 or #39 was found for this item.
