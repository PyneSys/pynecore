# DNSE v2 Fix Plan — architecture parity with the reference plugins

Status: COMPLETE v1 (2026-08-26) — phase plan verified against measured
findings; all three reference deep-dives folded in; consolidated ranking at
the end. Execution tracked on card #53 (full pipeline on pickup).

## Why (measured, not theoretical)

Every missing module corresponds to a live failure mode we have already
measured:

| Missing piece | Measured consequence | Live scenario that bites |
|---|---|---|
| Store/journal wiring | T16: same-label relaunch — NO adoption, cancel_all cannot reach the old order (stranded, manual cleanup); startup ownership clamp adopts ZERO (#48 matrix a-startup: `_durable_owned_signed_size()` sums a journal nobody writes) | crash mid-session → orphaned resting orders; engine-owned position invisible after restart |
| recovery.py | no reopen/re-own path at all (#36) | same crash, no automated way back |
| DisappearanceTracker wiring | #48 matrix b: an order that vanishes from every book is silent FOREVER (watch_orders is a change-detector over present rows; zero tracker references in plugins/) | operator cancels our order in-app (has happened); venue expires one — plugin never notices |
| reconcile.py | only the #48 warn-only drift detector exists (engine-side, 08-25) | external position changes = one warning, nothing more |
| broker_lab/ VenueProfile | no conformance suite; venue-semantics regressions rely entirely on the live-test campaign | a regression in venue handling ships silently until a live session catches it |

Contract caveat: `validate_plugin_contract` passes clean but validates the
INTERFACE only — and its disappearance check is measured to false-pass when
`watch_orders` is overridden (filed on #36). A clean pass never contradicted
this gap.

Structure debt: `broker.py` is 1,327 lines (repo rule: 800 max) and growing
(tick mode +~200). bybit ships 18 modules, capitalcom 16, each with a
broker_lab conformance suite; DNSE ships 6 files.

Operational mitigations that have kept us safe so far (procedures, not code —
they do not survive a crash): L0 before every run, venue-record grading
(venue.py), operator-close protocols, single-process runs, T10-proven
isolation, the #48 drift warning.

## Phases (each unlocks the next)

- **Phase A = #36 — store/journal + restart recovery (KEYSTONE).** Journal
  every order write (place/cancel/amend outcomes) via the store seam; restart
  re-owns via run_tag + journal (idempotency=SOFTWARE means no venue-side
  coid — the journal IS the bridge). Copy capitalcom's recovery contract and
  journal call patterns. Fixes T16 + the startup clamp in one move.
- **Phase B — DisappearanceTracker wiring.** 100% store-coupled
  (disappearance.py:424,478) so it needs A. Namespaces: NORMAL book, STOP
  book (two-book venue — an id ABSENT from both books may have MOVED to a
  child, so `confirm_missing` must walk the externalOrderId chain before
  declaring vanished; #48 matrix b-trap).
- **Phase C — module split.** Code-motion into the reference layout
  (`execution.py` / `reconcile.py` / `recovery.py` / `state.py` / feed module
  for #37's tick code) AFTER A/B so the modules are born with real content.
  Suites green at every step + one no-fill live regression session.
- **Phase D — broker_lab VenueProfile.** Encode the measured venue facts
  (two order books + Activated→child, #51 window, session phases, GTD clamp,
  amend-500, cancel-ACK semantics, latest-only tick source) as a conformance
  suite matching the reference plugins' broker_lab structure.

Sequencing with existing cards: #36 is Phase A (already head of the operator's
queue). #48's remaining scope (scenario-c design decision) and #50 (WS capture)
feed Phases B/D. #51's mechanism confirmation gates nothing here but shapes
recovery behavior (a recovering engine may be unable to cancel pre-crash
conditionals — recovery must RE-ADOPT, never blind-cancel).

## Reference deep-dive findings (2026-08-26)

### bybit (18 modules) — audited 2026-08-26

**1. Persist-first store wiring (THE pattern — Phase A's template).** Every
write path: record identity → persist row `state='submitted'` BEFORE the wire
call → `log_event('dispatch_submitted')` → POST → on success
`upsert_order(state='confirmed', exchange_order_id)` + `add_ref` / on failure
`mark_disposition_unknown` (execution.py:567-612, 791-829). Reason documented
in-code: a crash between send and ack must not orphan a fill. Fill-time:
`set_filled` from the authoritative row; retirement is ALWAYS the pair
`close_order` + `record_complete` — a stale envelope otherwise re-derives the
same spent coid, and DNSE has NO venue-side duplicate rejection to catch that.
*DNSE mapping:* our locally-minted coid is the store key (no venue coid
needed); the venue's int id, conditional string id, and adopted child id
become multiple `add_ref` rows on one coid — the two-book problem made durable.

**2. recovery.py (532 lines, runs at broker startup before first
get_position).** Three inviolable rules, each mapping to a measured DNSE
hazard: *never re-dispatch* (no venue idempotency), *not-found ≠ rejected* (a
DNSE conditional legitimately vanishes into its child — lookup must walk BOTH
books + externalOrderId before concluding gone), *any live exposure aborts the
orphan pass* (netting + operator on the same account = exposure often not
ours). Failed venue reads are inconclusive and skip the pass, never advance it.

**3. DisappearanceTracker wiring (~250 lines of hooks; core class is free).**
Two namespaces (orders by id / settled positions by symbol); a failed snapshot
read yields None (never an empty set); `confirm_missing` re-verifies against
FRESH venue reads with verdicts that refuse to retire on a stale fill cursor;
grace (25 s) deliberately decoupled from cadence (10 s). *DNSE mapping:*
`tracked_refs` returns a SET, so a triggered conditional is tracked under
parent AND child refs simultaneously — surviving the CLOSED transition without
a false stamp; #41's immortal Activated shells are the `is_exempt` case;
`confirm_missing` maps onto our `_cancel_took_effect` + `_resolve_child_detail`.

**4. Netting handling (closest match to our venue).** `_closed_position_siblings`:
a proven close retires ALL settled sibling rows of the symbol in one
transaction (netting merges pyramids — closing one row leaves siblings live
against a flat venue). Flat-sweep freshness gate: never trust a flat snapshot
older than this run's own newest fill. `external_activity_ignored` audit for
unattributable executions — MANDATORY for us (operator shares the account).

**5. Module split.** 18 modules; the enabling trick is `_base.py` (423 lines):
type-only attribute annotations + cross-mixin method signatures, with
`plugin.py` as the composition root. Audit's verbatim warning: **"do _base.py
first or the split just moves the monolith around."**

**6. broker_lab.** `OfflineBybit(Bybit)` subclasses the REAL plugin overriding
only the HTTP seam; ~13 profile subclasses each isolate one hostile venue
behavior; step kinds assert the exact barrier sequence was exercised. DNSE
step kinds write themselves: `trigger_conditional`, `withhold_external_order_id`
(#42-A), `operator_trade` (#51 lockout), `drop_next_place_ack`.

**7. Additional finds:**
- **F4 durable fill backfill** (REST-native, fixes our empty-`_last_seen`
  restart hole): watermark persisted as an audit event, fixed 60 s overlap
  re-read, and NEVER advance a cursor past a window that did not fully drain.
- **NEW DEFECT FOUND IN DNSE**: `watch_orders` uses bare
  `except Exception: continue` — an expired/refused token silently degrades
  into a permanently silent fill feed (the #51 failure shape, invisible).
  bybit classifies read errors so credential failures surface. → carded.
- Config hygiene (hosts.py): our inline api_version pin is the same hazard
  class (a floating date silently no-ops conditional cancels, measured 08-07)
  — deserves an isolated, verification-dated table.
- Side-mapping chokepoint: our NB/NS ↔ buy/sell ↔ long/short is a THREE-domain
  mapping with no single chokepoint (the #49 bug's habitat).
- live_provider reconnect ordering (for #50, and the hold-back-then-release
  cursor rule applies to today's REST tick feed too).

**bybit top-5 by live-risk reduction:** 1) persist-first wiring, 2) recovery.py,
3) DisappearanceTracker hooks, 4) netting sibling retirement + freshness gate,
5) durable poll-cursor backfill. broker_lab explicitly AFTER these (it verifies
them).
### capitalcom (16 modules) — audited 2026-08-26

**HEADLINE: the journal is 90% core-provided.** Capital.com owns only wire
format + venue verdicts; `DispatchJournal`, `DisappearanceTracker`,
`store_helpers`, `store_ctx`/`quarantine_sink`/`on_unexpected_cancel` are all
core, already on `BrokerPlugin` — and DNSE references NONE of them (0 hits).
**Phase A is wiring, not writing.**

**1. The 6-point persist-first ordering** (journal.py:761-880, the canonical
sequence): row `submitted` + audit event (with endpoint+body payload for
forensics) BEFORE the venue call → every failure class advances the row
(disposition_unknown with phase tag / rejected — NEVER left `submitted` after
a known outcome) → server ref recorded (refs table FIRST, then extras, so a
crash between leaves the ref reachable) → confirm/reject/timeout each mapped →
the returned ExchangeOrder is built FROM THE PERSISTED ROW, not the response.
The contract tests pin state × extras × refs × ORDERED event list per branch —
byte-level. Three load-bearing invariants: no confirm_level on non-filled
rows; the ref recorded before a rejected confirm; the id-ref alias is what
makes a row visible to the whole plugin.

**2. Recovery without venue client-ids (OUR exact problem).** Priority
ladder: kind-routed verdicts, stored-ref match first, then an activity
heuristic (symbol AND side AND qty AND ±3 s timestamp band); exactly-one
candidate confirms, zero = still_unknown (row untouched — the engine re-drives;
never mutate on doubt), two+ = `submission_ambiguous` + HALT with the
candidate ids. Every verdict carries a named `recovery_path` (30+ strings) —
the forensic channel. Orphan retirement MUST also `record_complete` (the
stale-envelope → same-coid → invisible-row trap, 12-line comment). And the
symmetric sweep — `_adopt_untracked_positions` (netting-only!) seeds synthetic
confirmed rows for live venue legs nobody tracks, because close paths derive
targets from confirmed rows: without it `close_all` STRANDS live exposure.
*DNSE inversion for #51:* a pre-crash conditional we cannot cancel goes
through re-adoption, never retire-and-replace.

**3. Activity/dedup discipline.** Content-addressed fingerprint over a FROZEN
field set (venue has no stable activity id — same as us), persisted as
`activity_processed` events, cursor replayed on restart; unmatched activity
stays RETRYABLE (don't fingerprint, don't advance the watermark — may be our
own add_ref racing); batch-wide defer clamp; breadcrumbs stamped BEFORE the
yield (a crash between yield and write turns a natural close into a false
UnexpectedCancelError). *This is our in-memory `_last_seen` made durable —
today every restart within the venue list window can RE-EMIT fills.*

**4. confirm_missing may return INCONCLUSIVE indefinitely.** Capital.com's
venue can't re-verify so it concludes CANCELLED; ours CAN (detail read) and
MUST return INCONCLUSIVE during the #51 refusal window — keep the stamp,
never conclude a cancel you cannot verify.

**5. Cancel dispatch table with named reason_path (the #45/#46 lesson,
systematized).** Per-target-type branches; NOT_FOUND absorbed as benign
`already_gone`; filled position = silent noop (never cancel a fill);
`strategy.cancel(entry)` never clears a software bracket (TV semantics); and
`cancel_pending` — a refused write marks the ROW pending-with-reason so the
reconciler retries, instead of our `_cancel_took_effect` blocking 6 reads then
returning a bare False nothing acts on. `reason_path='refused_conditional_
window'` is #51's slot.

**6. Bracket aggregator race fix.** TP+SL legs share ONE parked dispatch:
resolutions are aggregated per parent within a poll and written ONCE —
per-leg writes race the engine's consumption. Directly applicable to our OCO
umbrella + software-bracket paths.

**7. broker_lab fault-injection model.** `OfflineCapitalCom` overrides ONE
transport method; `_raise_post_write_fault` mutates venue state then
suppresses the response — the exact lost-reply crash window — paired with
CONTROL scenarios (`expected_violation`) proving each barrier assertion has
teeth; the runner auto-minimises failures and prints a reproduction line.
*Our override point (`client.py`, 85 lines) is even cleaner.*

**capitalcom top-5:** 1) journal wiring via run_entry, 2) recovery ladder +
#51-inverted re-adoption, 3) tracker with two-book refs + INCONCLUSIVE, 4)
durable content-addressed dedup, 5) cancel dispatch table with cancel_pending.
### ctrader (16 modules, protobuf/TCP transport) — audited 2026-08-26

**1. Persist-first with THREE-way failure classification + a post-write-drop
error class (ctrader-unique).** Row persisted `submitted` + audit event BEFORE
the wire send; then each failure class advances the row in lock-step:
disposition-unknown / rejected / pre-send-connection-error-leaves-row-submitted
(execution.py:434-481). The transport distinguishes a PRE-write drop (clean
retry) from a POST-write drop (bytes queued or ack lost → disposition
unknown) via a dedicated error class (wire.py:75-87) — neither sibling has
this. *DNSE mapping:* our httpx layer needs the same split — connect-error vs
read-timeout/post-send drop; with idempotency=SOFTWARE this is the difference
between a safe retry and a duplicate live order.

**2. Run-ownership isolation (ctrader-unique — THE operator-shares-account
machinery).** `_owned_position_ids()` = the set this run's own journal
recorded, filtering close-targeting, get_position adoption and
get_open_orders; identity resolution REFUSES non-run-unique handles (on a
netting account the position id is shared — a fill reverse-maps only through
the run-unique order id, everything else is dropped as external)
(_base.py:1102-1158, events.py:349-399). *DNSE mapping:* prevents the two
worst live failures on our account — adopting the operator's manual position
as bot exposure, and booking operator closes as bot exits.

**3. Reconcile mechanics.** Cadence piggybacked ONTO the watch_orders loop
(deadline checked at loop top — a busy stream cannot starve reconcile);
grace (25 s) decoupled from cadence (5 s). `confirm_missing` classifies a
vanished row FOUR ways — inconclusive (keep stamp) / filled-then-closed (the
"operator closed it manually" case → terminal close, never a synthetic
cancel) / false-premise (it actually filled) / conclusive-no-fill (genuine
unexpected cancel) — and never concludes a cancel from a truncated read
(reconcile.py:355-419).

**4. Recovery refinements (ctrader-unique).** Evidence-gated abandonment: a
still-unknown row is abandoned only when the history read was CONCLUSIVE
(paginated to exhaustion, no transport error) AND a 600 s TTL elapsed;
inconclusive-but-fills-visible records the refs WITHOUT confirming, so a late
fill push still reverse-maps instead of reading as external. The
`_adoption_baselined` guard stops recovery re-running on mid-session
reconnect (a bug-class we would otherwise hit the first time we add recovery).

**5. Netting position accounting (ctrader-unique, directly our shape).** One
venue handle, many local rows: the shared alias is pinned to the FIFO-OLDEST
entry (netting closes reduce oldest-first; last-write-wins mis-attributes
exits), every row mirrors the handle in extras so a full close flattens all
pyramid rows, and sharing emits a one-time audit event. *DNSE mapping:* our
triggered-conditional→child spawn is the same one-handle-many-rows problem.

**6. Transport patterns (for #50 and the REST tick feed).** Single router
task fans out one socket's messages (two consumers on one queue steal each
other's frames); inbound-idle watchdog (90 s) distinct from ping cadence
(10 s) catches half-open TCP; `on_reconnect` = replay-subscription-first,
then backfill, run OUTSIDE the per-bar timeout budget; backfill's `settled()`
predicate requires the NEWEST-expected slot specifically and degrades rather
than halts. Every timeout constant carries a measured rationale + regression
test.

**7. broker_lab.** Fake injected at the transport seam only (real plugin stack
above it); venue behaviors as fake toggles; and — worth copying wholesale —
**control scenarios**: 7 deliberately-broken plugin subclasses paired with
scenarios asserting the harness DETECTS each bug. A conformance suite that
proves its own teeth. *DNSE toggle candidates:* `refuse_conditional_writes_
after_operator_trade` (#51), `withhold_external_order_id` (#42-A),
`activated_shell_persists` (#41).

**8. Error taxonomy with a recovery dimension.** Code sets select WHICH
recovery runs (auth-loss checked BEFORE generic reject mapping so it never
surfaces as an order rejection); our errors.py has the classification but not
the which-recovery dimension.

**ctrader top-5:** 1) persist-first + post-write-drop class, 2) run-ownership
isolation, 3) tracker wiring with 4-way conclusive confirm_missing, 4)
FIFO-pinned alias for netting, 5) startup-orphan abstentions +
adoption-baseline barrier.

## Consolidated adoption ranking (three audits reconciled, 2026-08-26)

The three top-5 lists agree almost perfectly. Merged, by live-risk reduction:

| # | Adoption | Sources | Phase |
|---|---|---|---|
| 0 | **`_base.py` type-only extraction** — pure annotations + cross-mixin signatures, zero runtime risk; all three audits: "do this first or the split just moves the monolith" | all 3 | Phase 0 — can land ANY time, enables everything |
| 1 | **Persist-first journal wiring** (capitalcom's 6-point ordering + byte-level contract tests; ctrader's post-write-drop transport error class; refs: conditional id + child id as TWO refs on one row) | all 3 (#1 each) | A |
| 2 | **recovery.py** — verdict ladder, still_unknown touches nothing, record_complete on every retire, and the #51 INVERSION: re-adopt uncancellable pre-crash conditionals | all 3 | A |
| 3 | **DisappearanceTracker wiring** — two-book ref set, INCONCLUSIVE during the #51 window, is_exempt for #41 shells, fresh-read confirm_missing | all 3 | B |
| 4 | **Run-ownership isolation** — journal-derived owned-id set filters adoption/close-targeting; identity resolution refuses non-run-unique handles (operator shares the account) | ctrader-unique | A/B |
| 5 | **Durable content-addressed activity dedup + cursor** (replaces in-memory `_last_seen`; fixes restart fill re-emission) + never advance a cursor past an un-drained window | capitalcom + bybit F4 | A |
| 6 | **Netting accounting** — sibling retirement in one transaction + flat-sweep freshness gate + FIFO-pinned shared-handle alias | bybit + ctrader | B |
| 7 | **Cancel dispatch table** — per-target branches, named reason_path, `cancel_pending` for the #51 refusal window (replaces `_cancel_took_effect`'s blocking + bare False) | capitalcom | B/C |
| 8 | **Read-error classification in watch loops** (bare `except: continue` kills the fill feed silently — carded #54) | bybit | can land NOW |
| 9 | **broker_lab suite** — transport-seam fake, post-write fault + CONTROL scenarios, #51/#42-A/#41 as venue toggles | all 3 (explicitly LAST — it verifies 1-7) | D |

Cheap immediate wins independent of the phases: #54 (read-error classification),
the side-mapping chokepoint (three-domain NB/NS↔buy/sell↔long/short — the #49
habitat), the api_version pin isolated into a verification-dated table.
