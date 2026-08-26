# DNSE v2 Fix Plan — architecture parity with the reference plugins

Status: REVIEWED v2 (2026-08-26 — 11-agent adversarial round; see the verdicts section) — phase plan verified against measured
findings; all three reference deep-dives folded in; consolidated ranking at
the end. Execution tracked on card #53 (full pipeline on pickup).

## Per-item adversarial review — verdicts (2026-08-26, 11 Opus agents, full reviews on #53)

Every item was challenged for false positives with red-first probes, given 3
alternative designs, double-confirmed. NO item survived unchanged:

| # | Item | Verdict | Key correction |
|---|---|---|---|
| 0 | _base.py | CONFIRMED on a refuted premise | no type checker sees the plugin; value = the written contract. Base must include DNSEProvider in the MRO (verbatim reference shape BREAKS DNSE). Guard: anti-shadow MRO test (red-proven both ways) |
| 1 | journal wiring | CONFIRMED, worse than stated | a lost reply isn't even parked today (SDK re-raises; status==0 sentinel is DEAD code). Full run_entry BLOCKED (exit intents lack order_type; no run_exit) → persist-first at the `_place` chokepoint. Transport error split lands FIRST or the journal branches are unreachable. `_write`'s token retry = second POST behind one row |
| 2 | recovery + #51 inversion | PARTIAL FALSE POSITIVE | reference orphan passes never cancel live orders — the "cannot cancel" scenario never arises there. Two-class ladder: journalled ids = core re-points automatically (just don't retire); ACK-window crashes = HALT + operator list. Guard: adoption journal-sourced NEVER book-sourced. DISCOVERY: `metadata.ip` = server-written ownership signal (unmeasured — experiment) |
| 3 | DisappearanceTracker | PROCEED, shape was WRONG | union-visibility + immortal #41 shells = the planned ref set silently no-ops on exposure rows → lifecycle refs (parent XOR child). Motivation was a false positive (in-app cancels ARE detected today — 22 events in the corpus). #51-INCONCLUSIVE guards a non-hazard. Staged B0→B2; grace 30 s flat |
| 4 | run-ownership | PARTIAL FALSE POSITIVE (40/20/40) | identity-refusal already structural; adoption filter duplicates the engine clamp; survives as a precondition of item 2 + tri-state ownership (FOREIGN fails closed / INCONCLUSIVE fails open). An owned-set protects the bot from the operator, NOT vice versa — residual-decrease alarm added; FILL tier NOT unlocked |
| 5 | durable dedup | Justification refuted; RECLASSED | restarts are SILENT today (identity dies too); the duplicate emerges the moment Phase A restores identity → HARD PREREQUISITE OF A. Derive-on-restart + the SDK's executions/{orderId} endpoint (eventNo — in the SDK all along). Guard: cross-instance event reader (capitalcom's own rebuild is instance-scoped = broken) |
| 6 | netting accounting | SHRINK — 2/3 dissolved | sibling retirement → Phase A contract clause (engine FIFO already does it); FIFO alias → no reader exists. Survivor: the get_position envelope-completeness guard (NEW hazard: truncated page ≡ flat → external-flatten wipe → double exposure) |
| 7 | cancel dispatch | ADOPT, RESCOPED | cancel_pending doesn't exist as a retryable state anywhere — dropped. NEW HIGH DEFECT: ALREADY_FILLED never returned → cancel-raced-by-fill reads CANCEL_CONFIRMED → engine fires a SECOND MARKET (double-open, red-proven). Blocking verify measured 8.4 s of the 10 s grace on the shared loop. Fix not blocked on Phase A |
| 8 | read-errors (#54) | PARTIAL FP — wrong mechanism | the bare except never sees HTTP failures (18,936 silent polls under a permanent 401, measured): non-200s flow through _iter_orders' else as zero rows. Classify at status level + per-book health; narrow the except FIRST (it would swallow the new halt) |
| 9 | broker_lab | PROCEED, resequenced | ~60% of deliverables hollow; the exclusive value = composition-across-restart + post-write faults. D0 thin profile lands WITH Phase A (T16 reproduces RED under the runner today = A's spec). Guard: pytest wiring from commit 1. DEFECT FOUND: 5 tests PIN the anti-#51 token retry |
| 10 | WS track (#50) | Both justifications refuted | fill-latency is dead (engine drains at bar close); honest win = print fidelity (11% sampled today) + forming bars. TradingClient NOT production-grade (15 defects; silent-death branch on the 8 h close code; zero lines ever run live). S3: WS market-data only; order events stay REST behind shadow grading |

### Resequenced execution order (post-review)

**IMMEDIATE, independent of phases (money-path):** #55 — item 7's
verdict-object + ALREADY_FILLED fix (fix class confirmed blind by round 2);
the page-completeness family — #57 get_position envelope guard, #61 page-0
order-book reads, #62 the L0 gate itself green-lighting on a truncated read
(HIGH — raise on PROVEN truncation only, never infer flat); #54 — item 8's
status-level read classification + narrowed except + _iter_orders
`rows | None` fail-closed contract; #58 — retire `_write`'s #51-blind token
retry (and unpin its tests).

**Phase 0:** `_DNSEBase(DNSEProvider, BrokerPlugin[...])` + anti-shadow test,
split 0a-now / 0b-before-C (round 2); fix the validator false-pass alongside.
**Phase A:** transport error split → persist-first at `_place` (round-2
corrections adopted: urllib3 not httpx; a client timeout KILLS the run) →
item 5's journal-persisted cursor with derive-on-restart as the recovery
fallback (`executions/{orderId}` `eventNo`, both rounds) → D0 broker_lab thin
profile as the red-first spec (pytest-hosted per round 2) →
sibling-retirement contract clause → `ADOPTED_STARTUP_EXTRA_KEY` discipline.
**Phase B:** item 2's two-class recovery ladder + #60 `--run-label` adoption
hole → item 4 as a clamped-snapshot chokepoint over the four raw-net engine
consumers (tri-state ownership retained; write-side precondition + retirement
writer split into its own item) → item 3 REPLACED: terminal-state transitions
are the primary external-cancel channel; the tracker is demoted to a narrow
residue detector whose `confirm_missing` concludes only from paginated
`/orders/history`; first step is the `_iter_orders` `rows | None` rewrite.
**Phase C:** module split. **Phase D:** full broker_lab toggle/control suite.
**Parallel:** WS market-data track (S3 scope); `metadata.ip` ownership
experiment; TradingClient hardening list before any WS production use.

## Round-2 blind re-review (A/B) merged (2026-08-26, operator-approved)

A second, blinded review round ran against the pre-review plan version
(card [#59](https://github.com/rubycell/pynecore/issues/59); full comparison
in `dnse_v2_fix_plan_AB_comparison.md`). No verdict flipped. The resequenced
order above IS the merged truth; material corrections it absorbs:

- **Item 6 fix shape corrected**: raise on PROVEN truncation only — an
  inferred raise on an incomplete envelope would halt healthy runs.
- **Item 7**: the plan's capitalcom citation was the wrong artifact
  (`submit_cancel` is bool and always True); port
  `execute_cancel_with_outcome` + `_classify_cancel_via_activity`. Never emit
  CANCEL_CONFIRMED from inference — only a read-back terminal with
  `filled_qty == 0` confirms; everything else is UNKNOWN (engine retries).
- **Item 4 rescoped deeper** (~80% FP): the surviving fix is a
  clamped-snapshot chokepoint over the four raw-net engine consumers the
  clamp never reaches; halt/settle proofs keep the RAW venue net. The
  post-restart hole runs the OPPOSITE direction (our fills unbooked).
- **New defect cards**: #60 `--run-label` adoption hole, #61 page-0-only
  order-book reads, #62 L0-gate false-green on truncation (HIGH),
  #63 `fake_venue.get_instruments` shape mismatch.

**Fork resolutions** (leader defaults, operator-approved 2026-08-26):

| Fork | Resolution | Why |
|---|---|---|
| Item 1 journal mechanism | r1's persist-first at `_place`, adopting r2's corrections (urllib3 reality; timeout kills the run) | the chokepoint is where intent meets transport and Phase A is built around it |
| Item 3 surviving tracker role | r2's demotion (terminal transitions primary; residue detector; history-only CANCELLED) | only path that passed all bad-fix probes unconditionally; ref-set repair stays hostage to `_iter_orders` disciplines |
| Item 5 cursor source | synthesis: persist with the item-1 journal, derive-on-restart as fallback | persistence is ~free once the journal exists; derivation covers journal loss |
| Item 4 write-side asymmetry | split into its own item (r2) | it is the only protection for operator-partial-close; keeping it implicit is how it stayed unbuilt |

## Executive summary — skip / change / new (updated 2026-08-26 after operator review)

### Skipped / deferred

| What | Disposition | Why | From |
|---|---|---|---|
| ~~WS/streaming modules~~ | **UN-SKIPPED → current work (#50)** | "WS silent" REFUTED — auth-handshake methodology error; alive with every print, order book, forming bars, stock frames | bybit, capitalcom, ctrader |
| ~~Trading-WS (order events)~~ | **UN-SKIPPED → control plane PROVEN** | documented in our own mirror + implemented in the vendored `TradingClient` (never used) + the plugin never opened a socket; all 3 channels subscribe `active`; delivery pends ONE account event during a capture (empty ≠ conclusive) | our probe + SDK |
| ~~Spot inventory~~ | **DEFERRED-RELEVANT** | the account holds stocks; bybit's inventory port is the template; stock frames already flow on #50's socket | bybit |
| Hedge-mode machinery | skip machinery, **keep the seam** | netting hard-coded behind ONE `venue_mode` seam + documented upgrade path | ctrader |
| Unconditional `CANCELLED` in `confirm_missing` | skip (anti-pattern) | we CAN re-verify — return `INCONCLUSIVE`, esp. in the #51 window | capitalcom |
| Retire-and-replace orphans for conditionals | skip (inverted) | #51 — re-adopt what recovery cannot cancel | all 3, inverted |
| broker_lab **now** | Phase D (last) | verifies mechanisms, adds no safety itself | all 3 |

### Changed — existing DNSE code rebuilt

| What changes | Into what / why | From |
|---|---|---|
| `broker.py` monolith (1,327 lines) | module split; **Phase 0 = `_base.py` type-only extraction first** | all 3 |
| In-memory `_identity`/`_order_ids`/`_order_category` | persist-first journal rows; two books = two refs on one row | capitalcom + ctrader |
| In-memory `_last_seen` | durable content-addressed fingerprint + cursor (restarts re-emit fills today) | capitalcom + bybit F4 |
| `_cancel_took_effect` | per-target cancel dispatch table, named `reason_path`, `cancel_pending` on #51 refusals | capitalcom |
| `watch_orders` bare `except: continue` | read-error classification (**#54**, can land now) | bybit |
| **0.5 s order polling** (100k/h budget) | **order-event PUSH via TradingClient** once delivery proves — real-time fills, budget dissolves; REST poll demoted to fallback | our probe + all 3 |
| Tick feed source | S1' REST polling → documented fallback; WS per-print stream = target feed (#50) | our probe |
| Startup adoption (clamps to zero) | journal-backed owned size + re-adoption sweep | capitalcom, bybit |
| Side mapping / `api_version` pin / `errors.py` taxonomy | single chokepoint / verification-dated table / which-recovery dimension | bybit, bybit, ctrader |
| Misleading "WS connected" engine banner | honest per-plugin connect reporting (small fix, on #50) | our discovery |

### New — capabilities DNSE has zero of today

| What's new | Why it matters | From |
|---|---|---|
| Persist-first journal wiring (6-point ordering) | a lost conditional-POST reply is currently unrecoverable; 90% core-provided | capitalcom + ctrader |
| `recovery.py` (verdict ladder, never-mutate-on-doubt, `record_complete` on retire, #51-inverted re-adoption) | fixes T16/#36 on a shared netting account | capitalcom + bybit + ctrader |
| `DisappearanceTracker` wiring (two-book refs, `INCONCLUSIVE` on #51, `is_exempt` #41) | notices operator in-app cancels of our orders | all 3 |
| **Run-ownership isolation** | never adopt the operator's position as ours, never book their closes as our exits | **ctrader-unique** |
| Netting accounting (sibling retirement, freshness gate, FIFO-pinned alias) | one-row closes and last-write-wins silently corrupt position state | bybit + ctrader |
| **WS market-data + trading-event feeds via vendored `TradingClient`** | every print + order book + forming bars + real-time order events, one socket, no REST buckets | #50 probes + all 3 |
| Bracket per-parent aggregated resolution / `external_activity_ignored` audit | race fix / operator trades become observable non-events | capitalcom / bybit+ctrader |
| broker_lab suite (post-write faults, control scenarios, #51/#42-A/#41 toggles) | proves journal/recovery offline; `client.py` is an 85-line override seam | all 3 |

Sequencing: **Phase 0** → **A** → **B** → **C** → **D**, with #50 as a parallel
track (WS feeds) and four independent quick items: #54, the side-mapping
chokepoint, the version-pin table, and the 30-second order-event delivery
proof (one operator app action while a capture is open).

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

## Corrections after operator review (2026-08-26)

1. **The WS-skip rationale was WRONG — venue fact refuted by re-measurement.**
   The "trading-WS measured silent" fact was a methodology error: the WS
   requires an HMAC auth handshake within 30 s and refuses subscribes before
   it; the old probe never authenticated. Re-probed correctly
   (probe_ws_market_data.py): auth_success, 4 channels active, ~2,400
   frames/45 s — 136 trade prints/30 s on 41I1G9000 (every print, vs REST
   latest-only sampling a fraction), order-book depth, forming-bar `ohlc.{res}`
   channels, and HPG STOCK frames on the same socket. The vendored SDK even
   ships the full WS client + AuthManager (never used by the plugin).
   Consequence: the reference plugins' streaming patterns (router-task
   fan-out, inbound-idle watchdog, replay-then-backfill, hold-back ordering)
   move from "recorded for later" to CURRENT work under #50; the WS is the
   strictly superior tick source (#37's S2 premise partially refuted —
   recorded on the card) and the fleet path (#40: one socket, 100
   subscriptions, no REST bucket).
   **Trading-WS follow-through (same day):** order/position event channels are
   documented in our own mirror (trading_connect guide), implemented in the
   vendored SDK (`TradingClient.subscribe_order_event/positions/account` —
   never used), and the plugin NEVER OPENED A SOCKET (connect() is a REST
   no-op; the engine's "WS connected and subscribed" banner is generic text —
   misleading-log fix candidate). Probe: auth_success, all three trading
   channels subscribed ACTIVE; delivery proof pends ONE account event during a
   capture window (zero frames with zero account activity is EXPECTED — empty
   is not conclusive). If delivery proves out, order-event PUSH replaces the
   0.5 s order-poll entirely (fill latency AND the 100k/h Get-Orders budget,
   #40) — the architecture plan's events.py story becomes WS-first with REST
   polling as fallback. MANDATE (operator): build on the vendored
   TradingClient, never a hand-rolled WS client; note the /v1/stream path
   exists ONLY in the SDK — the docs alone cannot connect.
2. **The spot-inventory skip was WRONG — the account holds STOCKS** (HPG et
   al.), not just derivatives. bybit's `inventory.py` (spot inventory port)
   becomes the template when stock trading lands in the plugin; moved from
   SKIP to DEFERRED-RELEVANT, and the stock side should ride #50's WS work
   (stock frames already flow on the same socket).
3. **Hedge-mode future-prep** (netting hard-code stays right): put
   `venue_mode` in ONE seam (the broker_lab profile attribute + a capability),
   write Phase B's netting logic behind that switch rather than inline
   assumptions, and document the hedged upgrade path — so a future venue mode
   is a seam flip plus new tests, not an excavation.
