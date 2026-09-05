# DNSE live-test suite

How the DNSE broker plugin is tested, after the 2026-08-10 → 08-14 live campaign.
Four operational test types; every live verdict is graded from **DNSE's own records**
(order details / books), never from the run log alone — the log claimed success twice
while the venue disagreed (see findings below).

## Run order for a live session — MANDATORY escalation (operator rule, 2026-08-19)

Never start a session with a fill test. Escalate, and stop at the first failure:

1. **Live-L0-Gate** — venue semantics (`level0_venue_semantics/l0_order_semantics.py`),
   must exit 0. Proves auth, both books, place/rest/cancel round trips.
2. **NO-FILL smoke** — `live_staged_place_cancel.py` states **0–2 (T01–T03)**:
   long limit place/cancel, short limit place/cancel, limit + `exit(stop)` with
   both legs cancelled explicitly. ~10 min including warmup, and it touches BOTH
   books (the exit leg is a conditional), so it proves the whole
   Pine→engine→plugin→venue chain cheaply. Set `winEnd` so the run stops after
   state 2.
   *The full T01–T13 ladder is a periodic regression* — run it after plugin
   changes or weekly, not before every fill session (operator decision
   2026-08-19: the short smoke buys ~20 min of fill-window).
3. **FILL tests LAST** — the staged fill ladder (operator-in-the-loop, below), F10,
   F11. Only after 1 and 2 are green in THIS session.

Rationale: a fill test is the only thing that puts real exposure on the account, so
it runs only once everything cheaper has proven the plumbing is healthy today —
token, session phase, both books, cancel paths, engine wiring.

## Risk tiers — what may run when (decided 2026-08-18)

Case IDs are stable (never renamed); the TIER is the execution gate:

| Tier | Levels / cases | Precondition | Why |
|---|---|---|---|
| **NO-FILL** | L0, all L1 (T01–T17, T32), L4 | open session + GOOD token only — **safe with an open user position** | every order is ≥4.5% away or cancelled before it can fill; ORDER-level isolation is proven (Live-L1-T10 ran two engines on one account with zero cross-contamination) |
| **FILL** (highest tier) | all L2, all L3 (F01–F11) | **FLAT account or a dedicated sub-account, supervised** — NEVER with an open user position | VN derivatives accounts hold ONE NET position per symbol: a short fill SELLS the user's own contracts (2→1) and the flatten re-buys them — trading their position without consent; even long fills merge into and RECOMPUTE their average entry price; venue-record grading becomes unattributable |

The FILL tier is not about engine capability — order tracking is per-run — it is the
venue's position netting. Until the operator flattens or provides a sub-account
(`account_no` in `workdir/config/plugins/dnse_broker.toml`), FILL-tier cases are
**parked**, not skipped.

## Offline suites — map and red-first anchors (verified 2026-08-25)

Counts as of 2026-08-25 (they grow; re-run for current numbers):
**plugin suite 366** (`pytest plugins/dnse/tests/ -q`) and **core engine suite
1,664** (`pytest tests/ -q --ignore=tests/t00_pynecore/ast/test_045_…`).

| Plugin file | n | Focus |
|---|---|---|
| test_provider.py | 68 | OHLCV provider, symbol/timeframe mapping |
| test_broker_state.py | 52 | position/fills/watch loop + most red-first anchors |
| test_broker_orders.py | 51 | order construction, payloads, bands |
| test_errors.py | 43 | venue-code classification, cancel book-probe table |
| test_broker_lifecycle.py | 42 | place/cancel lifecycle, OCO resolver, `_with_outcome` |
| test_client.py / test_errors_helpers.py | 31+30 | transport seam / classification helpers |
| test_refresh_token.py / test_token_status.py | 11+7 | token tooling |
| test_stop_fill_price.py | 10 | stop→LO pricing (slippage floor) |
| test_tick_feed.py | 7 | #37 tick synthesis + guards |
| test_divergence_matrix.py | 6 | #48 executable baseline (engine+plugin) |
| test_fixes_end_to_end.py | 4 | fake-venue e2e |
| test_sdk.py / test_contract.py | 2+2 | vendored SDK / plugin authoring contract |

**Red-first anchor map** (each proven FAILING before its fix; the permanent
tripwire for that card — never weaken):
- **#42** `__test_activated_adoption_retries_when_external_id_is_late__`,
  `__test_adoption_gives_up_with_manual_intervention__` (broker_state)
- **#43** `__test_unresolved_oco_umbrella_child_fill_still_surfaces__` + the
  dead-umbrella / drain-give-up guards (broker_state)
- **#45** `__test_cancel_one_unknown_id_probes_every_book__` (test_errors)
- **#47** `__test_cancel_with_outcome_covers_adopted_children__` (broker_state)
- **#49** `__test_get_position_side_speaks_the_engine_contract_vocabulary__` (broker_state)
- **#39** has REGRESSION PINS only (written after the 08-19 fix; docstrings say
  "#39 regression") — its live proof is F11.

### How to read the registry (evidence classes — audit rule, 2026-08-25)
A registry STATUS is trustworthy when it carries a date and an evidence trail
(evidence file in `logs/`, or the measurement inline in the row from venue
records). SIDE-ANNOTATIONS (block reasons, cross-references) rot silently as
cards close — the F03–F08 rows carried a "#39 blocked" reason for 6 days after
#39 was fixed. Rules: annotations must cite the card they depend on; closing a
card includes sweeping the registry for annotations that cite it; a
double-check of these tables re-runs the suites and re-reads the evidence
files for recent rows, but historical measurements are vouched for by their
trail, not re-measurable.

## Master test plan v3 (2026-08-25) — dual-system execution

### Execution systems: every engine-driven case runs on BOTH feeds
The plugin now has two market-data systems (`feed_mode = "ohlc" | "tick"`, #37).
Policy (operator decision 2026-08-25): **every engine-driven live case runs as a
dual-mode pair** — one leg per feed — once the tick-delivery gate case (below)
has passed once. Rationale: all staged scripts are `calc_on_every_tick=false`,
so their behaviour must be equivalent in both modes; the two legs form a
**differential oracle** — any divergence between the legs' venue records is
automatically a finding, sharper than grading either leg alone.

- **Exempt**: L0 and every direct probe (T14/T15/T17/T18/T33) — they hit the
  client directly and consume no bars; there is nothing feed-dependent to dualize.
- **Expected divergence carve-out**: the session close. OHLC mode never delivers
  the final candle (withheld +903 s, Live-L4-T03); tick mode synth-closes it
  LOUDLY. Close-spanning dual runs grade this difference as EXPECTED.
- **Grading**: registry status becomes two-dimensional (`✅ ohlc <date> / ✅ tick
  <date>`); legs run sequentially in one window, ohlc first as control, config
  flipped between legs by the (planned) `run_dual.sh` wrapper.
- **Budget**: the tick leg adds ~1,800 req/h on `/trades/latest`'s own 10k/h
  bucket at the 2 s default — fine at n=1. Fleet ceiling on one key: FIVE
  tick strategies at 2 s (six breach), TWO at the 1 s floor (#40) — corrected
  2026-08-26, the ceiling-of-two had been the 1 s panel figure.

### Scheduling constraint — the #51 venue window (CRITICAL)
Conditional-book writes are venue-refused (misleading `INVALID_TRADING_TOKEN`)
after the operator's first EntradeX app trade of the day (#46/#51, unified
timeline on #51; rules in the repo CLAUDE.md). **Schedule L0 and every
conditional-involving live case pre-open (~08:20) or in an operator-idle
window.** Do not re-mint on that signature; do not retry into the window.

### Planned cases (v3 queue)
| Planned case | What it measures | Gate/priority |
|---|---|---|
| **Live-L1-T33-CrossProcessCancel** (direct probe) | process A places a far conditional and STAYS ALIVE; process B (`venue.py cancel`, no records) cancels it — exercises #45's probe-all-books fallback live AND measures the #51 window boundary (pre-open: must succeed; post-app-trade: measures the refusal) | first pre-open slot |
| **Live-L4-T04-TickDelivery** | tick mode delivers forming updates + closes with who-closed-each-bar grading (official within `tick_close_timeout` vs loud SYNTH fallback), 429 transitions logged; THE GATE for the dual-mode policy | first pre-open slot after T33 |
| **Live-L1-T27-TickDedup** (live_t27_tick_dedup.pine, script pre-exists) | print-dedup behaviour under the REAL tick feed — runnable only now that #37's tick mode exists | after L4-T04, same window |
| **Live-L4-T05-SynthParity** | recorder over a live window: synth bar captured at each rollover BEFORE official replacement → distributions of close-delta, H/L undershoot, V undershoot (the latest-only endpoint makes exact parity impossible BY CONSTRUCTION — this measures the error bound a calc_on_every_tick author must know) | after T04 |
| **L1 passive row: DriftDetector (#48)** | self-grades on every run the operator holds a position through; first live PASS 08-25 (2 correct warnings, 0 spam, T32 log) | passive, every session |
| **L0 tooling update** | classify the #51 signature as "blocked by venue window — reschedule pre-open" instead of generic FAIL (still exits non-zero) | with T33 |
| **Registry backfill: T19–T31** | the param-matrix states (T19-T30, measured 08-17/18, params_a/b evidence) AND T31 (live_t31_direction_gate.* exists with no registry row at all — found 08-25) have no rows; add them, and sweep all side-annotations for rot (the stale '#39 blocked' F-rows showed annotations decay even when statuses hold) | housekeeping |

### Cases deliberately WITHOUT live coverage (decided, not forgotten)
- **#43** (pending-OCO drain) and **#47** (cancel-with-outcome multi-id): their
  triggers are venue races (stale OCO replica at place time; ambiguous cancel
  disposition) that cannot be forced from outside — fake-venue red-first tests
  are primary evidence. The F-ladder protocol gains: if the `#43` "queued for
  poll-loop adoption" line ever appears in a live run, grade that run's venue
  record against the drain design as a bonus measurement.

## Canonical test registry — the ONLY names to use

Every live test case has one ID: **`Live-L<level>-<case>`**. Logs, plans, cards and
conversation all use these. (Script-internal log tags map as: `[L1] TEST n` -> Live-L1-Tnn,
`[F] Fn` -> Live-L3-Fnn.)

| ID | What it does | Live status |
|----|--------------|--------------------------|
| **Live-L0-Gate** | venue-semantics probe; MUST pass before every live run | ✅ passes (run per session) |
| **Live-L1-T01-LongLimitCancel** | long limit −5% place→cancel | ✅ 08-13, re-verified 08-17 |
| **Live-L1-T02-ShortLimitCancel** | short limit +5% place→cancel | ✅ 08-13, re-verified 08-17 |
| **Live-L1-T03-LimitWithStopExit** | limit + `exit(stop)`; both cancelled explicitly | ✅ 08-13, re-verified 08-17 |
| **Live-L1-T04-OcaCancelEntryOnly** | OCA entry+exit, cancel entry only | ✅ 08-13/14/17 (found #19; post-revert the exit orphan is EXPECTED — clean it manually) |
| **Live-L1-T05-NativeOcoCancelEntryOnly** | native-OCO exit, cancel entry only | ✅ 08-14/17 (forced cascade revert; orphan now expected) |
| **Live-L1-T06-AmendNormal** | amend on the NORMAL book | ✅ 08-14, re-verified 08-17 |
| **Live-L1-T07-AmendConditional500** | conditional amend → HTTP 500 park (#18) | ✅ 08-14, re-verified 08-17 |
| **Live-L1-T08-CancelAll** | `cancel_all()` across both books | ✅ 08-14, re-verified 08-17 |
| **Live-L1-T09-OrderFn** | `strategy.order()` routing | ✅ 08-14, re-verified 08-17 |
| **Live-L1-T10-DualStrategy** | two engines, one account (`run_t10_dual.sh`) | ✅ 08-14, re-verified 08-17 |
| **Live-L1-T11-OcaCancelMember** | `oca.cancel` ×3 across books; cancel one member — siblings must remain | ✅ **08-17** — cancel of one member swept nothing; cross-book group intact |
| **Live-L1-T12-OcaReduce** | `oca.reduce` pair rests full qty | ✅ **08-17** — both rested full qty; cancel_all swept both |
| **Live-L1-T13-OcaNone** | `oca.none` shared name = independent | ✅ **08-17** — sibling untouched by member cancel |
| **Live-L1-T16-RestartAdoption** | place-and-hold, kill engine, relaunch: the resting order must be adopted/quarantined EXPLICITLY and be reachable by cancel_all | ⚠️ **08-18 MEASURED — BUG FOUND**: TERM leaves the order at the venue (no shutdown sweep ✓); DIFFERENT-label relaunch correctly ignores it (T10 isolation ✓); SAME-label relaunch gives NO adoption line and cancel_all does NOT reach it → stranded working order, manual cleanup. Root cause (corrected 08-18 review): the store seam EXISTS and was active (broker.sqlite; same-label relaunch shares run_id) — but ordinary orders are never journaled (orders table holds only software-watch rows, order_refs empty) and idempotency=SOFTWARE sends no coid to the venue, so restart recognition has no bridge. Card #36 |
| **Live-L1-T14-AtcCancelRefusal** | place late CONT-PM, cancel DURING ATC → venue must refuse every attempt; both orders expire at close. Direct probe (`direct_probes_t14_t15_t17.py --case t14`, launch 14:20–14:29) — Live-L4-T03 proved no bars arrive in ATC, so Pine cannot fire the cancel | ✅ **08-18 provisional PASS** — 8/8 attempts refused across the full ATC (14:30–14:44), all CANNOT_CANCEL_THE_ORDER_IN_THE_ATC_SESSION http 400, zero accepts; orders 538916/538926 both **Expired** (fill=0) per venue history 08-19 — FINAL PASS. Operator-cancelled 1st-launch pair reads Canceled, confirming cancels work in CONT-PM and are refused only in ATC |
| **Live-L1-T15-CancelReplace** | #18 evidence: conditional cancel-ACK → immediate replacement on the STOP book; measures the exposure gap (`--case t15`, continuous/lunch) | ✅ **08-18** — ack 35 ms, replacement accepted +99 ms, A=Canceled B=New first poll: cancel+replace IS a viable amend substitute (#18) |
| **Live-L1-T17-ReplaceUnderAckLag** | identical NORMAL LO re-placed inside the ~10 s stale-replica window after a cancel ACK; final state must be exactly ONE working (`--case t17`, continuous) | ✅ **08-18** — exactly one working at every read; found ORDER_CANCEL_STATUS_REJECTED (double-cancel on NORMAL = terminal, classifier fixed) |
| **Live-L1-T18-ImmediateCancel** | cancel the instant the order is visible on the live book — no bar-clock wait (`--case t18`, both books) | ✅ **08-18** — NO minimum-rest rule: NORMAL visible 154 ms/gone 61 ms after cancel-ack 31 ms; STOP visible 101 ms/gone 197 ms. The next-candle wait was Pine's bar clock, not the venue |
| **Live-L1-T32-AtoProbe** | can a NORMAL limit and a conditional stop be PLACED during ATO/POST-ATO (08:45-09:15)? First-ever measurement (`live_staged_params.py startState=13`; bars don't arrive during pure ATO — POST-ATO 09:00-09:15 is the earliest a bar can carry the phase) | ✅ **08-25 PASS** — both order types ACCEPTED at the earliest POST-ATO bar (09:00:00), neither refused; swept clean by cancel_all() at CONT-AM (09:15:00), both Canceled per venue record. Feeds #28 (native ATC/ATO order-type support) — plain order types already work without special handling at the open |
| **Live-L2-SingleFill** | one market fill → flatten (`l2_fill_flatten`) | ✅ 08-12 |
| **Live-L2-BracketFill** | fill + TP/SL bracket → flatten (`l2b_…`) | ⚠️ 08-12 partial |
| **Live-L3-F01-LongMarket** … **F02-ShortMarket** | market fills | ❌ NOT RUN — parked by the FILL-tier precondition (flat account / sub-account); the plain-LO fill path itself is proven (F10) |
| **Live-L3-F03-LongStop** … **F04-ShortStop** | stop-entry fills | ❌ NOT RUN — the old "blocked by #39" reason is RESOLVED (fixed 08-19; F11 passed live on the fix, stop-child adoption proven); parked ONLY by the FILL-tier precondition |
| **Live-L3-F05-LongStopLimit** … **F06-ShortStopLimit** | stop-limit fills (#14 evidence) | ❌ NOT RUN — #39 block RESOLVED 08-19; parked ONLY by the FILL-tier precondition |
| **Live-L3-F07-OcaLongBreak** … **F08-OcaShortBreak** | OCA sibling-cancel on a real fill | ❌ NOT RUN — #39 block RESOLVED 08-19 (F11 = the live proof of exactly this chain); parked ONLY by the FILL-tier precondition |

| **Live-L3-F10-CrossedStopAtPlacement** | buy-stop trigger already BELOW market → must fill immediately (`live_crossed_stop.pine`; oracle measured 08-18: fills at next open; fix #34: plugin detects crossed trigger via last 1m close → marketable NORMAL LO, fail-open to conditional) | ✅ **08-18 PASS live** — [BROKER] 'crossed stop at placement -> immediate marketable LO', order 377636 FILLED @1876.3 (bars-to-fill=1), close 379966 FILLED @1876.9, venue flat. #34 fix verified end-to-end |
| **Live-L3-F11-OcaEntryGroupCancel** | oca.cancel ENTRY pair: near fills → far leg must be CANCELLED by the engine cascade (`live_oca_entry_group.pine`; oracle measured 08-18: backtest cancels it; fix #33 flipped oca_cancel→SOFTWARE) | ✅ **08-19 PASS** (after the #39 fix) — full chain live: `conditional ACTIVATED -> tracking child 49236`, `event FILLED id=49236 pine='NEAR'`, engine `position size=1.0`, then `cancelling CANCEL id='FAR'` → venue `FAR = Canceled`. **Verifies BOTH #39 (stop-entry fills visible) and #33 (oca cascade on a real fill).** Script safety fixed same day: held the position 2 bars 'to observe' → now flattens on sight + protection armed at placement |
| **Live-L4-T01-BarParity** | live @1/@3/@5 bars vs the venue's delayed OHLC + 1m aggregation cross-check (`level4_data_parity/`) | ✅ 08-17 smoke + PM-reopen chunk (exact match incl. aggregation; evidence `level4_data_parity/logs/l4_20260817_1310.json`) |
| **Live-L4-T02-BarLatency** | arrival delay + REST RTT per bar; red lines: bar later than next close, upward drift | ✅ 08-17 smoke + PM-reopen (≤4 s, no red-line hit; drift heuristic false-alarmed on a poll-phase plateau — probe fix queued on #24) |
| **Live-L4-T03-AtcBarDelivery** | does `watch_ohlcv` deliver bars DURING ATC (14:30–14:45) and for the auction print after close? Same recorder run over the 14:22–14:50 edge; feeds Live-L1-T14's bar-delivery caveat and the "can we trade the 14:30/14:45 candle" question | ✅ 08-17 MEASURED (3 chunks 14:22–14:50 + direct probes): NO bars during ATC on any TF; the session-final candle (@1 14:29 / @5 14:25) is WITHHELD and published together with the 14:45 ATC print at auction settlement (+903 s measured); with the 5-bar poll lookback a live @1/@3 strategy NEVER receives the final candle at all. Last actionable live bar: @1 14:28 arriving ~14:29. Chart feed ≠ OpenAPI feed at the close. Feeds T14 (needs direct-probe fallback for cancel-in-ATC) and #28/#29 |

Live-L3 backtest mode (past window) is the ORACLE — a pre-launch gate, never a live result.
Live-L3 requires a FLAT account. Live-L4 is PASSIVE (no orders, no trading token) but must
not run concurrently with order-placing tests (rate-limit contamination).

### FILL tests: the OPERATOR closes the position (protocol from 2026-08-19)

Pine's bar clock means a strategy cannot react to its own fill until the next bar
closes — measured 60–90 s (F10: filled 13:26, could only flatten 13:28), and F11's
earlier "observe 2 bars" design stretched that to ~3 minutes of UNPROTECTED
exposure until the operator flattened by hand. Faster polling does not help: the
plugin sees the fill in ~0.5 s, the *strategy* still waits for the bar.

All three live fill tests — the staged ladder, F10 and F11 — run
**operator-in-the-loop** (`operatorCloses` input, default `true`):

1. The strategy places the stage and, on seeing the fill, **cancels the resting
   legs only** (a cancel can never open a position) and SHOUTS
   `>>> OPERATOR: CLOSE THIS POSITION NOW <<<`.
2. **You close it in the DNSE app, ~3 s after the fill.** The strategy never sends
   a close — issuing one against a stale view *after* a manual close would
   REVERSE the position.
3. The stage advances when the engine sees flat. If it does not within
   `CLOSE_WAIT_BARS` (5), the run HALTS once, cancels everything, and says so:
   either the position is still open, or **the engine failed to detect an external
   close — which is itself a finding worth recording**.
4. `operatorCloses = false` restores self-closing for the BACKTEST oracle (no
   operator exists there); the oracle still completes all 8 stages.
5. **The protective exit stays armed until the position is actually flat** — it is
   the only cover during the operator's close window — and is retired immediately
   afterwards so it cannot linger as an orphan stop that later OPENS a position.
6. **F11 never cancels its FAR leg.** That leg's venue state IS the measurement of
   the engine cascade; a script-side sweep would make the test unfailable (#42-B).
   If the cascade is broken, FAR is left working for the operator to cancel by
   hand after grading.

Preconditions unchanged: FLAT account (or a dedicated sub-account), supervised,
DNSE app open — plus: **do not trade this account manually while a fill test runs**
(a netting venue cannot distinguish your fills from the test's).

### Poll cadence and the rate-limit budget

DNSE limits are **per API key AND per endpoint**, with two thresholds (requests/hour
and quota/day) — see `docs/dnse-openapi-documentation/guide-ratelimits.md`:

| Feed | Endpoint limit | Our default | One strategy | Ten strategies |
|---|---|---|---|---|
| Order books (2 req/cycle) | 100,000/h · 1M/day | **0.5 s** | 14,400/h = 14% | 144,000/h = **OVER** |
| Bars (1 req/cycle per symbol+TF) | 50,000/h · **100,000/day** | 3 s | 7,200/day = 7% | 72,000/day = **72%** |

Both are `DNSEBrokerConfig` fields (`order_poll_interval`, `bar_poll_interval`) —
raise them before running a fleet on one key. The budget is shared, so N strategies
cost N×, and the duplication is total (order books are account-wide and identical
for every strategy; same-symbol+TF bar polls are byte-identical). Pooling one poll
per asset/account is the prerequisite for a fleet — not yet built.

**Failure mode if exhausted:** 429s are treated as transient poll failures — bars
and fills silently stop arriving while strategies keep looping. No crash, no alarm.

Live smoke 2026-08-19: 98 order-book requests at 0.5 s → 98×200, zero 429, RTT
median 55 ms.

### Grading orders from a PREVIOUS day

The current-day detail endpoint returns `orderStatus: None` for yesterday's orders.
Use the history endpoint instead: `GET /accounts/{accountNo}/orders/history`
(`?marketType=&orderCategory=&from=YYYY-MM-DD&to=YYYY-MM-DD`) — note it answers with
a **`data`** array (not `orders`) and ids are **date-prefixed** (`20260818_538916`).
Measured 2026-08-19 while grading Live-L1-T14.

## The four test types

| # | Type | Command | Needs |
|---|------|---------|-------|
| 1 | **Unit + offline e2e** (incl. fake-venue replay of measured quirks, contract probe) | `pytest plugins/dnse/tests/` | nothing |
| 2 | **L0 venue-semantics gate** | `.venv/bin/python plugins/dnse/testing/live_test/level0_venue_semantics/l0_order_semantics.py` | token; any placement-capable hour |
| 3 | **Staged no-fill probe** T1–T9, T11–T13 (+ T10 dual-runner) | `pyne run … live_staged_place_cancel.py dnse_broker:VN30F1M@1 --broker` | token + open session |
| 4 | **Staged fill test** F1–F8 — backtest mode doubles as the oracle | `pyne run … live_staged_fill.py` (`dnse:` = oracle, `dnse_broker: --broker` = live) | live: token + session + **FLAT account** |

**Rules (non-negotiable):**
- **L0 must pass (exit 0) before EVERY live run.** It catches an expired token, a stale
  contract mapping after the monthly roll, and changed venue behaviour in ~30 s.
- **Offline dry-run before live**: point the trade window at a past session; the whole
  state machine must replay with **zero fills** (type 3) / balanced round-trips (type 4).
- **Never `--from`** (truncates the shared `.ohlcv` cache — see repo CLAUDE.md).
- Grade from the venue record; wait ≥1 bar after a flatten before judging "no orphan"
  (teardown lags one sync).

## The staged probes

Both are driven by a `var int state` machine with three `.toml` inputs:
`winStart`/`winEnd` (epoch-ms trade window — **live runs need `winStart` after your
launch time**, or warmup bars consume the stages) and `startState` (resume at any case).

### `live_staged_place_cancel.pine` — no-fill, T1–T13

Every order 1 contract, ≥4.5 % from market: rests, cannot fill.

| Test | State | Case |
|------|-------|------|
| T1/T2 | 0/1 | long/short limit place → cancel |
| T3 | 2 | limit + `exit(stop)` — exit reaches the venue **pre-fill**; both cancelled explicitly |
| T4 | 3 | OCA entry+exit, cancel **entry only** |
| T5 | 4 | native-OCO exit (umbrella book), cancel entry only |
| T6 | 5 | **amend** on the NORMAL book (place → re-issue at a new price → cancel) |
| T7 | 6 | **amend on a conditional → HTTP 500** (#18): must park+verify, not crash, and still cancel |
| T8 | 7 | `strategy.cancel_all()` across both books |
| T9 | 8 | `strategy.order()` routing |
| T11 | 9 | `oca.cancel` ×3 across both books — cancel one member, siblings must **remain** (OCA fires on fill, not cancel) |
| T12 | 10 | `oca.reduce` pair — both rest **full qty** (reduce acts only on a fill) |
| T13 | 11 | `oca.none`, shared name — fully independent |

**T10** (dual-strategy isolation) is its own runner: `run_t10_dual.sh [lead_minutes]` —
two concurrent engines on one account, B fires `cancel_all()` while A's order rests;
PASS = zero cross-contamination, both orders `Canceled` at the venue. Exit 0 = PASS.

### `live_staged_fill.pine` — fills, F1–F8 (= Level 3, oracle included)

One file, two modes; oracle and live test are the same byte-identical state machine.
Per stage: place (+protection exit armed at placement) → fill → flatten (`close` +
**explicit** cancels of exit/sibling — never rely on a cascade, #19) → flat → advance.
6-bar fill timeout: cancel, retry once, then SKIP with a log.

F1/F2 market · F3/F4 stop · F5/F6 stop-limit (the #14 evidence) · F7/F8 OCA where the
near leg fills and the far leg must be CANCELLED.

**Live preconditions:** flat account, supervised, DNSE app open for one-tap flatten.

## Measured venue facts (why the tests look like this)

- **A 2xx cancel is an ACK, not a completion** — the venue can report `New` for >12 s
  after `200 OK`; the plugin re-reads until the venue agrees (#20, fixed + live-verified).
- **DNSE does not cascade** an entry cancel to its exit legs — and a *plugin*-side cascade
  breaks the engine's ownership model (quarantine + re-placed orphan, measured; #19 is an
  ENGINE fix, pending). Interim rule: cancel exit ids explicitly.
- **Conditional amend → HTTP 500** (#18); NORMAL amend works. The engine parks and the
  order stays cancellable; the venue keeps the OLD level.
- **GTD must be a future day within the contract** — past `finalTradeDate` →
  `CO-ORD-006` (fixed: clamped to midnight UTC of the final trade date).
- **Session phases gate everything**: continuous 09:15–11:30 / 13:00–14:30 ICT (market
  orders FILL), lunch (market orders queue + cancel — the only phase L0 runs its market
  part), ATC 14:30–14:45 (**cancels refused**, resting orders fill in the auction),
  closed (nothing placeable at all).
- Two id shapes: NORMAL book = int ids, conditional book = string ids; a cancel against
  the wrong book 404s (`RESOURCE_NOT_FOUND`) — that's how the plugin probes.
- Exits reach the venue **before** their entry fills.
- **Detail reads can be stale/non-monotonic** — a Canceled order was served as `New`
  by a lagging replica ~10 s later (08-17); re-read before concluding a cancel failed.
  DNSE's cancel is **idempotent**: 200+record on an already-Canceled conditional, not
  `CO-ORD-013`.
- **A SIGKILLed run leaves a heartbeat row**; the engine's single-instance guard then
  blocks relaunch ("Active run_id already exists"). **Wait out the heartbeat-stale
  window and resume with the SAME label** — #36's journal adoption re-owns the resting
  orders only under the same run identity. A NEW `--run-label <x>` changes `run_id`
  and STRANDS the previous run's orders (they are reported loudly at startup and never
  adopted — #71/#60; the old recipe of always relaunching under a new label was the
  very thing that defeated recovery).

## History / older docs

`LIVE_TEST_PLAN.md` (L1 live1–3), `LIVE_TEST_PLAN_L2.md` (l2/l2b) and
`../TEST_PLAN.md` are the earlier-generation plans this suite grew from; superseded
material (t1–t8 oracle strategies, l3a–h matrix, graduated probes) is parked in
`backup/deleteable/superseded_20260814/`. Findings live on the cards:
rubycell/pynecore#16 (campaign log), #18, #19, #21.
