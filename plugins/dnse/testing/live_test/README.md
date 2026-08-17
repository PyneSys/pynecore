# DNSE live-test suite

How the DNSE broker plugin is tested, after the 2026-08-10 → 08-14 live campaign.
Four operational test types; every live verdict is graded from **DNSE's own records**
(order details / books), never from the run log alone — the log claimed success twice
while the venue disagreed (see findings below).

## Canonical test registry — the ONLY names to use

Every live test case has one ID: **`Live-L<level>-<case>`**. Logs, plans, cards and
conversation all use these. (Script-internal log tags map as: `[L1] TEST n` -> Live-L1-Tnn,
`[F] Fn` -> Live-L3-Fnn.)

| ID | What it does | Live status (2026-08-17) |
|----|--------------|--------------------------|
| **Live-L0-Gate** | venue-semantics probe; MUST pass before every live run | ✅ passes (run per session) |
| **Live-L1-T01-LongLimitCancel** | long limit −5% place→cancel | ✅ 08-13 |
| **Live-L1-T02-ShortLimitCancel** | short limit +5% place→cancel | ✅ 08-13 |
| **Live-L1-T03-LimitWithStopExit** | limit + `exit(stop)`; both cancelled explicitly | ✅ 08-13 |
| **Live-L1-T04-OcaCancelEntryOnly** | OCA entry+exit, cancel entry only | ✅ 08-13/14 (found #19) |
| **Live-L1-T05-NativeOcoCancelEntryOnly** | native-OCO exit, cancel entry only | ✅ 08-14 (forced cascade revert) |
| **Live-L1-T06-AmendNormal** | amend on the NORMAL book | ✅ 08-14 |
| **Live-L1-T07-AmendConditional500** | conditional amend → HTTP 500 park (#18) | ✅ 08-14 |
| **Live-L1-T08-CancelAll** | `cancel_all()` across both books | ✅ 08-14 |
| **Live-L1-T09-OrderFn** | `strategy.order()` routing | ✅ 08-14 |
| **Live-L1-T10-DualStrategy** | two engines, one account (`run_t10_dual.sh`) | ✅ 08-14 |
| **Live-L1-T11-OcaCancelMember** | `oca.cancel` ×3 across books; cancel one member — siblings must remain | ❌ **NOT RUN** |
| **Live-L1-T12-OcaReduce** | `oca.reduce` pair rests full qty | ❌ **NOT RUN** |
| **Live-L1-T13-OcaNone** | `oca.none` shared name = independent | ❌ **NOT RUN** |
| **Live-L2-SingleFill** | one market fill → flatten (`l2_fill_flatten`) | ✅ 08-12 |
| **Live-L2-BracketFill** | fill + TP/SL bracket → flatten (`l2b_…`) | ⚠️ 08-12 partial |
| **Live-L3-F01-LongMarket** … **F02-ShortMarket** | market fills | ❌ NOT RUN |
| **Live-L3-F03-LongStop** … **F04-ShortStop** | stop-entry fills | ❌ NOT RUN |
| **Live-L3-F05-LongStopLimit** … **F06-ShortStopLimit** | stop-limit fills (#14 evidence) | ❌ NOT RUN |
| **Live-L3-F07-OcaLongBreak** … **F08-OcaShortBreak** | OCA sibling-cancel on a real fill | ❌ NOT RUN |

Live-L3 backtest mode (past window) is the ORACLE — a pre-launch gate, never a live result.
Live-L3 requires a FLAT account.

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

## History / older docs

`LIVE_TEST_PLAN.md` (L1 live1–3), `LIVE_TEST_PLAN_L2.md` (l2/l2b) and
`../TEST_PLAN.md` are the earlier-generation plans this suite grew from; superseded
material (t1–t8 oracle strategies, l3a–h matrix, graduated probes) is parked in
`backup/deleteable/superseded_20260814/`. Findings live on the cards:
rubycell/pynecore#16 (campaign log), #18, #19, #21.
