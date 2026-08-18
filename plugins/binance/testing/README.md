# Binance broker — live-test registry (testnet)

Same discipline as `plugins/dnse/testing/live_test/README.md`: pytest is the
only network-free suite; everything live runs on the **spot testnet**
(`sandbox = true`, keys from https://testnet.binance.vision) behind the L0
gate. The broker refuses mainnet without `allow_mainnet = true`. Every live
verdict is graded from **Binance's own records** (fetch_orders / myTrades /
balances), never from the run log alone.

## Canonical test registry — `Live-B<case>` names only

Spot is LONG-ONLY (`short_selling` unsupported, enforced at startup), so the
DNSE suite's short-side cases (T2-short, F2/F4/F6/F8) are N/A here; each B
case notes its DNSE ancestor.

| ID | What it does (DNSE ancestor) | Live status |
|----|-------------------------------|-------------|
| **Live-B0-Gate** | `tools/l0_gate.py` — auth, clock skew, filters, book, balance; MUST exit 0 before every live run | ✅ 08-17 (skew 96–226 ms) |
| **Live-B1-T01-LongLimitCancel** (T1) | long limit −5% place→cancel | ✅ 08-17 |
| **Live-B1-T02-LongLimitCancel4** (T2, long-only) | long limit −4% place→cancel | ✅ 08-17 |
| **Live-B1-T03-LimitWithStopExit** (T3) | limit + `exit(stop)`; MEASURED: the pre-fill exit **reaches the venue** (sell STOP_LOSS_LIMIT rests against base inventory); both cancelled explicitly | ✅ 08-17 |
| **Live-B1-T04-OcaCancelEntryOnly** (T4) | OCA entry+exit, cancel entry only — X4 exit stayed working (orphan, engine #19 shape); swept by the T08 cancel_all | ✅ 08-17 |
| **Live-B1-T05-NativeOcoCancelEntryOnly** (T5) | entry + `exit(tp+sl)` → NATIVE spot OCO (LIMIT_MAKER + STOP_LOSS_LIMIT legs confirmed at venue), cancel entry only — OCO legs stayed (orphan, ditto) | ✅ 08-17 |
| **Live-B1-T06-AmendNormal** (T6) | limit re-issue → cancel+replace, fresh venue id, same coid | ✅ 08-17 |
| **Live-B1-T07-AmendConditional** (T7) | stop re-issue → cancel+replace clean (no DNSE-#18 500 analogue) | ✅ 08-17 |
| **Live-B1-T08-CancelAll** (T8) | `cancel_all()` — swept both fresh orders AND the T04/T05 orphans; expected-cancel sink kept quarantine quiet | ✅ 08-17 |
| **Live-B1-T09-OrderFn** (T9) | `strategy.order()` routing identical to `entry()` | ✅ 08-17 |
| **Live-B1-T11-OcaCancelMember** (T11) | `oca.cancel` ×3; cancel one member — venue snapshot shows exactly the two siblings still NEW | ✅ 08-17 |
| **Live-B1-T12-OcaReduce** (T12) | `oca.reduce` pair rests full qty; cancel_all sweeps | ✅ 08-17 |
| **Live-B1-T13-OcaNone** (T13) | `oca.none` shared name = independent | ✅ 08-17 |
| **Live-B2-F01-MarketFill** (F1) | market buy → fill → flatten; ledger position synthesis live | ✅ 08-17 |
| **Live-B2-F03-StopFill** (F3) | stop entry; crossed-at-placement → **market fallback** (found+fixed this run); rested stop fills through trigger | ✅ 08-17 |
| **Live-B2-F05-StopLimitFill** (F5) | stop-limit = ONE STOP_LOSS_LIMIT carrying its cap; crossed → falls back to LIMIT at the cap (post-run fix, unit-tested) | ✅ 08-17 |
| **Live-B2-F07-OcaEntryBreak** (F7) | entry-OCA near/far: near filled; **engine did NOT sibling-cancel under `oca_cancel=NATIVE`** — far leg survived until the script's explicit cancel → capability corrected to SOFTWARE | ✅ 08-17 (finding) |
| **Live-B2-F09-OcoBracketResolve** (new) | market fill → native OCO bracket → TP leg FILLED, venue auto-cancelled the SL sibling (status **EXPIRED**); engine observed the external cancel without quarantining | ✅ 08-17 |

## The four test types

| # | Type | Command |
|---|------|---------|
| 1 | Unit + offline e2e (fake ccxt venue, contract probe) | `pytest plugins/binance/tests/` |
| 2 | L0 venue gate | `.venv/bin/python plugins/binance/tools/l0_gate.py` |
| 3 | Staged no-fill probe B1–B13 | `pyne run plugins/binance/testing/live_staged_place_cancel.py binance_broker:BTC/USDT@1 --broker` |
| 4 | Staged fill test BF1–BF9 — backtest mode over a past window IS the oracle | `pyne run plugins/binance/testing/live_staged_fill.py …` (`binance_BINANCE_BTC_USDT_1` = oracle, `binance_broker:… --broker` = live) |

**Rules (inherited from DNSE, non-negotiable):** L0 exit 0 before EVERY live
run · offline oracle dry-run first (no-fill probe must replay with ZERO fills;
fill suite with balanced round trips) · never `--from` · grade from the venue
record · live `winStart` must be after launch.

Both staged scripts are driven by `winStart`/`winEnd`/`startState` in their
`.toml` (state machine identical in oracle and live mode; `startState` resumes
after a mid-run fix — used live on 08-17).

## Measured venue facts (why the code looks like this)

- **-2010 "Stop price would trigger immediately"**: Binance REJECTS a stop
  whose trigger is at/crossed by the market (DNSE would accept + activate).
  Pine semantics for a crossed stop = immediate fill → the plugin falls back
  to MARKET (plain stop) / LIMIT at the cap (stop-limit). Measured BF3/BF5.
- **Entry OCA groups have NO venue link** — only exit tp+sl pairs are
  venue-run (spot OCO orderList). `oca_cancel` must be SOFTWARE or the far
  entry leg survives its sibling's fill (measured BF7).
- **The auto-cancelled OCO sibling reports `EXPIRED`**, not `CANCELED`
  (order 3843764); the TP LIMIT_MAKER leg fills at its exact price.
- **Cancel is synchronous** (~1 poll cycle to CANCELED) and **idempotent
  coid reuse is allowed** once the previous bearer is terminal (amend's
  cancel+replace reused the coid, venue accepted).
- The runner derives run identity BEFORE `connect()` → `account_id` resolves
  lazily in the property.
- A pre-fill `strategy.exit` reaches the venue and rests against base
  inventory (needs base balance on spot; testnet grants 1 BTC).
- Entry-only cancels orphan their exit legs (engine #19 shape, same as
  DNSE) — a later `cancel_all` sweeps them; until #19 lands, strategies
  must cancel exit ids explicitly.
- Testnet grants ~10k USDT + 1 BTC + a long tail of assets; book resets
  periodically.

## Evidence (this directory)

`evidence_b1_nofill_2026-08-17.txt` (first probe),
`evidence_staged_nofill_2026-08-17.txt` + `evidence_venue_grade_nofill_…`
(B1–B13: 23 venue orders, ALL CANCELED, 0 fills, book empty),
`evidence_staged_fill_2026-08-17.txt` + `evidence_venue_grade_fill_…`
(BF: 10 FILLED = 5 balanced round trips, net BTC 0.000000, BTC back to 1.0,
+0.06 USDT, book empty).
