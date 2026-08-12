# DNSE broker — LIVE TEST LEVEL 2 (first real fill)

**Gated on Level 1.** Do NOT start this until `live1`/`live2`/`live3` (place → rest →
cancel, no fills — see `LIVE_TEST_PLAN.md`) have all passed on the live account.

Level 1 proved the write/cancel path with **zero position risk**. Level 2 is the first tier
that takes a **real position**: enter **one** contract (a real fill), then flatten it the
instant the fill is seen. Everything below exists to make that one contract's exposure as
small and as bounded as possible.

> **This tier is SUPERVISED. Never run it unattended.** An operator watches the whole run
> with the DNSE app open for a one-tap manual flatten.

## Risk model — what is actually at stake

- **One** VN30F1M contract: ~135 M VND notional, ~25 M VND margin (18.48 %).
- Held **~1 bar** (see timeframe below), then flattened.
- Realistic worst case in that window: a few index points against us ≈ tens of thousands of
  VND, plus spread + fees on the round-trip. The tail risk is a gap while held — which
  **Level 2b** closes with a venue-side stop.
- The exposure is bounded no matter what by: **1 lot**, an **enter-once latch**, a
  **realtime gate** (warmup never trades), a **targeted close** of our own entry, and a
  **supervising operator** with a one-tap abort.

## Hard preconditions (every one must hold before you go `--broker`)

1. **Level 1 passed** — all three place/cancel strategies behaved on the live account.
2. **Token GOOD** — `python plugins/dnse/tools/token_status.py` prints GOOD.
3. **Continuous session** — 09:15–11:30 or 13:00–14:25 ICT. NOT an auction (09:00 ATO,
   14:30 ATC). Early afternoon is the calmest window to watch a full cycle.
4. **FLAT account** — no open position, no working orders. This is not optional: with an
   external/adopted position present, `position_size` is nonzero from bar one and the
   flatten logic races. If the pre-existing external long is still there, close it (or
   confirm it's gone) first. Verify with a read-only positions/orders probe.
5. **Manual abort ready** — the DNSE mobile/web app open and logged in, so any position can
   be closed in one tap. This is the ultimate backstop; have it up *before* you start.

## The strategy — `l2_fill_flatten.pine`

```pine
var bool traded  = false
var bool closing = false

// (1) the single real entry — ONLY on the live bar, never during historical warmup
if barstate.isrealtime and not traded
    strategy.entry("E", strategy.long, comment="MKT 1-lot")
    traded := true

// (2) flatten OUR entry the instant a position exists (the "cancel immediately" step)
if traded and strategy.position_size != 0 and not closing
    strategy.close("E", comment="FLATTEN")
    closing := true
```

Grounded against the code, not assumed:
- `barstate.isrealtime` is set `True` only on live bars (`security_process.py`,
  `script_runner.py:1981`) and is `False` in warmup/backtest — so entry (1) can only fire
  live, exactly once.
- A price-less `strategy.entry` → a **marketable LO at the band edge** (`broker.py:15`):
  deterministic fill, slippage capped at ±7 %.
- `strategy.close("E")` → `execute_close` with software-enforced `reduce_only`
  (`broker.py:478,182`): reduces only our entry, can't flip or touch an external position.

## Step A — backtest gate (this proves warmup safety)

```bash
cd /home/mike/workspace/github/pine2pyne
.venv/bin/python -m pine2pyne \
  /home/mike/workspace/github/pynecore/plugins/dnse/testing/live_test/l2_fill_flatten.pine \
  -o /home/mike/workspace/github/pynecore/plugins/dnse/testing/live_test/l2_fill_flatten.py

cd /home/mike/workspace/github/pynecore
.venv/bin/pyne run plugins/dnse/testing/live_test/l2_fill_flatten.py dnse:VN30F1M@3
```

**Expected: ZERO trades, clean exit.** Because `barstate.isrealtime` is always `False` in
backtest, the entry never fires. A quiet backtest here is not a null result — it is the
proof that historical warmup cannot place a real order. If the backtest shows *any* trade,
STOP: the gate is not holding, do not go live.

## Step B — run live (supervised, background)

```bash
.venv/bin/pyne run plugins/dnse/testing/live_test/l2_fill_flatten.py \
  dnse_broker:VN30F1M@3 --broker
```

Watch the log in real time. Expected sequence:
1. Warmup bars replay silently — **no orders** (isrealtime False).
2. First live bar → **one** ENTRY dispatched (marketable LO) → `CREATED` → **Filled**,
   position → +1.
3. Next live bar → `close("E")` dispatched → **Filled** → position → **flat (0)**.
4. No further orders — the `traded`/`closing` latches hold. Round-trip PnL ≈ −(spread+fees).

Stop the run once you've seen fill → flatten → flat.

## Step C — verify & clean up

- Confirm on the venue: the account is **flat**, **no working orders** remain.
- Confirm the log shows exactly **one** entry and **one** close — no re-entry.
- If anything is left resting or open, flatten/cancel it (app one-tap, or a probe cancel).

## Abort / rollback (if it misbehaves)

- **Primary:** close the position in the DNSE app — one tap, most reliable.
- **If the process dies between fill and flatten:** in 2a the position is briefly
  *unprotected* — flatten it manually at once. (This exact window is what 2b removes.)
- Kill the run (`TaskStop`/Ctrl-C), then reconcile the account by hand to flat.

## Timeframe

**3 m recommended** — hold ≈ 1 bar ≈ 3 min: short enough to keep exposure small, long
enough to avoid the 1 m poll/fill/exit raciness seen in earlier live runs. **5 m** is the
conservative choice (matches prior `--broker` runs) at the cost of a ~5-min hold. Avoid
1 m for a position-holding test.

## Success criteria

- [ ] Backtest (Step A) places **zero** trades and exits clean (gate proven).
- [ ] Live: exactly **one** entry fills → position +1; then flattens to **0** on the next
      bar; **no** re-entry.
- [ ] Round-trip PnL ≈ −(spread + fees); no surprise size (never > 1 lot).
- [ ] Account ends **flat** with **no** working orders; any external position untouched.

---

## Level 2b (optional next step) — venue-protected fill

Same as 2a, but the entry also drops a **venue-side protective stop** the moment it's
placed, so a crash/hang between fill and flatten is still covered *at the exchange*. It also
exercises exit-order placement **and** its cancellation when we flatten.

```pine
if barstate.isrealtime and not traded
    strategy.entry("E", strategy.long, comment="MKT 1-lot")
    strategy.exit("X", from_entry="E", stop=close * 0.995, comment="PROT -0.5%")  // native STOP at venue
    traded := true

if traded and strategy.position_size != 0 and not closing
    strategy.close("E", comment="FLATTEN")   // flatten + the engine cancels the resting stop "X"
    closing := true
```

- `strategy.exit(stop=…)` → a native STOP resting at the venue (`broker.py:14,455`) ≈0.5 %
  below entry — the crash backstop that 2a lacks.
- On flatten, the engine cancels the orphaned STOP; **verify that cancel lands** (no stray
  resting stop left behind). Run 2b on **5 m** since the venue stop covers the longer hold.

## Out of scope (Level 3+)

Holding a position across many bars; take-profit / partial exits; a full OCO exit bracket
riding a real fill; reversal (long→short) on a live position. Each is a later, separately
planned tier.
