# DNSE broker — LIVE TEST LEVEL 3 (fill matrix)

**Gated on Level 1 and Level 2.** Do NOT start until place/cancel (L1) and the single
market fill (L2) have both passed on the live account.

Level 3 exercises **every entry order type × both sides**, each as a *fill → protect →
flatten → read-the-log* cycle on **1 m**. Each case takes ONE real contract, fills it, arms
a **protective stop-loss** on the position the moment it fills (learned from `t3`–`t6`),
flattens the instant the fill is seen, and then you grade pass/fail from the run log. The
stop-loss never has to fire — flatten-on-sight closes first — but it is a native venue-side
backstop for the fill-to-flatten window and any crash, and it makes the flatten also prove
the stop-loss is **cancelled** (no orphan).

> **SUPERVISED tier. Never unattended.** Operator watches every run, DNSE app open for a
> one-tap manual flatten.

## The matrix — 8 cases

| # | File | Entry | Expected order in the log | Pos | Flatten |
|---|------|-------|---------------------------|-----|---------|
| a | `l3a_long_market`     | `entry(long)`                       | marketable LO (NORMAL, band-edge) | +1 | `close("E")` |
| b | `l3b_long_stop`       | `entry(long, stop=high[1])`         | `orderCategory=STOP`              | +1 | `close("E")` |
| c | `l3c_long_stoplimit`  | `entry(long, stop=high[1], limit=high[1]*1.0001)` | `STOP` **with a limit price** (NOT OCO) | +1 | `close("E")` |
| d | `l3d_short_market`    | `entry(short)`                      | marketable LO (NORMAL, band-edge) | −1 | `close("E")` |
| e | `l3e_short_stop`      | `entry(short, stop=low[1])`         | `orderCategory=STOP`              | −1 | `close("E")` |
| f | `l3f_short_stoplimit` | `entry(short, stop=low[1], limit=low[1]*0.999)`   | `STOP` **with a limit price** (NOT OCO) | −1 | `close("E")` |
| g | `l3g_oca_longbreak`   | OCA: `Up=stop(high[1])` + `Dn=stop(low[1]*0.95)`  | two grouped STOPs → **UP fills, DN cancels** | +1 | `close_all` |
| h | `l3h_oca_shortbreak`  | OCA: `Up=stop(high[1]*1.05)` + `Dn=stop(low[1])`  | two grouped STOPs → **DN fills, UP cancels** | −1 | `close_all` |

Stops fill when price breaks the prior 1 m bar's high/low — normal within a few bars in an
active session. The OCA cases bias one leg near market (fills) and the sibling far (cancels).

**Every case also arms a protective stop-loss** the moment it fills, via
`strategy.exit("X", from_entry=…, stop=…, comment_loss="SL@…")` — a native venue-side STOP
at **`low[1]` for longs / `high[1]` for shorts** (the `t3`–`t6` pattern; for OCA it rides
the filled leg). It's tagged `SL@<price>` in the log so it's distinguishable from the entry
STOP and the flatten. Flatten-on-sight closes before it can trigger, so the flatten must
also **cancel** it.

## Shared safety harness (identical in all 8 files)

```pine
var bool traded  = false
var bool closing = false
if barstate.isrealtime and not traded      // (1) one entry, LIVE bar only
    <the entry(ies) for this case>
    traded := true
if strategy.position_size > 0              // (2) ALWAYS protect: native stop-loss (t3-t6)
    strategy.exit("X", from_entry="<long id>", stop=low[1],  comment_loss="SL@…")
if strategy.position_size < 0
    strategy.exit("X", from_entry="<short id>", stop=high[1], comment_loss="SL@…")
if traded and strategy.position_size != 0 and not closing   // (3) flatten on sight
    <close("E") | close_all>
    closing := true
// (4) -0.1% PROTECT last line (t1-t8): close_all if open PnL < -0.1%
```

Every guard is grounded in code, not assumed:
- **Enter-once latch** (`traded`) — exactly one entry per run; a Pine entry order *persists*
  until filled/cancelled, so not re-issuing it next bar keeps it resting (the engine won't
  cancel it out from under us).
- **Realtime gate** (`barstate.isrealtime`) — `False` in warmup/backtest, `True` only live
  (`security_process.py`, `script_runner.py:1981`). Historical warmup can NEVER trade ⇒ a
  plain `dnse:` backtest of each file places **ZERO** trades (verified for a/b/f/g). That
  quiet backtest is the safety proof.
- **1 lot, `pyramiding=0`** — never stacks.
- **Targeted flatten** — `close("E")` reduces only our entry (`reduce_only` is
  software-enforced, `broker.py:182`); OCA uses `close_all` (safe under the flat-account
  precondition).
- **Price-less market → marketable LO at the band edge** (`broker.py:15`) — capped slippage.
- **Mandatory stop-loss** — `strategy.exit(stop=…)` → a native venue-side STOP
  (`exit(stop) → STOP`, `broker.py:14`) armed on fill; the venue holds it even if our
  process hangs or dies between fill and flatten. Plus the `-0.1% PROTECT` software line
  from `t1`–`t8` as a second backstop.

## Hard preconditions (all must hold before each `--broker` run)

1. **L1 and L2 passed.**
2. **Token GOOD** — `python plugins/dnse/tools/token_status.py`.
3. **Continuous session** — 09:15–11:30 / 13:00–14:25 ICT. Not an auction.
4. **FLAT account** — no position, no working orders, before *each* case. With any residual
   position, `position_size` is nonzero from bar one and the flatten logic races. Confirm
   flat between cases (the previous case's flatten must have completed).
5. **Manual abort ready** — DNSE app open for one-tap close.

## Per-case procedure

Do them in order **a → h** (markets first: deterministic; then stops; then stop-limits;
then OCA). For each case `<C>`:

```bash
# 1. transpile (local pine2pyne)
cd /home/mike/workspace/github/pine2pyne
.venv/bin/python -m pine2pyne \
  /home/mike/workspace/github/pynecore/plugins/dnse/testing/live_test/level3_fill_matrix/<C>.pine \
  -o /home/mike/workspace/github/pynecore/plugins/dnse/testing/live_test/level3_fill_matrix/<C>.py

# 2. backtest gate — MUST show zero trades + clean exit (proves warmup can't fire)
cd /home/mike/workspace/github/pynecore
.venv/bin/pyne run plugins/dnse/testing/live_test/level3_fill_matrix/<C>.py dnse:VN30F1M@1

# 3. run LIVE on 1m, capturing the log to grade afterwards
mkdir -p plugins/dnse/testing/live_test/level3_fill_matrix/logs
.venv/bin/pyne run plugins/dnse/testing/live_test/level3_fill_matrix/<C>.py \
  dnse_broker:VN30F1M@1 --broker \
  2>&1 | tee plugins/dnse/testing/live_test/level3_fill_matrix/logs/<C>.log
```

Watch the run: entry dispatched → **Filled** → position reaches the expected sign → flatten
dispatched → **Filled** → position **0**. Stop the run once flat.

## Check the log — how to grade (do this after the flatten)

Pull the verdict-relevant lines from the captured log:

```bash
grep -iE "\[BROKER\]|dispatch|orderCategory|order type|SL@|FILLED|CANCELLED|REJECTED|position|error" \
  plugins/dnse/testing/live_test/level3_fill_matrix/logs/<C>.log
```

**PASS** requires ALL of:

- [ ] **Right order sent** — the dispatch matches the "Expected order in the log" column
      (STOP for b/e; STOP **with a limit** and **not** OCO for c/f; two grouped STOPs for g/h).
- [ ] **Filled** — a FILLED event for the entry; position reached the expected sign (±1).
- [ ] **Stop-loss armed** — a protective STOP tagged `SL@…` appears once the position opens
      (`low[1]` for longs, `high[1]` for shorts).
- [ ] **Flattened** — the close dispatched and FILLED; final position **0**.
- [ ] **Stop-loss cancelled** — the `SL@…` STOP shows `CANCELLED` on the flatten (it must
      NOT be left resting, and must NOT have fired ahead of the flatten).
- [ ] **No errors** — no `[BROKER] … ERROR` line and no `REJECTED` for our orders.
- [ ] **No orphans** — nothing left resting. For g/h the losing OCA leg MUST show
      `CANCELLED`; for c/f no second (OCO-style) order should exist.

Record the verdict per case. Any missing tick = FAIL for that case; stop and diagnose
before the next.

## Two cases that also verify open issues

- **Stop-limit (c, f)** — an `entry` carrying **both** `stop=` and `limit=` is the exact
  shape flagged in the backlog (stop-limit vs. OCO ambiguity). The log must show a single
  **STOP with a limit price**, NOT an OCO / dual order. This is the live confirmation the
  backlog item asked for.
- **OCA (g, h)** — the first time a leg actually **fills** live (Level 1's OCA never
  filled). This is the first real test of **one-cancels-other on a fill**: the winning leg
  fills, the losing leg must be `CANCELLED` with no stray resting stop after the flatten.

## Timeframe — 1 m (as requested)

All cases run on 1 m to keep the post-fill hold minimal (flatten ≈ 1 bar after the fill).
Note: prior runs found 1 m tight for the poll/fill/exit cadence — mitigated here by tiny
per-case scope, supervision, and the flat-account reset between cases. If a stop hasn't
filled within ~5 bars (quiet tape, price not breaking the prior high/low), cancel and
re-run rather than waiting indefinitely.

## Success criteria

- [ ] Each case's backtest gate: **zero** trades, clean exit.
- [ ] Each live case: correct order type sent → **one** fill (±1) → protective `SL@…` STOP
      armed → flatten to **0** → `SL@…` STOP `CANCELLED`; no re-entry, no errors, no orphans.
- [ ] g/h: losing OCA leg `CANCELLED`; c/f: single STOP-with-limit, not OCO.
- [ ] Account **flat** after every case.

## Abort / rollback

- **Primary:** close in the DNSE app (one tap).
- Kill the run, flatten manually, cancel any resting order, confirm flat before continuing.
- If a case leaves a position while unattended-for-a-moment: the mandatory venue-side
  stop-loss (armed on fill) is the automatic backstop, 1 m limits the window, and the DNSE
  app is the manual backstop.

## Out of scope (later tiers)

Multi-bar holds; take-profit / partial exits; a full OCO **exit** bracket on a live fill;
long→short reversal on a live position.
```
