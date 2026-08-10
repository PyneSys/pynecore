# DNSE Broker Plugin — Order-Semantics Test Plan (v2)

> **v2 corrects a wrong premise in v1.** v1 claimed "the plugin sends EVERY
> order as a marketable LO at the ±7% band edge." That is false — verified in
> `broker.py:429` + `execute_entry/exit/close`. The truth changes which tests
> are meaningful, so read "What the plugin actually does" before anything else.

## What the plugin actually does (CONFIRMED in code)

Every order is an `LO` (there are no other order types wired). The **price** is:

```python
# broker.py:429
"price": round(float(price) if price else self._marketable_price(side), 1)
```

where `price` comes from the intent, and `_marketable_price` (the ±7% band
edge) is the fallback used **only when no price is given** (market/close):

| Pine call | Price sent to DNSE | Result |
|---|---|---|
| `entry(limit=below-mkt)` | the limit price | **rests correctly** ✓ |
| `exit(limit=tp above-mkt)` | the tp price | **rests correctly** ✓ |
| `entry(stop=above-mkt)` | the stop price | **marketable → fills NOW, trigger ignored** ✗ |
| `exit(stop=sl below-mkt)` | the sl price | **marketable → fills NOW, trigger ignored** ✗ |
| `entry()` market / `close()` | band edge | fills now (correct) |
| `exit(trail_*)` | — | `OrderSkippedByPlugin` (unimplemented) |

**The bug is entirely about STOPS.** A stop trigger fed into a plain limit is
marketable *by construction* — a buy-stop sits above market, so a buy `LO` at
that price fills immediately. Proven live: buy-stop at 1925.3, market ~1921,
filled @ 1921.7. **Limits and take-profits already rest correctly** — the
protective (stop-loss) side is the failure, which is the worst place for it.

Two more confirmed facts that shape the tests:
- **REST polling, no intrabar data.** `watch_ohlcv` polls `/price/ohlc` (closed
  bars only); `watch_orders` polls `/accounts/{acct}/orders` every ~2 s. The
  backtest oracle (`SimPosition`) does an intrabar open→high→low→close walk, so
  a wick-triggered fill can differ between backtest and live purely from data
  granularity, and a live stop fills at *market after the closed bar*, not at
  the level.
- **Exits are deferred until the entry fills.** The engine holds an
  `exit(...)` until the referenced entry's fill is observed (`sync_engine.py`
  `_deferred_exits`; `tp_sl_bracket=SOFTWARE`). So entry and stop are NEVER on
  DNSE together: entry placed → fill seen on the next ~2 s poll → THEN exit
  placed. **The position is naked for up to the poll interval.** This — not
  intrabar ordering — is the real protection risk (test W1 below).

## Method (corrected)

1. Trigger placed FAR from market so *stop* tests diverge from correct behaviour.
   (Limit tests rest either way — see D0/D2, now regression tests.)
2. Backtest first as a reference. **A backtest-vs-live divergence is a
   *candidate* bug — not proof.** Before blaming the plugin, rule out
   (a) intrabar-vs-closed-bar granularity and (b) fill-at-market-vs-fill-at-level.
3. Then live, session open, one at a time.
4. **Safety: ANY stop test can fill immediately (the bug) and then the next-bar
   `strategy.cancel(...)` is a no-op against a filled order — leaving an open
   position the script never closes.** Applies to D1, D3, D4 (and D0/D2 only if
   the bug were real for limits, which it isn't). Always flatten + cancel_all
   after every test and confirm 0 positions / 0 live orders.

## Ground rules (learned the hard way)

- `strategy.cancel(id)` takes **no `comment`** — passing one crashes. Every
  cancel below is bare.
- VN30F1M tick = 0.1 pt; `profit=200` (ticks) = 20 pts.
- `margin_long/short=18.48` (%) → ~35.6M margin/contract; `initial_capital`
  500M is generous **headroom**, not a hard floor. D6's 2 contracts (~71M) fit
  easily. (Real placement floor is DNSE's account-side check — unverified.)
- Orders only place 09:00–11:30 / 13:00–14:45 ICT; cancel works after hours.

---

## PROBE 0 — does DNSE accept a RESTING stop? (do FIRST; gates D1/D3/D5 and D4's sl)

**This is the single highest-value action.** It decides whether the stop bug is
fixable (wire a real stop order) or a hard limitation (stops can't rest here).
Hardened script lives at `plugins/dnse/testing/probe_conditional_order.py`
(this doc shows intent, not the full source):

- Tries the **full matrix** category × orderType × trigger-field — do NOT pair
  each type with one guessed field; the real combo is unknown.
- Trigger set 3% AWAY from market so a genuine stop RESTS (fill 0). A plain
  `LO`/`price` at that level is NOT a stop test and is excluded from the
  "has stops?" decision.
- `try/finally` that **cancels every placed order AND flattens any position** —
  a fill on a marketable variant must not leave naked exposure.
- **Decision:** some variant rests (status New/Pending, fill 0) → DNSE has real
  stops; wire that orderType/field into `_place`. Every variant rejected OR
  fills immediately → treat as **UNKNOWN/none**, not proof of absence, and
  escalate to DNSE docs/support. Field names (`stopPrice`/`triggerPrice`/`STO`)
  are UNVERIFIED guesses — the probe's job is to find the real ones.

---

## Suite D — full .pine (transpile with pine2pyne, backtest, then live)

Shared header (only the title differs):

```pine
//@version=6
strategy("<title>", overlay=true, pyramiding=0, initial_capital=500000000,
     default_qty_type=strategy.fixed, default_qty_value=1,
     margin_long=18.48, margin_short=18.48,
     calc_on_every_tick=false, process_orders_on_close=false)
```

### D0 — Limit rest + cancel  ·  REGRESSION (should PASS today)

Not a bug demo — the plugin rests a non-marketable limit correctly. This
verifies that + the cancel path.

```pine
//@version=6
strategy("D0 Limit Rest + Cancel", overlay=true, pyramiding=0, initial_capital=500000000,
     default_qty_type=strategy.fixed, default_qty_value=1,
     margin_long=18.48, margin_short=18.48,
     calc_on_every_tick=false, process_orders_on_close=false)

// Buy limit 5% BELOW market -> non-marketable -> rests (both backtest AND the
// live plugin, since a limit is sent at its own price). Cancel next bar.
var int phase = 0
limitPx = close * 0.95

if phase == 0
    strategy.entry("L", strategy.long, qty=1, limit=limitPx,
                   comment="LIMIT@" + str.tostring(limitPx, format.mintick))
    phase := 1
else if phase == 1
    strategy.cancel("L")
    phase := 2

plot(strategy.position_size, "pos", display=display.data_window)
```
- **Expected today (backtest AND live):** 0 fills; order ends `Canceled`.
- **This would FAIL only if:** the plugin market-fills a resting limit (it does
  not) — i.e. a genuine regression.

### D1 — Stop entry far above  ·  BUG DEMO  ·  gated on Probe 0

```pine
//@version=6
strategy("D1 Stop Entry Far", overlay=true, pyramiding=0, initial_capital=500000000,
     default_qty_type=strategy.fixed, default_qty_value=1,
     margin_long=18.48, margin_short=18.48,
     calc_on_every_tick=false, process_orders_on_close=false)

// Buy STOP 3% ABOVE market. Correct: rests until a 3% rise (won't happen) -> 0.
// Bug: sent as LO at the stop price, which is ABOVE market -> marketable ->
// fills immediately at ~market on the placing bar.
var int phase = 0
stopPx = close * 1.03

if phase == 0
    strategy.entry("L", strategy.long, qty=1, stop=stopPx,
                   comment="STOP@" + str.tostring(stopPx, format.mintick))
    phase := 1
else if phase == 1
    strategy.cancel("L")
    phase := 2

plot(strategy.position_size, "pos", display=display.data_window)
```
- **Oracle (backtest):** 0 trades.
- **Bug signature (CONFIRMED today):** immediate long fill on the placing bar at
  ~market, well below the 1.03 trigger. Leaves an open long → cancel is a no-op
  → **flatten after.**

### D2 — Short limit far above  ·  REGRESSION (should PASS today)

```pine
//@version=6
strategy("D2 Short Limit Rest", overlay=true, pyramiding=0, initial_capital=500000000,
     default_qty_type=strategy.fixed, default_qty_value=1,
     margin_long=18.48, margin_short=18.48,
     calc_on_every_tick=false, process_orders_on_close=false)

// Sell limit 5% ABOVE market -> non-marketable -> rests. (Regression, not a bug
// demo: a sell limit is sent at its own price and rests correctly.)
var int phase = 0
limitPx = close * 1.05

if phase == 0
    strategy.entry("S", strategy.short, qty=1, limit=limitPx,
                   comment="SLIMIT@" + str.tostring(limitPx, format.mintick))
    phase := 1
else if phase == 1
    strategy.cancel("S")
    phase := 2

plot(strategy.position_size, "pos", display=display.data_window)
```
- **Expected today (backtest AND live):** 0 fills; order `Canceled`.

### D3 — Short stop far below  ·  BUG DEMO  ·  gated on Probe 0

```pine
//@version=6
strategy("D3 Short Stop Far", overlay=true, pyramiding=0, initial_capital=500000000,
     default_qty_type=strategy.fixed, default_qty_value=1,
     margin_long=18.48, margin_short=18.48,
     calc_on_every_tick=false, process_orders_on_close=false)

// Sell STOP 3% BELOW market. Correct: rests until a 3% drop -> 0. Bug: LO at
// the stop price is BELOW market -> marketable -> immediate short fill.
var int phase = 0
stopPx = close * 0.97

if phase == 0
    strategy.entry("S", strategy.short, qty=1, stop=stopPx,
                   comment="SSTOP@" + str.tostring(stopPx, format.mintick))
    phase := 1
else if phase == 1
    strategy.cancel("S")
    phase := 2

plot(strategy.position_size, "pos", display=display.data_window)
```
- **Oracle:** 0 trades. **Bug signature (CONFIRMED):** immediate short fill;
  leaves an open short → **flatten after.**

### D4 — Market entry + FAR bracket  ·  BUG DEMO (sl leg)

```pine
//@version=6
strategy("D4 Far Bracket", overlay=true, pyramiding=0, initial_capital=500000000,
     default_qty_type=strategy.fixed, default_qty_value=1,
     margin_long=18.48, margin_short=18.48,
     calc_on_every_tick=false, process_orders_on_close=false)

// Enter at market (fills), then a bracket 20 pts each way (profit/loss in
// TICKS; 200 ticks = 20 pts). Neither level is near market. Correct: position
// HELD across bars, sl rests 20 pts below. Bug: the sl leg (LO below market) is
// marketable -> fills immediately -> position closed on the bar after entry.
var bool entered = false

if not entered
    strategy.entry("L", strategy.long, qty=1, comment="MKT")
    entered := true

if strategy.position_size > 0
    strategy.exit("X", from_entry="L", profit=200, loss=200,
                  comment_profit="TP", comment_loss="SL")

plot(strategy.position_size, "pos", display=display.data_window)
```
- **Oracle:** 1 entry; position stays open (no 20-pt move). Flatten after.
- **Bug signature (CONFIRMED):** the sl fills on the bar after entry with no
  20-pt move (it was sent as a marketable LO below market). NOTE: distinguish
  this from W1 — here the exit *is* placed but fills wrong; W1 measures the gap
  *before* it is placed.

### D5 — Trailing stop  ·  DOCUMENTS UNSUPPORTED  ·  gated on Probe 0

```pine
//@version=6
strategy("D5 Trailing Stop", overlay=true, pyramiding=0, initial_capital=500000000,
     default_qty_type=strategy.fixed, default_qty_value=1,
     margin_long=18.48, margin_short=18.48,
     calc_on_every_tick=false, process_orders_on_close=false)

// Whole-row trailing exit -> reaches execute_exit -> tp/sl None -> the plugin
// raises OrderSkippedByPlugin (trailing not implemented). Confirms a genuine
// unprotected-position gap. (Verified path: not partial, so not engine-emulated.)
var bool entered = false

if not entered
    strategy.entry("L", strategy.long, qty=1)
    entered := true

if strategy.position_size > 0
    strategy.exit("X", from_entry="L", trail_points=100, trail_offset=50)

plot(strategy.position_size, "pos", display=display.data_window)
```
- **Oracle:** trailing fills on a pullback. **Bug signature (CONFIRMED):** live
  raises `OrderSkippedByPlugin` — the position has NO protective order. Flatten
  after (the entry is open and unprotected).

### D6 — Partial exit (qty_percent)  ·  PATH COVERAGE (engine-emulated, NOT the venue path)

```pine
//@version=6
strategy("D6 Partial Exit", overlay=true, pyramiding=0, initial_capital=500000000,
     default_qty_type=strategy.fixed, default_qty_value=2,
     margin_long=18.48, margin_short=18.48,
     calc_on_every_tick=false, process_orders_on_close=false)

// qty_percent=50 on 2 contracts -> is_partial -> routed through the SOFTWARE
// partial-bracket engine (a price-WATCH), NOT execute_exit. The stop is engine-
// side, close-bar granularity live, and never reaches DNSE as an order -> this
// does NOT test the marketable-LO path. Oracle below is DIRECTION-DEPENDENT.
var bool entered = false

if not entered
    strategy.entry("L", strategy.long, qty=2, comment="MKT2")
    entered := true

if strategy.position_size == 2
    strategy.exit("X1", from_entry="L", qty_percent=50, stop=low[1],
                  comment_loss="HALF")
else if strategy.position_size == 1
    strategy.close("L", comment="REST")

plot(strategy.position_size, "pos", display=display.data_window)
```
- **Oracle (only if price falls to `low[1]`):** 2 → 1 (partial) → 0. If price
  ticks up, the partial never triggers and size stays 2 — the `==1` branch never
  runs. Make the move deterministic or accept the direction dependency.
- **Tests:** qty_percent → 1-contract handling on a netted venue (engine path).

### D7 — order() + pyramiding stacking  ·  PATH COVERAGE

```pine
//@version=6
strategy("D7 order() Stacking", overlay=true, pyramiding=3, initial_capital=500000000,
     default_qty_type=strategy.fixed, default_qty_value=1,
     margin_long=18.48, margin_short=18.48,
     calc_on_every_tick=false, process_orders_on_close=false)

// order() is NOT pyramiding-capped. Two long order() -> net 2, then close_all().
var int phase = 0

if phase == 0
    strategy.order("O1", strategy.long, qty=1, comment="ORD1")
    phase := 1
else if phase == 1 and strategy.position_size >= 1
    strategy.order("O2", strategy.long, qty=1, comment="ORD2")
    phase := 2
else if phase == 2 and strategy.position_size >= 2
    strategy.close_all(comment="FLAT")
    phase := 3

plot(strategy.position_size, "pos", display=display.data_window)
```
- **Oracle:** 0 → 1 → 2 → 0. **Bug signature:** second order capped/rejected, or
  close_all leaves a residual.

### D8 — Protection close_all() fires  ·  PATH COVERAGE

```pine
//@version=6
strategy("D8 Protection Fires", overlay=true, pyramiding=0, initial_capital=500000000,
     default_qty_type=strategy.fixed, default_qty_value=1,
     margin_long=18.48, margin_short=18.48,
     calc_on_every_tick=false, process_orders_on_close=false)

// Enter market, then a deliberately tight -0.05% last-protection close_all so it
// fires on almost any adverse tick — exercising the close_all() path that never
// triggered in t1-t6. (Live fills land at ~market, so avg_price ~= market.)
var bool entered = false
PROTECT_PCT = -0.05

if not entered
    strategy.entry("L", strategy.long, qty=1, comment="MKT")
    entered := true

openPnlPct = 0.0
if strategy.position_size > 0
    openPnlPct := (close - strategy.position_avg_price) / strategy.position_avg_price * 100

if strategy.position_size != 0 and openPnlPct < PROTECT_PCT
    strategy.close_all(comment="PROTECT")

plot(strategy.position_size, "pos", display=display.data_window)
plot(openPnlPct, "pnl%", display=display.data_window)
```
- **Oracle:** enters, then close_all fires on the first bar with open P&L < -0.05%.
- **Bug signature:** protection never fires despite an adverse move.

---

## W1 — THE UNPROTECTED-WINDOW MEASUREMENT (the real risk; not a .pine)

D0–D8 read *final state* and cannot see the multi-second window where a filled
position has no stop on the venue (entry fill learned only on the ~2 s poll,
exit dispatched only after). Measure it directly, live:

```
1. Run a stop-loss strategy live (e.g. market entry + exit(stop=low[1])).
2. From an INDEPENDENT reader, poll /accounts/{acct}/positions and
   /accounts/{acct}/orders every ~0.5 s.
3. Record t_fill  = wall-clock when the entry position first appears.
   Record t_stop  = wall-clock when a protective (opposite-side) order first
                    appears on the book.
4. Report the gap (t_stop - t_fill). Expect it >= the ~2 s poll interval, plus a
   sync cycle. During that gap the position is provably unprotected.
```
- **This is the headline risk** and the only test that surfaces it. It needs no
  adverse move to prove the exposure exists; an engineered gap-down inside the
  window would demonstrate a real loss, but the timing measurement alone is the
  finding.

---

## Run procedure (per D-test)

```bash
cd /home/mike/workspace/github/pynecore
S=d1_stop_entry_far   # etc.
PYTHONPATH=/path/to/pine2pyne .venv/bin/python -m pine2pyne \
  plugins/dnse/testing/strategies/$S.pine -o /tmp/$S.py
cp /tmp/$S.py plugins/dnse/testing/strategies/$S.py workdir/scripts/
.venv/bin/pyne run $S "dnse:VN30F1M@1" -f 40            # BACKTEST oracle
.venv/bin/pyne run $S "dnse_broker:VN30F1M@1" -f -50 --live --broker --shutdown-timeout 5
# cleanup: flatten + cancel_all; confirm 0 positions / 0 live orders
```

## Corrected pass/fail matrix

| Test | Oracle | Current-plugin expectation | Kind |
|---|---|---|---|
| D0 | 0 fills, Canceled | **0 fills (PASSES today)** | regression (limit rests) |
| D1 | 0 fills (stop rests) | **fills immediately (BUG)** | stop-entry bug demo |
| D2 | 0 fills, Canceled | **0 fills (PASSES today)** | regression (limit rests) |
| D3 | 0 fills | **fills immediately (BUG)** | short-stop bug demo |
| D4 | holds; no exit | **sl fills instantly (BUG)** | bracket-sl bug demo |
| D5 | trailing fills | **OrderSkippedByPlugin** | unimplemented gap |
| D6 | 2→1→0 (if price drops) | engine-emulated watch | path coverage |
| D7 | 0→1→2→0 | order()/pyramiding | path coverage |
| D8 | close_all fires | protection path | path coverage |
| **W1** | n/a | **gap ≥ ~2 s, position unprotected** | **the real risk** |

**Headline:** only **D1, D3, D4** demonstrate the marketable-LO stop bug, and
**D5** an unimplemented gap. D0/D2 are regressions expected to pass. **W1 is the
test that matters** — it measures the unprotected window D0–D8 cannot see. None
of this proves protection works; the fix depends on Probe 0's answer about
resting stops. Run Probe 0 first.
