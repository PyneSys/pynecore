# DNSE broker — safest live-test plan

> **HISTORICAL DOCUMENT.** The canonical test registry and the ONLY test names to use
> (`Live-L<level>-<case>`, e.g. `Live-L1-T11-OcaCancelMember`) live in
> [`live_test/README.md`](README.md). This file predates that naming and is kept for history.

Validate the **live order path** (place → rest → cancel) against the real DNSE account
with **zero fills and zero position risk**. Every order is placed far enough from market
that it cannot fill or trigger in a session, so nothing is ever held — the only thing
exercised is the write/cancel round-trip. Fill / position / exit logic is covered by the
**backtest oracle** (`t3`–`t6`, balanced entry/exit) and is deliberately NOT run live here.

## MANDATORY PRE-FLIGHT — Level 0 must pass first

**Every live test below is gated on Level 0.** It runs at any hour (no session, no
candles needed) and exercises the whole chain against the real venue in ~30 s:
config -> trading token -> contract resolution -> place -> read back -> cancel.

```bash
cd /home/mike/workspace/github/pynecore
.venv/bin/python plugins/dnse/testing/live_test/level0_venue_semantics/l0_order_semantics.py
# exit 0 = cleared to proceed;  non-zero = STOP, do not run any live test
```

It catches, before you risk anything, what the higher tiers would only hit mid-flight:
an expired/broken trading token (a stronger check than `token_status.py` — L0 actually
places an order), a **stale contract mapping** after the monthly `VN30F1M` roll, a
changed venue behaviour, and whether native STOP/STOP-LIMIT orders still rest and cancel.

Run it **the same day** as the live test (the token lasts ~8 h). During an open session
L0 skips its market-order part automatically — that part would fill — so an in-session
run still validates the STOP / STOP-LIMIT paths, which is what matters here.

## Why this is safe

- **No marketable / market orders** → no fills → **no position, no money at risk**. The
  only orders sent are resting entries priced where they cannot execute.
- Orders sit **±5 % from market**: inside the ±7 % daily band, so they **rest** (a closer
  band-edge price risks an `INVALID_PRICE` rejection; a price *outside* the band is
  rejected outright), yet far enough they won't fill/trigger in a normal session
  (VN30F1M moves ~1–2 %/day).
- **1 contract** VN30F1M, always (`default_qty_value=1`).
- A **−0.1 % `close_all` guard** on every strategy — a pure backstop. With no position it
  never fires, but it flattens instantly if anything ever *did* fill.
- **Single-trigger orders only** (limit-only or stop-only). This side-steps the open
  stop-limit-vs-OCO ambiguity flagged in the backlog (an entry with *both* `stop=` and
  `limit=`), so the live plan can't be tripped by it.
- **Backtest first**, then live — same transpiled `.py`, so what you backtest is byte-for-byte
  what you run live.
- Run **in the background** — a foreground run gets SIGTERM'd at the 120 s tool cap and
  looks like a failure when it isn't.
- **Clean up** after every test: the run's shutdown does not cancel resting orders, so
  cancel them yourself and confirm the account is flat.

## The strategies (safest → broadest surface)

| # | File | On green it places | Proves |
|---|------|--------------------|--------|
| 1 | `live_test/live1_limit_cancel.pine` | one buy-limit −5 % (NORMAL LO) | place → rest → cancel of a plain limit order |
| 2 | `live_test/live2_stop_cancel.pine`  | one buy-stop +5 % (native STOP) | STOP place/cancel — the cancel-by-own-book fix |
| 3 | `live_test/live3_oca_cancel.pine`   | buy-stop +5 % **+** sell-stop −5 %, one OCA group | grouped conditional place + cancel of both legs |

All three: place on a **green** candle, cancel on a **red** candle, hold no position ever.
(#1 mirrors `t7_long_limit_cancel`, #3 mirrors `t8_oca_breakout`; kept here as the plan's
self-contained set so this folder is the whole story.)

## Pre-flight (once, before any live run)

```bash
cd /home/mike/workspace/github/pynecore
.venv/bin/python plugins/dnse/tools/token_status.py     # must print GOOD
```

Then confirm the market is in a **continuous** session — **09:15–11:30** or
**13:00–14:30 ICT**. Avoid the auctions (09:00 ATO, 11:30 lunch, 14:30 ATC): placement is
session-gated and, in an auction, a cancel can be refused. The safest window to watch a
full place-then-cancel cycle is early afternoon (13:00+).

## Procedure — per strategy

**Timeframe: 1 m.** These hold no position, so 1 m is ideal — fast place/cancel cycles and
quick feedback. 3 m / 5 m work too if you want to watch each step longer.

Using `live1` as the example (repeat for `live2`, `live3`):

**1. Transpile** (local pine2pyne — NOT the cloud API):
```bash
cd /home/mike/workspace/github/pine2pyne
.venv/bin/python -m pine2pyne \
  /home/mike/workspace/github/pynecore/plugins/dnse/testing/live_test/live1_limit_cancel.pine \
  -o /home/mike/workspace/github/pynecore/plugins/dnse/testing/live_test/live1_limit_cancel.py
```

**2. Backtest first** (must run clean — no exceptions, orders place & cancel in the log):
```bash
cd /home/mike/workspace/github/pynecore
.venv/bin/pyne run plugins/dnse/testing/live_test/live1_limit_cancel.py dnse:VN30F1M@1
```

**3. Run live** (background; `--broker` places REAL orders on the real account):
```bash
.venv/bin/pyne run plugins/dnse/testing/live_test/live1_limit_cancel.py \
  dnse_broker:VN30F1M@1 --broker
```
Run this **in the background** and watch the log. On a green bar you should see the order
dispatched and a `CREATED` event; on the next red bar, a cancel and a `CANCELLED` event.

**4. Verify on the venue** — open the DNSE order book (or a read-only `get_orders` probe)
and confirm: order appears **New** after a green bar, becomes **Canceled** after a red bar.

**5. Clean up** — stop the run, then cancel anything still resting and confirm flat:
```bash
.venv/bin/python plugins/dnse/testing/probe_conditional_order.py --list      # inspect books
# cancel any leftover working order (STOP / NORMAL / OCO), then re-list to confirm none rest
```
Leave the pre-existing external position untouched — this plan only adds and removes its
own resting orders.

Run them in order **1 → 2 → 3**. For each, stopping as soon as you've observed one place
**and** one cancel is enough — you don't need many cycles.

## Success criteria

- [ ] `live1`: a NORMAL limit order goes **New → Canceled**, no fill, account stays flat.
- [ ] `live2`: a native STOP goes **New → Canceled** via `orderCategory=STOP` (the fix), no
      `404 RESOURCE_NOT_FOUND` short-circuit leaving it resting.
- [ ] `live3`: **both** OCA legs place, **both** cancel; nothing triggers.
- [ ] After cleanup, zero orders the plan created remain working; the external position is
      unchanged.

## Out of scope (deliberately)

- **Fills / positions / exits** — never taken live (no marketable orders). Covered by the
  backtest oracle (`t3`–`t6`).
- **One-cancels-other on a real fill** — covered by the backtest OCA demo, not live.
- Validating a real fill live needs a separate, **riskier** plan (a 1-lot marketable entry
  + immediate `close_all`, tightly guarded). That is intentionally NOT part of this
  *safest* plan; treat it as a follow-up only if the venue's fill accounting must be seen
  end-to-end on the live account.
```
