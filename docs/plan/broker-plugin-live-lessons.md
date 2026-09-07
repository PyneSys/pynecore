# Broker-plugin live-market lessons (DNSE + Binance, joint findings)

Fork-only reference — not upstream content, do not merge into
`docs/development/broker-plugin-authoring.md` (upstream-authored). Written
2026-09-07 from a cross-session audit between the DNSE and Binance broker
plugin work, each side verifying the other's claims against actual code
rather than relaying them as fact. Update this file as new venues are added
or new incidents land; it is meant to outlive any single plugin's own
testing README.

## The core finding: one venue property predicts your failure mode

**Whether a venue's "list open orders" endpoint retains or drops terminal
rows determines both your external-cancel detection design AND your
dominant failure mode.** Same core machinery on both sides — `DisappearanceTracker`,
`quarantine_sink`, `on_unexpected_cancel` — opposite consequences depending
on this one measurable fact:

| | Retains terminal rows (DNSE's day-book) | Drops terminal rows immediately (Binance `openOrders`) |
|---|---|---|
| **What an external cancel looks like** | A status **transition** on a row that's still listed | A **disappearance** — no retained row to diff a status off |
| **The load-bearing signal** | The transition detector in `watch_orders`; `DisappearanceTracker` is at best a narrow residue backstop | Disappearance itself — a genuine `DisappearanceTracker` (or equivalent attribution) is load-bearing, not optional |
| **Dominant risk if built wrong** | **Mass FALSE quarantine** — a read failure returning `[]` instead of None-per-book reads as "every row vanished" = book-wide external cancel, and the default `on_unexpected_cancel=stop` halts trading while real exposure may still be open | **No detection at all** — every terminal transition looks identical, so an operator's manual cancel produces zero attribution, zero halt, zero alert |
| **Where it's guarded / not yet guarded** | Guarded: the `rows \| None` read contract (dnse-broker-v2 #54) | Not yet guarded: [rubycell/pynecore#79](https://github.com/rubycell/pynecore/issues/79) — filed 2026-09-07, no fix landed yet |

**Practical consequence for the next venue plugin:** measure this fact
before designing the cancel-attribution path, don't assume either shape.
Binance's is the industry-standard shape (exchange REST APIs almost
universally drop terminal orders from the "open orders" list); DNSE's
day-book retention is the less common case and is easy to over-generalize
from if DNSE was your first venue.

## Second venue property: do exit orders execute standalone?

Joint finding, 2026-09-07 (DNSE Live-L3-F05 → Binance staged probe B3). Same
shape as the retention axis above: one measurable venue property decides a
design branch, and each branch has its own failure mode if you pick wrong.

**The question: does a protective exit execute on its own, or is it attached
to a position and rejected without one?**

| | Standalone exits (DNSE conditional book, Binance spot) | Attach-semantics exits (position-attached SL/TP) |
|---|---|---|
| **What an exit is** | An independent working order on the book. DNSE: a conditional-book order. Binance spot: a plain SELL resting against account base inventory — spot has no position rows at all | A protective level attached to a position row; the venue rejects it when no position exists |
| **Correct engine behavior** | **Withhold** the exit until the parent entry fills (`exit_orders_execute_standalone = True` arms the engine's #82b guard, which also clamps exit qty to the live position) | **Pre-dispatch** at placement — this is the deliberate contract, and the partial-fill bracket-amend flow depends on it |
| **Failure mode if you pick wrong** | **Naked position from a pre-fill protection.** The protection executes with nothing behind it: DNSE F5 opened a naked SHORT while flat; Binance would SELL foreign/other-strategy base inventory it does not own | **Misclassified bracket rejects.** Without pre-dispatch the engine's whole bracket-reject recovery family (`BracketAttachAfterFillRejectedError` and the defensive-close path) has nothing to recover from and the attach contract breaks — this is why the #82b guard is capability-scoped, not universal (a universal version broke 20 upstream contract tests) |
| **Measured evidence** | DNSE F5 2026-09-07 (naked short, real money, ladder stopped for safety); Binance probe B3 red 2026-08-17 (exit 3828722 resting while flat) → green 2026-09-07 (0 SELL orders) | The 20 upstream attach-contract tests that the universal guard broke |

**Known cost of the withhold, verified in code 2026-09-07:** the engine's
`sync()` is called **once per bar** from the script runner (keyed to
`lib.last_bar_time`); there is no fill-triggered re-sync. So a withheld
protection is dispatched at the *next bar close* after the entry fills —
an unprotected window of **up to one full bar**, which scales with the
strategy's timeframe (≤60 s at 1m, but ≤15 min at 15m). Mild on unleveraged
spot; material for tight stops or leveraged venues. Know it exists before
choosing a timeframe for a stop-dependent strategy.

## Verified findings, by topic

### `execute_cancel_all` — settled, dead path on both sides

`strategy.cancel_all()` never calls `BrokerPlugin.execute_cancel_all` —
grepped the whole engine, zero call sites. `BrokerPosition._cancel_all_orders()`
just clears the local order dicts; the next `OrderSyncEngine.sync()` diffs
against `_active_intents` and dispatches ordinary per-order `execute_cancel`
calls. `execute_cancel_all` is a **plugin-initiated side door** only — for a
plugin's own code to call a venue's real bulk-cancel endpoint and arm
`enqueue_native_cancel_all_expected` first so the resulting `CANCELLED`
pushes don't trip quarantine. It matters only for a future *external* caller
(a kill-switch API, a `pyne runs` CLI) invoking it directly for speed.
Independently confirmed by both DNSE (dnse-broker-v2 #77, rejected as a
candidate fix for exactly this reason) and Binance (this audit) on the same
day. Binance implements it anyway (real bulk-cancel endpoint); DNSE doesn't;
neither matters for Pine-script correctness today.

### The stale-journal-extras hazard — scoped, not a core risk today

DNSE's #77 fix (`_scan_live_entry_anchors_for_restart`, now in
`src/pynecore/core/broker/sync_engine.py` — core, shared by every plugin)
had a second-layer bug: a deterministic client-order-id means the
underlying `orders` row can be *reused* across runs, and a prior run's
`terminal_status` extras key survived the reopen, causing the restart scan
to wrongly filter out a freshly-resting live order.

**Scoping, confirmed by code read:** `terminal_status` / `last_raw_status` /
`last_fill_venue_id` are keys DNSE's own `journal_wiring.py` invented (a
custom `_merged_extras` helper) — not a core convention. Grepped: zero hits
outside `plugins/dnse/`. `BinanceBroker` never calls `upsert_order`/touches
`.extras` at all. **The hazard is real only for a plugin that (a) invents
its own `extras` keys on (b) a deterministic-coid row that gets reused.**
It is not a risk for Binance today, and would only become a core concern
if pynecore ever grows a generic plugin-facing journal layer that all
plugins share — worth remembering if that's ever built.

### Read-failure contract — both sides correct, verify this on every new venue

The rule (dnse-broker-v2 #54, independently satisfied by Binance): a failed
order-book read must **never** collapse to an empty list — that's what turns
into the mass-false-quarantine risk in the table above. DNSE reports
`rows | None` explicitly. Binance's `get_open_orders` raises (via `_venue()`'s
exception mapping) rather than ever catching-and-returning `[]`; the
`BrokerPlugin` base class's central safety net parks a raised
`ExchangeConnectionError`/retryable transient for retry-next-cycle instead
of reading it as "the book is empty." Both are valid implementations of the
same contract. **Test this explicitly for every new venue plugin** — it's
the kind of bug that only shows up during a real outage, never in a happy-path
test.

### rubycell/pynecore#79 — Binance's external-cancel attribution gap

Filed 2026-09-07. `BinanceBroker` detects terminal transitions by polling
each tracked id in `_live_ids` via `fetch_order`, and reports every
transition as a plain `cancelled`/`rejected` event — with no distinction
between a bot-initiated cancel and an external one. `quarantine_sink` is
wired but only reaches the spot-inventory-conflict path, never the
order-watch path. `UnexpectedCancelError` is never raised. Net:
`on_unexpected_cancel` is completely inert for Binance regardless of
configured policy.

**Suggested fix shape**, refined jointly: track a set of coids currently
mid-`execute_cancel`/`_cancel_one` (bot-initiated); when `_poll_once`
observes a terminal transition on a tracked id NOT in that set, route it
through `self.on_unexpected_cancel` instead of a plain event.
**Caveat (DNSE, from their own `enqueue_native_cancel_all_expected`
experience): the expected-cancel set must be bounded/TTL'd, not an
unbounded accumulator** — cancel/replace churn (e.g. the amend path, which
Binance implements as cancel+replace) will otherwise produce a slow memory
leak of stale expected-cancel entries that never get consumed because the
matching event already arrived via a different path.

### Atomic amend vs. cancel+replace — no universal winner, document the tradeoff per venue

DNSE implements an atomic amend for `modify_entry`/`modify_exit`; Binance
deliberately uses the `BrokerPlugin` base class's cancel+replace default
(`amend_order = SOFTWARE`). Measured tradeoffs on both sides:

- DNSE's atomic amend **misbehaves on conditional/stop orders** — a
  conditional amend returns HTTP 500 (dnse-broker-v2 #18); the engine parks
  and the order stays cancellable, but the **venue silently keeps the old
  level** until the parked state resolves. This is exactly the order type
  you'd most want to move atomically.
- Binance's cancel+replace has a **measured exposure gap**: dnse-broker-v2's
  T15 probe clocked cancel-ACK at 35ms and the replacement accepted at
  +99ms — roughly a 100ms window with no protective order resting on the
  venue at all. If a protective stop ever goes through cancel+replace live,
  that gap is real naked exposure, however brief.

Neither approach is free. State which one a given venue plugin uses and why
in its own testing docs (already done: `plugins/binance/testing/README.md`
documents the SOFTWARE choice as measured, not assumed).

### Restart / persistence discipline — shared, no code divergence

- **Same-label relaunch is mandatory.** Orphan adoption in
  `src/pynecore/core/broker/storage.py` keys on `run_id`, which is derived
  from (among other things) `--run-label`. A relaunch under a *different*
  label after a crash changes `run_id` and orphan adoption never fires —
  the prior run's live orders become invisible strands. This bit DNSE live
  (dnse-broker-v2 #60) when a documented crash-relaunch recipe used a new
  label; it applies identically to Binance or any future plugin.
- **The 5-minute stale-run window is a real relaunch blocker**, not just a
  theoretical guard. `open_run()` raises `"Active run_id already exists"` if
  you relaunch under the same `run_id` before the prior instance's
  heartbeat ages out (~5 min after a SIGKILL; immediate after a clean
  SIGTERM teardown). Fast crash-relaunch under the same label within that
  window will be refused — wait it out, or confirm teardown actually ran.
- **Under SOFTWARE idempotency (no venue-echoed client order id), the
  plugin's own journal is the *only* restart bridge.** DNSE's original T16
  bug (#36) was simply that nothing was journaled, so restart had nothing to
  adopt from. Under NATIVE idempotency (Binance), the venue's echoed
  `clientOrderId` already carries this information and no custom journal is
  needed for restart adoption — but ANY plugin state that isn't
  persist-first-before-the-POST (partial fills, bracket construction state)
  is equally vulnerable to a crash-in-the-window loss on either venue type.

## Method note: re-read the other venue's evidence before writing a new probe

When a new bug class lands on one venue, **re-read the other venue's existing
evidence for the same shape before writing any new probe — the measurement is
often already on disk, unrecognised.**

Both sides have now hit this. The #82 naked-exit precondition was sitting in
Binance's own 2026-08-17 B3 evidence (`dispatched EXIT id='X3' … ->
['3828722']`, an exit resting at the venue while its entry was unfilled) for
three weeks; it read as a neutral "measured: the pre-fill exit reaches the
venue" until DNSE's F5 gave the shape a name and a consequence. DNSE reports
the same pattern in their #51 timeline reconstruction. The cost of the
re-read is minutes; the cost of missing it is a live incident on the second
venue.

## Open items

- **rubycell/pynecore#79** — Binance external-cancel attribution fix, not
  yet designed in detail beyond the shape above. Needs a fake-venue test
  that cancels a tracked order behind the plugin's back and asserts the
  configured `on_unexpected_cancel` policy actually fires.
- DNSE is mid-way through a live T16 restart re-grade as of this writing;
  this doc's restart/persistence section may need a follow-up pass if that
  changes anything.
- If a third venue plugin (bybit/capitalcom/ctrader, or a new one) is ever
  built against this fork, measure its open-orders terminal-row retention
  policy FIRST, before writing its disappearance/quarantine wiring — that's
  the one fact that decides the whole design, per the table above.
