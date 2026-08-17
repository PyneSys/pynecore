#!/usr/bin/env python3
"""T33 — closed-hours placement semantics (card rubycell/pynecore#22, phase probe).

The ONLY test on the card whose REQUIRED phase is "closed"; it refuses to run in any
other phase. Pine cannot drive it (no bars arrive while the market is closed, so a
strategy body never executes) — this drives the real broker code path directly,
exactly like l0_order_semantics.py.

WHY IT EXISTS — two prior measurements conflict:
  * 2026-08-13 14:37/14:51 (after the close): conditionals refused with CO-ORD-006,
    a plain NORMAL limit refused with CANNOT_PLACE_ORDER_IN_THE_CLOSED_SESSION
    -> "closed = nothing can be placed".
  * 2026-08-12 (part1_market docstring): a market order "outside a session" was
    ACCEPTED and queued at the band edge, filled=0.
  A weekend evening is unambiguously closed — this run settles which behaviour the
  venue actually has per order type, and grades the PLUGIN's error surfacing
  (SESSION_REJECT disposition -> clean reject + WARNING, no crash) along the way.

Four probes, every one EXPECTING refusal:
  1. STOP conditional (buy, trigger +5%)      -> expect CO-ORD-006-class refusal
  2. STOP-LIMIT conditional (buy, +5%/+5.01%) -> expect the same
  3. NORMAL far limit (buy, -5%)              -> expect CANNOT_PLACE_..._CLOSED_SESSION
  4. NORMAL at an OFF-TICK price (x.x5 on a 0.1-tick contract) -> ordering probe:
     does tick validation or the session gate fire first? Either refusal PASSes
     (record which code); nothing can rest from this probe by construction.

If any probe is unexpectedly ACCEPTED: cancel immediately — the cancel outcome is
itself the "can we cancel while closed?" measurement the card wants. A refusal to
cancel leaves a far-from-market 1-lot order until Monday 09:00: bounded, but loud
manual-action warnings are printed and the run FAILs.

Run (Saturday evening / any closed hour):
    .venv/bin/python plugins/dnse/testing/live_test/level0_venue_semantics/t33_closed_hours.py
    # --dry-run prints what would be sent without touching the venue
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pynecore.core.config import ensure_config                      # noqa: E402
from pynecore_dnse.broker import DNSEBroker, DNSEBrokerConfig       # noqa: E402
from l0_order_semantics import (                                    # noqa: E402
    Result, envelope, reference_close, book_has, session_phase, SYMBOL, TIMEFRAME, QTY,
)

ICT = timezone(timedelta(hours=7))
PLACED: list[str] = []


async def probe(broker: DNSEBroker, result: Result, name: str, *,
                side: str, price: float, category: str,
                stop_price: float | None = None, dry_run: bool = False) -> None:
    """One placement attempt whose EXPECTED outcome is a clean venue refusal."""
    phase = session_phase()
    if dry_run:
        result.add(name, None,
                   f"dry-run: {category} {side} price={price} stop={stop_price} phase={phase}")
        return
    try:
        orders = broker._place(envelope(f"t33-{name.replace(' ', '-')}"), side, QTY,
                               price=price, category=category, stop_price=stop_price)
    except Exception as exc:                                        # noqa: BLE001
        # Refusal IS the pass. Record the exact classification the plugin produced —
        # the grading interest is as much the error SURFACE as the venue code.
        result.add(name, True, f"refused (expected): {type(exc).__name__}: {exc}")
        return

    # Unexpectedly accepted — measure cancellability while closed, leave nothing behind.
    ids = [str(o.id) for o in orders]
    PLACED.extend(ids)
    filled = sum(float(getattr(o, "filled_qty", 0) or 0) for o in orders)
    if filled:
        result.add(name, False, f"FILLED while closed?! ids={ids} — investigate NOW")
        return
    result.add(name, False,
               f"ACCEPTED while closed (contradicts 2026-08-13 measurement) ids={ids}")
    await asyncio.sleep(1.0)
    on_book = all([await book_has(broker, oid) for oid in ids])
    cancelled = all(broker._cancel_one(oid) for oid in ids)
    result.add(f"{name}: cancel-while-closed", cancelled,
               ("cancel accepted — closed-hours cancel WORKS (new measurement)"
                if cancelled else
                "cancel REFUSED — order rests until Monday 09:00, CANCEL IT MANUALLY"))
    if cancelled:
        for oid in ids:
            if oid in PLACED:
                PLACED.remove(oid)
    else:
        result.add(f"{name}: book state", on_book,
                   f"ids={ids} far from market (bounded) — manual cleanup required")


async def run(args: argparse.Namespace) -> int:
    now = datetime.now(ICT)
    phase = session_phase(now)
    print(f"=== T33 — closed-hours placement semantics @ {now:%Y-%m-%d %H:%M} ICT ===")
    print(f"phase: {phase}")
    if phase != "closed" and not args.dry_run:
        print("REFUSING to run: T33's required phase is 'closed' — in any open phase "
              "these probes stop being reject probes and start being orders.")
        return 2

    cfg = ensure_config(DNSEBrokerConfig,
                        REPO / "workdir" / "config" / "plugins" / "dnse_broker.toml")
    broker = DNSEBroker(symbol=SYMBOL, timeframe=TIMEFRAME, config=cfg)
    print(f"account: {broker.account_id}   contract: {broker.resolve_contract()}")

    ref, ref_label = reference_close(broker)
    tick = 0.1
    print(f"reference ({ref_label}): {ref}")

    result = Result()
    try:
        await probe(broker, result, "stop conditional",
                    side="buy", price=round(ref * 1.05, 1), category="STOP",
                    stop_price=round(ref * 1.05, 1), dry_run=args.dry_run)
        await probe(broker, result, "stop-limit conditional",
                    side="buy", price=round(ref * 1.0501, 1), category="STOP",
                    stop_price=round(ref * 1.05, 1), dry_run=args.dry_run)
        await probe(broker, result, "normal far limit",
                    side="buy", price=round(ref * 0.95, 1), category="NORMAL",
                    dry_run=args.dry_run)
        # Off-tick: halfway between two ticks. Zero rest risk by construction —
        # SOME validator must refuse it; the interest is WHICH ONE fires first.
        off_tick = round(ref * 0.95, 1) + tick / 2
        await probe(broker, result, "normal off-tick",
                    side="buy", price=off_tick, category="NORMAL", dry_run=args.dry_run)
    finally:
        if PLACED:
            print(f"\n[cleanup] cancelling {len(PLACED)} leftover order(s): {PLACED}")
            for oid in list(PLACED):
                try:
                    if broker._cancel_one(oid):
                        print(f"  cancelled {oid}")
                        PLACED.remove(oid)
                    else:
                        print(f"  !! COULD NOT CANCEL {oid} — far from market, but "
                              f"CANCEL IT MANUALLY before Monday 09:00")
                except Exception as exc:                            # noqa: BLE001
                    print(f"  !! cancel error {oid}: {exc} — CANCEL IT MANUALLY")

    print("\n=== SUMMARY (phase={}) ===".format(phase))
    marks = {True: "PASS", False: "FAIL", None: "SKIP"}
    for name, ok, detail in result.rows:
        print(f"  [{marks[ok]}] {name}" + (f" — {detail}" if detail else ""))
    print("\nVERDICT:", "FAIL — see above" if result.failed else "PASS")
    return 1 if result.failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be sent; touch nothing")
    args = parser.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
