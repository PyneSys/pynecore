#!/usr/bin/env python3
"""LIVE TEST LEVEL 0 — venue order semantics. Runs ANY TIME, including after hours.

Answers the question the Pine-driven tests cannot answer outside a session: **does
DNSE actually accept and REST each order type we send, and can we cancel it again?**
No candles required, no strategy, no engine — it drives the real v2 broker code path
(``DNSEBroker._place`` / ``_cancel_one`` / ``get_open_orders``) directly.

Three parts, each self-verifying:

  1. MARKET long + short  -> expected to FAIL after hours (a market order cannot be
     placed outside a continuous session). A clean rejection IS the pass.
     SKIPPED during a continuous session, where it would FILL and open a real
     position — pass ``--allow-market-in-session`` to override (you almost never
     should).
  2. STOP long + short, triggers +/-5% from the reference close -> must REST on the
     conditional book. Read back, confirm present, cancel, confirm gone.
  3. STOP-LIMIT long + short (stop AND limit), same +/-5% -> must REST. Read back,
     confirm present, cancel, confirm gone.

Why +/-5%: inside the +/-7% daily band so the venue accepts it, far enough that it
cannot trigger. Every order is 1 contract.

SAFETY: a ``finally`` block cancels every order this script placed and reports any
position it sees, even on exception/Ctrl-C. Nothing is left resting.

Run:
    .venv/bin/python plugins/dnse/testing/live_test/level0_venue_semantics/l0_order_semantics.py
    # add --dry-run to print what WOULD be sent without touching the venue
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))

from pynecore.core.config import ensure_config                      # noqa: E402
from pynecore_dnse.broker import DNSEBroker, DNSEBrokerConfig       # noqa: E402

ICT = timezone(timedelta(hours=7))
SYMBOL = "VN30F1M"
TIMEFRAME = "1"
QTY = 1.0
AWAY = 0.05          # 5% from the reference close — rests, cannot trigger


# --------------------------------------------------------------------------- utils

class Result:
    """Collects per-step verdicts so the run ends with one readable table."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool | None, str]] = []

    def add(self, name: str, ok: bool | None, detail: str = "") -> None:
        self.rows.append((name, ok, detail))
        mark = {True: "PASS", False: "FAIL", None: "SKIP"}[ok]
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    @property
    def failed(self) -> bool:
        return any(ok is False for _, ok, _ in self.rows)


def in_continuous_session(now: datetime | None = None) -> bool:
    """VN30F1M continuous trading: Mon-Fri 09:15-11:30 and 13:00-14:30 ICT."""
    now = now or datetime.now(ICT)
    if now.weekday() > 4:
        return False
    minutes = now.hour * 60 + now.minute
    return (9 * 60 + 15) <= minutes < (11 * 60 + 30) or \
           (13 * 60) <= minutes < (14 * 60 + 30)


def envelope(tag: str) -> SimpleNamespace:
    """Minimal stand-in for the engine's order envelope — enough for ``_place``."""
    return SimpleNamespace(
        intent=SimpleNamespace(intent_key=f"l0-{tag}", pine_id=f"L0-{tag}",
                               from_entry=None, limit=None, stop=None))


def reference_close(broker: DNSEBroker) -> tuple[float, str]:
    """Last DAILY close — the anchor for the +/-5% trigger levels.

    ``/price/ohlc`` keys off the ALIAS (``VN30F1M``), not the resolved KRX contract —
    the provider's own ``download_ohlcv`` passes ``self.symbol``. Falls back to the
    last 1-minute close if the daily series comes back empty (e.g. before the first
    daily bar of a new contract).
    """
    to_ts = int(datetime.now(timezone.utc).timestamp())
    for resolution, span, label in (("1D", 30 * 24 * 3600, "daily close"),
                                    ("1", 3 * 24 * 3600, "last 1m close")):
        status, body = broker.client.get_ohlc(broker.market_type, {
            "symbol": broker.symbol, "resolution": resolution,
            "from": to_ts - span, "to": to_ts,
        })
        if status == 200 and isinstance(body, dict) and body.get("c"):
            return float(body["c"][-1]), label
    raise RuntimeError("cannot read a reference close from /price/ohlc "
                       f"(symbol={broker.symbol})")


async def book_has(broker: DNSEBroker, order_id: str) -> bool:
    """Is ``order_id`` present and non-terminal on either DNSE order book?"""
    for order in await broker.get_open_orders(SYMBOL):
        if str(order.id) == str(order_id):
            return True
    return False


# --------------------------------------------------------------------------- parts

async def part1_market(broker: DNSEBroker, result: Result, *,
                       allow_in_session: bool, dry_run: bool) -> None:
    """After hours a MARKET order must NOT FILL — and must be cancellable.

    Measured 2026-08-12: DNSE **accepts** a market order outside a session rather
    than refusing it. It is priced as a NORMAL LO at the band edge and simply
    queues (``filled=0``). So the invariant worth asserting is *no fill*, not
    *no acceptance* — and, critically, that we can cancel it again: a queued
    band-edge buy left in the book would fill at the next open, ~130 points
    through the market. Either outcome (refused, or accepted-and-cancelled)
    passes; a FILL or an uncancellable order fails.
    """
    print("\n[1] MARKET long + short — after hours: must NOT fill, must be cancellable")
    if in_continuous_session() and not allow_in_session:
        result.add("market long", None, "session is OPEN — would FILL; skipped")
        result.add("market short", None, "session is OPEN — would FILL; skipped")
        return

    for side in ("buy", "sell"):
        name = f"market {'long' if side == 'buy' else 'short'}"
        price = broker._marketable_price(side)
        if dry_run:
            result.add(name, None, f"dry-run: NORMAL LO @ band edge {price}")
            continue
        try:
            orders = broker._place(envelope(f"mkt-{side}"), side, QTY,
                                   price=price, category="NORMAL")
        except Exception as exc:                       # noqa: BLE001 — refusal is fine too
            result.add(name, True, f"refused by venue: {type(exc).__name__}: {exc}")
            continue

        ids = [str(o.id) for o in orders]
        PLACED.extend(ids)
        filled = sum(float(getattr(o, "filled_qty", 0) or 0) for o in orders)
        if filled:
            result.add(name, False,
                       f"FILLED OUT OF SESSION ({filled}) — real position opened! ids={ids}")
            continue
        result.add(name, True, f"queued at band edge {price}, filled=0, ids={ids}")

        cancelled = all(broker._cancel_one(oid) for oid in ids)
        result.add(f"{name}: cancel", cancelled,
                   "cancelled — nothing left to fill at the open" if cancelled
                   else "CANCEL FAILED — WOULD FILL AT THE OPEN, cancel it manually!")
        if cancelled:
            for oid in ids:
                if oid in PLACED:
                    PLACED.remove(oid)


async def part_resting(broker: DNSEBroker, result: Result, ref: float, *,
                       stop_limit: bool, dry_run: bool) -> None:
    """STOP (or STOP-LIMIT) both sides: place -> confirm resting -> cancel -> confirm gone."""
    label = "STOP-LIMIT" if stop_limit else "STOP"
    part = "3" if stop_limit else "2"
    print(f"\n[{part}] {label} long + short — expect RESTING, then cancellable")

    for side in ("buy", "sell"):
        way = "long" if side == "buy" else "short"
        name = f"{label.lower()} {way}"
        trigger = round(ref * (1 + AWAY) if side == "buy" else ref * (1 - AWAY), 1)
        # entry(stop)        -> price = stop        (the LO price once triggered)
        # entry(stop, limit) -> price = limit       (mirrors broker.execute_entry)
        limit = round(trigger * (1.0001 if side == "buy" else 0.999), 1) if stop_limit else trigger

        if dry_run:
            result.add(name, None, f"dry-run: STOP trigger={trigger} price={limit}")
            continue

        # --- place
        try:
            orders = broker._place(envelope(f"{label}-{side}"), side, QTY,
                                   price=limit, category="STOP", stop_price=trigger)
        except Exception as exc:                                   # noqa: BLE001
            result.add(f"{name}: place", False, f"{type(exc).__name__}: {exc}")
            continue
        ids = [str(o.id) for o in orders]
        PLACED.extend(ids)
        filled = sum(float(getattr(o, "filled_qty", 0) or 0) for o in orders)
        if filled:
            result.add(f"{name}: place", False,
                       f"FILLED ON PLACEMENT ({filled}) — not a resting stop! ids={ids}")
        else:
            result.add(f"{name}: place", True, f"accepted, filled=0, ids={ids}")

        # --- confirm it is really on the book
        await asyncio.sleep(1.0)
        present = all([await book_has(broker, oid) for oid in ids])
        result.add(f"{name}: rests on book", present,
                   "found on order book" if present else "NOT on the order book")

        # --- cancel
        cancelled = all(broker._cancel_one(oid) for oid in ids)
        result.add(f"{name}: cancel", cancelled,
                   "cancel accepted" if cancelled else "cancel FAILED")

        # --- confirm gone
        await asyncio.sleep(1.0)
        gone = not any([await book_has(broker, oid) for oid in ids])
        result.add(f"{name}: gone after cancel", gone,
                   "absent from book" if gone else "STILL RESTING — orphan!")
        if gone:
            for oid in ids:
                if oid in PLACED:
                    PLACED.remove(oid)


# --------------------------------------------------------------------------- main

PLACED: list[str] = []


async def run(args: argparse.Namespace) -> int:
    cfg = ensure_config(DNSEBrokerConfig,
                        REPO / "workdir" / "config" / "plugins" / "dnse_broker.toml")
    broker = DNSEBroker(symbol=SYMBOL, timeframe=TIMEFRAME, config=cfg)

    now = datetime.now(ICT)
    session = in_continuous_session(now)
    print(f"=== LEVEL 0 — DNSE order semantics @ {now:%Y-%m-%d %H:%M} ICT ===")
    print(f"session: {'OPEN (continuous)' if session else 'CLOSED (after hours / break)'}")
    print(f"account: {broker.account_id}   contract: {broker.resolve_contract()}")

    ref, ref_label = reference_close(broker)
    print(f"reference ({ref_label}): {ref}   -> long trigger {round(ref * 1.05, 1)}"
          f" / short trigger {round(ref * 0.95, 1)}")

    result = Result()
    try:
        await part1_market(broker, result, allow_in_session=args.allow_market_in_session,
                           dry_run=args.dry_run)
        await part_resting(broker, result, ref, stop_limit=False, dry_run=args.dry_run)
        await part_resting(broker, result, ref, stop_limit=True, dry_run=args.dry_run)
    finally:
        if PLACED:
            print(f"\n[cleanup] cancelling {len(PLACED)} leftover order(s): {PLACED}")
            for oid in list(PLACED):
                try:
                    if broker._cancel_one(oid):
                        print(f"  cancelled {oid}")
                        PLACED.remove(oid)
                    else:
                        print(f"  !! COULD NOT CANCEL {oid} — CANCEL IT MANUALLY")
                except Exception as exc:                            # noqa: BLE001
                    print(f"  !! cancel error {oid}: {exc} — CANCEL IT MANUALLY")
        if not args.dry_run:
            try:
                pos = await broker.get_position(SYMBOL)
                print(f"[cleanup] position now: {pos}")
            except Exception as exc:                                # noqa: BLE001
                print(f"[cleanup] position read failed: {exc}")

    print("\n=== SUMMARY ===")
    marks = {True: "PASS", False: "FAIL", None: "SKIP"}
    for name, ok, detail in result.rows:
        print(f"  [{marks[ok]}] {name}" + (f" — {detail}" if detail else ""))
    print("\nVERDICT:", "FAIL — see above" if result.failed else "PASS")
    return 1 if result.failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be sent; touch nothing")
    parser.add_argument("--allow-market-in-session", action="store_true",
                        help="DANGEROUS: run the market-order part during an open "
                             "session, where it WILL fill and open a real position")
    args = parser.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
