#!/usr/bin/env python3
"""Live verification of the cancel ACK-vs-completion fix (rubycell/pynecore#20).

Runs at ANY hour — it only uses conditional (STOP) orders, which DNSE accepts outside a
session. Places one 1-contract STOP ~5% from market (inside the +/-7% band, so it rests and
cannot trigger), then instruments ``_cancel_one`` to prove the fixed contract holds against
the real venue:

  1. ``_cancel_one`` returns True only when the VENUE reports the order terminal.
  2. It actually polls the order back (``get_order_detail`` call count > 0) rather than
     trusting the cancel's 2xx.
  3. The venue's own record says ``Canceled`` afterwards.

Before 902c156 a 2xx alone returned True while the order stayed ``New`` — measured
2026-08-13, three consecutive cancels reported success on an order that was still live.

SAFETY: the order is far from market and 1 contract; a ``finally`` block cancels it and
reports the position even on error.

    .venv/bin/python plugins/dnse/testing/live_test/level0_venue_semantics/verify_cancel_ack.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))

from pynecore.core.config import ensure_config                      # noqa: E402
from pynecore_dnse.broker import DNSEBroker, DNSEBrokerConfig       # noqa: E402

SYMBOL, TIMEFRAME, QTY, AWAY = "VN30F1M", "1", 1.0, 0.05


def envelope(tag: str) -> SimpleNamespace:
    return SimpleNamespace(intent=SimpleNamespace(
        intent_key=f"vca-{tag}", pine_id=f"VCA-{tag}", from_entry=None))


def venue_status(broker: DNSEBroker, order_id: str,
                 category: str = "STOP") -> str | None:
    status, body = broker.client.get_order_detail(
        broker.account_id, order_id, broker.market_type, order_category=category)
    if status == 200 and isinstance(body, dict):
        return body.get("orderStatus")
    return f"HTTP {status}"


def main() -> int:
    cfg = ensure_config(DNSEBrokerConfig,
                        REPO / "workdir" / "config" / "plugins" / "dnse_broker.toml")
    broker = DNSEBroker(symbol=SYMBOL, timeframe=TIMEFRAME, config=cfg)

    to_ts = int(datetime.now(timezone.utc).timestamp())
    st, body = broker.client.get_ohlc(broker.market_type, {
        "symbol": broker.symbol, "resolution": "1", "from": to_ts - 3 * 24 * 3600, "to": to_ts})
    ref = float(body["c"][-1])
    trigger = round(ref * (1 + AWAY), 1)

    print(f"=== verify cancel ACK-vs-completion (#20) @ "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
    print(f"account {broker.account_id}  ref {ref}  buy-stop trigger {trigger}")

    # count the readback polls the fix performs
    polls = {"n": 0}
    real_detail = broker.client.get_order_detail

    def counting_detail(*a, **k):
        polls["n"] += 1
        return real_detail(*a, **k)

    broker.client.get_order_detail = counting_detail            # type: ignore[assignment]

    # Which instrument can we safely place right now?
    #   * inside the trading day -> a conditional STOP (the book the bug was found on)
    #   * once the day is over    -> conditionals are rejected (CO-ORD-006), so fall back
    #     to a NORMAL limit priced FAR below market. It still exercises the identical
    #     code path (cancel -> 2xx -> readback), and a buy-limit 5% under cannot fill,
    #     not even at the next open, so nothing can be stranded.
    from datetime import timedelta
    sys.path.insert(0, str(Path(__file__).parent))
    from l0_order_semantics import session_phase                    # noqa: E402
    phase = session_phase()
    use_stop = phase in ("continuous", "lunch")
    far_limit = round(ref * (1 - AWAY), 1)

    order_id = None
    failures = []
    try:
        if use_stop:
            orders = broker._place(envelope("stop"), "buy", QTY,
                                   price=trigger, category="STOP", stop_price=trigger)
            kind = f"STOP trigger={trigger}"
        else:
            orders = broker._place(envelope("lim"), "buy", QTY, price=far_limit)
            kind = f"NORMAL limit={far_limit} (-5%, cannot fill)"
        order_id = str(orders[0].id)
        print(f"\nphase={phase} -> placed {kind}")
        print(f"placed id={order_id}  filled={orders[0].filled_qty}")
        print(f"  venue status before cancel: {venue_status(broker, order_id, "STOP" if use_stop else "NORMAL")}")

        polls["n"] = 0
        t0 = time.monotonic()
        confirmed = broker._cancel_one(order_id)
        elapsed = time.monotonic() - t0
        after = venue_status(broker, order_id, "STOP" if use_stop else "NORMAL")

        print(f"\n_cancel_one -> {confirmed}   readback polls: {polls['n']}   "
              f"elapsed {elapsed:.2f}s")
        print(f"  venue status after cancel:  {after}")

        # --- the contract -------------------------------------------------
        if polls["n"] < 1:
            failures.append("the fix must re-read the order; no get_order_detail call was made")
        if confirmed and after not in ("Canceled", "Cancelled", "Filled", "Rejected", "Expired"):
            failures.append(f"reported cancelled while the venue says {after!r} "
                            f"— this is exactly the bug #20 fixed")
        if not confirmed and after in ("Canceled", "Cancelled"):
            failures.append(f"venue says {after!r} but the plugin reported NOT cancelled "
                            "(over-cautious: the poll budget may be too small)")
        if not confirmed and after not in ("Canceled", "Cancelled"):
            failures.append(f"cancel did not take effect at all (venue={after!r}) — "
                            "correctly reported False, but the order needs manual cleanup")
        order_id = None if confirmed and after in ("Canceled", "Cancelled") else order_id
    finally:
        broker.client.get_order_detail = real_detail            # type: ignore[assignment]
        if order_id:
            print(f"\n[cleanup] order {order_id} may still be working — cancelling")
            try:
                print("  cancel ->", broker._cancel_one(order_id),
                      "| venue:", venue_status(broker, order_id, "STOP" if use_stop else "NORMAL"))
            except Exception as exc:                            # noqa: BLE001
                print(f"  !! cleanup failed: {exc} — CANCEL IT MANUALLY")
        try:
            print("[cleanup] position:", asyncio.run(broker.get_position(SYMBOL)))
        except Exception as exc:                                # noqa: BLE001
            print(f"[cleanup] position read failed: {exc}")

    print("\n=== RESULT ===")
    for f in failures:
        print(f"  [FAIL] {f}")
    print("VERDICT:", "FAIL" if failures else
          "PASS — a 2xx is only trusted once the venue agrees")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
