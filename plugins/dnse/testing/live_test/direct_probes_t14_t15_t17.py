#!/usr/bin/env python3
"""Direct venue probes — Live-L1-T14 / T15 / T17 (card rubycell/pynecore#22).

These three cases need timing the Pine bar clock cannot express, so they follow
the L0 pattern (raw ``DNSEBroker._place`` / ``client.cancel_order``), not a
staged .pine state:

  T14  AtcCancelRefusal   place two NORMAL LOs in late CONT-PM, then attempt to
                          cancel ONE of them repeatedly DURING the ATC — the venue
                          must REFUSE each attempt (measured code:
                          CANNOT_CANCEL_THE_ORDER_IN_THE_ATC_SESSION); the rider is
                          never cancelled. Both then EXPIRE at the close — grade
                          their terminal state from the venue the NEXT morning.
                          Live-L4-T03 proved no bars are delivered during ATC, so
                          this cancel can only be fired clock-driven, not from Pine.
  T15  CancelReplace      the #18 evidence: conditional amend -> HTTP 500, so can
                          cancel-then-replace substitute? Measure the AGGRESSIVE
                          gap: raw cancel ACK -> immediate replacement place on the
                          conditional book; report the exposure window and both
                          terminal states.
  T17  ReplaceUnderAckLag same probe on the NORMAL book with an IDENTICAL
                          replacement, placed inside the ~10 s stale-replica window
                          (measured 2026-08-17: a Canceled order read back as `New`)
                          — the venue must end with exactly ONE resting order.

House rules: 1 contract, >=4.5% away (nothing can fill), phase-stamped evidence
lines, explicit cleanup, L0 gate (separate script) before ANY live run, and
``--plan`` prints the intended steps without touching the venue (the offline gate).

    .venv/bin/python .../direct_probes_t14_t15_t17.py --case t15 [--plan]
    .venv/bin/python .../direct_probes_t14_t15_t17.py --case t14   # launch ~14:26 ICT
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent / "level0_venue_semantics"))

from pynecore.core.config import ensure_config                       # noqa: E402
from pynecore_dnse.broker import DNSEBroker, DNSEBrokerConfig        # noqa: E402
from l0_order_semantics import (                                     # noqa: E402
    envelope, reference_close, book_has, session_phase,
)

ICT = timezone(timedelta(hours=7))
SYMBOL = "VN30F1M"
QTY = 1
AWAY = 0.05


def now_str() -> str:
    return datetime.now(ICT).strftime("%H:%M:%S")


def stamp(case: str, msg: str) -> None:
    print(f"[{case}] {now_str()} phase={session_phase()} {msg}", flush=True)


def new_broker() -> DNSEBroker:
    cfg = ensure_config(DNSEBrokerConfig,
                        REPO / "workdir" / "config" / "plugins" / "dnse_broker.toml")
    return DNSEBroker(symbol=SYMBOL, timeframe="1", config=cfg)


def raw_cancel(b: DNSEBroker, order_id: str, category: str) -> tuple[int, dict | str]:
    """One raw cancel call — returns the venue's (status, body) unfiltered."""
    return b._write(lambda tok: b.client.cancel_order(
        b.account_id, order_id, b.market_type, tok, order_category=category))


def detail(b: DNSEBroker, order_id: str, category: str) -> str:
    status, body = b.client.get_order_detail(b.account_id, order_id, b.market_type,
                                             order_category=category)
    if isinstance(body, dict) and body.get("orderStatus"):
        return str(body["orderStatus"])
    return f"HTTP {status}"


def place(b: DNSEBroker, tag: str, *, price: float, category: str = "NORMAL",
          stop_price: float | None = None) -> str:
    orders = b._place(envelope(tag), "buy", QTY, price=price,
                      category=category, stop_price=stop_price)
    oid = str(orders[0].id)
    stamp(tag, f"PLACED id={oid} cat={category} price={price} stop={stop_price} "
               f"filled={getattr(orders[0], 'filled_qty', 0)}")
    return oid


def cleanup(b: DNSEBroker, ids: list[tuple[str, str]], case: str) -> bool:
    """Best-effort cancel of every (id, category) we placed; True if book is clean."""
    for oid, _cat in ids:
        try:
            b._cancel_one(oid)
        except Exception as exc:                                     # noqa: BLE001
            stamp(case, f"cleanup cancel {oid} raised {type(exc).__name__}: {exc}")
    time.sleep(3)
    left = [oid for oid, _ in ids if asyncio.run(book_has(b, oid))]
    stamp(case, f"CLEANUP {'clean' if not left else 'STILL WORKING: ' + ','.join(left)}")
    return not left


# ------------------------------------------------------------------ T15
def t15_cancel_replace(plan: bool) -> int:
    """Conditional book: cancel ACK -> immediate replacement. #18 evidence."""
    if plan:
        print("[t15/plan] place buy-STOP A trigger=+5%; confirm resting; raw cancel A "
              "(record ACK latency); IMMEDIATELY place buy-STOP B trigger=+4.5%; "
              "record the gap; poll A->Canceled and B->resting (<=30 s, stale-replica "
              "readings logged, not failed); cleanup cancel B; PASS = B accepted, no "
              "ghost, book clean. Continuous or lunch phase only.")
        return 0
    if session_phase() not in ("continuous", "lunch"):
        stamp("t15", "REFUSING to start — conditionals need continuous/lunch phase")
        return 1
    b = new_broker()
    ref, src = reference_close(b)
    stamp("t15", f"reference {ref} ({src})")
    a_trig = round(ref * (1 + AWAY), 1)
    b_trig = round(ref * (1 + AWAY - 0.005), 1)
    a_id = place(b, "t15A", price=a_trig, category="STOP", stop_price=a_trig)
    ids = [(a_id, "STOP")]
    try:
        time.sleep(2)
        if not asyncio.run(book_has(b, a_id)):
            stamp("t15", "FAIL — A not resting after 2 s")
            return 1
        t0 = time.monotonic()
        status, body = raw_cancel(b, a_id, "STOP")
        t_ack = time.monotonic() - t0
        stamp("t15", f"CANCEL A ack http={status} in {t_ack*1000:.0f} ms body={str(body)[:80]}")
        b_id = place(b, "t15B", price=b_trig, category="STOP", stop_price=b_trig)
        ids.append((b_id, "STOP"))
        t_gap = time.monotonic() - t0
        stamp("t15", f"GAP cancel-ack -> replacement-placed = {t_gap*1000:.0f} ms")
        deadline = time.time() + 30
        a_st = b_st = "?"
        while time.time() < deadline:
            a_st, b_st = detail(b, a_id, "STOP"), detail(b, b_id, "STOP")
            stamp("t15", f"poll A={a_st} B={b_st}")
            if a_st in ("Canceled", "Cancelled") and b_st in ("New", "PendingNew", "Pending"):
                break
            time.sleep(5)
        ok = a_st in ("Canceled", "Cancelled") and b_st in ("New", "PendingNew", "Pending")
        print(f"\nVERDICT t15: {'PASS' if ok else 'FAIL'} — A={a_st} B={b_st} "
              f"exposure-gap={t_gap*1000:.0f} ms (feeds the #18 cancel+replace decision)")
        return 0 if ok else 1
    finally:
        cleanup(b, ids, "t15")


# ------------------------------------------------------------------ T17
def t17_replace_under_ack_lag(plan: bool) -> int:
    """NORMAL book: identical replacement inside the stale-replica window."""
    if plan:
        print("[t17/plan] place buy LO A at -5%; confirm resting; raw cancel A (2xx "
              "ACK); IMMEDIATELY place IDENTICAL LO A' (same price/qty); poll both "
              "<=30 s logging every reading (stale replica may show A as New — "
              "documented, not failed); PASS = final state exactly ONE working (A' "
              "resting, A Canceled); cleanup cancel A'. Continuous phase only.")
        return 0
    if session_phase() != "continuous":
        stamp("t17", "REFUSING to start — NORMAL LO probe wants continuous phase")
        return 1
    b = new_broker()
    ref, src = reference_close(b)
    stamp("t17", f"reference {ref} ({src})")
    px = round(ref * (1 - AWAY), 1)
    a_id = place(b, "t17A", price=px)
    ids = [(a_id, "NORMAL")]
    try:
        time.sleep(2)
        if not asyncio.run(book_has(b, a_id)):
            stamp("t17", "FAIL — A not resting after 2 s")
            return 1
        status, body = raw_cancel(b, a_id, "NORMAL")
        stamp("t17", f"CANCEL A ack http={status} body={str(body)[:80]}")
        a2_id = place(b, "t17A2", price=px)          # identical, inside the lag window
        ids.append((a2_id, "NORMAL"))
        deadline = time.time() + 30
        a_st = a2_st = "?"
        both_working_final = False
        while time.time() < deadline:
            a_st, a2_st = detail(b, a_id, "NORMAL"), detail(b, a2_id, "NORMAL")
            working = [s for s in (a_st, a2_st) if s in ("New", "PendingNew", "Pending", "PartiallyFilled")]
            stamp("t17", f"poll A={a_st} A'={a2_st} working={len(working)}")
            both_working_final = len(working) > 1
            if a_st in ("Canceled", "Cancelled") and a2_st in ("New", "PendingNew", "Pending"):
                break
            time.sleep(5)
        ok = (a_st in ("Canceled", "Cancelled")
              and a2_st in ("New", "PendingNew", "Pending")
              and not both_working_final)
        print(f"\nVERDICT t17: {'PASS' if ok else 'FAIL'} — final A={a_st} A'={a2_st} "
              f"(transient double-readings are the documented stale replica, only the "
              f"FINAL state grades)")
        return 0 if ok else 1
    finally:
        cleanup(b, ids, "t17")


# ------------------------------------------------------------------ T14
def t14_atc_cancel_refusal(plan: bool) -> int:
    """Late-CONT-PM placement, clock-driven cancel attempts through the ATC."""
    if plan:
        print("[t14/plan] launch 14:20-14:29 ICT; place buy LO T14 at -5% and rider "
              "T14r at -6% (NORMAL, day duration); wait for phase==atc; from 14:30:30 "
              "attempt raw cancel of T14 every 120 s until 14:44:30 — EXPECT refusal "
              "each time (record http+code, e.g. CANNOT_CANCEL_THE_ORDER_IN_THE_ATC_"
              "SESSION); T14r is NEVER cancelled; after 14:45 read both details and "
              "exit. GRADE NEXT MORNING from the venue: both must be Expired, "
              "nothing working. If a cancel unexpectedly SUCCEEDS in ATC that is a "
              "FINDING (venue behavior change) — the probe then cancels the rider "
              "too and reports.")
        return 0
    now = datetime.now(ICT)
    if not (now.hour == 14 and 20 <= now.minute <= 29):
        stamp("t14", f"REFUSING to start — launch window is 14:20-14:29 ICT (now {now_str()})")
        return 1
    b = new_broker()
    ref, src = reference_close(b)
    stamp("t14", f"reference {ref} ({src})")
    t14_id = place(b, "t14", price=round(ref * (1 - AWAY), 1))
    rider_id = place(b, "t14r", price=round(ref * (1 - AWAY - 0.01), 1))
    stamp("t14", f"resting into the auction: T14={t14_id} rider={rider_id} — "
                 f"cancel attempts start at 14:30:30")
    refusals, successes = 0, 0
    while datetime.now(ICT).hour == 14 and datetime.now(ICT).minute < 45:
        now = datetime.now(ICT)
        if session_phase() != "atc":
            time.sleep(10)
            continue
        if now.minute >= 45:
            break
        status, body = raw_cancel(b, t14_id, "NORMAL")
        code = body.get("code") if isinstance(body, dict) else ""
        if status in (200, 204):
            successes += 1
            stamp("t14", f"CANCEL UNEXPECTEDLY ACCEPTED http={status} — FINDING "
                         f"(venue behavior change); cancelling rider too")
            raw_cancel(b, rider_id, "NORMAL")
            break
        refusals += 1
        stamp("t14", f"CANCEL REFUSED http={status} code={code} "
                     f"msg={str(body)[:100]} (attempt {refusals})")
        time.sleep(120)
    while datetime.now(ICT) < datetime.now(ICT).replace(hour=14, minute=45, second=30):
        time.sleep(5)
    a_st, r_st = detail(b, t14_id, "NORMAL"), detail(b, rider_id, "NORMAL")
    print(f"\nVERDICT t14 (provisional): refusals={refusals} unexpected-accepts={successes} "
          f"post-close T14={a_st} rider={r_st}")
    print("GRADE NEXT MORNING at the venue: both orders must be Expired, nothing "
          "of ours working. Refusal code expected on every ATC attempt.")
    return 0 if refusals > 0 and successes == 0 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True, choices=["t14", "t15", "t17", "t18"])
    ap.add_argument("--plan", action="store_true",
                    help="print the intended steps, touch nothing (offline gate)")
    args = ap.parse_args(argv)
    return {"t14": t14_atc_cancel_refusal,
            "t15": t15_cancel_replace,
            "t17": t17_replace_under_ack_lag,
            "t18": t18_immediate_cancel}[args.case](args.plan)




# ------------------------------------------------------------------ T18
def t18_immediate_cancel(plan: bool) -> int:
    """Cancel the instant the order is VISIBLE on the live book — no bar-clock wait.

    The staged suite cancels on the NEXT 1m bar (Pine's clock); this measures the
    bar-free floor on BOTH books: place -> poll get_open_orders (~100 ms) until the
    id appears -> cancel immediately -> poll until the venue agrees it is gone.
    PASS = venue accepts the immediate cancel (no minimum-rest rule) on both books.
    """
    if plan:
        print("[t18/plan] NORMAL leg: place buy LO -5%; poll open orders every ~100 ms "
              "until visible; cancel IMMEDIATELY; poll detail to Canceled. Repeat with "
              "a conditional STOP (+5% trigger). Report place-ack / book-visible / "
              "cancel-ack / confirmed-gone latencies per book. Continuous or lunch "
              "phase; nothing can fill (>=5% away).")
        return 0
    if session_phase() not in ("continuous", "lunch"):
        stamp("t18", "REFUSING to start — needs continuous/lunch phase")
        return 1
    b = new_broker()
    ref, src = reference_close(b)
    stamp("t18", f"reference {ref} ({src})")
    failures = 0
    for label, category, price, stop in (
            ("NORMAL", "NORMAL", round(ref * (1 - AWAY), 1), None),
            ("STOP", "STOP", round(ref * (1 + AWAY), 1), round(ref * (1 + AWAY), 1))):
        t0 = time.monotonic()
        oid = place(b, f"t18-{label.lower()}", price=price, category=category,
                    stop_price=stop)
        t_place = time.monotonic() - t0
        ids = [(oid, category)]
        try:
            visible = None
            deadline = time.time() + 15
            while time.time() < deadline:
                if asyncio.run(book_has(b, oid)):
                    visible = time.monotonic() - t0
                    break
                time.sleep(0.1)
            if visible is None:
                stamp("t18", f"{label}: FAIL — never visible on the book within 15 s")
                failures += 1
                continue
            t1 = time.monotonic()
            status, body = raw_cancel(b, oid, category)
            t_ack = time.monotonic() - t1
            code = body.get("code") if isinstance(body, dict) else ""
            gone = None
            deadline = time.time() + 20
            while time.time() < deadline:
                st = detail(b, oid, category)
                if st in ("Canceled", "Cancelled"):
                    gone = time.monotonic() - t1
                    break
                time.sleep(0.2)
            ok = status in (200, 204) and gone is not None
            stamp("t18", f"{label}: place-ack={t_place*1000:.0f}ms book-visible={visible*1000:.0f}ms "
                         f"cancel http={status} code={code or '-'} ack={t_ack*1000:.0f}ms "
                         f"confirmed-gone={gone*1000:.0f}ms -> {'OK' if ok else 'FAIL'}")
            if not ok:
                failures += 1
        finally:
            cleanup(b, ids, "t18")
    print(f"\nVERDICT t18: {'PASS — immediate cancel accepted on both books, no minimum-rest rule' if failures == 0 else 'FAIL'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
