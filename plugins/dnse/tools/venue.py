#!/usr/bin/env python3
"""DNSE venue toolkit — the answers we keep needing, as commands instead of
hand-written probes.

WHY THIS EXISTS: every live session used to re-type the same
``sys.path.insert`` + ``ensure_config`` + ``DNSEBroker(...)`` heredoc a dozen
times, and each retype was a fresh chance to invent an API. On 2026-08-19 one of
those improvised probes called a method that does not exist (``fetch_position``,
which is a CAPABILITY flag, not a method — the method is ``get_position``),
guarded it with ``hasattr``, got ``None``, and reported the account FLAT while it
held +2 contracts. A wrapped, tested command cannot make that mistake.

EXIT CODES — the whole point, so a script can trust the answer:
    0  question answered, affirmative/neutral (e.g. flat, order is terminal)
    1  question answered, negative (e.g. NOT flat, order still working)
    2  COULD NOT DETERMINE — the read failed. Never confuse this with 0.

Read-only by default. ``cancel`` and ``sweep`` write, and ``sweep`` refuses to run
without ``--yes`` because it touches every working order on the account —
INCLUDING THE OPERATOR'S OWN (DNSE nets per symbol; the venue has no notion of
"this run's" orders).

    venue.py status                 # phase, token, position, working orders
    venue.py flat                   # exit 0 only if flat AND no working orders
    venue.py order <id> [<id>...]   # detail, auto book, previous-day fallback
    venue.py cancel <id> [<id>...]  # cancel and VERIFY terminal
    venue.py sweep --yes            # cancel every working order (destructive)
    venue.py history [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import asyncio

from pynecore.core.broker.models import CancelDispositionOutcome

#: #55: bool ``_cancel_one`` is gone; OK = positively cancelled or
#: terminal-without-fill. ALREADY_FILLED is NOT ok (a fill is not a cancel).
_CANCEL_OK = (CancelDispositionOutcome.CANCEL_CONFIRMED,
              CancelDispositionOutcome.TOO_LATE_TO_CANCEL)
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "plugins" / "dnse" / "testing" / "live_test"
                      / "level0_venue_semantics"))

from pynecore.core.config import ensure_config                       # noqa: E402
from pynecore_dnse.broker import DNSEBroker, DNSEBrokerConfig        # noqa: E402

ICT = timezone(timedelta(hours=7))
SYMBOL = "VN30F1M"
BOOKS = ("NORMAL", "STOP", "OCO")

EXIT_OK, EXIT_NEGATIVE, EXIT_UNKNOWN = 0, 1, 2


def broker(symbol: str = SYMBOL) -> DNSEBroker:
    return DNSEBroker(symbol=symbol, timeframe="1", config=ensure_config(
        DNSEBrokerConfig, REPO / "workdir" / "config" / "plugins" / "dnse_broker.toml"))


def session_phase() -> str:
    """Reuse L0's phase logic — one definition, not two that drift.

    Display-only holiday annotation (#70): the phase TOKEN stays "closed"
    (exact-string consumers), the WHY is appended here for the operator."""
    try:
        from l0_order_semantics import session_phase as _phase
        phase = _phase()
        try:
            from l0_order_semantics import is_exchange_holiday
            if phase == "closed" and is_exchange_holiday(datetime.now(ICT)):
                return "closed (exchange holiday)"
        except Exception:                                             # noqa: BLE001
            pass                     # annotation is cosmetic, never load-bearing
        return phase
    except Exception as exc:                                          # noqa: BLE001
        return f"UNKNOWN ({type(exc).__name__})"


def token_verdict(state: str | None = None) -> str:
    """The canonical check, run as its own tool rather than reimplemented.

    ``state`` overrides the token file (used to exercise the not-GOOD path
    without touching the real token — see rule 2: a check must be red-first).
    """
    tool = REPO / "plugins" / "dnse" / "tools" / "token_status.py"
    cmd = [sys.executable, str(tool)] + (["--state", state] if state else [])
    try:
        out = subprocess.run(cmd, capture_output=True,
                             text=True, timeout=60, cwd=REPO)
        for line in reversed(out.stdout.splitlines()):
            if line.startswith("VERDICT:"):
                return line.split(":", 1)[1].strip()
        return "UNKNOWN (no verdict line)"
    except Exception as exc:                                          # noqa: BLE001
        return f"UNKNOWN ({type(exc).__name__})"


def read_state(b: DNSEBroker, symbol: str):
    """-> (position, working_orders). Raises if either read fails — a failed read
    must NEVER look like 'nothing there' (that is the bug this tool exists for)."""
    position = asyncio.run(b.get_position(symbol))     # raises on read failure
    working = asyncio.run(b.get_open_orders(symbol))   # raises if books unavailable
    return position, working


def classify_working(b: DNSEBroker, orders: list) -> tuple[list, list]:
    """Split working orders into (LIVE, PHANTOM).

    A conditional that has already triggered stays ``Activated`` on the STOP book
    forever — its child on the NORMAL book did the actual work — but the books
    still list it as working (issue #41). Counting those as live makes "is the
    account clean?" cry wolf after every triggered stop, so they are reported
    separately rather than ignored.
    """
    live, phantom = [], []
    for o in orders:
        _, row = _detail_today(b, str(o.id))
        if (row and str(row.get("orderStatus", "")).upper() == "ACTIVATED"
                and row.get("externalOrderId")):
            phantom.append((o, row.get("externalOrderId")))
        else:
            live.append(o)
    return live, phantom


def fmt_order(o) -> str:
    return (f"{str(o.id):22s} {str(getattr(o, 'side', '?')):4s} "
            f"qty={getattr(o, 'qty', '?')} price={getattr(o, 'price', '?')} "
            f"status={getattr(o, 'status', '?')}")


# --------------------------------------------------------------------- commands

def cmd_status(args) -> int:
    b = broker(args.symbol)
    print(f"phase   : {session_phase()}   ({datetime.now(ICT):%Y-%m-%d %H:%M:%S} ICT)")
    print(f"token   : {token_verdict(args.token_state)}")
    try:
        position, working = read_state(b, args.symbol)
    except Exception as exc:                                          # noqa: BLE001
        print(f"position: COULD NOT READ — {type(exc).__name__}: {exc}")
        print("working : COULD NOT READ")
        return EXIT_UNKNOWN
    live, phantom = classify_working(b, working)
    print(f"position: {'FLAT' if position is None else position}")
    print(f"working : {len(live)} live, {len(phantom)} phantom (consumed conditionals, #41)")
    for o in live:
        print(f"   LIVE    {fmt_order(o)}")
    for o, child in phantom:
        print(f"   phantom {fmt_order(o)}  -> child {child} did the work")
    return EXIT_OK


def cmd_flat(args) -> int:
    """The question that was answered wrongly on 2026-08-19."""
    b = broker(args.symbol)
    try:
        position, working = read_state(b, args.symbol)
    except Exception as exc:                                          # noqa: BLE001
        print(f"UNDETERMINED — the read FAILED ({type(exc).__name__}: {exc}). "
              f"This is NOT 'flat'.")
        return EXIT_UNKNOWN
    live, phantom = classify_working(b, working)
    if position is None and not live:
        print(f"FLAT — no position, no LIVE orders (both reads succeeded)"
              + (f"; {len(phantom)} phantom shell(s) ignored, see #41" if phantom else ""))
        return EXIT_OK
    if position is not None:
        print(f"NOT FLAT — {position}")
    if live:
        print(f"NOT CLEAN — {len(live)} LIVE order(s):")
        for o in live:
            print(f"  {fmt_order(o)}")
    return EXIT_NEGATIVE


def _detail_today(b: DNSEBroker, oid: str):
    """(book, status_dict) for an order that still lives in today's books."""
    for book in BOOKS:
        status, body = b.client.get_order_detail(b.account_id, oid, b.market_type,
                                                 order_category=book)
        if status == 200 and isinstance(body, dict) and body.get("orderStatus"):
            return book, body
    return None, None


def _detail_history(b: DNSEBroker, oid: str, days: int = 7):
    """Previous-day orders are NOT on the detail endpoint (it answers None) —
    they live in /orders/history — the vendored SDK's ``get_order_history`` (one
    call covers BOTH books; rows are date-prefixed, 20260818_538916, and arrive
    under 'data', not 'orders'). Measured 2026-08-19."""
    today = datetime.now(ICT).date()
    st, body = b.client.get_order_history(
        b.account_id, b.market_type, from_date=str(today - timedelta(days=days)),
        to_date=str(today), page_size=200)
    if st != 200 or not isinstance(body, dict):
        return None, None
    for row in body.get("data") or []:
        if str(row.get("id", "")).split("_")[-1] == str(oid):
            return "history", row
    return None, None


def cmd_order(args) -> int:
    b = broker(args.symbol)
    worst = EXIT_OK
    for oid in args.ids:
        book, row = _detail_today(b, oid)
        if row is None:
            book, row = _detail_history(b, oid)
        if row is None:
            print(f"{oid}: NOT FOUND in any book or in 7 days of history — "
                  f"UNDETERMINED (not proof it never existed)")
            worst = max(worst, EXIT_UNKNOWN)
            continue
        print(f"{oid} [{book}]: status={row.get('orderStatus')} "
              f"price={row.get('price')} stop={row.get('stopPrice')} "
              f"qty={row.get('quantity')} filled={row.get('fillQuantity')} "
              f"child={row.get('externalOrderId')}")
    return worst


def cmd_cancel(args) -> int:
    # Writes need a live trading token (~8h TTL, so it expires most mornings).
    # Refuse with a clear message rather than surfacing an AuthenticationError
    # from inside the cancel loop after some ids were already attempted.
    verdict = token_verdict(args.token_state)
    if not verdict.startswith("GOOD"):
        print(f"REFUSING to write — token is not GOOD: {verdict}\n"
              f"mint one first: .venv/bin/python plugins/dnse/tools/refresh_token.py "
              f"--send   (then --otp <code>)")
        return EXIT_UNKNOWN
    b = broker(args.symbol)
    worst = EXIT_OK
    for oid in args.ids:
        try:
            ok = asyncio.run(b._cancel_one_disposition(oid)) in _CANCEL_OK
        except Exception as exc:                                      # noqa: BLE001
            print(f"{oid}: cancel RAISED {type(exc).__name__}: {exc}")
            worst = max(worst, EXIT_UNKNOWN)
            continue
        book, row = _detail_today(b, oid)
        state = row.get("orderStatus") if row else "unreadable"
        terminal = state in ("Canceled", "Cancelled", "Filled", "Expired", "Rejected")
        print(f"{oid}: cancel_one={ok} venue={state} "
              f"{'(terminal)' if terminal else '(STILL LIVE — recheck; DNSE detail '
                                              'reads are eventually consistent)'}")
        if not terminal:
            worst = max(worst, EXIT_NEGATIVE)
    return worst


def cmd_sweep(args) -> int:
    b = broker(args.symbol)
    try:
        _, working = read_state(b, args.symbol)
    except Exception as exc:                                          # noqa: BLE001
        print(f"UNDETERMINED — cannot list working orders ({type(exc).__name__}: {exc})")
        return EXIT_UNKNOWN
    if not working:
        print("nothing working — nothing to sweep")
        return EXIT_OK
    print(f"about to cancel {len(working)} working order(s) — this includes ANY "
          f"order on the account, not just a test run's:")
    for o in working:
        print(f"  {fmt_order(o)}")
    if not args.yes:
        print("\nrefusing without --yes")
        return EXIT_NEGATIVE
    args.ids = [str(o.id) for o in working]
    return cmd_cancel(args)


def cmd_history(args) -> int:
    b = broker(args.symbol)
    day = args.date or str(datetime.now(ICT).date())
    st, body = b.client.get_order_history(b.account_id, b.market_type,
                                          from_date=day, to_date=day, page_size=200)
    if st != 200 or not isinstance(body, dict):
        print(f"history read FAILED: HTTP {st} — UNDETERMINED, not 'no orders'")
        return EXIT_UNKNOWN
    rows = body.get("data") or []
    for row in rows:
        print(f"{str(row.get('id')):24s} {str(row.get('side')):3s} "
              f"{str(row.get('orderStatus')):10s} price={row.get('price')} "
              f"filled={row.get('fillQuantity')}")
    print(f"-- {len(rows)} order(s) on {day}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default=SYMBOL)
    ap.add_argument("--token-state", default=None,
                    help="override the trading-token state file (for testing the "
                         "not-GOOD path without touching the real token)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("flat")
    p_order = sub.add_parser("order"); p_order.add_argument("ids", nargs="+")
    p_cancel = sub.add_parser("cancel"); p_cancel.add_argument("ids", nargs="+")
    p_sweep = sub.add_parser("sweep"); p_sweep.add_argument("--yes", action="store_true")
    p_hist = sub.add_parser("history"); p_hist.add_argument("--date")
    args = ap.parse_args(argv)
    return {"status": cmd_status, "flat": cmd_flat, "order": cmd_order,
            "cancel": cmd_cancel, "sweep": cmd_sweep,
            "history": cmd_history}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
