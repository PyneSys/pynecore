"""#91 — fill-tier flatten: close FIRST, then sweep OUR protection.

Measured live 2026-09-08 (operator-caught): the old close-only flatten left
the closed position's protective stop ARMED on a flat account — with the
engine down, a naked entry-stop until the 14:45 DAY expiry (#82 class).

Design (card #91, panel-adjudicated, unanimous):

* **Close-first** — a rejected close under protection-first is one failure
  from an UNBOUNDED naked open (near-market closes CAN reject, measured
  F6), and re-arming protection after a refused close is a conditional
  write of the same class that just refused. Close-first's failure state
  leaves the account unchanged and retryable.
* **The sweep runs only after the close is CONFIRMED flat** — never on an
  unconfirmed close.
* **Attribution is a READ-ONLY store query** (:func:`owned_live_ids`),
  never ``connect()``/``open_run``: ``open_run`` raises on the live-row
  collision for the whole 5-minute stale window (storage.py:886) — exactly
  the post-crash emergency this tool exists for — and would steal the run
  identity from the #77/T16 restart adoption. A missing/unreadable store
  is "attribution UNAVAILABLE" (``None``) and the caller must exit 2 —
  ``BrokerStore`` is never instantiated here because its constructor
  CREATES the file, which would make absence read as "clean".
* **The netting account is SHARED** — a venue order outside the owned set
  is the operator's: reported, never cancelled (hard rule).
* Sweep cancels go through the plugin's disposition core
  (``_cancel_one_disposition`` — the same path ``venue.py cancel`` uses):
  a #41 phantom shell resolves ``ALREADY_FILLED`` via its child, a
  double-cancel racing a live engine answers the measured-terminal
  ``ORDER_CANCEL_STATUS_REJECTED``, and a #51 refusal classifies
  ``UNKNOWN`` — reported and exit 1, never retried (#58 write policy).

Exit codes (venue.py convention): 0 flat AND owned orders swept/resolved;
1 not flat, or an owned order's disposition stayed UNKNOWN; 2 could not
determine (position unreadable, or attribution unavailable).
"""
import asyncio
import sqlite3
import time
from pathlib import Path

from pynecore.core.broker.models import (
    CancelDispositionOutcome, CloseIntent, DispatchEnvelope,
)

#: Sweep outcomes that mean "this order is provably done" (fill outranks —
#: an ALREADY_FILLED here is a consumed entry/shell, not a working order).
_RESOLVED = frozenset({
    CancelDispositionOutcome.CANCEL_CONFIRMED,
    CancelDispositionOutcome.ALREADY_FILLED,
    CancelDispositionOutcome.TOO_LATE_TO_CANCEL,
    CancelDispositionOutcome.STILL_OPEN,
})


def owned_live_ids(store_path) -> "set[str] | None":
    """Venue ids the BOT owns, from the journal store — read-only.

    Every un-closed ``orders`` row with a venue id in this store was
    written by the bot (the store is the bot's journal; foreign book rows
    are never journalled, #36). Deliberately NO symbol filter: the store's
    ``symbol`` column holds the WIRE symbol (``41I1G9000`` — measured on
    the live store; the #77 wire-symbol join lesson), so a Pine-symbol
    filter silently matches NOTHING and the sweep passes vacuously. The
    sweep intersects this set with the venue's working orders for the
    target symbol, so cross-symbol ids are inert. ``None`` = attribution
    UNAVAILABLE (store missing or unreadable) — the caller must treat
    that as exit 2, never as an empty owned set. The file is NEVER
    created here.
    """
    path = Path(store_path)
    if not path.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT DISTINCT exchange_order_id FROM orders "
                "WHERE closed_ts_ms IS NULL "
                "  AND exchange_order_id IS NOT NULL "
                "  AND exchange_order_id != ''").fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        print(f"attribution UNAVAILABLE: store read failed: {exc}")
        return None
    return {str(row[0]) for row in rows}


def _read_position_size(broker, symbol: str) -> "float | None":
    """Signed net size, or ``None`` when the read failed (never 'flat')."""
    try:
        pos = asyncio.run(broker.get_position(symbol))
    except Exception as exc:                                  # noqa: BLE001
        print(f"COULD NOT READ position: {type(exc).__name__}: {exc}")
        return None
    if pos is None:
        return 0.0
    return float(pos.size or 0.0)


def flatten(broker, symbol: str, owned_ids: "set[str] | None",
            *, close_wait_s: int = 25) -> int:
    """Close-first flatten + owned-order sweep. See module docstring."""
    size = _read_position_size(broker, symbol)
    if size is None:
        return 2

    if size != 0.0:
        close_side = "sell" if size > 0 else "buy"
        qty = abs(size)
        print(f"position: {'long' if size > 0 else 'short'} {qty} -> "
              f"closing via execute_close ({close_side} {int(qty)} @ band edge)")
        envelope = DispatchEnvelope(
            intent=CloseIntent(pine_id="F-FLATTEN", symbol=symbol,
                               side=close_side, qty=qty, immediately=True),
            run_tag="flat", bar_ts_ms=int(time.time() * 1000),
            retry_seq=0, coid_max_len=30)
        order = asyncio.run(broker.execute_close(envelope))
        print(f"close order placed: id={order.id} {order.side} {order.qty}")
        deadline = time.time() + close_wait_s
        flat = False
        while time.time() < deadline:
            time.sleep(2)
            size_now = _read_position_size(broker, symbol)
            if size_now == 0.0:
                flat = True
                break               # FIRST flat observation -> proceed
        if not flat:
            # Unconfirmed close: NEVER sweep (cancelling protection over a
            # possibly-open position is the naked-open the panel rejected).
            print(f"NOT FLAT after {close_wait_s}s — sweep WITHHELD; "
                  f"check venue.py status NOW")
            return 1
        print("FLAT confirmed")
    else:
        print("FLAT already")

    if owned_ids is None:
        print("attribution UNAVAILABLE — cannot prove which resting orders "
              "are ours; position is flat but the protection sweep could "
              "not run. Inspect venue.py status and the store path.")
        return 2

    try:
        working = asyncio.run(broker.get_open_orders(symbol))
    except Exception as exc:                                  # noqa: BLE001
        print(f"COULD NOT READ open orders: {type(exc).__name__}: {exc}")
        return 2
    ours = [order for order in working if str(order.id) in owned_ids]
    foreign = [order for order in working if str(order.id) not in owned_ids]
    for order in foreign:
        # SHARED account: report, never cancel (hard rule).
        print(f"foreign live order (operator's — NOT touched): "
              f"id={order.id} {order.side} qty={order.qty}")
    if not ours:
        print("sweep: no owned working orders — clean")
        return 0

    unresolved = []
    for order in ours:
        outcome = asyncio.run(broker._cancel_one_disposition(str(order.id)))
        print(f"sweep: cancel {order.id} -> {outcome.value}")
        if outcome not in _RESOLVED:
            unresolved.append(str(order.id))
    if unresolved:
        print(f"sweep INCOMPLETE — disposition UNKNOWN for {unresolved}; "
              f"the venue may still hold them (e.g. a #51 refusal window). "
              f"Cancel via the app or retry later. NOT retrying (#58).")
        return 1
    print(f"sweep complete: {len(ours)} owned order(s) resolved; "
          f"{len(foreign)} foreign order(s) reported untouched")
    return 0
