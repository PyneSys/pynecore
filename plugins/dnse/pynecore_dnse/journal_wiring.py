"""Persist-first journal wiring over the engine's BrokerStore (#36, Phase A2).

Every function no-ops when ``store_ctx is None`` (backtests, unit tests,
degraded runs keep today's behavior byte-exact — the panel's G2). The row is
the RESTART BRIDGE: DNSE sends no client-order-id to the venue
(``idempotency=SOFTWARE``), so only these rows link a venue order back to a
run after a crash/TERM (Live-L1-T16, measured).

Design facts this module encodes (card #36 panel, leader-verified):

- **Persist-FIRST**: the ``submitted`` row is written BEFORE the POST, so a
  crash in any window leaves an auditable row (`find_pending_dispatch`).
- The child of a triggered conditional is a SECOND ref on the PARENT's row —
  via ``add_ref`` with its own ref type, never ``record_server_ref`` (which
  would demote the row's state and collide in the journal's ref dict).
- ``close_order`` deletes refs by design — adoption reads LIVE rows only.
- #67's transport ``phase`` ('sent'/'connect') never escapes
  ``errors.classify``; callers pass it from the sentinel BODY.
"""
from dataclasses import dataclass

from pynecore.core.broker.store_helpers import (
    STATE_REJECTED, STATE_SERVER_REF_SEEN, STATE_SUBMITTED,
    mark_disposition_unknown, mark_rejected,
)

#: Ref types on one row. The placed (tracked) venue id is the primary ref;
#: a conditional's NORMAL-book child and an OCO's umbrella are secondary.
REF_EXCHANGE_ORDER_ID = "exchange_order_id"
REF_CHILD_ORDER_ID = "child_order_id"
REF_UMBRELLA_ORDER_ID = "umbrella_order_id"


def _merged_extras(store_ctx, coid: str, **updates) -> dict:
    """``upsert_order(extras=...)`` overwrites the WHOLE dict — merge first."""
    row = store_ctx.get_order(coid)
    extras = dict(row.extras or {}) if row is not None else {}
    extras.update(updates)
    return extras


def journal_submitted(store_ctx, *, coid, symbol, side, qty, intent_key,
                      pine_id, from_entry, leg_kind, category, order_type,
                      price=None) -> None:
    """The persist-FIRST row: written BEFORE the POST leaves the process.

    ``price`` is EVIDENCE for a future recovery matcher (#71: today's ladder
    deliberately has none to act on) — persisted now, consumed by nothing."""
    if store_ctx is None or not coid:
        return
    row = store_ctx.get_order(coid)
    if row is not None and row.closed_ts_ms is not None:
        store_ctx.reopen_order(coid)
    store_ctx.upsert_order(
        coid, symbol=symbol, side=side, qty=float(qty),
        state=STATE_SUBMITTED, intent_key=intent_key or "",
        pine_entry_id=pine_id or "", from_entry=from_entry,
        extras=_merged_extras(store_ctx, coid,
                              dnse_category=category, order_type=order_type,
                              leg_kind=leg_kind or "",
                              submitted_price=price))


def journal_server_ref(store_ctx, *, coid, venue_id, category,
                       umbrella_id=None) -> None:
    """The venue answered: record the tracked id as the primary ref."""
    if store_ctx is None or not coid:
        return
    store_ctx.add_ref(coid, REF_EXCHANGE_ORDER_ID, str(venue_id))
    if umbrella_id is not None and str(umbrella_id) != str(venue_id):
        store_ctx.add_ref(coid, REF_UMBRELLA_ORDER_ID, str(umbrella_id))
    store_ctx.upsert_order(
        coid, state=STATE_SERVER_REF_SEEN, exchange_order_id=str(venue_id),
        extras=_merged_extras(store_ctx, coid, dnse_category=category))


def journal_rejected(store_ctx, *, coid) -> None:
    if store_ctx is None or not coid:
        return
    mark_rejected(store_ctx, coid=coid)


def journal_disposition_unknown(store_ctx, *, coid, phase=None,
                                transport=None) -> None:
    """#67 join: the lost-reply row. ``phase`` comes from the sentinel BODY
    (it never escapes ``errors.classify`` — panel P1)."""
    if store_ctx is None or not coid:
        return
    store_ctx.upsert_order(
        coid, extras=_merged_extras(store_ctx, coid,
                                    transport_phase=phase or "unknown",
                                    transport_kind=transport or ""))
    mark_disposition_unknown(store_ctx, coid=coid)


def _coid_for_venue_id(store_ctx, venue_id: str):
    """Reverse lookup: which live row owns this venue id (any ref type)."""
    wanted = str(venue_id)
    for row in store_ctx.iter_live_orders():
        if str(row.exchange_order_id or "") == wanted:
            return row.client_order_id
        for _ref_type, ref_value in store_ctx.iter_refs_for_coid(row.client_order_id):
            if str(ref_value) == wanted:
                return row.client_order_id
    return None


def journal_child_ref(store_ctx, *, parent_venue_id, child_id) -> None:
    """A triggered conditional named its NORMAL-book child (#41/#42-A):
    SECOND ref on the parent's row, state PRESERVED (panel G4 correction)."""
    if store_ctx is None:
        return
    coid = _coid_for_venue_id(store_ctx, parent_venue_id)
    if coid is None:
        return                       # foreign / already closed: nothing to root
    store_ctx.add_ref(coid, REF_CHILD_ORDER_ID, str(child_id))
    store_ctx.upsert_order(
        coid, extras=_merged_extras(store_ctx, coid, child_order_id=str(child_id)))


def journal_fill_progress(store_ctx, *, venue_id, filled_qty, raw_status) -> None:
    """#56/item 5: persist the fill watermark on a LIVE partial — a restart
    seeds ``_last_seen`` from ``row.filled_qty`` + ``last_raw_status`` and
    re-emits nothing (the in-memory cursor IS ``_last_seen``; this is its
    only durable copy)."""
    if store_ctx is None:
        return
    coid = _coid_for_venue_id(store_ctx, venue_id)
    if coid is None:
        return
    store_ctx.set_filled(coid, float(filled_qty))
    store_ctx.upsert_order(
        coid, extras=_merged_extras(store_ctx, coid,
                                    last_raw_status=str(raw_status),
                                    last_fill_venue_id=str(venue_id)))


def journal_terminal(store_ctx, *, venue_id, terminal_status,
                     filled_qty=None) -> None:
    """A terminal observation (fill / cancel / reject) closes the row.
    Idempotent: an already-closed or unknown id is a no-op — the watch scan
    and the cancel core can both observe the same terminal."""
    if store_ctx is None:
        return
    coid = _coid_for_venue_id(store_ctx, venue_id)
    if coid is None:
        return
    if filled_qty:
        store_ctx.set_filled(coid, float(filled_qty))
    store_ctx.upsert_order(
        coid, extras=_merged_extras(store_ctx, coid,
                                    terminal_status=str(terminal_status)))
    store_ctx.close_order(coid)


@dataclass
class JournalIdentity:
    """One live journalled order, materialised for connect-time adoption."""
    coid: str
    intent_key: str
    pine_id: str
    from_entry: "str | None"
    leg_kind: str
    category: str
    venue_ids: "list[str]"           # primary + child + umbrella refs
    child_id: "str | None"
    state: str
    filled_qty: float                # #56: the persisted fill watermark
    last_raw_status: "str | None"
    last_fill_venue_id: "str | None"


def iter_journal_identities(store_ctx):
    """Live rows -> adoption material. REJECTED rows are skipped (no venue
    order exists); DISPOSITION_UNKNOWN rows ARE yielded — the order may
    exist, and the chase/adoption is how it is re-owned."""
    if store_ctx is None:
        return
    for row in store_ctx.iter_live_orders():
        if row.state == STATE_REJECTED:
            continue
        extras = row.extras or {}
        venue_ids: list[str] = []
        if row.exchange_order_id:
            venue_ids.append(str(row.exchange_order_id))
        child_id = None
        for ref_type, ref_value in store_ctx.iter_refs_for_coid(row.client_order_id):
            value = str(ref_value)
            if value not in venue_ids:
                venue_ids.append(value)
            if ref_type == REF_CHILD_ORDER_ID:
                child_id = value
        yield JournalIdentity(
            coid=row.client_order_id,
            intent_key=row.intent_key or "",
            pine_id=row.pine_entry_id or "",
            from_entry=row.from_entry,
            leg_kind=str(extras.get("leg_kind") or ""),
            category=str(extras.get("dnse_category") or "NORMAL"),
            venue_ids=venue_ids,
            child_id=child_id,
            state=row.state,
            filled_qty=float(row.filled_qty or 0.0),
            last_raw_status=extras.get("last_raw_status"),
            last_fill_venue_id=extras.get("last_fill_venue_id"),
        )
