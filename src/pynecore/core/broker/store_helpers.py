"""
Typed helpers over :class:`~pynecore.core.broker.storage.RunContext`.

The :class:`BrokerStore` exposes :meth:`upsert_order` with a free-form
``extras`` dict and individual setters per column. Each broker plugin
currently re-implements the same micro-patterns on top of that surface:
seed a row with ``{'kind': ..., 'order_type': ...}`` extras, record the
server reference, lift the state machine to ``'confirmed'`` with the
fill price persisted under ``extras['confirm_level']``. The keys are
implicit, the ordering of writes is implicit, and a typo causes silent
divergence between Pine intent and broker truth.

These helpers turn the most common ``execute_entry`` lifecycle steps
into typed function calls. They are intentionally a *thin* layer:
nothing in this module talks to the exchange, retries, sleeps, or makes
state-machine decisions beyond what the orchestrating
:class:`~pynecore.core.broker.journal.DispatchJournal` instructs. The
module is the single place where the canonical ``extras`` schema for
non-bracket entries lives.

Scope is intentionally limited to the M1 proof-of-shape — entry orders
only, no bracket legs, no trail state, no natural-close flags. Those
land in later milestones once a second broker plugin confirms the
shape.

See ``docs/pynecore/plugin-system/broker/broker-plugin-responsibility-review.md``
section §4 for the rationale.
"""
from collections.abc import Iterator
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pynecore.core.broker.storage import OrderRow, RunContext

__all__ = [
    'ENTRY_KIND_POSITION',
    'ENTRY_KIND_WORKING',
    'STATE_SUBMITTED',
    'STATE_SERVER_REF_SEEN',
    'STATE_CONFIRMED',
    'STATE_REJECTED',
    'STATE_DISPOSITION_UNKNOWN',
    'PENDING_DISPATCH_STATES',
    'create_entry_order_row',
    'record_server_ref',
    'mark_confirmed_with_fill',
    'mark_disposition_unknown',
    'mark_rejected',
    'find_pending_dispatch',
]


# === Canonical extras values ===============================================

# ``extras['kind']`` for an entry that opens a position immediately
# (MARKET order). The plugin's reconcile / activity paths key off this
# to know the row maps to ``/positions`` rather than ``/workingorders``.
ENTRY_KIND_POSITION = 'position'

# ``extras['kind']`` for an entry placed as a working order (LIMIT or
# STOP). The reconcile / activity paths key off this to know the row
# maps to ``/workingorders`` rather than ``/positions``.
ENTRY_KIND_WORKING = 'working'


# === Canonical states ======================================================

# Order row persisted, REST POST not yet attempted.
STATE_SUBMITTED = 'submitted'

# REST POST returned a server reference (``dealReference`` for
# Capital.com). Confirm / readback is the next step.
STATE_SERVER_REF_SEEN = 'server_ref_seen'

# Confirm succeeded; ``exchange_order_id`` populated. For MARKET orders
# this also implies a fill, with the fill price under
# ``extras['confirm_level']`` and ``filled_qty`` non-zero.
STATE_CONFIRMED = 'confirmed'

# Exchange rejected the dispatch synchronously (confirm REJECTED, or a
# 4xx response with a known reject reason). Terminal.
STATE_REJECTED = 'rejected'

# Submission outcome unknown — typically a network timeout between the
# POST and the response, or a successful POST without a server
# reference. The recovery path uses these rows on startup to replay or
# reconcile against the exchange.
STATE_DISPOSITION_UNKNOWN = 'disposition_unknown'

# States the recovery path must examine after a restart. ``submitted``
# is here because a crash between :func:`create_entry_order_row` and the
# REST POST leaves the row in this state — same recovery semantics as
# ``disposition_unknown``.
PENDING_DISPATCH_STATES: frozenset[str] = frozenset({
    STATE_SUBMITTED,
    STATE_SERVER_REF_SEEN,
    STATE_DISPOSITION_UNKNOWN,
})


# === Entry order row lifecycle =============================================

def create_entry_order_row(
        store: 'RunContext',
        *,
        coid: str,
        symbol: str,
        side: str,
        qty: float,
        intent_key: str,
        pine_entry_id: str,
        kind: str,
        order_type: str,
) -> None:
    """Insert the initial ``submitted`` row for an entry dispatch.

    Replaces the manual
    ``store.upsert_order(coid, ..., extras={'kind': ..., 'order_type': ...})``
    that every plugin re-implements. The function does **not** write an
    audit event — the orchestrator does that *after* the upsert so a
    crash between the two does not leave an event without its row.

    :param store: The active run context (`plugin.store_ctx`).
    :param coid: The dispatch's canonical client-order-id.
    :param symbol: Exchange-side symbol (epic for Capital.com, etc.).
    :param side: ``'buy'`` or ``'sell'`` — Pine intent's side.
    :param qty: Already quantized to the broker's lot step.
    :param intent_key: The Pine intent's diff key (e.g. ``pine_id``).
    :param pine_entry_id: The Pine ``strategy.entry(id=...)`` value.
    :param kind: One of :data:`ENTRY_KIND_POSITION`,
        :data:`ENTRY_KIND_WORKING`. Decides how downstream recovery and
        activity reconcile interpret the row.
    :param order_type: The ``OrderType`` enum's ``.value`` (``'market'``,
        ``'limit'``, ``'stop'``). Persisted into extras so recovery can
        replay an entry without re-deriving it from the intent.
    """
    if kind not in (ENTRY_KIND_POSITION, ENTRY_KIND_WORKING):
        raise ValueError(
            f"create_entry_order_row: kind must be one of "
            f"{{ENTRY_KIND_POSITION, ENTRY_KIND_WORKING}}, got {kind!r}"
        )
    store.upsert_order(
        coid,
        symbol=symbol,
        side=side,
        qty=qty,
        state=STATE_SUBMITTED,
        intent_key=intent_key,
        pine_entry_id=pine_entry_id,
        extras={'kind': kind, 'order_type': order_type},
    )


def record_server_ref(
        store: 'RunContext',
        *,
        coid: str,
        deal_reference: str,
        kind: str,
        order_type: str,
) -> None:
    """Record the exchange's submission reference + advance state.

    Two writes in a defined order:

    1. ``order_refs`` row keyed by ``'deal_reference'`` so activity /
       reconcile can resolve the reference back to the COID with a
       single indexed lookup.
    2. ``orders.extras`` updated to mirror the reference *together
       with* ``orders.state`` advanced to
       :data:`STATE_SERVER_REF_SEEN` — single
       :meth:`RunContext.upsert_order` transaction, so state + extras
       can never disagree.

    Crash-safety: if the process crashes between (1) and (2), the row
    is still in :data:`STATE_SUBMITTED` (pending), and the resume hook
    sees the ``deal_reference`` via the ``order_refs`` table —
    :func:`~pynecore.core.broker.journal._collect_refs_for` materialises
    it from there. That is the contract :func:`find_pending_dispatch`
    relies on.

    :param store: The active run context.
    :param coid: The dispatch's COID.
    :param deal_reference: Server-allocated reference for the POST.
    :param kind: Same value originally passed to
        :func:`create_entry_order_row`. Repeated here because
        ``upsert_order(extras=...)`` overwrites the whole dict.
    :param order_type: Same value originally passed to
        :func:`create_entry_order_row`. Repeated for the same reason.
    """
    store.add_ref(coid, 'deal_reference', deal_reference)
    store.upsert_order(
        coid,
        state=STATE_SERVER_REF_SEEN,
        extras={
            'kind': kind,
            'order_type': order_type,
            'deal_reference': deal_reference,
        },
    )


def mark_confirmed_with_fill(
        store: 'RunContext',
        *,
        coid: str,
        exchange_id: str | None,
        is_filled: bool,
        filled_qty: float,
        fill_price: float | None,
) -> None:
    """Promote the row to ``confirmed``, optionally recording a fill.

    A MARKET entry that fills immediately needs four facts persisted in
    a single logical step:

    - ``order_refs`` row keyed by ``'deal_id'`` (so activity tail rows
      that carry only the deal id can map back).
    - ``orders.exchange_order_id`` populated.
    - ``orders.state`` = :data:`STATE_CONFIRMED`.
    - ``orders.filled_qty`` set and ``extras['confirm_level']`` set
      from the confirm response, as a recovery fallback when the
      ``/history/activity`` row arrives with ``level=0`` and the
      ``/positions`` snapshot is also empty for this deal id.

    LIMIT / STOP entries call this with ``is_filled=False``; only the
    first three facts apply.

    :param store: The active run context.
    :param coid: The dispatch's COID.
    :param exchange_id: Exchange-allocated id (``dealId`` for
        Capital.com). May be ``None`` if confirm returns no id — the
        function still advances state but skips the id-related writes.
    :param is_filled: ``True`` only for MARKET-side fills that confirm
        as OPEN. LIMIT / STOP submissions land as live working orders
        and pass ``False`` here; the fill arrives later via the
        activity stream.
    :param filled_qty: Confirmed fill quantity. Ignored when
        ``is_filled`` is ``False``.
    :param fill_price: Confirm-side fill price. Persisted under
        ``extras['confirm_level']`` only when ``is_filled`` is ``True``
        and the value is strictly positive — a zero/negative level is
        a no-quote artefact and would corrupt the recovery fallback.

    Crash-safety: ``exchange_order_id``, ``state``, ``filled_qty``, and
    the merged ``extras`` (with ``confirm_level``) are written in a
    *single* :meth:`RunContext.upsert_order` transaction, so a crash
    cannot leave the row in :data:`STATE_CONFIRMED` (terminal) without
    its fill details. The ``deal_id`` ref is written first as a separate
    transaction — if a crash occurs between the ref write and the
    consolidated update, the row's state is still
    :data:`STATE_SERVER_REF_SEEN` (pending), so
    :func:`find_pending_dispatch` re-yields it and the resume hook can
    rebuild the confirmation from the ``deal_id`` already in
    ``order_refs``.
    """
    if exchange_id:
        store.add_ref(coid, 'deal_id', exchange_id)

    fields: dict[str, Any] = {'state': STATE_CONFIRMED}
    if exchange_id:
        fields['exchange_order_id'] = exchange_id
    if is_filled:
        fields['filled_qty'] = filled_qty
        if fill_price is not None and fill_price > 0.0:
            existing = store.get_order(coid)
            merged = dict(existing.extras or {}) if existing is not None else {}
            merged['confirm_level'] = fill_price
            fields['extras'] = merged
    store.upsert_order(coid, **fields)


def mark_disposition_unknown(
        store: 'RunContext',
        *,
        coid: str,
) -> None:
    """Flip the row to :data:`STATE_DISPOSITION_UNKNOWN`.

    Used when a POST times out or the response is missing the server
    reference. The row is *not* deleted — recovery on the next restart
    re-evaluates it against the exchange's authoritative view.
    """
    store.set_order_state(coid, STATE_DISPOSITION_UNKNOWN)


def mark_rejected(
        store: 'RunContext',
        *,
        coid: str,
) -> None:
    """Flip the row to :data:`STATE_REJECTED` (terminal).

    Used when the exchange returns a definitive reject — confirm
    ``REJECTED``, a 4xx response, or a synchronous reason string the
    plugin maps to ``ExchangeOrderRejectedError``. The row stays in
    place for audit; the engine clears its intent slot on the next
    sync.
    """
    store.set_order_state(coid, STATE_REJECTED)


# === Recovery query ========================================================

def find_pending_dispatch(store: 'RunContext') -> Iterator['OrderRow']:
    """Yield live rows whose dispatch the journal still owns.

    A row is *pending* when its state is in
    :data:`PENDING_DISPATCH_STATES`. The journal's
    :meth:`~pynecore.core.broker.journal.DispatchJournal.recover_pending`
    iterates these on startup and asks the plugin's
    ``resume_pending_dispatch`` hook for a verdict.

    The query relies on the partial ``idx_orders_live`` index, so the
    cost is O(log n) per call.
    """
    for row in store.iter_live_orders():
        if row.state in PENDING_DISPATCH_STATES:
            yield row
