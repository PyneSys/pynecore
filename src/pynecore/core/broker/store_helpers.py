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
from collections.abc import Iterator, Mapping
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pynecore.core.broker.storage import OrderRow, RunContext

__all__ = [
    'ENTRY_KIND_POSITION',
    'ENTRY_KIND_WORKING',
    'KIND_FULL_CLOSE',
    'KIND_PARTIAL_CLOSE',
    'KIND_CANCEL',
    'KIND_MODIFY_ENTRY',
    'KIND_MODIFY_EXIT',
    'CLOSE_KINDS',
    'STATE_SUBMITTED',
    'STATE_SERVER_REF_SEEN',
    'STATE_CONFIRMED',
    'STATE_REJECTED',
    'STATE_DISPOSITION_UNKNOWN',
    'STATE_CLOSING',
    'STATE_CANCEL_PENDING',
    'PENDING_DISPATCH_STATES',
    'create_entry_order_row',
    'record_server_ref',
    'mark_confirmed_with_fill',
    'mark_disposition_unknown',
    'mark_rejected',
    'find_pending_dispatch',
    'create_close_target_row',
    'record_close_server_ref',
    'mark_closing',
    'mark_close_completed',
    'create_cancel_command_row',
    'mark_cancel_completed',
    'create_modify_entry_row',
    'create_modify_exit_row',
    'mark_modify_completed',
    'mark_reconcile_filled',
    'mark_reconcile_terminal_close',
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

# ``extras['kind']`` for a close that targets every unit of one or more
# live positions. The plugin issues a DELETE per target ``dealId``; no
# server reference is allocated (DELETE is fire-and-forget). Recovery
# reconciles the persisted ``extras['targets']`` against the live
# positions snapshot.
KIND_FULL_CLOSE = 'full_close'

# ``extras['kind']`` for a close that reduces an existing position by
# a strict subset of its units. Emulated via an opposite-direction
# POST to ``/positions`` (Capital.com has no native partial-close
# endpoint), so the row owns a server reference like an entry would.
# Recovery uses the ``deal_reference`` confirm GET plus a unit-count
# delta against the pre/post position snapshot.
KIND_PARTIAL_CLOSE = 'partial_close_emulated'

# Convenience set used by close-aware recovery branches that route by
# ``extras['kind']`` and need to handle both full and partial close
# rows identically.
CLOSE_KINDS: frozenset[str] = frozenset({KIND_FULL_CLOSE, KIND_PARTIAL_CLOSE})

# ``extras['kind']`` for a cancel command row. Carries the per-dispatch
# audit trail for a ``CancelIntent``; the row does not itself land on
# the exchange. The targets it sweeps are referenced via
# ``extras['target_coids']`` so a mid-loop crash can re-evaluate which
# targets still need cancelling on recovery.
KIND_CANCEL = 'cancel'

# ``extras['kind']`` for a working-order amend (PUT level). The row
# carries the audit trail for the amend dispatch and stores the new
# level under ``extras['new_level']`` so recovery can verify whether
# the broker landed the change.
KIND_MODIFY_ENTRY = 'modify_entry'

# ``extras['kind']`` for a position bracket amend (PUT TP / SL /
# trailing). The row is the entry-side audit trail; the synthetic
# bracket leg rows that mirror the resulting TP / SL state remain
# under the plugin's leg state machine for the duration of M4.
KIND_MODIFY_EXIT = 'modify_exit'


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

# A full-close DELETE landed at the exchange but the final fill /
# settlement has not yet been observed in the activity stream. The
# recovery path treats this state like a pending dispatch: the targets
# are re-queried against the live positions snapshot to confirm they
# have actually vanished.
STATE_CLOSING = 'closing'

# A cancel command row is mid-flight: the per-target loop has started
# but not finished. Recovery re-evaluates the targets against the
# snapshot and retries the DELETEs that did not land.
STATE_CANCEL_PENDING = 'cancel_pending'

# States the recovery path must examine after a restart. ``submitted``
# is here because a crash between the initial helper and the REST call
# leaves the row in this state — same recovery semantics as
# ``disposition_unknown``. The non-entry pending states (``closing``,
# ``cancel_pending``) are added to this set by the corresponding M4
# phases once the matching ``resume_pending_dispatch`` branches exist
# on the plugin side; until then the journal does not surface them to
# the recovery loop.
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


# === Close lifecycle =======================================================

def create_close_target_row(
        store: 'RunContext',
        *,
        coid: str,
        symbol: str,
        side: str,
        qty: float,
        intent_key: str,
        kind: str,
        pine_entry_id: str | None = None,
        extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """Insert the initial ``submitted`` row for a close dispatch.

    Both full-close and partial-close dispatches start with the same
    persist-first row; the ``kind`` discriminator (:data:`KIND_FULL_CLOSE`
    vs :data:`KIND_PARTIAL_CLOSE`) decides the downstream lifecycle and
    the recovery contract.

    :param store: The active run context.
    :param coid: The dispatch's COID
        (``envelope.client_order_id(KIND_CLOSE)``).
    :param symbol: Exchange-side symbol (epic).
    :param side: Side of the closing leg — for a full close this is
        the side of the position being closed; for a partial close
        this is the opposite-direction emulation side.
    :param qty: Quantity to close, already quantized.
    :param intent_key: The Pine intent's diff key.
    :param kind: :data:`KIND_FULL_CLOSE` or :data:`KIND_PARTIAL_CLOSE`.
    :param pine_entry_id: Optional Pine ``strategy.entry(id=...)`` value
        when the close is tied to a single named entry. ``None`` when
        the close sweeps multiple positions (e.g. full close across
        ids).
    :param extra_payload: Optional extras to merge in alongside the
        canonical ``kind`` key. Use for plugin-specific bookkeeping
        the recovery contract needs (e.g. ``pre_total_units``,
        ``intent_units`` for the partial-close emulation).
    """
    if kind not in CLOSE_KINDS:
        raise ValueError(
            f"create_close_target_row: kind must be one of {{KIND_FULL_CLOSE, "
            f"KIND_PARTIAL_CLOSE}}, got {kind!r}"
        )
    extras: dict[str, Any] = {'kind': kind}
    if extra_payload:
        extras.update(extra_payload)
    store.upsert_order(
        coid,
        symbol=symbol,
        side=side,
        qty=qty,
        state=STATE_SUBMITTED,
        intent_key=intent_key,
        pine_entry_id=pine_entry_id,
        extras=extras,
    )


def record_close_server_ref(
        store: 'RunContext',
        *,
        coid: str,
        deal_reference: str,
        kind: str,
        extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """Record the partial-close POST's server reference + advance state.

    Only the partial-close path issues a POST that allocates a
    ``dealReference``; the full-close DELETE returns no such id, so
    this helper is not called there. The two-write order matches
    :func:`record_server_ref`: ``order_refs`` first, then a consolidated
    ``upsert_order`` that mirrors the reference into ``extras`` and
    advances state to :data:`STATE_SERVER_REF_SEEN`.

    :param store: The active run context.
    :param coid: The dispatch's COID.
    :param deal_reference: Server-allocated reference for the POST.
    :param kind: Must be :data:`KIND_PARTIAL_CLOSE`. Re-stated because
        ``upsert_order(extras=...)`` overwrites the whole dict.
    :param extra_payload: Optional extras to preserve through the
        state advance (e.g. ``pre_total_units``, ``intent_units``).
    """
    if kind != KIND_PARTIAL_CLOSE:
        raise ValueError(
            f"record_close_server_ref: kind must be {KIND_PARTIAL_CLOSE!r}, "
            f"got {kind!r}"
        )
    store.add_ref(coid, 'deal_reference', deal_reference)
    # ``upsert_order(extras=...)`` overwrites the whole dict, so read the
    # current extras first and merge: plugin-side context written between
    # row creation and this helper (e.g. ``pre_total_units`` /
    # ``intent_units`` from the partial-close emulation's pre-snapshot)
    # must survive the state advance so recovery has them.
    existing = store.get_order(coid)
    extras: dict[str, Any] = dict(existing.extras or {}) if existing else {}
    extras['kind'] = kind
    extras['deal_reference'] = deal_reference
    if extra_payload:
        extras.update(extra_payload)
    store.upsert_order(
        coid,
        state=STATE_SERVER_REF_SEEN,
        extras=extras,
    )


def mark_closing(
        store: 'RunContext',
        *,
        coid: str,
        kind: str,
        targets: list[str],
        extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """Flip a full-close row to :data:`STATE_CLOSING` and pin its targets.

    Called after every DELETE has been issued but before the activity
    stream has confirmed the resulting fills. The ``targets`` list is
    persisted under ``extras['targets']`` so a restart-mid-stream can
    reconcile each ``dealId`` against the live positions snapshot.

    :param store: The active run context.
    :param coid: The dispatch's COID.
    :param kind: Must be :data:`KIND_FULL_CLOSE`. Partial-close uses
        :func:`mark_close_completed` directly because its emulation
        finishes synchronously inside the POST.
    :param targets: Exchange ``dealId`` strings the DELETE chain
        targeted. Empty list is a no-op close (the engine should have
        elided the dispatch, but this is harmless).
    :param extra_payload: Optional extras to preserve.
    """
    if kind != KIND_FULL_CLOSE:
        raise ValueError(
            f"mark_closing: kind must be {KIND_FULL_CLOSE!r}, got {kind!r}"
        )
    extras: dict[str, Any] = {
        'kind': kind,
        'targets': list(targets),
    }
    if extra_payload:
        extras.update(extra_payload)
    store.upsert_order(
        coid,
        state=STATE_CLOSING,
        extras=extras,
    )


def mark_close_completed(
        store: 'RunContext',
        *,
        coid: str,
        kind: str,
        extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """Finalise a close dispatch row's state to ``confirmed``.

    For full close, called once every target's vanished position has
    been observed (or the recovery path has decided the targets are
    gone). For partial close, called when the POST + post-snapshot
    reconcile confirm the unit-count delta matches.

    Does **not** call :meth:`RunContext.close_order` — the journal
    issues that as a separate step *after* it has emitted the
    ``confirmed`` audit event, so the event order is
    ``dispatch_submitted → deal_reference_seen → confirmed →
    order_closed``.

    :param store: The active run context.
    :param coid: The dispatch's COID.
    :param kind: :data:`KIND_FULL_CLOSE` or :data:`KIND_PARTIAL_CLOSE`.
    :param extra_payload: Optional extras to preserve.
    """
    if kind not in CLOSE_KINDS:
        raise ValueError(
            f"mark_close_completed: kind must be one of {{KIND_FULL_CLOSE, "
            f"KIND_PARTIAL_CLOSE}}, got {kind!r}"
        )
    fields: dict[str, Any] = {'state': STATE_CONFIRMED}
    if extra_payload:
        existing = store.get_order(coid)
        merged = dict(existing.extras or {}) if existing is not None else {}
        merged.update(extra_payload)
        fields['extras'] = merged
    store.upsert_order(coid, **fields)


# === Cancel lifecycle ======================================================

def create_cancel_command_row(
        store: 'RunContext',
        *,
        coid: str,
        symbol: str,
        side: str,
        qty: float,
        intent_key: str,
        pine_entry_id: str | None = None,
        from_entry: str | None = None,
        target_coids: list[str] | None = None,
        extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """Insert the initial ``submitted`` row for a cancel dispatch.

    A cancel row is the per-dispatch audit trail for a ``CancelIntent``:
    it carries the list of target COIDs the per-target loop will sweep,
    so a mid-loop crash can resume cleanly. The row itself does not
    land at the exchange — only its targets do (via DELETE / PUT-null).

    :param store: The active run context.
    :param coid: The cancel dispatch's COID
        (``envelope.client_order_id(KIND_CANCEL)``).
    :param symbol: Exchange-side symbol of the targets.
    :param side: Side of the targets (uniform across a single dispatch).
    :param qty: Aggregate quantity of the targets — for diagnostics
        only; the recovery contract uses ``target_coids`` directly.
    :param intent_key: The Pine intent's diff key.
    :param pine_entry_id: ``strategy.entry(id=...)`` value the cancel
        addresses, or ``None`` for cross-entry cancels.
    :param from_entry: ``ExitIntent`` ``from_entry`` echo when the
        cancel sweeps a bracket leg, ``None`` otherwise.
    :param target_coids: COIDs the per-target loop intends to sweep.
        Persisted under ``extras['target_coids']`` so recovery can
        re-evaluate after a mid-loop crash. Empty list is permitted —
        the journal will still emit the audit trail and finalise the
        row with :func:`mark_cancel_completed`.
    :param extra_payload: Optional extras to merge in.
    """
    extras: dict[str, Any] = {
        'kind': KIND_CANCEL,
        'target_coids': list(target_coids or ()),
    }
    if extra_payload:
        extras.update(extra_payload)
    store.upsert_order(
        coid,
        symbol=symbol,
        side=side,
        qty=qty,
        state=STATE_SUBMITTED,
        intent_key=intent_key,
        pine_entry_id=pine_entry_id,
        from_entry=from_entry,
        extras=extras,
    )


def mark_cancel_completed(
        store: 'RunContext',
        *,
        coid: str,
        reason_path: str,
        extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """Finalise a cancel command row's state to ``confirmed``.

    Called after the per-target loop has finished. ``reason_path``
    records why the cancel resolved this way (``'deleted'`` when the
    targets were swept, ``'already_gone'`` when every target had
    vanished benignly, ``'noop'`` when nothing matched). The value is
    preserved in ``extras['reason_path']`` for forensics. Does **not**
    call :meth:`RunContext.close_order` — see
    :func:`mark_close_completed` for the rationale.

    :param store: The active run context.
    :param coid: The cancel dispatch's COID.
    :param reason_path: One of ``'deleted'``, ``'already_gone'``,
        ``'noop'`` (the canonical :class:`CancelOutcome.reason_path`
        values).
    :param extra_payload: Optional extras to preserve.
    """
    existing = store.get_order(coid)
    merged: dict[str, Any] = dict(existing.extras or {}) if existing is not None else {}
    merged['reason_path'] = reason_path
    if extra_payload:
        merged.update(extra_payload)
    store.upsert_order(
        coid,
        state=STATE_CONFIRMED,
        extras=merged,
    )


# === Modify lifecycle ======================================================

def create_modify_entry_row(
        store: 'RunContext',
        *,
        coid: str,
        target_coid: str,
        symbol: str,
        side: str,
        qty: float,
        intent_key: str,
        new_level: float,
        pine_entry_id: str | None = None,
        extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """Insert the initial ``submitted`` row for a working-order amend.

    The amend dispatch produces its own COID + audit row separate from
    the working order it mutates. ``target_coid`` and ``new_level``
    travel through ``extras`` so a restart-mid-amend can verify the
    broker landed the change.

    :param store: The active run context.
    :param coid: The amend dispatch's COID
        (``envelope.client_order_id(KIND_MODIFY)`` for the new envelope).
    :param target_coid: COID of the working order being amended.
    :param symbol: Exchange-side symbol.
    :param side: Side of the target order.
    :param qty: Order quantity (unchanged by the amend).
    :param intent_key: The Pine intent's diff key.
    :param new_level: Requested new ``level`` for the working order.
    :param pine_entry_id: ``strategy.entry(id=...)`` echo.
    :param extra_payload: Optional extras to merge in.
    """
    extras: dict[str, Any] = {
        'kind': KIND_MODIFY_ENTRY,
        'target_coid': target_coid,
        'new_level': new_level,
    }
    if extra_payload:
        extras.update(extra_payload)
    store.upsert_order(
        coid,
        symbol=symbol,
        side=side,
        qty=qty,
        state=STATE_SUBMITTED,
        intent_key=intent_key,
        pine_entry_id=pine_entry_id,
        extras=extras,
    )


def create_modify_exit_row(
        store: 'RunContext',
        *,
        coid: str,
        target_coid: str,
        symbol: str,
        side: str,
        qty: float,
        intent_key: str,
        new_tp: float | None,
        new_sl: float | None,
        new_trail: float | None,
        new_trail_price: float | None = None,
        pine_entry_id: str | None = None,
        from_entry: str | None = None,
        extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """Insert the initial ``submitted`` row for a position bracket amend.

    Mirrors :func:`create_modify_entry_row` but for ``ExitIntent``
    amends: the target is the entry-row representing the position and
    the requested change covers TP / SL / trailing levels. Recovery
    reads ``new_tp`` / ``new_sl`` / ``new_trail`` / ``new_trail_price``
    from ``extras`` and compares them against the post-amend snapshot.

    ``new_trail_price`` is Pine's trailing-stop *activation* price
    (``strategy.exit(trail_price=...)``). When both ``new_trail`` and
    ``new_trail_price`` are set the bracket is **pending trailing** —
    the broker carries no native trailing stop yet (the local
    activation monitor will PUT one once price crosses the threshold),
    so the post-amend snapshot legitimately shows ``trailingStop=False``
    and recovery must NOT require ``trailingStop=True`` to declare the
    amend landed. Persisting the activation price is what lets the
    snapshot verdict distinguish the two trailing shapes.

    Synthetic bracket leg rows (``leg_kind`` ``'tp'`` / ``'sl'``)
    remain under the plugin's leg state machine for M4; this helper
    persists only the entry-side audit trail.
    """
    extras: dict[str, Any] = {
        'kind': KIND_MODIFY_EXIT,
        'target_coid': target_coid,
        'new_tp': new_tp,
        'new_sl': new_sl,
        'new_trail': new_trail,
        'new_trail_price': new_trail_price,
    }
    if extra_payload:
        extras.update(extra_payload)
    store.upsert_order(
        coid,
        symbol=symbol,
        side=side,
        qty=qty,
        state=STATE_SUBMITTED,
        intent_key=intent_key,
        pine_entry_id=pine_entry_id,
        from_entry=from_entry,
        extras=extras,
    )


def mark_modify_completed(
        store: 'RunContext',
        *,
        coid: str,
        extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """Finalise a modify dispatch row's state to ``confirmed``.

    Both :data:`KIND_MODIFY_ENTRY` and :data:`KIND_MODIFY_EXIT` use
    this terminal helper. The target order it amended is untouched;
    only the modify command row transitions. Does **not** call
    :meth:`RunContext.close_order` — see :func:`mark_close_completed`
    for the rationale.

    :param store: The active run context.
    :param coid: The modify dispatch's COID.
    :param extra_payload: Optional extras to preserve (e.g. the
        amended values echoed back from the confirm response).
    """
    fields: dict[str, Any] = {'state': STATE_CONFIRMED}
    if extra_payload:
        existing = store.get_order(coid)
        merged = dict(existing.extras or {}) if existing is not None else {}
        merged.update(extra_payload)
        fields['extras'] = merged
    store.upsert_order(coid, **fields)


# === Reconcile-path terminal helpers =======================================

def mark_reconcile_filled(
        store: 'RunContext',
        *,
        coid: str,
        filled_qty: float,
        new_state: str,
        extras_patch: Mapping[str, Any] | None,
) -> None:
    """Promote a row to ``confirmed`` from a reconcile-path observation.

    Used by :meth:`~pynecore.core.broker.journal.DispatchJournal.apply_reconcile_outcome`
    for the working→position case: a row sitting in
    :data:`STATE_SERVER_REF_SEEN` is observed in the live positions
    snapshot, so the journal records the fill and flips
    ``extras['kind']`` from :data:`ENTRY_KIND_WORKING` to
    :data:`ENTRY_KIND_POSITION` in a single transaction.

    :param store: The active run context.
    :param coid: The row's client-order-id.
    :param filled_qty: Confirmed fill quantity from the snapshot.
    :param new_state: The state to land in (always :data:`STATE_CONFIRMED`
        for the current call sites; passed through for clarity).
    :param extras_patch: Plugin-supplied extras to merge into the row
        (e.g. ``{'kind': 'position', 'entry_filled_at': now_ts}``).
        Merged on top of the existing ``extras`` — the journal does not
        validate the keys.
    """
    fields: dict[str, Any] = {
        'state': new_state,
        'filled_qty': filled_qty,
    }
    if extras_patch:
        existing = store.get_order(coid)
        merged = dict(existing.extras or {}) if existing is not None else {}
        merged.update(extras_patch)
        fields['extras'] = merged
    store.upsert_order(coid, **fields)


def mark_reconcile_terminal_close(
        store: 'RunContext',
        *,
        coid: str,
        new_state: str,
        extras_patch: Mapping[str, Any] | None,
        close_row: bool,
) -> None:
    """Terminate a row from a reconcile-path observation.

    Used by :meth:`~pynecore.core.broker.journal.DispatchJournal.apply_reconcile_outcome`
    for every terminal close that originates in the reconciler:
    bracket sibling retire on mixed-bracket rejection, pending-trail
    sibling parent-rejection cascade, missing-pending grace expiry,
    unexpected-cancel cascade, and eager-teardown follow-up after a
    natural close.

    The state mutation, the optional extras merge, and the
    :meth:`RunContext.close_order` (when ``close_row`` is ``True``)
    happen in this single helper; the journal then writes a separate
    audit event. The split mirrors :func:`mark_confirmed_with_fill`
    style — terminal facts persisted first, audit logging by the
    caller.

    :param store: The active run context.
    :param coid: The row's client-order-id.
    :param new_state: Terminal state to land in — typically
        :data:`STATE_REJECTED` for bracket retire / cascade paths.
        ``close_row=True`` is what drops the row from the live set.
    :param extras_patch: Plugin-supplied extras to merge before the
        terminal transition (rarely used on this path; usually
        ``None``).
    :param close_row: When ``True``, also calls
        :meth:`RunContext.close_order` after the state update — this
        is the steady-state default for reconcile terminal closures.
    """
    fields: dict[str, Any] = {'state': new_state}
    if extras_patch:
        existing = store.get_order(coid)
        merged = dict(existing.extras or {}) if existing is not None else {}
        merged.update(extras_patch)
        fields['extras'] = merged
    store.upsert_order(coid, **fields)
    if close_row:
        store.close_order(coid)
