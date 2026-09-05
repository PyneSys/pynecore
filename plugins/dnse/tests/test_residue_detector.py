"""Phase B3 repro — the external-cancel RESIDUE detector (#59 item-3, demoted).

The #59 item-3 round established (comment 5420854335, leader-verified): DNSE
reports an external cancel as a STATUS TRANSITION on a row the day book
RETAINS — that channel already works end-to-end live (T5-T9 evidence), so the
DisappearanceTracker's presence-diffing premise is dead. What remains is the
narrow RESIDUE: an id that vanished from every readable book WITHOUT a
terminal transition ever observed (restart gaps aside, mostly our own read
artifacts — page truncation, previous-day rollover, failed reads).

The contract under test (Path 2 of that review):

- a residue may conclude CANCELLED **only** from a positive
  ``/orders/history`` row read without transport error (paginated to
  exhaustion — the #69 gap rides here);
- absence anywhere (empty history, unreadable book, blip) is INCONCLUSIVE
  forever — never a verdict;
- a #41 Activated shell is never a residue subject (its child does the work);
- the normal terminal-transition channel stays the primary — no duplicates.

R1 is RED on the unmodified tree (a vanished id is silent forever —
``watch_orders`` is a change-detector over PRESENT rows and nothing walks the
complement). R2-R5 pin the abstention/exclusion rules the implementation must
never break.
"""
import asyncio
import logging

import pytest
import pynecore.lib as lib

lib.bar_index = 0  # let the [BROKER] log formatter render during broker._emit()

from pynecore_dnse import broker
from pynecore.core.broker.storage import BrokerStore
from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.models import DispatchEnvelope, EntryIntent, OrderType

_SECDEF_ROW = [{"ceilingPrice": "1550", "floorPrice": "1450", "securityGroupId": "FU"}]
_LOAN_OK = (200, {"loanPackages": [{"id": 42}]})
_PLACED = (201, {"id": "437346", "symbol": "VN30F1M", "side": "NB",
                 "quantity": 1, "orderStatus": "New"})
_EMPTY_BOOK = (200, {"orders": [], "totalPages": 1})
_EMPTY_HISTORY = (200, {"data": [], "total": 0})
_HISTORY_CANCELLED = (200, {"data": [{"id": "20260904_437346",
                                      "symbol": "VN30F1M", "side": "NB",
                                      "quantity": 1,
                                      "orderStatus": "Canceled"}],
                            "total": 1})


def _broker(fake_client, tmp_path, **client_responses):
    responses = {"get_security_definition": (200, _SECDEF_ROW),
                 "get_loan_packages": _LOAN_OK}
    responses.update(client_responses)
    config = broker.DNSEBrokerConfig(
        api_key="k", api_secret="s", account_no="ACC001", trading_token="tok-A",
        token_file=str(tmp_path / "missing_token.json"))
    instance = broker.DNSEBroker(symbol="VN30F1M", timeframe="15", config=config)
    instance._client = fake_client(**responses)
    instance._poll_interval = 0.001
    # Future residue knob: fire immediately in tests. Harmless (unread) on
    # the unmodified tree; the implementation MUST honor it (grace >= 30 s
    # flat in production — the measured ~10 s stale-replica lag, x3).
    instance._residue_grace_s = 0.0
    return instance


def _open_store_ctx(tmp_path, instance, label="residue"):
    store = BrokerStore(tmp_path / "broker.sqlite",
                        plugin_name=instance.plugin_name)
    ctx = store.open_run(
        RunIdentity(strategy_id=label, symbol="VN30F1M", timeframe="15",
                    account_id="ACC001"),
        script_source="// residue")
    instance.store_ctx = ctx
    return store, ctx


def _entry_envelope(pine_id="L"):
    return DispatchEnvelope(
        intent=EntryIntent(pine_id=pine_id, symbol="VN30F1M", side="buy", qty=1,
                           order_type=OrderType.LIMIT, limit=1500.0),
        run_tag="abcd", bar_ts_ms=1_700_000_000_000, retry_seq=0, coid_max_len=30)


def _collect_events(instance, *, min_polls, wall_clock=2.0):
    """Drive ``watch_orders`` until ``min_polls`` book polls served (or the
    wall clock expires); return every yielded event."""
    async def _run():
        events = []
        stream = instance.watch_orders()
        task = asyncio.ensure_future(stream.__anext__())
        deadline = asyncio.get_running_loop().time() + wall_clock
        try:
            while asyncio.get_running_loop().time() < deadline:
                if task.done():
                    try:
                        events.append(task.result())
                    except (StopAsyncIteration, Exception):     # noqa: BLE001
                        break
                    task = asyncio.ensure_future(stream.__anext__())
                    continue
                if instance._client.count("get_orders") >= min_polls:
                    break
                await asyncio.sleep(0.005)
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):          # noqa: BLE001
                pass
            await stream.aclose()
        return events
    return asyncio.run(_run())


def _cancelled_for(events, venue_id):
    return [e for e in events
            if e.event_type == "cancelled" and str(e.order.id) == str(venue_id)]


def _loud_records(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


# --- R1 (RED): vanished id + positive history row -> residue concluded -------

def __test_vanished_id_with_history_cancelled_row_is_concluded__(
        fake_client, tmp_path, caplog):
    """The placed order disappears from every (readable, drained) book with
    no terminal transition ever observed; ``/orders/history`` holds a
    positive Canceled row for it. The residue detector must surface it — a
    'cancelled' OrderEvent for the id, or at minimum a WARNING+ naming it.
    Today: total silence, forever (watch_orders is a change-detector over
    PRESENT rows; nothing walks the complement — #48 matrix b / FP6)."""
    b = _broker(fake_client, tmp_path, post_order=_PLACED,
                get_orders=_EMPTY_BOOK,
                get_order_detail=(400, {"code": "RESOURCE_NOT_FOUND"}),
                get_order_history=_HISTORY_CANCELLED)
    store, _ctx = _open_store_ctx(tmp_path, b)
    caplog.clear()
    try:
        asyncio.run(b.execute_entry(_entry_envelope()))
        with caplog.at_level(logging.DEBUG):
            events = _collect_events(b, min_polls=8)

        assert _cancelled_for(events, "437346") or _loud_records(caplog), (
            "order 437346 vanished from every readable book with no terminal "
            "transition, /orders/history holds a positive Canceled row — and "
            "the run said NOTHING: the exit intent leaks and the strategy "
            "believes the order still works (Phase B3 R1)")
    finally:
        store.close()


# --- R2: vanished id + NO history row -> INCONCLUSIVE forever ----------------

def __test_vanished_id_without_history_row_is_never_concluded_cancelled__(
        fake_client, tmp_path):
    """Absence is not a disposition (#55's rule, extended to the residue):
    with no positive history row the verdict stays INCONCLUSIVE forever —
    emitting 'cancelled' from absence alone re-opens the double-order hole
    on this SOFTWARE-idempotency venue. Pinned against the implementation."""
    b = _broker(fake_client, tmp_path, post_order=_PLACED,
                get_orders=_EMPTY_BOOK,
                get_order_detail=(400, {"code": "RESOURCE_NOT_FOUND"}),
                get_order_history=_EMPTY_HISTORY)
    store, _ctx = _open_store_ctx(tmp_path, b)
    try:
        asyncio.run(b.execute_entry(_entry_envelope()))
        events = _collect_events(b, min_polls=8)

        assert not _cancelled_for(events, "437346"), (
            "a residue with NO positive history row was concluded CANCELLED "
            "— absence is never a disposition")
    finally:
        store.close()


# --- R3: unreadable book -> no residue verdict (abstention) ------------------

def __test_unreadable_book_never_yields_a_residue_verdict__(
        fake_client, tmp_path):
    """A failed book read looks exactly like universal absence. The rows|None
    contract (#54/#62) marks it unreadable; the residue detector must not
    stamp, conclude, or emit from such a cycle — even with a history row
    ready to 'confirm'. A 429/500 storm must degrade to non-detection,
    never to false cancels (probes D vs E of the item-3 review)."""
    b = _broker(fake_client, tmp_path, post_order=_PLACED,
                get_orders=(500, {"message": "boom"}),
                get_order_detail=(400, {"code": "RESOURCE_NOT_FOUND"}),
                get_order_history=_HISTORY_CANCELLED)
    store, _ctx = _open_store_ctx(tmp_path, b)
    try:
        asyncio.run(b.execute_entry(_entry_envelope()))
        events = _collect_events(b, min_polls=8)

        assert not _cancelled_for(events, "437346"), (
            "an UNREADABLE book cycle produced a residue verdict — a read "
            "failure was converted into a false external cancel")
    finally:
        store.close()


# --- R4: a #41 Activated shell is never a residue subject --------------------

def __test_activated_shell_is_never_a_residue__(fake_client, tmp_path):
    """A triggered conditional stays Activated on the STOP book forever
    while its NORMAL-book child does the work (#41). The shell's row must be
    excluded via its journalled child ref — never concluded cancelled, even
    while the child is temporarily invisible (replica lag) and history holds
    nothing."""
    parent_id = "da203hg6p09g1n1vipog"
    parent_row = {"id": parent_id, "symbol": "VN30F1M", "side": "NB",
                  "quantity": 1, "orderStatus": "Activated",
                  "externalOrderId": "437400"}

    def _books(*_args, **kwargs):
        if kwargs.get("order_category") == "STOP":
            return (200, {"orders": [parent_row], "totalPages": 1})
        return _EMPTY_BOOK                      # child not visible (lag)

    b = _broker(fake_client, tmp_path,
                post_order=(201, {**parent_row, "orderStatus": "New"}),
                get_orders=_books,
                get_order_detail=(200, parent_row),
                get_order_history=_EMPTY_HISTORY)
    store, _ctx = _open_store_ctx(tmp_path, b)
    try:
        asyncio.run(b.execute_entry(DispatchEnvelope(
            intent=EntryIntent(pine_id="S", symbol="VN30F1M", side="buy",
                               qty=1, order_type=OrderType.STOP, stop=1520.0),
            run_tag="abcd", bar_ts_ms=1_700_000_000_000, retry_seq=0,
            coid_max_len=30)))
        events = _collect_events(b, min_polls=8)

        assert not _cancelled_for(events, parent_id), (
            "a #41 Activated shell was concluded as an external-cancel "
            "residue — shells are phantoms, their child does the work")
    finally:
        store.close()


def __test_absent_shell_with_history_row_is_still_never_concluded__(
        fake_client, tmp_path):
    """R4b (panel P3: R4 alone is non-discriminating — the shell is PRESENT
    there). Here the shell is ABSENT from every book AND history holds a
    Canceled row for the SHELL id: the discriminating case. The shell must
    be excluded at POPULATION (tracked by its journalled child ref only) —
    a detector that tracks the shell id would confirm-and-conclude it."""
    from pynecore_dnse.journal_wiring import journal_child_ref

    parent_id = "da203hg6p09g1n1vipog"
    shell_history = (200, {"data": [{"id": f"20260904_{parent_id}",
                                     "symbol": "VN30F1M", "side": "NB",
                                     "quantity": 1,
                                     "orderStatus": "Canceled"}],
                           "total": 1})
    b = _broker(fake_client, tmp_path,
                post_order=(201, {"id": parent_id, "symbol": "VN30F1M",
                                  "side": "NB", "quantity": 1,
                                  "orderStatus": "New"}),
                get_orders=_EMPTY_BOOK,
                get_order_detail=(400, {"code": "RESOURCE_NOT_FOUND"}),
                get_order_history=shell_history)
    store, ctx = _open_store_ctx(tmp_path, b)
    try:
        asyncio.run(b.execute_entry(DispatchEnvelope(
            intent=EntryIntent(pine_id="S", symbol="VN30F1M", side="buy",
                               qty=1, order_type=OrderType.STOP, stop=1520.0),
            run_tag="abcd", bar_ts_ms=1_700_000_000_000, retry_seq=0,
            coid_max_len=30)))
        journal_child_ref(ctx, parent_venue_id=parent_id, child_id="437400")
        events = _collect_events(b, min_polls=8)

        assert not _cancelled_for(events, parent_id), (
            "an absent #41 shell with a history Canceled row was concluded "
            "— the shell id must never be a residue subject (its child ref "
            "is the tracked identity)")
    finally:
        store.close()


# --- R6: exposure-ledger rows (#73) are never residue subjects ---------------

def __test_own_filled_position_row_is_never_a_residue__(fake_client, tmp_path):
    """#73 keeps FILLED rows LIVE as the exposure ledger — their ids
    legitimately leave the day book. Concluding 'cancelled' on one would be
    catastrophic (panel P1+P2 convergent): the engine would release the
    intent over a REAL position. Even with a pathological Canceled history
    row for the id, an exposure row must never be confirmed."""
    from pynecore_dnse.journal_wiring import journal_terminal

    b = _broker(fake_client, tmp_path, post_order=_PLACED,
                get_orders=_EMPTY_BOOK,
                get_order_detail=(400, {"code": "RESOURCE_NOT_FOUND"}),
                get_order_history=_HISTORY_CANCELLED)
    store, ctx = _open_store_ctx(tmp_path, b)
    try:
        asyncio.run(b.execute_entry(_entry_envelope()))
        journal_terminal(ctx, venue_id="437346", terminal_status="Filled",
                         filled_qty=1.0)          # row stays LIVE (#73)
        events = _collect_events(b, min_polls=8)

        assert not _cancelled_for(events, "437346"), (
            "a live exposure-ledger row (our own filled position) was "
            "concluded as an external-cancel residue")
    finally:
        store.close()


# --- R5 (control): normal terminal transition stays the primary, once -------

def __test_terminal_transition_fires_once_with_no_residue_duplicate__(
        fake_client, tmp_path):
    """The proven live channel (status transition on a RETAINED row) stays
    primary: exactly ONE cancelled event, and the residue detector never
    duplicates it."""
    cancelled_row = {**_PLACED[1], "orderStatus": "Canceled"}
    b = _broker(fake_client, tmp_path, post_order=_PLACED,
                get_orders=(200, {"orders": [cancelled_row], "totalPages": 1}),
                get_order_detail=(200, cancelled_row),
                get_order_history=_HISTORY_CANCELLED)
    store, _ctx = _open_store_ctx(tmp_path, b)
    try:
        asyncio.run(b.execute_entry(_entry_envelope()))
        events = _collect_events(b, min_polls=8)

        assert len(_cancelled_for(events, "437346")) == 1, (
            f"expected exactly one cancelled event for the transition, got "
            f"{len(_cancelled_for(events, '437346'))} — the residue detector "
            f"must never double-report the primary channel")
    finally:
        store.close()
