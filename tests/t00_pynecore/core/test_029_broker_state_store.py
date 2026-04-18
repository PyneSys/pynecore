"""
Tests for the cross-restart broker recovery layer (WS1.5).

Two layers under test:

- :class:`pynecore.core.broker.state_store.StateStore` — the append-only JSONL
  format itself. Round-trip writes / replay; torn-line tolerance; the
  ``complete`` self-compaction rule that keeps the file small under churn.

- :class:`OrderSyncEngine` integration — restarting the engine with the same
  store regenerates identical ``client_order_id``s for live intents and
  recovers parked-verification entries via the ``get_open_orders`` matching
  path that already existed for in-process recovery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from pynecore import lib
from pynecore.core.broker.exceptions import OrderDispositionUnknownError
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.state_store import (
    EnvelopeRecord,
    PendingRecord,
    StateStore,
)
from pynecore.core.broker.sync_engine import OrderSyncEngine
from pynecore.core.broker.models import (
    DispatchEnvelope,
    ExchangeCapabilities,
    ExchangeOrder,
    OrderStatus,
    OrderType,
)
from pynecore.lib.strategy import Order, _order_type_entry


SYMBOL = "BTCUSDT"
RUN_TAG = "test"
BAR_TS = 1_700_000_000_000


@pytest.fixture(autouse=True)
def _stub_script():
    prev = lib._script
    lib._script = SimpleNamespace(initial_capital=1_000_000.0)
    try:
        yield
    finally:
        lib._script = prev


# === StateStore unit tests ===


def __test_state_store_round_trip_envelope_and_pending__(tmp_path: Path) -> None:
    path = tmp_path / "state.jsonl"
    with StateStore(path) as store:
        store.record_envelope(key="Long", bar_ts_ms=BAR_TS, retry_seq=0)
        store.record_envelope(key="TP\0Long", bar_ts_ms=BAR_TS, retry_seq=0)
        store.record_park(coid="abcd-pid12345-0jw3qkz00-e0", key="Long")

    with StateStore(path) as store:
        envelopes, pending = store.replay()

    assert envelopes == {
        "Long": EnvelopeRecord(key="Long", bar_ts_ms=BAR_TS, retry_seq=0),
        "TP\0Long": EnvelopeRecord(key="TP\0Long", bar_ts_ms=BAR_TS, retry_seq=0),
    }
    assert pending == {
        "abcd-pid12345-0jw3qkz00-e0": PendingRecord(
            key="Long", coid="abcd-pid12345-0jw3qkz00-e0",
        ),
    }


def __test_state_store_complete_drops_envelope_and_pending_for_key__(
        tmp_path: Path,
) -> None:
    path = tmp_path / "state.jsonl"
    with StateStore(path) as store:
        store.record_envelope(key="Long", bar_ts_ms=BAR_TS, retry_seq=0)
        store.record_park(coid="coid-1", key="Long")
        store.record_complete(key="Long")

    envelopes, pending = StateStore(path).replay()
    assert envelopes == {}
    assert pending == {}, "complete must drop pending entries attached to the key"


def __test_state_store_unpark_drops_only_that_coid__(tmp_path: Path) -> None:
    path = tmp_path / "state.jsonl"
    with StateStore(path) as store:
        store.record_envelope(key="Long", bar_ts_ms=BAR_TS, retry_seq=0)
        store.record_park(coid="coid-1", key="Long")
        store.record_park(coid="coid-2", key="Long")
        store.record_unpark(coid="coid-1")

    envelopes, pending = StateStore(path).replay()
    assert "Long" in envelopes
    assert list(pending) == ["coid-2"]


def __test_state_store_torn_final_line_is_ignored__(tmp_path: Path) -> None:
    """Crash mid-write must not lose previously-flushed records."""
    path = tmp_path / "state.jsonl"
    with StateStore(path) as store:
        store.record_envelope(key="Long", bar_ts_ms=BAR_TS, retry_seq=0)
        store.record_envelope(key="Short", bar_ts_ms=BAR_TS, retry_seq=0)
    # Simulate a torn append: the writer crashed before the newline / mid-JSON.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"op":"envelope","key":"Tor')

    envelopes, pending = StateStore(path).replay()
    assert set(envelopes) == {"Long", "Short"}
    assert pending == {}


def __test_state_store_replay_on_missing_file_is_empty__(tmp_path: Path) -> None:
    envelopes, pending = StateStore(tmp_path / "missing.jsonl").replay()
    assert envelopes == {}
    assert pending == {}


def __test_state_store_unknown_op_is_skipped__(tmp_path: Path) -> None:
    path = tmp_path / "state.jsonl"
    with StateStore(path) as store:
        store.record_envelope(key="Long", bar_ts_ms=BAR_TS, retry_seq=0)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"op":"unknown","key":"X"}\n')

    envelopes, _ = StateStore(path).replay()
    assert "Long" in envelopes


# === Integration: restart recovery via the sync engine ===


@dataclass
class _MockBroker:
    """Minimal broker for restart-recovery scenarios."""
    entry_calls: list[DispatchEnvelope] = field(default_factory=list)
    cancel_calls: list[DispatchEnvelope] = field(default_factory=list)
    modify_entry_calls: list[tuple[DispatchEnvelope, DispatchEnvelope]] = field(
        default_factory=list,
    )
    open_orders: list[ExchangeOrder] = field(default_factory=list)
    raise_on_next_entry: Exception | None = None
    capabilities: ExchangeCapabilities = field(default_factory=ExchangeCapabilities)
    _next_id: int = 0

    def get_capabilities(self) -> ExchangeCapabilities:
        return self.capabilities

    def _mk_order(self, envelope: DispatchEnvelope) -> ExchangeOrder:
        self._next_id += 1
        intent = envelope.intent
        return ExchangeOrder(
            id=f"xchg-{self._next_id}",
            symbol=getattr(intent, 'symbol', SYMBOL),
            side=getattr(intent, 'side', 'buy'),
            order_type=OrderType.LIMIT,
            qty=getattr(intent, 'qty', 0.0),
            filled_qty=0.0,
            remaining_qty=getattr(intent, 'qty', 0.0),
            price=None,
            stop_price=None,
            average_fill_price=None,
            status=OrderStatus.OPEN,
            timestamp=0.0,
            fee=0.0,
            fee_currency="",
            client_order_id=envelope.client_order_id('e'),
        )

    async def execute_entry(self, envelope):
        self.entry_calls.append(envelope)
        if self.raise_on_next_entry is not None:
            err = self.raise_on_next_entry
            self.raise_on_next_entry = None
            raise err
        return [self._mk_order(envelope)]

    async def execute_exit(self, envelope):  # pragma: no cover — unused here
        return [self._mk_order(envelope)]

    async def execute_close(self, envelope):  # pragma: no cover — unused here
        return self._mk_order(envelope)

    async def execute_cancel(self, envelope):
        self.cancel_calls.append(envelope)
        return True

    async def modify_entry(self, old, new):
        self.modify_entry_calls.append((old, new))
        return [self._mk_order(new)]

    async def modify_exit(self, old, new):  # pragma: no cover — unused here
        return [self._mk_order(new)]

    async def get_open_orders(self, symbol=None):
        return list(self.open_orders)

    async def get_position(self, symbol):  # pragma: no cover — unused here
        return None

    def watch_orders(self):  # pragma: no cover — unused here
        raise NotImplementedError


def _mk_engine(
        broker: _MockBroker, store: StateStore | None,
) -> tuple[OrderSyncEngine, BrokerPosition]:
    pos = BrokerPosition()
    engine = OrderSyncEngine(
        broker=broker,  # type: ignore[arg-type]
        position=pos,
        symbol=SYMBOL,
        run_tag=RUN_TAG,
        mintick=1.0,
        state_store=store,
    )
    return engine, pos


def __test_restart_regenerates_same_client_order_id_for_live_intent__(
        tmp_path: Path,
) -> None:
    """Restart with same store → first dispatch produces the *same* CO-ID.

    Without persistence the post-restart engine would re-anchor the envelope to
    the new ``bar_ts_ms`` and emit a brand-new ``client_order_id`` — which the
    exchange would treat as a new order, not a duplicate of the in-flight one.
    """
    state_path = tmp_path / "state.jsonl"

    broker_a = _MockBroker()
    store_a = StateStore(state_path)
    engine_a, pos_a = _mk_engine(broker_a, store_a)
    pos_a.entry_orders["L"] = Order("L", 1.0, order_type=_order_type_entry, limit=50_000.0)
    engine_a.sync(BAR_TS)
    coid_first = broker_a.entry_calls[0].client_order_id('e')
    store_a.close()

    # Process restart — fresh broker, fresh engine, fresh position; same store path
    # and same Pine order book (the script reproduces it on every run).
    broker_b = _MockBroker()
    store_b = StateStore(state_path)
    engine_b, pos_b = _mk_engine(broker_b, store_b)
    pos_b.entry_orders["L"] = Order("L", 1.0, order_type=_order_type_entry, limit=50_000.0)
    engine_b.sync(BAR_TS + 60_000)  # later bar — would differ without persistence
    coid_second = broker_b.entry_calls[0].client_order_id('e')

    assert coid_first == coid_second, (
        "post-restart dispatch must use the persisted bar_ts_ms anchor"
    )


def __test_restart_completed_intent_does_not_replay__(tmp_path: Path) -> None:
    """An intent that was cancelled before restart must NOT resurrect.

    The ``complete`` marker dropped during cancel removes the envelope from the
    persisted state, so a fresh engine that no longer sees the Pine order does
    not anchor a stale ``bar_ts_ms``.
    """
    state_path = tmp_path / "state.jsonl"

    broker_a = _MockBroker()
    store_a = StateStore(state_path)
    engine_a, pos_a = _mk_engine(broker_a, store_a)
    pos_a.entry_orders["L"] = Order("L", 1.0, order_type=_order_type_entry, limit=50_000.0)
    engine_a.sync(BAR_TS)
    # Pine cancels the order — diff engine emits cancel + complete.
    pos_a.entry_orders.clear()
    engine_a.sync(BAR_TS + 1)
    assert len(broker_a.cancel_calls) == 1
    store_a.close()

    # Restart with no Pine order — replay must be empty.
    envelopes, pending = StateStore(state_path).replay()
    assert envelopes == {}
    assert pending == {}


def __test_restart_recovers_parked_dispatch_via_get_open_orders__(
        tmp_path: Path,
) -> None:
    """A pre-restart parked dispatch is matched on the next sync's open-orders view.

    Sequence:

    1. Engine A dispatches an entry → broker raises ``OrderDispositionUnknownError``.
    2. Engine A persists the parked CO-ID and key.
    3. Process restarts. Engine B replays the store, but its in-memory dict is empty.
    4. The exchange did receive the order. Engine B's first sync sees it via
       ``get_open_orders`` matched by the persisted CO-ID and registers the
       exchange order under the right ``intent_key`` — without re-dispatching.
    """
    state_path = tmp_path / "state.jsonl"

    broker_a = _MockBroker()

    # Compute the deterministic CO-ID the engine will allocate for this entry.
    expected_envelope = DispatchEnvelope(
        intent=SimpleNamespace(pine_id="L"),  # type: ignore[arg-type]
        run_tag=RUN_TAG,
        bar_ts_ms=BAR_TS,
        retry_seq=0,
    )
    expected_coid = expected_envelope.client_order_id('e')

    broker_a.raise_on_next_entry = OrderDispositionUnknownError(
        "simulated network timeout",
        client_order_id=expected_coid,
        cause=TimeoutError("simulated network timeout"),
    )

    store_a = StateStore(state_path)
    engine_a, pos_a = _mk_engine(broker_a, store_a)
    pos_a.entry_orders["L"] = Order("L", 1.0, order_type=_order_type_entry, limit=50_000.0)
    engine_a.sync(BAR_TS)
    assert expected_coid in engine_a.pending_verification, "park did not happen"
    store_a.close()

    # Restart. The exchange happens to have the order — broker_b returns it from
    # get_open_orders with a matching client_order_id.
    broker_b = _MockBroker()
    broker_b.open_orders = [
        ExchangeOrder(
            id="xchg-from-restart",
            symbol=SYMBOL,
            side="buy",
            order_type=OrderType.LIMIT,
            qty=1.0,
            filled_qty=0.0,
            remaining_qty=1.0,
            price=50_000.0,
            stop_price=None,
            average_fill_price=None,
            status=OrderStatus.OPEN,
            timestamp=0.0,
            fee=0.0,
            fee_currency="",
            client_order_id=expected_coid,
        ),
    ]
    store_b = StateStore(state_path)
    engine_b, pos_b = _mk_engine(broker_b, store_b)
    pos_b.entry_orders["L"] = Order("L", 1.0, order_type=_order_type_entry, limit=50_000.0)
    engine_b.sync(BAR_TS + 60_000)

    # No re-dispatch: the engine reused the in-flight order via the persisted park.
    assert len(broker_b.entry_calls) == 0, (
        "post-restart engine must NOT re-dispatch a parked entry that the "
        "exchange already has"
    )
    assert "xchg-from-restart" in engine_b.order_mapping["L"]
    # The persisted park is consumed.
    envelopes, pending = StateStore(state_path).replay()
    assert pending == {}
    assert "L" in envelopes  # envelope still live (entry not yet filled / cancelled)


def __test_no_state_store_means_no_persistence__(tmp_path: Path) -> None:
    """Backwards-compat: omitting ``state_store`` keeps behaviour pre-WS1.5.

    No file should be created and a fresh engine must be free to anchor the
    envelope to its own ``bar_ts_ms``.
    """
    broker = _MockBroker()
    engine, pos = _mk_engine(broker, store=None)
    pos.entry_orders["L"] = Order("L", 1.0, order_type=_order_type_entry, limit=50_000.0)
    engine.sync(BAR_TS)
    # No file in tmp_path because the engine was given store=None.
    assert list(tmp_path.iterdir()) == []
