"""
End-to-end integration: :class:`OrderSyncEngine` × :class:`BrokerStore`.

Ezek a tesztek lefedik azt a teljes szalagot, amit a régi WS1.5 recovery
réteg adott, de a mostani egyesített SQLite-based storage-on keresztül:

- Restart után azonos ``client_order_id`` képződik egy élő intenthez (a
  persisted ``bar_ts_ms`` anchor visszaolvasása).
- Parked dispatchek recovery-je a következő sync ``get_open_orders`` view-ja
  alapján, újra-dispatch nélkül.
- A ``record_complete`` valóban eldobja a perzisztált sort — cancelált
  intent nem resurrect-ál.
- Stale-run cleanup a következő ``open_run`` hívásra lezárja a korábbi
  crash-elt run-t (SIGKILL szimuláció).

Az egységtesztek a ``BrokerStore`` API-járól a ``test_029_broker_store.py``
fájlban élnek; itt kizárólag a két réteg együttműködését vizsgáljuk.
"""
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from pynecore import lib
from pynecore.core.broker.exceptions import OrderDispositionUnknownError
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.storage import BrokerStore, RunContext
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
BAR_TS = 1_700_000_000_000
PLUGIN = "Mock"


@pytest.fixture(autouse=True)
def _stub_script():
    prev = lib._script
    lib._script = SimpleNamespace(initial_capital=1_000_000.0)
    try:
        yield
    finally:
        lib._script = prev


@dataclass
class _MockBroker:
    """Duck-typed broker for integration forgatókönyvek."""
    entry_calls: list[DispatchEnvelope] = field(default_factory=list)
    cancel_calls: list[DispatchEnvelope] = field(default_factory=list)
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
            price=None, stop_price=None, average_fill_price=None,
            status=OrderStatus.OPEN,
            timestamp=0.0, fee=0.0, fee_currency="",
            client_order_id=envelope.client_order_id('e'),
        )

    async def execute_entry(self, envelope):
        self.entry_calls.append(envelope)
        if self.raise_on_next_entry is not None:
            err = self.raise_on_next_entry
            self.raise_on_next_entry = None
            raise err
        return [self._mk_order(envelope)]

    async def execute_exit(self, envelope):  # pragma: no cover — unused
        return [self._mk_order(envelope)]

    async def execute_close(self, envelope):  # pragma: no cover — unused
        return self._mk_order(envelope)

    async def execute_cancel(self, envelope):
        self.cancel_calls.append(envelope)
        return True

    async def modify_entry(self, old, new):  # pragma: no cover — unused
        return [self._mk_order(new)]

    async def modify_exit(self, old, new):  # pragma: no cover — unused
        return [self._mk_order(new)]

    async def get_open_orders(self, symbol=None):
        return list(self.open_orders)

    async def get_position(self, symbol):  # pragma: no cover — unused
        return None

    def watch_orders(self):  # pragma: no cover — unused
        raise NotImplementedError


def _open_ctx(store: BrokerStore) -> RunContext:
    identity = RunIdentity(
        strategy_id="integration", symbol=SYMBOL, timeframe="60",
        account_id="test-account", label=None,
    )
    return store.open_run(identity, script_source="// integration test")


def _mk_engine(
        broker: _MockBroker, ctx: RunContext | None,
) -> tuple[OrderSyncEngine, BrokerPosition]:
    pos = BrokerPosition()
    run_tag = ctx.run_tag if ctx is not None else "test"
    engine = OrderSyncEngine(
        broker=broker,  # type: ignore[arg-type]
        position=pos,
        symbol=SYMBOL,
        run_tag=run_tag,
        mintick=1.0,
        store_ctx=ctx,
    )
    return engine, pos


# === Core restart round-trip ==============================================


def __test_restart_regenerates_same_client_order_id_for_live_intent__(
        tmp_path: Path,
) -> None:
    """Restart → első dispatch ugyanazt a CO-ID-t produkálja.

    Perzisztencia nélkül a post-restart engine új ``bar_ts_ms``-re anchor-elne,
    és egy vadonatúj ``client_order_id``-t emittálna — a broker ezt új
    orderként kezelné, nem duplikátumként. A BrokerStore visszaolvassa a
    korábbi anchor-t, így a retry CO-ID változatlan marad.
    """
    db = tmp_path / "broker.sqlite"

    broker_a = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store_a:
        ctx_a = _open_ctx(store_a)
        engine_a, pos_a = _mk_engine(broker_a, ctx_a)
        pos_a.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=50_000.0,
        )
        engine_a.sync(BAR_TS)
        coid_first = broker_a.entry_calls[0].client_order_id('e')
        ctx_a.close()  # happy-path run lezárás

    # Process-restart szimuláció: új BrokerStore + új run ugyanezen
    # identity-re. Az előző run már lezárt — collision check engedi.
    broker_b = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store_b:
        ctx_b = _open_ctx(store_b)
        engine_b, pos_b = _mk_engine(broker_b, ctx_b)
        pos_b.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=50_000.0,
        )
        # Késleltetett "új" bar — anélkül, hogy replay-elne, a CO-ID eltérne.
        engine_b.sync(BAR_TS + 60_000)
        coid_second = broker_b.entry_calls[0].client_order_id('e')

    assert coid_first == coid_second, (
        "post-restart dispatch a perzisztált bar_ts_ms anchor-t kell "
        "használja, hogy a CO-ID stabil maradjon"
    )


def __test_restart_completed_intent_does_not_replay__(tmp_path: Path) -> None:
    """A restart előtt lezárt intent nem resurrect-ál.

    A ``record_complete`` törli a ``envelopes`` sort, így egy olyan fresh
    engine, amelyik már nem látja a Pine-oldali order-t, nem anchor-ol
    stale ``bar_ts_ms``-t.
    """
    db = tmp_path / "broker.sqlite"

    broker_a = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store_a:
        ctx_a = _open_ctx(store_a)
        engine_a, pos_a = _mk_engine(broker_a, ctx_a)
        pos_a.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=50_000.0,
        )
        engine_a.sync(BAR_TS)
        # Pine cancel: a diff engine cancel + complete eseményt bocsát ki.
        pos_a.entry_orders.clear()
        engine_a.sync(BAR_TS + 1)
        assert len(broker_a.cancel_calls) == 1
        ctx_a.close()

    with BrokerStore(db, plugin_name=PLUGIN) as store_b:
        ctx_b = _open_ctx(store_b)
        envelopes, pending = ctx_b.replay()
        assert envelopes == {}
        assert pending == {}


def __test_restart_recovers_parked_dispatch_via_get_open_orders__(
        tmp_path: Path,
) -> None:
    """Restart előtt parked dispatch a következő sync open-orders nézete
    alapján re-register-álódik, újra-dispatch nélkül.
    """
    db = tmp_path / "broker.sqlite"

    broker_a = _MockBroker()
    # A CO-ID determinisztikus — az engine ugyanazt generálja az A és a B
    # oldalon, ha a run_tag ugyanaz (ami a perzisztált identity miatt igaz).
    expected_envelope = DispatchEnvelope(
        intent=SimpleNamespace(pine_id="L"),  # type: ignore[arg-type]
        run_tag="",  # csak a CO-ID számításához; a helyes tag a ctx-ből jön
        bar_ts_ms=BAR_TS,
        retry_seq=0,
    )
    # Pre-computáljuk az A-oldali tagból a várt CO-ID-t a run megnyitása után.

    with BrokerStore(db, plugin_name=PLUGIN) as store_a:
        ctx_a = _open_ctx(store_a)
        expected_envelope = DispatchEnvelope(
            intent=SimpleNamespace(pine_id="L"),  # type: ignore[arg-type]
            run_tag=ctx_a.run_tag,
            bar_ts_ms=BAR_TS,
            retry_seq=0,
        )
        expected_coid = expected_envelope.client_order_id('e')

        broker_a.raise_on_next_entry = OrderDispositionUnknownError(
            "simulated network timeout",
            client_order_id=expected_coid,
            cause=TimeoutError("simulated network timeout"),
        )

        engine_a, pos_a = _mk_engine(broker_a, ctx_a)
        pos_a.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=50_000.0,
        )
        engine_a.sync(BAR_TS)
        assert expected_coid in engine_a.pending_verification, "park nem történt meg"
        ctx_a.close()

    # Restart. A bróker oldalán ott van az order a pending CO-ID-vel.
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
    with BrokerStore(db, plugin_name=PLUGIN) as store_b:
        ctx_b = _open_ctx(store_b)
        engine_b, pos_b = _mk_engine(broker_b, ctx_b)
        pos_b.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=50_000.0,
        )
        engine_b.sync(BAR_TS + 60_000)

        # Nem történt újra-dispatch: az engine felvette a pending dispatch-et.
        assert len(broker_b.entry_calls) == 0, (
            "post-restart engine nem dispatch-elhet egy parked entryt, "
            "ha a bróker oldalán már ott van"
        )
        assert "xchg-from-restart" in engine_b.order_mapping["L"]

        # A persisted park consumed, az envelope még él (entry még nem fill-ed).
        envelopes, pending = ctx_b.replay()
        assert pending == {}
        assert "L" in envelopes


# === Stale-run cleanup ====================================================


def __test_crashed_run_is_cleaned_on_next_open_run__(tmp_path: Path) -> None:
    """SIGKILL szimuláció: az első run nem hívja ctx.close()-t, heartbeat
    elöregül, a második ``open_run`` automatikusan lezárja az elsőt és
    indulhat saját instance-szal.
    """
    db = tmp_path / "broker.sqlite"

    broker_a = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store_a:
        ctx_a = _open_ctx(store_a)
        engine_a, pos_a = _mk_engine(broker_a, ctx_a)
        pos_a.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=50_000.0,
        )
        engine_a.sync(BAR_TS)

        # Crash szimuláció: NEM hívunk ctx_a.close()-t. Helyette bemocskoljuk
        # a last_heartbeat_ts_ms-t: stale küszöb alá állítjuk.
        store_a._conn.execute(
            "UPDATE runs SET last_heartbeat_ts_ms = 1 WHERE run_instance_id = ?",
            (ctx_a.run_instance_id,),
        )
        store_a._conn.commit()

    # Restart — az open_run cleanup végigmegy a stale soron.
    with BrokerStore(db, plugin_name=PLUGIN) as store_b:
        ctx_b = _open_ctx(store_b)
        assert ctx_b.run_instance_id != ctx_a.run_instance_id

        # A crash-elt sor most már ``ended_ts_ms``-elt és van egy
        # ``stale_run_cleaned`` event-je.
        old = store_b._conn.execute(
            "SELECT ended_ts_ms FROM runs WHERE run_instance_id = ?",
            (ctx_a.run_instance_id,),
        ).fetchone()
        assert old['ended_ts_ms'] is not None

        ev = store_b._conn.execute(
            "SELECT kind FROM events WHERE run_instance_id = ? AND kind = 'stale_run_cleaned'",
            (ctx_a.run_instance_id,),
        ).fetchone()
        assert ev is not None
