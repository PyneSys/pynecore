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
import asyncio
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


# === Capital.com plugin × OrderSyncEngine integration ======================
#
# The tests below stand up a real ``CapitalCom`` plugin (with the live
# ``_call`` method replaced by an in-memory response table) wired through
# the real ``OrderSyncEngine`` and ``BrokerStore``. Three end-to-end
# scenarios are exercised:
#
# 1. Entry → confirm → restart → ``deal_id`` ref survives.
# 2. POST-confirm timeout parks the dispatch, restart recovery promotes
#    the row to ``confirmed``, next sync's ``get_open_orders`` unparks
#    the envelope without re-dispatching.
# 3. Market entry + bracket exit in the same sync cycle — the Capital
#    plugin receives both envelopes and attaches the bracket to the
#    confirmed entry row.
#
# These complement the plugin-local unit tests (which stub REST at
# ``_call``) and the sync-engine-local tests above (which stub the broker
# at the plugin interface) by exercising both layers together.


import httpx  # noqa: E402

from pynecore.lib.strategy import _order_type_close  # noqa: E402


class _FakeCapitalCom:  # Forward-declared; replaced at import time below.
    pass


# Conditional import — the capitalcom plugin is editable-installed in the
# monorepo, but tests should not hard-fail if the plugin tree is missing
# (e.g. a pynecore-only sdist). ``pytest.skip`` provides a soft gate.
try:
    from pynecore_capitalcom import CapitalCom, CapitalComConfig
except ImportError:  # pragma: no cover — defensive; plugin is installed in repo
    CapitalCom = None  # type: ignore[assignment]
    CapitalComConfig = None  # type: ignore[assignment]


if CapitalCom is not None:

    class _FakeCapitalCom(CapitalCom):  # type: ignore[no-redef]
        """Capital plugin with REST stubbed via an (endpoint, method) table.

        Mirrors the plugin-local test harness, but lives here so the
        integration tests can drive the plugin end-to-end through the real
        :class:`OrderSyncEngine` without importing the plugin's private
        test helpers.
        """

        def __init__(self, *, config, responses=None):
            super().__init__(config=config)
            self._responses: dict = responses or {}
            self._calls: list = []

        async def _call(self, endpoint, *, data=None, method='post'):
            self._calls.append((endpoint, method, data))
            err = self._responses.get(('error', endpoint, method))
            if err is not None:
                # One-shot errors — consume so restart replays can succeed.
                del self._responses[('error', endpoint, method)]
                raise err
            return self._responses.get((endpoint, method), {})


_CAP_SYMBOL = "EURUSD"
_CAP_BAR = 1_700_000_000_000
# Minimal dealingRules payload so ``_get_instrument_rules`` returns a
# usable :class:`_InstrumentRules` without each test re-declaring it.
_CAP_RULES = {
    'dealingRules': {
        'minStepDistance': {'value': 0.01},
        'minDealSize': {'value': 0.01},
        'minNormalStopOrLimitDistance': {'value': 0.0001},
    },
    'instrument': {'lotSize': 0.01},
}


def _cap_config():
    return CapitalComConfig(
        demo=True, user_email="t@example.com",
        api_key="k", api_password="p",
    )


def _open_cap_ctx(store: BrokerStore) -> RunContext:
    identity = RunIdentity(
        strategy_id="cap-integration", symbol=_CAP_SYMBOL, timeframe="60",
        account_id="cap-acct", label=None,
    )
    return store.open_run(identity, script_source="// cap integration")


def _mk_cap_engine(
        broker, ctx: RunContext,
) -> tuple[OrderSyncEngine, BrokerPosition]:
    pos = BrokerPosition()
    engine = OrderSyncEngine(
        broker=broker,  # type: ignore[arg-type]
        position=pos,
        symbol=_CAP_SYMBOL,
        run_tag=ctx.run_tag,
        mintick=0.0001,
        store_ctx=ctx,
    )
    return engine, pos


pytestmark_cap = pytest.mark.skipif(
    CapitalCom is None, reason="pynecore_capitalcom plugin not installed",
)


@pytestmark_cap
def __test_integration_capitalcom_entry_confirm_persists_deal_id_and_maps_order__(
        tmp_path: Path,
) -> None:
    """LIMIT entry dispatched through the engine → plugin's PERSIST-FIRST
    chain commits the deal_id ref and the engine's order_mapping reflects
    the exchange id in a single sync cycle.

    Exercises the full wiring: ``OrderSyncEngine._dispatch_new`` →
    :meth:`CapitalCom.execute_entry` → BrokerStore upserts/refs →
    return into :attr:`OrderSyncEngine.order_mapping`.
    """
    db = tmp_path / "broker.sqlite"

    broker = _FakeCapitalCom(config=_cap_config(), responses={
        (f'markets/{_CAP_SYMBOL}', 'get'): _CAP_RULES,
        ('workingorders', 'post'): {'dealReference': 'ref-A'},
        ('confirms/ref-A', 'get'): {
            'dealStatus': 'ACCEPTED', 'dealId': 'deal-A',
            'level': 1.0800, 'size': 1.0, 'status': 'WORKING',
        },
    })

    with BrokerStore(db, plugin_name="capitalcom") as store:
        ctx = _open_cap_ctx(store)
        broker.store_ctx = ctx
        engine, pos = _mk_cap_engine(broker, ctx)
        pos.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=1.0800,
        )
        engine.sync(_CAP_BAR)

        # engine.order_mapping is keyed by intent_key ('L') and values are
        # the exchange ids — the plugin must return deal-A for the LIMIT
        # entry.
        assert engine.order_mapping["L"] == ["deal-A"]
        # deal_id ref committed before execute_entry returned.
        row = ctx.find_by_ref('deal_id', 'deal-A')
        assert row is not None
        assert row.exchange_order_id == 'deal-A'
        assert row.state == 'confirmed'
        # A deal_reference ref is also attached from the PERSIST-FIRST
        # intermediate state — tests the whole §5.1 ordering.
        assert ctx.find_by_ref('deal_reference', 'ref-A') is not None
        ctx.close()


@pytestmark_cap
def __test_integration_capitalcom_parked_dispatch_recovers_and_unparks__(
        tmp_path: Path,
) -> None:
    """POST timeout parks the dispatch; a subsequent recovery pass + a new
    engine sync unpark it via ``get_open_orders`` without re-POSTing.

    Same run_instance throughout — the current architecture scopes
    ``orders`` / ``order_refs`` per run_instance, so this is the
    widest recovery cycle the plugin + engine actually support. A
    cross-restart variant would need run_instance-spanning reads, which
    is out of scope here.
    """
    db = tmp_path / "broker.sqlite"

    broker = _FakeCapitalCom(config=_cap_config(), responses={
        (f'markets/{_CAP_SYMBOL}', 'get'): _CAP_RULES,
        # First POST times out — engine parks the envelope.
        ('error', 'workingorders', 'post'): httpx.TimeoutException(
            "simulated POST timeout",
        ),
    })

    with BrokerStore(db, plugin_name="capitalcom") as store:
        ctx = _open_cap_ctx(store)
        broker.store_ctx = ctx
        engine, pos = _mk_cap_engine(broker, ctx)
        pos.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=1.0800,
        )
        engine.sync(_CAP_BAR)

        # The POST timeout path parks the envelope and marks the row
        # ``disposition_unknown`` — both persisted invariants of §5.1.
        assert len(engine.pending_verification) == 1
        parked_coid = next(iter(engine.pending_verification))
        row = ctx.get_order(parked_coid)
        assert row is not None
        assert row.state == 'disposition_unknown'

        # Reset the one-shot error and seed the recovery endpoints so the
        # next call graph succeeds. The plugin's recovery uses activity
        # with a ±3 s createdDateUTC window against the row's
        # created_ts_ms — align it to the row timestamp.
        assert row.created_ts_ms is not None
        import datetime as _dt
        act_iso = _dt.datetime.fromtimestamp(
            row.created_ts_ms / 1000.0, tz=_dt.UTC,
        ).strftime('%Y-%m-%dT%H:%M:%S.000')
        broker._responses.update({
            ('positions', 'get'): {'positions': []},
            ('workingorders', 'get'): {
                'workingOrders': [
                    {'market': {'epic': _CAP_SYMBOL},
                     'workingOrderData': {
                         'dealId': 'deal-P', 'direction': 'BUY',
                         'orderType': 'LIMIT', 'orderLevel': 1.0800,
                         'orderSize': 1.0,
                         'createdDateUTC': act_iso,
                     }},
                ],
            },
            ('history/activity', 'get'): {'activities': [
                {'epic': _CAP_SYMBOL, 'direction': 'BUY', 'size': 1.0,
                 'dateUTC': act_iso, 'dealId': 'deal-P',
                 'type': 'WORKING_ORDER', 'status': 'ACCEPTED'},
            ]},
        })

        # Run the plugin-side recovery directly — connect() would do the
        # same before the WS subscribes, minus the network I/O.
        asyncio.run(broker._recover_in_flight_submissions())

        # ``disposition_unknown`` → single activity match → confirmed.
        row = ctx.get_order(parked_coid)
        assert row.state == 'confirmed'
        assert row.exchange_order_id == 'deal-P'

        # Drop the POST-time error so a re-dispatch would succeed (for
        # the asserting-that-it-does-NOT-happen test below).
        broker._responses.pop(('error', 'workingorders', 'post'), None)
        pre_sync_calls = list(broker._calls)

        # Second sync: ``_verify_pending_dispatches`` hits get_open_orders
        # (GET /workingorders), looks the coid up via the ``deal_id`` ref
        # the recovery just committed, and unparks.
        engine.sync(_CAP_BAR + 60_000)
        new_calls = broker._calls[len(pre_sync_calls):]
        assert not any(c[0] == 'workingorders' and c[1] == 'post'
                       for c in new_calls), (
            "the parked envelope must unpark via get_open_orders echo, "
            "not via a fresh POST — that would duplicate the order"
        )
        assert 'deal-P' in engine.order_mapping.get("L", [])
        assert parked_coid not in engine.pending_verification

        ctx.close()


@pytestmark_cap
def __test_integration_capitalcom_full_roundtrip_entry_bracket_tp_fill__(
        tmp_path: Path,
) -> None:
    """MARKET entry + bracket exit through the engine → plugin attaches
    the bracket to the just-confirmed entry row inside a single sync cycle.

    The engine dispatches the entry intent first, which flips the row to
    ``state='confirmed'`` before the exit intent's dispatch reads it —
    the ordering invariant the plugin's exit handler depends on.
    """
    db = tmp_path / "broker.sqlite"

    broker = _FakeCapitalCom(config=_cap_config(), responses={
        (f'markets/{_CAP_SYMBOL}', 'get'): _CAP_RULES,
        # Entry (MARKET) — POST /positions + confirm
        ('positions', 'post'): {'dealReference': 'ref-E'},
        ('confirms/ref-E', 'get'): {
            'dealStatus': 'ACCEPTED', 'dealId': 'deal-E',
            'level': 1.0800, 'size': 1.0, 'status': 'OPEN',
        },
        # Exit bracket PUT
        ('positions/deal-E', 'put'): {'dealReference': 'ref-X'},
        ('confirms/ref-X', 'get'): {
            'dealStatus': 'ACCEPTED', 'dealId': 'deal-E',
        },
    })

    with BrokerStore(db, plugin_name="capitalcom") as store:
        ctx = _open_cap_ctx(store)
        broker.store_ctx = ctx
        engine, pos = _mk_cap_engine(broker, ctx)

        pos.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry,
        )
        pos.exit_orders["L"] = Order(
            "L", -1.0, order_type=_order_type_close, exit_id="TP",
            limit=1.0900, stop=1.0750,
        )
        engine.sync(_CAP_BAR)

        # Entry confirmed, bracket legs persisted.
        entry_row = ctx.find_by_ref('deal_id', 'deal-E')
        assert entry_row is not None
        assert entry_row.state == 'confirmed'

        # Bracket legs share the parent deal id under a composite
        # client_order_id convention managed by the plugin.
        leg_coids = [r.client_order_id for r in ctx.iter_live_orders()
                     if (r.extras or {}).get('leg_kind') in ('tp', 'sl')]
        assert len(leg_coids) == 2, (
            "both TP and SL bracket legs must be persisted after the "
            "exit dispatch completes"
        )

        # Both bracket legs are mapped under intent_key 'TP\0L' (the
        # engine's composite key for strategy.exit("TP", from_entry="L")).
        tp_sl_ids = engine.order_mapping.get("TP\0L", [])
        assert any(':tp' in oid for oid in tp_sl_ids), (
            "TP leg composite id missing from engine.order_mapping"
        )
        assert any(':sl' in oid for oid in tp_sl_ids), (
            "SL leg composite id missing from engine.order_mapping"
        )

        ctx.close()
