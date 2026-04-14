"""
Integration tests for :class:`ScriptRunner` in broker (live trading) mode.

These tests construct a ScriptRunner with a :class:`MockBrokerPlugin` and
drive a minimal ``@pyne`` strategy through it, verifying the contract between
the Pine order book, the :class:`OrderSyncEngine` and the position tracker:

- Startup capability validation rejects incompatible scripts.
- ``script.position`` is swapped to a :class:`BrokerPosition`.
- ``strategy.entry`` / ``strategy.close`` route to ``execute_*`` calls.
- :class:`OrderEvent` fills update :class:`BrokerPosition`.

No live thread, no real exchange — the engine runs with
``event_loop=None`` (one-shot ``asyncio.run`` per broker call).
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from pynecore.core.broker.models import (
    CancelIntent,
    CloseIntent,
    EntryIntent,
    ExchangeCapabilities,
    ExchangeOrder,
    ExchangePosition,
    ExitIntent,
    OrderEvent,
    OrderStatus,
    OrderType,
    LegType,
)
from pynecore.core.broker.exceptions import ExchangeCapabilityError
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.syminfo import SymInfo
from pynecore.types.ohlcv import OHLCV


# === Synthetic bars + syminfo ===


def _make_syminfo() -> SymInfo:
    """Minimal SymInfo for a crypto-like instrument."""
    return SymInfo(
        prefix="TEST",
        ticker="BTCUSDT",
        description="Test",
        currency="USDT",
        basecurrency="BTC",
        period="1D",
        type="crypto",
        mintick=0.01,
        pricescale=100.0,
        pointvalue=1.0,
        timezone="UTC",
        volumetype="base",
        opening_hours=[],
        session_starts=[],
        session_ends=[],
    )


def _make_bars(count: int, start_price: float = 50_000.0) -> list[OHLCV]:
    """A monotonically increasing, closed-bar sequence."""
    bars: list[OHLCV] = []
    for i in range(count):
        price = start_price + i * 100.0
        bars.append(OHLCV(
            timestamp=1_700_000_000 + i * 86_400,
            open=price, high=price + 50.0, low=price - 50.0,
            close=price, volume=1.0,
        ))
    return bars


# === Mock BrokerPlugin ===


@dataclass
class MockBrokerPlugin:
    """Duck-typed stand-in for :class:`BrokerPlugin` — matches only the
    methods :class:`ScriptRunner` and :class:`OrderSyncEngine` call."""
    capabilities: ExchangeCapabilities = field(default_factory=ExchangeCapabilities)
    entry_calls: list[EntryIntent] = field(default_factory=list)
    exit_calls: list[ExitIntent] = field(default_factory=list)
    close_calls: list[CloseIntent] = field(default_factory=list)
    cancel_calls: list[CancelIntent] = field(default_factory=list)
    _next_id: int = 0

    def get_capabilities(self) -> ExchangeCapabilities:
        return self.capabilities

    def _mk_order(self, intent) -> ExchangeOrder:
        self._next_id += 1
        return ExchangeOrder(
            id=f"xchg-{self._next_id}",
            symbol=getattr(intent, 'symbol', 'BTCUSDT'),
            side=getattr(intent, 'side', 'buy'),
            order_type=OrderType.MARKET,
            qty=getattr(intent, 'qty', 0.0),
            filled_qty=0.0,
            remaining_qty=getattr(intent, 'qty', 0.0),
            price=None, stop_price=None, average_fill_price=None,
            status=OrderStatus.OPEN,
            timestamp=0.0, fee=0.0, fee_currency="",
        )

    async def execute_entry(self, intent):
        self.entry_calls.append(intent)
        return [self._mk_order(intent)]

    async def execute_exit(self, intent):
        self.exit_calls.append(intent)
        return [self._mk_order(intent)]

    async def execute_close(self, intent):
        self.close_calls.append(intent)
        return self._mk_order(intent)

    async def execute_cancel(self, intent):
        self.cancel_calls.append(intent)
        return True

    async def modify_entry(self, old, new):
        return [self._mk_order(new)]

    async def modify_exit(self, old, new):
        return [self._mk_order(new)]

    async def get_open_orders(self, symbol=None):
        return []

    async def get_position(self, symbol):
        return None


# === Script templates ===


_MARKET_ENTRY_SCRIPT = '''\
"""
@pyne
"""
from pynecore.lib import script, strategy, bar_index

@script.strategy("MarketEntry")
def main():
    if bar_index == 0:
        strategy.entry("L", strategy.long, qty=1.0)
'''


_LIMIT_EXIT_BRACKET_SCRIPT = '''\
"""
@pyne
"""
from pynecore.lib import script, strategy, bar_index

@script.strategy("LimitBracket")
def main():
    if bar_index == 0:
        strategy.entry("L", strategy.long, qty=1.0)
        strategy.exit("TP", from_entry="L", limit=51_000.0, stop=49_000.0)
'''


_ENTRY_AND_CLOSE_SCRIPT = '''\
"""
@pyne
"""
from pynecore.lib import script, strategy, bar_index

@script.strategy("EntryClose")
def main():
    if bar_index == 0:
        strategy.entry("L", strategy.long, qty=1.0)
    if bar_index == 1:
        strategy.close("L")
'''


_script_counter = [0]


def _write_script(tmp_path: Path, code: str) -> Path:
    """Write the script to a uniquely-named file.

    Each test gets a fresh filename so Python's module cache doesn't
    serve a sibling test's script when the same tmp_path is recycled.
    """
    _script_counter[0] += 1
    p = tmp_path / f"strategy_test_{_script_counter[0]}.py"
    p.write_text(code)
    return p


# === Tests ===


def __test_broker_mode_swaps_position_to_broker_position__(tmp_path):
    plugin = MockBrokerPlugin(capabilities=ExchangeCapabilities())
    script_path = _write_script(tmp_path, _MARKET_ENTRY_SCRIPT)

    runner = ScriptRunner(
        script_path=script_path,
        ohlcv_iter=_make_bars(1),
        syminfo=_make_syminfo(),
        broker_plugin=plugin,  # type: ignore[arg-type]
    )

    assert isinstance(runner.script.position, BrokerPosition)
    assert runner._order_sync_engine is not None


def __test_startup_validation_rejects_incompatible_script__(tmp_path):
    """Script uses TP+SL bracket, plugin declares no tp_sl_bracket."""
    plugin = MockBrokerPlugin(capabilities=ExchangeCapabilities())
    script_path = _write_script(tmp_path, _LIMIT_EXIT_BRACKET_SCRIPT)

    runner = ScriptRunner(
        script_path=script_path,
        ohlcv_iter=_make_bars(2),
        syminfo=_make_syminfo(),
        broker_plugin=plugin,  # type: ignore[arg-type]
    )
    with pytest.raises(ExchangeCapabilityError) as excinfo:
        list(runner.run_iter())
    assert "TP+SL" in str(excinfo.value)


def __test_startup_validation_accepts_compatible_script__(tmp_path):
    # The bracket script needs tp_sl_bracket AND stop_orders from its
    # syntactic keywords; the plugin must advertise both.
    plugin = MockBrokerPlugin(
        capabilities=ExchangeCapabilities(
            tp_sl_bracket=True, stop_order=True,
        ),
    )
    script_path = _write_script(tmp_path, _LIMIT_EXIT_BRACKET_SCRIPT)

    runner = ScriptRunner(
        script_path=script_path,
        ohlcv_iter=_make_bars(2),
        syminfo=_make_syminfo(),
        broker_plugin=plugin,  # type: ignore[arg-type]
    )
    # Must not raise.
    list(runner.run_iter())


def __test_market_entry_dispatches_execute_entry__(tmp_path):
    plugin = MockBrokerPlugin(capabilities=ExchangeCapabilities())
    script_path = _write_script(tmp_path, _MARKET_ENTRY_SCRIPT)

    runner = ScriptRunner(
        script_path=script_path,
        ohlcv_iter=_make_bars(2),
        syminfo=_make_syminfo(),
        broker_plugin=plugin,  # type: ignore[arg-type]
    )
    list(runner.run_iter())

    assert len(plugin.entry_calls) == 1
    call = plugin.entry_calls[0]
    assert call.pine_id == "L"
    assert call.side == "buy"
    assert call.qty == 1.0
    assert call.order_type is OrderType.MARKET


def __test_close_dispatches_execute_close__(tmp_path):
    """``strategy.close`` only emits an order when there is an open position;
    in broker mode that requires a real exchange fill first."""
    plugin = MockBrokerPlugin(capabilities=ExchangeCapabilities())
    # 3-bar script: enter on bar 0, hold, close on bar 2 once filled.
    script_path = _write_script(tmp_path, textwrap.dedent('''\
        """
        @pyne
        """
        from pynecore.lib import script, strategy, bar_index

        @script.strategy("EntryClose")
        def main():
            if bar_index == 0:
                strategy.entry("L", strategy.long, qty=1.0)
            if bar_index == 2:
                strategy.close("L")
    '''))

    runner = ScriptRunner(
        script_path=script_path,
        ohlcv_iter=_make_bars(5),
        syminfo=_make_syminfo(),
        broker_plugin=plugin,  # type: ignore[arg-type]
    )

    it = iter(runner.run_iter())
    next(it)  # bar 0 — entry created (Pine), not yet dispatched
    next(it)  # bar 1 — sync dispatches entry; we now inject a fill

    assert len(plugin.entry_calls) == 1
    exch_id = runner._order_sync_engine._order_mapping["L"][0]
    runner._order_sync_engine.on_order_event(OrderEvent(
        order=ExchangeOrder(
            id=exch_id, symbol="BTCUSDT", side="buy",
            order_type=OrderType.MARKET, qty=1.0, filled_qty=1.0,
            remaining_qty=0.0, price=None, stop_price=None,
            average_fill_price=50_000.0, status=OrderStatus.FILLED,
            timestamp=0.0, fee=0.0, fee_currency="",
        ),
        event_type='filled', fill_price=50_000.0, fill_qty=1.0,
        timestamp=0.0, pine_id="L", leg_type=LegType.ENTRY,
    ))

    next(it)  # bar 2 — sync drains fill, script calls close (now allowed)
    next(it)  # bar 3 — sync dispatches the close order

    assert len(plugin.close_calls) == 1
    assert plugin.close_calls[0].pine_id == "L"
    assert plugin.close_calls[0].side == "sell"


def __test_order_event_fill_updates_broker_position__(tmp_path):
    plugin = MockBrokerPlugin(capabilities=ExchangeCapabilities())
    script_path = _write_script(tmp_path, _MARKET_ENTRY_SCRIPT)

    runner = ScriptRunner(
        script_path=script_path,
        ohlcv_iter=_make_bars(5),
        syminfo=_make_syminfo(),
        broker_plugin=plugin,  # type: ignore[arg-type]
    )

    # Bar-by-bar drive. Sync runs BEFORE the script on each bar, so the
    # entry created on bar 0 is dispatched only on bar 1's sync.
    it = iter(runner.run_iter())
    next(it)  # bar 0 — script creates the entry order
    next(it)  # bar 1 — sync dispatches execute_entry

    assert len(plugin.entry_calls) == 1

    # Simulate the exchange filling the order.
    exch_id = runner._order_sync_engine._order_mapping["L"][0]
    fill = OrderEvent(
        order=ExchangeOrder(
            id=exch_id, symbol="BTCUSDT", side="buy",
            order_type=OrderType.MARKET, qty=1.0, filled_qty=1.0,
            remaining_qty=0.0, price=None, stop_price=None,
            average_fill_price=50_000.0, status=OrderStatus.FILLED,
            timestamp=0.0, fee=0.0, fee_currency="",
        ),
        event_type='filled', fill_price=50_000.0, fill_qty=1.0,
        timestamp=0.0, pine_id="L", leg_type=LegType.ENTRY,
    )
    runner._order_sync_engine.on_order_event(fill)

    next(it)  # bar 2 — sync drains the fill, updates BrokerPosition

    pos = runner.script.position
    assert isinstance(pos, BrokerPosition)
    assert pos.size == 1.0
    assert len(pos.open_trades) == 1
    assert pos.open_trades[0].entry_price == 50_000.0


def __test_unchanged_intent_not_redispatched__(tmp_path):
    """A pending limit entry that Pine re-emits bar-after-bar must not
    trigger repeated execute_entry calls."""
    plugin = MockBrokerPlugin(capabilities=ExchangeCapabilities())
    script_path = _write_script(tmp_path, textwrap.dedent('''\
        """
        @pyne
        """
        from pynecore.lib import script, strategy, bar_index

        @script.strategy("PendingLimit")
        def main():
            strategy.entry("L", strategy.long, qty=1.0, limit=48_000.0)
    '''))

    runner = ScriptRunner(
        script_path=script_path,
        ohlcv_iter=_make_bars(5),
        syminfo=_make_syminfo(),
        broker_plugin=plugin,  # type: ignore[arg-type]
    )
    list(runner.run_iter())

    # Pine re-creates the same order every bar; the engine must see it as
    # unchanged from bar 2 onwards.
    assert len(plugin.entry_calls) == 1
