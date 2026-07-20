"""
Regression: a no-quantity ``strategy.exit`` on a filled live position must size
the protective legs to the current net position, not to the entry order plus the
resulting open trade counted twice (issue BYBIT-001).

In live (broker) mode the entry :class:`~pynecore.lib.strategy.Order` stays in
:attr:`BrokerPosition.entry_orders` for intent stability, while
:meth:`BrokerPosition.record_fill` moves the filled quantity into
:attr:`open_trades`. Before the fix, ``strategy.exit``'s bound-size reservation
summed both, so a no-qty bracket on a 0.001 position dispatched 0.002 legs.
"""
from types import SimpleNamespace

import pytest

from pynecore import lib
from pynecore.lib import strategy, syminfo
from pynecore.lib.strategy import Order, _order_type_entry
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.intent_builder import build_intents
from pynecore.core.broker.models import (
    ExchangeOrder,
    ExitIntent,
    OrderEvent,
    OrderStatus,
    OrderType,
    LegType,
)


def _buy_fill(qty: float, price: float, *, pine_id: str) -> OrderEvent:
    order = ExchangeOrder(
        id=f"xchg-{pine_id}-buy-{qty}",
        symbol="BTCUSDT",
        side="buy",
        order_type=OrderType.MARKET,
        qty=qty,
        filled_qty=qty,
        remaining_qty=0.0,
        price=None,
        stop_price=None,
        average_fill_price=price,
        status=OrderStatus.FILLED,
        timestamp=0.0,
        fee=0.0,
        fee_currency="USDT",
    )
    return OrderEvent(
        order=order,
        event_type="filled",
        fill_price=price,
        fill_qty=qty,
        timestamp=0.0,
        pine_id=pine_id,
        from_entry=None,
        leg_type=LegType.ENTRY,
        fee=0.0,
        fee_currency="USDT",
    )


@pytest.fixture
def live_env():
    """Wire a live :class:`BrokerPosition` plus a BTCUSDT-like symbol grid."""
    prev = {
        "script": lib._script,
        "sem": lib._lib_semaphore,
        "supp": lib._strategy_suppressed,
        "bar": getattr(lib, "bar_index", 0),
        "pricescale": syminfo.pricescale,
        "minmove": syminfo.minmove,
        "mintick": syminfo.mintick,
        "srf": getattr(syminfo, "_size_round_factor", None),
    }

    pos = BrokerPosition()
    lib._script = SimpleNamespace(initial_capital=1_000_000.0, position=pos, pyramiding=1)
    lib._lib_semaphore = False
    lib._strategy_suppressed = False
    lib.bar_index = 5
    syminfo.pricescale = 100
    syminfo.minmove = 1
    syminfo.mintick = 0.01
    syminfo._size_round_factor = 1000  # 0.001 lot step

    try:
        yield pos
    finally:
        lib._script = prev["script"]
        lib._lib_semaphore = prev["sem"]
        lib._strategy_suppressed = prev["supp"]
        lib.bar_index = prev["bar"]
        syminfo.pricescale = prev["pricescale"]
        syminfo.minmove = prev["minmove"]
        syminfo.mintick = prev["mintick"]
        syminfo._size_round_factor = prev["srf"]


def __test_record_fill_marks_retained_entry_order__(live_env):
    """A fill records its quantity on the retained entry order."""
    entry = Order("Long", 0.001, order_type=_order_type_entry)
    live_env.entry_orders["Long"] = entry

    live_env.record_fill(_buy_fill(0.001, 50_000.0, pine_id="Long"))

    assert live_env.size == pytest.approx(0.001)
    assert entry.filled_qty == pytest.approx(0.001)


def __test_no_qty_exit_sizes_to_net_position_not_double__(live_env):
    """No-qty bracket on a filled 0.001 long must reserve exactly 0.001."""
    entry = Order("Long", 0.001, order_type=_order_type_entry)
    live_env.entry_orders["Long"] = entry
    live_env.record_fill(_buy_fill(0.001, 50_000.0, pine_id="Long"))

    strategy.exit("Bracket", from_entry="Long", limit=51_500.0, stop=48_500.0)

    exit_order = live_env.exit_orders[("Bracket", "Long")]
    assert abs(exit_order.size) == pytest.approx(0.001)

    intents = build_intents(
        live_env.entry_orders,
        live_env.exit_orders,
        "BTCUSDT",
        open_trades=live_env.open_trades,
    )
    exit_intents = [i for i in intents if isinstance(i, ExitIntent)]
    assert len(exit_intents) == 1
    assert exit_intents[0].qty == pytest.approx(0.001)
    # A whole-row bracket, not a partial-qty one.
    assert exit_intents[0].is_partial_qty_bracket is False
