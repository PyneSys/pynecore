"""
Parity of the live binding book with the simulator's.

The same fill sequence is booked on :class:`~pynecore.lib.strategy.SimPosition`
(through its fill routine, the one a backtest runs) and on
:class:`~pynecore.core.broker.position.BrokerPosition` (through
:meth:`~pynecore.core.broker.position.BrokerPosition.record_fill`, as a plugin
reports it); both must end with the same binding book.
"""
from types import SimpleNamespace

import pytest

import pynecore.lib.strategy as strategy_module
from pynecore import lib
from pynecore.core.broker.intent_builder import CLOSE_ALL_EXIT_ID, CLOSE_EXIT_ID_PREFIX
from pynecore.core.broker.models import (
    ExchangeOrder,
    LegType,
    OrderEvent,
    OrderStatus,
    OrderType,
)
from pynecore.core.broker.position import BrokerPosition
from pynecore.lib import syminfo
from pynecore.lib.strategy import (
    Order,
    SimPosition,
    _order_type_close,
    _order_type_entry,
    _order_type_normal,
)

# A step of a sequence:
#   ("entry", id, signed qty)            -- strategy.entry fill
#   ("slices", id, (signed qty, ...))    -- one entry order filled in slices
#   ("exit", id, qty)                    -- a strategy.exit leg of the id's oldest fill
#   ("close", id, qty)                   -- strategy.close(id)
#   ("close_all",)                       -- strategy.close_all
#   ("order", signed qty)                -- an opposite strategy.order (unbound)
#   ("reverse", id, signed qty)          -- a reversing strategy.entry, its full quantity
_SEQUENCES = {
    "c432_pyramid": [
        ("entry", "L1", 1.0), ("entry", "L2", 1.0), ("entry", "L3", 1.0),
        ("exit", "L3", 1.0), ("exit", "L2", 1.0),
    ],
    "close_id_spans_two_fills": [
        ("entry", "L1", 1.0), ("entry", "L2", 2.0), ("entry", "L1", 1.0),
        ("close", "L1", 1.5),
    ],
    "partial_exit_then_close_all": [
        ("entry", "L1", 2.0), ("entry", "L2", 1.0),
        ("exit", "L2", 0.5), ("close_all",), ("entry", "L3", 1.0),
    ],
    "unbound_order_reduction": [
        ("entry", "L1", 1.0), ("entry", "L2", 2.0), ("entry", "L3", 1.0),
        ("order", -2.5),
    ],
    "exit_then_reversal_then_exit": [
        ("entry", "L1", 1.0), ("entry", "L2", 1.0), ("exit", "L1", 1.0),
        ("reverse", "S", -3.0), ("exit", "S", 1.0),
    ],
    "short_pyramid": [
        ("entry", "S1", -1.0), ("entry", "S2", -2.0), ("entry", "S3", -1.0),
        ("exit", "S2", 2.0), ("close", "S1", 1.0),
    ],
    "sliced_entry": [
        ("entry", "L1", 1.0), ("slices", "L2", (0.4, 0.6)), ("entry", "L1", 2.0),
        ("exit", "L2", 0.25), ("order", -1.5),
    ],
}


@pytest.fixture
def _sim_env(monkeypatch):
    """The script and symbol state the simulator's fill routine reads."""
    script = SimpleNamespace(
        initial_capital=1_000_000.0,
        commission_type=strategy_module.commission.percent,
        commission_value=0.0,
        close_entries_rule='FIFO',
    )
    monkeypatch.setattr(lib, '_script', script)
    monkeypatch.setattr(strategy_module, '_conv_identity_script', script)
    monkeypatch.setattr(syminfo, 'mintick', 0.01, raising=False)
    monkeypatch.setattr(syminfo, '_size_round_factor', 1e6, raising=False)
    monkeypatch.setattr(syminfo, 'pointvalue', 1.0, raising=False)
    monkeypatch.setattr(lib, 'bar_index', 0, raising=False)
    monkeypatch.setattr(lib, '_time', 0, raising=False)


def _sim_step(p: SimPosition, step: tuple) -> None:
    """Book one step on a simulator position."""
    kind = step[0]
    if kind == "entry":
        p._fill_order(Order(step[1], step[2], order_type=_order_type_entry), 100.0)
    elif kind == "slices":
        p._fill_order(Order(step[1], sum(step[2]), order_type=_order_type_entry), 100.0)
    elif kind == "exit":
        binding = next(b for b in p._entry_book if b.entry_id == step[1])
        leg = Order(step[1], -binding.sign * step[2], order_type=_order_type_close,
                    exit_id=f"{step[1]}-X")
        leg.entry_seq = binding.seq
        p._fill_order(leg, 105.0)
    elif kind == "close":
        p._fill_order(Order(step[1], -p.sign * step[2], order_type=_order_type_close,
                            exit_id=CLOSE_EXIT_ID_PREFIX + step[1]), 105.0)
    elif kind == "close_all":
        p._fill_order(Order(None, -p.size, order_type=_order_type_close,
                            exit_id=CLOSE_ALL_EXIT_ID), 105.0)
    elif kind == "order":
        p._fill_order(Order("O", step[1], order_type=_order_type_normal), 105.0)
    else:
        # The simulator books a reversal as its closing and its opening leg.
        p._fill_order(Order(step[1], -p.size, order_type=_order_type_entry), 105.0)
        p._fill_order(Order(step[1], step[2], order_type=_order_type_entry), 105.0)


def _event(signed_qty: float, price: float, *, pine_id: str | None, order_id: str,
           leg: LegType = LegType.ENTRY, from_entry: str | None = None) -> OrderEvent:
    """A fill event as a plugin reports it."""
    side = "buy" if signed_qty > 0.0 else "sell"
    qty = abs(signed_qty)
    order = ExchangeOrder(
        id=order_id, symbol="BTCUSDT", side=side, order_type=OrderType.MARKET,
        qty=qty, filled_qty=qty, remaining_qty=0.0, price=None, stop_price=None,
        average_fill_price=price, status=OrderStatus.FILLED, timestamp=0.0,
        fee=0.0, fee_currency="USDT",
    )
    return OrderEvent(
        order=order, event_type="filled", fill_price=price, fill_qty=qty,
        timestamp=0.0, pine_id=pine_id, from_entry=from_entry, leg_type=leg,
    )


def _live_step(p: BrokerPosition, order_id: str, step: tuple) -> None:
    """Book one step on a live position, its fills under venue order ``order_id``."""
    kind = step[0]
    if kind == "entry":
        p.record_fill(_event(step[2], 100.0, pine_id=step[1], order_id=order_id))
    elif kind == "slices":
        for qty in step[2]:
            p.record_fill(_event(qty, 100.0, pine_id=step[1], order_id=order_id))
    elif kind == "exit":
        p.record_fill(_event(-p.sign * step[2], 105.0, pine_id=f"{step[1]}-X",
                             order_id=order_id, leg=LegType.TAKE_PROFIT,
                             from_entry=step[1]))
    elif kind == "close":
        p.record_fill(_event(-p.sign * step[2], 105.0, pine_id=step[1],
                             order_id=order_id, leg=LegType.CLOSE),
                      close_entry_id=step[1])
    elif kind == "close_all":
        p.record_fill(_event(-p.size, 105.0, pine_id="", order_id=order_id,
                             leg=LegType.CLOSE))
    elif kind == "order":
        p.record_fill(_event(step[1], 105.0, pine_id="O", order_id=order_id))
    else:
        p.record_fill(_event(step[2] - p.size, 105.0, pine_id=step[1], order_id=order_id))


@pytest.mark.parametrize("sequence", list(_SEQUENCES.values()), ids=list(_SEQUENCES))
def __test_live_binding_book_matches_the_simulator__(_sim_env, sequence):
    """The same fills leave the same binding book on the simulator and on the live position.

    Compared after every step, per binding, oldest first: entry id, side and
    bound quantity. The simulator fills an entry order whole, so a sliced
    live entry order is one simulator fill of the slices' sum.
    """
    sim = SimPosition()
    sim.c = 100.0
    live = BrokerPosition()
    for n, step in enumerate(sequence):
        _sim_step(sim, step)
        _live_step(live, f"o{n}", step)

        assert ([(b.entry_id, b.sign, pytest.approx(b.bound)) for b in live._entry_book]
                == [(b.entry_id, b.sign, b.bound) for b in sim._entry_book]), step
        assert live.size == pytest.approx(sim.size), step
        assert sum(b.bound for b in live._entry_book) == pytest.approx(abs(live.size)), step
