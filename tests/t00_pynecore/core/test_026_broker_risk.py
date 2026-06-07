"""
Unit tests for :class:`BrokerPosition` risk management hooks.

The Phase 2b broker-side enforcement reuses the same predicates as the sim
(`_is_max_drawdown_breached`, `_is_max_intraday_loss_breached`,
`_is_max_cons_loss_days_breached`, `_is_intraday_filled_cap_reached`,
`_adjust_for_max_position_size`, `_is_direction_allowed`), so these tests
exercise the broker-side wiring: pre-submit gates in :meth:`_add_order`,
day-rollover bookkeeping in :meth:`_handle_bar_open_risk`, and post-bar
checks in :meth:`_enforce_post_bar_risk`. Engine-level dispatch of the
queued risk-close is covered by ``test_025_order_sync_engine.py``.
"""
from datetime import datetime, timezone as tz
from types import SimpleNamespace

import pytest

from pynecore import lib
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.models import (
    ExchangeOrder,
    OrderEvent,
    OrderStatus,
    OrderType,
    LegType,
)
from pynecore.lib.strategy import (
    Order,
    cash,
    direction,
    percent_of_equity,
    _order_type_entry,
    _order_type_close,
)


@pytest.fixture(autouse=True)
def _stub_script():
    prev_script = lib._script
    prev_dt = lib._datetime
    prev_period = lib.syminfo.period
    prev_opening_hours = lib.syminfo._opening_hours
    lib._script = SimpleNamespace(initial_capital=10_000.0)
    # ``_handle_bar_open_risk`` keys the trading day off ``lib.time_tradingday()``,
    # which needs a valid timeframe and consults the opening hours. With no overnight
    # session the trading day is the plain UTC calendar day, so each ``_set_day`` bar
    # produces a distinct key. Individual tests move the day via ``_set_day``.
    lib.syminfo.period = "1D"
    lib.syminfo._opening_hours = []
    lib._datetime = datetime(2024, 1, 1, tzinfo=tz.utc)
    try:
        yield
    finally:
        lib._script = prev_script
        lib._datetime = prev_dt
        lib.syminfo.period = prev_period
        lib.syminfo._opening_hours = prev_opening_hours


def _set_day(day: int) -> None:
    """Move ``lib._datetime`` to a fixed UTC midnight on the given day."""
    lib._datetime = datetime(2024, 1, day, tzinfo=tz.utc)


def _day_key(day: int) -> int:
    """Expected ``time_tradingday`` value (00:00 UTC ms) for a ``2024-01-<day>`` bar."""
    return int(datetime(2024, 1, day, tzinfo=tz.utc).timestamp() * 1000)


def _fill(side: str, qty: float, price: float, *,
          pine_id: str = "L", leg: LegType = LegType.ENTRY,
          fee: float = 0.0) -> OrderEvent:
    order = ExchangeOrder(
        id=f"xchg-{pine_id}-{side}-{qty}",
        symbol="BTCUSDT",
        side=side,
        order_type=OrderType.MARKET,
        qty=qty,
        filled_qty=qty,
        remaining_qty=0.0,
        price=None,
        stop_price=None,
        average_fill_price=price,
        status=OrderStatus.FILLED,
        timestamp=0.0,
        fee=fee,
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
        leg_type=leg,
        fee=fee,
        fee_currency="USDT",
    )


# === Pre-submit gates (`_add_order`) ===


def __test_pre_submit_max_position_size_trims_intent_qty__():
    """``risk_max_position_size`` adapts the order size at submit time."""
    p = BrokerPosition()
    p.risk_max_position_size = 5.0
    order = Order("L", 8.0, order_type=_order_type_entry)
    p._add_order(order)
    assert order.size == pytest.approx(5.0), (
        "Pre-submit trim should reduce intent qty to remaining cap room"
    )
    assert "L" in p.entry_orders


def __test_pre_submit_max_position_size_rejects_when_cap_full__():
    """If cap already hit, the new entry is silently dropped (no queue write)."""
    p = BrokerPosition()
    p.risk_max_position_size = 5.0
    p.record_fill(_fill("buy", 5.0, 100.0))
    order = Order("L2", 1.0, order_type=_order_type_entry)
    p._add_order(order)
    assert "L2" not in p.entry_orders, (
        "Cap-full pre-submit must drop the entry — no broker dispatch"
    )


def __test_pre_submit_allow_entry_in_blocks_disallowed_direction__():
    """``risk_allowed_direction = long`` rejects a short entry at submit."""
    p = BrokerPosition()
    p.risk_allowed_direction = direction.long
    short_order = Order("S", -1.0, order_type=_order_type_entry)
    p._add_order(short_order)
    assert "S" not in p.entry_orders, (
        "allow_entry_in=long should silently drop a short entry intent"
    )
    long_order = Order("L", 1.0, order_type=_order_type_entry)
    p._add_order(long_order)
    assert "L" in p.entry_orders, (
        "allow_entry_in=long must still permit a long entry"
    )


def __test_pre_submit_intraday_filled_cap_blocks_new_entry__():
    """After ``risk_max_intraday_filled_orders`` fills, new entries drop at submit."""
    p = BrokerPosition()
    p.risk_max_intraday_filled_orders = 2
    # Two fills bring the counter to the cap.
    p.record_fill(_fill("buy", 1.0, 100.0, pine_id="E1"))
    p.record_fill(_fill("sell", 1.0, 105.0, pine_id="X1",
                        leg=LegType.TAKE_PROFIT))
    assert p.risk_intraday_filled_orders == 2
    rejected = Order("E2", 1.0, order_type=_order_type_entry)
    p._add_order(rejected)
    assert "E2" not in p.entry_orders, (
        "Pre-submit intraday cap must reject the third entry"
    )


# === Day-rollover bookkeeping (`_handle_bar_open_risk`) ===


def __test_day_rollover_initialises_anchors_on_first_bar__():
    """First bar: counters reset, no halt, no cons_loss increment."""
    p = BrokerPosition()
    _set_day(1)
    p._handle_bar_open_risk()
    assert p.risk_last_trading_day == _day_key(1)
    assert p.risk_cons_loss_days == 0
    assert p.risk_intraday_start_equity == pytest.approx(10_000.0)
    assert p.risk_halt_trading is False


def __test_day_rollover_increments_cons_loss_days_on_equity_drop__():
    """Equity lower than prior day-end → ``risk_cons_loss_days`` += 1."""
    p = BrokerPosition()
    _set_day(1)
    p._handle_bar_open_risk()
    # Simulate a losing day: realized P&L drops equity below day-1 close.
    p.netprofit = -500.0
    _set_day(2)
    p._handle_bar_open_risk()
    assert p.risk_cons_loss_days == 1
    assert p.risk_last_day_equity == pytest.approx(9_500.0)


def __test_day_rollover_resets_cons_loss_days_on_winning_day__():
    """Equity higher than prior day-end → counter resets to 0."""
    p = BrokerPosition()
    p.risk_cons_loss_days = 2
    p.risk_last_day_equity = 9_000.0
    p.risk_last_trading_day = _day_key(1)
    p.netprofit = 500.0  # equity = 10_500
    _set_day(2)
    p._handle_bar_open_risk()
    assert p.risk_cons_loss_days == 0


def __test_max_cons_loss_days_halts_on_day_rollover__():
    """Reaching the cap on a new-day rollover halts trading immediately."""
    p = BrokerPosition()
    p.risk_max_cons_loss_days = 3
    p.risk_cons_loss_days = 2  # one more loss day will breach
    p.risk_last_day_equity = 9_000.0
    p.risk_last_trading_day = _day_key(1)
    p.netprofit = -500.0  # equity = 9_500 < 9_000? NO, 9_500 > 9_000, so use stronger drop
    p.netprofit = -1_500.0  # equity = 8_500 < 9_000 → loss day
    _set_day(2)
    p._handle_bar_open_risk()
    assert p.risk_cons_loss_days == 3
    assert p.risk_halt_trading is True, (
        "Cons-loss-day cap reached on rollover should halt at bar open"
    )


# === Post-bar enforce (`_enforce_post_bar_risk`) ===


def __test_max_drawdown_halts_and_queues_close_at_bar_end__():
    """Drawdown ≥ cash limit closes the position via a queued market close."""
    p = BrokerPosition()
    p.risk_max_drawdown_value = 100.0
    p.risk_max_drawdown_type = cash
    # Open long at 100, mark peak equity.
    p.record_fill(_fill("buy", 10.0, 100.0))
    p.update_unrealized_pnl(100.0)
    assert p.max_equity == pytest.approx(10_000.0)
    # Price drops to 80 → openprofit = -200, drawdown = 200 > limit 100.
    p.update_unrealized_pnl(80.0)
    assert p.max_drawdown >= 100.0
    p._enforce_post_bar_risk()
    assert p.risk_halt_trading is True
    # A market close for the full 10-unit position must be queued for sync.
    assert p.size == 10.0, "Halt only queues the close, fill comes from broker"
    queued = list(p.exit_orders.values())
    assert len(queued) == 1
    close = queued[0]
    assert close.exit_id == 'Risk management close'
    assert close.size == pytest.approx(-10.0)
    assert close.order_type == _order_type_close
    assert "Max drawdown" in (close.comment or "")


def __test_max_intraday_loss_halts_within_day__():
    """Equity drop beyond the cash limit triggers a halt at bar end."""
    p = BrokerPosition()
    p.risk_max_intraday_loss_value = 100.0
    p.risk_max_intraday_loss_type = cash
    # Anchor start-of-day equity, then take an unrealized loss.
    p.risk_intraday_start_equity = 10_000.0
    p.risk_last_trading_day = _day_key(1)
    p.record_fill(_fill("buy", 10.0, 100.0))
    p.update_unrealized_pnl(88.0)  # -120 loss > 100 limit
    p._enforce_post_bar_risk()
    assert p.risk_halt_trading is True
    queued = list(p.exit_orders.values())
    assert len(queued) == 1
    assert "Max intraday loss" in (queued[0].comment or "")


def __test_max_drawdown_percent_uses_peak_equity__():
    """``percent_of_equity`` measures threshold off the running peak."""
    p = BrokerPosition()
    p.risk_max_drawdown_value = 5.0  # 5%
    p.risk_max_drawdown_type = percent_of_equity
    # Push equity up to 12_000, then down to 11_300 (drawdown 700 = 5.83%).
    p.netprofit = 2_000.0  # equity = 12_000
    p.update_unrealized_pnl(0.0)  # no open trades — only refreshes peak
    assert p.max_equity == pytest.approx(12_000.0)
    p.netprofit = 1_300.0  # equity = 11_300, drawdown = 700
    p.update_unrealized_pnl(0.0)
    assert p.max_drawdown >= 700.0
    # 5% of peak 12_000 = 600 → 700 > 600 → breach.
    p._enforce_post_bar_risk()
    assert p.risk_halt_trading is True


def __test_post_bar_intraday_filled_cap_does_not_self_halt__():
    """Intraday cap does NOT trigger via post-bar enforce — pre-submit/sim handle it.

    The broker-side post-bar hook checks only drawdown / intraday loss /
    cons loss days; the intraday filled-orders rule is enforced at submit
    time by :meth:`_add_order` (drop new entry) and the counter increments
    in :meth:`record_fill`. There is no broker-side analogue to the sim's
    inline post-fill close-all halt because that path lives in the sim's
    fill loop, which the broker does not run.
    """
    p = BrokerPosition()
    p.risk_max_intraday_filled_orders = 2
    p.record_fill(_fill("buy", 1.0, 100.0, pine_id="E1"))
    p.record_fill(_fill("sell", 1.0, 105.0, pine_id="X1",
                        leg=LegType.TAKE_PROFIT))
    p._enforce_post_bar_risk()
    assert p.risk_halt_trading is False


def __test_halt_clears_pending_entries_and_exits__():
    """``_trigger_risk_halt`` cancels every queued order before queueing the close."""
    p = BrokerPosition()
    p.risk_max_drawdown_value = 50.0
    p.risk_max_drawdown_type = cash
    p.record_fill(_fill("buy", 10.0, 100.0))
    p.update_unrealized_pnl(100.0)
    # Pre-existing pending orders that must be cancelled by the halt.
    p._add_order(Order("PendingEntry", 1.0, order_type=_order_type_entry))
    p._add_order(Order(None, -1.0, exit_id="PendingExit",
                       order_type=_order_type_close))
    assert "PendingEntry" in p.entry_orders
    assert ("PendingExit", None) in p.exit_orders
    # Trigger drawdown halt.
    p.update_unrealized_pnl(94.0)  # loss -60 > 50
    p._enforce_post_bar_risk()
    assert p.risk_halt_trading is True
    assert "PendingEntry" not in p.entry_orders, "Halt must cancel pending entries"
    # Only the queued risk-close remains.
    assert list(p.exit_orders.keys()) == [(None, None)] or \
        any(o.exit_id == 'Risk management close' for o in p.exit_orders.values())


def __test_no_halt_when_no_open_position_and_no_breach__():
    """Sanity: no rules configured → no halt, no spurious queued orders."""
    p = BrokerPosition()
    _set_day(1)
    p._handle_bar_open_risk()
    p._enforce_post_bar_risk()
    assert p.risk_halt_trading is False
    assert p.exit_orders == {}
    assert p.entry_orders == {}
