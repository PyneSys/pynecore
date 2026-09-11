"""Unit tests for cumulative closed-trade reporting statistics."""

from dataclasses import asdict

import pytest

from pynecore import lib
from pynecore.core.broker.models import ExchangeOrder, LegType, OrderEvent, OrderStatus, OrderType
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.closed_trade_stats import ClosedTradeStats
import pynecore.core.script as script_module
from pynecore.core.strategy_stats import calculate_strategy_statistics
from pynecore.lib.strategy import Trade


_REPORT_FIELDS = (
    "commission_paid",
    "avg_trade",
    "avg_trade_percent",
    "avg_winning_trade",
    "avg_winning_trade_percent",
    "avg_losing_trade",
    "avg_losing_trade_percent",
    "ratio_avg_win_loss",
    "largest_winning_trade",
    "largest_winning_trade_percent",
    "largest_losing_trade",
    "largest_losing_trade_percent",
    "avg_bars_in_trades",
    "avg_bars_in_winning_trades",
    "avg_bars_in_losing_trades",
    "long_trades",
    "long_winning_trades",
    "long_net_profit",
    "long_net_profit_percent",
    "long_gross_profit",
    "long_gross_profit_percent",
    "long_gross_loss",
    "long_gross_loss_percent",
    "long_avg_trade",
    "long_avg_trade_percent",
    "long_largest_winning_trade",
    "long_largest_winning_trade_percent",
    "long_largest_losing_trade",
    "long_largest_losing_trade_percent",
    "long_avg_bars",
    "short_trades",
    "short_winning_trades",
    "short_net_profit",
    "short_net_profit_percent",
    "short_gross_profit",
    "short_gross_profit_percent",
    "short_gross_loss",
    "short_gross_loss_percent",
    "short_avg_trade",
    "short_avg_trade_percent",
    "short_largest_winning_trade",
    "short_largest_winning_trade_percent",
    "short_largest_losing_trade",
    "short_largest_losing_trade_percent",
    "short_avg_bars",
    "max_cons_winning_trades",
    "max_cons_losing_trades",
)


def _trade(
        index: int,
        *,
        profit: float,
        profit_percent: float,
        sign: int,
        commission: float = 0.0,
        entry_bar_index: int | None = None,
        exit_bar_index: int | None = None,
        entry_time: int | None = None,
        exit_time: int | None = None,
) -> Trade:
    entry_bar = index if entry_bar_index is None else entry_bar_index
    exit_bar = entry_bar + 1 if exit_bar_index is None else exit_bar_index
    opened_at = index * 10 if entry_time is None else entry_time
    closed_at = opened_at + 1 if exit_time is None else exit_time
    trade = Trade(
        size=float(sign),
        entry_id=f"E{index}",
        entry_bar_index=entry_bar,
        entry_time=opened_at,
        entry_price=100.0,
        commission=commission,
    )
    trade.exit_bar_index = exit_bar
    trade.exit_time = closed_at
    trade.exit_price = 100.0 + profit * sign
    trade.profit = profit
    trade.profit_percent = profit_percent
    return trade


def _add_report_trade(position: BrokerPosition, trade: Trade) -> None:
    """Book one finalized trade into a real position's cumulative report state."""
    profit = float(trade.profit)
    ratio = float(trade.profit_percent) / 100.0
    position.closed_trades.append(trade)
    position._closed_trade_stats.add(trade)
    position.closed_trades_count += 1
    position.netprofit += profit
    position.sum_profit_ratio += ratio
    if profit > 0.0:
        position.wintrades += 1
        position.grossprofit += profit
        position.sum_win_profit_ratio += ratio
    elif profit < 0.0:
        position.losstrades += 1
        position.grossloss += profit
        position.sum_loss_profit_ratio += ratio
    else:
        position.eventrades += 1


def _full_history_oracle(
        trades: list[Trade],
        position: BrokerPosition,
        initial_capital: float,
) -> dict[str, float | int]:
    """Calculate report fields directly from the complete trade history."""
    result: dict[str, float | int] = {field: 0.0 for field in _REPORT_FIELDS}
    if not trades:
        return result

    winning = [trade for trade in trades if float(trade.profit) > 0.0]
    losing = [trade for trade in trades if float(trade.profit) < 0.0]
    long_trades = [trade for trade in trades if trade.sign > 0.0]
    short_trades = [trade for trade in trades if trade.sign < 0.0]

    result["commission_paid"] = sum(float(trade.commission) for trade in trades)
    result["avg_trade"] = position.netprofit / len(trades)
    result["avg_trade_percent"] = float(position.sum_profit_ratio) / len(trades) * 100.0

    if winning:
        result["avg_winning_trade"] = sum(float(trade.profit) for trade in winning) / len(winning)
        result["avg_winning_trade_percent"] = (
            float(position.sum_win_profit_ratio) / len(winning) * 100.0
        )
        largest_win = max(winning, key=lambda trade: float(trade.profit))
        result["largest_winning_trade"] = float(largest_win.profit)
        result["largest_winning_trade_percent"] = float(largest_win.profit_percent)
        winning_bars = [
            trade.exit_bar_index - trade.entry_bar_index
            for trade in winning
            if trade.exit_bar_index >= 0
        ]
        if winning_bars:
            result["avg_bars_in_winning_trades"] = sum(winning_bars) / len(winning_bars)

    if losing:
        result["avg_losing_trade"] = sum(float(trade.profit) for trade in losing) / len(losing)
        result["avg_losing_trade_percent"] = (
            float(position.sum_loss_profit_ratio) / len(losing) * 100.0
        )
        largest_loss = min(losing, key=lambda trade: float(trade.profit))
        result["largest_losing_trade"] = float(largest_loss.profit)
        result["largest_losing_trade_percent"] = float(largest_loss.profit_percent)
        losing_bars = [
            trade.exit_bar_index - trade.entry_bar_index
            for trade in losing
            if trade.exit_bar_index >= 0
        ]
        if losing_bars:
            result["avg_bars_in_losing_trades"] = sum(losing_bars) / len(losing_bars)

    if result["avg_losing_trade"] != 0.0:
        result["ratio_avg_win_loss"] = abs(
            float(result["avg_winning_trade"]) / float(result["avg_losing_trade"])
        )

    all_bars = [
        trade.exit_bar_index - trade.entry_bar_index
        for trade in trades
        if trade.exit_bar_index >= 0
    ]
    if all_bars:
        result["avg_bars_in_trades"] = sum(all_bars) / len(all_bars)

    for prefix, side_trades in (("long", long_trades), ("short", short_trades)):
        if not side_trades:
            continue
        side_winning = [trade for trade in side_trades if float(trade.profit) > 0.0]
        side_losing = [trade for trade in side_trades if float(trade.profit) < 0.0]
        net_profit = sum(float(trade.profit) for trade in side_trades)
        net_profit_percent = net_profit / initial_capital * 100.0
        gross_profit = sum(float(trade.profit) for trade in side_winning)
        gross_loss = sum(float(trade.profit) for trade in side_losing)

        result[f"{prefix}_trades"] = len(side_trades)
        result[f"{prefix}_winning_trades"] = len(side_winning)
        result[f"{prefix}_net_profit"] = net_profit
        result[f"{prefix}_net_profit_percent"] = net_profit_percent
        result[f"{prefix}_gross_profit"] = gross_profit
        result[f"{prefix}_gross_profit_percent"] = gross_profit / initial_capital * 100.0
        result[f"{prefix}_gross_loss"] = gross_loss
        result[f"{prefix}_gross_loss_percent"] = gross_loss / initial_capital * 100.0
        result[f"{prefix}_avg_trade"] = net_profit / len(side_trades)
        result[f"{prefix}_avg_trade_percent"] = net_profit_percent / len(side_trades)

        if side_winning:
            largest_side_win = max(side_winning, key=lambda trade: float(trade.profit))
            result[f"{prefix}_largest_winning_trade"] = float(largest_side_win.profit)
            result[f"{prefix}_largest_winning_trade_percent"] = float(
                largest_side_win.profit_percent
            )
        if side_losing:
            largest_side_loss = min(side_losing, key=lambda trade: float(trade.profit))
            result[f"{prefix}_largest_losing_trade"] = float(largest_side_loss.profit)
            result[f"{prefix}_largest_losing_trade_percent"] = float(
                largest_side_loss.profit_percent
            )

        side_bars = [
            trade.exit_bar_index - trade.entry_bar_index
            for trade in side_trades
            if trade.exit_bar_index >= 0
        ]
        if side_bars:
            result[f"{prefix}_avg_bars"] = sum(side_bars) / len(side_bars)

    current_wins = 0
    current_losses = 0
    max_wins = 0
    max_losses = 0
    for trade in trades:
        profit = float(trade.profit)
        if profit > 0.0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        elif profit < 0.0:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
        else:
            current_wins = 0
            current_losses = 0
    result["max_cons_winning_trades"] = max_wins
    result["max_cons_losing_trades"] = max_losses
    return result


def _fill(side: str, qty: float, price: float, *, pine_id: str, leg: LegType) -> OrderEvent:
    order = ExchangeOrder(
        id=f"exchange-{pine_id}",
        symbol="TEST",
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
        fee=0.0,
        fee_currency="USD",
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
        fee=0.0,
        fee_currency="USD",
    )


def __test_closed_trade_stats_tracks_mixed_history_and_first_equal_extrema__():
    summary = ClosedTradeStats()
    trades = [
        _trade(0, profit=10.0, profit_percent=10.0, sign=1, commission=1.0,
               entry_bar_index=1, exit_bar_index=4),
        _trade(1, profit=10.0, profit_percent=8.0, sign=-1, commission=2.0,
               entry_bar_index=5, exit_bar_index=9),
        _trade(2, profit=-5.0, profit_percent=-5.0, sign=1, commission=3.0,
               entry_bar_index=10, exit_bar_index=12),
        _trade(3, profit=-5.0, profit_percent=-4.0, sign=-1, commission=4.0,
               entry_bar_index=14, exit_bar_index=-1),
        _trade(4, profit=0.0, profit_percent=0.0, sign=1, commission=5.0,
               entry_bar_index=20, exit_bar_index=21),
        _trade(5, profit=4.0, profit_percent=4.0, sign=1, commission=6.0,
               entry_bar_index=25, exit_bar_index=-1),
        _trade(6, profit=-7.0, profit_percent=-6.0, sign=-1, commission=7.0,
               entry_bar_index=30, exit_bar_index=35),
        _trade(7, profit=10.0, profit_percent=99.0, sign=1, commission=8.0,
               entry_bar_index=40, exit_bar_index=42),
        _trade(8, profit=3.0, profit_percent=30.0, sign=1, commission=9.0,
               entry_bar_index=50, exit_bar_index=53),
        _trade(9, profit=1.0, profit_percent=1.0, sign=-1, commission=10.0,
               entry_bar_index=60, exit_bar_index=61),
    ]
    for trade in trades:
        summary.add(trade)

    assert not hasattr(summary, "__dict__")
    assert not hasattr(summary.long, "__dict__")
    assert not hasattr(summary, "trades")
    assert summary.commission == pytest.approx(55.0)
    assert summary.winning_count == 6
    assert summary.losing_count == 3
    assert summary.winning_profit_sum == pytest.approx(38.0)
    assert summary.losing_profit_sum == pytest.approx(-17.0)
    assert summary.largest_winning_trade == pytest.approx(10.0)
    assert summary.largest_winning_trade_percent == pytest.approx(10.0)
    assert summary.largest_losing_trade == pytest.approx(-7.0)
    assert summary.largest_losing_trade_percent == pytest.approx(-6.0)
    assert (summary.bar_length_sum, summary.bar_length_count) == (21, 8)
    assert (summary.winning_bar_length_sum, summary.winning_bar_length_count) == (13, 5)
    assert (summary.losing_bar_length_sum, summary.losing_bar_length_count) == (7, 2)
    assert summary.current_winning_streak == 3
    assert summary.current_losing_streak == 0
    assert summary.max_winning_streak == 3
    assert summary.max_losing_streak == 2

    assert summary.long.count == 6
    assert summary.long.winning_count == 4
    assert summary.long.net_profit == pytest.approx(22.0)
    assert summary.long.gross_profit == pytest.approx(27.0)
    assert summary.long.gross_loss == pytest.approx(-5.0)
    assert summary.long.largest_winning_trade == pytest.approx(10.0)
    assert summary.long.largest_winning_trade_percent == pytest.approx(10.0)
    assert summary.long.largest_losing_trade == pytest.approx(-5.0)
    assert summary.long.largest_losing_trade_percent == pytest.approx(-5.0)
    assert (summary.long.bar_length_sum, summary.long.bar_length_count) == (11, 5)

    assert summary.short.count == 4
    assert summary.short.winning_count == 2
    assert summary.short.net_profit == pytest.approx(-1.0)
    assert summary.short.gross_profit == pytest.approx(11.0)
    assert summary.short.gross_loss == pytest.approx(-12.0)
    assert summary.short.largest_winning_trade == pytest.approx(10.0)
    assert summary.short.largest_winning_trade_percent == pytest.approx(8.0)
    assert summary.short.largest_losing_trade == pytest.approx(-7.0)
    assert summary.short.largest_losing_trade_percent == pytest.approx(-6.0)
    assert (summary.short.bar_length_sum, summary.short.bar_length_count) == (10, 3)


def __test_report_matches_full_history_after_retained_deque_truncation__():
    initial_capital = 100_000.0
    position = BrokerPosition()
    trades = [
        _trade(0, profit=1_000.0, profit_percent=12.5, sign=1, commission=2.0,
               entry_bar_index=0, exit_bar_index=100),
        _trade(1, profit=20.0, profit_percent=2.0, sign=-1, commission=3.0,
               entry_bar_index=100, exit_bar_index=120),
        _trade(2, profit=20.0, profit_percent=2.5, sign=1, commission=4.0,
               entry_bar_index=120, exit_bar_index=140),
        _trade(3, profit=-900.0, profit_percent=-11.0, sign=-1, commission=5.0,
               entry_bar_index=140, exit_bar_index=230),
        _trade(4, profit=-20.0, profit_percent=-2.0, sign=1, commission=6.0,
               entry_bar_index=230, exit_bar_index=-1),
    ]
    for index in range(5, 9_005):
        phase = index % 3
        profit = 7.0 if phase == 0 else -3.0 if phase == 1 else 0.0
        percent = 0.7 if phase == 0 else -0.3 if phase == 1 else 0.0
        sign = 1 if index % 2 == 0 else -1
        exit_bar = -1 if index % 11 == 0 else index + index % 5
        trades.append(_trade(
            index,
            profit=profit,
            profit_percent=percent,
            sign=sign,
            commission=(index % 7) * 0.1,
            exit_bar_index=exit_bar,
        ))

    for trade in trades:
        _add_report_trade(position, trade)

    assert position.closed_trades_count == 9_005
    assert len(position.closed_trades) == 9_000
    assert position.closed_trades[0] is trades[5]

    expected = _full_history_oracle(trades, position, initial_capital)
    first = calculate_strategy_statistics(position, initial_capital)
    second = calculate_strategy_statistics(position, initial_capital)

    for field in _REPORT_FIELDS:
        assert getattr(first, field) == pytest.approx(expected[field]), field
    assert asdict(second) == asdict(first)
    assert position.closed_trades_count == 9_005
    assert len(position.closed_trades) == 9_000


def __test_report_uses_position_peak_for_same_timestamp_closed_trades__():
    position = BrokerPosition()
    for index in range(2):
        trade = _trade(
            index,
            profit=1.0,
            profit_percent=1.0,
            sign=1,
            entry_time=100,
            exit_time=100,
        )
        _add_report_trade(position, trade)
    position.max_contracts_held_long = 2.0
    position.max_contracts_held_short = 1.0

    report = calculate_strategy_statistics(position, 1_000.0)

    assert report.max_contracts_held == pytest.approx(2.0)


@pytest.mark.parametrize("size", (2.0, -3.0))
def __test_report_preserves_adopted_position_peak_after_close__(monkeypatch, size: float):
    script = script_module.Script(initial_capital=1_000.0)
    monkeypatch.setattr(lib, "_script", script)
    position = BrokerPosition()
    script.position = position
    position.size = size
    position.sign = 1.0 if size > 0 else -1.0
    position.avg_price = 100.0
    position.reconstruct_parent_trade(entry_id="Adopted", size=size, entry_price=100.0)

    assert calculate_strategy_statistics(position, 1_000.0).max_contracts_held == abs(size)
    position.record_fill(_fill(
        "sell" if size > 0 else "buy", abs(size), 100.0,
        pine_id="Close", leg=LegType.CLOSE,
    ))
    assert position.size == 0.0
    assert calculate_strategy_statistics(position, 1_000.0).max_contracts_held == abs(size)


def __test_broker_fifo_trade_counts_drive_win_loss_percentage_averages__(monkeypatch):
    script = script_module.Script(initial_capital=1_000.0)
    monkeypatch.setattr(lib, "_script", script)
    position = BrokerPosition()
    script.position = position

    position.record_fill(_fill("buy", 1.0, 100.0, pine_id="L1", leg=LegType.ENTRY))
    position.record_fill(_fill("buy", 1.0, 120.0, pine_id="L2", leg=LegType.ENTRY))
    position.record_fill(_fill(
        "sell", 2.0, 110.0, pine_id="Close", leg=LegType.TAKE_PROFIT,
    ))

    assert [float(trade.profit) for trade in position.closed_trades] == pytest.approx([10.0, -10.0])
    assert (position.wintrades, position.losstrades, position.eventrades) == (0, 0, 1)
    assert position._closed_trade_stats.winning_count == 1
    assert position._closed_trade_stats.losing_count == 1

    report = calculate_strategy_statistics(position, 1_000.0)

    assert report.avg_winning_trade == pytest.approx(10.0)
    assert report.avg_losing_trade == pytest.approx(-10.0)
    assert report.avg_winning_trade_percent == pytest.approx(10.0)
    assert report.avg_losing_trade_percent == pytest.approx(-10.0 / 120.0 * 100.0)
