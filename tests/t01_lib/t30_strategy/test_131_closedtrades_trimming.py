"""
@pyne

Closed-trade numbers and report statistics span the complete trading range.
"""
import pytest

from pynecore.core.strategy_stats import calculate_strategy_statistics
from pynecore.lib import bar_index, input, na, plot, script, strategy
from pynecore.types.na import NA
from pynecore.types.ohlcv import OHLCV
from pynecore.types.strategy import Commission


@script.strategy("Closed trade trimming", initial_capital=10000000, pyramiding=2,
                 default_qty_type=strategy.fixed, default_qty_value=1)
def main(partial=input.bool(False, "Partial closes")):
    if partial:
        cycle = bar_index % 8
        if cycle == 0:
            strategy.entry("L1", strategy.long, qty=4)
        if cycle == 1:
            strategy.entry("L2", strategy.long, qty=2)
        if cycle == 2:
            strategy.close("L1", qty=1)
        if cycle == 3:
            strategy.close_all()
        if cycle == 4:
            strategy.entry("S1", strategy.short, qty=4)
        if cycle == 5:
            strategy.entry("S2", strategy.short, qty=2)
        if cycle == 6:
            strategy.close("S1", qty=1)
        if cycle == 7:
            strategy.close_all()
    elif bar_index % 2 == 0:
        strategy.entry("L", strategy.long, comment="entry-long")
    else:
        strategy.entry("S", strategy.short, comment="entry-short")
    plot(strategy.closedtrades, "closed_count")
    plot(strategy.closedtrades.first_index, "first_index")
    plot(strategy.closedtrades.entry_bar_index(0), "old_entry_bar")
    plot(strategy.closedtrades.size(0), "old_size")
    plot(strategy.closedtrades.entry_bar_index(strategy.closedtrades.first_index), "first_entry_bar")
    plot(strategy.closedtrades.entry_bar_index(strategy.closedtrades - 1), "last_entry_bar")
    plot(1 if strategy.closedtrades > strategy.closedtrades[1] else 0, "closed_changed")


# Measured on CAPITALCOM:EURUSD 240: missing P&L fields are zero, prices/times are na.
# An na index normalizes to global trade zero before applying the retained-window offset.
__test_helper_zero_fields = (
    "commission", "max_drawdown", "max_drawdown_percent", "max_runup", "max_runup_percent",
    "profit", "profit_percent", "size",
)
__test_helper_na_fields = (
    "entry_bar_index", "entry_comment", "entry_id", "entry_price", "entry_time",
    "exit_bar_index", "exit_comment", "exit_id", "exit_price", "exit_time",
)


def __test_helper_bars(count):
    for index in range(count):
        price = 100.0 if index % 2 else 101.0
        yield OHLCV(timestamp=1704067200 + index * 300, open=price, high=price,
                    low=price, close=price, volume=10.0)


def __test_helper_missing(index):
    for field in __test_helper_zero_fields:
        assert getattr(strategy.closedtrades, field)(index) == 0.0, (index, field)
    for field in __test_helper_na_fields:
        assert na(getattr(strategy.closedtrades, field)(index)), (index, field)


def __test_helper_retained(index, trade):
    for field in __test_helper_zero_fields + __test_helper_na_fields:
        actual = getattr(strategy.closedtrades, field)(index)
        expected = getattr(trade, field)
        if expected is None or na(expected):
            assert na(actual), (index, field)
        else:
            assert actual == expected, (index, field, actual, expected)


def __test_closedtrades_keeps_global_numbers_after_trimming__(runner):
    """Every closure remains observable after old trade rows leave the bounded list."""
    run = runner(__test_helper_bars(20000), syminfo_override={
        "mintick": 1.0, "pricescale": 1, "mincontract": 1.0,
    })
    total_events = 0
    checkpoints = set()
    for index, (_candle, values, closed) in enumerate(run.run_iter()):
        total_events += len(closed)
        count = max(index - 1, 0)
        first = max(count - 9000, 0)
        assert total_events == count
        assert values["closed_count"] == count
        assert values["first_index"] == first
        assert values["closed_changed"] == (1 if index >= 2 else 0)
        if count and not first:
            assert values["old_entry_bar"] == 1
            assert values["old_size"] == 1
        else:
            assert na(values["old_entry_bar"])
            assert values["old_size"] == 0
        if count:
            assert values["first_entry_bar"] == first + 1
            assert values["last_entry_bar"] == count

        if count not in (0, 1, 8999, 9000, 9001, 19998):
            continue
        checkpoints.add(count)
        position = run.script.position
        assert len(position.closed_trades) == min(count, 9000)
        __test_helper_missing(-1)
        if count and not first:
            __test_helper_retained(NA(int), position.closed_trades[0])
        else:
            __test_helper_missing(NA(int))
        __test_helper_missing(count)
        __test_helper_missing(count + 100)
        if first:
            __test_helper_missing(0)
            __test_helper_missing(first - 1)
            __test_helper_missing(first - 0.25)
        if count:
            __test_helper_retained(first, position.closed_trades[0])
            __test_helper_retained(first + 0.75, position.closed_trades[0])
            __test_helper_retained(count - 1, position.closed_trades[-1])
            stats = calculate_strategy_statistics(position, run.script.initial_capital)
            assert stats.total_trades == count
            assert stats.net_profit == count
            assert stats.avg_trade == 1.0
            assert stats.long_trades == (count + 1) // 2
            assert stats.short_trades == count // 2
            assert stats.max_cons_winning_trades == count
            assert stats.max_contracts_held == 1.0
            assert stats == calculate_strategy_statistics(position, run.script.initial_capital)
    assert total_events == 19998
    assert checkpoints == {0, 1, 8999, 9000, 9001, 19998}


@pytest.mark.parametrize("commission_type, commission_value", (
    (strategy.commission.percent, 0.05),
    (strategy.commission.cash_per_contract, 0.05),
    (strategy.commission.cash_per_order, 0.12),
))
def __test_closed_trade_summary_books_finalized_partial_fills__(
        runner, commission_type: Commission, commission_value: float):
    """Partial and multi-trade exits enter report totals once, after all exit fees."""
    run = runner(__test_helper_bars(17), syminfo_override={
        "mintick": 1.0, "pricescale": 1, "mincontract": 1.0,
    }, inputs={"partial": True})
    run.script.commission_type = commission_type
    run.script.commission_value = commission_value
    history = []
    for _candle, _values, closed in run.run_iter():
        history.extend(closed)
        stats = calculate_strategy_statistics(run.script.position, run.script.initial_capital)
        assert stats.total_trades == len(history)
        assert stats.commission_paid == pytest.approx(sum(trade.commission for trade in history))
        assert stats.long_trades == sum(trade.sign > 0 for trade in history)
        assert stats.short_trades == sum(trade.sign < 0 for trade in history)
        wins = [trade for trade in history if trade.profit > 0]
        losses = [trade for trade in history if trade.profit < 0]
        if history:
            assert stats.avg_trade_percent == pytest.approx(
                sum(trade.profit_percent for trade in history) / len(history), abs=1e-12)
        if wins:
            assert stats.avg_winning_trade == pytest.approx(
                sum(trade.profit for trade in wins) / len(wins))
            assert stats.avg_winning_trade_percent == pytest.approx(
                sum(trade.profit_percent for trade in wins) / len(wins), abs=1e-12)
        if losses:
            assert stats.avg_losing_trade == pytest.approx(
                sum(trade.profit for trade in losses) / len(losses))
            assert stats.avg_losing_trade_percent == pytest.approx(
                sum(trade.profit_percent for trade in losses) / len(losses), abs=1e-12)
        assert stats == calculate_strategy_statistics(run.script.position, run.script.initial_capital)
    assert len(history) == 12
    final_stats = calculate_strategy_statistics(run.script.position, run.script.initial_capital)
    assert final_stats.long_trades == 6
    assert final_stats.short_trades == 6
