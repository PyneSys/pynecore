"""The strategy report's Sharpe and Sortino ratios, from monthly realized returns."""

import csv
from pathlib import Path

import pytest

from pynecore.core.broker.position import BrokerPosition
from pynecore.core.closed_trade_stats import ClosedTradeStats, utc_month_index
from pynecore.core.strategy_stats import calculate_strategy_statistics
from pynecore.lib.strategy import Trade

# The trades of TradingView strategy-report probes on CAPITALCOM:EURUSD and GBPUSD, 60 and
# 1D (2026-10-31). PyneCore reproduced each probe's trades and its plotted netprofit,
# openprofit and equity bit for bit on TradingView's own bars; the profits are those trades'.
_TRADES_CSV = Path(__file__).parent / "data" / "sharpe_sortino_trades.csv"

# Per trade list: the position's open profit at the last bar, and the last bar's time (ms)
_LISTS = {
    "gbpusd_ema_60": (0.0, 1793390400000),  # 977 closed
    "eurusd_sma_60": (-4.299999999999304, 1793390400000),  # 877 closed, 1 open
    "eurusd_pyramid_60": (-23.000000000001908, 1793390400000),  # 1391 closed, 2 open
    "eurusd_window_1d": (0.0, 1793307600000),  # 50 closed, empty months on both ends
    "eurusd_steps_h1_1d": (0.0, 1793307600000),  # first trade closes in its entry month
    "eurusd_steps_h2_1d": (0.0, 1793307600000),  # first trade closes 3 months after entry
    "eurusd_steps_h3_1d": (0.0, 1793307600000),  # last trade closes in the last month but one
    "eurusd_open_tail_1d": (2251.000000000003, 1793307600000),  # 1 closed, 1 open since 2026
    "eurusd_open_only_1d": (-2123.5000000000005, 1793307600000),  # 1 open, nothing closed
    "eurusd_one_win_1d": (0.0, 1793307600000),  # a single month
    "eurusd_two_wins_1d": (0.0, 1793307600000),  # no month below the risk-free rate
    "eurusd_one_loss_1d": (0.0, 1793307600000),  # a single losing month
}

# (trade list, initial capital, risk_free_rate, TradingView's sharpeRatio and sortinoRatio);
# None is a ratio TradingView reports none for, which the report gives as 0.0
_REPORTS = [
    ("gbpusd_ema_60", 10000.0, 4.3, -0.7045345833874509, -0.6164212733374417),
    ("gbpusd_ema_60", 100000.0, 4.3, -6.025436287437427, -0.9865062472768094),
    ("gbpusd_ema_60", 10000.0, 2.0, -0.39341845478971316, -0.431051578860701),
    ("gbpusd_ema_60", 10000.0, 0.0, -0.12288269079168021, -0.16974962720095557),
    ("gbpusd_ema_60", 1000000000.0, 3.7, -50846.16270139965, -0.9999999998066013),
    ("eurusd_sma_60", 1000000000.0, 3.7, -13949.58043146072, -0.9999999974305054),
    ("eurusd_pyramid_60", 20000.0, 3.7, -1.056168454117931, -0.7344631039001895),
    ("eurusd_window_1d", 50000.0, 3.7, -0.34342063567062564, -0.4824234036320506),
    ("eurusd_steps_h1_1d", 100000.0, 2.0, 0.19785960175507494, 1.298467999081771),
    ("eurusd_steps_h2_1d", 100000.0, 2.0, -0.27103193253550745, -0.31764202177260115),
    ("eurusd_steps_h3_1d", 100000.0, 2.0, -5.523319751058251, -0.9840718482220718),
    ("eurusd_open_tail_1d", 100000.0, 2.0, -0.4902591362343388, -0.7940825339406062),
    ("eurusd_open_only_1d", 100000.0, 2.0, None, None),
    ("eurusd_one_win_1d", 100000.0, 2.0, None, None),
    ("eurusd_two_wins_1d", 100000.0, 0.0, 1.3369412065383353, None),
    ("eurusd_one_loss_1d", 100000.0, 2.0, None, None),
]


def _trade(entry_time: int, profit: float, exit_time: int = -1) -> Trade:
    trade = Trade(size=1.0, entry_id="L", entry_bar_index=0, entry_time=entry_time,
                  entry_price=1.0, commission=0.0)
    trade.exit_time = exit_time
    trade.profit = profit
    return trade


# noinspection PyProtectedMember
def _position(trades: str) -> BrokerPosition:
    """A position holding one trade list's closed and open trades."""
    position = BrokerPosition()
    with open(_TRADES_CSV, newline="") as file:
        for row in csv.DictReader(file):
            if row["trades"] != trades:
                continue
            if row["kind"] == "closed":
                trade = _trade(int(row["entry_time"]), float(row["profit"]), int(row["exit_time"]))
                trade.exit_bar_index = 1
                position._closed_trade_stats.add(trade)
            else:
                position.open_trades.append(_trade(int(row["entry_time"]), float(row["profit"])))
    position.openprofit = _LISTS[trades][0]
    return position


def _same(value: float, expected: float | None) -> bool:
    """Bit-equal, or 0.0 where TradingView reports no value."""
    return value == (0.0 if expected is None else expected)


def __test_utc_month_index_numbers_calendar_months__():
    """Months are UTC calendar months, consecutive across years and before 1970"""
    assert utc_month_index(0) == 1970 * 12
    assert utc_month_index(-1) == 1969 * 12 + 11
    assert utc_month_index(1722470399999) == 2024 * 12 + 6  # 2024-07-31T23:59:59.999Z
    assert utc_month_index(1722470400000) == 2024 * 12 + 7  # 2024-08-01T00:00:00.000Z
    assert utc_month_index(1735689600000) - utc_month_index(1735689599999) == 1  # into 2025


def __test_closed_trade_profit_books_into_its_utc_exit_month__():
    """Each closed trade's profit is summed into the month it exits in, in closing order"""
    stats = ClosedTradeStats()
    stats.add(_trade(1722300000000, 1.25, exit_time=1722470399999))
    stats.add(_trade(1722200000000, -0.5, exit_time=1722470400000))
    stats.add(_trade(1722470400000, 0.1, exit_time=1722470500000))
    stats.add(_trade(1722470500000, 0.2, exit_time=1722470600000))

    assert stats.month_profit == {
        2024 * 12 + 6: 1.25,
        2024 * 12 + 7: (-0.5 + 0.1) + 0.2,
    }
    assert stats.first_entry_time == 1722200000000


@pytest.mark.parametrize(
    "trades, initial_capital, risk_free_rate, sharpe, sortino", _REPORTS,
    ids=[f"{trades}-{capital:g}-{rate:g}" for trades, capital, rate, _, _ in _REPORTS],
)
def __test_report_ratios_reproduce_tradingview__(
        trades: str, initial_capital: float, risk_free_rate: float,
        sharpe: float | None, sortino: float | None):
    """Sharpe and Sortino ratios reproduce TradingView's strategy report bit for bit"""
    report = calculate_strategy_statistics(
        _position(trades), initial_capital,
        risk_free_rate=risk_free_rate, last_bar_time=_LISTS[trades][1],
    )
    assert _same(report.sharpe_ratio, sharpe), report.sharpe_ratio
    assert _same(report.sortino_ratio, sortino), report.sortino_ratio
