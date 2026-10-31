"""
Strategy statistics calculation module for PyneCore.
Calculates comprehensive trading statistics similar to TradingView's Strategy Tester.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .. import lib
from ..types.na import NA
from .closed_trade_stats import utc_month_index
from .csv_file import CSVWriter
from ..lib.strategy import PositionBase


@dataclass
class StrategyStatistics:
    """Complete strategy statistics matching TradingView's output"""

    # Overview metrics
    net_profit: float = 0.0
    net_profit_percent: float = 0.0
    gross_profit: float = 0.0
    gross_profit_percent: float = 0.0
    gross_loss: float = 0.0
    gross_loss_percent: float = 0.0
    max_equity_runup: float = 0.0
    max_equity_runup_percent: float = 0.0
    max_equity_drawdown: float = 0.0
    max_equity_drawdown_percent: float = 0.0
    buy_and_hold_return: float = 0.0
    buy_and_hold_return_percent: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    percent_profitable: float = 0.0
    avg_trade: float = 0.0
    avg_trade_percent: float = 0.0
    avg_winning_trade: float = 0.0
    avg_winning_trade_percent: float = 0.0
    avg_losing_trade: float = 0.0
    avg_losing_trade_percent: float = 0.0
    largest_winning_trade: float = 0.0
    largest_winning_trade_percent: float = 0.0
    largest_losing_trade: float = 0.0
    largest_losing_trade_percent: float = 0.0
    avg_bars_in_trades: float = 0.0
    avg_bars_in_winning_trades: float = 0.0
    avg_bars_in_losing_trades: float = 0.0

    # Long/Short breakdown
    long_trades: int = 0
    long_winning_trades: int = 0
    long_net_profit: float = 0.0
    long_net_profit_percent: float = 0.0
    long_gross_profit: float = 0.0
    long_gross_profit_percent: float = 0.0
    long_gross_loss: float = 0.0
    long_gross_loss_percent: float = 0.0
    long_avg_trade: float = 0.0
    long_avg_trade_percent: float = 0.0
    long_largest_winning_trade: float = 0.0
    long_largest_winning_trade_percent: float = 0.0
    long_largest_losing_trade: float = 0.0
    long_largest_losing_trade_percent: float = 0.0
    long_avg_bars: float = 0.0

    short_trades: int = 0
    short_winning_trades: int = 0
    short_net_profit: float = 0.0
    short_net_profit_percent: float = 0.0
    short_gross_profit: float = 0.0
    short_gross_profit_percent: float = 0.0
    short_gross_loss: float = 0.0
    short_gross_loss_percent: float = 0.0
    short_avg_trade: float = 0.0
    short_avg_trade_percent: float = 0.0
    short_largest_winning_trade: float = 0.0
    short_largest_winning_trade_percent: float = 0.0
    short_largest_losing_trade: float = 0.0
    short_largest_losing_trade_percent: float = 0.0
    short_avg_bars: float = 0.0

    # Other metrics
    margin_calls: int = 0
    max_contracts_held: float = 0.0
    commission_paid: float = 0.0
    total_open_trades: int = 0

    # Drawdown metrics
    max_cons_winning_trades: int = 0
    max_cons_losing_trades: int = 0

    # Additional ratio calculations
    ratio_avg_win_loss: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        """Convert statistics to dictionary for CSV export"""
        return {
            # Overview
            "Net Profit": self.net_profit,
            "Net Profit %": self.net_profit_percent,
            "Gross Profit": self.gross_profit,
            "Gross Profit %": self.gross_profit_percent,
            "Gross Loss": self.gross_loss,
            "Gross Loss %": self.gross_loss_percent,
            "Max Equity Run-up": self.max_equity_runup,
            "Max Equity Run-up %": self.max_equity_runup_percent,
            "Max Equity Drawdown": self.max_equity_drawdown,
            "Max Equity Drawdown %": self.max_equity_drawdown_percent,
            "Buy & Hold Return": self.buy_and_hold_return,
            "Buy & Hold Return %": self.buy_and_hold_return_percent,
            "Sharpe Ratio": self.sharpe_ratio,
            "Sortino Ratio": self.sortino_ratio,
            "Profit Factor": self.profit_factor,

            # Trades
            "Total Trades": self.total_trades,
            "Winning Trades": self.winning_trades,
            "Losing Trades": self.losing_trades,
            "Percent Profitable": self.percent_profitable,
            "Avg Trade": self.avg_trade,
            "Avg Trade %": self.avg_trade_percent,
            "Avg Winning Trade": self.avg_winning_trade,
            "Avg Winning Trade %": self.avg_winning_trade_percent,
            "Avg Losing Trade": self.avg_losing_trade,
            "Avg Losing Trade %": self.avg_losing_trade_percent,
            "Ratio Avg Win/Loss": self.ratio_avg_win_loss,
            "Largest Winning Trade": self.largest_winning_trade,
            "Largest Winning Trade %": self.largest_winning_trade_percent,
            "Largest Losing Trade": self.largest_losing_trade,
            "Largest Losing Trade %": self.largest_losing_trade_percent,
            "Avg # Bars in Trades": self.avg_bars_in_trades,
            "Avg # Bars in Winning Trades": self.avg_bars_in_winning_trades,
            "Avg # Bars in Losing Trades": self.avg_bars_in_losing_trades,
            "Max Consecutive Wins": self.max_cons_winning_trades,
            "Max Consecutive Losses": self.max_cons_losing_trades,

            # Long trades
            "Long Trades": self.long_trades,
            "Long Winning Trades": self.long_winning_trades,
            "Long Net Profit": self.long_net_profit,
            "Long Net Profit %": self.long_net_profit_percent,
            "Long Gross Profit": self.long_gross_profit,
            "Long Gross Profit %": self.long_gross_profit_percent,
            "Long Gross Loss": self.long_gross_loss,
            "Long Gross Loss %": self.long_gross_loss_percent,
            "Long Avg Trade": self.long_avg_trade,
            "Long Avg Trade %": self.long_avg_trade_percent,
            "Long Largest Winning Trade": self.long_largest_winning_trade,
            "Long Largest Winning Trade %": self.long_largest_winning_trade_percent,
            "Long Largest Losing Trade": self.long_largest_losing_trade,
            "Long Largest Losing Trade %": self.long_largest_losing_trade_percent,
            "Long Avg # Bars": self.long_avg_bars,

            # Short trades
            "Short Trades": self.short_trades,
            "Short Winning Trades": self.short_winning_trades,
            "Short Net Profit": self.short_net_profit,
            "Short Net Profit %": self.short_net_profit_percent,
            "Short Gross Profit": self.short_gross_profit,
            "Short Gross Profit %": self.short_gross_profit_percent,
            "Short Gross Loss": self.short_gross_loss,
            "Short Gross Loss %": self.short_gross_loss_percent,
            "Short Avg Trade": self.short_avg_trade,
            "Short Avg Trade %": self.short_avg_trade_percent,
            "Short Largest Winning Trade": self.short_largest_winning_trade,
            "Short Largest Winning Trade %": self.short_largest_winning_trade_percent,
            "Short Largest Losing Trade": self.short_largest_losing_trade,
            "Short Largest Losing Trade %": self.short_largest_losing_trade_percent,
            "Short Avg # Bars": self.short_avg_bars,

            # Other
            "Margin Calls": self.margin_calls,
            "Max Contracts Held": self.max_contracts_held,
            "Commission Paid": self.commission_paid,
            "Total Open Trades": self.total_open_trades,
        }


# noinspection PyProtectedMember
def calculate_strategy_statistics(
        position: PositionBase,
        initial_capital: float,
        first_price: float | None = None,
        last_price: float | None = None,
        risk_free_rate: float = 2.0,
        last_bar_time: int | None = None
) -> StrategyStatistics:
    """
    Calculate comprehensive strategy statistics from position data.

    :param position: PositionBase object containing all trade data
    :param initial_capital: Initial capital for percentage calculations
    :param first_price: First price for buy & hold calculation
    :param last_price: Last price for buy & hold calculation
    :param risk_free_rate: The script's yearly risk-free rate in percent, for the Sharpe and
                           Sortino ratios (``strategy()``'s default is 2)
    :param last_bar_time: UNIX time (ms) of the last bar, the month the open trades' profit
                          belongs to; the current bar's time when omitted
    :return: StrategyStatistics object with all calculated metrics
    """
    stats = StrategyStatistics()
    closed_trade_stats = position._closed_trade_stats

    # Basic metrics from position
    stats.net_profit = float(position.netprofit) if not isinstance(position.netprofit, NA) else 0.0
    stats.gross_profit = float(position.grossprofit) if not isinstance(position.grossprofit, NA) else 0.0
    stats.gross_loss = float(position.grossloss) if not isinstance(position.grossloss, NA) else 0.0
    stats.max_equity_drawdown = float(position.max_drawdown) if not isinstance(position.max_drawdown, NA) else 0.0
    stats.max_equity_runup = float(position.max_runup) if not isinstance(position.max_runup, NA) else 0.0
    # Each equity excursion's percent is measured against its own higher endpoint —
    # the peak the drop fell from, the top the rise reached — and tracked apart from
    # the currency maximum, so it is read off the position, not divided out here.
    stats.max_equity_drawdown_percent = float(position.max_drawdown_percent)
    stats.max_equity_runup_percent = float(position.max_runup_percent)

    # Every percentage of the initial capital stays 0.0 without a capital to measure against
    has_capital = initial_capital > 0
    if has_capital:
        stats.net_profit_percent = (stats.net_profit / initial_capital) * 100
        stats.gross_profit_percent = (stats.gross_profit / initial_capital) * 100
        stats.gross_loss_percent = (stats.gross_loss / initial_capital) * 100

    # Buy & Hold calculation
    if first_price and last_price and first_price > 0:
        buy_hold_shares = initial_capital / first_price
        buy_hold_value = buy_hold_shares * last_price
        stats.buy_and_hold_return = buy_hold_value - initial_capital
        if has_capital:
            stats.buy_and_hold_return_percent = (stats.buy_and_hold_return / initial_capital) * 100

    stats.total_trades = position.closed_trades_count
    stats.winning_trades = position.wintrades
    stats.losing_trades = position.losstrades
    stats.total_open_trades = len(position.open_trades)

    # Percent profitable
    if stats.total_trades > 0:
        stats.percent_profitable = (stats.winning_trades / stats.total_trades) * 100

    # Profit factor
    if stats.gross_loss != 0:
        stats.profit_factor = abs(stats.gross_profit / stats.gross_loss)

    # Calculate trade statistics from the fixed-size cumulative summary.
    if position.closed_trades_count > 0:
        stats.commission_paid = closed_trade_stats.commission
        stats.avg_trade = stats.net_profit / position.closed_trades_count
        # The percent averages are the mean of the per-trade profit RATIOS, not the
        # average profit taken against the initial capital. Each ratio's denominator
        # is that trade's own entry cost including the entry commission, and the
        # position keeps the running sums (see SimPosition's fill loop).
        stats.avg_trade_percent = (
            float(position.sum_profit_ratio) / position.closed_trades_count * 100
        )

        if closed_trade_stats.winning_count > 0:
            stats.avg_winning_trade = (
                closed_trade_stats.winning_profit_sum / closed_trade_stats.winning_count
            )
            stats.avg_winning_trade_percent = (
                float(position.sum_win_profit_ratio) / closed_trade_stats.winning_count * 100
            )
            stats.largest_winning_trade = closed_trade_stats.largest_winning_trade
            stats.largest_winning_trade_percent = closed_trade_stats.largest_winning_trade_percent
            if closed_trade_stats.winning_bar_length_count > 0:
                stats.avg_bars_in_winning_trades = (
                    closed_trade_stats.winning_bar_length_sum
                    / closed_trade_stats.winning_bar_length_count
                )

        if closed_trade_stats.losing_count > 0:
            stats.avg_losing_trade = (
                closed_trade_stats.losing_profit_sum / closed_trade_stats.losing_count
            )
            stats.avg_losing_trade_percent = (
                float(position.sum_loss_profit_ratio) / closed_trade_stats.losing_count * 100
            )
            stats.largest_losing_trade = closed_trade_stats.largest_losing_trade
            stats.largest_losing_trade_percent = closed_trade_stats.largest_losing_trade_percent
            if closed_trade_stats.losing_bar_length_count > 0:
                stats.avg_bars_in_losing_trades = (
                    closed_trade_stats.losing_bar_length_sum
                    / closed_trade_stats.losing_bar_length_count
                )

        if stats.avg_losing_trade != 0:
            stats.ratio_avg_win_loss = abs(stats.avg_winning_trade / stats.avg_losing_trade)

        if closed_trade_stats.bar_length_count > 0:
            stats.avg_bars_in_trades = (
                closed_trade_stats.bar_length_sum / closed_trade_stats.bar_length_count
            )

        long_stats = closed_trade_stats.long
        if long_stats.count > 0:
            stats.long_trades = long_stats.count
            stats.long_winning_trades = long_stats.winning_count
            stats.long_net_profit = long_stats.net_profit
            stats.long_gross_profit = long_stats.gross_profit
            stats.long_gross_loss = long_stats.gross_loss
            if has_capital:
                stats.long_net_profit_percent = stats.long_net_profit / initial_capital * 100
                stats.long_gross_profit_percent = stats.long_gross_profit / initial_capital * 100
                stats.long_gross_loss_percent = stats.long_gross_loss / initial_capital * 100
            stats.long_avg_trade = stats.long_net_profit / long_stats.count
            # Side percentages express the average profit against initial capital.
            stats.long_avg_trade_percent = stats.long_net_profit_percent / long_stats.count
            stats.long_largest_winning_trade = long_stats.largest_winning_trade
            stats.long_largest_winning_trade_percent = long_stats.largest_winning_trade_percent
            stats.long_largest_losing_trade = long_stats.largest_losing_trade
            stats.long_largest_losing_trade_percent = long_stats.largest_losing_trade_percent
            if long_stats.bar_length_count > 0:
                stats.long_avg_bars = long_stats.bar_length_sum / long_stats.bar_length_count

        short_stats = closed_trade_stats.short
        if short_stats.count > 0:
            stats.short_trades = short_stats.count
            stats.short_winning_trades = short_stats.winning_count
            stats.short_net_profit = short_stats.net_profit
            stats.short_gross_profit = short_stats.gross_profit
            stats.short_gross_loss = short_stats.gross_loss
            if has_capital:
                stats.short_net_profit_percent = stats.short_net_profit / initial_capital * 100
                stats.short_gross_profit_percent = stats.short_gross_profit / initial_capital * 100
                stats.short_gross_loss_percent = stats.short_gross_loss / initial_capital * 100
            stats.short_avg_trade = stats.short_net_profit / short_stats.count
            # Side percentages express the average profit against initial capital.
            stats.short_avg_trade_percent = stats.short_net_profit_percent / short_stats.count
            stats.short_largest_winning_trade = short_stats.largest_winning_trade
            stats.short_largest_winning_trade_percent = short_stats.largest_winning_trade_percent
            stats.short_largest_losing_trade = short_stats.largest_losing_trade
            stats.short_largest_losing_trade_percent = short_stats.largest_losing_trade_percent
            if short_stats.bar_length_count > 0:
                stats.short_avg_bars = short_stats.bar_length_sum / short_stats.bar_length_count

        stats.max_cons_winning_trades = closed_trade_stats.max_winning_streak
        stats.max_cons_losing_trades = closed_trade_stats.max_losing_streak

    stats.max_contracts_held = max(
        position.max_contracts_held_long,
        position.max_contracts_held_short,
    )

    stats.sharpe_ratio, stats.sortino_ratio = _sharpe_sortino_ratios(
        position, initial_capital, risk_free_rate,
        lib._time if last_bar_time is None else last_bar_time)

    return stats


# noinspection PyProtectedMember
def _sharpe_sortino_ratios(position: PositionBase, initial_capital: float,
                           risk_free_rate: float, last_bar_time: int) -> tuple[float, float]:
    """
    Calculate the Sharpe and Sortino ratios from the monthly returns of the realized profit.

    Every closed trade's profit belongs to the UTC calendar month it closed in, and the open
    trades add the position's open profit to the month of the last bar. The months run from
    the first entry's month — or the month after it, when no trade closed in it — to the
    last month with profit; a month without a close counts as a zero return. A month's
    return is its profit over the equity rolled forward month by month from the initial
    capital. The excess return is taken over the yearly risk-free rate spread over twelve
    months, both deviations divide by the month count, and neither ratio is annualized.

    :param position: The position whose closed-trade statistics hold the monthly profit
    :param initial_capital: The equity the first month's return is measured against
    :param risk_free_rate: The yearly risk-free rate in percent
    :param last_bar_time: UNIX time (ms) of the last bar
    :return: The Sharpe and the Sortino ratio, 0.0 where there is none, like the report's
             other undefined ratios: without a closed trade, with fewer than two months, or
             with a zero deviation
    """
    # Measured on TradingView's strategy report (17 CAPITALCOM probes, 2026-10-31), bit
    # for bit: a trade-ledger equity (initial capital + netprofit) as the month's base
    # misses by ULPs, and so does a ``(end - start) / start`` return
    ledger = position._closed_trade_stats
    month_profit = ledger.month_profit
    first_entry_time = ledger.first_entry_time
    if not month_profit or first_entry_time is None:
        return 0.0, 0.0
    for trade in position.open_trades:
        if trade.entry_time < first_entry_time:
            first_entry_time = trade.entry_time

    first_month = utc_month_index(first_entry_time)
    start = first_month if first_month in month_profit else first_month + 1
    end = utc_month_index(last_bar_time) if position.open_trades else max(month_profit)

    returns: list[float] = []
    equity = initial_capital
    for month in range(start, end + 1):
        profit = month_profit.get(month, 0.0)
        if month == end and position.open_trades:
            profit = profit + float(position.openprofit)
        if equity == 0.0:
            return 0.0, 0.0
        returns.append(profit / equity)
        equity = equity + profit

    count = len(returns)
    if count < 2:
        return 0.0, 0.0

    # Plain left-to-right sums: the built-in ``sum()`` of floats is compensated since
    # Python 3.12, which TradingView's sums are not. Every square is the exact product,
    # never ``** 2``, which is the platform's ``pow()``
    total = 0.0
    for monthly_return in returns:
        total += monthly_return
    mean = total / count
    monthly_risk_free = risk_free_rate / 12 / 100
    excess = mean - monthly_risk_free

    deviation_sum = 0.0
    downside_sum = 0.0
    for monthly_return in returns:
        deviation = monthly_return - mean
        deviation_sum += deviation * deviation
        if monthly_return < monthly_risk_free:
            shortfall = monthly_return - monthly_risk_free
            downside_sum += shortfall * shortfall

    sharpe = excess / math.sqrt(deviation_sum / count) if deviation_sum > 0.0 else 0.0
    sortino = excess / math.sqrt(downside_sum / count) if downside_sum > 0.0 else 0.0
    return sharpe, sortino


def write_strategy_statistics_csv(
        stats: StrategyStatistics,
        csv_writer: CSVWriter
) -> None:
    """
    Write strategy statistics to CSV file in TradingView format.

    :param stats: Calculated strategy statistics
    :param csv_writer: CSV writer instance (already opened)
    """
    # Row 1: Net profit
    csv_writer.write("Net profit",
                     stats.net_profit, stats.net_profit_percent,
                     stats.long_net_profit, stats.long_net_profit_percent,
                     stats.short_net_profit, stats.short_net_profit_percent
                     )
    # Row 2: Gross profit
    csv_writer.write("Gross profit",
                     stats.gross_profit, stats.gross_profit_percent,
                     stats.long_gross_profit, stats.long_gross_profit_percent,
                     stats.short_gross_profit, stats.short_gross_profit_percent
                     )
    # Row 3: Gross loss
    csv_writer.write("Gross loss",
                     stats.gross_loss, stats.gross_loss_percent,
                     stats.long_gross_loss, stats.long_gross_loss_percent,
                     stats.short_gross_loss, stats.short_gross_loss_percent
                     )
    # Row 4: Commission paid
    csv_writer.write("Commission paid",
                     stats.commission_paid, "",
                     stats.commission_paid, "",
                     0, ""
                     )
    # Row 5: Buy & hold return
    csv_writer.write("Buy & hold return",
                     stats.buy_and_hold_return, stats.buy_and_hold_return_percent,
                     "", "", "", ""
                     )
    # Row 6: Max equity run-up
    csv_writer.write("Max equity run-up",
                     stats.max_equity_runup, stats.max_equity_runup_percent,
                     "", "", "", ""
                     )
    # Row 7: Max equity drawdown
    csv_writer.write("Max equity drawdown",
                     stats.max_equity_drawdown, stats.max_equity_drawdown_percent,
                     "", "", "", ""
                     )
    # Row 8: Max contracts held
    csv_writer.write("Max contracts held",
                     stats.max_contracts_held, "",
                     stats.max_contracts_held, "",
                     0, ""
                     )

    # Empty row
    csv_writer.write("", "", "", "", "", "", "")

    # Trade statistics section
    csv_writer.write("Total trades",
                     stats.total_trades, "",
                     stats.long_trades, "",
                     stats.short_trades, ""
                     )
    csv_writer.write("Total open trades",
                     stats.total_open_trades, "",
                     stats.total_open_trades, "",
                     0, ""
                     )
    csv_writer.write("Winning trades",
                     stats.winning_trades, "",
                     stats.long_winning_trades, "",
                     stats.short_winning_trades, ""
                     )
    csv_writer.write("Losing trades",
                     stats.losing_trades, "",
                     stats.long_trades - stats.long_winning_trades, "",
                     stats.short_trades - stats.short_winning_trades, ""
                     )

    # Calculate percentages with safe division
    all_percent = stats.percent_profitable
    long_percent = (stats.long_winning_trades / stats.long_trades * 100) if stats.long_trades > 0 else 0
    short_percent = (stats.short_winning_trades / stats.short_trades * 100) if stats.short_trades > 0 else 0

    csv_writer.write("Percent profitable",
                     "", all_percent,
                     "", long_percent,
                     "", short_percent
                     )
    csv_writer.write("Avg P&L",
                     stats.avg_trade, stats.avg_trade_percent,
                     stats.long_avg_trade, stats.long_avg_trade_percent,
                     stats.short_avg_trade, stats.short_avg_trade_percent
                     )
    csv_writer.write("Avg winning trade",
                     stats.avg_winning_trade, stats.avg_winning_trade_percent,
                     stats.avg_winning_trade, stats.avg_winning_trade_percent,
                     0, ""
                     )
    csv_writer.write("Avg losing trade",
                     stats.avg_losing_trade, stats.avg_losing_trade_percent,
                     stats.avg_losing_trade, stats.avg_losing_trade_percent,
                     0, ""
                     )
    csv_writer.write("Ratio avg win / avg loss",
                     stats.ratio_avg_win_loss, "",
                     stats.ratio_avg_win_loss, "",
                     0, ""
                     )
    csv_writer.write("Largest winning trade",
                     stats.largest_winning_trade, "",
                     stats.long_largest_winning_trade, "",
                     stats.short_largest_winning_trade, ""
                     )
    csv_writer.write("Largest winning trade percent",
                     "", stats.largest_winning_trade_percent,
                     "", stats.long_largest_winning_trade_percent,
                     "", stats.short_largest_winning_trade_percent
                     )
    csv_writer.write("Largest losing trade",
                     stats.largest_losing_trade, "",
                     stats.long_largest_losing_trade, "",
                     stats.short_largest_losing_trade, ""
                     )
    csv_writer.write("Largest losing trade percent",
                     "", stats.largest_losing_trade_percent,
                     "", stats.long_largest_losing_trade_percent,
                     "", stats.short_largest_losing_trade_percent
                     )
    csv_writer.write("Avg # bars in trades",
                     stats.avg_bars_in_trades, "",
                     stats.long_avg_bars, "",
                     stats.short_avg_bars, ""
                     )
    csv_writer.write("Avg # bars in winning trades",
                     stats.avg_bars_in_winning_trades, "",
                     stats.avg_bars_in_winning_trades, "",
                     0, ""
                     )
    csv_writer.write("Avg # bars in losing trades",
                     stats.avg_bars_in_losing_trades, "",
                     stats.avg_bars_in_losing_trades, "",
                     0, ""
                     )

    # Empty row
    csv_writer.write("", "", "", "", "", "", "")

    # Additional statistics
    csv_writer.write("Sharpe ratio",
                     stats.sharpe_ratio, "",
                     "", "", "", ""
                     )
    csv_writer.write("Sortino ratio",
                     stats.sortino_ratio, "",
                     "", "", "", ""
                     )
    csv_writer.write("Profit factor",
                     stats.profit_factor, "",
                     stats.profit_factor, "",
                     0, ""
                     )
    csv_writer.write("Margin calls",
                     stats.margin_calls, "",
                     stats.margin_calls, "",
                     stats.margin_calls, ""
                     )
