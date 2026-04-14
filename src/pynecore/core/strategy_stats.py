"""
Strategy statistics calculation module for PyneCore.
Calculates comprehensive trading statistics similar to TradingView's Strategy Tester.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..types.na import NA
from ..lib.strategy import Trade
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


def calculate_strategy_statistics(
        position: PositionBase,
        initial_capital: float,
        equity_curve: list[float] | None = None,
        first_price: float | None = None,
        last_price: float | None = None
) -> StrategyStatistics:
    """
    Calculate comprehensive strategy statistics from position data.

    :param position: Position object containing all trade data
    :param initial_capital: Initial capital for percentage calculations
    :param equity_curve: List of equity values for Sharpe/Sortino calculations
    :param first_price: First price for buy & hold calculation
    :param last_price: Last price for buy & hold calculation
    :return: StrategyStatistics object with all calculated metrics
    """
    stats = StrategyStatistics()

    # Basic metrics from position
    stats.net_profit = float(position.netprofit) if not isinstance(position.netprofit, NA) else 0.0
    stats.gross_profit = float(position.grossprofit) if not isinstance(position.grossprofit, NA) else 0.0
    stats.gross_loss = float(position.grossloss) if not isinstance(position.grossloss, NA) else 0.0
    stats.max_equity_drawdown = float(position.max_drawdown) if not isinstance(position.max_drawdown, NA) else 0.0
    stats.max_equity_runup = float(position.max_runup) if not isinstance(position.max_runup, NA) else 0.0

    # Calculate percentages
    if initial_capital > 0:
        stats.net_profit_percent = (stats.net_profit / initial_capital) * 100
        stats.gross_profit_percent = (stats.gross_profit / initial_capital) * 100
        stats.gross_loss_percent = (stats.gross_loss / initial_capital) * 100
        stats.max_equity_drawdown_percent = (stats.max_equity_drawdown / initial_capital) * 100
        stats.max_equity_runup_percent = (stats.max_equity_runup / initial_capital) * 100

    # Buy & Hold calculation
    if first_price and last_price and first_price > 0:
        buy_hold_shares = initial_capital / first_price
        buy_hold_value = buy_hold_shares * last_price
        stats.buy_and_hold_return = buy_hold_value - initial_capital
        stats.buy_and_hold_return_percent = (stats.buy_and_hold_return / initial_capital) * 100

    # Get all trades (closed + open)
    all_trades: list[Trade] = list(position.closed_trades) + position.open_trades
    closed_trades = list(position.closed_trades)

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

    # Calculate trade statistics
    if closed_trades:
        # Commission
        stats.commission_paid = float(sum(trade.commission for trade in closed_trades))

        # Average calculations
        stats.avg_trade = stats.net_profit / len(closed_trades)
        stats.avg_trade_percent = stats.net_profit_percent / len(closed_trades)

        # Separate winning and losing trades
        winning_trades = [t for t in closed_trades if float(t.profit) > 0]
        losing_trades = [t for t in closed_trades if float(t.profit) < 0]

        # Winning trades statistics
        if winning_trades:
            total_win_profit = sum(float(t.profit) for t in winning_trades)
            stats.avg_winning_trade = total_win_profit / len(winning_trades)
            stats.avg_winning_trade_percent = stats.avg_winning_trade / initial_capital * 100

            # Largest winning trade
            max_win = max(winning_trades, key=lambda t: float(t.profit))
            stats.largest_winning_trade = float(max_win.profit)
            stats.largest_winning_trade_percent = float(max_win.profit_percent)

            # Average bars in winning trades
            bars_in_wins = [t.exit_bar_index - t.entry_bar_index for t in winning_trades if t.exit_bar_index >= 0]
            if bars_in_wins:
                stats.avg_bars_in_winning_trades = sum(bars_in_wins) / len(bars_in_wins)

        # Losing trades statistics
        if losing_trades:
            total_loss_profit = sum(float(t.profit) for t in losing_trades)
            stats.avg_losing_trade = total_loss_profit / len(losing_trades)
            stats.avg_losing_trade_percent = stats.avg_losing_trade / initial_capital * 100

            # Largest losing trade
            max_loss = min(losing_trades, key=lambda t: float(t.profit))
            stats.largest_losing_trade = float(max_loss.profit)
            stats.largest_losing_trade_percent = float(max_loss.profit_percent)

            # Average bars in losing trades
            bars_in_losses = [t.exit_bar_index - t.entry_bar_index for t in losing_trades if t.exit_bar_index >= 0]
            if bars_in_losses:
                stats.avg_bars_in_losing_trades = sum(bars_in_losses) / len(bars_in_losses)

        # Ratio of average win to average loss
        if stats.avg_losing_trade != 0:
            stats.ratio_avg_win_loss = abs(stats.avg_winning_trade / stats.avg_losing_trade)

        # Average bars in all trades
        bars_in_trades = [t.exit_bar_index - t.entry_bar_index for t in closed_trades if t.exit_bar_index >= 0]
        if bars_in_trades:
            stats.avg_bars_in_trades = sum(bars_in_trades) / len(bars_in_trades)

        # Long/Short breakdown
        long_trades = [t for t in closed_trades if t.sign > 0]
        short_trades = [t for t in closed_trades if t.sign < 0]

        # Long statistics
        if long_trades:
            stats.long_trades = len(long_trades)
            long_winning = [t for t in long_trades if float(t.profit) > 0]
            long_losing = [t for t in long_trades if float(t.profit) < 0]

            stats.long_winning_trades = len(long_winning)
            stats.long_net_profit = sum(float(t.profit) for t in long_trades)
            stats.long_net_profit_percent = (stats.long_net_profit / initial_capital) * 100

            if long_winning:
                stats.long_gross_profit = sum(float(t.profit) for t in long_winning)
                stats.long_gross_profit_percent = (stats.long_gross_profit / initial_capital) * 100
                max_long_win = max(long_winning, key=lambda t: float(t.profit))
                stats.long_largest_winning_trade = float(max_long_win.profit)
                stats.long_largest_winning_trade_percent = float(max_long_win.profit_percent)

            if long_losing:
                stats.long_gross_loss = sum(float(t.profit) for t in long_losing)
                stats.long_gross_loss_percent = (stats.long_gross_loss / initial_capital) * 100
                max_long_loss = min(long_losing, key=lambda t: float(t.profit))
                stats.long_largest_losing_trade = float(max_long_loss.profit)
                stats.long_largest_losing_trade_percent = float(max_long_loss.profit_percent)

            stats.long_avg_trade = stats.long_net_profit / len(long_trades)
            stats.long_avg_trade_percent = stats.long_net_profit_percent / len(long_trades)

            long_bars = [t.exit_bar_index - t.entry_bar_index for t in long_trades if t.exit_bar_index >= 0]
            if long_bars:
                stats.long_avg_bars = sum(long_bars) / len(long_bars)

        # Short statistics
        if short_trades:
            stats.short_trades = len(short_trades)
            short_winning = [t for t in short_trades if float(t.profit) > 0]
            short_losing = [t for t in short_trades if float(t.profit) < 0]

            stats.short_winning_trades = len(short_winning)
            stats.short_net_profit = sum(float(t.profit) for t in short_trades)
            stats.short_net_profit_percent = (stats.short_net_profit / initial_capital) * 100

            if short_winning:
                stats.short_gross_profit = sum(float(t.profit) for t in short_winning)
                stats.short_gross_profit_percent = (stats.short_gross_profit / initial_capital) * 100
                max_short_win = max(short_winning, key=lambda t: float(t.profit))
                stats.short_largest_winning_trade = float(max_short_win.profit)
                stats.short_largest_winning_trade_percent = float(max_short_win.profit_percent)

            if short_losing:
                stats.short_gross_loss = sum(float(t.profit) for t in short_losing)
                stats.short_gross_loss_percent = (stats.short_gross_loss / initial_capital) * 100
                max_short_loss = min(short_losing, key=lambda t: float(t.profit))
                stats.short_largest_losing_trade = float(max_short_loss.profit)
                stats.short_largest_losing_trade_percent = float(max_short_loss.profit_percent)

            stats.short_avg_trade = stats.short_net_profit / len(short_trades)
            stats.short_avg_trade_percent = stats.short_net_profit_percent / len(short_trades)

            short_bars = [t.exit_bar_index - t.entry_bar_index for t in short_trades if t.exit_bar_index >= 0]
            if short_bars:
                stats.short_avg_bars = sum(short_bars) / len(short_bars)

        # Max consecutive wins/losses
        if closed_trades:
            current_wins = 0
            current_losses = 0
            max_wins = 0
            max_losses = 0

            for trade in closed_trades:
                profit = float(trade.profit)
                if profit > 0:
                    current_wins += 1
                    current_losses = 0
                    max_wins = max(max_wins, current_wins)
                elif profit < 0:
                    current_losses += 1
                    current_wins = 0
                    max_losses = max(max_losses, current_losses)
                else:
                    current_wins = 0
                    current_losses = 0

            stats.max_cons_winning_trades = max_wins
            stats.max_cons_losing_trades = max_losses

    # Max contracts held
    if all_trades:
        max_size = 0.0
        current_positions: list[Trade] = []

        # Sort all trades by entry time
        sorted_trades = sorted(all_trades, key=lambda t: t.entry_time)

        for trade in sorted_trades:
            # Add to current positions
            current_positions.append(trade)

            # Remove closed positions that exit before this entry
            current_positions = [t for t in current_positions
                                 if t.exit_time < 0 or t.exit_time > trade.entry_time]

            # Calculate current size
            current_size = sum(abs(t.size) for t in current_positions)
            max_size = max(max_size, current_size)

        stats.max_contracts_held = max_size

    # Sharpe and Sortino ratios (if equity curve provided)
    if equity_curve and len(equity_curve) > 1:
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] != 0:
                ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                returns.append(ret)

        if returns:
            avg_return = sum(returns) / len(returns)

            # Sharpe ratio calculation
            if len(returns) > 1:
                variance = sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
                std_dev = math.sqrt(variance)
                if std_dev > 0:
                    # Annualized Sharpe ratio (assuming daily returns and 252 trading days)
                    stats.sharpe_ratio = (avg_return * 252) / (std_dev * math.sqrt(252))

            # Sortino ratio calculation
            downside_returns = [r for r in returns if r < 0]
            if len(downside_returns) > 1:
                downside_variance = sum(r ** 2 for r in downside_returns) / len(downside_returns)
                downside_std = math.sqrt(downside_variance)
                if downside_std > 0:
                    # Annualized Sortino ratio
                    stats.sortino_ratio = (avg_return * 252) / (downside_std * math.sqrt(252))

    return stats


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
