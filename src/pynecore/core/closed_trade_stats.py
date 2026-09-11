"""Fixed-size cumulative statistics for finalized closed trades."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..lib.strategy import Trade


class _SideClosedTradeStats:
    """Cumulative closed-trade statistics for one position direction."""

    __slots__ = (
        "count",
        "winning_count",
        "net_profit",
        "gross_profit",
        "gross_loss",
        "largest_winning_trade",
        "largest_winning_trade_percent",
        "largest_losing_trade",
        "largest_losing_trade_percent",
        "bar_length_sum",
        "bar_length_count",
    )

    def __init__(self) -> None:
        self.count: int = 0
        self.winning_count: int = 0
        self.net_profit: float = 0.0
        self.gross_profit: float = 0.0
        self.gross_loss: float = 0.0
        self.largest_winning_trade: float = 0.0
        self.largest_winning_trade_percent: float = 0.0
        self.largest_losing_trade: float = 0.0
        self.largest_losing_trade_percent: float = 0.0
        self.bar_length_sum: int = 0
        self.bar_length_count: int = 0

    def add(self, profit: float, profit_percent: float, bar_length: int | None) -> None:
        """Add one finalized trade from this direction."""
        self.count += 1
        self.net_profit += profit

        if profit > 0.0:
            self.winning_count += 1
            self.gross_profit += profit
            if profit > self.largest_winning_trade:
                self.largest_winning_trade = profit
                self.largest_winning_trade_percent = profit_percent
        elif profit < 0.0:
            self.gross_loss += profit
            if profit < self.largest_losing_trade:
                self.largest_losing_trade = profit
                self.largest_losing_trade_percent = profit_percent

        if bar_length is not None:
            self.bar_length_sum += bar_length
            self.bar_length_count += 1


class ClosedTradeStats:
    """Fixed-size cumulative statistics for finalized closed trades."""

    __slots__ = (
        "commission",
        "winning_count",
        "losing_count",
        "winning_profit_sum",
        "losing_profit_sum",
        "largest_winning_trade",
        "largest_winning_trade_percent",
        "largest_losing_trade",
        "largest_losing_trade_percent",
        "bar_length_sum",
        "bar_length_count",
        "winning_bar_length_sum",
        "winning_bar_length_count",
        "losing_bar_length_sum",
        "losing_bar_length_count",
        "current_winning_streak",
        "current_losing_streak",
        "max_winning_streak",
        "max_losing_streak",
        "long",
        "short",
    )

    def __init__(self) -> None:
        self.commission: float = 0.0
        self.winning_count: int = 0
        self.losing_count: int = 0
        self.winning_profit_sum: float = 0.0
        self.losing_profit_sum: float = 0.0
        self.largest_winning_trade: float = 0.0
        self.largest_winning_trade_percent: float = 0.0
        self.largest_losing_trade: float = 0.0
        self.largest_losing_trade_percent: float = 0.0
        self.bar_length_sum: int = 0
        self.bar_length_count: int = 0
        self.winning_bar_length_sum: int = 0
        self.winning_bar_length_count: int = 0
        self.losing_bar_length_sum: int = 0
        self.losing_bar_length_count: int = 0
        self.current_winning_streak: int = 0
        self.current_losing_streak: int = 0
        self.max_winning_streak: int = 0
        self.max_losing_streak: int = 0
        self.long = _SideClosedTradeStats()
        self.short = _SideClosedTradeStats()

    def add(self, trade: 'Trade') -> None:
        """Add a finalized closed trade to the cumulative statistics."""
        profit = float(trade.profit)
        profit_percent = float(trade.profit_percent)
        self.commission += float(trade.commission)

        bar_length: int | None = None
        if trade.exit_bar_index >= 0:
            bar_length = trade.exit_bar_index - trade.entry_bar_index
            self.bar_length_sum += bar_length
            self.bar_length_count += 1

        if profit > 0.0:
            self.winning_count += 1
            self.winning_profit_sum += profit
            self.current_winning_streak += 1
            self.current_losing_streak = 0
            if self.current_winning_streak > self.max_winning_streak:
                self.max_winning_streak = self.current_winning_streak
            if profit > self.largest_winning_trade:
                self.largest_winning_trade = profit
                self.largest_winning_trade_percent = profit_percent
            if bar_length is not None:
                self.winning_bar_length_sum += bar_length
                self.winning_bar_length_count += 1
        elif profit < 0.0:
            self.losing_count += 1
            self.losing_profit_sum += profit
            self.current_losing_streak += 1
            self.current_winning_streak = 0
            if self.current_losing_streak > self.max_losing_streak:
                self.max_losing_streak = self.current_losing_streak
            if profit < self.largest_losing_trade:
                self.largest_losing_trade = profit
                self.largest_losing_trade_percent = profit_percent
            if bar_length is not None:
                self.losing_bar_length_sum += bar_length
                self.losing_bar_length_count += 1
        else:
            self.current_winning_streak = 0
            self.current_losing_streak = 0

        if trade.sign > 0.0:
            self.long.add(profit, profit_percent, bar_length)
        elif trade.sign < 0.0:
            self.short.add(profit, profit_percent, bar_length)
