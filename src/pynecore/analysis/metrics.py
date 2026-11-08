"""
Summary statistics of a sequence of per-trade values.

The values are additive: R multiples, account currency, or percent returns taken as
a fixed-size series. The drawdown is measured on the running sum of the closed
trades, so it is not the intrabar equity drawdown of the strategy report.
"""

import math
import statistics
from dataclasses import dataclass, asdict
from typing import Any

TOP_TRADES = 3
"""Number of best trades the concentration figures take out."""


@dataclass(slots=True, frozen=True)
class Summary:
    """
    Statistics of one sequence of per-trade values.

    :ivar trades: Number of values
    :ivar wins: Values above zero
    :ivar losses: Values below zero
    :ivar hit_rate: ``wins / trades``, ``nan`` without trades
    :ivar total: Sum of the values
    :ivar mean: Mean value, ``nan`` without trades
    :ivar median: Median value, ``nan`` without trades
    :ivar stdev: Sample standard deviation, ``nan`` under two trades
    :ivar profit_factor: Sum of the wins over the sum of the losses, ``inf`` without
        losses, ``nan`` without trades
    :ivar max_drawdown: Largest fall of the running sum from a previous peak (the
        start counts as a peak at zero), a non-negative number
    :ivar largest_win: Largest value, ``nan`` without trades
    :ivar largest_loss: Smallest value, ``nan`` without trades
    :ivar top_total: Sum of the :data:`TOP_TRADES` largest values
    :ivar longest_losing_streak: Most consecutive values below zero
    """

    trades: int
    wins: int
    losses: int
    hit_rate: float
    total: float
    mean: float
    median: float
    stdev: float
    profit_factor: float
    max_drawdown: float
    largest_win: float
    largest_loss: float
    top_total: float
    longest_losing_streak: int

    @property
    def total_without_top(self) -> float:
        """
        The total without the :data:`TOP_TRADES` largest values.

        A result carried by a few outliers falls far here.

        :return: ``total - top_total``
        """
        return self.total - self.top_total

    def to_dict(self) -> dict[str, Any]:
        """
        :return: The fields and the derived figures, JSON-ready (``nan`` and ``inf``
            become ``None``)
        """
        data = asdict(self)
        data['total_without_top'] = self.total_without_top
        return {k: _finite_or_none(v) for k, v in data.items()}


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def max_drawdown(values: list[float]) -> float:
    """
    Largest fall of the running sum from a previous peak.

    :param values: Per-trade values in trade order
    :return: The drawdown as a non-negative number; the start counts as a peak at zero
    """
    peak = 0.0
    equity = 0.0
    worst = 0.0
    for value in values:
        equity += value
        if equity > peak:
            peak = equity
        elif peak - equity > worst:
            worst = peak - equity
    return worst


def summarize(values: list[float]) -> Summary:
    """
    Summarize per-trade values.

    :param values: Per-trade values in trade order
    :return: The statistics
    """
    n = len(values)
    nan = math.nan
    if n == 0:
        return Summary(trades=0, wins=0, losses=0, hit_rate=nan, total=0.0, mean=nan,
                       median=nan, stdev=nan, profit_factor=nan, max_drawdown=0.0,
                       largest_win=nan, largest_loss=nan, top_total=0.0,
                       longest_losing_streak=0)

    wins = 0
    losses = 0
    gains = 0.0
    pains = 0.0
    streak = 0
    longest = 0
    for value in values:
        if value > 0.0:
            wins += 1
            gains += value
            streak = 0
        elif value < 0.0:
            losses += 1
            pains -= value
            streak += 1
            if streak > longest:
                longest = streak
        else:
            streak = 0

    total = math.fsum(values)
    ordered = sorted(values, reverse=True)
    return Summary(
        trades=n,
        wins=wins,
        losses=losses,
        hit_rate=wins / n,
        total=total,
        mean=total / n,
        median=statistics.median(values),
        stdev=statistics.stdev(values) if n > 1 else nan,
        profit_factor=gains / pains if pains > 0.0 else math.inf,
        max_drawdown=max_drawdown(values),
        largest_win=ordered[0],
        largest_loss=ordered[-1],
        top_total=math.fsum(ordered[:TOP_TRADES]),
        longest_losing_streak=longest,
    )


def percentile(ordered: list[float], q: float) -> float:
    """
    Percentile of sorted values, by linear interpolation between the closest ranks.

    :param ordered: Values in ascending order
    :param q: The percentile, ``0`` to ``100``
    :return: The value at the percentile, ``nan`` without values
    """
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight
