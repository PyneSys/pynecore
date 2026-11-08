"""
Monte Carlo resampling of the trade sequence.

One backtest is one path through its trades. Resampling the trades gives many paths
with the same per-trade results, which shows how much of the drawdown and of the
final result came from the order the trades happened to arrive in:

- ``shuffle`` keeps every trade exactly once and only changes the order. The
  additive total stays the same; the drawdown and the path change.
- ``bootstrap`` draws as many trades as there were, with replacement, so the total
  varies as well.

Both treat the trades as independent and describe this backtest, not the future.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Literal

from .metrics import percentile

Method = Literal['shuffle', 'bootstrap']

PERCENTILES: tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0)
"""Percentiles reported for every distribution and for the path bands."""

DEFAULT_DRAWS = 10_000
MAX_BAND_POINTS = 100
"""Most trade positions the path bands are reported at."""


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _percentiles(values: list[float]) -> dict[str, float | None]:
    ordered = sorted(values)
    return {f"p{q:g}": _finite(percentile(ordered, q)) for q in PERCENTILES}


@dataclass(slots=True)
class MonteCarloResult:
    """
    Distribution of resampled trade paths.

    In the additive form a path is the running sum of the per-trade values. In the
    account form it is an account that compounds the per-trade returns from
    ``initial_capital``; ``ruin`` and ``target`` are levels of that account.

    :ivar method: The resampling method
    :ivar draws: Number of paths
    :ivar seed: Seed of the random generator
    :ivar trades: Trades per path
    :ivar compound: True for the account form
    :ivar initial_capital: Starting account of the account form, ``0.0`` otherwise
    :ivar final: Percentiles of the path's final value
    :ivar max_drawdown: Percentiles of the path's largest fall from a previous peak;
        a fraction of the peak in the account form
    :ivar share_losing: Share of paths that ended at or below their start
    :ivar ruin_level: Account level that ends a path as ruined, account form only
    :ivar share_ruined: Share of paths that touched the ruin level
    :ivar target_level: Account level counted as reaching the target, account form only
    :ivar share_target: Share of paths that reached the target before any ruin
    :ivar bands: Path percentiles at evenly spaced trade positions:
        ``(trade_number, {percentile: value})``
    """

    method: Method
    draws: int
    seed: int
    trades: int
    compound: bool
    initial_capital: float
    final: dict[str, float | None]
    max_drawdown: dict[str, float | None]
    share_losing: float
    ruin_level: float | None = None
    share_ruined: float | None = None
    target_level: float | None = None
    share_target: float | None = None
    bands: list[tuple[int, dict[str, float | None]]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """:return: JSON-ready form"""
        return {
            'method': self.method,
            'draws': self.draws,
            'seed': self.seed,
            'trades': self.trades,
            'compound': self.compound,
            'initial_capital': self.initial_capital,
            'final': self.final,
            'max_drawdown': self.max_drawdown,
            'share_losing': self.share_losing,
            'ruin_level': self.ruin_level,
            'share_ruined': self.share_ruined,
            'target_level': self.target_level,
            'share_target': self.share_target,
            'bands': [{'trade': n, **values} for n, values in self.bands],
        }


def _band_positions(n: int) -> list[int]:
    """
    Trade numbers the bands are reported at: every one, or evenly spaced ones.

    :param n: Trades per path
    :return: Ascending trade numbers ``1..n``, the last one always included
    """
    if n <= MAX_BAND_POINTS:
        return list(range(1, n + 1))
    step = n / MAX_BAND_POINTS
    return sorted({max(1, round(step * (i + 1))) for i in range(MAX_BAND_POINTS)})


def monte_carlo(values: list[float], *, method: Method = 'bootstrap', draws: int = DEFAULT_DRAWS,
                seed: int = 0, compound: bool = False, initial_capital: float = 1.0,
                ruin_fraction: float | None = None,
                target_multiple: float | None = None) -> MonteCarloResult:
    """
    Resample the trade sequence.

    :param values: Per-trade values in trade order. Additive form: the values
        themselves (R, currency, percent). Account form: per-trade returns as
        fractions of the account (``0.01`` is +1%).
    :param method: ``shuffle`` or ``bootstrap``
    :param draws: Number of paths
    :param seed: Seed of the random generator
    :param compound: Compound the values as returns of an account
    :param initial_capital: Starting account of the account form
    :param ruin_fraction: Account form: the share of the starting account whose loss
        ends a path as ruined (``0.5``: the account fell to half). ``None`` for no
        ruin level.
    :param target_multiple: Account form: the multiple of the starting account
        counted as reaching the target (``100``: $100 to $10,000). ``None`` for no
        target.
    :return: The distributions
    """
    n = len(values)
    rng = random.Random(seed)
    positions = _band_positions(n)
    columns: list[list[float]] = [[] for _ in positions]
    finals: list[float] = []
    drawdowns: list[float] = []
    losing = 0
    ruined = 0
    reached = 0
    start = initial_capital if compound else 0.0
    ruin_level = (initial_capital * (1.0 - ruin_fraction)
                  if compound and ruin_fraction is not None else None)
    target_level = (initial_capital * target_multiple
                    if compound and target_multiple is not None else None)
    order = list(values)

    for _ in range(draws if n else 0):
        if method == 'shuffle':
            rng.shuffle(order)
            path = order
        else:
            path = rng.choices(values, k=n)
        equity = start
        peak = start
        worst = 0.0
        is_ruined = False
        is_reached = False
        column = 0
        for number, value in enumerate(path, 1):
            if not is_ruined:
                if compound:
                    equity *= 1.0 + value
                    if equity < 0.0:
                        equity = 0.0
                else:
                    equity += value
                if equity > peak:
                    peak = equity
                else:
                    fall = ((peak - equity) / peak if peak > 0.0 else 0.0) if compound \
                        else peak - equity
                    if fall > worst:
                        worst = fall
                if target_level is not None and not is_reached and equity >= target_level:
                    is_reached = True
                if ruin_level is not None and equity <= ruin_level:
                    # A ruined account stops trading and keeps its level
                    is_ruined = True
            if column < len(positions) and number == positions[column]:
                columns[column].append(equity)
                column += 1
        finals.append(equity)
        drawdowns.append(worst)
        if equity <= start:
            losing += 1
        if is_ruined:
            ruined += 1
        # A ruined path stops trading, so a reached target was reached before the ruin
        if is_reached:
            reached += 1

    used = len(finals)
    return MonteCarloResult(
        method=method,
        draws=used,
        seed=seed,
        trades=n,
        compound=compound,
        initial_capital=initial_capital if compound else 0.0,
        final=_percentiles(finals),
        max_drawdown=_percentiles(drawdowns),
        share_losing=losing / used if used else math.nan,
        ruin_level=ruin_level,
        share_ruined=ruined / used if used and ruin_level is not None else None,
        target_level=target_level,
        share_target=reached / used if used and target_level is not None else None,
        bands=[(position, _percentiles(column)) for position, column in zip(positions, columns)],
    )
