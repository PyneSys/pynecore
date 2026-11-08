"""
Confidence intervals for the mean trade and the hit rate.

Trades of one strategy are rarely independent: they cluster in the weeks the market
suits the strategy. The block bootstrap resamples whole calendar blocks of trades,
quiet blocks included, so the interval keeps that clustering. It does not cover
dependence longer than one block, such as a market regime of several months.
"""

import math
import random
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, UTC
from typing import Any, Literal

from .metrics import percentile

Block = Literal['trade', 'week', 'month']
"""Resampling block: single trades (independent), calendar weeks, or calendar months."""

DEFAULT_DRAWS = 10_000
DEFAULT_CONFIDENCE = 0.95


@dataclass(slots=True, frozen=True)
class BootstrapResult:
    """
    Bootstrap interval of the mean value per trade.

    :ivar block: The resampling block
    :ivar blocks: Number of blocks the trades fell into, empty ones included
    :ivar draws: Number of resamples drawn
    :ivar used_draws: Resamples that contained at least one trade
    :ivar seed: Seed of the random generator
    :ivar confidence: Coverage of the interval
    :ivar mean: Mean value per trade of the original trades
    :ivar low: Lower end of the interval
    :ivar high: Upper end of the interval
    :ivar share_not_above_zero: Share of the resampled means at or below zero
    """

    block: Block
    blocks: int
    draws: int
    used_draws: int
    seed: int
    confidence: float
    mean: float
    low: float
    high: float
    share_not_above_zero: float

    @property
    def excludes_zero(self) -> bool:
        """:return: True when the whole interval is on one side of zero"""
        return self.low > 0.0 or self.high < 0.0

    def to_dict(self) -> dict[str, Any]:
        """:return: JSON-ready form"""
        data = asdict(self)
        data['excludes_zero'] = self.excludes_zero
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                for k, v in data.items()}


@dataclass(slots=True, frozen=True)
class WilsonInterval:
    """
    Wilson score interval of a hit rate.

    :ivar wins: Number of winning trades
    :ivar trades: Number of trades
    :ivar confidence: Coverage of the interval
    :ivar rate: The measured hit rate
    :ivar low: Lower end of the interval
    :ivar high: Upper end of the interval
    """

    wins: int
    trades: int
    confidence: float
    rate: float
    low: float
    high: float

    def to_dict(self) -> dict[str, Any]:
        """:return: JSON-ready form"""
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                for k, v in asdict(self).items()}


def _block_start(time_ms: int, block: Block) -> datetime:
    moment = datetime.fromtimestamp(time_ms / 1000.0, UTC)
    day = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    if block == 'week':
        return day - timedelta(days=day.weekday())
    return day.replace(day=1)


def _next_block(start: datetime, block: Block) -> datetime:
    if block == 'week':
        return start + timedelta(days=7)
    if start.month == 12:
        return start.replace(year=start.year + 1, month=1)
    return start.replace(month=start.month + 1)


def _blocks(values: list[float], times: list[int], block: Block) -> list[tuple[float, int]]:
    """
    Group the values into consecutive calendar blocks.

    :param values: Per-trade values
    :param times: Time of each trade, Unix milliseconds
    :param block: The block kind
    :return: ``(sum, count)`` per block from the first trade's block to the last
        one's, empty blocks included
    """
    if block == 'trade':
        return [(v, 1) for v in values]
    grouped: dict[datetime, list[float]] = {}
    for value, time_ms in zip(values, times):
        grouped.setdefault(_block_start(time_ms, block), []).append(value)
    if not grouped:
        return []
    result: list[tuple[float, int]] = []
    current = min(grouped)
    last = max(grouped)
    while current <= last:
        members = grouped.get(current, [])
        result.append((math.fsum(members), len(members)))
        current = _next_block(current, block)
    return result


def block_bootstrap_mean(values: list[float], times: list[int], *, block: Block = 'week',
                         draws: int = DEFAULT_DRAWS, seed: int = 0,
                         confidence: float = DEFAULT_CONFIDENCE) -> BootstrapResult:
    """
    Bootstrap interval of the mean value per trade.

    Every resample draws as many blocks as there are, with replacement; its statistic
    is the sum of the drawn values over the number of drawn trades. A resample
    without any trade has no mean and is not counted.

    :param values: Per-trade values in trade order
    :param times: Time of each trade (the exit time), Unix milliseconds
    :param block: The resampling block
    :param draws: Number of resamples
    :param seed: Seed of the random generator
    :param confidence: Coverage of the interval, between 0 and 1
    :return: The interval
    """
    blocks = _blocks(values, times, block)
    mean = math.fsum(values) / len(values) if values else math.nan
    rng = random.Random(seed)
    k = len(blocks)
    means: list[float] = []
    if k:
        for _ in range(draws):
            total = 0.0
            count = 0
            for block_sum, block_count in rng.choices(blocks, k=k):
                total += block_sum
                count += block_count
            if count:
                means.append(total / count)
    means.sort()
    tail = (1.0 - confidence) / 2.0 * 100.0
    not_above = sum(1 for m in means if m <= 0.0)
    return BootstrapResult(
        block=block,
        blocks=k,
        draws=draws,
        used_draws=len(means),
        seed=seed,
        confidence=confidence,
        mean=mean,
        low=percentile(means, tail),
        high=percentile(means, 100.0 - tail),
        share_not_above_zero=not_above / len(means) if means else math.nan,
    )


def wilson_interval(wins: int, trades: int,
                    confidence: float = DEFAULT_CONFIDENCE) -> WilsonInterval:
    """
    Wilson score interval of a hit rate.

    Unlike the normal approximation it stays inside 0..1 and holds up with few trades
    and with rates near 0 or 1.

    :param wins: Number of winning trades
    :param trades: Number of trades
    :param confidence: Coverage of the interval, between 0 and 1
    :return: The interval
    """
    if trades == 0:
        nan = math.nan
        return WilsonInterval(wins=0, trades=0, confidence=confidence, rate=nan, low=nan, high=nan)
    z = statistics.NormalDist().inv_cdf(0.5 + confidence / 2.0)
    rate = wins / trades
    z2 = z * z
    denominator = 1.0 + z2 / trades
    center = (rate + z2 / (2.0 * trades)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / trades + z2 / (4.0 * trades * trades)) / denominator
    return WilsonInterval(wins=wins, trades=trades, confidence=confidence, rate=rate,
                          low=max(0.0, center - margin), high=min(1.0, center + margin))
