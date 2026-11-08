"""
Coin-flip tests: does the strategy beat the same trades with a random choice?

Two choices of a strategy can be randomized, one at a time:

- ``direction``: every entry stays at its moment and price, and a coin decides
  whether it goes long or short. Its stop and target are mirrored through the fill
  price, so they keep their distances. The test asks whether the signal knows
  where the price goes.
- ``timing``: every trade keeps its direction, size, stop distance and holding
  time, and enters at a random moment instead. The test asks whether the entry
  signal is better than entering blind. The script's own exits are replaced too:
  they are tied to its entry signals, and a random entry has none.

Both run the strategy in the engine, so the real and the random trades fill by the
same rules. Every random decision is a pure function of the seed, the draw and the
bar, so a run repeats exactly, and the re-executions of one bar (calc on order
fills, the bar magnifier) agree with each other.
"""

import hashlib
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

from .. import lib
from ..lib import strategy as _strategy
from .metrics import percentile
from .montecarlo import PERCENTILES
from .trades import TradeRecord, TradeRecorder, Unit, trade_value

if TYPE_CHECKING:
    from ..core.script_runner import ScriptRunner
    from ..core.syminfo import SymInfo
    from ..lib.strategy import SimPosition

Mode = Literal['direction', 'timing']

DEFAULT_DRAWS = 10_000
"""Draws of the direction test's fast path, where a draw costs no backtest."""
DEFAULT_RUNS = 200
"""Backtests of the tests that run the strategy once per draw."""
MIN_PAIRED_SHARE = 0.9
"""Share of the real trades the direction test's fast path must pair to give a verdict."""

_TIMING_ID = 'coinflip'

_FillKey = tuple[int, str | None]


@dataclass(slots=True)
class CoinFlipResult:
    """
    The real result against the random ones.

    :ivar mode: What was randomized: ``direction`` or ``timing``
    :ivar method: ``paired`` (two backtests, the draws combine their trades) or
        ``runs`` (one backtest per draw)
    :ivar unit: Unit of the totals
    :ivar draws: Number of random totals
    :ivar seed: Seed of the random decisions
    :ivar real: Total of the real trades the random ones are compared with
    :ivar trades: Real trades in the comparison
    :ivar distribution: Percentiles of the random totals
    :ivar percentile: Share of the random totals the real one beat, ties counted
        half (mid-rank)
    :ivar share_better: Share of the random totals above the real one
    :ivar totals: Every random total, in draw order
    :ivar unpaired: ``paired``: real trades the flipped backtest had no counterpart
        for, left out of the comparison
    :ivar valid: False when too few trades paired (see :data:`MIN_PAIRED_SHARE`):
        the result then gives no verdict, and its percentile and share are NaN
    :ivar not_placed: ``timing``: random entries that found no flat moment before
        the end of the data, per draw on average
    :ivar notes: Warnings about the parts of the script the test could not randomize
    :ivar seconds: Run time of the test
    """

    mode: Mode
    method: str
    unit: Unit
    draws: int
    seed: int
    real: float
    trades: int
    distribution: dict[str, float | None]
    percentile: float
    share_better: float
    totals: list[float] = field(default_factory=list)
    unpaired: int = 0
    valid: bool = True
    not_placed: float = 0.0
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def description(self) -> str:
        """:return: One sentence on what was randomized"""
        if self.mode == 'direction':
            return "Every entry kept, a coin flipped for its direction"
        return ("Every trade's direction, size, stop distance and holding time kept, "
                "its entry moment drawn at random")

    @property
    def verdict(self) -> str:
        """:return: The comparison in plain words"""
        if not self.valid:
            return (f"not valid for this script: only {self.trades} of "
                    f"{self.trades + self.unpaired} trades had a counterpart in the flipped "
                    f"backtest (at least {MIN_PAIRED_SHARE:.0%} needed)")
        return (f"{self.share_better:.0%} of the random {self.mode} draws did better; "
                f"the real rules beat {self.percentile:.0%} of them")

    def to_dict(self) -> dict[str, Any]:
        """:return: JSON-ready form, without the individual totals"""
        return {
            'mode': self.mode, 'method': self.method, 'unit': self.unit, 'draws': self.draws,
            'seed': self.seed, 'real': _finite(self.real), 'trades': self.trades,
            'distribution': self.distribution, 'percentile': _finite(self.percentile),
            'share_better': _finite(self.share_better), 'unpaired': self.unpaired,
            'valid': self.valid,
            'not_placed': self.not_placed, 'notes': list(self.notes),
            'description': self.description, 'verdict': self.verdict,
            'seconds': self.seconds,
        }


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def coin(seed: int, draw: int, *keys: Any) -> bool:
    """
    One fair coin, a pure function of its inputs.

    :param seed: Seed of the test
    :param draw: Number of the draw
    :param keys: What the coin decides about (a bar index, an entry id)
    :return: Heads or tails
    """
    text = ':'.join(str(k) for k in (seed, draw, *keys))
    return hashlib.blake2b(text.encode(), digest_size=8).digest()[0] & 1 == 1


def _compare(real: float, totals: list[float]) -> tuple[dict[str, float | None], float, float]:
    ordered = sorted(totals)
    n = len(ordered)
    dist: dict[str, float | None] = {}
    for q in PERCENTILES:
        value = percentile(ordered, q)
        dist[f"p{q:g}"] = value if math.isfinite(value) else None
    if not n:
        return dist, math.nan, math.nan
    below = sum(1 for v in ordered if v < real)
    equal = sum(1 for v in ordered if v == real)
    above = n - below - equal
    return dist, (below + 0.5 * equal) / n, above / n


def _fill_values(trades: list[TradeRecord], unit: Unit) -> dict[_FillKey, float]:
    """
    One value per entry fill: a partial exit splits a fill into several trades.

    :param trades: Closed trades
    :param unit: Unit of the values
    :return: Value per ``(entry bar, entry id)``; a fill without a value in the unit
        is left out
    """
    groups: dict[_FillKey, list[TradeRecord]] = {}
    for trade in trades:
        groups.setdefault((trade.entry_bar_index, trade.entry_id), []).append(trade)
    values: dict[_FillKey, float] = {}
    for key, pieces in groups.items():
        if unit == 'r':
            risk = math.fsum(t.risk for t in pieces if t.risk is not None)
            if len(pieces) == sum(1 for t in pieces if t.risk is not None) and risk > 0.0:
                values[key] = math.fsum(t.profit for t in pieces) / risk
        elif unit == 'currency':
            values[key] = math.fsum(t.profit for t in pieces)
        else:
            parts = [trade_value(t, unit) for t in pieces]
            if all(p is not None for p in parts):
                values[key] = math.fsum(p for p in parts if p is not None)
    return values


def _record(runner: 'ScriptRunner', syminfo: 'SymInfo',
            on_bar: Callable[[], None] | None = None) -> list[TradeRecord]:
    position = runner.script.position
    recorder = TradeRecorder(position, syminfo)
    for _ in runner.run_iter(copy_results=False):
        if on_bar is not None:
            on_bar()
        recorder.on_bar()
    return recorder.trades


# noinspection PyProtectedMember
def _flip_notes(flipper: '_strategy._EntryFlip') -> list[str]:
    notes: list[str] = []
    if flipper.stop_limit_entries:
        notes.append(f"{flipper.stop_limit_entries} stop-limit entry calls have no mirror image "
                     f"and kept their direction")
    if flipper.order_calls:
        notes.append(f"{flipper.order_calls} strategy.order calls were not flipped")
    return notes


def _run_flipped(make_runner: Callable[[], 'ScriptRunner'], syminfo: 'SymInfo',
                 decide: Callable[[str, int], bool]) -> tuple[list[TradeRecord], list[str]]:
    runner = make_runner()
    # noinspection PyProtectedMember
    flipper = _strategy._EntryFlip(decide)
    _strategy._entry_flip = flipper
    try:
        trades = _record(runner, syminfo)
    finally:
        _strategy._entry_flip = None
    return trades, _flip_notes(flipper)


def direction_test(make_runner: Callable[[], 'ScriptRunner'], syminfo: 'SymInfo',
                   real_trades: list[TradeRecord], unit: Unit, *, seed: int = 0,
                   draws: int = DEFAULT_DRAWS, exact: bool = False, runs: int = DEFAULT_RUNS,
                   on_progress: Callable[[int, int], None] | None = None) -> CoinFlipResult:
    """
    The real direction choices against coin flips on the same entries.

    The fast path runs one more backtest with every entry flipped and pairs its
    trades with the real ones by entry bar and id; each draw then picks the real or
    the flipped outcome of every trade by a coin. It assumes the trades do not
    depend on each other: a flipped trade that closes earlier, or a position size
    that compounds, can change the later entries, and those trades fall out of the
    pairing. ``exact`` runs a full backtest per draw instead, with a coin per entry,
    which keeps such paths.

    :param make_runner: Builds a fresh runner of the strategy on its data
    :param syminfo: Symbol information of the chart
    :param real_trades: The trades of the strategy's own backtest
    :param unit: Unit of the totals
    :param seed: Seed of the coins
    :param draws: Draws of the fast path
    :param exact: Run a backtest per draw
    :param runs: Backtests of the exact path
    :param on_progress: Called with the backtests done and the backtests in all
    :return: The comparison
    """
    started = time.perf_counter()
    real_values = _fill_values(real_trades, unit)

    if not exact:
        if on_progress is not None:
            on_progress(0, 1)
        flipped_trades, notes = _run_flipped(make_runner, syminfo, lambda _id, _bar: True)
        if on_progress is not None:
            on_progress(1, 1)
        flipped_values = _fill_values(flipped_trades, unit)
        keys = [k for k in real_values if k in flipped_values]
        pairs = [(real_values[k], flipped_values[k]) for k in keys]
        unpaired = len(real_values) - len(pairs)
        if unpaired:
            notes.append(f"{unpaired} of {len(real_values)} trades had no counterpart in the "
                         f"flipped backtest and were left out")
        if not real_values or len(pairs) < MIN_PAIRED_SHARE * len(real_values):
            # Too few pairs say nothing about the strategy's direction choices; the
            # exact path, a backtest per draw, needs no pairing
            notes.append("the trades of the flipped backtest depend on each other too much "
                         "for pairing: run the exact test (a backtest per draw) instead")
            empty: dict[str, float | None] = {f"p{q:g}": None for q in PERCENTILES}
            return CoinFlipResult(mode='direction', method='paired', unit=unit, draws=0,
                                  seed=seed, real=math.fsum(v for v, _ in pairs),
                                  trades=len(pairs), distribution=empty, percentile=math.nan,
                                  share_better=math.nan, unpaired=unpaired, valid=False,
                                  notes=notes, seconds=time.perf_counter() - started)
        rng = random.Random(seed)
        totals: list[float] = []
        for _ in range(draws):
            total = 0.0
            for real_value, flipped_value in pairs:
                total += flipped_value if rng.random() < 0.5 else real_value
            totals.append(total)
        real = math.fsum(v for v, _ in pairs)
        dist, rank, better = _compare(real, totals)
        return CoinFlipResult(mode='direction', method='paired', unit=unit, draws=len(totals),
                              seed=seed, real=real, trades=len(pairs), distribution=dist,
                              percentile=rank, share_better=better, totals=totals,
                              unpaired=unpaired, notes=notes,
                              seconds=time.perf_counter() - started)

    totals = []
    notes = []
    for draw in range(runs):
        trades, run_notes = _run_flipped(
            make_runner, syminfo,
            lambda entry_id, bar, d=draw: coin(seed, d, bar, entry_id))
        if not notes:
            notes = run_notes
        totals.append(math.fsum(_fill_values(trades, unit).values()))
        if on_progress is not None:
            on_progress(draw + 1, runs)
    real = math.fsum(real_values.values())
    dist, rank, better = _compare(real, totals)
    return CoinFlipResult(mode='direction', method='runs', unit=unit, draws=len(totals),
                          seed=seed, real=real, trades=len(real_values), distribution=dist,
                          percentile=rank, share_better=better, totals=totals, notes=notes,
                          seconds=time.perf_counter() - started)


@dataclass(slots=True, frozen=True)
class _Plan:
    """One real trade as the timing test re-enters it."""

    direction: int
    size: float
    risk_fraction: float | None
    hold_bars: int


def _plans(trades: list[TradeRecord]) -> list[_Plan]:
    """
    The real trades as entry fills, in entry order.

    :param trades: Closed trades; the pieces of one fill are merged
    :return: One plan per fill
    """
    groups: dict[_FillKey, list[TradeRecord]] = {}
    for trade in trades:
        groups.setdefault((trade.entry_bar_index, trade.entry_id), []).append(trade)
    plans: list[_Plan] = []
    for (entry_bar, _), pieces in sorted(groups.items(), key=lambda item: item[0][0]):
        first = pieces[0]
        stop = first.initial_stop
        risk_fraction = (abs(first.entry_price - stop) / first.entry_price
                         if stop is not None and first.entry_price > 0.0 else None)
        plans.append(_Plan(direction=first.direction,
                           size=math.fsum(t.size for t in pieces),
                           risk_fraction=risk_fraction,
                           hold_bars=max(t.exit_bar_index for t in pieces) - entry_bar))
    return plans


class _TimingDriver:
    """
    Trades the plans at random moments, one position at a time.

    Called after every bar, once its orders are settled. An order placed here
    executes from the next bar on, like one the script places in its own run.
    """

    __slots__ = ('_plans', '_schedule', '_position', '_mintick', '_next', '_open_plan',
                 '_entry_bar', '_closing')

    def __init__(self, plans: list[_Plan], schedule: list[int], position: 'SimPosition',
                 mintick: float):
        self._plans = plans
        self._schedule = schedule
        self._position = position
        self._mintick = mintick
        self._next = 0
        self._open_plan: _Plan | None = None
        self._entry_bar = -1
        self._closing = False

    @property
    def not_placed(self) -> int:
        """:return: Plans that never entered"""
        return len(self._plans) - self._next

    def on_bar(self) -> None:
        """Manage the open trade, or enter the next plan when its moment has come."""
        position = self._position
        bar = int(lib.bar_index)
        _strategy._orders_blocked = False
        try:
            if position.open_trades:
                plan = self._open_plan
                if plan is not None and not self._closing:
                    if self._entry_bar < 0:
                        self._entry_bar = int(position.open_trades[0].entry_bar_index)
                    # The close placed now fills on the next bar's open
                    if bar + 1 - self._entry_bar >= plan.hold_bars:
                        _strategy.close(_TIMING_ID)
                        self._closing = True
                return
            self._open_plan = None
            self._closing = False
            self._entry_bar = -1
            if position.entry_orders or self._next >= len(self._plans):
                return
            if self._schedule[self._next] > bar:
                return
            plan = self._plans[self._next]
            self._next += 1
            self._open_plan = plan
            _strategy.entry(_TIMING_ID, _strategy.long if plan.direction > 0 else _strategy.short,
                            qty=plan.size)
            if plan.risk_fraction is not None and self._mintick > 0.0:
                # The real trade's stop distance, as a share of the price
                ticks = plan.risk_fraction * float(lib.close) / self._mintick
                _strategy.exit(_TIMING_ID + 'x', _TIMING_ID, loss=ticks)
        finally:
            _strategy._orders_blocked = True


def timing_test(make_runner: Callable[[], 'ScriptRunner'], syminfo: 'SymInfo',
                real_trades: list[TradeRecord], unit: Unit, *, seed: int = 0,
                runs: int = DEFAULT_RUNS,
                on_progress: Callable[[int, int], None] | None = None) -> CoinFlipResult:
    """
    The real entry moments against random ones.

    Each draw is a backtest in which the script places no order. The real trades
    enter in their own order, each at a random bar between the first real entry and
    the last real exit, once the previous one has closed: same direction and size,
    a stop at the real trade's stop distance (as a share of the price) and a close
    after the real trade's holding time. A trade without a stop keeps only the
    holding time.

    :param make_runner: Builds a fresh runner of the strategy on its data
    :param syminfo: Symbol information of the chart
    :param real_trades: The trades of the strategy's own backtest
    :param unit: Unit of the totals
    :param seed: Seed of the random moments
    :param runs: Backtests, one per draw
    :param on_progress: Called with the backtests done and the backtests in all
    :return: The comparison
    """
    started = time.perf_counter()
    plans = _plans(real_trades)
    real_values = _fill_values(real_trades, unit)
    first_bar = min(t.entry_bar_index for t in real_trades)
    last_bar = max(t.exit_bar_index for t in real_trades)
    span = range(first_bar, max(first_bar + 1, last_bar))
    count = min(len(plans), len(span))
    mintick = float(syminfo.mintick or 0.0)

    totals: list[float] = []
    not_placed = 0
    for draw in range(runs):
        rng = random.Random(f"{seed}:{draw}")
        schedule = sorted(rng.sample(span, count))
        runner = make_runner()
        driver = _TimingDriver(plans[:count], schedule, runner.script.position, mintick)
        _strategy._orders_blocked = True
        try:
            trades = _record(runner, syminfo, driver.on_bar)
        finally:
            _strategy._orders_blocked = False
        not_placed += driver.not_placed
        totals.append(math.fsum(_fill_values(trades, unit).values()))
        if on_progress is not None:
            on_progress(draw + 1, runs)

    real = math.fsum(real_values.values())
    dist, rank, better = _compare(real, totals)
    notes: list[str] = []
    if any(p.risk_fraction is None for p in plans):
        notes.append("trades without a stop were re-entered with their holding time only")
    average_missing = not_placed / runs if runs else 0.0
    if average_missing:
        notes.append(f"on average {average_missing:.1f} random entries found no flat moment "
                     f"before the end of the data")
    return CoinFlipResult(mode='timing', method='runs', unit=unit, draws=len(totals), seed=seed,
                          real=real, trades=len(real_values), distribution=dist, percentile=rank,
                          share_better=better, totals=totals, not_placed=average_missing,
                          notes=notes, seconds=time.perf_counter() - started)


def coinflip_test(mode: Mode, make_runner: Callable[[], 'ScriptRunner'], syminfo: 'SymInfo',
                  real_trades: list[TradeRecord], unit: Unit, *, seed: int = 0,
                  draws: int = DEFAULT_DRAWS, exact: bool = False, runs: int = DEFAULT_RUNS,
                  on_progress: Callable[[int, int], None] | None = None) -> CoinFlipResult:
    """
    Run one coin-flip test.

    :param mode: ``direction`` or ``timing``
    :param make_runner: Builds a fresh runner of the strategy on its data
    :param syminfo: Symbol information of the chart
    :param real_trades: The trades of the strategy's own backtest
    :param unit: Unit of the totals
    :param seed: Seed of the random decisions
    :param draws: Draws of the direction test's fast path
    :param exact: Direction test: run a backtest per draw
    :param runs: Backtests of the tests that run one per draw
    :param on_progress: Called with the backtests done and the backtests in all
    :return: The comparison
    """
    if mode == 'direction':
        return direction_test(make_runner, syminfo, real_trades, unit, seed=seed, draws=draws,
                              exact=exact, runs=runs, on_progress=on_progress)
    return timing_test(make_runner, syminfo, real_trades, unit, seed=seed, runs=runs,
                       on_progress=on_progress)

