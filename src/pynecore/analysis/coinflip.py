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

The tests that run a backtest per draw can spread the draws over worker processes
(``jobs``) and stop once their verdict is clear (``early_stop``). A draw does not
depend on the others or on the worker that runs it, and the stop is decided on the
draws in draw order, so the result is the same with any number of workers.
"""

import hashlib
import math
import multiprocessing.context
import os
import pickle
import random
import signal
import sys
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from multiprocessing.pool import Pool
from types import FrameType
from typing import Any, Literal, TYPE_CHECKING

from .. import lib
from ..lib import strategy as _strategy
from .bootstrap import wilson_interval
from .metrics import percentile
from .montecarlo import PERCENTILES
from .trades import TradeRecord, TradeRecorder, Unit, trade_value

if TYPE_CHECKING:
    from ..core.script_runner import ScriptRunner
    from ..core.syminfo import SymInfo
    from ..lib.strategy import SimPosition

Mode = Literal['direction', 'timing']
EarlyStop = Literal['edge', 'better', 'off']

DEFAULT_DRAWS = 10_000
"""Draws of the direction test's fast path, where a draw costs no backtest."""
DEFAULT_RUNS = 200
"""Backtests of the tests that run the strategy once per draw."""
MIN_PAIRED_SHARE = 0.9
"""Share of the real trades the direction test's fast path must pair to give a verdict."""
STOP_MARKS: dict[str, float] = {'edge': 0.95, 'better': 0.5}
"""
Share of the random draws the real result is tested against, per early-stop rule:
``edge`` stops once it is clear whether the real rules beat 95% of the random draws,
``better`` once it is clear whether they beat half of them.
"""
STOP_MIN_RUNS = 30
"""Backtests before the first early-stop check."""
STOP_EVERY = 10
"""Backtests between two early-stop checks."""
STOP_CONFIDENCE = 0.99
"""
Coverage of the interval an early stop needs to be clear of the mark. It is above
the usual 95% because the interval is checked again and again.
"""

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
    :ivar planned: ``runs``: backtests the test runs without an early stop
    :ivar stop_reason: ``runs``: why the test stopped before ``planned`` backtests,
        None when it ran them all
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
    planned: int = 0
    stop_reason: str | None = None
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
        verdict = (f"{self.share_better:.0%} of the random {self.mode} draws did better; "
                   f"the real rules beat {self.percentile:.0%} of them")
        if self.stop_reason is not None:
            verdict += (f" (stopped after {self.draws} of {self.planned} backtests: "
                        f"{self.stop_reason})")
        return verdict

    def to_dict(self) -> dict[str, Any]:
        """:return: JSON-ready form, without the individual totals"""
        return {
            'mode': self.mode, 'method': self.method, 'unit': self.unit, 'draws': self.draws,
            'seed': self.seed, 'real': _finite(self.real), 'trades': self.trades,
            'distribution': self.distribution, 'percentile': _finite(self.percentile),
            'share_better': _finite(self.share_better), 'unpaired': self.unpaired,
            'valid': self.valid, 'not_placed': self.not_placed, 'planned': self.planned,
            'stop_reason': self.stop_reason, 'notes': list(self.notes),
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
                   jobs: int = 1, early_stop: EarlyStop = 'off',
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
    :param jobs: Worker processes of the exact path (see :func:`_run_draws`)
    :param early_stop: Early-stop rule of the exact path (see :data:`STOP_MARKS`)
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

    real = math.fsum(real_values.values())
    task = _DrawTask(mode='direction', make_runner=make_runner, syminfo=syminfo, unit=unit,
                     seed=seed)
    results, stop_reason = _run_draws(task, real, runs, jobs=jobs, early_stop=early_stop,
                                      on_progress=on_progress)
    totals = [r.total for r in results]
    notes = next((r.notes for r in results if r.notes), [])
    dist, rank, better = _compare(real, totals)
    return CoinFlipResult(mode='direction', method='runs', unit=unit, draws=len(totals),
                          seed=seed, real=real, trades=len(real_values), distribution=dist,
                          percentile=rank, share_better=better, totals=totals, planned=runs,
                          stop_reason=stop_reason, notes=notes,
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


@dataclass(slots=True)
class _DrawTask:
    """
    What one draw of a test that runs a backtest per draw needs.

    It travels to the worker processes, so with more than one job every field,
    ``make_runner`` included, must be picklable.
    """

    mode: Mode
    make_runner: Callable[[], 'ScriptRunner']
    syminfo: 'SymInfo'
    unit: Unit
    seed: int
    plans: list[_Plan] = field(default_factory=list)
    span: range = range(0)
    mintick: float = 0.0


@dataclass(slots=True)
class _DrawResult:
    """The outcome of one draw."""

    total: float
    notes: list[str] = field(default_factory=list)
    not_placed: int = 0


def _run_draw(task: _DrawTask, draw: int) -> _DrawResult:
    """
    Run the backtest of one draw.

    :param task: The test
    :param draw: Number of the draw; with the seed it decides every random choice
    :return: The draw's total
    """
    if task.mode == 'direction':
        seed = task.seed
        trades, notes = _run_flipped(task.make_runner, task.syminfo,
                                     lambda entry_id, bar: coin(seed, draw, bar, entry_id))
        return _DrawResult(math.fsum(_fill_values(trades, task.unit).values()), notes)

    rng = random.Random(f"{task.seed}:{draw}")
    schedule = sorted(rng.sample(task.span, len(task.plans)))
    runner = task.make_runner()
    driver = _TimingDriver(task.plans, schedule, runner.script.position, task.mintick)
    _strategy._orders_blocked = True
    try:
        trades = _record(runner, task.syminfo, driver.on_bar)
    finally:
        _strategy._orders_blocked = False
    return _DrawResult(math.fsum(_fill_values(trades, task.unit).values()),
                       not_placed=driver.not_placed)


class _WorkerProcess(multiprocessing.context.SpawnProcess):
    """
    A pool worker that may start processes of its own.

    ``request.security`` runs each context in a child process, which a daemon
    process cannot start, so the worker never becomes one. The pool terminates its
    workers itself, and an orphaned security child exits on its own.
    """

    @property
    def daemon(self) -> bool:
        """:return: Always False"""
        return False

    @daemon.setter
    def daemon(self, value: bool) -> None:
        """Ignore the pool's request to make the worker a daemon."""


class _WorkerContext(multiprocessing.context.SpawnContext):
    """
    Workers that may have children, each in a fresh interpreter: forking a process
    that may run threads is not safe.
    """

    Process = _WorkerProcess


_worker_task: _DrawTask | None = None


def _stop_worker(_signum: int, _frame: FrameType | None) -> None:
    """Unwind the running backtest, so it stops its security children and frees its memory."""
    sys.exit(0)


def _init_worker(task: _DrawTask) -> None:
    global _worker_task
    _worker_task = task
    # A worker is a replica of the recorded backtest: its imports of the script must not
    # rewrite the script's toml at the same time as the other workers
    os.environ['PYNE_SAVE_SCRIPT_TOML'] = '0'
    signal.signal(signal.SIGTERM, _stop_worker)


def _worker_draw(draw: int) -> tuple[int, _DrawResult]:
    assert _worker_task is not None
    return draw, _run_draw(_worker_task, draw)


def _iter_draws(task: _DrawTask, runs: int,
                jobs: int) -> Generator[tuple[int, _DrawResult], None, None]:
    """
    The draws as they finish: in draw order in this process, in any order in workers.

    Closing the generator stops the draws still running.
    """
    if jobs <= 1 or runs <= 1:
        for draw in range(runs):
            yield draw, _run_draw(task, draw)
        return
    try:
        pickle.dumps(task)
    except (pickle.PicklingError, TypeError, AttributeError) as e:
        raise TypeError("running the backtests in parallel needs a make_runner that can be "
                        "pickled: a module-level function or class instance, not a closure; "
                        "or run them with jobs=1") from e
    pool = Pool(min(jobs, runs), initializer=_init_worker, initargs=(task,),
                context=_WorkerContext())
    try:
        yield from pool.imap_unordered(_worker_draw, range(runs))
    finally:
        pool.terminate()
        pool.join()


def _stop_reason(real: float, totals: list[float], mark: float) -> str | None:
    """
    :param real: The real total
    :param totals: The random totals so far
    :param mark: Share of the random draws the real result is tested against
    :return: Why the verdict is already clear, or None while it is not
    """
    below = sum(1 for v in totals if v < real)
    equal = sum(1 for v in totals if v == real)
    # Ties count against the side being proven, so a stop never rests on them
    if wilson_interval(below + equal, len(totals), STOP_CONFIDENCE).high < mark:
        return f"the real rules clearly beat fewer than {mark:.0%} of the random draws"
    if wilson_interval(below, len(totals), STOP_CONFIDENCE).low > mark:
        return f"the real rules clearly beat more than {mark:.0%} of the random draws"
    return None


def _run_draws(task: _DrawTask, real: float, runs: int, *, jobs: int = 1,
               early_stop: EarlyStop = 'off',
               on_progress: Callable[[int, int], None] | None = None,
               ) -> tuple[list[_DrawResult], str | None]:
    """
    Run the backtests of a test, one per draw.

    With an early-stop rule the draws are checked after :data:`STOP_MIN_RUNS`
    backtests and every :data:`STOP_EVERY` after that, always on the first draws in
    draw order. The test stops at the first check whose interval of the share of
    random draws the real result beat (:data:`STOP_CONFIDENCE`) is clear of the
    rule's mark: below it or above it.

    :param task: The test
    :param real: The real total
    :param runs: Backtests without an early stop
    :param jobs: Worker processes; above 1 the task must be picklable
    :param early_stop: ``edge``, ``better`` or ``off`` (see :data:`STOP_MARKS`)
    :param on_progress: Called with the backtests done and the backtests in all
    :return: The draws kept, in draw order, and why the test stopped early (None when
        it ran every backtest)
    """
    mark = STOP_MARKS.get(early_stop)
    results: dict[int, _DrawResult] = {}
    in_order = 0
    check = STOP_MIN_RUNS
    kept = runs
    reason: str | None = None
    draws = _iter_draws(task, runs, jobs)
    try:
        for draw, result in draws:
            results[draw] = result
            if on_progress is not None:
                on_progress(len(results), runs)
            if mark is None:
                continue
            while in_order in results:
                in_order += 1
            # Workers finish out of order: every check runs on its own first draws,
            # so the stop does not depend on the number of workers
            while reason is None and check < runs and check <= in_order:
                reason = _stop_reason(real, [results[i].total for i in range(check)], mark)
                if reason is None:
                    check += STOP_EVERY
                else:
                    kept = check
            if reason is not None:
                break
    finally:
        draws.close()
    if reason is not None and on_progress is not None:
        on_progress(kept, kept)
    return [results[i] for i in range(kept)], reason


def timing_test(make_runner: Callable[[], 'ScriptRunner'], syminfo: 'SymInfo',
                real_trades: list[TradeRecord], unit: Unit, *, seed: int = 0,
                runs: int = DEFAULT_RUNS, jobs: int = 1, early_stop: EarlyStop = 'off',
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
    :param jobs: Worker processes (see :func:`_run_draws`)
    :param early_stop: Early-stop rule (see :data:`STOP_MARKS`)
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

    real = math.fsum(real_values.values())
    task = _DrawTask(mode='timing', make_runner=make_runner, syminfo=syminfo, unit=unit,
                     seed=seed, plans=plans[:count], span=span,
                     mintick=float(syminfo.mintick or 0.0))
    results, stop_reason = _run_draws(task, real, runs, jobs=jobs, early_stop=early_stop,
                                      on_progress=on_progress)
    totals = [r.total for r in results]
    not_placed = sum(r.not_placed for r in results)
    dist, rank, better = _compare(real, totals)
    notes: list[str] = []
    if any(p.risk_fraction is None for p in plans):
        notes.append("trades without a stop were re-entered with their holding time only")
    average_missing = not_placed / len(results) if results else 0.0
    if average_missing:
        notes.append(f"on average {average_missing:.1f} random entries found no flat moment "
                     f"before the end of the data")
    return CoinFlipResult(mode='timing', method='runs', unit=unit, draws=len(totals), seed=seed,
                          real=real, trades=len(real_values), distribution=dist, percentile=rank,
                          share_better=better, totals=totals, not_placed=average_missing,
                          planned=runs, stop_reason=stop_reason, notes=notes,
                          seconds=time.perf_counter() - started)


def coinflip_test(mode: Mode, make_runner: Callable[[], 'ScriptRunner'], syminfo: 'SymInfo',
                  real_trades: list[TradeRecord], unit: Unit, *, seed: int = 0,
                  draws: int = DEFAULT_DRAWS, exact: bool = False, runs: int = DEFAULT_RUNS,
                  jobs: int = 1, early_stop: EarlyStop = 'off',
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
    :param jobs: Worker processes of the tests that run one per draw; above 1
        ``make_runner`` must be picklable (see :func:`_run_draws`)
    :param early_stop: Early-stop rule of the tests that run one per draw (see
        :data:`STOP_MARKS`)
    :param on_progress: Called with the backtests done and the backtests in all
    :return: The comparison
    """
    if mode == 'direction':
        return direction_test(make_runner, syminfo, real_trades, unit, seed=seed, draws=draws,
                              exact=exact, runs=runs, jobs=jobs, early_stop=early_stop,
                              on_progress=on_progress)
    return timing_test(make_runner, syminfo, real_trades, unit, seed=seed, runs=runs,
                       jobs=jobs, early_stop=early_stop, on_progress=on_progress)

