"""
Runtime protocol for request.security() — chart-side and security-side functions.

The SecurityTransformer (Phase 1) rewrites request.security() calls into four protocol
functions: __sec_signal__, __sec_write__, __sec_read__, __sec_wait__. This module provides
the runtime implementations that coordinate via shared memory and multiprocessing Events.

Architecture:
- Chart process: signals security processes, waits for results, reads from shared memory
- Security process: receives signals, runs bars, writes results to shared memory
- Cross-context reads: security processes read other contexts' latest values immediately
"""
from __future__ import annotations

import logging
import os
import threading
from bisect import bisect_right
from time import monotonic
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import Enum, auto
from multiprocessing import Event, Lock, connection
from pathlib import Path
from typing import Any, TYPE_CHECKING
from zoneinfo import ZoneInfo

from .datetime import parse_timezone
from .lookahead import ALLOW_LOOKAHEAD
from .security_shm import (
    SyncBlock, ResultBlock, ResultReader, INITIAL_RESULT_SIZE,
    FLAG_IS_DEVELOPING, FLAG_CLOSED_OVERRIDE, FLAG_DEV_HISTORICAL,
    FLAG_LTF_WINDOW, FLAG_LTF_CHART_DEVELOPING, FLAG_LTF_LIVE_PHASE,
    FLAG_MORE_STEPS, FLAG_BATCH_ROUND, FLAG_DEV_BATCH_ROUND,
    INITIAL_RING_CAPACITY, INITIAL_RING_ARENA, RING_RUNAHEAD_ENTRIES,
    RingReader, RingWriter, write_result,
)

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess
    from multiprocessing.synchronize import (
        Condition as ConditionType, Event as EventType, Lock as LockType,
    )
    from multiprocessing.connection import Connection
    from typing import Callable, Iterator
    from .ohlcv import ChartBarWindow, OHLCVReader
    from .resampler import Resampler
    from .htf_aggregator import HTFAggregator
    from .syminfo import SymInfo, SymInfoSession, SymInfoInterval

logger = logging.getLogger(__name__)


class Lookahead(Enum):
    """Lookahead mode for a security context.

    OFF
        TV-faithful default. The security context advances to the most
        recent HTF bar that has CLOSED at or before the chart bar's CLOSE
        instant — the HTF period's last chart bar already carries the
        period's final value. In historical mode this matches TradingView's
        ``barmerge.lookahead_off`` exactly. In live mode the chart-side
        ``HTFAggregator`` ships each freshly closed HTF bar to the
        subprocess via the SyncBlock (the static ``.ohlcv`` file cannot
        grow at runtime) as soon as the confirmed chart bar completing the
        period is folded — no developing-bar exposure.

    LAST_CLOSED
        PyneSys-native, repaint-free alternative. Always returns the most
        recently closed security bar. In historical mode it is functionally
        equivalent to ``OFF``; in live mode it uses the same closed-bar
        transport as ``OFF`` (no developing exposure) and remains
        repaint-free. Recommended for non-charting backtests when the TV
        ``close[1]`` idiom is not desired.

    ON
        TV ``lookahead_on`` semantics — the security context steps into
        the bar that *contains* the chart bar's time, and that bar runs
        with ``barstate.isconfirmed=False`` over OHLCV aggregated from the
        chart timeframe by ``HTFAggregator``; on HTF period boundaries the
        closed bar is delivered first (snapshot saved), then the fresh
        developing bar. **Historical and live mode take the same path**:
        the aggregator is fed on every chart bar including warmup, and the
        subprocess is given the developing OHLCV rather than being allowed
        to read the containing period's already-complete bar from its own
        data file.

        That last point is a deliberate divergence from TradingView. TV
        exposes the containing period's FINAL close and high on every chart
        bar of the period (measured), which is future data everywhere except
        the period's last bar. PyneCore never reproduces lookahead, so a
        bare ``close`` here reads the period as it has built so far.

        The inner-``[1]`` daily-pivot idiom
        ``request.security(sym, "D", close[1], lookahead_on)`` is unaffected
        and matches TradingView: ``close[1]`` is the just-closed prior period
        (yesterday) whether or not the current period is still developing.
        The outer-``[1]`` form ``request.security(..., close, lookahead_on)[1]``
        reads the previously delivered chart-series value. Use ``LAST_CLOSED``
        when you want explicit last-closed semantics.

        **Cross-symbol HTF** — when the security symbol differs from the
        chart symbol there is no chart-side aggregator (the chart OHLCV
        is the wrong instrument). The chart-side read returns ``na`` while
        an HTF period is open (``na_on_developing``); at the period
        boundary the chart receives the just-closed cross-symbol HTF
        close, and the TV ``request.security(..., lookahead_on)[1]``
        idiom continues to deliver that value on the next chart bar.
        Behaviour is identical in historical and live mode.
    """
    OFF = auto()
    LAST_CLOSED = auto()
    ON = auto()


def _lookahead_mode(value) -> Lookahead:
    """Map a ``barmerge.lookahead_*`` singleton (or None) to a :class:`Lookahead`."""
    from pynecore.lib import barmerge
    if value is barmerge.lookahead_on:
        return Lookahead.ON
    if value is barmerge.lookahead_last_closed:
        return Lookahead.LAST_CLOSED
    return Lookahead.OFF


# Upper bound of a single wait on the registry event while a child waits for a
# runtime-resolved peer's record. Not a sleep: the event ends the wait the
# moment the record lands; this only bounds how often ``stop_event`` is
# re-checked.
_REGISTRY_WAIT_SECONDS = 0.1

# How long a child waits for a runtime-resolved peer's record before saying so.
_REGISTRY_WARN_SECONDS = 30.0

# Liveness poll interval for security-process waits without a death watcher
# (legacy fallback). Short enough to detect a crashed child quickly.
_LIVENESS_POLL_SECONDS = 0.5

# ``PYNE_NO_SECURITY_BATCH=1`` keeps every security context on the per-bar round
# path, whatever its shape. A batch round must be an OPTIMIZATION and nothing
# else — same values, same round sequence in the child — so the two paths have
# to be comparable on the same run, and the equality tests in
# ``tests/t01_lib/t40_security`` run a script both ways and diff the output. It
# is also how a batch-round performance claim is measured.
NO_BATCH = os.environ.get("PYNE_NO_SECURITY_BATCH", "").strip().lower() in (
    "1", "true", "yes", "on",
)


def watch_security_child(
    sec_id: str,
    proc: 'BaseProcess',
    failed_children: set[str],
    events: 'tuple[EventType, ...]',
    stop_events: 'tuple[EventType, ...]' = (),
) -> None:
    """
    Start a daemon thread that watches a security child for abnormal death.

    The per-bar chart waits must be UNTIMED ``event.wait()`` calls: macOS has
    no ``sem_timedwait``, so a timed multiprocessing wait falls back to
    CPython's ``sem_timedwait_save`` emulation — a ``sem_trywait`` +
    ``select()`` polling loop with millisecond-growing sleeps that adds up to
    ~20ms of wake latency PER WAIT regardless of the timeout value. On a per-
    bar signalled context that quantization dominated the whole run (measured
    ~2ms/bar, >80% of wall time). Death detection therefore moves out of the
    wait: this watcher blocks on the process sentinel (no polling), and on a
    non-zero exit registers the sec_id in ``failed_children`` BEFORE setting
    the events, so a blocked ``_wait_with_liveness`` wakes immediately and
    raises instead of deadlocking. A clean exit (code 0) registers nothing —
    a child never exits cleanly while the chart still waits on it.

    :param sec_id: Security context id the process serves
    :param proc: The started child process
    :param failed_children: Shared registry of abnormally died sec_ids
    :param events: Events a chart wait may block on for this context
    :param stop_events: Every child's shutdown event. One child's death has to
        release them ALL: a sibling blocked on this context's ring would never
        wake, and the chart waiting on that sibling would freeze instead of
        raising.
    """
    def _watch() -> None:
        connection.wait([proc.sentinel])
        if proc.exitcode not in (0, None):
            failed_children.add(sec_id)
            # Release every OTHER child too: a peer blocked on this one's ring
            # would otherwise never wake, and the chart would wait on that peer.
            for ev in stop_events:
                ev.set()
            for ev in events:
                ev.set()

    threading.Thread(target=_watch, daemon=True,
                     name=f"sec-watch-{sec_id}").start()


def _wait_with_liveness(
    event: 'EventType',
    sec_id: str,
    sec_processes: 'dict[str, BaseProcess] | None',
    failed_children: 'set[str] | None' = None,
) -> None:
    """
    Wait for ``event`` without deadlocking on a dead security process.

    With a ``failed_children`` registry (see :func:`watch_security_child`)
    the wait is a plain unbounded ``event.wait()`` — the cheap, non-polling
    path (macOS emulates TIMED multiprocessing waits with a select() polling
    loop whose wake latency is disastrous per bar) — and a wake caused by the
    death watcher raises ``RuntimeError``. Without a registry, fall back to
    polling ``proc.is_alive()`` on a timed wait.

    Same-context and ignored sec_ids have no associated Process — they use
    the plain unbounded ``event.wait()`` because their signalling is driven
    by the chart itself, not a separate process.
    """
    if sec_processes is None or sec_id not in sec_processes:
        event.wait()
        return
    if failed_children is not None:
        event.wait()
        if failed_children:
            # ANY child's death can freeze this wait, not only the awaited
            # one's: contexts read each other, so a dead consumer leaves a
            # producer blocked and the chart waiting on that producer. Report
            # the awaited context when it is the dead one, otherwise the child
            # that actually died.
            dead = sec_id if sec_id in failed_children else next(iter(failed_children))
            proc = sec_processes.get(dead)
            raise RuntimeError(
                f"Security process for '{dead}' died unexpectedly "
                f"(exit code: {proc.exitcode if proc is not None else '?'})"
            )
        return
    proc = sec_processes[sec_id]
    while not event.wait(timeout=_LIVENESS_POLL_SECONDS):
        if not proc.is_alive():
            raise RuntimeError(
                f"Security process for '{sec_id}' died unexpectedly "
                f"(exit code: {proc.exitcode})"
            )


@dataclass
class SecurityState:
    """Per-security-context runtime state."""
    sec_id: str
    timeframe: str
    gaps_on: bool
    same_timeframe: bool
    resampler: Resampler | None  # None only if same_timeframe AND same_symbol
    tz: ZoneInfo

    # Multiprocessing events (shared between chart and security processes)
    data_ready: EventType = field(default_factory=Event)
    advance_event: EventType = field(default_factory=Event)
    done_event: EventType = field(default_factory=Event)
    stop_event: EventType = field(default_factory=Event)

    # Cross-process mutex protecting this slot's ResultBlock + sync metadata.
    # Held by writers (write_result/write_na) and by cross-context readers in
    # security children. Chart-side reads also acquire it for uniformity, but
    # never contend (data_ready already gates them).
    result_lock: LockType = field(default_factory=Lock)

    # LTF mode (lower timeframe → array return)
    is_ltf: bool = False

    # Plain ``request.security()`` with a timeframe FINER than the chart's
    # (scalar return, unlike ``is_ltf``). TradingView merge rule (verified on
    # captured references): ``lookahead_off`` returns the expression's value on
    # the LAST intrabar of each chart bar, ``lookahead_on`` on the FIRST. The
    # chart side therefore targets the chart bar's own period end (OFF) or open
    # (ON) and the child's last per-intrabar write wins. ``plain_ltf_span_ms``
    # caches the security period in ms for the live developing-bar clamp.
    plain_ltf: bool = False
    plain_ltf_span_ms: int = 0

    # Synthetic chart type requested via ``ticker.heikinashi()`` etc. ``None`` is
    # an ordinary feed. When set (currently only ``"heikinashi"``), it is passed
    # to the security child, which applies the chart-type transform per bar
    # (backtest and live alike) and flips the matching ``chart.*`` builtin. LTF
    # (sub-bar) chart types are rejected at spawn.
    chart_type: str | None = None

    # Feed the chart-type transform warms up from: an ``.ohlcv`` at the CONTEXT's
    # resolution whose bars reach before the chart's first one. Only set for a
    # context the chart feed serves (:meth:`ScriptRunner._resolve_security_data`),
    # where the file is a seed source and never the context's data. ``None``
    # leaves the recurrence cold, which is what TradingView does when its own
    # feed starts with the chart.
    chart_type_warmup: str | None = None

    # LTF prefix-skip (chart-side, backtest/file-backed only). The LTF child's
    # ``.ohlcv`` feed first bar open, in ms. The child includes intrabars with
    # ``bar_open <= target_time`` and the historical target is the chart bar's
    # last ms (``chart_off``), so a chart bar contains an intrabar only when its
    # period end reaches the feed (``target_time >= ltf_first_ms``). A chart bar
    # whose whole period ends before the feed's first open therefore yields an
    # empty array unconditionally (TradingView returns ``na`` before the LTF
    # series begins). ``__sec_signal__`` then skips the per-bar signal+wait
    # handshake for that idle prefix and ``__sec_read__`` returns the empty-array
    # default without touching shared memory. ``None`` disables the optimization
    # (live ``PluginSymbol`` streams have no static first bar), restoring the
    # original per-chart-bar signal. Populated by ``load_ltf_first_ms`` at child
    # spawn.
    ltf_first_ms: int | None = None

    # Lookahead mode (Pine `lookahead=barmerge.lookahead_*`). Drives whether
    # the security process should emit a ghost-bar write step on chart bars
    # that fall inside an unclosed HTF period.
    lookahead: Lookahead = Lookahead.OFF

    # Per-sec_id HTF aggregator (chart-side). Populated by
    # ``setup_security_states`` for every same-symbol HTF context — drives
    # the live-mode closed-bar transport (all lookahead modes) and, for
    # ``Lookahead.ON``, the developing-bar transport. None for same-TF, LTF,
    # and cross-symbol HTF (chart-derived OHLCV would be the wrong instrument).
    htf_aggregator: HTFAggregator | None = None

    # Cross-symbol HTF + ``Lookahead.ON``: the containing developing bar
    # cannot be aggregated (chart OHLCV is the wrong instrument), so the
    # chart-side read returns ``na`` on every chart bar inside an open HTF
    # period. The subprocess still advances on HTF period closes, so
    # ``close[1]`` on the first chart bar of a fresh HTF period returns the
    # just-closed cross-symbol HTF close — the TV ``lookahead_on + close[1]``
    # idiom continues to work. Applies in both historical and live mode;
    # backtest never silently emits a value live could not produce.
    na_on_developing: bool = False

    # True once the ScriptRunner enters live mode (``barstate.ishistory=False``).
    # Chart-side ``__sec_signal__`` consults this to gate the developing-bar
    # transport — historical bars never emit developing OHLCV.
    is_live: bool = False

    # True when this is an LTF (``request.security_lower_tf``) context backed by a
    # live streaming source (a :class:`PluginSymbol`, no static ``.ohlcv`` file).
    # Such a context has no ``ltf_first_ms`` (the loader is skipped) and its
    # subprocess pulls intrabars from its own streamer, so ``__sec_signal__``
    # routes every round — warmup replay included — through the LTF-window path
    # (``FLAG_LTF_WINDOW``) rather than the file-backed read-ahead path. Set at
    # setup; distinct from ``is_live`` (a lifecycle phase that flips only after
    # warmup) and from ``ltf_first_ms is None`` (which also matches an empty
    # static feed that has no streamer).
    ltf_live_stream: bool = False

    # Intraday session anchoring (chart-side). Populated by
    # ``setup_security_states`` only when this security's session opens off the
    # requested HTF grid (e.g. equities 09:30 at 1H). ``None`` selects the pure
    # UTC clock-floor fast path in ``Resampler.get_bar_time`` — zero overhead for
    # 24/7, on-hour, and session-aligned instruments. ``session_tz`` is the
    # security's own exchange timezone (correct even for cross-symbol HTF).
    session_starts: 'list[SymInfoSession] | None' = None
    session_tz: ZoneInfo | None = None
    # The security's own ``opening_hours``, carried alongside ``session_starts``
    # so the D/W/M trading-day roll can be read from real session bounds instead
    # of being inferred from opens alone — the last trading day of an overnight
    # market's week ends without a new open (see ``overnight_opens``).
    session_opening_hours: 'list[SymInfoInterval] | None' = None

    # Daily/weekly/monthly HTF confirmation (chart-side). The child's data file
    # realizes the actual trading calendar, so confirmation rides the child's
    # real bar opens instead of an arithmetic grid that assumes a bar on every
    # calendar period — essential for sparse daily series (ECONOMICS macro data,
    # dividends) where the grid would confirm phantom periods and the subprocess
    # would ``write_na`` into empty windows, wiping the ``gaps_off`` forward-fill.
    # Two strategies, by ``bar_opens_multiperiod`` (see ``_get_confirmed_time``):
    #   * multi-period (nD/nW/nM): WALK the opens — the grid cannot reproduce
    #     TradingView's scheduled multi-period boundaries (holiday calendar).
    #   * single-period (1D/1W/1M): the grid gives the correct calendar close
    #     instant; CLAMP it to the latest real open so a sparse child still
    #     forward-fills its last value between bars instead of confirming late.
    # Populated by ``load_htf_bar_opens`` at child spawn (backtest only).
    # ``chart_off`` is the chart bar span minus one ms for
    # intraday/seconds charts (0 for D/W/M charts); ``_get_confirmed_time``
    # derives the chart bar's close instant from it for HTF confirmation.
    # ``sec_grid_args`` are the security's own (tz, session_starts,
    # opening_hours, mode) for the past-end-of-data fallback grid.
    bar_opens: list[int] | None = None
    # Session-bounded intraday HTF only: the scheduled session-end instant (ms)
    # of each ``bar_opens`` entry, derived from ``opening_hours``.
    # ``_get_confirmed_time`` then confirms such a bar on its session end
    # (calendar-known) instead of the arithmetic next-period boundary, which a
    # non-trading gap before the next session would push a full period late.
    # ``None`` for D/W/M, sessionless and dense feeds (they keep the grid clamp).
    bar_closes: list[int] | None = None
    bar_opens_multiperiod: bool = False
    bar_ptr: int = -1
    chart_off: int = 0
    sec_grid_args: tuple | None = None

    # The security's own trading schedule, populated by ``load_htf_bar_opens``.
    # Drives ``actual_bar_close`` for this context's bars and, together with the
    # chart's own calendar, the as-of calendar extension (principle 6).
    calendar: 'BarCalendar | None' = None
    # The CHART's schedule — the same object for every context, so the
    # ``same_calendar`` test against ``calendar`` is a plain comparison.
    chart_calendar: 'BarCalendar | None' = None
    # The chart's own timeframe string, for the chart bar's ``actual_bar_close``.
    chart_timeframe: str = ''
    # True when this context's timeframe is daily/weekly/monthly. Only such a
    # peer gets the as-of calendar extension.
    is_dwm: bool = False

    # Sids this context's expression depends on (transitive closure of the
    # transformer's DIRECT ``depends`` lists), and whether its write block sits
    # inside a loop. Chart-side they only travel to the child at spawn.
    depends: frozenset[str] = frozenset()
    in_loop: bool = False

    # Whether the transformer proved this context's ``__sec_read__`` runs on
    # every chart bar (``always_read`` in ``__security_contexts__``). Only such
    # contexts join the cold group start in ``_start``.
    always_read: bool = False

    # Whether the transformer proved this context's ``__sec_signal__`` stands
    # in the script entry's own top guard block and nowhere else
    # (``signal_per_bar`` in ``__security_contexts__``), so it runs exactly once
    # per ``main()`` invocation. A developing batch replays a planned round per
    # chart bar, so a signal that can be skipped or repeated must stay per-bar.
    signal_per_bar: bool = False

    # Live step queue (``__sec_signal__`` enqueues, ``__sec_read__`` and the
    # runner's bar-end hook drive it). Each entry is a zero-argument callable
    # that writes the slot and sets ``advance_event`` for one round.
    pending_live: list = field(default_factory=list)
    # The developing-transport rounds of a context not started yet, in bar
    # order. ``_start`` runs them as one burst at the first read, so a context
    # read late gets exactly the round sequence an eager start would have run.
    # Every round carries its own OHLCV values, captured when it was queued.
    deferred_steps: list = field(default_factory=list)
    # Rounds this context launched vs. the child's ``rounds_done`` counter in
    # the SyncBlock. "One round outstanding" waits for equality instead of a
    # bool ``done_event``, because one chart bar can launch several rounds.
    rounds_launched: int = 0

    # LTF window on a daily/weekly/monthly CHART (chart-side, file-backed). A
    # single-period civil D/W/M chart bar has no fixed arithmetic span
    # (``chart_off == 0``), so the Phase 1 ``chart_time + chart_off`` target
    # degrades to the bar open and the child would re-collect the *previous*
    # period. When the chart is a single-period civil D/W/M timeframe,
    # ``chart_resampler`` (with ``chart_dwm_modifier`` 'D'/'W'/'M') is set for
    # every single-period D/W/M chart so ``__sec_signal__`` can target the chart
    # bar's civil period end (``_next_civil_period_open`` minus one ms) — the
    # bar's OWN period ``[T, next_civil_open)``. Both stay ``None``/``''`` for
    # intraday charts (the ``chart_off`` fast path) and for multi-period D/W/M
    # charts (excluded at setup). Session-anchored D/W/M charts DO get these set,
    # but the per-bar civil-anchored guard in ``__sec_signal__``
    # (``get_bar_time(chart_time) == chart_time``) falls back to ``chart_off``
    # for a bar that does not open on the civil boundary — a correct window there
    # needs the chart's real bar opens, not a civil-calendar guess, and no
    # TradingView ground truth exists for it (documented limitation).
    chart_resampler: Resampler | None = None
    chart_dwm_modifier: str = ''

    # Lazy start (chart-side). ``True`` once this context's child process has
    # been started and its rounds are actually launched. A context is started at
    # its first ``__sec_read__``, not at its first ``__sec_signal__``: the
    # transformer hoists every context's signal to the top of ``main()``, so a
    # ``request.security()`` call sitting in a branch this run never takes is
    # signalled on every bar — and would otherwise cost a process plus a round
    # per bar for a value nothing ever reads.
    started: bool = False
    # A round the signal prepared (target time and flags are already written to
    # the SyncBlock) while this context was not started yet. ``_start`` performs
    # it; because the child runs its feed up to the round's target, that one
    # late round also covers every bar skipped meanwhile.
    pending_launch: bool = False

    # Tracking (chart-side only)
    last_confirmed: int = 0
    needs_wait: bool = False
    new_period: bool = False

    # ``Lookahead.ON``: set once the one-time historical prefill in
    # ``__sec_signal__`` has replayed the child's own ``.ohlcv`` bars that
    # closed before the first containing period. The developing transport
    # never reads that file, so without the prefill the child's HTF series
    # would begin at the chart's first bar.
    htf_prefilled: bool = False

    # Set by ``__sec_signal__`` when an LTF chart bar precedes the feed (see
    # ``ltf_first_ms``): no handshake ran this bar, so ``__sec_read__`` returns
    # the empty-array default directly instead of waiting on shared memory.
    ltf_skip: bool = False

    # Historical BATCH round (see :func:`plan_historical_batch`). Non-zero
    # ``batch_target`` means the chart launches this context ONCE for the whole
    # historical phase instead of once per chart bar, and pairs the values out
    # of the child's ring as-of each chart bar. What this removes is the
    # wake-up round trip per chart bar per context; the child's work and the
    # values it publishes are unchanged.
    batch_target: int = 0
    # Ring entries the batch round publishes, for pre-sizing the ring block.
    batch_capacity: int = 0
    # Payload arena size (bytes) to pre-size that ring block with.
    batch_arena: int = 0
    # Whether the batch round has been launched (the first signal does it).
    batch_launched: bool = False
    # The chart's as-of instant for this context on the CURRENT bar — the same
    # instant ``_get_confirmed_time`` paired the target with, kept for the ring
    # lookup in ``__sec_read__``.
    batch_asof: int = 0
    # The PREVIOUS bar's as-of for this context, the lower bound of the window a
    # developing batch's entry for the current bar may close in (see
    # ``_batch_read``).
    batch_prev_asof: int = 0

    # Historical DEVELOPING batch round (see :func:`prepare_developing_batch`).
    # What the child reproduces its round sequence from, ``None`` for every other
    # shape. Set alongside ``batch_target``, so the read routes through
    # ``_batch_read`` and the cold group start covers this context too — only the
    # child's round source differs (its own replay instead of the per-bar pushes).
    dev_batch_spec: 'DevBatchSpec | None' = None


def _same_value(a, b) -> bool:
    """Repeat-write equality: Pine's ``na == na``, and tuples elementwise.

    A security write block standing in a loop body runs several times per bar.
    The first write publishes; an identical repeat is a no-op. Only this
    comparison decides which, so ``na`` (a float NaN, never equal to itself)
    has to count as unchanged.
    """
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b))
    if a is b:
        return True
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return bool(a == b)


def chart_asof(state: SecurityState, chart_time: int,
               next_chart_time: int = 0, round_tick: int = 0) -> int:
    """
    The instant the CHART is as-of for this context in the current round.

    The chart bar's own scheduled close (:func:`actual_bar_close`), or — on a
    developing round — the round's fixed tick instant, which the runner reads
    once per bar cycle so every signal, target and clamp of that bar agree.

    For a daily/weekly/monthly peer keeping the CHART's calendar the base time
    is extended to the end of the scheduled break it falls in
    (:func:`break_end_after`): an equity hourly bar closing at 16:00 sits in the
    break, so it already sees the daily bar closing at 16:00, while the
    09:30-10:30 bar (closing inside the session) does not. A peer on another
    calendar gets no extension — it may trade during this one's break.

    :param state: Security context state.
    :param chart_time: Current chart bar open time in ms.
    :param next_chart_time: Open of the chart bar after this one, 0 when none.
    :param round_tick: The round's fixed tick instant for a developing round;
        0 selects the chart bar's scheduled close.
    :return: The chart's as-of instant for this context, in ms.
    """
    if round_tick:
        base = round_tick
    elif state.chart_calendar is not None and state.chart_timeframe:
        base = actual_bar_close(chart_time, next_chart_time,
                                state.chart_calendar, state.chart_timeframe)
    else:
        # No chart calendar (unit-test / legacy callers): the nominal chart span.
        base = chart_time + state.chart_off + 1
    if (state.is_dwm and state.chart_calendar is not None
            and state.calendar is not None
            and same_calendar(state.calendar, state.chart_calendar)):
        return break_end_after(base, state.chart_calendar)
    return base


def _ltf_bar_index(state: SecurityState, chart_time: int,
                   next_chart_time: int, round_tick: int) -> int:
    """
    Index of the LAST intrabar the chart bar may see, ``-1`` when there is none.

    The same pairing rule the rest of the module uses: the peer's last bar whose
    scheduled close (``close_A``) is at or before the consumer's as-of instant
    (:func:`chart_asof`). For a lower-timeframe context this is what keeps an
    intrabar that straddles the chart bar's close — a non-nested grid (a 4-minute
    context on a 3-minute chart) or a session-shortened intrabar — out of this
    chart bar: its close lies after the chart bar's own close, so it belongs to
    the next chart bar's round.

    ``-1`` means the chart bar precedes the context's first usable intrabar; the
    caller then skips the cross-process handshake and answers with the default
    (an empty array, or ``na`` for the scalar merge), which is what TradingView
    returns before the lower-timeframe series begins.

    Requires the loaded feed (:func:`load_htf_bar_opens`); a live stream with no
    static file keeps the nominal span target and its own closed-intrabar clamp.

    :param state: Lower-timeframe security context state.
    :param chart_time: Current chart bar open time in ms.
    :param next_chart_time: Open of the chart bar after this one, 0 when none.
    :param round_tick: The round's fixed tick instant on a developing round.
    :return: Index into ``state.bar_opens`` / ``state.bar_closes``, or ``-1``.
    """
    opens = state.bar_opens
    closes = state.bar_closes
    assert opens is not None and closes is not None
    asof = chart_asof(state, chart_time, next_chart_time, round_tick)
    # ``asof`` never decreases across rounds and the closes ascend, so the
    # persistent pointer only ever advances.
    n = len(closes)
    ptr = state.bar_ptr
    while ptr + 1 < n and closes[ptr + 1] <= asof:
        ptr += 1
    state.bar_ptr = ptr
    return ptr


def _get_confirmed_time(state: SecurityState, chart_time: int,
                        next_chart_time: int = 0, round_tick: int = 0) -> int:
    """
    Determine which security period the subprocess should advance to.

    ONE rule for every peer kind (HTF daily/weekly/monthly, intraday HTF,
    same-timeframe cross-symbol, plain lower-timeframe): the context's LAST bar
    whose scheduled close (``close_A``) is at or before the chart's as-of
    instant for it (:func:`chart_asof`); the target is that bar's open. The
    three TradingView pairings measured for this rule (weekly consumer of a
    daily producer, daily consumer of a weekly producer, same-timeframe
    cross-symbol) all reduce to it, and so does the child-side peer read — the
    chart is just another consumer.

    Confirmation therefore rides the child's REAL bars and their real closes:
    a sparse child (scattered macro days) forward-fills its last value instead
    of confirming phantom calendar periods, a session-closing stub bar is
    confirmed on the chart bar its session end reaches, and a bar straddling
    the chart bar's close is left for the next chart bar rather than exposing a
    price from after that close.

    Falls back to the arithmetic period grid only when the child's bars are not
    loaded (live streams, unit tests without a feed).

    :param state: Security context state
    :param chart_time: Current chart bar time in milliseconds
    :param next_chart_time: Open time (ms) of the chart bar after this one, 0
                            when none is known (last historical bar, live)
    :param round_tick: The round's fixed tick instant on a developing round
    :return: Target time in milliseconds
    """
    if (ALLOW_LOOKAHEAD and state.lookahead is Lookahead.ON
            and state.htf_aggregator is not None):
        # ``PYNE_ALLOW_LOOKAHEAD`` only: step into the period CONTAINING the
        # chart bar instead of the last closed one, so a script can be measured
        # with TradingView's future-leak and without it. ``chart_time`` (not the
        # close instant) selects the period, so the period's last chart bar
        # still maps to that period rather than the next.
        resampler = state.resampler
        assert resampler is not None
        if state.session_starts is not None:
            return resampler.get_bar_time(
                chart_time, state.session_tz, state.session_starts,
                state.session_opening_hours)
        return resampler.get_bar_time(chart_time, state.tz)

    asof = chart_asof(state, chart_time, next_chart_time, round_tick)

    opens = state.bar_opens
    closes = state.bar_closes
    if opens is not None and closes is not None:
        # ``asof`` is monotonically non-decreasing across chart bars and the
        # closes are ascending, so the persistent ``bar_ptr`` only ever advances.
        # ``ptr == -1`` means the chart still precedes the context's first bar:
        # ``last_confirmed`` (0) confirms nothing and the read stays ``na``.
        n = len(opens)
        ptr = state.bar_ptr
        while ptr + 1 < n and closes[ptr + 1] <= asof:
            ptr += 1
        state.bar_ptr = ptr
        if ptr >= 0 and closes[ptr] <= asof:
            return opens[ptr]
        return state.last_confirmed

    if state.same_timeframe:
        # A chart bar and its same-time security bar close at the same instant.
        return chart_time

    resampler = state.resampler
    assert resampler is not None
    if state.session_starts is not None:
        # Off-grid intraday session -> anchor the context's bars to the session
        # open, using the security's own exchange timezone.
        period = resampler.get_bar_time(
            asof, state.session_tz, state.session_starts,
            state.session_opening_hours)
        return resampler.get_bar_time(
            period - 1, state.session_tz, state.session_starts,
            state.session_opening_hours)
    period = resampler.get_bar_time(asof, state.tz)
    return resampler.get_bar_time(period - 1, state.tz)


# How many security bars past the chart's own last as-of the batch round runs.
# The chart's final as-of is derived here from ``last_bar_time`` with no
# following bar, while the run itself may pair that bar with a real next open;
# a couple of spare bars make the frontier reach whatever the run actually asks
# for. Publishing them cannot leak anything: every read pairs by close against
# its own as-of, so a bar closing later is never selected.
_BATCH_MARGIN_BARS = 2

# Bytes reserved per ring entry when pre-sizing a batch ring. A pickled scalar
# or short tuple stays well under this; a larger value only makes the writer
# grow the arena once, which is correct either way.
_BATCH_ARENA_PER_ENTRY = 128


def plan_historical_batch(state: SecurityState, last_chart_time: int) -> bool:
    """
    Plan this context's single historical batch round, if it can have one.

    The chart's as-of for its LAST bar (:func:`chart_asof`) decides how far the
    child has to run: the last of its bars whose scheduled close reaches that
    instant, plus :data:`_BATCH_MARGIN_BARS`. That target is exactly where the
    per-bar rounds would have left the child, so one round replaces all of them
    without running the feed past the chart.

    Requires the child's real bars (:func:`load_htf_bar_opens`), which is also
    what proves the context is file-backed: a live streamer has no static feed
    to run ahead on.

    :param state: Security context state, after ``load_htf_bar_opens``.
    :param last_chart_time: Open time (ms) of the chart's last historical bar.
    :return: Whether a batch round was planned (``state.batch_target`` set).
    """
    opens = state.bar_opens
    closes = state.bar_closes
    if not opens or not closes or len(closes) != len(opens):
        return False
    asof_end = chart_asof(state, last_chart_time, 0, 0)
    idx = bisect_right(closes, asof_end) - 1
    if idx < 0:
        # The chart ends before this context's first bar closes: nothing to
        # batch, every bar reads ``na`` anyway.
        return False
    idx = min(idx + _BATCH_MARGIN_BARS, len(opens) - 1)
    state.batch_target = opens[idx]
    state.batch_capacity = idx + 1
    state.batch_arena = state.batch_capacity * _BATCH_ARENA_PER_ENTRY
    return True


def batch_eligible(state: SecurityState, sec_id: str, *, is_live: bool) -> bool:
    """
    Whether a context's historical phase may run as ONE batch round.

    Every excluded shape needs the chart to decide, per bar, what the child
    should do next — so its round cannot be planned up front:

    * a live run (the child would end up past the chart's history, and the
      per-round transports below take over at the transition anyway);
    * a context whose timeframe is COARSER than the chart's: the closed-only
      flow already launches a round only when a fresh period confirms, so
      there is barely a handshake to remove — while the ring lookup the batch
      read costs is paid on every read (MEASURED: a daily context on a 30m
      chart got ~20% slower). The whole win is in contexts that confirm a
      fresh period on every chart bar, which after the lower-timeframe
      exclusions below means the chart's own timeframe;
    * a context with a non-empty ``depends`` set. Pairing a peer out of its
      ring would be correct (it is as-of based like everything else), but the
      round's chart tick is the batch's ONE tick, so the peer read's cap has to
      go — and then the child runs its whole feed gated on peers that advance
      at the chart's pace, one ring wait per dependency per bar. MEASURED on a
      19-context screener whose contexts are all (spuriously) tainted by each
      other: 277 s against 64 s for the per-bar rounds;
    * ``Lookahead.ON``: the developing-bar transport pushes OHLCV into the slot
      per chart bar, and with ``PYNE_ALLOW_LOOKAHEAD`` the target is the
      CONTAINING period rather than the last closed one — neither is an as-of
      pairing the ring can answer;
    * lower-timeframe contexts (``request.security_lower_tf`` and the scalar
      ``plain_ltf`` merge), whose per-bar target IS the merge rule;
    * the synthetic ``__auto_rate_*`` feeds, driven by ``signal_rate_sources``
      rather than by a Pine call.

    :param state: Security context state, after ``load_htf_bar_opens``.
    :param sec_id: The context's id.
    :param is_live: Whether the run has a live phase.
    :return: Whether the context may be batched
        (never with :data:`NO_BATCH`).
    """
    return (not is_live
            and not NO_BATCH
            and state.same_timeframe
            and not state.depends
            and not state.is_ltf
            and not state.plain_ltf
            and not state.ltf_live_stream
            and state.lookahead is not Lookahead.ON
            and not state.na_on_developing
            and not sec_id.startswith('__auto_rate_'))


# One developing-batch record per push the per-bar path would have made:
# ``(kind, period_start, open, high, low, close, volume, tick,
# sched_next_open)``, where the last two are the chart bar's round tick and its
# scheduled next session open — the two values ``_launch`` writes into the slot.
# ``DEV_BATCH_PREFILL`` carries only its target in ``period_start``.
#
# The records are never materialized as a sequence: the CHILD produces them one
# at a time from :class:`DevBatchSpec` while it replays them (see
# :func:`iter_dev_batch_records`), so neither process ever holds more than the
# record it is on, whatever the chart's length.

# Record kinds, mirroring the three steps ``__sec_signal__`` builds per bar.
DEV_BATCH_PREFILL = 0
DEV_BATCH_CLOSED = 1
DEV_BATCH_DEVELOPING = 2


@dataclass(frozen=True, slots=True)
class DevBatchSpec:
    """Everything a child needs to reproduce its developing-batch rounds.

    Travels to the child in the spawn arguments (a few hundred bytes) in place
    of the round sequence itself. Two halves:

    * the chart's bar production — the window over the chart's own OHLCV file
      and the cleaning parameters the bar loop publishes its bars with, so the
      child walks the very bars the loop runs on;
    * the chart-side context state the per-bar path derived each push from — the
      aggregator's configuration and the chart's own schedule.

    :param window: The run's chart bar window (see :class:`ChartBarWindow`).
    :param round_decimals: Mintick decimals of the chart's price rounding.
    :param lossless_volume: The chart feed reads its volume back exactly.
    :param lossless_prices: The chart feed reads its OHLC back exactly.
    :param timeframe: The context's timeframe.
    :param tz: The context's session timezone.
    :param session_starts: Its intraday session anchoring, or ``None``.
    :param chart_span_ms: The aggregator's close-instant span (0 disables it).
    :param chart_off: The chart bar's own span minus one, in ms.
    :param chart_timeframe: The chart's timeframe, for the bar close rule.
    :param chart_calendar: The chart's trading schedule, or ``None``.
    """

    window: 'ChartBarWindow'
    round_decimals: int | None
    lossless_volume: bool
    lossless_prices: bool
    timeframe: str
    tz: ZoneInfo
    session_starts: 'list[SymInfoSession] | None'
    chart_span_ms: int
    chart_off: int
    chart_timeframe: str | None
    chart_calendar: 'BarCalendar | None'


def iter_dev_batch_records(spec: DevBatchSpec) -> 'Iterator[tuple]':
    """Produce the whole historical developing-transport sequence of a context.

    Walks the CHART's bars — from ``spec.window``, the same production the bar
    loop takes its bars from, cleaned by the same :func:`_clean_bar` the loop
    publishes through — with a private :class:`HTFAggregator` configured exactly
    like the runtime one, and yields per bar what ``__sec_signal__``'s
    ``Lookahead.ON`` branch would have pushed: the one-time prefill target, then
    the closed step of a period that just completed and the developing step of
    the period in progress.

    A GENERATOR on purpose: the child consumes each record as it replays it, so
    the memory this costs is one record and one chart bar rather than the
    ~72 bytes per chart bar a materialized sequence took in BOTH processes.

    NOTHING is compressed. A period's developing bar is re-pushed on every chart
    bar inside it, and every one of those pushes is a record: the child rolls a
    same-period re-tick's ``var`` and function-instance slots back, but not its
    ``varip`` slots or its ``IBPersistent`` storage, so dropping a re-tick would
    leave it in a state the per-bar path never reaches.

    :param spec: The context's batch specification.
    :return: Iterator of ``(kind, period_start, open, high, low, close, volume,
        tick, sched_next_open)`` records, in push order.
    """
    # Neither can be imported at module level: ``htf_aggregator`` pulls in
    # ``resampler``, which imports ``lib``, which imports THIS module, and
    # ``script_runner`` imports ``lib`` too.
    from .htf_aggregator import HTFAggregator
    from .script_runner import _clean_bar

    cal = spec.chart_calendar
    # A private aggregator, same construction as ``setup_security_states``': the
    # runtime one keeps feeding the per-bar ``new_period`` bookkeeping and must
    # not see this walk.
    aggregator = HTFAggregator(
        spec.timeframe, spec.tz, session_starts=spec.session_starts,
        chart_span_ms=spec.chart_span_ms)
    chart_off = spec.chart_off
    chart_timeframe = spec.chart_timeframe
    round_decimals = spec.round_decimals
    lossless_volume = spec.lossless_volume
    lossless_prices = spec.lossless_prices
    last_confirmed = 0
    prefilled = False

    def _bar_records(bar: tuple, next_time: int) -> list[tuple]:
        """Every record ONE chart bar produces.

        :param bar: The cleaned chart bar ``(open_ms, o, h, l, c, v)``.
        :param next_time: Open time of the NEXT chart bar, 0 for the last one.
        :return: The bar's records, in push order.
        """
        nonlocal last_confirmed, prefilled
        bar_time, b_open, b_high, b_low, b_close, b_volume = bar
        if cal is not None and chart_timeframe:
            tick = actual_bar_close(bar_time, next_time, cal, chart_timeframe)
        else:
            tick = bar_time + chart_off + 1
        if cal is not None and cal.opening_hours:
            extended = break_end_after(tick, cal)
            sched_next_open = 0 if extended == tick else extended
        else:
            sched_next_open = 0

        _, dev_bar, closed_bar = aggregator.update(
            bar_time, b_open, b_high, b_low, b_close, b_volume,
            chart_confirmed=True)

        out: list[tuple] = []
        if not prefilled:
            prefilled = True
            containing = dev_bar if dev_bar is not None else closed_bar
            prefill_target = (containing.period_start - 1
                              if containing is not None else 0)
            if prefill_target > last_confirmed:
                last_confirmed = prefill_target
                out.append((DEV_BATCH_PREFILL, prefill_target,
                            0.0, 0.0, 0.0, 0.0, 0.0, tick, sched_next_open))
        if closed_bar is not None:
            last_confirmed = closed_bar.period_start
            out.append((DEV_BATCH_CLOSED, closed_bar.period_start,
                        closed_bar.open, closed_bar.high, closed_bar.low,
                        closed_bar.close, closed_bar.volume,
                        tick, sched_next_open))
        if dev_bar is not None:
            out.append((DEV_BATCH_DEVELOPING, dev_bar.period_start,
                        dev_bar.open, dev_bar.high, dev_bar.low,
                        dev_bar.close, dev_bar.volume,
                        tick, sched_next_open))
        return out

    # One bar of lookahead: a bar's round tick is its close instant, which the
    # trading schedule resolves from the NEXT bar's open time.
    pending: tuple | None = None
    for candle in spec.window.bars():
        o, h, lo, c, v = _clean_bar(candle, round_decimals,
                                    lossless_volume, lossless_prices)
        cleaned = (candle.timestamp, o, h, lo, c, v)
        if pending is not None:
            for record in _bar_records(pending, cleaned[0]):
                yield record
        pending = cleaned
    if pending is not None:
        for record in _bar_records(pending, 0):
            yield record


def dev_batch_eligible(state: SecurityState, sec_id: str, *,
                       is_live: bool, has_consumers: bool) -> bool:
    """
    Whether a ``lookahead_on`` HTF context's history may run as ONE batch round.

    The shape this covers is the expensive one: a same-symbol HIGHER-timeframe
    context with ``lookahead=barmerge.lookahead_on`` in a backtest. Its bars are
    not read from the child's file at all — the chart aggregates the developing
    HTF bar from its own bars and pushes it into the child's slot, then blocks on
    the child's answer. That is a round trip per chart bar (MEASURED on a
    9-context 30m script: 26,784 rounds for 2,976 chart bars, 87% of the wall
    clock spent with the chart blocked). Every push is a function of the chart's
    own bar stream, so the whole sequence can be planned up front and replayed
    by the child while it runs ahead of the chart.

    Excluded, and why the exclusion is not just conservatism:

    * a live run — the pushes are not knowable up front;
    * anything but the developing transport: no aggregator (cross-symbol HTF,
      where the chart's OHLCV is the wrong instrument), the same timeframe (the
      closed batch's shape), ``na_on_developing``, or
      ``PYNE_ALLOW_LOOKAHEAD``, which routes ``ON`` through the closed-only flow;
    * a context with dependencies or consumers: the planned rounds all carry
      their own bar's tick, but a peer read's cap and a consumer's pairing are
      derived from the round the CHART is on — the child would be running
      ahead of the instants those are relative to;
    * lower-timeframe contexts, whose per-bar target IS the merge rule, and the
      synthetic ``__auto_rate_*`` feeds, which no Pine call signals;
    * a context whose signal is not proved to run exactly once per chart bar
      (``signal_per_bar``): the plan holds one round per chart bar and every
      signal consumes the next one, so a signal behind a branch, inside a
      conditionally called helper or on a helper invoked twice would pair the
      bar with another bar's round.

    :param state: Security context state, after ``load_htf_bar_opens``.
    :param sec_id: The context's id.
    :param is_live: Whether the run has a live phase.
    :param has_consumers: Whether another context reads this one.
    :return: Whether the context may run its history as a developing batch
        (never with :data:`NO_BATCH`).
    """
    return (not is_live
            and not NO_BATCH
            and not ALLOW_LOOKAHEAD
            and state.signal_per_bar
            and state.lookahead is Lookahead.ON
            and state.htf_aggregator is not None
            and not state.same_timeframe
            and not state.depends
            and not has_consumers
            and not state.is_ltf
            and not state.plain_ltf
            and not state.ltf_live_stream
            and not state.na_on_developing
            and not sec_id.startswith('__auto_rate_'))


def prepare_developing_batch(state: SecurityState, spec: DevBatchSpec,
                             chart_bar_count: int) -> bool:
    """
    Arm the historical developing batch of one context.

    Only the ROUTING is decided here — the round sequence itself is produced in
    the child, record by record, from ``spec`` (see
    :func:`iter_dev_batch_records`). The chart therefore never walks its bars a
    second time and never holds a plan: what this costs is independent of the
    chart's length.

    :param state: Security context state (nothing of its own is walked).
    :param spec: The batch specification the child reproduces its rounds from.
    :param chart_bar_count: Number of chart bars in the run's window.
    :return: Whether the batch was armed (``state.dev_batch_spec`` set).
    """
    if chart_bar_count <= 0 or not spec.window.to_ts:
        return False
    state.dev_batch_spec = spec
    # ``batch_target`` is what routes the read through the ring and joins the
    # cold group start. The child's replay carries the target of every one of
    # its rounds in the record itself, so what the slot holds only has to be an
    # instant no planned period start can exceed: the window's own end.
    state.batch_target = spec.window.to_ts or 0
    # Pre-size the ring to the run-ahead bound rather than to the whole
    # sequence: the child never holds more live entries than that, and the GC
    # compacts below the chart's watermark. At most one prefill plus a closed
    # and a developing record per chart bar can exist.
    state.batch_capacity = min(2 * chart_bar_count + 1, RING_RUNAHEAD_ENTRIES * 2)
    state.batch_arena = state.batch_capacity * _BATCH_ARENA_PER_ENTRY
    return True


def _next_civil_period_open(modifier: str, current_ms: int, tz: ZoneInfo) -> int:
    """
    Open time (ms) of the civil period immediately following the one that
    contains ``current_ms``, for a single-period daily/weekly/monthly chart.

    The next local calendar boundary (next day's midnight, next Monday, or the
    first of next month) is constructed *directly* in ``tz`` and then converted
    back to epoch ms — never by adding a fixed 24h / 7d / nominal-month delta —
    so the result is correct across DST transitions and variable month lengths.
    Used to window LTF intrabars into a D/W/M chart bar's own period
    ``[T, next_open)`` (the caller targets ``next_open - 1``).

    :param modifier: Chart timeframe modifier, one of ``'D'``, ``'W'``, ``'M'``.
    :param current_ms: The chart bar's open time in milliseconds.
    :param tz: The chart's timezone (defines where the civil boundary falls).
    :return: The next civil period's open time in milliseconds.
    """
    cur = datetime.fromtimestamp(current_ms / 1000, tz)
    if modifier == 'D':
        nd = cur.date() + timedelta(days=1)
        nxt = datetime(nd.year, nd.month, nd.day, tzinfo=tz)
    elif modifier == 'W':
        # Anchor to the bar's Monday, then step a full week — robust even if the
        # bar open is not exactly the Monday boundary.
        monday = cur.date() - timedelta(days=cur.weekday())
        nd = monday + timedelta(days=7)
        nxt = datetime(nd.year, nd.month, nd.day, tzinfo=tz)
    else:  # 'M'
        year, month = (cur.year + 1, 1) if cur.month == 12 else (cur.year, cur.month + 1)
        nxt = datetime(year, month, 1, tzinfo=tz)
    return int(nxt.timestamp()) * 1000


def create_chart_protocol(
    states: dict[str, SecurityState],
    sync_block: SyncBlock,
    deferred_resolve_fn: 'Callable[[str, str, str | None], None] | None' = None,
    prepare_fn: 'Callable[[str], None] | None' = None,
    lazy_spawn_fn: 'Callable[[str], None] | None' = None,
    same_context_ids: 'set[str] | frozenset[str]' = frozenset(),
    no_process_ids: 'set[str] | frozenset[str]' = frozenset(),
    result_blocks: dict[str, ResultBlock] | None = None,
    currency_conversions: dict[str, tuple[str, str]] | None = None,
    sec_processes: 'dict[str, BaseProcess] | None' = None,
    auto_rate_sec_ids: frozenset[str] = frozenset(),
    failed_children: 'set[str] | None' = None,
    ring_conditions: 'dict[str, ConditionType] | None' = None,
    consumers_by_sid: 'dict[str, list[str]] | None' = None,
) -> tuple:
    """
    Create protocol functions for the **chart** process.

    :param states: Per-security-context runtime states
    :param sync_block: Shared memory sync block
    :param deferred_resolve_fn: Optional callback for resolving deferred security contexts.
                                Called with (sec_id, symbol, timeframe) on first __sec_signal__.
    :param prepare_fn: Optional callback preparing a static context on its FIRST
                       ``__sec_signal__``: it loads the child's real bar grid and
                       plans its batch round, both of which every chart-side
                       target computation from that bar on reads. It starts no
                       process.
    :param lazy_spawn_fn: Optional callback for lazy-spawning static security processes.
                          Called with sec_id on the context's FIRST ``__sec_read__``
                          (see ``_start``), so a context the run never reads never
                          gets a process.
    :param same_context_ids: Security IDs that share the chart's symbol+timeframe.
                             These are handled directly by the chart (no separate process).
    :param no_process_ids: Security IDs that have no process (same-context + ignored).
                           Signal/wait are skipped for these.
    :param result_blocks: Result blocks for writing same-context values to shared memory.
    :param currency_conversions: Maps sec_id → (from_currency, to_currency) for auto-conversion.
    :param sec_processes: Live ``sec_id → Process`` map. Captured by reference, so
                          entries added by lazy/deferred spawn become visible to the
                          read/wait protocol functions. When provided, blocked waits
                          poll ``proc.is_alive()`` and raise instead of deadlocking
                          if a child dies.
    :param auto_rate_sec_ids: Hidden ``__auto_rate_*`` sec_ids driving currency
                              rate sources. No Pine call signals them, so the
                              chart loop must call ``signal_rate_sources()``
                              once per bar to advance their subprocess and
                              refresh the ResultBlock the
                              :class:`CurrencyRateProvider` reads from.
    :param failed_children: Shared registry filled by
                            :func:`watch_security_child` when a child dies
                            abnormally. Enables the cheap UNTIMED waits; when
                            None the waits fall back to liveness polling.
    :param ring_conditions: Per-sid ``multiprocessing.Condition``, shared with
                            the children. Needed for the chart-context
                            producers, whose ring the chart itself writes.
    :param consumers_by_sid: Sids consuming each context, for the ring GC
                             watermark minimum.
    :return: (sec_signal, sec_write, sec_read, sec_wait, cleanup,
              signal_rate_sources, begin_bar, end_bar)
    """
    # Module-level import would close a ``core`` <-> ``lib`` cycle; done once
    # here rather than per bar inside the protocol functions.
    from pynecore import lib

    readers: dict[str, ResultReader] = {
        sid: ResultReader(sid, sync_block.block_prefix(sid)) for sid in states
    }

    resolved: set[str] = set()
    # Whether the cold group start has run (see ``_start``). One flag for the
    # whole run: after it, every always-read batched context is running.
    cold_batch_start = [False]
    ring_conditions = ring_conditions or {}
    consumers_by_sid = consumers_by_sid or {}

    # Ring readers of the BATCHED contexts (see ``SecurityState.batch_target``).
    # The chart is an ordinary ring consumer for them: it waits on the
    # producer's frontier and pairs by close, exactly as a child does for a
    # peer — only its GC watermark lives in the extra row the SyncBlock keeps
    # for it rather than in a slot of its own.
    batch_readers: dict[str, RingReader] = {}
    chart_wm_index = sync_block.chart_index

    # Chart-context producers: a sid the chart itself evaluates (same symbol and
    # timeframe) whose value another context consumes. The chart owns its ring.
    chart_writers: dict[str, RingWriter] = {}
    chart_consumer_indexes: dict[str, list[int]] = {}
    chart_bar_written: set[str] = set()
    chart_bar_value: dict[str, object] = {}

    def _ensure_chart_writer(_sid: str) -> None:
        """Give a chart-context producer its ring writer, if it has consumers.

        Called both at startup and from ``__sec_signal__``: a context whose
        symbol or timeframe is only known at runtime can be downgraded to the
        chart's own context by the deferred resolver, long after this factory
        ran. It is advertised as a producer the moment that happens, so its
        ring has to exist by then — otherwise its writes never reach a ring and
        every dependent child blocks on a frontier nothing moves.
        """
        if _sid in chart_writers or _sid not in same_context_ids:
            return
        _consumers = consumers_by_sid.get(_sid)
        if not _consumers or _sid not in ring_conditions:
            return
        chart_writers[_sid] = RingWriter(_sid, sync_block, ring_conditions[_sid])
        chart_consumer_indexes[_sid] = [sync_block.index_of(c) for c in _consumers]

    # The round's fixed instants, read ONCE per chart bar cycle by ``begin_bar``.
    # Every target, live clamp and slot round-context of the bar uses the same
    # values: a child reading a second later must not ask for a bar the chart
    # did not target in this round.
    round_state = {'tick': 0, 'sched_next_open': 0, 'chart_time': 0, 'next_time': 0}

    def _chart_calendar() -> 'BarCalendar | None':
        for st in states.values():
            if st.chart_calendar is not None:
                return st.chart_calendar
        return None

    def begin_bar(chart_time: int, next_chart_time: int, confirmed: bool) -> None:
        """Open a chart bar cycle: fix the round's tick instant.

        Read once, here, and used by every signal, target and live clamp of the
        bar — and handed to each child in its slot. Reading a fresh clock per
        signal would let a consumer ask for a peer bar the producer was not
        targeted at in this round, which nothing would ever publish.

        :param chart_time: The chart bar's open time in ms.
        :param next_chart_time: Open of the following chart bar, 0 when none.
        :param confirmed: Whether the chart bar is closed (historical/confirmed).
        """
        cal = _chart_calendar()
        round_state['chart_time'] = chart_time
        round_state['next_time'] = next_chart_time
        if cal is not None and _chart_tf[0]:
            bar_close = actual_bar_close(chart_time, next_chart_time, cal, _chart_tf[0])
        else:
            bar_close = chart_time + _chart_off[0] + 1
        if confirmed:
            tick = bar_close
        else:
            # A developing chart bar's as-of is "now" — but it must stay INSIDE
            # the bar. The wall clock only does that when the feed really is
            # real time; replaying recorded bars through the live transport
            # puts it far beyond them, and an as-of past a peer's developing
            # close_A asks for a bar nothing will ever publish (a developing
            # bar never appends, so the peer's frontier stops one tick below
            # its own close_A).
            tick = int(datetime.now().timestamp() * 1000)
            if tick >= bar_close:
                tick = bar_close - 1
            if tick < chart_time:
                tick = chart_time
        round_state['tick'] = tick
        if cal is not None and cal.opening_hours:
            extended = break_end_after(tick, cal)
            round_state['sched_next_open'] = 0 if extended == tick else extended
        else:
            round_state['sched_next_open'] = 0

    _chart_tf = ['']
    _chart_off = [0]
    for _st in states.values():
        _chart_tf[0] = _st.chart_timeframe
        _chart_off[0] = _st.chart_off
        break

    def _settle_round(sec_id: str, state: SecurityState) -> None:
        """Wait until the child finished every round this context launched.

        One chart bar can launch SEVERAL rounds on a live context (prefill,
        closed override, developing), so a boolean ``done_event`` cannot say
        whether the child is still unpacking an earlier round's slot — a
        counter can. Overwriting the slot under an unpacking child mixes two
        bars' OHLCV in one read.
        """
        if not state.needs_wait:
            return
        while sync_block.get_rounds_done(sec_id) < state.rounds_launched:
            _wait_with_liveness(state.done_event, sec_id, sec_processes, failed_children)
            state.done_event.clear()
        state.done_event.clear()
        state.needs_wait = False

    def _settle_no_round(sec_id: str, state: SecurityState) -> None:
        """Publish the round's as-of as the frontier of a context NOT launched.

        A producer's frontier reaches exactly as far as what it published: a
        round goes no further than its own as-of instant, deliberately — a wider
        claim would let a consumer running on the NEXT tick pass its wait and
        read the value of this one. So whoever settles the round has to
        raise the frontier: the producer itself at its write when the chart
        launches it, and the chart here, at the producer's own signal, when it
        decides there is no round to run.

        Skipping the launch IS the statement that no bar of this context closes
        at or before the chart's as-of beyond what its ring already holds, so
        the raise is exactly as true as that decision — and a consumer's own
        as-of never exceeds it, which is what releases the waiter.

        Nothing can be waiting on a context that has not been started: a
        consumer's child only exists once ``_start`` ran for it, and that starts
        every producer it depends on FIRST. An unstarted producer therefore
        publishes no frontier — it would be a claim about a ring no round of its
        own has filled yet.
        """
        if not state.started:
            return
        cond = ring_conditions.get(sec_id)
        if cond is None or not consumers_by_sid.get(sec_id):
            return
        asof = chart_asof(state, round_state['chart_time'], round_state['next_time'],
                          0 if lib.barstate.isconfirmed else round_state['tick'])
        with cond:
            if asof > sync_block.get_frontier_close(sec_id):
                sync_block.set_frontier_close(sec_id, asof)
            cond.notify_all()

    def _start(sec_id: str, state: SecurityState, *, run_pending: bool = True) -> None:
        """Start a context: its producers, its child process, its first round.

        Called from the FIRST ``__sec_read__`` of the context (and from
        ``_launch`` for the shapes that cannot be started late, see below).
        Until then the signal did all its per-bar bookkeeping but launched
        nothing, so the whole start is one late round against the target the
        signal last wrote — the child runs its feed up to that target, which is
        exactly where the per-bar rounds would have left it.

        Producers first, transitively: this context's child reads every sid in
        ``depends`` out of that sid's ring and waits on its frontier, so a
        producer must be running — and launched for this bar — before the
        consumer's child exists. ``started`` is set before the recursion, so two
        contexts reading each other cannot recurse forever.

        A CONSUMER of this context is deliberately NOT started: a consumer that
        is never read stays unstarted and keeps its watermark at zero, which
        only stops the producer's ring from being garbage-collected (the writer
        grows the ring instead of ever blocking), and keeping the whole history
        is also what lets that consumer start correctly later.

        The first cold start of a BATCHED context also starts every other
        batched context the transformer flagged ``always_read``, because their
        cold starts are what the chart's first bar would otherwise queue up one
        behind the other. A context read only behind a branch is never in that
        group — it starts at its own first read, or not at all.

        :param sec_id: The context's id.
        :param state: Its runtime state.
        :param run_pending: Whether to run the round the signal prepared. False
                            from ``_launch``, which is about to launch its own.
        """
        state.started = True
        for producer_id in state.depends:
            producer = states.get(producer_id)
            if producer is not None and not producer.started:
                _start(producer_id, producer)
        if sec_id in no_process_ids:
            # Chart-served (same symbol+timeframe) or ignored: no child, no
            # round — the inline write/read path answers it.
            return
        if lazy_spawn_fn is not None:
            lazy_spawn_fn(sec_id)
            if state.batch_target and not cold_batch_start[0]:
                # Cold start of a BATCHED context: every other batched context
                # the transformer proved is read on every bar starts with it. A
                # batch round is the child's whole historical phase, and the
                # chart's first read of it blocks until that child has booted
                # and caught up to this bar — so starting them one read after
                # the other serializes all of it into the first chart bar
                # (MEASURED on a 19-context screener: 7.3 s became 32.8 s, all
                # of it the 19 cold starts standing in a queue on bar one).
                # Running ahead of the chart IN PARALLEL is what a batch round
                # is for, so the group is started as a group. ``always_read``
                # is what keeps that strictly lazy: a context read only behind
                # a branch never joins the group, so a branch this run does not
                # take costs no child process at all.
                cold_batch_start[0] = True
                for other_id, other in states.items():
                    if (other_id != sec_id and other.batch_target
                            and other.always_read and not other.started):
                        _start(other_id, other)
        if not run_pending:
            return
        if state.deferred_steps:
            # Developing transport: run the whole queued round sequence now.
            # The first step goes out here and ``__sec_read__`` drives the rest
            # (``_drive_pending``), exactly as it does for a live chart bar's
            # own multi-step round.
            steps = state.deferred_steps
            state.deferred_steps = []
            state.pending_live = steps[1:]
            steps[0]()
        elif state.pending_launch:
            _launch(sec_id, state)
            if sync_block.get_flags(sec_id) & (FLAG_BATCH_ROUND
                                               | FLAG_DEV_BATCH_ROUND):
                # The deferred round IS the child's whole historical phase: its
                # values reach the chart through the ring while it runs, so no
                # later bar may block on it — the same invariant the batch
                # signal sets when it launches the round itself.
                state.needs_wait = False
        else:
            # Nothing to run: every bar so far decided this context has no fresh
            # bar for the chart. Its consumers still have to be released.
            _settle_no_round(sec_id, state)

    def _launch(sec_id: str, state: SecurityState, *, lazy_ok: bool = True) -> None:
        """Hand the prepared slot to the child and count the round.

        A context the chart has not read yet has no child process: the slot the
        caller prepared stays in shared memory and only the wake-up is deferred
        (``pending_launch``), for ``_start`` to perform at the first read. One
        late round then reaches the same target the per-bar rounds would have
        left the child at, because the child runs its feed up to the target it
        is given.

        ``lazy_ok=False`` is for the rounds that cannot be replaced by a later
        one: the LTF paths (``request.security_lower_tf`` and the scalar
        ``plain_ltf`` merge), whose round IS the chart bar's own intrabar window
        — a later round would window a different bar — and the developing-bar
        transport's steps, which carry one specific chart bar's aggregated
        OHLCV. Those steps are QUEUED while the context is unstarted
        (``deferred_steps``) and run in order at the first read, so by the time
        they call this the context is started anyway; the flag only matters for
        a developing-transport context that produces for or consumes another,
        which is never queued and so starts at its first signal as before.

        The round context travels WITH the target, before the advance: the
        child unpacks both at the start of its round and derives its chart-side
        as-of cap from its own slot, never from a peer's, whose slot the chart
        may already have written for the next round.

        ``FLAG_MORE_STEPS`` rides along: one chart bar can queue several rounds
        for the same context (prefill, closed, developing) that all share the
        SAME ``round_tick``, and a step that is not the last one must not
        advertise a frontier reaching that tick -- a consumer released by it
        would read this step's value instead of waiting for the pending
        developing append at the very same instant. ``pending_live`` holds the
        steps still to come, so it is filled BEFORE the first step runs and each
        further step pops itself off before launching.
        """
        if not state.started:
            if lazy_ok:
                state.pending_launch = True
                return
            _start(sec_id, state, run_pending=False)
        state.pending_launch = False
        flags = sync_block.get_flags(sec_id)
        if state.pending_live:
            flags |= FLAG_MORE_STEPS
        else:
            flags &= ~FLAG_MORE_STEPS
        sync_block.set_flags(sec_id, flags)
        sync_block.set_round_context(sec_id, round_state['tick'],
                                     round_state['sched_next_open'])
        state.data_ready.clear()
        state.rounds_launched += 1
        state.advance_event.set()
        state.needs_wait = True

    def _drive_pending(sec_id: str, state: SecurityState) -> None:
        """Run the queued live steps of one context to the end.

        ``__sec_signal__`` launches the FIRST step and returns without waiting,
        so the chart can go on to signal further contexts; each further step
        waits for the previous step's value (``data_ready``, set at the child's
        write) and then launches the next. Driving them from the read — and from
        the runner's bar-end hook for the steps no read reaches — keeps a chart
        bar free of any wait that a child's own peer read could extend.
        """
        while state.pending_live:
            _wait_with_liveness(state.data_ready, sec_id, sec_processes, failed_children)
            step = state.pending_live.pop(0)
            step()

    def _batch_read(sec_id: str, state: SecurityState, default):
        """Pair one chart bar's value out of a batched context's ring.

        The same rule the rest of the module pairs by: the context's last bar
        whose scheduled close is at or before the chart's as-of instant for it
        — the very bar ``_get_confirmed_time`` picked as this round's target,
        since both read the same ascending ``bar_closes``. The child runs far
        ahead of the chart, so the wait below is normally already satisfied;
        it blocks only while the batch is still catching up, on the producer's
        condition rather than on a per-bar handshake. The wait also parks the
        chart's GC watermark at this as-of, which is both what the ring may
        collect below and what tells the producer whether waking the chart can
        release it.

        :param sec_id: The batched context's id.
        :param state: Its runtime state (``batch_asof`` set by the signal).
        :param default: Pine ``na`` for this read.
        :return: The paired value, or ``default`` when nothing has closed yet.
        """
        reader = batch_readers.get(sec_id)
        if reader is None:
            reader = RingReader(sec_id, sync_block, ring_conditions[sec_id])
            batch_readers[sec_id] = reader
        asof = state.batch_asof
        while not reader.wait_for_close(asof, state.stop_event,
                                        timeout=_LIVENESS_POLL_SECONDS,
                                        watermark_index=chart_wm_index,
                                        spin=state.dev_batch_spec is not None):
            # Only a dead producer can keep this from being satisfied: its next
            # unpublished bar closes above the chart's as-of, so publishing it
            # raises the frontier past this instant.
            if failed_children:
                dead = sec_id if sec_id in failed_children else next(iter(failed_children))
                proc = sec_processes.get(dead) if sec_processes is not None else None
                raise RuntimeError(
                    f"Security process for '{dead}' died unexpectedly "
                    f"(exit code: {proc.exitcode if proc is not None else '?'})"
                )
            if state.stop_event.is_set():
                return default
            proc = sec_processes.get(sec_id) if sec_processes is not None else None
            if proc is not None and not proc.is_alive():
                raise RuntimeError(
                    f"Security process for '{sec_id}' died unexpectedly "
                    f"(exit code: {proc.exitcode})"
                )
        entry = reader.last_close_at_or_before(asof)
        if entry is None:
            return default
        if (state.dev_batch_spec is not None
                and entry[1] <= state.batch_prev_asof):
            # Every entry a developing batch publishes for ONE chart bar closes
            # inside that bar's own window — after the previous bar's as-of and
            # at or before this one's. A developing record closes exactly at the
            # round tick; a record closing a security period carries that
            # period's own scheduled close, which is later than the previous
            # chart tick but can fall BEFORE this one when the two grids do not
            # align (a 60-minute context on a 45-minute chart completes the
            # hourly period at 01:00 on the bar ticking at 01:30). The child
            # reproduces the chart's bars from the same window and the same
            # cleaning the bar loop publishes them with, so the sequences are
            # the same by construction; if they were not, pairing would silently
            # answer with an EARLIER bar's value instead of failing. O(1), and
            # it covers every bar rather than only the ticks a pre-walked plan
            # could have been compared against.
            raise RuntimeError(
                f"security context '{sec_id}': the developing batch published "
                f"no round in ({state.batch_prev_asof}, {asof}] "
                f"(nearest close {entry[1]}) — the child's chart bar stream "
                f"diverged from the bar loop's"
            )
        return entry[2]

    def __sec_signal__(sec_id: str, symbol: str | None = None,
                       timeframe: str | None = None, lookahead=None,
                       _scope_id=None):
        state = states[sec_id]

        # Resolve deferred symbol/timeframe on first call. The two callbacks are
        # NOT alternatives: in a script with both deferred and static contexts the
        # deferred resolver no-ops for a static sec_id (and the runtime symbol
        # argument is always present), so an elif here would leave every static
        # context's bar grid unloaded and every target of this bar computed off
        # an arithmetic guess. ``prepare_fn`` itself skips sids that already
        # have a process (a deferred context prepares inside the resolver).
        if sec_id not in resolved:
            resolved.add(sec_id)
            if lookahead is not None:
                # Input-derived (Pine "simple") lookahead: the transformer stored
                # None in __security_contexts__ and passes the actual value here.
                # Resolve the mode BEFORE the deferred symbol/timeframe callback,
                # which recomputes ``na_on_developing`` from ``state.lookahead``.
                state.lookahead = _lookahead_mode(lookahead)
                # Static symbol/timeframe with deferred lookahead: mirror the
                # setup-time cross-symbol decision (no aggregator ⇒ cross-symbol
                # HTF). A deferred symbol/timeframe context is recomputed by the
                # resolver below instead.
                state.na_on_developing = (
                    not state.is_ltf and not state.plain_ltf
                    and not state.same_timeframe
                    and state.htf_aggregator is None
                    and state.lookahead is Lookahead.ON
                )
            if deferred_resolve_fn is not None and symbol is not None:
                deferred_resolve_fn(sec_id, symbol, timeframe)
                # The resolver may have turned this context into the chart's
                # own: it now produces for its consumers, so it needs a ring.
                _ensure_chart_writer(sec_id)
            if prepare_fn is not None:
                prepare_fn(sec_id)

        # No-process contexts (same-context, ignored): skip advance/wait
        if sec_id in no_process_ids:
            # Nothing to start: the chart serves this context inline, so it is
            # "started" from the first bar — its consumers read the ring the
            # chart itself writes.
            state.started = True
            if sec_id in same_context_ids:
                state.new_period = True
                state.data_ready.clear()
            return

        # One outstanding round per context. ``__sec_wait__`` normally collects
        # the previous round, but it only runs where the script actually reads
        # the context — a read behind a conditional leaves the round in flight,
        # and the writes below would then overwrite the SyncBlock slot the
        # subprocess is still unpacking. ``set_developing_bar`` packs 48 bytes
        # in one call, not one atomic store, so the child can observe a mix of
        # both bars: MEASURED on ICT Master Suite — parent wrote
        # (o=115055.03 h=115127.81 l=112650.0 c=113848.3 t=1754352000000), the
        # child read (o=115055.03 h=0 l=0 c=0 t=0) and the script divided by it.
        _settle_round(sec_id, state)

        # noinspection PyProtectedMember
        chart_time = lib._time

        if state.plain_ltf:
            # Plain (scalar) request.security with a timeframe FINER than the
            # chart's. TradingView merge rule: ``lookahead_off`` returns the
            # expression value on the LAST intrabar of the chart bar's own
            # period, ``lookahead_on`` on the FIRST. The child's historical
            # loop runs every feed bar with ``bar_open <= target`` and the
            # last per-intrabar write wins, so the target IS the merge rule:
            # the bar's period end for OFF, the bar's open for ON.
            if state.lookahead is Lookahead.ON:
                target_time = chart_time
                # Prefix skip: a chart bar opening before the LTF feed's first
                # bar has no intrabar to merge — TradingView returns ``na``
                # before the lower-timeframe series begins.
                if state.ltf_first_ms is not None and target_time < state.ltf_first_ms:
                    state.ltf_skip = True
                    state.new_period = True
                    state.needs_wait = False
                    _settle_no_round(sec_id, state)
                    return
            elif state.bar_opens is not None and state.bar_closes is not None:
                # File-backed: the last intrabar whose scheduled close reaches
                # the chart bar's own close. A straddling intrabar closes after
                # it and is left for the next chart bar.
                idx = _ltf_bar_index(
                    state, chart_time, round_state['next_time'],
                    0 if lib.barstate.isconfirmed else round_state['tick'])
                if idx < 0:
                    state.ltf_skip = True
                    state.new_period = True
                    state.needs_wait = False
                    _settle_no_round(sec_id, state)
                    return
                target_time = state.bar_opens[idx]
            elif (state.chart_dwm_modifier and state.chart_resampler is not None
                    and state.chart_resampler.get_bar_time(chart_time, state.tz)
                    == chart_time):
                # Single-period civil D/W/M chart bar: no fixed arithmetic
                # span (``chart_off`` is 0) — target the civil period end.
                target_time = _next_civil_period_open(
                    state.chart_dwm_modifier, chart_time, state.tz) - 1
            else:
                target_time = chart_time + state.chart_off
            # Live stream with no static file: a developing chart bar's period
            # end lies in the future, so clamp to the last surely-closed
            # intrabar instead. The round's fixed tick, not a fresh clock read:
            # a peer whose target was set microseconds earlier must not be asked
            # for a bar it was never targeted at in this round.
            if (state.bar_closes is None and state.is_live
                    and not lib.barstate.isconfirmed and state.plain_ltf_span_ms):
                now_ms = round_state['tick']
                elapsed_close = (now_ms // state.plain_ltf_span_ms
                                 ) * state.plain_ltf_span_ms - 1
                if elapsed_close < target_time:
                    target_time = elapsed_close
            state.ltf_skip = False
            state.new_period = True
            sync_block.set_target_time(sec_id, target_time)
            _launch(sec_id, state, lazy_ok=False)
            return

        if state.is_ltf:
            # Live streaming LTF (PluginSymbol source): the chart bar may be
            # developing, so read-ahead is impossible. The subprocess pulls
            # intrabars from its own LTF streamer and builds the chart period's
            # window (closed intrabars + the developing intrabar as the live last
            # element). The parent ships only the period bounds and whether the
            # chart bar is still developing; warmup replay (confirmed chart bars)
            # flows through the same path and yields full closed periods.
            if state.ltf_live_stream:
                period_start = chart_time
                if (state.chart_dwm_modifier and state.chart_resampler is not None
                        and state.chart_resampler.get_bar_time(chart_time, state.tz)
                        == chart_time):
                    period_end_exclusive = _next_civil_period_open(
                        state.chart_dwm_modifier, chart_time, state.tz)
                else:
                    period_end_exclusive = chart_time + state.chart_off + 1
                ltf_flags = sync_block.get_flags(sec_id) & ~(
                    FLAG_IS_DEVELOPING | FLAG_CLOSED_OVERRIDE
                )
                ltf_flags |= FLAG_LTF_WINDOW
                if lib.barstate.isconfirmed:
                    ltf_flags &= ~FLAG_LTF_CHART_DEVELOPING
                else:
                    ltf_flags |= FLAG_LTF_CHART_DEVELOPING
                if state.is_live:
                    ltf_flags |= FLAG_LTF_LIVE_PHASE
                else:
                    ltf_flags &= ~FLAG_LTF_LIVE_PHASE
                sync_block.set_flags(sec_id, ltf_flags)
                sync_block.set_target_time(sec_id, period_start)
                sync_block.set_ltf_period_end(sec_id, period_end_exclusive)
                state.ltf_skip = False
                state.new_period = True
                _launch(sec_id, state, lazy_ok=False)
                return

            # Historical/file-backed LTF: the child includes intrabars with
            # ``bar_open <= target_time``. The target is the LAST intrabar whose
            # scheduled close reaches the chart bar's own close, so the array
            # never carries a price from after that close: on a nested grid this
            # is exactly the chart bar's own period, and on a non-nested one (a
            # 4-minute context on a 3-minute chart) or behind a shortened
            # session the straddling intrabar goes to the next chart bar's
            # round. ``bar_opens`` is ``None`` only for an empty static feed (no
            # streamer), which has no intrabars to window at all.
            if state.bar_opens is None or state.bar_closes is None:
                ltf_target_time = chart_time
            else:
                idx = _ltf_bar_index(
                    state, chart_time, round_state['next_time'],
                    0 if lib.barstate.isconfirmed else round_state['tick'])
                if idx < 0:
                    # Nothing has closed inside this chart bar yet: the read is
                    # the empty-array default, so skip the cross-process
                    # signal+wait entirely.
                    state.ltf_skip = True
                    state.new_period = True
                    state.needs_wait = False
                    _settle_no_round(sec_id, state)
                    return
                ltf_target_time = state.bar_opens[idx]
            state.ltf_skip = False
            # LTF: every chart bar needs intrabar data — always signal. The
            # period start bounds the flushed array to the bar's OWN intrabars:
            # the child still replays any earlier feed bars for expression
            # state, but their values are prefix, not array content.
            state.new_period = True
            sync_block.set_ltf_period_start(sec_id, chart_time)
            sync_block.set_target_time(sec_id, ltf_target_time)
            _launch(sec_id, state, lazy_ok=False)
            return

        # Live HTF transport — the chart aggregates its own OHLCV into the
        # containing HTF bar via ``HTFAggregator`` and ships it to the
        # subprocess on the SyncBlock. The static ``.ohlcv`` file cannot
        # grow at runtime, so this transport is the *only* way for any
        # lookahead mode to advance an HTF security context live.
        #
        # Phase 1 (closed-bar override): every HTF period close pushes the
        # newly closed OHLCV to the subprocess synchronously. All lookahead
        # modes use this phase — it is the live equivalent of reading the
        # next ``.ohlcv`` bar in historical mode.
        #
        # Phase 2 (developing bar): only ``Lookahead.ON`` exposes the
        # in-progress HTF bar with ``barstate.isconfirmed=False``; OFF and
        # LAST_CLOSED stay repaint-free and skip this phase.
        #
        # Seed the aggregator on EVERY chart bar (warmup included). If we
        # only fed it once ``is_live`` flipped, a live transition that
        # happens mid-HTF-period would lose all warmup bars belonging to
        # the in-progress period, and the first developing/closed override
        # emitted live would carry partial OHLCV (open/high/low/volume
        # missing the prior chart bars).
        if state.htf_aggregator is not None:
            chart_open = float(lib.open)
            chart_high = float(lib.high)
            chart_low = float(lib.low)
            chart_close = float(lib.close)
            raw_vol = lib.volume
            chart_volume = 0.0 if raw_vol is None else float(raw_vol)

            _, dev_bar, closed_bar = state.htf_aggregator.update(
                chart_time, chart_open, chart_high, chart_low,
                chart_close, chart_volume,
                chart_confirmed=bool(lib.barstate.isconfirmed),
            )

            if state.dev_batch_spec is not None:
                # Historical DEVELOPING batch: the child produces and replays
                # its whole round sequence while the chart walks its bars, and
                # ``__sec_read__`` pairs each bar's value out of the ring by
                # as-of. Nothing is pushed and nothing is launched per bar; the
                # aggregator ran above only for the ``new_period`` bookkeeping
                # (which chart bar opens a fresh security period, the ``gaps_on``
                # na/value selection), and the as-of the read pairs with is this
                # chart bar's own round tick — exactly the close the developing
                # record of this bar carries.
                state.batch_prev_asof = state.batch_asof
                state.batch_asof = round_state['tick']
                # The per-bar path sets ``new_period`` True whenever a developing
                # step goes out, and otherwise from whether a period just closed.
                state.new_period = (dev_bar is not None
                                    or closed_bar is not None)
                if not state.batch_launched:
                    state.batch_launched = True
                    sync_block.set_flags(sec_id, (sync_block.get_flags(sec_id) & ~(
                        FLAG_IS_DEVELOPING | FLAG_CLOSED_OVERRIDE
                        | FLAG_DEV_HISTORICAL
                    )) | FLAG_DEV_BATCH_ROUND)
                    sync_block.set_target_time(sec_id, state.batch_target)
                    _launch(sec_id, state)
                    # The chart never settles this round as a whole — its values
                    # arrive through the ring while it runs — so no later bar is
                    # allowed to block on it.
                    state.needs_wait = False
                return

            # ``Lookahead.ON`` takes this transport in HISTORICAL mode too. The
            # child's own data file holds the containing period's COMPLETE bar,
            # so letting it read that bar hands the script the period's final
            # close and high on the period's very first chart bar — TV does
            # exactly that, and PyneCore does not reproduce lookahead (see the
            # module docstring). The aggregator is already fed on every chart
            # bar including warmup, so the partial bar needed to avoid the leak
            # is available here at no extra cost. ``FLAG_DEV_HISTORICAL`` tells
            # the subprocess to keep history barstate: the transport is the live
            # one, the bar is not.
            # With ``ALLOW_LOOKAHEAD`` a historical ``ON`` falls through to the
            # closed-only flow instead, where ``_get_confirmed_time`` steps into
            # the containing period and the child reads its complete bar. Live
            # mode keeps the developing transport either way — a live bar has no
            # completed future to read.
            dev_transport = state.is_live or (
                state.lookahead is Lookahead.ON and not ALLOW_LOOKAHEAD)
            hist_phase = 0 if state.is_live else FLAG_DEV_HISTORICAL

            # Phase 0 (one-time historical prefill): the developing transport
            # never lets the child read its own ``.ohlcv`` file, so on its own
            # the child's HTF series would begin at the chart's first bar and
            # every ``[1]`` read inside the first period would return na.
            # MEASURED on the daily-pivot idiom
            # ``security(tickerid, "D", close[1], lookahead_on)``: a 30m chart
            # opening 2025-01-01 got TV's 2024-12-31 pivot but na from PyneCore
            # for that whole first day.
            #
            # Replaying the periods that closed BEFORE the containing one leaks
            # nothing — they are complete past bars, exactly the ones the
            # closed-only flow below would have delivered. Only the containing
            # period stays on the aggregated developing bar, so the no-lookahead
            # rule is untouched. ``last_confirmed`` guards the live case, where
            # the closed-only warmup already advanced the child further.
            if dev_transport:
                # The live steps of this chart bar, in order. Only the FIRST is
                # launched here; the rest are driven by ``__sec_read__`` (and by
                # the runner's bar-end hook for the steps no read reaches), each
                # waiting for the previous step's VALUE rather than for the
                # child's whole ``main()``. Launching them all synchronously in
                # the signal block deadlocks two contexts that read each other:
                # the chart would sit in a wait that a child's own peer read
                # extends.
                steps: list = []
                # Whether this bar's rounds are queued for the first read
                # instead of being run now. Only for a context that neither
                # produces for nor consumes another: every round of the burst
                # carries the FIRST READ's chart tick, which is what a ring
                # entry's developing close and a peer read's as-of cap are
                # derived from — an isolated context publishes no ring and caps
                # no peer, so nothing observes the difference.
                defer_steps = (not state.started and not state.depends
                               and not consumers_by_sid.get(sec_id))

                if not state.htf_prefilled:
                    state.htf_prefilled = True
                    containing = dev_bar if dev_bar is not None else closed_bar
                    prefill_target = (containing.period_start - 1
                                      if containing is not None else 0)
                    if prefill_target > state.last_confirmed:
                        state.last_confirmed = prefill_target

                        def _prefill_step(_target=prefill_target):
                            sync_block.set_flags(
                                sec_id,
                                sync_block.get_flags(sec_id) & ~(
                                    FLAG_IS_DEVELOPING | FLAG_CLOSED_OVERRIDE
                                    | FLAG_DEV_HISTORICAL
                                ),
                            )
                            sync_block.set_target_time(sec_id, _target)
                            _launch(sec_id, state, lazy_ok=False)

                        steps.append(_prefill_step)

                if closed_bar is not None:
                    state.last_confirmed = closed_bar.period_start

                    def _closed_step(_bar=closed_bar):
                        sync_block.set_developing_bar(
                            sec_id, _bar.open, _bar.high, _bar.low,
                            _bar.close, _bar.volume, _bar.period_start,
                        )
                        sync_block.set_flags(sec_id, (
                            sync_block.get_flags(sec_id)
                            & ~(FLAG_IS_DEVELOPING | FLAG_DEV_HISTORICAL)
                        ) | FLAG_CLOSED_OVERRIDE | hist_phase)
                        sync_block.set_target_time(sec_id, _bar.period_start)
                        _launch(sec_id, state, lazy_ok=False)

                    steps.append(_closed_step)

                # Developing bar — only for ``Lookahead.ON``. ``dev_bar`` is
                # None when the confirmed chart bar just completed the period
                # (the closed step delivered it); no fresh developing bar exists
                # until the next chart bar.
                if state.lookahead is Lookahead.ON and dev_bar is not None:
                    # The aggregator keeps ONE developing bar per period and
                    # mutates it on every chart bar, so the values are copied
                    # into the step instead of the object: a queued step
                    # (``deferred_steps``) runs bars later and has to carry the
                    # OHLCV of the bar it was built for.
                    def _developing_step(_o=dev_bar.open, _h=dev_bar.high,
                                         _l=dev_bar.low, _c=dev_bar.close,
                                         _v=dev_bar.volume,
                                         _ps=dev_bar.period_start):
                        sync_block.set_developing_bar(sec_id, _o, _h, _l, _c,
                                                      _v, _ps)
                        sync_block.set_flags(sec_id, (
                            sync_block.get_flags(sec_id)
                            & ~(FLAG_CLOSED_OVERRIDE | FLAG_DEV_HISTORICAL)
                        ) | FLAG_IS_DEVELOPING | hist_phase)
                        sync_block.set_target_time(sec_id, _ps)
                        _launch(sec_id, state, lazy_ok=False)

                    steps.append(_developing_step)
                    state.new_period = True
                else:
                    # ``new_period`` reflects whether a fresh HTF close just
                    # landed (drives the ``gaps_on`` na/value selection in
                    # ``__sec_read__``).
                    state.new_period = closed_bar is not None
                    if closed_bar is None:
                        # Clear any stale developing flag from a prior
                        # ``Lookahead.ON`` session (same SyncBlock slot).
                        sync_block.set_flags(sec_id, sync_block.get_flags(sec_id) & ~(
                            FLAG_IS_DEVELOPING | FLAG_CLOSED_OVERRIDE
                        ))

                if steps:
                    if defer_steps:
                        # Not read yet: keep this bar's rounds for the first
                        # read instead of running them. EVERY round is kept, in
                        # bar order: the child rolls back a developing re-tick's
                        # var and function-instance slots but not its ``varip``
                        # slots or its ``IBPersistent`` storage, so a dropped
                        # developing round would leave the child in a state an
                        # eager start never reaches.
                        state.deferred_steps.extend(steps)
                    else:
                        # The queue is filled FIRST: ``_launch`` reads it to
                        # decide whether this round is the chart bar's last
                        # publication.
                        state.pending_live = steps[1:]
                        steps[0]()
                else:
                    _settle_no_round(sec_id, state)
                return
            # Historical OFF / LAST_CLOSED warmup falls through to the
            # closed-only flow below; the aggregator state has already advanced
            # so the live transition starts with the correct in-progress HTF bar.

        # Closed-only flow (historical / lookahead_off / lookahead_last_closed)
        target_time = _get_confirmed_time(
            state, chart_time, round_state['next_time'],
            0 if lib.barstate.isconfirmed else round_state['tick'])

        if state.batch_target:
            # Historical BATCH: the child runs its whole historical phase in a
            # single round while the chart walks its bars, and ``__sec_read__``
            # pairs each bar's value out of the ring. Nothing is launched per
            # bar; only the ``gaps_on`` bookkeeping (which chart bar opens a
            # fresh security period) stays, and the as-of the read pairs with.
            state.batch_asof = chart_asof(
                state, chart_time, round_state['next_time'],
                0 if lib.barstate.isconfirmed else round_state['tick'])
            state.new_period = target_time > state.last_confirmed
            if state.new_period:
                state.last_confirmed = target_time
            if not state.batch_launched:
                state.batch_launched = True
                sync_block.set_flags(sec_id, (sync_block.get_flags(sec_id) & ~(
                    FLAG_IS_DEVELOPING | FLAG_CLOSED_OVERRIDE
                )) | FLAG_BATCH_ROUND)
                sync_block.set_target_time(sec_id, state.batch_target)
                _launch(sec_id, state)
                # The chart never settles this round as a whole — its values
                # arrive through the ring while it runs — so no later bar is
                # allowed to block on it.
                state.needs_wait = False
            elif target_time > state.batch_target:
                # Safety net: the chart asked past what the batch was planned
                # for. Settle it (the child must not have its slot rewritten
                # while it is still unpacking one) and run an ordinary round.
                state.needs_wait = True
                _settle_round(sec_id, state)
                state.batch_target = target_time
                sync_block.set_flags(sec_id, sync_block.get_flags(sec_id) & ~(
                    FLAG_IS_DEVELOPING | FLAG_CLOSED_OVERRIDE | FLAG_BATCH_ROUND
                ))
                sync_block.set_target_time(sec_id, target_time)
                _launch(sec_id, state)
                state.needs_wait = False
            return

        if target_time > state.last_confirmed:
            state.last_confirmed = target_time
            state.new_period = True
            # Make sure no stale developing/override flag leaks across modes.
            stale_flags = sync_block.get_flags(sec_id) & ~(
                FLAG_IS_DEVELOPING | FLAG_CLOSED_OVERRIDE
            )
            sync_block.set_flags(sec_id, stale_flags)
            sync_block.set_target_time(sec_id, target_time)
            _launch(sec_id, state)
        else:
            state.new_period = False
            _settle_no_round(sec_id, state)

    def __sec_write__(sec_id: str, value, _scope_id=None):
        if sec_id not in same_context_ids or result_blocks is None:
            return
        state = states[sec_id]
        if sec_id in chart_bar_written:
            # A second write on one chart bar: legitimate only from a loop body,
            # and only with a loop-INVARIANT value (a consumer may already have
            # read the first one).
            if not state.in_loop:
                raise AssertionError(
                    f"security context '{sec_id}' wrote twice on one bar "
                    f"outside a loop"
                )
            if _same_value(chart_bar_value.get(sec_id), value):
                return
            raise RuntimeError("loop-varying security expression is not supported")
        chart_bar_written.add(sec_id)
        chart_bar_value[sec_id] = value
        with state.result_lock:
            write_result(result_blocks[sec_id], sync_block, value)
        state.data_ready.set()
        writer = chart_writers.get(sec_id)
        if writer is not None and round_state['tick']:
            # A chart-context producer publishes on the chart's own grid: the
            # chart bar's open and its scheduled close. A developing chart bar
            # is never appended — its re-runs would scatter entries sharing one
            # open — so the frontier stops one ms below its close.
            # noinspection PyProtectedMember
            confirmed = bool(lib.barstate.isconfirmed)
            if confirmed:
                writer.append(round_state['chart_time'], round_state['tick'], value,
                              chart_consumer_indexes.get(sec_id),
                              round_state['tick'] - 1)
            else:
                # The developing value stays unpublished, but the frontier must
                # still reach this round's tick: a consumer's as-of never goes
                # above it (``begin_bar`` clamps the tick inside the bar), so
                # without this an HTF ``lookahead_on`` peer would wait for a bar
                # this producer never appends — while the chart is parked on
                # that peer's result. Pairing then answers with the last CLOSED
                # chart bar, which is what a chart-context producer exposes.
                writer.set_frontier_close(round_state['tick'])

    def _convert_currency(conversion: tuple[str, str], result):
        """Apply a context's ``currency=`` conversion to a read value."""
        from ..lib import request
        from math import isnan
        from_cur, to_cur = conversion
        rate = request.currency_rate(from_cur, to_cur)
        if isnan(rate):
            return result
        if isinstance(result, (int, float)):
            return result * rate
        if isinstance(result, tuple):
            return tuple(
                v * rate if isinstance(v, (int, float)) else v for v in result
            )
        return result

    def __sec_read__(sec_id: str, default=None, _scope_id=None):
        # ``ignore_invalid_symbol=True`` may downgrade a live security to
        # ``no-process`` after syminfo prefetch fails — no subprocess is
        # ever spawned, so ``data_ready`` would never be set and a plain
        # ``_wait_with_liveness`` would deadlock here. Short-circuit to
        # ``default`` (Pine ``na``) so the script keeps running.
        if sec_id in no_process_ids and sec_id not in same_context_ids:
            return default
        state = states[sec_id]
        if state.ltf_skip:
            # LTF chart bar before the feed began: ``__sec_signal__`` skipped the
            # handshake, so the result is the empty-array default (identical to
            # the empty-buffer flush an unskipped bar would have produced). The
            # flag is current-bar-fresh: SecurityTransformer emits every
            # context's ``__sec_signal__`` at ``main()``'s start, ahead of any
            # ``__sec_read__``, so each read observes this bar's flag — the same
            # signal-before-read invariant ``new_period``/``needs_wait`` rely on.
            return default
        if not state.started:
            # FIRST read of this context: start its child now and run the round
            # its signal prepared for this bar (see ``_start``). A context no
            # read ever reaches never gets here — and never gets a process.
            _start(sec_id, state)
        if state.batch_target:
            # Batched context: no per-bar handshake ran, the value comes from
            # the ring. ``gaps_on`` still emits ``na`` on a chart bar that opens
            # no fresh security period, exactly as below.
            batch_result = _batch_read(sec_id, state, default)
            if state.gaps_on and not state.new_period:
                batch_result = default
            if (currency_conversions and sec_id in currency_conversions
                    and batch_result is not default):
                return _convert_currency(currency_conversions[sec_id], batch_result)
            return batch_result

        if state.pending_live:
            _drive_pending(sec_id, state)
        _wait_with_liveness(state.data_ready, sec_id, sec_processes, failed_children)

        if not state.is_ltf and not state.new_period:
            # gaps_on emits ``na`` between HTF closes (Pine semantics).
            # na_on_developing emits ``na`` while inside an open cross-symbol
            # HTF period when lookahead_on is requested (developing bar cannot
            # be aggregated). Both share the same shape: ``na`` whenever the
            # chart bar is not opening a fresh HTF period.
            if state.gaps_on or state.na_on_developing:
                return default

        with state.result_lock:
            result = readers[sec_id].read(sync_block, default)

        if currency_conversions and sec_id in currency_conversions and result is not default:
            result = _convert_currency(currency_conversions[sec_id], result)

        return result

    def __sec_wait__(sec_id: str, _scope_id=None):
        _settle_round(sec_id, states[sec_id])

    def end_bar() -> None:
        """Close out a chart bar cycle, whatever ``main()`` did.

        Runs from the runner after ``main()`` returns, so an early ``return``
        that skipped a write block or a ``__sec_wait__`` cannot strand anything:
        a chart-context producer that did not write still raises its frontier
        (or a consumer waiting on it would never be released), and the live
        steps no read reached are driven to the end here.
        """
        for sec_id, state in states.items():
            if state.pending_live:
                _drive_pending(sec_id, state)
                _wait_with_liveness(state.data_ready, sec_id, sec_processes,
                                    failed_children)
        if round_state['tick']:
            # A confirmed bar's own close is the round tick, so the frontier
            # stops one ms below it; a developing round publishes nothing, so
            # its frontier may reach the tick itself (see ``__sec_write__``).
            # noinspection PyProtectedMember
            frontier = (round_state['tick'] - 1 if lib.barstate.isconfirmed
                        else round_state['tick'])
            for sec_id, writer in chart_writers.items():
                if sec_id not in chart_bar_written:
                    writer.set_frontier_close(frontier)
        chart_bar_written.clear()
        chart_bar_value.clear()

    def cleanup():
        for r in readers.values():
            r.close()
        for br in batch_readers.values():
            br.close()
        for w in chart_writers.values():
            w.finish()
            w.close()
            w.unlink()

    def signal_rate_sources():
        """Advance every auto-spawned rate-source subprocess by one bar.

        No Pine call drives ``__auto_rate_*`` sec_ids (they are synthetic
        contexts created by ``_autospawn_rate_sources``), so the chart loop
        is the only place that can tick them forward. Each rate-source
        subprocess runs the lightweight close-only loop in
        ``security_process._run_rate_source_loop``: advance → drain
        newly-closed bars → write the latest close to its ResultBlock → set
        data_ready. We wait synchronously for data_ready so the rate value
        :meth:`CurrencyRateProvider._lookup_sec` reads later in the same
        chart bar reflects the bars closed up to ``chart_time``.
        """
        if not auto_rate_sec_ids:
            return
        # noinspection PyProtectedMember
        chart_time = lib._time
        for sec_id in auto_rate_sec_ids:
            if sec_id in no_process_ids or sec_id not in states:
                continue
            state = states[sec_id]
            sync_block.set_target_time(sec_id, chart_time)
            state.data_ready.clear()
            state.advance_event.set()
            _wait_with_liveness(state.data_ready, sec_id, sec_processes, failed_children)

    for _sid in consumers_by_sid:
        _ensure_chart_writer(_sid)

    return (
        __sec_signal__, __sec_write__, __sec_read__, __sec_wait__,
        cleanup, signal_rate_sources, begin_bar, end_bar,
    )


def __ltf_unzip__(rows, n):
    """Transpose a row-major LTF tuple buffer into Pine's column-major arrays.

    ``request.security_lower_tf(sym, tf, (e0, ..., e{n-1}))`` returns a tuple of
    ``n`` arrays, where array ``i`` holds the per-intrabar values of ``e_i``. The
    LTF subprocess accumulates one ``(e0, ..., e{n-1})`` tuple per intrabar, so the
    raw result is row-major (a list of ``n``-tuples). This transposes it into the
    ``n`` column arrays the tuple-unpack expects, returning ``n`` empty arrays when
    the chart bar has no intrabars (e.g. the lower-timeframe feed does not reach
    that period).

    Inserted by ``SecurityTransformer`` only for tuple-valued
    ``request.security_lower_tf()`` calls; scalar calls read the array directly.

    :param rows: Per-intrabar value tuples (possibly empty when there are no
        intrabars).
    :param n: Tuple arity (number of expression elements).
    :return: Tuple of ``n`` lists, column-major.
    """
    if not rows:
        return tuple([] for _ in range(n))
    return tuple(list(col) for col in zip(*rows))


class SecurityChildContext:
    """
    Per-round / per-bar context the security child's protocol functions read.

    The bar loop fills these in before every ``main()`` run; ``__sec_write__``
    publishes with them and ``__sec_read__`` derives its as-of instant from
    them. A plain attribute holder on purpose — this sits in the per-bar hot
    path and must not grow a lookup layer.

    :ivar bar_open: Open instant (ms) of the bar being run.
    :ivar bar_close: That bar's scheduled close (``close_A``), in ms.
    :ivar frontier: Frontier close to publish with the bar's append — the next
        unpublished bar's ``close_A`` minus one ms, or the round's own as-of on
        a developing run, which is as far as such a run publishes.
    :ivar is_round_last: Whether this is the round's last bar, i.e. the chart
        may be released as soon as the value is written.
    :ivar developing: Whether this run is a developing (unconfirmed) one. Such a
        run appends at the round's fixed tick instead of a scheduled close, and
        takes that same tick as its own as-of base — the scheduled close lies in
        the future.
    :ivar round_tick: The round's fixed tick instant, written by the chart into
        this context's slot together with the target.
    :ivar round_sched_next_open: End of the scheduled break containing
        ``round_tick`` under the CHART's calendar, ``0`` when it is inside a
        session or the chart symbol trades round the clock.
    """

    __slots__ = ('bar_open', 'bar_close', 'frontier', 'is_round_last',
                 'developing', 'round_tick', 'round_sched_next_open')

    def __init__(self) -> None:
        self.bar_open = 0
        self.bar_close = 0
        self.frontier = 0
        self.is_round_last = False
        self.developing = False
        self.round_tick = 0
        self.round_sched_next_open = 0


class _PeerReadState:
    """Per-peer read state of one consumer, keyed by consumer bar and as-of.

    Same bar and same as-of returns the cached value; the same bar with a
    larger as-of (a developing tick, a live lower-timeframe clamp) is a fresh
    wait; a new bar that brings no new entry forward-fills (``gaps_off``) or
    yields ``na`` (``gaps_on``).
    """

    __slots__ = ('bar', 'asof', 'entry_close', 'value')

    def __init__(self) -> None:
        self.bar: int = -1
        self.asof: int = -1
        self.entry_close: int = -1
        self.value: Any = None


def create_security_protocol(
    sec_id: str,
    sync_block: SyncBlock,
    result_block: ResultBlock,
    all_sec_ids: list[str],
    result_locks: 'dict[str, LockType]',
    is_ltf: bool = False,
    *,
    registry: 'dict[str, dict] | None' = None,
    chart_calendar: 'BarCalendar | None' = None,
    ring_conditions: 'dict[str, ConditionType] | None' = None,
    consumer_ids: 'list[str] | None' = None,
    stop_event: 'EventType | None' = None,
    data_ready_event: 'EventType | None' = None,
    registry_pipe: 'Connection | None' = None,
    chart_ring_capacity: int = 0,
    chart_ring_arena: int = 0,
) -> tuple:
    """
    Create protocol functions for a **security** process.

    ``__sec_signal__`` and ``__sec_wait__`` are no-ops (the chart drives the
    rounds). ``__sec_write__`` publishes this context's value; ``__sec_read__``
    reads a peer context's value by the ONE pairing rule: the peer's last bar
    whose scheduled close is at or before this consumer's as-of instant.

    **Publication** (``__sec_write__``): the slot (so the chart can read it),
    then — when this context has consumers — a ring append of
    ``(bar_open, close_A, value)`` that raises the producer's frontier close in
    the same, condition-held step, and finally ``data_ready`` on the round's
    LAST bar. The chart therefore waits for the VALUE, never for the child's
    whole ``main()``; a peer read standing after this write can no longer hold
    the chart up.

    **As-of** (``__sec_read__``): the minimum of this bar's own scheduled close
    (a developing run takes the round's fixed tick instead — its close lies in
    the future), the scheduled-break extension for a daily/weekly/monthly peer
    keeping this context's calendar, and the chart's own as-of for that peer in
    this round, computed from this context's OWN round context. That last cap
    is what makes the wait terminate: a consumer can never ask for a peer bar
    the chart does not confirm in this round or an earlier one.

    **Waiting** (deadlock-freedom): the wait ends when an entry closes exactly
    at the as-of instant or the peer's frontier reaches it. The peer's next
    unpublished bar closes ABOVE the chart's as-of for it, hence above this
    one, so the frontier passes the as-of as soon as that bar is published —
    the wait always terminates, even when the chart does not wake the peer
    again this round. A consumer can only depend on producers whose write site
    precedes its own read in program order, and those have already published.

    :param sec_id: This security context's ID (the only slot it writes to).
    :param sync_block: Shared memory sync block
    :param result_block: Shared memory result block for writing
    :param all_sec_ids: All security context IDs (for cross-context reads)
    :param result_locks: Per-slot ``multiprocessing.Lock`` keyed by sec_id.
    :param is_ltf: If True, enable LTF accumulation mode.
    :param registry: Peer records by sec_id (``timeframe``, ``is_dwm``,
        ``gaps_on``, ``has_producer``, ``calendar``, ``depends``, ``in_loop``).
        ``None`` disables peer pairing entirely (no dependencies in the script).
    :param chart_calendar: The chart's trading schedule, for the chart-as-of cap.
    :param ring_conditions: Per-sid ``multiprocessing.Condition``, created by the
        parent and shared with every child.
    :param consumer_ids: Sids consuming THIS context — drives the ring GC
        watermark minimum and whether a ring is allocated at all.
    :param chart_ring_capacity: Non-zero when the CHART consumes this context's
        ring (its historical phase runs as one batch round). A ring is then
        allocated even with no security consumer, pre-sized to this many
        entries, and the chart's watermark row bounds its GC.
    :param chart_ring_arena: Payload arena size in bytes for that pre-sizing.
    :param stop_event: Shutdown event; a peer wait ends when it is set.
    :param data_ready_event: Set at the round's last write so the chart is
        released on the VALUE rather than at the end of ``main()``.
    :param registry_pipe: Child end of the parent's registry pipe. A context
        whose symbol or timeframe is only known at runtime is resolved AFTER
        its consumers were spawned, so its record cannot be in their snapshot;
        the chart pushes it down this pipe the moment it resolves, and a read
        of a not-yet-known dependency blocks on the pipe until it arrives.
    :return: (sec_signal, sec_write, sec_read, sec_wait, cleanup, flush,
             ltf_take_value, ltf_publish, buffer_len, ctx, after_bar, finish,
             prime_ring)
    """
    own_lock = result_locks[sec_id]
    ctx = SecurityChildContext()

    own_record = (registry or {}).get(sec_id) or {}
    own_calendar: BarCalendar = own_record.get('calendar') or BarCalendar()
    depends: frozenset[str] = frozenset(own_record.get('depends') or ())
    in_loop: bool = bool(own_record.get('in_loop', False))

    ring_conditions = ring_conditions or {}
    consumer_indexes: list[int] | None = None
    writer: RingWriter | None = None
    if (consumer_ids or chart_ring_capacity) and sec_id in ring_conditions:
        consumer_indexes = [sync_block.index_of(cid) for cid in (consumer_ids or ())]
        if chart_ring_capacity:
            # The chart pairs this context's values out of the ring for the
            # whole historical phase, so it bounds the GC like any other
            # consumer — from its own watermark row, having no slot of its own.
            consumer_indexes.append(sync_block.chart_index)
            writer = RingWriter(
                sec_id, sync_block, ring_conditions[sec_id],
                capacity=max(INITIAL_RING_CAPACITY, chart_ring_capacity),
                arena_size=max(INITIAL_RING_ARENA, chart_ring_arena))
        else:
            writer = RingWriter(sec_id, sync_block, ring_conditions[sec_id])

    known: dict[str, dict] = dict(registry or {})
    registry_lock = threading.Lock()
    registry_arrived = threading.Event()

    def _registry_reader() -> None:
        """Absorb registry records as they arrive, off the bar loop.

        The chart pushes every freshly resolved context's record to every
        child. A child parked in a ring wait cannot poll its pipe, so draining
        it only from the bar loop lets the records pile up until the pipe
        buffer is full and the chart's ``send`` blocks — and the chart is
        exactly who has to resolve the context that child is waiting for.
        Reading here keeps the chart's send non-blocking whatever the child is
        doing.
        """
        while True:
            try:
                record = registry_pipe.recv()  # type: ignore[union-attr]
            except (EOFError, OSError):
                # The chart closed the pipe (shutdown); nothing more arrives.
                break
            with registry_lock:
                known.update(record)
            registry_arrived.set()

    if registry_pipe is not None:
        threading.Thread(target=_registry_reader, name=f'sec-registry-{sec_id}',
                         daemon=True).start()

    def _peer_record(sid: str) -> 'dict | None':
        """This peer's record, waiting for it when the chart has yet to send it.

        A context whose symbol comes out of a user function only becomes real
        at its own inline ``__sec_signal__``, which can run long after this
        child was spawned. Blocking here is safe — and always ends:

        A consumer X blocks on a peer P only at P's OWN call site. If X's site
        precedes P's, X already released the chart at its write, so the chart
        runs on and reaches P's signal. If P's site precedes X's, the chart
        signalled P before this round of X even started. Either way the chart
        resolves P — spawning it, downgrading it to its own context, or marking
        it producerless — and sends the record down this pipe.

        The wait is a bounded wait on the event the reader thread sets, so a
        dead chart or a ``stop_event`` ends it instead of hanging.
        """
        record = known.get(sid)
        if record is not None:
            return record
        if registry_pipe is None:
            return None
        warned = False
        deadline = monotonic() + _REGISTRY_WARN_SECONDS
        while True:
            if stop_event is not None and stop_event.is_set():
                return None
            registry_arrived.clear()
            with registry_lock:
                record = known.get(sid)
            if record is not None:
                return record
            registry_arrived.wait(_REGISTRY_WAIT_SECONDS)
            if not warned and monotonic() >= deadline:
                warned = True
                # The one shape that can stall here: the peer's call site sits
                # behind a chart-side branch that this bar did not take, so the
                # chart never resolves it. Say so instead of hanging silently.
                logger.warning(
                    "Security context '%s' is still waiting for peer '%s' to be "
                    "resolved by the chart after %.0fs. The peer's "
                    "request.security() call may sit behind a branch the chart "
                    "did not execute on this bar.",
                    sec_id, sid, _REGISTRY_WARN_SECONDS,
                )

    peer_readers: dict[str, RingReader] = {}
    peer_states: dict[str, _PeerReadState] = {}
    own_index = sync_block.index_of(sec_id)

    # Values written on the current bar: the first one publishes, an identical
    # repeat inside a loop is a no-op, a differing one is an error.
    bar_value: list = [None]
    bar_written: list[bool] = [False]
    last_own_value: list = [None]

    def __sec_signal__(_sid: str, _symbol=None, _timeframe=None, _lookahead=None,
                       _scope_id=None):
        pass

    def _peer_asof(record: dict) -> int:
        """This consumer bar's as-of instant for one peer (principle 6)."""
        base = ctx.round_tick if ctx.developing else ctx.bar_close
        peer_cal: BarCalendar = record.get('calendar') or BarCalendar()
        peer_dwm = bool(record.get('is_dwm'))
        if peer_dwm and own_calendar.opening_hours and same_calendar(peer_cal, own_calendar):
            # The scheduled break the base instant falls in ends with a session
            # open; a D/W/M peer closing inside that break is already final.
            base = break_end_after(base, own_calendar)
        cap = ctx.round_tick
        if (peer_dwm and chart_calendar is not None
                and ctx.round_sched_next_open
                and same_calendar(peer_cal, chart_calendar)):
            cap = ctx.round_sched_next_open
        return base if base < cap else cap

    if is_ltf:
        _buffer: list = []

        def __sec_write__(_sid: str, value, _scope_id=None):
            _buffer.append(value)
            if writer is not None and not ctx.developing and ctx.bar_close:
                # Each intrabar is its own ring entry, so a consumer can take the
                # slice covering ITS period instead of the chart bar's window.
                # ``bar_close`` is unset on the live LTF-window path, which is
                # not a pairable producer (see ``__sec_read__``).
                writer.append(ctx.bar_open, ctx.bar_close, value,
                              consumer_indexes, ctx.frontier)

        def flush(skip: int = 0):
            """Publish the round's intrabar array. ``skip`` drops the first N
            buffered values — feed bars the round replayed for expression
            state but which open BEFORE the chart bar's own period (a cold
            start mid-feed, or intrabars in a chart session gap). TradingView
            arrays carry only the bar's own period."""
            published = _buffer[skip:]
            with own_lock:
                write_result(result_block, sync_block, published)
            last_own_value[0] = published
            _buffer.clear()

        def buffer_len() -> int:
            return len(_buffer)

        def ltf_take_value():
            """Return the value written by the latest intrabar run, clearing the
            buffer. Used by the live LTF-window path to capture one intrabar's
            expression value per ``__run_script_main`` instead of the whole
            accumulated array (which the path manages via its own window)."""
            if not _buffer:
                return None
            value = _buffer[-1]
            _buffer.clear()
            return value

        def ltf_publish(values):
            """Write a live LTF-window array under the result lock."""
            published = list(values)
            with own_lock:
                write_result(result_block, sync_block, published)
            last_own_value[0] = published
    else:
        def __sec_write__(_sid: str, value, _scope_id=None):
            if bar_written[0]:
                # A second write on the same bar. Only a write block sitting in
                # a loop body can do this legitimately, and only with a
                # loop-INVARIANT value: the chart may already have read the
                # first one, so a differing value has no single answer.
                if not in_loop:
                    raise AssertionError(
                        f"security context '{sec_id}' wrote twice on one bar "
                        f"outside a loop"
                    )
                if _same_value(bar_value[0], value):
                    return
                raise RuntimeError(
                    "loop-varying security expression is not supported"
                )
            bar_written[0] = True
            bar_value[0] = value
            last_own_value[0] = value
            with own_lock:
                write_result(result_block, sync_block, value)
            if writer is not None:
                # The frontier rises AT the append, not at the end of the bar:
                # a consumer whose as-of the newly published bar covers must be
                # released immediately, or two contexts writing on the same bar
                # would wait for each other.
                #
                # A DEVELOPING bar has no scheduled close yet, so it enters the
                # ring at the round's as-of instant instead: that is exactly how
                # far it has aggregated, and it is what the chart's own
                # ``lookahead_on`` read of this context returns on this bar.
                # Publishing it is what keeps the dependent form equal to the
                # nested one — a consumer of the same round would otherwise see
                # the PREVIOUS closed bar while the chart sees this one. It
                # cannot look ahead: the entry is paired by its as-of, and a
                # consumer never asks beyond its own. The frontier goes no
                # further than the round's own as-of (``ctx.frontier``), so a
                # consumer running on a LATER tick waits for that tick's round
                # instead of passing the wait and reading this one.
                close_ms = ctx.round_tick if ctx.developing else ctx.bar_close
                writer.append(ctx.bar_open, close_ms, value,
                              consumer_indexes, ctx.frontier)
            if ctx.is_round_last and data_ready_event is not None:
                data_ready_event.set()

        flush = None
        ltf_take_value = None
        ltf_publish = None
        buffer_len = None

    def __sec_read__(sid: str, default=None, _scope_id=None):
        if sid == sec_id:
            # Nothing published yet -- the child replays the WHOLE script, so a
            # context's own read can run before its write does on the first bar.
            # The caller's default is what every other empty answer here returns,
            # and an array read asks for `[]`: the alternative is handing the
            # script a None no `array.*` builtin accepts. A published value is
            # never None (na is a float), so the test cannot swallow a real one.
            own = last_own_value[0]
            return default if own is None else own
        if sid not in depends:
            # Not a dependency of this context: the read cannot influence this
            # expression, so it never waits. ``depends`` also excludes producers
            # whose call site comes AFTER this context's own: a warmup round
            # replays many bars at once, so waiting for a peer the chart has not
            # launched yet would block this context's remaining writes — and the
            # chart is parked on one of them. Such a peer can only reach this
            # expression as a previous-bar carry, which is already published.
            return default
        record = _peer_record(sid)
        if (record is None or not record.get('has_producer')
                or sid not in ring_conditions
                or (record.get('is_ltf') and record.get('ltf_live_stream'))):
            # A context nothing produces for — ``ignore_invalid_symbol``
            # downgraded it — or a lower-timeframe array fed by a LIVE stream,
            # which publishes whole chart-bar windows and keeps no per-intrabar
            # ring. Answer with the default; waiting would never end, nothing
            # moves that frontier.
            return default
        asof = _peer_asof(record)
        state = peer_states.get(sid)
        if state is None:
            state = _PeerReadState()
            peer_states[sid] = state
        if state.bar == ctx.bar_open and state.asof == asof:
            return state.value
        reader = peer_readers.get(sid)
        if reader is None:
            reader = RingReader(sid, sync_block, ring_conditions[sid])
            peer_readers[sid] = reader
        reader.wait_for_close(asof, stop_event)
        if record.get('is_ltf'):
            # A lower-timeframe ARRAY peer: this consumer bar's own slice of the
            # producer's intrabars — those closing after this bar opened and at
            # or before its as-of. An intrabar straddling either end belongs to
            # the neighbouring bar, exactly as it does on the chart.
            values = [v for _o, _c, v in reader.range_by_close(ctx.bar_open, asof)]
            state.bar = ctx.bar_open
            state.asof = asof
            state.entry_close = asof
            state.value = values
            return values
        entry = reader.last_close_at_or_before(asof)
        new_bar = state.bar != ctx.bar_open
        state.bar = ctx.bar_open
        state.asof = asof
        if entry is None:
            state.entry_close = -1
            state.value = default
            return default
        _entry_open, entry_close, value = entry
        if new_bar and entry_close == state.entry_close:
            # No new peer bar closed in this consumer bar: TradingView holds the
            # last value (``gaps_off``) or emits ``na`` (``gaps_on``).
            if record.get('gaps_on'):
                state.value = default
                return default
            state.value = value
            return value
        state.entry_close = entry_close
        state.value = value
        return value

    def __sec_wait__(_sid: str, _scope_id=None):
        pass

    def after_bar() -> None:
        """Close out one bar of this context, whatever the script did.

        Runs from the bar loop, not from script-emitted code: an early
        ``return`` in ``main()`` skips every protocol call, and a conditional
        write leaves the bar without an append — yet the frontier must still
        rise, or a consumer waiting on this context's as-of would never be
        released. Also parks the GC watermark for every peer this context can
        read, so a conditional read cannot pin a producer's ring forever.
        """
        if writer is not None and not bar_written[0]:
            writer.set_frontier_close(ctx.frontier, consumer_indexes)
        bar_written[0] = False
        bar_value[0] = None
        if depends:
            for sid in depends:
                # Never blocks: a peer the chart has not resolved yet has no
                # ring to pin, so there is no watermark to park.
                record = known.get(sid)
                if record is None or not record.get('has_producer'):
                    continue
                # A lower-timeframe array peer is read from this bar's period
                # START, so only entries below THAT may be collected.
                mark = ctx.bar_open if record.get('is_ltf') else _peer_asof(record)
                sync_block.set_watermark(
                    own_index, sync_block.index_of(sid), mark)

    def prime_ring(frontier_close: int) -> None:
        """Publish the frontier this producer starts with.

        A producer the chart has not woken yet has published nothing, and a
        frontier left at zero would hold up any consumer whose own first bar
        closes before this context's first one — the chart never targets this
        context in that round, so nothing would ever raise it. The runner
        therefore primes the frontier from the feed's first bar as soon as the
        producer is ready: "nothing of mine closes at or before this".

        :param frontier_close: Initial frontier close in epoch ms.
        """
        if writer is not None:
            writer.set_frontier_close(frontier_close)

    def finish() -> None:
        """Mark this producer done: the frontier goes to ``+inf``."""
        if writer is not None:
            writer.finish()

    def cleanup():
        if registry_pipe is not None:
            registry_pipe.close()
        for r in peer_readers.values():
            r.close()
        if writer is not None:
            writer.close()
            writer.unlink()

    return (__sec_signal__, __sec_write__, __sec_read__, __sec_wait__, cleanup,
            flush, ltf_take_value, ltf_publish, buffer_len, ctx, after_bar, finish,
            prime_ring)


# Representative dates for the off-grid session probe — one on each side of the
# DST boundary so a session that lands on the tf grid only half the year is still
# detected. Both are Mondays, so the weekday offset arithmetic below is exact.
_WINTER_PROBE = date(2024, 1, 15)
_SUMMER_PROBE = date(2024, 7, 15)

# One civil day in ms — the probe step of the D/W/M period-end search.
_DAY_MS = 86_400_000
# Upper bound of that search: long enough for a 12M period (multi-period yearly
# grids are the widest shape a D/W/M timeframe can take) plus DST headroom.
_DWM_PROBE_DAYS = 400
# Upper bound of the break scan: no market closes for more than a week.
_BREAK_SCAN_DAYS = 9


def _needs_session_anchor(
    session_starts: 'list[SymInfoSession]',
    tzinfo: ZoneInfo | None,
    timeframe: str,
) -> bool:
    """
    Whether ``timeframe`` bars need session anchoring for this market.

    Session anchoring changes nothing when every declared session open already
    lands on the ``timeframe`` grid, so those markets keep the zero-overhead
    clock-floor fast path. For intraday timeframes the grid is the UTC epoch one
    (24/7, on-hour, session-aligned markets are already on it) and the open is
    probed on both a winter and a summer date to cover both DST offsets. For
    D/W/M the grid is the exchange timezone's midnight, so anchoring matters
    exactly for markets whose trading day opens at another hour — FX at 17:00,
    futures at 18:00. The test is deliberately conservative: it anchors whenever
    any open is off-grid.

    :param session_starts: Per-trading-day primary opens.
    :param tzinfo: The market's exchange timezone.
    :param timeframe: The requested HTF string (e.g. ``"60"``).
    :return: True if the security loop must pass ``session_starts`` to anchor.
    """
    if not session_starts:
        return False
    # Local import: ``core`` ↔ ``lib`` would otherwise form an import cycle
    # (mirrors the existing ``from pynecore.lib import ...`` uses in this file).
    from ..lib import timeframe as tf_module
    # noinspection PyProtectedMember
    modifier, _ = tf_module._process_tf(timeframe)
    if modifier not in ('S', ''):
        return any(s.time != time(0, 0) for s in session_starts)
    tf_seconds = tf_module._in_seconds(timeframe)
    for probe in (_WINTER_PROBE, _SUMMER_PROBE):
        for s in session_starts:
            d = probe + timedelta(days=(s.day - probe.weekday()) % 7)
            open_sec = int(datetime(
                d.year, d.month, d.day,
                s.time.hour, s.time.minute, s.time.second,
                tzinfo=tzinfo,
            ).timestamp())
            if open_sec % tf_seconds != 0:
                return True
    return False


def resolve_session_anchor(
    si: 'SymInfo | None',
    timeframe: str,
    fallback_tz: ZoneInfo,
) -> 'tuple[list[SymInfoSession] | None, ZoneInfo | None, list[SymInfoInterval] | None]':
    """
    Decide HTF session anchoring for one security context.

    Returns ``(session_starts, session_tz, opening_hours)`` to store on the
    ``SecurityState`` so ``_get_confirmed_time`` anchors HTF bars to the session
    open, or all-``None`` when the market opens on the ``timeframe`` grid (the
    clock-floor fast path). ``session_tz`` is the security's own exchange
    timezone — correct even for a cross-symbol HTF in a different session.

    :param si: The security's own ``SymInfo`` (``None`` → no anchoring).
    :param timeframe: The resolved HTF string (e.g. ``"60"``).
    :param fallback_tz: Timezone used if the syminfo timezone is missing/invalid.
    """
    if si is None or not getattr(si, 'session_starts', None):
        return None, None, None
    if si.timezone:
        try:
            # parse_timezone resolves both IANA names and UTC/GMT±HHMM offset
            # forms (e.g. "UTC-5"), which bare ZoneInfo() rejects.
            si_tz = parse_timezone(si.timezone)
        except (ValueError, KeyError):
            # Unknown / malformed timezone (ZoneInfoNotFoundError is a KeyError;
            # TimezoneNotFoundError is a ValueError).
            si_tz = fallback_tz
    else:
        si_tz = fallback_tz
    if _needs_session_anchor(si.session_starts, si_tz, timeframe):
        return si.session_starts, si_tz, getattr(si, 'opening_hours', None) or None
    return None, None, None


def _session_bar_closes(
        opens: list[int],
        tz: ZoneInfo | None,
        opening_hours: list[SymInfoInterval],
        period_ms: int,
        corrections: dict[date, tuple[SymInfoInterval, ...]] | None = None,
) -> list[int] | None:
    """
    Close instant (epoch ms) of each intraday HTF bar: the earlier of its period
    end and its session's scheduled end.

    An HTF bar covers ``[open, open + period)`` but never extends past its trading
    session, so it closes at ``min(open + period, session_end)``. When the period
    is at least as long as the session (one bar per session, e.g. a 720-minute bar
    on a 3-session palm-oil contract) the session end wins, and ``_get_confirmed_
    time`` confirms the bar there instead of at the arithmetic next-period boundary
    that the non-trading gap before the next session would push a full period late.
    When several bars fit inside a session (e.g. a 60-minute HTF) the period end
    wins and behaviour matches the plain grid. Each open's session end comes from
    the ``opening_hours`` interval that contains it (overnight intervals — ``end <=
    start`` — close on the following calendar day). A bar opening *after* midnight
    is matched to the PREVIOUS calendar day's overnight interval, whose session it
    belongs to (e.g. a ``21:00->02:00`` night session's ``01:00`` bar closes at the
    ``02:00`` session end, not a full period later).

    A date listed in ``corrections`` trades on its own hours instead of its
    weekday's — an exchange early close (US equities' 13:00 half-days) ends the
    session hours before the regular close, and a bar opening in the final hour
    then closes at the early close, not a full period later. The correction
    applies per calendar date on both branches, so an overnight session's
    after-midnight leg takes the correction of the date its session STARTED.

    :param opens: HTF bar opens in epoch ms, ascending.
    :param tz: The security's exchange timezone.
    :param opening_hours: The security's ``SymInfo.opening_hours`` intervals.
    :param period_ms: The HTF period length in milliseconds.
    :param corrections: The security's ``SymInfo.session_corrections``, or ``None``.
    :return: A parallel list of close instants (epoch ms), or ``None`` if any open
        has no containing interval — the schedule does not fully describe the
        feed, so the caller keeps the arithmetic grid clamp rather than risk a
        wrong session end.
    """
    closes: list[int] = []
    # One prepared entry per exchange-local DATE, shared by every bar opening on
    # it. The schedule work -- the correction lookups, the overnight
    # classification of each interval and the end-instant arithmetic -- depends
    # only on the date (:func:`_day_session_ends`); a 30-minute feed puts 48 bars
    # on one date and each of them only needs the time-of-day comparison. The
    # cache is local to this walk, so it cannot outlive the schedule it was
    # built from.
    prepared: 'dict[date, tuple[tuple[time | None, time | None, int], ...]]' = {}
    for open_ms in opens:
        open_dt = datetime.fromtimestamp(open_ms / 1000, tz=tz)
        open_date = open_dt.date()
        ends = prepared.get(open_date)
        if ends is None:
            # Exclusive ends, as in :func:`actual_bar_close`: a 24h schedule's
            # 23:59:59 marker closes its last bar at midnight, not a second early.
            ends = tuple(
                (lo, hi, _exclusive_session_end(end, tz))
                for lo, hi, end in _day_session_ends(open_date, tz, opening_hours, corrections))
            prepared[open_date] = ends
        end_ms = _match_session_end(open_dt.time(), ends)
        if end_ms is None:
            return None
        # Whichever comes first: the bar's own period end, or the session end (a
        # non-trading gap before the next session must not delay confirmation).
        closes.append(min(open_ms + period_ms, end_ms))
    return closes


def _session_end_of_open(
        open_ms: int,
        tz: ZoneInfo | None,
        opening_hours: 'list[SymInfoInterval]',
        corrections: 'dict[date, tuple[SymInfoInterval, ...]] | None' = None,
) -> int | None:
    """
    Scheduled end instant (epoch ms) of the session that contains ``open_ms``.

    Extracted from :func:`_session_bar_closes` so a single bar open can be
    resolved on its own — :func:`actual_bar_close` needs exactly this for the
    intraday branch, and the D/W/M branch needs it per trading day.

    :param open_ms: Bar open (or any instant inside a session) in epoch ms.
    :param tz: The security's exchange timezone.
    :param opening_hours: The security's ``SymInfo.opening_hours`` intervals.
    :param corrections: The security's ``SymInfo.session_corrections``, or ``None``.
    :return: The session end in epoch ms, or ``None`` when no interval contains
        ``open_ms`` (the schedule does not describe this instant).
    """
    open_dt = datetime.fromtimestamp(open_ms / 1000, tz=tz)
    return _match_session_end(
        open_dt.time(),
        _day_session_ends(open_dt.date(), tz, opening_hours, corrections))


def _day_session_ends(
        open_date: date,
        tz: ZoneInfo | None,
        opening_hours: 'list[SymInfoInterval]',
        corrections: 'dict[date, tuple[SymInfoInterval, ...]] | None' = None,
) -> 'tuple[tuple[time | None, time | None, int], ...]':
    """
    Every session end a bar opening on ``open_date`` can take, as time-of-day
    bounds plus the end instant.

    The per-DATE half of :func:`_session_end_of_open`: which intervals are
    effective (a date listed in ``corrections`` trades on its own hours instead
    of its weekday's), which of them are overnight, and what epoch instant each
    close falls on. None of that depends on the bar's time of day, so a walk
    over a whole feed prepares it once per date and then only compares times of
    day (:func:`_match_session_end`).

    Each entry is ``(lo, hi, end_ms)`` and covers an ``open_time`` when
    ``lo <= open_time`` (``lo`` of ``None``: no lower bound) and
    ``open_time < hi`` (``hi`` of ``None``: no upper bound):

    * a same-day interval bounds both sides and closes on ``open_date``;
    * the pre-midnight leg of an overnight interval (``end <= start``) has no
      upper bound -- every time of day from its open on belongs to it -- and
      closes on the following calendar day;
    * the after-midnight leg of the PREVIOUS day's overnight interval has no
      lower bound and closes on ``open_date`` (a ``21:00->02:00`` night
      session's ``01:00`` bar closes at that ``02:00``). Its correction is the
      one of the date its session STARTED.

    :param open_date: The bar's exchange-local calendar date.
    :param tz: The security's exchange timezone.
    :param opening_hours: The security's ``SymInfo.opening_hours`` intervals.
    :param corrections: The security's ``SymInfo.session_corrections``, or ``None``.
    :return: The date's candidate session ends, in schedule order.
    """
    from .resampler import crosses_midnight
    weekday = open_date.weekday()
    prev_weekday = (weekday - 1) % 7
    if corrections:
        today_hours = corrections.get(open_date, opening_hours)
        prev_hours = corrections.get(open_date - timedelta(days=1), opening_hours)
    else:
        today_hours = prev_hours = opening_hours
    ends: 'list[tuple[time | None, time | None, int]]' = []
    for interval in today_hours:
        if interval.day != weekday:
            continue
        overnight = crosses_midnight(interval.start, interval.end)
        end_date = open_date + timedelta(days=1 if overnight else 0)
        ends.append((
            interval.start,
            None if overnight else interval.end,
            int(datetime.combine(end_date, interval.end,
                                 tzinfo=tz).timestamp() * 1000),
        ))
    for interval in prev_hours:
        if interval.day != prev_weekday or not crosses_midnight(interval.start,
                                                                interval.end):
            continue
        ends.append((
            None,
            interval.end,
            int(datetime.combine(open_date, interval.end,
                                 tzinfo=tz).timestamp() * 1000),
        ))
    return tuple(ends)


def _match_session_end(
        open_time: time,
        ends: 'tuple[tuple[time | None, time | None, int], ...]',
) -> int | None:
    """
    The earliest prepared session end (:func:`_day_session_ends`) whose bounds
    cover ``open_time``.

    :param open_time: The bar's exchange-local time of day.
    :param ends: The date's candidate session ends.
    :return: The session end in epoch ms, or ``None`` when no session covers it.
    """
    end_ms: int | None = None
    for lo, hi, candidate in ends:
        if lo is not None and open_time < lo:
            continue
        if hi is not None and open_time >= hi:
            continue
        if end_ms is None or candidate < end_ms:
            end_ms = candidate
    return end_ms


def _dated_session_bar_closes(
        opens: list[int],
        tz: ZoneInfo | None,
        si: SymInfo,
        period_ms: int,
        overnight: dict[int, time],
) -> list[int] | None:
    """
    Close instants for an HTF feed whose exchange changed its session hours within
    the data range (effective-dated schedule history).

    Like :func:`_session_bar_closes`, but each bar open is matched to the session
    schedule *variant* effective on its exchange-local trading day, so a backtest
    spanning a session-hours change confirms each side with its own schedule. The
    trading-day key (not the raw calendar date of the open) is what
    ``request.security`` already uses to attribute overnight bars: a night bar
    opening 21:00 the evening before belongs to the next trading day and must take
    that day's variant -- keying on the raw open date would mis-assign the boundary
    bar by one day, exactly where a schedule change lives.

    Consecutive opens resolving to the same variant index are grouped into one
    segment and handed to the UNCHANGED :func:`_session_bar_closes` with that
    variant's ``opening_hours``, so every segment runs the same, already-tested
    close-instant arithmetic. Grouping is by variant *index* (not object identity),
    so an ``A -> B -> A`` history yields three segments and the result is stable
    even if the resolver ever returns copies. Any segment the schedule cannot fully
    describe returns ``None``, propagated so the caller keeps the arithmetic grid
    clamp.

    :param opens: HTF bar opens in epoch ms, ascending.
    :param tz: The security's exchange timezone.
    :param si: The security's :class:`SymInfo` (carries ``session_schedules``).
    :param period_ms: The HTF period length in milliseconds.
    :param overnight: Per-weekday rolling opens from ``overnight_opens``, used to
        roll each open to its trading day.
    :return: A parallel list of close instants, or ``None`` if any variant fails to
        describe its bars.
    """
    from .resampler import trading_day
    # Resolve every open's variant index in one pass (trading-day keyed), then walk
    # maximal same-index runs. Setup-time only -- never on the per-bar hot path.
    idx = [si.schedule_index_for(trading_day(o // 1000, tz, overnight)) for o in opens]
    closes: list[int] = []
    i, n = 0, len(opens)
    while i < n:
        k = idx[i]
        j = i
        while j < n and idx[j] == k:
            j += 1
        oh = si.session_schedules[k].opening_hours
        seg = _session_bar_closes(opens[i:j], tz, oh, period_ms, si.session_corrections)
        if seg is None:
            return None
        closes.extend(seg)
        i = j
    return closes


@dataclass
class BarCalendar:
    """
    The trading schedule a security context's bars live on.

    Everything :func:`actual_bar_close`, :func:`break_end_after` and
    :func:`same_calendar` need, in one plain record: the exchange timezone, the
    ``opening_hours`` intervals, the ``session_starts`` template, the
    effective-dated session corrections and the scheduled-grid mode. Built once
    per context (chart and children alike) and then only read.

    An empty ``opening_hours`` means "schedule unknown": the close falls back to
    the civil/arithmetic period end and no calendar extension ever applies.
    """
    tz: ZoneInfo | None = None
    opening_hours: 'tuple[SymInfoInterval, ...]' = ()
    session_starts: 'tuple[SymInfoSession, ...]' = ()
    corrections: 'dict[date, tuple[SymInfoInterval, ...]] | None' = None
    grid_mode: str | None = None


def _day_hours(day: date, cal: BarCalendar) -> 'list[SymInfoInterval]':
    """The intervals effective on one calendar date: its correction, or the template.

    :param day: The calendar date.
    :param cal: The context's calendar.
    :return: The date's opening-hours intervals (empty when it does not trade).
    """
    if cal.corrections:
        corrected = cal.corrections.get(day)
        if corrected is not None:
            return list(corrected)
    return list(cal.opening_hours)


def _trading_day_end_hours(day: date, cal: BarCalendar) -> 'list[SymInfoInterval]':
    """The intervals that can END trading day ``day``, each taken from its own
    OPENING date.

    An effective-dated correction replaces the template for the date a session
    OPENS on, and an interval's ``day`` is its opening weekday — so a rolling
    (overnight) session that ends inside ``day`` is corrected on ``day - 1``.
    Reading both legs off the closing date would apply the wrong day's
    correction to an overnight market.

    :param day: The trading day whose end is being resolved.
    :param cal: The context's calendar.
    :return: Intervals ending inside ``day``, resolved per opening date.
    """
    from .resampler import rolls_trading_day
    weekday = day.weekday()
    hours = [iv for iv in _day_hours(day, cal)
             if iv.day == weekday and not rolls_trading_day(iv.start, iv.end)]
    prev = day - timedelta(days=1)
    prev_weekday = prev.weekday()
    hours += [iv for iv in _day_hours(prev, cal)
              if iv.day == prev_weekday and rolls_trading_day(iv.start, iv.end)]
    return hours


def same_calendar(a: BarCalendar, b: BarCalendar) -> bool:
    """
    Whether two contexts keep the same trading schedule.

    Same exchange timezone, same ``opening_hours`` and the same effective-dated
    corrections: only then does one context's scheduled break coincide with the
    other's, which is what the as-of calendar extension (principle 6) requires.
    The same tickerid satisfies it trivially. Two contexts sharing a weekly
    template but differing on a half-day do NOT keep the same schedule on that
    day, and extending one's as-of over the other's break would hand it a bar
    from past its own close.

    :param a: One context's calendar.
    :param b: The other context's calendar.
    :return: ``True`` when both keep the same schedule.
    """
    return (a.tz == b.tz and tuple(a.opening_hours) == tuple(b.opening_hours)
            and (a.corrections or {}) == (b.corrections or {}))


def break_end_after(ms: int, cal: BarCalendar) -> int:
    """
    ``ms`` itself when it falls inside a scheduled session, else the scheduled
    session open that ENDS the break containing it.

    Sessions are half-open ``[start, end)``, so a bar's own close instant (an
    exclusive end) lands in the break that follows it — which is exactly the
    case the as-of calendar extension is for: the 15:00-16:00 hourly bar of a
    09:30-16:00 equity closes at 16:00, in the break, and therefore sees the
    daily bar closing at 16:00; the 09:30-10:30 bar closes at 10:30, inside the
    session, and does not. A 24h symbol is always inside a session, so it is
    never extended.

    The bound is the SCHEDULE, never the next existing record: a data gap is
    indistinguishable from a holiday, and riding records would hand a consumer
    data from past its own close.

    :param ms: Base instant in epoch ms.
    :param cal: The consumer's own calendar.
    :return: ``ms``, or the next scheduled session open.
    """
    oh = cal.opening_hours
    if not oh:
        return ms
    if _session_end_of_open(ms, cal.tz, list(oh), cal.corrections) is not None:
        return ms
    base = datetime.fromtimestamp(ms / 1000, tz=cal.tz)
    for offset in range(_BREAK_SCAN_DAYS):
        day = base.date() + timedelta(days=offset)
        weekday = day.weekday()
        best: int | None = None
        # A corrected date opens on ITS hours; a date corrected to nothing is
        # closed and contributes no candidate at all.
        for interval in _day_hours(day, cal):
            if interval.day != weekday:
                continue
            candidate = int(
                datetime.combine(day, interval.start, tzinfo=cal.tz).timestamp() * 1000)
            if candidate >= ms and (best is None or candidate < best):
                best = candidate
        if best is not None:
            return best
    return ms


def _exclusive_session_end(ms: int, tz: ZoneInfo | None) -> int:
    """
    A scheduled session end as an EXCLUSIVE instant.

    ``23:59:59`` is PyneCore's end-of-day marker for a round-the-clock schedule
    — every writer of a 24/7 ``SymInfo`` emits it — so the session it ends
    really runs up to midnight. Without this the daily bar of a 24/7 symbol
    would close one second early and its own last intraday bar, closing exactly
    at midnight, would fall outside it.

    :param ms: Session end instant in epoch ms.
    :param tz: The security's exchange timezone.
    :return: The exclusive session end in epoch ms.
    """
    dt = datetime.fromtimestamp(ms / 1000, tz=tz)
    if (dt.hour, dt.minute, dt.second, dt.microsecond) == (23, 59, 59, 0):
        return ms + 1000
    return ms


def _dwm_period_end(open_ms: int, cal: BarCalendar, timeframe: str) -> int:
    """
    Exclusive end of the session-anchored D/W/M period opening at ``open_ms``.

    Probes the period grid one day at a time: :meth:`Resampler.get_bar_time`
    answers with the OPEN of the period a probe falls in, so the first probe
    landing in a later period already IS this period's exclusive end. Never
    nominal seconds — a month has no fixed length and a week ends where the
    schedule says.

    :param open_ms: The bar's open in epoch ms.
    :param cal: The security's calendar.
    :param timeframe: The context's D/W/M timeframe string.
    :return: The next period's open in epoch ms.
    """
    from .resampler import Resampler
    resampler = Resampler.get_resampler(timeframe)
    starts = list(cal.session_starts) or None
    hours = list(cal.opening_hours) or None
    probe = open_ms
    for _ in range(_DWM_PROBE_DAYS):
        probe += _DAY_MS
        nxt = resampler.get_bar_time(probe, cal.tz, starts, hours, cal.grid_mode)
        if nxt > open_ms:
            return nxt
    return probe


def actual_bar_close(open_ms: int, next_open_ms: int, cal: BarCalendar,
                     timeframe: str) -> int:
    """
    The bar's SCHEDULED close instant (``close_A``) — exclusive, in epoch ms.

    One rule pairs every consumer with every producer: a consumer sees the
    peer's last bar whose ``close_A`` is at or before its own as-of instant. So
    the close has to be the real one, not an arithmetic guess:

    * intraday: ``min(open + span, next_open, session_end)`` — a session-closing
      stub (15:30 -> 16:00 on a 60-minute grid) and a chart bar shortened by the
      next bar's open both close where they really do;
    * D/W/M: the end of the LAST scheduled session inside the session-anchored
      period (an equity weekly bar closes Friday 16:00, an FX weekly bar Friday
      17:00 New York — never the next Monday's open), falling back to the civil
      period end for a 24h symbol, which has no session bounds.

    Holidays and unscheduled early closes are NOT in the schedule and cannot be
    told apart from a data gap, so such a bar becomes visible one consumer bar
    late (see the as-of calendar extension) instead of being guessed from the
    bar grid.

    :param open_ms: The bar's open in epoch ms.
    :param next_open_ms: Open of the following bar in epoch ms, ``0`` when
        unknown (last historical bar, live edge). Intraday only.
    :param cal: The bar's own calendar.
    :param timeframe: The bar's timeframe string.
    :return: The bar's scheduled close instant in epoch ms.
    """
    from ..lib import timeframe as tf_module
    # noinspection PyProtectedMember
    modifier, _multiplier = tf_module._process_tf(timeframe)

    if modifier not in ('D', 'W', 'M'):
        # noinspection PyProtectedMember
        close = open_ms + tf_module._in_seconds(timeframe) * 1000
        if next_open_ms and next_open_ms < close:
            close = next_open_ms
        if cal.opening_hours:
            session_end = _session_end_of_open(
                open_ms, cal.tz, list(cal.opening_hours), cal.corrections)
            if session_end is not None:
                session_end = _exclusive_session_end(session_end, cal.tz)
                if session_end < close:
                    close = session_end
        return close

    return dwm_session_close(open_ms, _dwm_period_end(open_ms, cal, timeframe), cal)


def dwm_session_close(open_ms: int, period_end: int, cal: BarCalendar) -> int:
    """
    Close instant of a D/W/M bar spanning ``[open_ms, period_end)``.

    The end of the LAST scheduled session inside the period (an equity weekly
    bar closes Friday 16:00, an FX weekly bar Friday 17:00 New York — never the
    next Monday's open), falling back to the civil period end for a symbol with
    no usable session bounds.

    :param open_ms: The bar's open in epoch ms.
    :param period_end: The period's exclusive civil end in epoch ms.
    :param cal: The bar's own calendar.
    :return: The bar's close instant in epoch ms.
    """
    if not cal.opening_hours:
        # No schedule (or a 24h symbol with no usable session bounds): the civil
        # period end IS the close.
        return period_end
    from .resampler import overnight_opens, trading_day, trading_day_end_sec
    overnight = overnight_opens(list(cal.opening_hours), list(cal.session_starts) or None)
    day = trading_day((period_end - 1) / 1000, cal.tz, overnight)
    first_day = trading_day(open_ms / 1000, cal.tz, overnight)
    while day >= first_day:
        # An effective-dated correction REPLACES the weekly template for the
        # date a session OPENS on (an empty one means the day is closed),
        # exactly as the intraday branch resolves it — a D/W/M close computed
        # off the uncorrected template would expose an extended session's bar
        # hours early and misdate a shortened one.
        end_sec = trading_day_end_sec(day, cal.tz, _trading_day_end_hours(day, cal))
        if end_sec is not None:
            end_ms = _exclusive_session_end(end_sec * 1000, cal.tz)
            if open_ms < end_ms <= period_end:
                return end_ms
        day -= timedelta(days=1)
    return period_end


def _is_dense_feed(reader: OHLCVReader, real_bar_count: int, period_sec: int) -> bool:
    """
    Decide whether a feed's real bars tile the requested timeframe grid.

    A feed is dense only when its bars sit exactly one period apart. The row
    count alone is not enough: a session-spaced feed whose bars are wider than
    the period (e.g. a gap-free, 24h-spaced 720-minute night future) has no gap
    fills, yet its bars do NOT tile the period grid and must ride the
    session-close path. A file with fewer than two bars has no spacing to
    contradict the grid and stays dense.

    A file declaring its own period and density answers directly; a legacy file
    declares neither, so the spacing is measured from its first two records —
    which is also where its gap fills show up as a shorter real-bar count.

    :param reader: Open reader of the child's feed.
    :param real_bar_count: Number of real (non gap-fill) bars read from the feed.
    :param period_sec: The requested timeframe in seconds.
    :return: Whether the arithmetic grid already matches the feed.
    """
    from ..lib import timeframe as tf_module
    if reader.size < 2:
        return True
    if reader.dense is not None and reader.period is not None:
        return reader.dense and tf_module._in_seconds(reader.period) == period_sec
    return (real_bar_count == reader.size
            and reader.read(1).timestamp - reader.read(0).timestamp == period_sec * 1000)


def load_htf_bar_opens(state: SecurityState, data_path: str) -> None:
    """
    Load the child's real bar opens for HTF confirmation against the actual feed.

    The arithmetic grid in ``_get_confirmed_time`` assumes a child bar exists at
    every grid period — true only for a DENSE feed. Two cases break it, and both
    confirm by riding the child's real bar opens instead:

    * D/W/M — including the single-period ``D``/``W``/``M`` case: macro aggregates
      (ECONOMICS series, dividends) carry a bar only on scattered days, so the
      grid would emit a confirmation boundary for every calendar period and a
      chart bar landing on a day with no real child bar would advance the
      subprocess into an empty window — writing ``na`` and destroying the
      ``gaps_off`` (TV default) forward-fill.
    * Gappy intraday HTF: a session-gapped futures feed (e.g. a 720-minute HTF on
      a 3-session palm-oil contract) has no bar over its non-trading spans, so the
      grid would confirm periods the child never reaches. Dense intraday feeds
      keep the cheaper arithmetic grid (this stays a no-op for them).

    Lower-timeframe contexts (``request.security_lower_tf`` arrays and the scalar
    ``plain_ltf`` merge) load the same two lists: their chart-side target is the
    LAST intrabar whose ``close_A`` is at or before the chart bar's own close, so
    an intrabar straddling that close is left for the next chart bar instead of
    exposing a price from after it.

    A gappy SAME-TF cross-symbol feed (a session-bounded symbol requested at the
    chart's own TF on a 24/7 chart) rides the same intraday path:
    ``_get_confirmed_time`` clamps the chart bar time to these opens so gap bars
    confirm nothing new.

    Riding the real opens (clamp for single-period / intraday, walk for
    multi-period D/W/M) makes ``new_period`` fire only on real bars: between them
    ``gaps_off`` holds the last value and ``gaps_on`` emits ``na``, both matching
    TradingView. The security's own grid parameters are loaded from its TOML for
    the past-end-of-data fallback.

    :param state: Security context state (``state.timeframe`` already resolved)
    :param data_path: Path to the child's OHLCV data file
    """
    # Local import: ``core`` ↔ ``lib`` would otherwise form an import cycle.
    from ..lib import timeframe as tf_module
    from .ohlcv import OHLCVReader
    from .resampler import grid_mode, overnight_opens, trading_day
    from .syminfo import SymInfo
    # noinspection PyProtectedMember
    modifier, multiplier = tf_module._process_tf(state.timeframe)
    is_dwm = modifier in ('D', 'W', 'M')

    if not is_dwm:
        with OHLCVReader(data_path) as reader:
            start_ts = reader.start_timestamp
            if start_ts is None:
                return
            opens = [candle.timestamp for candle in reader.read_from(start_ts)]
        state.bar_opens_multiperiod = False
    else:
        # Multi-period (nD/nW/nM) walks the opens directly (the arithmetic grid
        # cannot reproduce TradingView's scheduled multi-period calendar). Single
        # period (1D/1W/1M) instead uses the grid for the calendar close instant
        # and only *clamps* to these opens — see ``_get_confirmed_time``.
        state.bar_opens_multiperiod = multiplier > 1
        with OHLCVReader(data_path) as reader:
            start_ts = reader.start_timestamp
            opens = ([] if start_ts is None else
                     [candle.timestamp for candle in reader.read_from(start_ts)])
            if not opens and reader.size == 1:
                # The lone record is a legacy phantom gap fill, which range reads
                # skip — ``opens`` would stay empty and ``_get_confirmed_time``
                # would never confirm the bar (the child reads ``na`` forever).
                # Read it directly so its open still anchors the clamp.
                opens = [reader.read(0).timestamp]

    state.bar_opens = opens
    state.bar_ptr = -1

    sec_tz: ZoneInfo | None = state.tz
    sec_starts = sec_hours = mode = None
    si: 'SymInfo | None' = None
    toml_path = Path(data_path).with_suffix('.toml')
    if toml_path.exists():
        si = SymInfo.load_toml(toml_path)
        try:
            sec_tz = parse_timezone(si.timezone) if si.timezone else state.tz
        except (ValueError, KeyError):
            sec_tz = state.tz
        sec_starts = si.session_starts or None
        sec_hours = si.opening_hours or None
        mode = grid_mode(si.type, si.opening_hours)

        # Session-bounded intraday feed (e.g. a futures contract's day/night
        # sessions, or an equity whose 09:30 open puts its bars off the chart's
        # grid): confirm each bar on its scheduled session end instead of the
        # arithmetic next-period boundary (see ``_get_confirmed_time``). Needs the
        # session schedule; ``None`` (no schedule, or a bar outside it) keeps the
        # grid clamp. Same-TF contexts need it for the same reason as HTF ones —
        # only their offset comes from the session, not from the period length.
        if not is_dwm and sec_hours:
            period_ms = tf_module._in_seconds(state.timeframe) * 1000
            if si.has_schedule_history:
                # The trading-day roll keys off the flat (newest) session opens;
                # this Core path assumes the session OPEN / trading-day attribution
                # is stable across variants (close-only era changes, e.g. a futures
                # contract that shortened its night session). A symbol that shifts
                # its session START across eras needs the deferred session-anchoring
                # work -- the assumption is stated here in code, not only the docs.
                overnight = overnight_opens(sec_hours, sec_starts)
                # Surface that unsupported shape instead of silently mis-confirming:
                # an earlier variant whose overnight session OPENS at a different
                # time than the newest one (a session-START shift, not a close-only
                # change) is rolled to the wrong trading day by the newest-keyed
                # ``overnight`` above and can pick the wrong variant. Only a weekday
                # that is overnight in BOTH variants but at a different time counts;
                # a structurally different (e.g. day-only) era is handled by the
                # ``None`` fallback below, not a START shift.
                for variant in si.session_schedules[:-1]:
                    vo = overnight_opens(variant.opening_hours, variant.session_starts)
                    if any(overnight.get(d) is not None and t != overnight[d]
                           for d, t in vo.items()):
                        logger.warning(
                            "%s:%s session schedule history changes the overnight "
                            "session OPEN at variant effective %s; the dated HTF "
                            "path attributes every bar by the newest variant's open, "
                            "so bars near that change may confirm against the wrong "
                            "variant. Session-START shifts are not yet supported "
                            "(close-only era changes are).",
                            si.prefix, si.ticker, variant.effective_from)
                        break
                if opens:
                    first_td = trading_day(opens[0] // 1000, sec_tz, overnight)
                    earliest = si.session_schedules[0].effective_from
                    if first_td < earliest:
                        logger.warning(
                            "%s:%s session schedule history starts %s but the HTF "
                            "feed opens on trading day %s; the oldest variant was "
                            "applied to the earlier bars. Add an earlier "
                            "[[session_schedules]] variant for an exact backtest "
                            "across that range.",
                            si.prefix, si.ticker, earliest, first_td)
                state.bar_closes = _dated_session_bar_closes(
                    opens, sec_tz, si, period_ms, overnight)
            else:
                state.bar_closes = _session_bar_closes(
                    opens, sec_tz, sec_hours, period_ms, si.session_corrections)

    state.calendar = BarCalendar(
        tz=sec_tz,
        opening_hours=tuple(sec_hours or ()),
        session_starts=tuple(sec_starts or ()),
        corrections=(si.session_corrections if si is not None else None),
        grid_mode=mode,
    )
    state.is_dwm = is_dwm

    if state.bar_closes is None:
        # Every context confirms on real close instants (principle 2), so a feed
        # whose schedule the branches above could not describe — a sessionless or
        # 24h intraday one, and every D/W/M one — gets its closes here. The
        # intraday branches above are the schedule-history and correction-aware
        # specializations of the same formula and are left alone where they
        # applied.
        state.bar_closes = [
            actual_bar_close(o, opens[i + 1] if i + 1 < len(opens) else 0,
                             state.calendar, state.timeframe)
            for i, o in enumerate(opens)
        ]

    state.sec_grid_args = (sec_tz, sec_starts, sec_hours, mode)


def load_ltf_first_ms(state: SecurityState, data_path: str) -> None:
    """
    Record the LTF child feed's first bar open for the chart-side prefix skip.

    ``request.security_lower_tf()`` makes the chart block on a cross-process
    handshake for *every* chart bar, because any chart bar may contain
    intrabars. Chart bars whose whole period ends before the feed's very first
    bar never can: the child includes intrabars with ``bar_open <= target_time``
    and the historical target is the chart bar's last ms, so a chart bar whose
    period ends strictly below the feed's first open yields an empty intrabar
    array unconditionally — matching TradingView, which returns ``na`` before the
    lower-timeframe series begins. Recording that first open lets
    ``__sec_signal__`` skip the signal+wait over the idle prefix and
    ``__sec_read__`` return the empty-array default without touching shared
    memory. Backtest/file-backed only: a live ``PluginSymbol`` stream has no
    static first bar, so ``ltf_first_ms`` stays ``None`` and every chart bar
    signals as before. No-op for non-LTF contexts (``plain_ltf`` — the scalar
    lower-timeframe merge — uses the same prefix skip, so it loads too).

    :param state: LTF security context state.
    :param data_path: Path to the child's OHLCV data file.
    """
    if not state.is_ltf and not state.plain_ltf:
        return
    from .ohlcv import OHLCVReader
    with OHLCVReader(data_path) as reader:
        state.ltf_first_ms = reader.start_timestamp


def setup_security_states(
    contexts: dict[str, dict],
    chart_timeframe: str,
    tz: 'ZoneInfo',
    chart_symbol: str | None = None,
    chart_syminfo: 'SymInfo | None' = None,
    sec_syminfos: 'dict[str, SymInfo] | None' = None,
) -> tuple[dict[str, SecurityState], SyncBlock, dict[str, ResultBlock]]:
    """
    Initialize security states, shared memory, and events from ``__security_contexts__``.

    :param contexts: The ``__security_contexts__`` dict from the script module.
                     Keys are sec_ids, values are dicts with 'symbol', 'timeframe', 'gaps'.
    :param chart_timeframe: The chart's timeframe string (e.g., "5", "1D").
    :param tz: The chart's timezone.
    :param chart_symbol: The chart's ticker (e.g. ``"AAPL"``). Drives same-symbol
                        gating for the live HTF transport — a cross-symbol HTF
                        context gets no ``HTFAggregator`` because the chart-side
                        OHLCV would be the wrong instrument. A context naming the
                        chart qualified (``"NASDAQ:AAPL"``) or empty counts as
                        same-symbol too. ``None`` (unit-test / legacy callers
                        without symbol context) is treated as "every HTF is
                        same-symbol".
    :param chart_syminfo: The chart symbol's ``SymInfo``, used as the session
                        source for same-symbol HTF anchoring and as the source of
                        the exchange prefix of the qualified chart symbol.
                        ``None`` disables anchoring unless a per-security syminfo
                        is supplied.
    :param sec_syminfos: ``sec_id → SymInfo`` for cross-symbol contexts, used so
                        each security anchors to its own session/timezone. ``None``
                        falls back to ``chart_syminfo``.
    :return: (states, sync_block, result_blocks)
    """
    from pynecore.lib import barmerge
    from pynecore.lib import timeframe as tf_module
    from .resampler import Resampler

    # Chart bar open -> last instant offset: multi-period boundaries resolve a
    # chart bar by the instant it ends at, so the bar containing a session
    # open counts as the new period's first bar. D/W/M chart bars are
    # session-aligned by construction and need no offset.
    # noinspection PyProtectedMember
    chart_mod, chart_mult = tf_module._process_tf(chart_timeframe)
    chart_off = (tf_module._in_seconds(chart_timeframe) * 1000 - 1
                 if chart_mod in ('', 'S') else 0)

    # Single-period civil daily/weekly/monthly chart: the LTF window cannot use
    # the (zero) ``chart_off`` span; ``__sec_signal__`` instead targets the
    # chart bar's civil period end via this resampler. Multi-period and
    # intraday charts keep the ``chart_off`` path. Only attached to LTF states.
    chart_ltf_resampler = None
    chart_ltf_modifier = ''
    if chart_mod in ('D', 'W', 'M') and chart_mult == 1:
        chart_ltf_resampler = Resampler.get_resampler(chart_timeframe)
        chart_ltf_modifier = chart_mod

    # Every spelling of the chart instrument, for the same-symbol gating below:
    # bare (``syminfo.ticker``), exchange qualified (``syminfo.tickerid``, the
    # form a literal ``"BINANCE:BTCUSDT"`` also takes) and empty (an empty symbol
    # IS the chart's own instrument in Pine). ``None`` keeps the "every HTF is
    # same-symbol" fallback of callers without symbol context.
    chart_symbols: set[str] | None = None
    if chart_symbol is not None:
        _symbols = {'', chart_symbol}
        if chart_syminfo is not None and chart_syminfo.prefix:
            _symbols.add(f"{chart_syminfo.prefix}:{chart_symbol}")
        chart_symbols = _symbols

    # The chart's own schedule — the single source of the chart bar's scheduled
    # close and of the ``same_calendar`` test every D/W/M peer's as-of extension
    # is gated on.
    from .resampler import grid_mode
    chart_calendar = BarCalendar(tz=tz)
    if chart_syminfo is not None:
        chart_cal_tz = tz
        if chart_syminfo.timezone:
            try:
                chart_cal_tz = parse_timezone(chart_syminfo.timezone)
            except (ValueError, KeyError):
                chart_cal_tz = tz
        chart_calendar = BarCalendar(
            tz=chart_cal_tz,
            opening_hours=tuple(chart_syminfo.opening_hours or ()),
            session_starts=tuple(chart_syminfo.session_starts or ()),
            corrections=chart_syminfo.session_corrections or None,
            grid_mode=grid_mode(chart_syminfo.type, chart_syminfo.opening_hours),
        )

    sec_ids = list(contexts.keys())
    sync_block = SyncBlock(sec_ids)
    states: dict[str, SecurityState] = {}
    result_blocks: dict[str, ResultBlock] = {}

    for sec_id, ctx in contexts.items():
        tf_val = ctx.get('timeframe', chart_timeframe)
        if tf_val is None or tf_val == '':
            # Runtime-dependent (deferred) timeframe gets the chart TF as a
            # placeholder until the runtime ``__sec_signal__`` resolves it;
            # an empty string IS the chart's timeframe (Pine semantics)
            tf_val = chart_timeframe
        timeframe = str(tf_val)
        is_ltf = bool(ctx.get('is_ltf', False))

        htf_aggregator: HTFAggregator | None = None
        na_on_developing = False
        anchor_starts: 'list[SymInfoSession] | None' = None
        anchor_tz: ZoneInfo | None = None
        anchor_oh: 'list[SymInfoInterval] | None' = None
        plain_ltf = False
        if is_ltf:
            is_gaps_on = False
            same_tf = False
            resampler = None  # chart-side resampler not needed for LTF
            lookahead_mode = Lookahead.OFF  # LTF has no lookahead concept
        else:
            gaps_val = ctx.get('gaps', barmerge.gaps_off)
            is_gaps_on = gaps_val is barmerge.gaps_on
            same_tf = (timeframe == chart_timeframe)
            # Plain security with a FINER timeframe than the chart: scalar
            # LTF merge (last/first intrabar of the chart bar), no resampler,
            # no HTF aggregator — the chart targets its own bar period.
            if not same_tf:
                sec_seconds = tf_module._in_seconds(timeframe)
                chart_seconds = tf_module._in_seconds(chart_timeframe)
                plain_ltf = 0 < sec_seconds < chart_seconds
            resampler = (None if same_tf or plain_ltf
                         else Resampler.get_resampler(timeframe))

            # A None value is a runtime-deferred (input-derived) lookahead; OFF
            # serves as the placeholder until the first ``__sec_signal__``
            # delivers the actual value.
            lookahead_mode = _lookahead_mode(ctx.get('lookahead'))

            # Live HTF transport via the chart's ``HTFAggregator`` (closed-bar
            # override for all lookahead modes, plus developing-bar for
            # ``Lookahead.ON``) requires same-symbol chart→HTF aggregation.
            # The chart bar OHLCV must belong to the same instrument as the
            # security, so cross-symbol HTF keeps no aggregator: in backtest
            # it reads from the security's own ``.ohlcv`` file; in live mode
            # the security subprocess drives its own provider (warmup
            # download + WS stream) so the cross-symbol context advances on
            # real feed bars instead of staying inert.
            from .htf_aggregator import HTFAggregator

            if not same_tf and resampler is not None:
                sym = ctx.get('symbol')
                if sym is not None:
                    # Strip any chart-type marker (``ticker.heikinashi()``) so a
                    # static same-symbol chart-type HTF resolves as same-symbol
                    # (and gets an aggregator), not misrouted as cross-symbol.
                    from ..lib.ticker import _split_chart_type
                    sym, _ = _split_chart_type(str(sym))
                is_same_symbol = (
                    chart_symbols is None
                    or sym is None
                    or str(sym) in chart_symbols
                )

                # Intraday session anchoring: align HTF bars to the session open
                # (TradingView behaviour) when the open is off the requested tf
                # grid. Use the security's OWN syminfo — correct even for a
                # cross-symbol HTF in a different exchange session.
                si = (sec_syminfos.get(sec_id)
                      if sec_syminfos is not None else None) or chart_syminfo
                anchor_starts, anchor_tz, anchor_oh = resolve_session_anchor(
                    si, timeframe, tz)

                if is_same_symbol:
                    htf_aggregator = HTFAggregator(
                        timeframe, tz, session_starts=anchor_starts,
                        chart_span_ms=chart_off + 1 if chart_off else 0)
                elif lookahead_mode is Lookahead.ON:
                    # Cross-symbol HTF + lookahead_on: developing bar cannot
                    # be aggregated from chart OHLCV (wrong instrument). The
                    # subprocess still advances on closed cross-symbol HTF
                    # bars, but the chart-side read returns ``na`` on every
                    # chart bar inside an open HTF period — backtest never
                    # silently exposes a value live could not produce, and
                    # the ``close[1]`` idiom keeps working at the period
                    # boundary.
                    na_on_developing = True

        state = SecurityState(
            sec_id=sec_id,
            timeframe=timeframe,
            gaps_on=is_gaps_on,
            same_timeframe=same_tf,
            resampler=resampler,
            tz=tz,
            is_ltf=is_ltf,
            plain_ltf=plain_ltf,
            plain_ltf_span_ms=(tf_module._in_seconds(timeframe) * 1000
                               if plain_ltf else 0),
            lookahead=lookahead_mode,
            htf_aggregator=htf_aggregator,
            na_on_developing=na_on_developing,
            session_starts=anchor_starts,
            session_tz=anchor_tz,
            session_opening_hours=anchor_oh,
            chart_off=chart_off,
            chart_calendar=chart_calendar,
            chart_timeframe=chart_timeframe,
            depends=frozenset(ctx.get('depends') or ()),
            in_loop=bool(ctx.get('in_loop', False)),
            always_read=bool(ctx.get('always_read', False)),
            signal_per_bar=bool(ctx.get('signal_per_bar', False)),
            chart_resampler=chart_ltf_resampler if (is_ltf or plain_ltf) else None,
            chart_dwm_modifier=chart_ltf_modifier if (is_ltf or plain_ltf) else '',
        )
        # data_ready starts SET so reads before first signal return na (via result_size=0)
        state.data_ready.set()

        states[sec_id] = state

        # noinspection PyProtectedMember
        state.is_dwm = tf_module._process_tf(timeframe)[0] in ('D', 'W', 'M')

        result_block = ResultBlock(sec_id, create=True, version=0, size=INITIAL_RESULT_SIZE,
                                   prefix=sync_block.block_prefix(sec_id))
        result_blocks[sec_id] = result_block

    # The transformer records DIRECT producers per sid; a consumer must wait for
    # everything it transitively depends on, so close the relation here. The set
    # is only a FILTER and an error scope — over-approximating it costs extra
    # waiting, never correctness (deadlock-freedom rests on program order, not on
    # this set).
    direct = {sid: st.depends for sid, st in states.items()}
    for sid, st in states.items():
        closure: set[str] = set()
        stack = list(direct[sid])
        while stack:
            dep = stack.pop()
            if dep in closure or dep not in direct:
                continue
            closure.add(dep)
            stack.extend(direct[dep])
        st.depends = frozenset(closure)

    return states, sync_block, result_blocks


def inject_protocol(module, signal_fn, write_fn, read_fn, wait_fn,
                    active_security=None,
                    same_context: 'set[str] | frozenset[str]' = frozenset()):
    """
    Inject protocol functions and __active_security__ into a script module's globals.

    :param module: The script module
    :param signal_fn: __sec_signal__ implementation
    :param write_fn: __sec_write__ implementation
    :param read_fn: __sec_read__ implementation
    :param wait_fn: __sec_wait__ implementation
    :param active_security: None for chart context, sec_id for security context
    :param same_context: Frozenset of sec_ids sharing the chart's symbol+timeframe
    """
    module.__sec_signal__ = signal_fn
    module.__sec_write__ = write_fn
    module.__sec_read__ = read_fn
    module.__sec_wait__ = wait_fn
    module.__active_security__ = active_security
    module.__same_context__ = same_context


def cleanup_shared_memory(
    sync_block: SyncBlock,
    result_blocks: dict[str, ResultBlock],
):
    """
    Clean up all shared memory resources.

    :param sync_block: The sync block to close and unlink
    :param result_blocks: Result blocks to close and unlink
    """
    for rb in result_blocks.values():
        rb.close()
        rb.unlink()
    sync_block.close()
    sync_block.unlink()
