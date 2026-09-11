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
import threading
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
    FLAG_MORE_STEPS, RingReader, RingWriter, write_result,
)

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess
    from multiprocessing.synchronize import (
        Condition as ConditionType, Event as EventType, Lock as LockType,
    )
    from multiprocessing.connection import Connection
    from typing import Callable
    from .ohlcv import OHLCVReader
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

    # Live step queue (``__sec_signal__`` enqueues, ``__sec_read__`` and the
    # runner's bar-end hook drive it). Each entry is a zero-argument callable
    # that writes the slot and sets ``advance_event`` for one round.
    pending_live: list = field(default_factory=list)
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
    :param lazy_spawn_fn: Optional callback for lazy-spawning static security processes.
                          Called with sec_id on first __sec_signal__ for static contexts.
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
    ring_conditions = ring_conditions or {}
    consumers_by_sid = consumers_by_sid or {}

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
        """
        cond = ring_conditions.get(sec_id)
        if cond is None or not consumers_by_sid.get(sec_id):
            return
        asof = chart_asof(state, round_state['chart_time'], round_state['next_time'],
                          0 if lib.barstate.isconfirmed else round_state['tick'])
        with cond:
            if asof > sync_block.get_frontier_close(sec_id):
                sync_block.set_frontier_close(sec_id, asof)
            cond.notify_all()

    def _launch(sec_id: str, state: SecurityState) -> None:
        """Hand the prepared slot to the child and count the round.

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

    def __sec_signal__(sec_id: str, symbol: str | None = None,
                       timeframe: str | None = None, lookahead=None,
                       _scope_id=None):
        state = states[sec_id]

        # Resolve deferred symbol/timeframe on first call. The two callbacks are
        # NOT alternatives: in a script with both deferred and static contexts the
        # deferred resolver no-ops for a static sec_id (and the runtime symbol
        # argument is always present), so an elif here would leave every static
        # context's subprocess unspawned and its first real read deadlocked.
        # ``lazy_spawn_fn`` itself skips sids that already have a process.
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
            if lazy_spawn_fn is not None:
                lazy_spawn_fn(sec_id)

        # No-process contexts (same-context, ignored): skip advance/wait
        if sec_id in no_process_ids:
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
            _launch(sec_id, state)
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
                _launch(sec_id, state)
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
            _launch(sec_id, state)
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
                            _launch(sec_id, state)

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
                        _launch(sec_id, state)

                    steps.append(_closed_step)

                # Developing bar — only for ``Lookahead.ON``. ``dev_bar`` is
                # None when the confirmed chart bar just completed the period
                # (the closed step delivered it); no fresh developing bar exists
                # until the next chart bar.
                if state.lookahead is Lookahead.ON and dev_bar is not None:
                    def _developing_step(_bar=dev_bar):
                        sync_block.set_developing_bar(
                            sec_id, _bar.open, _bar.high, _bar.low,
                            _bar.close, _bar.volume, _bar.period_start,
                        )
                        sync_block.set_flags(sec_id, (
                            sync_block.get_flags(sec_id)
                            & ~(FLAG_CLOSED_OVERRIDE | FLAG_DEV_HISTORICAL)
                        ) | FLAG_IS_DEVELOPING | hist_phase)
                        sync_block.set_target_time(sec_id, _bar.period_start)
                        _launch(sec_id, state)

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
                    # The queue is filled FIRST: ``_launch`` reads it to decide
                    # whether this round is the chart bar's last publication.
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
            from ..lib import request
            from math import isnan
            from_cur, to_cur = currency_conversions[sec_id]
            rate = request.currency_rate(from_cur, to_cur)
            if not isnan(rate):
                if isinstance(result, (int, float)):
                    result = result * rate
                elif isinstance(result, tuple):
                    result = tuple(
                        v * rate if isinstance(v, (int, float)) else v for v in result
                    )

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
    if consumer_ids and sec_id in ring_conditions:
        consumer_indexes = [sync_block.index_of(cid) for cid in consumer_ids]
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
            return last_own_value[0]
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
            writer.set_frontier_close(ctx.frontier)
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
    for open_ms in opens:
        end_ms = _session_end_of_open(open_ms, tz, opening_hours, corrections)
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
    from .resampler import crosses_midnight
    open_dt = datetime.fromtimestamp(open_ms / 1000, tz=tz)
    open_date = open_dt.date()
    weekday = open_dt.weekday()
    prev_weekday = (weekday - 1) % 7
    open_time = open_dt.time()
    if corrections:
        today_hours = corrections.get(open_date, opening_hours)
        prev_hours = corrections.get(open_date - timedelta(days=1), opening_hours)
    else:
        today_hours = prev_hours = opening_hours
    end_ms: int | None = None
    for interval in today_hours:
        overnight = crosses_midnight(interval.start, interval.end)
        if (interval.day == weekday and interval.start <= open_time
                and (overnight or open_time < interval.end)):
            # Same-day session, or the pre-midnight leg of an overnight one
            # (which closes on the following calendar day).
            end_date = open_date + timedelta(days=1 if overnight else 0)
        else:
            continue
        candidate = int(
            datetime.combine(end_date, interval.end, tzinfo=tz).timestamp() * 1000)
        if end_ms is None or candidate < end_ms:
            end_ms = candidate
    for interval in prev_hours:
        # After-midnight leg of the PREVIOUS day's overnight session: the bar
        # opens today but its session started yesterday and closes today
        # (e.g. a 21:00->02:00 night session's 01:00 bar).
        if (crosses_midnight(interval.start, interval.end)
                and interval.day == prev_weekday and open_time < interval.end):
            candidate = int(
                datetime.combine(open_date, interval.end, tzinfo=tz).timestamp() * 1000)
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

    period_end = _dwm_period_end(open_ms, cal, timeframe)
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
    from .htf_aggregator import HTFAggregator

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
