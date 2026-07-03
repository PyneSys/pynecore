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
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from multiprocessing import Event, Lock, connection
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from .datetime import parse_timezone
from .security_shm import (
    SyncBlock, ResultBlock, ResultReader, INITIAL_RESULT_SIZE,
    write_result,
)

if TYPE_CHECKING:
    from multiprocessing import Process
    from multiprocessing.synchronize import Event as EventType, Lock as LockType
    from typing import Callable
    from .resampler import Resampler
    from .syminfo import SymInfo, SymInfoSession, SymInfoInterval

logger = logging.getLogger(__name__)


# Liveness poll interval for security-process waits without a death watcher
# (legacy fallback). Short enough to detect a crashed child quickly.
_LIVENESS_POLL_SECONDS = 0.5


def watch_security_child(
    sec_id: str,
    proc: 'Process',
    failed_children: set[str],
    events: 'tuple[EventType, ...]',
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
    """
    def _watch() -> None:
        connection.wait([proc.sentinel])
        if proc.exitcode not in (0, None):
            failed_children.add(sec_id)
            for ev in events:
                ev.set()

    threading.Thread(target=_watch, daemon=True,
                     name=f"sec-watch-{sec_id}").start()


def _wait_with_liveness(
    event: 'EventType',
    sec_id: str,
    sec_processes: 'dict[str, Process] | None',
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
        if sec_id in failed_children:
            proc = sec_processes[sec_id]
            raise RuntimeError(
                f"Security process for '{sec_id}' died unexpectedly "
                f"(exit code: {proc.exitcode})"
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

    # Synthetic chart type requested via ``ticker.heikinashi()`` etc. ``None`` is
    # an ordinary feed. When set (currently only ``"heikinashi"``), the chart
    # process transforms this context's ``.ohlcv`` before the child reads it, so
    # the subprocess consumes the chart-type bars with no child-side change.
    # Backtest (file-backed) only — a live source raises at spawn.
    chart_type: str | None = None

    # LTF prefix-skip (chart-side, backtest/file-backed only). The LTF child's
    # ``.ohlcv`` feed first bar open, in ms. The child includes intrabars with
    # ``bar_open <= target_time`` and the historical target is the chart bar's
    # last ms (``chart_off``), so a chart bar contains an intrabar only when its
    # period end reaches the feed (``target_time >= ltf_first_ms``). A chart bar
    # whose whole period ends before the feed's first open therefore yields an
    # empty array unconditionally (TradingView returns ``na`` before the LTF
    # series begins). ``__sec_signal__`` then skips the per-bar signal+wait
    # handshake for that idle prefix and ``__sec_read__`` returns the empty-array
    # default without touching shared memory. ``None`` disables the optimization,
    # restoring the original per-chart-bar signal. Populated by
    # ``load_ltf_first_ms`` at child spawn.
    ltf_first_ms: int | None = None

    # Intraday session anchoring (chart-side). Populated by
    # ``setup_security_states`` only when this security's session opens off the
    # requested HTF grid (e.g. equities 09:30 at 1H). ``None`` selects the pure
    # UTC clock-floor fast path in ``Resampler.get_bar_time`` — zero overhead for
    # 24/7, on-hour, and session-aligned instruments. ``session_tz`` is the
    # security's own exchange timezone.
    session_starts: 'list[SymInfoSession] | None' = None
    session_tz: ZoneInfo | None = None

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

    # Set by ``__sec_signal__`` when an LTF chart bar precedes the feed (see
    # ``ltf_first_ms``): no handshake ran this bar, so ``__sec_read__`` returns
    # the empty-array default directly instead of waiting on shared memory.
    ltf_skip: bool = False


def _get_confirmed_time(state: SecurityState, chart_time: int) -> int:
    """
    Determine which security period is confirmed at the current chart time.

    For same timeframe: every chart bar is a confirmed bar (return chart_time).
    For HTF: the period is confirmed on its own LAST chart bar — the one whose
    CLOSE instant (``chart_time + chart_off + 1``) reaches the period end —
    per TradingView's historical ``lookahead_off`` merge rule, not one bar late
    on the next period's first bar. Daily/weekly/monthly HTF rides the child's
    actual bar opens when loaded (see ``SecurityState.bar_opens``) so a sparse
    child forward-fills its last value instead of confirming phantom calendar
    periods: multi-period (nD/nW/nM) walks the opens directly, single-period
    (1D/1W/1M) clamps the arithmetic grid's calendar-close target to the latest
    real open. Without loaded opens, the close instant is floored to the
    preceding resampler period (intraday HTF, live streams, unit tests).

    :param state: Security context state
    :param chart_time: Current chart bar time in milliseconds
    :return: Confirmed time in milliseconds
    """
    if state.same_timeframe:
        return chart_time

    resampler = state.resampler
    assert resampler is not None

    # The chart bar's close instant. ``chart_off`` is span-1 for intraday and
    # seconds charts; D/W/M chart bars have no fixed arithmetic span
    # (``chart_off == 0``), so confirmation degrades to the bar's open instant.
    close_time = chart_time + state.chart_off + 1

    if state.bar_opens is not None and state.bar_opens_multiperiod:
        # Multi-period walk: advance the pointer to the child bar the chart bar's
        # close instant falls in; entering bar ``ptr`` closes every bar before it
        opens = state.bar_opens
        ptr = state.bar_ptr
        n = len(opens)
        advanced = False
        while ptr + 1 < n and opens[ptr + 1] <= close_time:
            ptr += 1
            advanced = True
        state.bar_ptr = ptr
        if advanced and ptr >= 1:
            return opens[ptr - 1]
        if n and ptr == n - 1 and state.sec_grid_args is not None:
            # The chart marched past the child's last bar: no child bar
            # realizes the next period, so the arithmetic grid decides when
            # the last bar is closed
            sec_tz, ss, oh, mode = state.sec_grid_args
            if resampler.get_bar_time(close_time, sec_tz, ss, oh, mode) > opens[-1]:
                return opens[-1]
        return state.last_confirmed

    # OFF / lookahead_off: the period preceding the one ``close_time`` falls in.
    # When the chart bar's close lands exactly on a period boundary,
    # ``get_bar_time`` floors it to that boundary and the period ending there is
    # returned — the just-closed HTF bar is confirmed on its own last chart bar.
    if state.session_starts is not None:
        # Off-grid intraday session → anchor HTF bars to the session open,
        # using the security's own exchange timezone.
        period = resampler.get_bar_time(
            close_time, state.session_tz, state.session_starts)
        grid_target = resampler.get_bar_time(
            period - 1, state.session_tz, state.session_starts)
    else:
        period = resampler.get_bar_time(close_time, state.tz)
        grid_target = resampler.get_bar_time(period - 1, state.tz)

    if state.bar_opens is None:
        return grid_target

    opens = state.bar_opens
    n = len(opens)
    ptr = state.bar_ptr

    if state.bar_closes is not None:
        # Session-bounded intraday HTF: each real bar closes at its session's
        # scheduled end (calendar-known via ``opening_hours``), NOT at the
        # arithmetic next-period boundary — a non-trading gap before the next
        # session (e.g. the dead time between a futures day and night session)
        # would otherwise delay confirmation by a full period. TradingView
        # ``lookahead_off`` confirms the bar on its own last chart bar, whose
        # close coincides with the session end. Confirm the latest real bar whose
        # session end the chart bar's close (``close_time``) has reached.
        # ``close_time`` and ``bar_closes`` are both monotonic across chart bars,
        # so the persistent ``bar_ptr`` only ever advances.
        closes = state.bar_closes
        while ptr + 1 < n and closes[ptr + 1] <= close_time:
            ptr += 1
        state.bar_ptr = ptr
        if ptr >= 0 and closes[ptr] <= close_time:
            return opens[ptr]
        return state.last_confirmed

    # Single-period D/W/M with a loaded child: the grid above gives the correct
    # calendar close instant, but the child may carry a bar only on scattered
    # days (sparse macro series). Clamp the grid target to the latest real child
    # open so the subprocess never advances into an empty window (which would
    # ``write_na`` and wipe the forward-fill). Dense data has a bar on every
    # period, so the clamp is a no-op and behaviour is identical to the bare
    # grid; sparse data holds its last real value, matching TV ``gaps_off``.
    # ``grid_target`` is monotonically non-decreasing across chart bars (it is
    # ``get_bar_time(period - 1)`` of a monotonically increasing ``close_time``),
    # so the persistent ``bar_ptr`` only ever advances. ``last_confirmed`` is a
    # timestamp (0 = before any bar), returned while the chart precedes the first
    # real open so nothing is confirmed yet.
    while ptr + 1 < n and opens[ptr + 1] <= grid_target:
        ptr += 1
    state.bar_ptr = ptr
    if ptr >= 0 and opens[ptr] <= grid_target:
        return opens[ptr]
    return state.last_confirmed


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
    sec_processes: 'dict[str, Process] | None' = None,
    failed_children: 'set[str] | None' = None,
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
    :param failed_children: Shared registry filled by
                            :func:`watch_security_child` when a child dies
                            abnormally. Enables the cheap UNTIMED waits; when
                            None the waits fall back to liveness polling.
    :return: (sec_signal, sec_write, sec_read, sec_wait, cleanup)
    """
    readers: dict[str, ResultReader] = {
        sid: ResultReader(sid) for sid in states
    }

    resolved: set[str] = set()

    def __sec_signal__(sec_id: str, symbol: str | None = None,
                       timeframe: str | None = None, _scope_id=None):
        from pynecore import lib
        state = states[sec_id]

        # Resolve deferred symbol/timeframe on first call. The two callbacks are
        # NOT alternatives: in a script with both deferred and static contexts the
        # deferred resolver no-ops for a static sec_id (and the runtime symbol
        # argument is always present), so an elif here would leave every static
        # context's subprocess unspawned and its first real read deadlocked.
        # ``lazy_spawn_fn`` itself skips sids that already have a process.
        if sec_id not in resolved:
            resolved.add(sec_id)
            if deferred_resolve_fn is not None and symbol is not None:
                deferred_resolve_fn(sec_id, symbol, timeframe)
            if lazy_spawn_fn is not None:
                lazy_spawn_fn(sec_id)

        # No-process contexts (same-context, ignored): skip advance/wait
        if sec_id in no_process_ids:
            if sec_id in same_context_ids:
                state.new_period = True
                state.data_ready.clear()
            return

        # noinspection PyProtectedMember
        chart_time = lib._time

        if state.is_ltf:
            # Historical/file-backed LTF: the child includes intrabars with
            # ``bar_open <= target_time``. Target the chart bar's last ms so the
            # child returns the bar's OWN period — matching TradingView. Adjacent
            # bars tile with no gap or overlap (the prior bar targeted that ms
            # minus one). For live streams (``ltf_first_ms is None``) read-ahead
            # is impossible, so keep targeting the chart open (Phase 3).
            if state.ltf_first_ms is None:
                ltf_target_time = chart_time
            elif (state.chart_dwm_modifier and state.chart_resampler is not None
                  and state.chart_resampler.get_bar_time(chart_time, state.tz)
                  == chart_time):
                # Single-period civil D/W/M chart: ``chart_off`` is 0, so the
                # period end is the next civil open minus one ms (not a fixed
                # span). The civil-anchored guard above means we only do this
                # when the chart bar actually opens on the civil boundary; an
                # off-grid (session-anchored) D/W/M bar falls through to the
                # ``chart_off`` path, keeping the documented limitation rather
                # than mis-windowing on a civil-calendar guess.
                ltf_target_time = _next_civil_period_open(
                    state.chart_dwm_modifier, chart_time, state.tz) - 1
            else:
                # Intraday/seconds chart: ``chart_off`` == span-1 gives the
                # bar's own period ``[T, T+tf)``.
                ltf_target_time = chart_time + state.chart_off
            # Prefix skip: a chart bar whose whole period ends before the LTF
            # feed's first bar (``target_time < ltf_first_ms``) cannot contain an
            # intrabar, so the read is an empty array. Skip the cross-process
            # signal+wait entirely — ``__sec_read__`` returns the empty-array
            # default. Disabled (``ltf_first_ms is None``) for live streams,
            # which keep signalling every bar.
            if state.ltf_first_ms is not None and ltf_target_time < state.ltf_first_ms:
                state.ltf_skip = True
                state.new_period = True
                state.needs_wait = False
            else:
                state.ltf_skip = False
                # LTF: every chart bar needs intrabar data — always signal
                state.new_period = True
                state.data_ready.clear()
                sync_block.set_target_time(sec_id, ltf_target_time)
                state.advance_event.set()
                state.needs_wait = True
        else:
            target_time = _get_confirmed_time(state, chart_time)

            if target_time > state.last_confirmed:
                state.last_confirmed = target_time
                state.new_period = True
                state.data_ready.clear()
                sync_block.set_target_time(sec_id, target_time)
                state.advance_event.set()
                state.needs_wait = True
            else:
                state.new_period = False

    def __sec_write__(sec_id: str, value, _scope_id=None):
        if sec_id in same_context_ids and result_blocks is not None:
            with states[sec_id].result_lock:
                write_result(result_blocks[sec_id], sync_block, value)
            states[sec_id].data_ready.set()

    def __sec_read__(sec_id: str, default=None, _scope_id=None):
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
        _wait_with_liveness(state.data_ready, sec_id, sec_processes, failed_children)

        if not state.is_ltf and state.gaps_on and not state.new_period:
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
        state = states[sec_id]
        if state.needs_wait:
            _wait_with_liveness(state.done_event, sec_id, sec_processes, failed_children)
            state.done_event.clear()
            state.needs_wait = False

    def cleanup():
        for r in readers.values():
            r.close()

    return __sec_signal__, __sec_write__, __sec_read__, __sec_wait__, cleanup


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


def create_security_protocol(
    sec_id: str,
    sync_block: SyncBlock,
    result_block: ResultBlock,
    all_sec_ids: list[str],
    result_locks: 'dict[str, LockType]',
    is_ltf: bool = False,
) -> tuple:
    """
    Create protocol functions for a **security** process.

    In security context, __sec_signal__ and __sec_wait__ are no-ops (guarded by
    AST ``if __active_security__ is None`` checks). __sec_write__ writes to shared
    memory. __sec_read__ reads immediately without waiting (no deadlock).

    When ``is_ltf=True``, __sec_write__ appends to an internal buffer instead of
    writing to shared memory. The caller must invoke ``flush()`` at the end of
    each round to write the accumulated array.

    :param sec_id: This security context's ID (the only slot it writes to).
    :param sync_block: Shared memory sync block
    :param result_block: Shared memory result block for writing
    :param all_sec_ids: All security context IDs (for cross-context reads)
    :param result_locks: Per-slot ``multiprocessing.Lock`` keyed by sec_id.
                         Writers acquire ``result_locks[sec_id]``; cross-context
                         readers acquire ``result_locks[<peer sid>]``.
    :param is_ltf: If True, enable LTF accumulation mode.
    :return: (sec_signal, sec_write, sec_read, sec_wait, cleanup, flush)
             flush is None when is_ltf=False.
    """
    readers: dict[str, ResultReader] = {
        sid: ResultReader(sid) for sid in all_sec_ids
    }
    own_lock = result_locks[sec_id]

    def __sec_signal__(_sid: str, _symbol=None, _timeframe=None, _scope_id=None):
        pass

    if is_ltf:
        _buffer: list = []

        def __sec_write__(_sid: str, value, _scope_id=None):
            _buffer.append(value)

        def flush():
            with own_lock:
                write_result(result_block, sync_block, _buffer.copy())
            _buffer.clear()
    else:
        def __sec_write__(_sid: str, value, _scope_id=None):
            with own_lock:
                write_result(result_block, sync_block, value)

        flush = None

    def __sec_read__(sid: str, default=None, _scope_id=None):
        with result_locks[sid]:
            return readers[sid].read(sync_block, default)

    def __sec_wait__(_sid: str, _scope_id=None):
        pass

    def cleanup():
        for r in readers.values():
            r.close()

    return __sec_signal__, __sec_write__, __sec_read__, __sec_wait__, cleanup, flush


# Representative dates for the off-grid session probe — one on each side of the
# DST boundary so a session that lands on the tf grid only half the year is still
# detected. Both are Mondays, so the weekday offset arithmetic below is exact.
_WINTER_PROBE = date(2024, 1, 15)
_SUMMER_PROBE = date(2024, 7, 15)


def _needs_session_anchor(
    session_starts: 'list[SymInfoSession]',
    tzinfo: ZoneInfo | None,
    timeframe: str,
) -> bool:
    """
    Whether intraday ``timeframe`` bars need session anchoring for this market.

    Session anchoring changes nothing when every declared session open already
    lands on the ``timeframe`` UTC-epoch grid (24/7, on-hour, session-aligned
    markets), so those keep the zero-overhead clock-floor fast path. The open is
    probed on both a winter and a summer date to cover both DST offsets. The test
    is deliberately conservative: it anchors whenever any open is off-grid.
    """
    if not session_starts:
        return False
    # Local import: ``core`` ↔ ``lib`` would otherwise form an import cycle.
    from ..lib import timeframe as tf_module
    # noinspection PyProtectedMember
    modifier, _ = tf_module._process_tf(timeframe)
    if modifier not in ('S', ''):
        return False  # D/W/M alignment is timezone-driven, not session-anchored
    tf_seconds = tf_module.in_seconds(timeframe)
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
) -> 'tuple[list[SymInfoSession] | None, ZoneInfo | None]':
    """
    Decide intraday HTF session anchoring for one security context.

    Returns ``(session_starts, session_tz)`` to store on the ``SecurityState`` so
    ``_get_confirmed_time`` anchors HTF bars to the session open, or ``(None,
    None)`` when the market opens on the ``timeframe`` grid (the clock-floor fast
    path).

    :param si: The security's own ``SymInfo`` (``None`` → no anchoring).
    :param timeframe: The resolved HTF string (e.g. ``"60"``).
    :param fallback_tz: Timezone used if the syminfo timezone is missing/invalid.
    """
    if si is None or not getattr(si, 'session_starts', None):
        return None, None
    try:
        si_tz = parse_timezone(si.timezone) if si.timezone else fallback_tz
    except (ValueError, KeyError):
        si_tz = fallback_tz
    if _needs_session_anchor(si.session_starts, si_tz, timeframe):
        return si.session_starts, si_tz
    return None, None


def _session_bar_closes(
        opens: list[int],
        tz: ZoneInfo | None,
        opening_hours: list[SymInfoInterval],
        period_ms: int,
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

    :param opens: HTF bar opens in epoch ms, ascending.
    :param tz: The security's exchange timezone.
    :param opening_hours: The security's ``SymInfo.opening_hours`` intervals.
    :param period_ms: The HTF period length in milliseconds.
    :return: A parallel list of close instants (epoch ms), or ``None`` if any open
        has no containing interval — the schedule does not fully describe the
        feed, so the caller keeps the arithmetic grid clamp rather than risk a
        wrong session end.
    """
    from .resampler import crosses_midnight
    closes: list[int] = []
    for open_ms in opens:
        open_dt = datetime.fromtimestamp(open_ms / 1000, tz=tz)
        weekday = open_dt.weekday()
        prev_weekday = (weekday - 1) % 7
        open_time = open_dt.time()
        end_ms: int | None = None
        for interval in opening_hours:
            overnight = crosses_midnight(interval.start, interval.end)
            if (interval.day == weekday and interval.start <= open_time
                    and (overnight or open_time < interval.end)):
                # Same-day session, or the pre-midnight leg of an overnight one
                # (which closes on the following calendar day).
                end_date = open_dt.date() + timedelta(days=1 if overnight else 0)
            elif overnight and interval.day == prev_weekday and open_time < interval.end:
                # After-midnight leg of the PREVIOUS day's overnight session: the
                # bar opens today but its session started yesterday and closes
                # today (e.g. a 21:00->02:00 night session's 01:00 bar).
                end_date = open_dt.date()
            else:
                continue
            candidate = int(
                datetime.combine(end_date, interval.end, tzinfo=tz).timestamp() * 1000)
            if end_ms is None or candidate < end_ms:
                end_ms = candidate
        if end_ms is None:
            return None
        # Whichever comes first: the bar's own period end, or the session end (a
        # non-trading gap before the next session must not delay confirmation).
        closes.append(min(open_ms + period_ms, end_ms))
    return closes


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
        seg = _session_bar_closes(opens[i:j], tz, oh, period_ms)
        if seg is None:
            return None
        closes.extend(seg)
        i = j
    return closes


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
    * Gappy intraday HTF: ``OHLCVWriter`` forward-fills a session-gapped futures
      feed (e.g. a 720-minute HTF on a 3-session palm-oil contract) to a
      continuous grid, but the security child reads only the real bars
      (gap-compacted, see ``security_process``). The grid would then confirm
      phantom periods on the fills' timestamps. Dense intraday feeds keep the
      cheaper arithmetic grid (this stays a no-op for them); LTF contexts run
      their own intrabar machinery, not HTF confirmation.

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
    from .ohlcv_file import OHLCVReader
    from .resampler import grid_mode, overnight_opens, trading_day
    from .syminfo import SymInfo
    # noinspection PyProtectedMember
    modifier, multiplier = tf_module._process_tf(state.timeframe)
    is_dwm = modifier in ('D', 'W', 'M')

    if not is_dwm:
        # Intraday HTF: only a GAPPY feed needs the real-opens clamp (see above);
        # a dense feed keeps the arithmetic grid. LTF runs its own machinery.
        if state.is_ltf:
            return
        period_sec = tf_module.in_seconds(state.timeframe)
        with OHLCVReader(data_path) as reader:
            start_ts = reader.start_timestamp
            if start_ts is None:
                return
            opens = [candle.timestamp * 1000 for candle in reader.read_from(start_ts)]
            # A feed is dense only when its real bars tile the timeframe grid (file
            # interval == period). The row count alone is not enough: a session-
            # spaced feed whose bars sit wider than the period (e.g. a gap-free,
            # 24h-spaced 720-minute night future) has no gap fills, so
            # ``len(opens) == reader.size`` holds, yet its bars do NOT tile the
            # period grid and must still ride the session-close path below. A
            # single-record file has no interval to compare, so it keeps the dense
            # fast path as before.
            if len(opens) == reader.size and reader.interval in (None, period_sec):
                return  # dense feed: the arithmetic grid is already correct
        # Gappy fixed-span intraday HTF: keep the arithmetic (fixed-span) grid for
        # the close instant, but CLAMP it to the latest real open so an empty
        # (gap) period holds the last real bar instead of advancing into a phantom
        # period and writing na — see ``_get_confirmed_time``.
        state.bar_opens_multiperiod = False
    else:
        # Multi-period (nD/nW/nM) walks the opens directly (the arithmetic grid
        # cannot reproduce TradingView's scheduled multi-period calendar). Single
        # period (1D/1W/1M) instead uses the grid for the calendar close instant
        # and only *clamps* to these opens — see ``_get_confirmed_time``.
        state.bar_opens_multiperiod = multiplier > 1
        with OHLCVReader(data_path) as reader:
            if reader.size == 1:
                # A single-record feed has no derivable interval, so
                # ``start_timestamp`` stays ``None`` and ``read_from`` bails —
                # ``opens`` would be empty and ``_get_confirmed_time`` would then
                # never confirm the lone bar (the child reads ``na`` forever).
                # This is reachable when a finer base feed spans exactly one
                # requested D/W/M period and resamples to a single aggregate.
                # Read the one bar directly so its open still anchors the clamp.
                opens = [reader.read(0).timestamp * 1000]
            else:
                start_ts = reader.start_timestamp
                opens = ([] if start_ts is None else
                         [candle.timestamp * 1000 for candle in reader.read_from(start_ts)])

    state.bar_opens = opens
    state.bar_ptr = -1

    sec_tz: ZoneInfo | None = state.tz
    sec_starts = sec_hours = mode = None
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

        # Session-bounded intraday HTF (e.g. a futures contract's day/night
        # sessions): confirm each bar on its scheduled session end instead of the
        # arithmetic next-period boundary (see ``_get_confirmed_time``). Needs the
        # session schedule; ``None`` (no schedule, or a bar outside it) keeps the
        # grid clamp.
        if not is_dwm and sec_hours:
            period_ms = tf_module.in_seconds(state.timeframe) * 1000
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
                state.bar_closes = _session_bar_closes(opens, sec_tz, sec_hours, period_ms)
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
    memory. ``ltf_first_ms`` stays ``None`` (optimization disabled) when the feed
    has no first bar. No-op for non-LTF contexts.

    :param state: LTF security context state.
    :param data_path: Path to the child's OHLCV data file.
    """
    if not state.is_ltf:
        return
    from .ohlcv_file import OHLCVReader
    with OHLCVReader(data_path) as reader:
        start_ts = reader.start_timestamp
    state.ltf_first_ms = None if start_ts is None else int(start_ts * 1000)


def setup_security_states(
    contexts: dict[str, dict],
    chart_timeframe: str,
    tz: 'ZoneInfo',
    chart_syminfo: 'SymInfo | None' = None,
) -> tuple[dict[str, SecurityState], SyncBlock, dict[str, ResultBlock]]:
    """
    Initialize security states, shared memory, and events from ``__security_contexts__``.

    :param contexts: The ``__security_contexts__`` dict from the script module.
                     Keys are sec_ids, values are dicts with 'symbol', 'timeframe', 'gaps'.
    :param chart_timeframe: The chart's timeframe string (e.g., "5", "1D").
    :param tz: The chart's timezone.
    :param chart_syminfo: The chart symbol's ``SymInfo``, used as the session
                     source for intraday HTF session anchoring. ``None`` disables
                     anchoring (pure clock-floor).
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
    chart_off = (tf_module.in_seconds(chart_timeframe) * 1000 - 1
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

        anchor_starts: 'list[SymInfoSession] | None' = None
        anchor_tz: ZoneInfo | None = None
        if is_ltf:
            is_gaps_on = False
            same_tf = False
            resampler = None  # chart-side resampler not needed for LTF
        else:
            gaps_val = ctx.get('gaps', barmerge.gaps_off)
            is_gaps_on = gaps_val is barmerge.gaps_on
            same_tf = (timeframe == chart_timeframe)
            resampler = None if same_tf else Resampler.get_resampler(timeframe)

            # Intraday session anchoring: align HTF bars to the session open
            # (TradingView behaviour) when the open is off the requested tf grid.
            if not same_tf:
                anchor_starts, anchor_tz = resolve_session_anchor(
                    chart_syminfo, timeframe, tz)

        state = SecurityState(
            sec_id=sec_id,
            timeframe=timeframe,
            gaps_on=is_gaps_on,
            same_timeframe=same_tf,
            resampler=resampler,
            tz=tz,
            is_ltf=is_ltf,
            session_starts=anchor_starts,
            session_tz=anchor_tz,
            chart_off=chart_off,
            chart_resampler=chart_ltf_resampler if is_ltf else None,
            chart_dwm_modifier=chart_ltf_modifier if is_ltf else '',
        )
        # data_ready starts SET so reads before first signal return na (via result_size=0)
        state.data_ready.set()

        states[sec_id] = state

        result_block = ResultBlock(sec_id, create=True, version=0, size=INITIAL_RESULT_SIZE)
        result_blocks[sec_id] = result_block

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
