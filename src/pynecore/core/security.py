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

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum, auto
from multiprocessing import Event, Lock
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from .datetime import parse_timezone
from .security_shm import (
    SyncBlock, ResultBlock, ResultReader, INITIAL_RESULT_SIZE,
    FLAG_IS_DEVELOPING, FLAG_CLOSED_OVERRIDE,
    write_result,
)

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess
    from multiprocessing.synchronize import Event as EventType, Lock as LockType
    from typing import Callable
    from .resampler import Resampler
    from .htf_aggregator import HTFAggregator
    from .syminfo import SymInfo, SymInfoSession


class Lookahead(Enum):
    """Lookahead mode for a security context.

    OFF
        TV-faithful default. The security context advances only to the bar
        that has CLOSED at or before the chart bar's time. In historical
        mode this matches TradingView's ``barmerge.lookahead_off`` exactly.
        In live mode the chart-side ``HTFAggregator`` ships each freshly
        closed HTF bar to the subprocess via the SyncBlock (the static
        ``.ohlcv`` file cannot grow at runtime), so the security series
        advances on every HTF period close — no developing-bar exposure.

    LAST_CLOSED
        PyneSys-native, repaint-free alternative. Always returns the most
        recently closed security bar. In historical mode it is functionally
        equivalent to ``OFF``; in live mode it uses the same closed-bar
        transport as ``OFF`` (no developing exposure) and remains
        repaint-free. Recommended for non-charting backtests when the TV
        ``close[1]`` idiom is not desired.

    ON
        TV ``lookahead_on`` semantics — the security context steps into
        the bar that *contains* the chart bar's time. In live mode the
        bar runs with ``barstate.isconfirmed=False`` and OHLCV aggregated
        from the chart timeframe by ``HTFAggregator``; on HTF period
        boundaries the closed bar is delivered first (snapshot saved),
        then the fresh developing bar. In historical mode there is no
        chart-derived developing OHLCV, so ``_get_confirmed_time`` falls
        back to ``OFF`` semantics (most recently closed bar) — historical
        backtests therefore never expose a developing security close,
        avoiding TV's classical historical future-leak. The TV idiom
        ``request.security(..., lookahead_on)[1]`` remains TV-compatible
        in both modes because ``close[1]`` is always the previously
        closed bar regardless of whether ``close[0]`` is developing or
        the same closed bar.

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


# Liveness poll interval for security-process waits. Short enough to detect a
# crashed child quickly, large enough that the wakeup overhead is negligible
# vs. the typical per-bar processing time.
_LIVENESS_POLL_SECONDS = 0.5


def _wait_with_liveness(
    event: 'EventType',
    sec_id: str,
    sec_processes: 'dict[str, BaseProcess] | None',
) -> None:
    """
    Wait for ``event`` while polling the owning security process for liveness.

    If the security process dies before signalling the event, raise
    ``RuntimeError`` instead of deadlocking the chart forever.

    Same-context and ignored sec_ids have no associated Process — they fall
    back to a plain unbounded ``event.wait()`` because their signalling is
    driven by the chart itself, not a separate process.
    """
    if sec_processes is None or sec_id not in sec_processes:
        event.wait()
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

    # Intraday session anchoring (chart-side). Populated by
    # ``setup_security_states`` only when this security's session opens off the
    # requested HTF grid (e.g. equities 09:30 at 1H). ``None`` selects the pure
    # UTC clock-floor fast path in ``Resampler.get_bar_time`` — zero overhead for
    # 24/7, on-hour, and session-aligned instruments. ``session_tz`` is the
    # security's own exchange timezone (correct even for cross-symbol HTF).
    session_starts: 'list[SymInfoSession] | None' = None
    session_tz: ZoneInfo | None = None

    # Tracking (chart-side only)
    last_confirmed: int = 0
    prev_chart_time: int | None = None
    needs_wait: bool = False
    new_period: bool = False


def _get_confirmed_time(state: SecurityState, chart_time: int) -> int:
    """
    Determine which security period the subprocess should advance to.

    Same timeframe: every chart bar is itself a confirmed bar (return chart_time).

    HTF:
      * ``OFF`` / ``LAST_CLOSED``: target is the previously CLOSED period's
        opening time. The subprocess advances only when a new HTF period opens.
        Historically these two modes are equivalent.
      * ``ON``: target is the CONTAINING period's opening time — the subprocess
        steps into the developing HTF bar (live mode supplies developing OHLCV
        via the SyncBlock; historical mode falls back to OFF semantics because
        there is no chart-derived developing OHLCV to feed the subprocess).

    :param state: Security context state
    :param chart_time: Current chart bar time in milliseconds
    :return: Target time in milliseconds
    """
    if state.same_timeframe:
        return chart_time

    prev_chart_time = state.prev_chart_time
    if prev_chart_time is None:
        return state.last_confirmed

    resampler = state.resampler
    assert resampler is not None
    if state.session_starts is not None:
        # Off-grid intraday session → anchor HTF bars to the session open,
        # using the security's own exchange timezone.
        current_period = resampler.get_bar_time(
            chart_time, state.session_tz, state.session_starts)
        prev_period = resampler.get_bar_time(
            prev_chart_time, state.session_tz, state.session_starts)
    else:
        current_period = resampler.get_bar_time(chart_time, state.tz)
        prev_period = resampler.get_bar_time(prev_chart_time, state.tz)

    if (state.lookahead is Lookahead.ON and state.is_live
            and state.htf_aggregator is not None):
        # Live ``lookahead_on`` with a chart-side aggregator: step into the
        # containing (developing) period. Cross-symbol HTF has no aggregator
        # (chart OHLCV is the wrong instrument), so the subprocess keeps
        # closed-bar semantics instead and the chart-side read returns ``na``
        # while a period is open (``na_on_developing``).
        return current_period

    if current_period != prev_period:
        # New period started → previous period is now confirmed
        return prev_period
    else:
        return state.last_confirmed


def create_chart_protocol(
    states: dict[str, SecurityState],
    sync_block: SyncBlock,
    deferred_resolve_fn: 'Callable[[str, str, str | None], None] | None' = None,
    lazy_spawn_fn: 'Callable[[str], None] | None' = None,
    same_context_ids: frozenset[str] = frozenset(),
    no_process_ids: 'set[str] | frozenset[str]' = frozenset(),
    result_blocks: dict[str, ResultBlock] | None = None,
    currency_conversions: dict[str, tuple[str, str]] | None = None,
    sec_processes: 'dict[str, BaseProcess] | None' = None,
    auto_rate_sec_ids: frozenset[str] = frozenset(),
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
    :return: (sec_signal, sec_write, sec_read, sec_wait, cleanup,
              signal_rate_sources)
    """
    readers: dict[str, ResultReader] = {
        sid: ResultReader(sid) for sid in states
    }

    resolved: set[str] = set()

    def __sec_signal__(sec_id: str, symbol: str | None = None,
                       timeframe: str | None = None, _scope_id=None):
        from pynecore import lib
        state = states[sec_id]

        # Resolve deferred symbol/timeframe on first call
        if sec_id not in resolved:
            resolved.add(sec_id)
            if deferred_resolve_fn is not None and symbol is not None:
                deferred_resolve_fn(sec_id, symbol, timeframe)
            elif lazy_spawn_fn is not None:
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
            # LTF: every chart bar needs intrabar data — always signal
            state.new_period = True
            state.data_ready.clear()
            sync_block.set_target_time(sec_id, chart_time)
            state.advance_event.set()
            state.needs_wait = True
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
            )

            if state.is_live:
                state.prev_chart_time = chart_time

                # Phase 1: synchronously deliver any just-closed HTF bar
                if closed_bar is not None:
                    sync_block.set_developing_bar(
                        sec_id,
                        closed_bar.open, closed_bar.high, closed_bar.low,
                        closed_bar.close, closed_bar.volume,
                        closed_bar.period_start,
                    )
                    base_flags = (
                        sync_block.get_flags(sec_id) & ~FLAG_IS_DEVELOPING
                    ) | FLAG_CLOSED_OVERRIDE
                    sync_block.set_flags(sec_id, base_flags)
                    sync_block.set_target_time(sec_id, closed_bar.period_start)
                    state.last_confirmed = closed_bar.period_start
                    state.data_ready.clear()
                    state.advance_event.set()
                    # Block until the subprocess finishes processing the closed
                    # bar (writes result, saves var_snapshot). For ON, this also
                    # ensures the developing-bar phase below cannot race ahead
                    # of the closed phase.
                    _wait_with_liveness(state.done_event, sec_id, sec_processes)
                    state.done_event.clear()

                # Phase 2: developing bar — only for ``Lookahead.ON``.
                if state.lookahead is Lookahead.ON:
                    sync_block.set_developing_bar(
                        sec_id,
                        dev_bar.open, dev_bar.high, dev_bar.low,
                        dev_bar.close, dev_bar.volume, dev_bar.period_start,
                    )
                    base_flags = (
                        sync_block.get_flags(sec_id) & ~FLAG_CLOSED_OVERRIDE
                    ) | FLAG_IS_DEVELOPING
                    sync_block.set_flags(sec_id, base_flags)
                    sync_block.set_target_time(sec_id, dev_bar.period_start)
                    state.new_period = True
                    state.data_ready.clear()
                    state.advance_event.set()
                    state.needs_wait = True
                    return

                # OFF / LAST_CLOSED in live mode: closed-bar transport only.
                # ``new_period`` reflects whether a fresh HTF close just landed
                # (drives ``gaps_on`` na/value selection in ``__sec_read__``).
                # We already waited synchronously inside Phase 1, so no further
                # wait is needed in ``__sec_wait__``.
                state.new_period = closed_bar is not None
                state.needs_wait = False
                # Clear any stale developing flag from a prior ``Lookahead.ON``
                # session (defensive — same SyncBlock slot).
                if closed_bar is None:
                    stale_flags = sync_block.get_flags(sec_id) & ~(
                        FLAG_IS_DEVELOPING | FLAG_CLOSED_OVERRIDE
                    )
                    sync_block.set_flags(sec_id, stale_flags)
                return
            # Warmup with an HTF aggregator falls through to the closed-only
            # flow below; the aggregator state has already advanced so the
            # live transition starts with the correct in-progress HTF bar.

        # Closed-only flow (historical / lookahead_off / lookahead_last_closed)
        target_time = _get_confirmed_time(state, chart_time)
        state.prev_chart_time = chart_time

        if target_time > state.last_confirmed:
            state.last_confirmed = target_time
            state.new_period = True
            state.data_ready.clear()
            # Make sure no stale developing/override flag leaks across modes.
            stale_flags = sync_block.get_flags(sec_id) & ~(
                FLAG_IS_DEVELOPING | FLAG_CLOSED_OVERRIDE
            )
            sync_block.set_flags(sec_id, stale_flags)
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
        # ``ignore_invalid_symbol=True`` may downgrade a live security to
        # ``no-process`` after syminfo prefetch fails — no subprocess is
        # ever spawned, so ``data_ready`` would never be set and a plain
        # ``_wait_with_liveness`` would deadlock here. Short-circuit to
        # ``default`` (Pine ``na``) so the script keeps running.
        if sec_id in no_process_ids and sec_id not in same_context_ids:
            return default
        state = states[sec_id]
        _wait_with_liveness(state.data_ready, sec_id, sec_processes)

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
        state = states[sec_id]
        if state.needs_wait:
            _wait_with_liveness(state.done_event, sec_id, sec_processes)
            state.done_event.clear()
            state.needs_wait = False

    def cleanup():
        for r in readers.values():
            r.close()

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
        from pynecore import lib
        # noinspection PyProtectedMember
        chart_time = lib._time
        for sec_id in auto_rate_sec_ids:
            if sec_id in no_process_ids or sec_id not in states:
                continue
            state = states[sec_id]
            sync_block.set_target_time(sec_id, chart_time)
            state.data_ready.clear()
            state.advance_event.set()
            _wait_with_liveness(state.data_ready, sec_id, sec_processes)

    return (
        __sec_signal__, __sec_write__, __sec_read__, __sec_wait__,
        cleanup, signal_rate_sources,
    )


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
    path). ``session_tz`` is the security's own exchange timezone — correct even
    for a cross-symbol HTF in a different session.

    :param si: The security's own ``SymInfo`` (``None`` → no anchoring).
    :param timeframe: The resolved HTF string (e.g. ``"60"``).
    :param fallback_tz: Timezone used if the syminfo timezone is missing/invalid.
    """
    if si is None or not getattr(si, 'session_starts', None):
        return None, None
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
        return si.session_starts, si_tz
    return None, None


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
                        OHLCV would be the wrong instrument. ``None`` (unit-test
                        / legacy callers without symbol context) is treated as
                        "every HTF is same-symbol".
    :param chart_syminfo: The chart symbol's ``SymInfo``, used as the session
                        source for same-symbol HTF anchoring. ``None`` disables
                        anchoring unless a per-security syminfo is supplied.
    :param sec_syminfos: ``sec_id → SymInfo`` for cross-symbol contexts, used so
                        each security anchors to its own session/timezone. ``None``
                        falls back to ``chart_syminfo``.
    :return: (states, sync_block, result_blocks)
    """
    from pynecore.lib import barmerge
    from .resampler import Resampler
    from .htf_aggregator import HTFAggregator

    sec_ids = list(contexts.keys())
    sync_block = SyncBlock(sec_ids)
    states: dict[str, SecurityState] = {}
    result_blocks: dict[str, ResultBlock] = {}

    for sec_id, ctx in contexts.items():
        timeframe = str(ctx.get('timeframe', chart_timeframe))
        is_ltf = bool(ctx.get('is_ltf', False))

        htf_aggregator: HTFAggregator | None = None
        na_on_developing = False
        anchor_starts: 'list[SymInfoSession] | None' = None
        anchor_tz: ZoneInfo | None = None
        if is_ltf:
            is_gaps_on = False
            same_tf = False
            resampler = None  # chart-side resampler not needed for LTF
            lookahead_mode = Lookahead.OFF  # LTF has no lookahead concept
        else:
            gaps_val = ctx.get('gaps', barmerge.gaps_off)
            is_gaps_on = gaps_val is barmerge.gaps_on
            same_tf = (timeframe == chart_timeframe)
            resampler = None if same_tf else Resampler.get_resampler(timeframe)

            lookahead_val = ctx.get('lookahead', barmerge.lookahead_off)
            if lookahead_val is barmerge.lookahead_on:
                lookahead_mode = Lookahead.ON
            elif lookahead_val is barmerge.lookahead_last_closed:
                lookahead_mode = Lookahead.LAST_CLOSED
            else:
                lookahead_mode = Lookahead.OFF

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
                is_same_symbol = (
                    chart_symbol is None
                    or sym is None
                    or str(sym) == chart_symbol
                )

                # Intraday session anchoring: align HTF bars to the session open
                # (TradingView behaviour) when the open is off the requested tf
                # grid. Use the security's OWN syminfo — correct even for a
                # cross-symbol HTF in a different exchange session.
                si = (sec_syminfos.get(sec_id)
                      if sec_syminfos is not None else None) or chart_syminfo
                anchor_starts, anchor_tz = resolve_session_anchor(si, timeframe, tz)

                if is_same_symbol:
                    htf_aggregator = HTFAggregator(
                        timeframe, tz, session_starts=anchor_starts)
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
            lookahead=lookahead_mode,
            htf_aggregator=htf_aggregator,
            na_on_developing=na_on_developing,
            session_starts=anchor_starts,
            session_tz=anchor_tz,
        )
        # data_ready starts SET so reads before first signal return na (via result_size=0)
        state.data_ready.set()

        states[sec_id] = state

        result_block = ResultBlock(sec_id, create=True, version=0, size=INITIAL_RESULT_SIZE)
        result_blocks[sec_id] = result_block

    return states, sync_block, result_blocks


def inject_protocol(module, signal_fn, write_fn, read_fn, wait_fn,
                    active_security=None, same_context: frozenset[str] = frozenset()):
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
