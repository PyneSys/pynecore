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
from multiprocessing import Event
from typing import TYPE_CHECKING

from .security_shm import (
    SyncBlock, ResultBlock, ResultReader, INITIAL_RESULT_SIZE,
    write_result, write_na,
)

if TYPE_CHECKING:
    from typing import Callable
    from zoneinfo import ZoneInfo
    from .resampler import Resampler


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
    data_ready: Event = field(default_factory=Event)
    advance_event: Event = field(default_factory=Event)
    done_event: Event = field(default_factory=Event)
    stop_event: Event = field(default_factory=Event)

    # Tracking (chart-side only)
    last_confirmed: int = 0
    prev_chart_time: int | None = None
    needs_wait: bool = False
    new_period: bool = False


def _get_confirmed_time(state: SecurityState, chart_time: int) -> int:
    """
    Determine which security period is confirmed at the current chart time.

    For same timeframe: every chart bar is a confirmed bar (return chart_time).
    For HTF: a period is confirmed only when a NEW period starts — the previous
    period's opening time is returned (lookahead_off semantics).

    :param state: Security context state
    :param chart_time: Current chart bar time in milliseconds
    :return: Confirmed time in milliseconds
    """
    if state.same_timeframe:
        return chart_time

    prev_chart_time = state.prev_chart_time
    if prev_chart_time is None:
        return state.last_confirmed

    resampler = state.resampler
    current_period = resampler.get_bar_time(chart_time, state.tz)
    prev_period = resampler.get_bar_time(prev_chart_time, state.tz)

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
    no_process_ids: frozenset[str] = frozenset(),
    result_blocks: dict[str, ResultBlock] | None = None,
) -> tuple:
    """
    Create protocol functions for the **chart** process.

    :param deferred_resolve_fn: Optional callback for resolving deferred security contexts.
                                Called with (sec_id, symbol, timeframe) on first __sec_signal__.
    :param lazy_spawn_fn: Optional callback for lazy-spawning static security processes.
                          Called with sec_id on first __sec_signal__ for static contexts.
    :param same_context_ids: Security IDs that share the chart's symbol+timeframe.
                             These are handled directly by the chart (no separate process).
    :param no_process_ids: Security IDs that have no process (same-context + ignored).
                           Signal/wait are skipped for these.
    :param result_blocks: Result blocks for writing same-context values to shared memory.
    :return: (sec_signal, sec_write, sec_read, sec_wait, cleanup)
    """
    readers: dict[str, ResultReader] = {
        sid: ResultReader(sid) for sid in states
    }

    resolved: set[str] = set()

    def __sec_signal__(sec_id: str, symbol: str | None = None,
                       timeframe: str | None = None, scope_id=None):
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

        chart_time = lib._time

        target_time = _get_confirmed_time(state, chart_time)
        state.prev_chart_time = chart_time

        if target_time > state.last_confirmed:
            state.last_confirmed = target_time
            state.new_period = True
            state.data_ready.clear()
            sync_block.set_target_time(sec_id, target_time)
            state.advance_event.set()
            state.needs_wait = True
        else:
            state.new_period = False

    def __sec_write__(sec_id: str, value, scope_id=None):
        if sec_id in same_context_ids and result_blocks is not None:
            write_result(result_blocks[sec_id], sync_block, value)
            states[sec_id].data_ready.set()

    def __sec_read__(sec_id: str, default=None, scope_id=None):
        state = states[sec_id]
        state.data_ready.wait()

        if state.gaps_on and not state.new_period:
            return default

        return readers[sec_id].read(sync_block, default)

    def __sec_wait__(sec_id: str, scope_id=None):
        state = states[sec_id]
        if state.needs_wait:
            state.done_event.wait()
            state.done_event.clear()
            state.needs_wait = False

    def cleanup():
        for r in readers.values():
            r.close()

    return __sec_signal__, __sec_write__, __sec_read__, __sec_wait__, cleanup


def create_security_protocol(
    sec_id: str,
    sync_block: SyncBlock,
    result_block: ResultBlock,
    all_sec_ids: list[str],
) -> tuple:
    """
    Create protocol functions for a **security** process.

    In security context, __sec_signal__ and __sec_wait__ are no-ops (guarded by
    AST ``if __active_security__ is None`` checks). __sec_write__ writes to shared
    memory. __sec_read__ reads immediately without waiting (no deadlock).

    :return: (sec_signal, sec_write, sec_read, sec_wait, cleanup)
    """
    readers: dict[str, ResultReader] = {
        sid: ResultReader(sid) for sid in all_sec_ids
    }

    def __sec_signal__(sid: str, symbol=None, timeframe=None, scope_id=None):
        pass

    def __sec_write__(sid: str, value, scope_id=None):
        write_result(result_block, sync_block, value)

    def __sec_read__(sid: str, default=None, scope_id=None):
        return readers[sid].read(sync_block, default)

    def __sec_wait__(sid: str, scope_id=None):
        pass

    def cleanup():
        for r in readers.values():
            r.close()

    return __sec_signal__, __sec_write__, __sec_read__, __sec_wait__, cleanup


def setup_security_states(
    contexts: dict[str, dict],
    chart_timeframe: str,
    tz: 'ZoneInfo',
) -> tuple[dict[str, SecurityState], SyncBlock, dict[str, ResultBlock]]:
    """
    Initialize security states, shared memory, and events from ``__security_contexts__``.

    :param contexts: The ``__security_contexts__`` dict from the script module.
                     Keys are sec_ids, values are dicts with 'symbol', 'timeframe', 'gaps'.
    :param chart_timeframe: The chart's timeframe string (e.g., "5", "1D").
    :param tz: The chart's timezone.
    :return: (states, sync_block, result_blocks)
    """
    from pynecore.lib import barmerge
    from .resampler import Resampler

    sec_ids = list(contexts.keys())
    sync_block = SyncBlock(sec_ids)
    states: dict[str, SecurityState] = {}
    result_blocks: dict[str, ResultBlock] = {}

    for sec_id, ctx in contexts.items():
        timeframe = str(ctx.get('timeframe', chart_timeframe))
        gaps_val = ctx.get('gaps', barmerge.gaps_off)
        is_gaps_on = gaps_val is barmerge.gaps_on

        same_tf = (timeframe == chart_timeframe)

        resampler = None if same_tf else Resampler.get_resampler(timeframe)

        state = SecurityState(
            sec_id=sec_id,
            timeframe=timeframe,
            gaps_on=is_gaps_on,
            same_timeframe=same_tf,
            resampler=resampler,
            tz=tz,
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
