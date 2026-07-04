"""
Security process loop — multiprocessing.Process target for request.security() contexts.

Each security context runs as a separate OS process with its own Python interpreter,
lib module, and Series state. The process loads its own OHLCV data, re-imports the
script module (triggering AST transformation), and runs the main() function per bar.

Communication with the chart process uses shared memory + Events (see security.py).
"""
from __future__ import annotations

import logging
import os
import threading
from functools import partial
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING

from .security_shm import (
    SyncBlock, ResultBlock, write_na,
)
from .security import (
    create_security_protocol, inject_protocol,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any
    from multiprocessing.synchronize import Lock as LockType

# Seconds between parent-liveness checks in the orphan watchdog.
_ORPHAN_CHECK_INTERVAL = 2.0


def _start_parent_death_watchdog() -> None:
    """Hard-exit this security process if its parent dies without cleaning up.

    A security context runs as a ``daemon=True`` :class:`multiprocessing.Process`.
    A *clean* parent exit tears it down (daemon atexit + the runner's ``finally``
    that sets ``stop_event`` and joins). But a *hard* kill of the parent — a
    ``SIGKILL``, or a ``subprocess`` timeout that kills only the direct child —
    skips all of that: the child reparents to init, never receives ``stop_event``,
    and on macOS (where timed Event waits fall back to select() polling) spins at
    100% CPU while pinning its OHLCV and interpreter memory. macOS has no
    ``PR_SET_PDEATHSIG``, so the portable safety net is a watchdog thread that
    notices the reparent and exits.

    The captured parent PID is the spawning runner; when ``os.getppid()`` changes,
    the parent is gone and this orphan exits via ``os._exit`` — skipping
    atexit/``finally``, which could deadlock on shared memory the dead parent
    still holds.
    """
    parent_pid = os.getppid()

    def _watch() -> None:
        while True:
            sleep(_ORPHAN_CHECK_INTERVAL)  # watchdog tick, not a poll-retry
            if os.getppid() != parent_pid:
                logger.warning(
                    "Security process %d orphaned (parent %d gone); exiting.",
                    os.getpid(), parent_pid,
                )
                os._exit(1)

    threading.Thread(target=_watch, daemon=True, name="sec-parent-watchdog").start()


# noinspection PyProtectedMember
def security_process_main(
        sec_id: str,
        script_path: str,
        ohlcv_path: str,
        sync_block_name: str,
        all_sec_ids: list[str],
        # Events (multiprocessing.Event — picklable across spawn)
        data_ready_event,
        advance_event,
        done_event,
        stop_event,
        is_ltf: bool = False,
        result_locks: 'dict[str, LockType] | None' = None,
        ohlcv_fields: 'list[str] | None' = None,
        ohlcv_tuple: bool = False,
        same_timeframe: bool = False,
):
    assert result_locks is not None, "result_locks must be provided by script_runner"
    """
    Entry point for a security process (multiprocessing.Process target).

    Re-registers import hooks (needed for spawn mode on macOS/Windows),
    re-imports the script, and runs the bar loop.

    :param sec_id: This security context's unique ID
    :param script_path: Path to the script .py file
    :param ohlcv_path: Path to the OHLCV data file (.ohlcv)
    :param sync_block_name: SharedMemory name of the SyncBlock
    :param all_sec_ids: List of ALL security context IDs (for cross-reads)
    :param data_ready_event: Event signaling data is available for reading
    :param advance_event: Event signaling this process should advance
    :param done_event: Event signaling this process finished its current round
    :param stop_event: Event signaling this process should shut down
    :param is_ltf: If True, accumulate expression values into array per round
    :param ohlcv_fields: When set, the requested expression is only raw price
        series (open/high/low/close/volume/hl2/hlc3/ohlc4/hlcc4); the per-bar
        run skips main() and writes these fields straight from the bar.
    :param ohlcv_tuple: True when ``ohlcv_fields`` came from a tuple/list
        expression (write a tuple), False for a scalar expression.
    :param same_timeframe: True when the security TF equals the chart TF. The
        chart then signals every chart bar (``_get_confirmed_time`` returns the
        bar time verbatim), so gap compaction must NOT apply — a gappy
        cross-symbol feed forward-fills (``gaps_off``) through the chart's bars
        instead of compacting to real bars (which is only correct for a true
        HTF series).
    """
    # Safety net first: exit if the parent is hard-killed (see the watchdog docstring).
    _start_parent_death_watchdog()

    # Re-register import hooks (spawn mode starts a fresh Python process)
    from . import import_hook  # noqa

    # Open shared memory blocks
    sync_block = SyncBlock(all_sec_ids, create=False, name=sync_block_name)
    result_block = ResultBlock(sec_id, create=False, version=0)

    # Create protocol functions for security context
    signal_fn, write_fn, read_fn, wait_fn, cleanup, flush_fn = create_security_protocol(
        sec_id, sync_block, result_block, all_sec_ids, result_locks, is_ltf=is_ltf,
    )

    # Load OHLCV data
    from .ohlcv_file import OHLCVReader
    reader = OHLCVReader(ohlcv_path)
    reader.open()

    # Load syminfo from TOML (same directory, same base name)
    from .syminfo import SymInfo
    ohlcv_base = Path(ohlcv_path)
    toml_path = ohlcv_base.with_suffix('.toml')
    syminfo = SymInfo.load_toml(toml_path)

    # Gap-compacted bar view for HTF security contexts. ``OHLCVWriter``
    # forward-fills non-trading session/calendar gaps with ``volume == -1`` flat
    # bars, so a session-gapped intraday feed (e.g. a 720-minute HTF on Bursa palm
    # oil) becomes a continuous 24h grid, and a weekday-only D/W/M feed grows
    # synthetic weekend/holiday bars. The chart side drops these via
    # ``read_from(skip_gaps=True)`` (and ``bar_opens`` rides the real opens only);
    # the security child must too. Otherwise the child re-runs ``main()`` over the
    # phantom bars: bar-count history reads (``ta.highest``/``ta.lowest``/``[n]``)
    # span fewer real periods than TradingView, and stateful series like
    # ``ta.sma(close, 3)`` accumulate the flat fill bars (a Friday->Monday daily
    # gap would otherwise average two synthetic weekend closes). TradingView builds
    # its HTF series from real bars only. LTF keeps the fills (its intrabar windows
    # are intentionally continuous). Same-TF cross-symbol is excluded: the chart
    # signals every chart bar there, so compacting away the writer's fills would
    # emit ``na`` in a gap instead of forward-filling the prior real bar (TV
    # ``gaps_off``). ``None`` = no compaction (no gaps, LTF, or same-TF).
    real_index_map: list[int] | None = None
    if not is_ltf and not same_timeframe:
        from pynecore.lib.timeframe import in_seconds
        if in_seconds(syminfo.period) > 0:
            # Mirror ``read_from(skip_gaps=True)`` exactly: a gap is ``volume < 0``
            # (the writer's -1 fill). ``>= 0`` would also drop NaN-volume real bars
            # (no-volume instruments import as ``volume == na``), which the reader keeps.
            rim = [i for i in range(reader.size) if not (reader.read(i).volume < 0)]
            if len(rim) != reader.size:
                real_index_map = rim

    # Import the script module (triggers AST transformation)
    from .script_runner import import_script, _set_lib_properties, _set_lib_syminfo_properties
    from pynecore import lib
    from pynecore.lib import barstate
    from pynecore.core import instance_state, script as script_mod

    # Set syminfo BEFORE importing the script
    _set_lib_syminfo_properties(syminfo, lib)

    # Parse timezone
    from pynecore.lib import _parse_timezone
    tz = _parse_timezone(syminfo.timezone)

    # Mintick decimals for OHLC grid-snapping in ``_set_lib_properties``
    # (``None`` when the symbol has no real mintick -> falls back to the
    # significant-digit clean-up). Mirrors ``ScriptRunner._round_decimals``.
    from .syminfo import mintick_decimals
    _sec_mintick = getattr(syminfo, 'mintick', 0.0) or 0.0
    round_decimals = mintick_decimals(_sec_mintick) if _sec_mintick > 0 else None

    # A security child is a read-only replica of the user's script, not a place
    # to persist config. ``script.indicator``/``strategy`` re-saves the script's
    # ``.toml`` on import when ``pytest`` is absent (always true in a spawned
    # child), so several contexts would race to rewrite — and can corrupt — the
    # same user file. Disable the save for this process before importing.
    os.environ['PYNE_SAVE_SCRIPT_TOML'] = '0'

    # Import the script
    script_module = import_script(Path(script_path))

    # Inject security protocol into module globals
    inject_protocol(script_module, signal_fn, write_fn, read_fn, wait_fn,
                    active_security=sec_id)

    # Fresh per-process state: drop anything inherited (fork start method) and
    # build the root state vectors of the script's main and every registered
    # library main, mirroring ``ScriptRunner``'s bound-entries scheme so the
    # security child uses identical per-instance state keys as the chart.
    instance_state.reset()
    main_func = script_module.main
    bound_entries: dict[int, Callable[[], Any]] = {}
    seen_keys: set[str] = set()
    root_keys: list[str] = []
    for entry_func in [main_func] + [f for _title, f in script_mod._registered_libraries]:
        if id(entry_func) in bound_entries:
            continue
        entry_layout = getattr(entry_func, '__pyne_layout__', None)
        if entry_layout is None:
            bound_entries[id(entry_func)] = entry_func
            continue
        root_key = f'{entry_func.__module__}.{entry_func.__qualname__}'
        if root_key in seen_keys:
            root_key = f'{root_key}#{len(root_keys)}'
        seen_keys.add(root_key)
        root_keys.append(root_key)
        bound_entries[id(entry_func)] = partial(
            entry_func, instance_state.create_root(root_key, entry_layout))
    run_main = bound_entries[id(main_func)]
    lib_mains = [bound_entries[id(f)] for _title, f in script_mod._registered_libraries]

    # Set lib semaphore to suppress plot/strategy/alert side effects
    lib._lib_semaphore = True

    def _run_script_main():
        """Mirror the chart's ``_run_libs_and_main``: registered library mains
        initialize their exported-function proxies, so they must run before the
        script's ``main()`` on every bar — otherwise a script that calls an
        imported library function dies here with "Exported proxy has not been
        initialized". ``lib._lib_semaphore`` stays True for both (every side
        effect is suppressed in a security child)."""
        for run_lib_main in lib_mains:
            run_lib_main()
        run_main()

    # Plain-OHLCV fast path: the requested expression is only raw price series,
    # all of which ``_set_lib_properties`` already wrote onto ``lib`` for the
    # current bar (byte-identical to what main() would read). Replace the per-bar
    # main() re-run with a direct write of those fields — every loop path
    # (historical, live developing/closed, live LTF window) calls
    # ``_run_script_main`` and so picks this up through the closure cell.
    if ohlcv_fields:
        if ohlcv_tuple:
            _pt_fields = tuple(ohlcv_fields)

            def _run_script_main():
                write_fn(sec_id, tuple(getattr(lib, _f) for _f in _pt_fields))
        else:
            _pt_field = ohlcv_fields[0]

            def _run_script_main():
                write_fn(sec_id, getattr(lib, _pt_field))

    # Set up file-based logging if PYNE_SECURITY_LOG is set
    security_log_path = os.environ.get("PYNE_SECURITY_LOG")
    if security_log_path:
        context_label = f"{syminfo.ticker} {syminfo.period}"
        from pynecore.lib.log import setup_security_file_log
        setup_security_file_log(security_log_path, context_label)

    try:
        current_bar = 0
        total_bars = len(real_index_map) if real_index_map is not None else reader.size

        while True:
            # Wait for chart to signal this process
            advance_event.wait()
            advance_event.clear()

            # Check for shutdown
            if stop_event.is_set():
                break

            # Read target time from sync block
            target_time = sync_block.get_target_time(sec_id)

            # Advance: run bars until we reach or pass target_time
            bars_run = False
            while current_bar < total_bars:
                ohlcv = reader.read(
                    real_index_map[current_bar] if real_index_map is not None else current_bar)
                # Convert timestamp to milliseconds for comparison (a UTC->tz
                # datetime roundtrip preserves the instant, so the raw
                # timestamp is already the answer)
                bar_time_ms = int(ohlcv.timestamp * 1000)

                if bar_time_ms > target_time:
                    break

                # Set lib properties for this bar
                _set_lib_properties(ohlcv, current_bar, tz, lib, round_decimals)
                lib.last_bar_index = total_bars - 1

                # Set barstate
                barstate.isfirst = (current_bar == 0)
                barstate.islast = (current_bar == total_bars - 1)
                barstate.isconfirmed = True

                # Run the script
                _run_script_main()

                current_bar += 1
                bars_run = True

            if is_ltf:
                # LTF: flush accumulated array (empty list if no bars)
                flush_fn()
            elif not bars_run:
                # HTF: no bars for this time period (session gap)
                # Write na so chart reader doesn't deadlock
                with result_locks[sec_id]:
                    write_na(result_block, sync_block)

            # Signal: data is ready for reading
            data_ready_event.set()
            # Signal: this round is complete
            done_event.set()

    finally:
        cleanup()
        reader.close()
        result_block.close()
        sync_block.close()
        lib._lib_semaphore = False
