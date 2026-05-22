"""
Security process loop — multiprocessing.Process target for request.security() contexts.

Each security context runs as a separate OS process with its own Python interpreter,
lib module, and Series state. The process loads its own OHLCV data, re-imports the
script module (triggering AST transformation), and runs the main() function per bar.

Communication with the chart process uses shared memory + Events (see security.py).

Three flavors of advance are supported, distinguished by SyncBlock flags:

  (1) Historical closed bar — neither FLAG_IS_DEVELOPING nor FLAG_CLOSED_OVERRIDE
      is set. The subprocess reads the next OHLCV bar from its local file and
      runs ``main()`` with ``barstate.isconfirmed=True``. This is the path used
      in historical / backtest runs.

  (2) Live closed bar — FLAG_CLOSED_OVERRIDE is set, FLAG_IS_DEVELOPING is not.
      The chart supplies the closed HTF OHLCV directly via the SyncBlock
      (because the .ohlcv file is static and cannot be appended to at runtime).
      Used after the LIVE_TRANSITION sentinel by ``lookahead_on`` contexts when
      an HTF period closes.

  (3) Live developing bar — FLAG_IS_DEVELOPING is set. The chart supplies the
      in-progress HTF OHLCV via the SyncBlock. The subprocess re-runs ``main()``
      with ``barstate.isconfirmed=False`` against the same ``bar_index`` as the
      first developing tick of this period (Series.add → set, no new bar push).
      ``VarSnapshot`` rolls var globals back to the period baseline before each
      such re-execution; ``function_isolation.reset()`` clears per-call slots.
"""
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, UTC
from typing import TYPE_CHECKING

from .security_shm import (
    SyncBlock, ResultBlock, write_na,
    FLAG_IS_DEVELOPING, FLAG_CLOSED_OVERRIDE,
)
from .security import (
    create_security_protocol, inject_protocol,
)

if TYPE_CHECKING:
    from multiprocessing.synchronize import Lock as LockType


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
    """
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

    # Import the script module (triggers AST transformation)
    from .script_runner import import_script, _set_lib_properties, _set_lib_syminfo_properties
    from pynecore import lib
    from pynecore.lib import barstate
    from pynecore.core import function_isolation, script as script_mod
    from .var_snapshot import VarSnapshot
    from ..types.ohlcv import OHLCV

    # Set syminfo BEFORE importing the script
    _set_lib_syminfo_properties(syminfo, lib)

    # Parse timezone
    from pynecore.lib import _parse_timezone
    tz = _parse_timezone(syminfo.timezone)

    # Import the script
    script_module = import_script(Path(script_path))

    # Inject security protocol into module globals
    inject_protocol(script_module, signal_fn, write_fn, read_fn, wait_fn,
                    active_security=sec_id)

    # Reset function isolation for fresh state
    function_isolation.reset()

    # Set lib semaphore to suppress plot/strategy/alert side effects
    lib._lib_semaphore = True

    # Set up file-based logging if PYNE_SECURITY_LOG is set
    security_log_path = os.environ.get("PYNE_SECURITY_LOG")
    if security_log_path:
        context_label = f"{syminfo.ticker} {syminfo.period}"
        from pynecore.lib.log import setup_security_file_log
        setup_security_file_log(security_log_path, context_label)

    # VarSnapshot is created lazily after the script's persistent-var
    # globals are populated by the first ``main()`` run; before that the
    # snapshot would capture nothing useful.
    var_snapshot: VarSnapshot | None = None

    # Tracks the last developing HTF period start (ms) the subprocess
    # advanced into. Used to distinguish "new dev period" (allocate a new
    # bar_index) from "another tick within the same dev period" (re-run
    # against the saved baseline).
    last_dev_period_start: int | None = None

    # Set after the first live bar has been consumed. The historical loop
    # leaves ``current_bar`` already pointing at the *next* unprocessed
    # security index, so the very first live bar (developing or closed
    # override) must reuse that slot. Only subsequent transitions between
    # distinct HTF periods advance ``current_bar``.
    seen_live_bar: bool = False

    def _ensure_snapshot() -> VarSnapshot | None:
        nonlocal var_snapshot
        if var_snapshot is None:
            var_snapshot = VarSnapshot(
                script_module, script_mod._registered_libraries,
            )
        return var_snapshot if var_snapshot.has_vars else None

    try:
        current_bar = 0
        total_bars = reader.size

        while True:
            # Wait for chart to signal this process
            advance_event.wait()
            advance_event.clear()

            # Check for shutdown
            if stop_event.is_set():
                break

            target_time = sync_block.get_target_time(sec_id)
            flags = sync_block.get_flags(sec_id)
            is_developing = bool(flags & FLAG_IS_DEVELOPING)
            closed_override = bool(flags & FLAG_CLOSED_OVERRIDE)

            # ── (3) Live developing bar ──
            if is_developing:
                dev_open, dev_high, dev_low, dev_close, dev_volume, dev_time_ms = (
                    sync_block.get_developing_bar(sec_id)
                )
                dev_ts_sec = dev_time_ms // 1000
                ohlcv = OHLCV(
                    timestamp=dev_ts_sec,
                    open=dev_open, high=dev_high, low=dev_low,
                    close=dev_close, volume=dev_volume,
                )

                is_new_dev_period = (last_dev_period_start != dev_time_ms)
                if is_new_dev_period:
                    # Step the subprocess into a fresh bar slot — but reuse
                    # the slot already pointed at by ``current_bar`` for the
                    # very first live bar (the historical loop leaves it on
                    # the next unprocessed index).
                    if seen_live_bar:
                        current_bar += 1
                    seen_live_bar = True
                    last_dev_period_start = dev_time_ms
                else:
                    # Same dev period: restore var globals to the period
                    # baseline (saved either after the prior closed run or
                    # at the start of this dev period) and re-run.
                    snap = _ensure_snapshot()
                    if snap is not None:
                        snap.restore()
                    function_isolation.reset()

                _set_lib_properties(ohlcv, current_bar, tz, lib)
                lib.last_bar_index = current_bar

                barstate.isfirst = (current_bar == 0)
                barstate.islast = True
                barstate.isconfirmed = False
                barstate.ishistory = False
                barstate.isrealtime = True
                barstate.islastconfirmedhistory = False
                barstate.isnew = is_new_dev_period

                if is_new_dev_period:
                    snap = _ensure_snapshot()
                    if snap is not None:
                        snap.save()

                script_module.main()

                data_ready_event.set()
                done_event.set()
                continue

            # ── (2) Live closed bar (OHLCV from SyncBlock) ──
            if closed_override:
                dev_open, dev_high, dev_low, dev_close, dev_volume, dev_time_ms = (
                    sync_block.get_developing_bar(sec_id)
                )
                ts_sec = dev_time_ms // 1000
                ohlcv = OHLCV(
                    timestamp=ts_sec,
                    open=dev_open, high=dev_high, low=dev_low,
                    close=dev_close, volume=dev_volume,
                )

                # TV semantics: a developing HTF bar and its eventual close
                # share the same security-series index (Series.add() degrades
                # to set() because the bar_index hasn't moved). Only allocate
                # a NEW bar_index when this closed bar is not the closing of
                # an in-flight dev period (e.g. live closed bar arriving with
                # no prior dev — currently unused, but kept correct). The
                # very first live bar reuses the next-unprocessed index the
                # historical loop left in ``current_bar``.
                if last_dev_period_start == dev_time_ms:
                    # Same HTF bar — restore var baseline, then re-run as
                    # confirmed close. Series writes overwrite the dev value.
                    snap = _ensure_snapshot()
                    if snap is not None:
                        snap.restore()
                    function_isolation.reset()
                    is_new_closed_period = False
                else:
                    if seen_live_bar:
                        current_bar += 1
                    seen_live_bar = True
                    is_new_closed_period = True

                last_dev_period_start = None

                _set_lib_properties(ohlcv, current_bar, tz, lib)
                lib.last_bar_index = current_bar
                barstate.isfirst = (current_bar == 0)
                barstate.islast = False
                barstate.isconfirmed = True
                barstate.ishistory = False
                barstate.isrealtime = True
                barstate.islastconfirmedhistory = False
                barstate.isnew = is_new_closed_period

                script_module.main()

                # Snapshot AFTER the closed run completes — baseline for
                # subsequent developing iterations of the next HTF period.
                snap = _ensure_snapshot()
                if snap is not None:
                    snap.save()

                data_ready_event.set()
                done_event.set()
                continue

            # Historical path resets dev-period tracking.
            last_dev_period_start = None

            # ── (1) Historical closed bar from .ohlcv file ──
            bars_run = False
            while current_bar < total_bars:
                ohlcv_file_bar = reader.read(current_bar)
                bar_time_ms = int(
                    datetime.fromtimestamp(ohlcv_file_bar.timestamp, UTC)
                    .astimezone(tz).timestamp() * 1000
                )
                if bar_time_ms > target_time:
                    break

                _set_lib_properties(ohlcv_file_bar, current_bar, tz, lib)
                lib.last_bar_index = total_bars - 1
                barstate.isfirst = (current_bar == 0)
                barstate.islast = (current_bar == total_bars - 1)
                barstate.isconfirmed = True

                script_module.main()

                current_bar += 1
                bars_run = True

            if bars_run:
                snap = _ensure_snapshot()
                if snap is not None:
                    snap.save()

            if is_ltf:
                flush_fn()
            elif not bars_run:
                with result_locks[sec_id]:
                    write_na(result_block, sync_block)

            data_ready_event.set()
            done_event.set()

    finally:
        cleanup()
        reader.close()
        result_block.close()
        sync_block.close()
        lib._lib_semaphore = False
