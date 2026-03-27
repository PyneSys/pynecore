"""
Security process loop — multiprocessing.Process target for request.security() contexts.

Each security context runs as a separate OS process with its own Python interpreter,
lib module, and Series state. The process loads its own OHLCV data, re-imports the
script module (triggering AST transformation), and runs the main() function per bar.

Communication with the chart process uses shared memory + Events (see security.py).
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, UTC

from .security_shm import (
    SyncBlock, ResultBlock, write_na,
)
from .security import (
    create_security_protocol, inject_protocol,
)


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
):
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
    """
    # Re-register import hooks (spawn mode starts a fresh Python process)
    from . import import_hook  # noqa

    # Open shared memory blocks
    sync_block = SyncBlock(all_sec_ids, create=False, name=sync_block_name)
    result_block = ResultBlock(sec_id, create=False, version=0)

    # Create protocol functions for security context
    signal_fn, write_fn, read_fn, wait_fn, cleanup = create_security_protocol(
        sec_id, sync_block, result_block, all_sec_ids,
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
    from pynecore.core import function_isolation

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

            # Read target time from sync block
            target_time = sync_block.get_target_time(sec_id)

            # Advance: run bars until we reach or pass target_time
            bars_run = False
            while current_bar < total_bars:
                ohlcv = reader.read(current_bar)
                # Convert timestamp to milliseconds for comparison
                bar_time_ms = int(
                    datetime.fromtimestamp(ohlcv.timestamp, UTC)
                    .astimezone(tz).timestamp() * 1000
                )

                if bar_time_ms > target_time:
                    break

                # Set lib properties for this bar
                _set_lib_properties(ohlcv, current_bar, tz, lib)
                lib.last_bar_index = total_bars - 1

                # Set barstate
                barstate.isfirst = (current_bar == 0)
                barstate.islast = (current_bar == total_bars - 1)
                barstate.isconfirmed = True

                # Run the script
                script_module.main()

                current_bar += 1
                bars_run = True

            if not bars_run:
                # No bars for this time period (session gap)
                # Write na so chart reader doesn't deadlock
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
