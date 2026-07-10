"""
Security process loop — multiprocessing.Process target for request.security() contexts.

Each security context runs as a separate OS process with its own Python interpreter,
lib module, and Series state. The process loads its own OHLCV data, re-imports the
script module (triggering AST transformation), and runs the main() function per bar.

Communication with the chart process uses shared memory + Events (see security.py).

Two data-source modes (selected by the ``data_source`` arg type):

  Backtest / file mode (``str`` path):
      The subprocess opens a static ``.ohlcv`` file via :class:`OHLCVReader`
      and steps through it indexed by ``current_bar``. Used by historical
      runs and same-symbol HTF live runs (where the chart-side aggregator
      ships closed bars over the SyncBlock).

  Live cross-symbol mode (:class:`PluginSymbol`):
      The subprocess instantiates the named :class:`LiveProviderPlugin`,
      downloads warmup history in-memory via
      :func:`download_warmup_in_memory`, then streams closed bars from the
      provider's WS feed via :class:`LiveBarStreamer`. No ``.ohlcv`` file is
      involved.

Three flavors of advance are supported, distinguished by SyncBlock flags:

  (1) Historical closed bar — neither FLAG_IS_DEVELOPING nor FLAG_CLOSED_OVERRIDE
      is set. The subprocess pulls the next available bar from its source and
      runs ``main()`` with ``barstate.isconfirmed=True``. In file mode the
      source is the local ``.ohlcv`` file; in live cross-symbol mode it is the
      warmup buffer plus the WS stream.

  (2) Live closed bar — FLAG_CLOSED_OVERRIDE is set, FLAG_IS_DEVELOPING is not.
      The chart supplies the closed HTF OHLCV directly via the SyncBlock.
      Used by same-symbol HTF contexts where the chart-side
      :class:`HTFAggregator` builds the closed bar from the chart's own
      OHLCV stream.

  (3) Live developing bar — FLAG_IS_DEVELOPING is set. The chart supplies the
      in-progress HTF OHLCV via the SyncBlock. The subprocess re-runs ``main()``
      with ``barstate.isconfirmed=False`` against the same ``bar_index`` as the
      first developing tick of this period (Series.add → set, no new bar push).
      ``RootVarSnapshot`` rolls the root vectors' var slots back to the period
      baseline before each such re-execution; ``instance_state.reset()`` drops
      the function instances.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from functools import partial
from pathlib import Path
from datetime import datetime, timedelta, UTC
from time import monotonic, sleep
from typing import TYPE_CHECKING, Callable, cast

from .security_shm import (
    SyncBlock, ResultBlock, write_na,
    FLAG_IS_DEVELOPING, FLAG_CLOSED_OVERRIDE,
    is_ltf_window, is_ltf_chart_developing, is_ltf_live_phase,
)
from .security import (
    create_security_protocol, inject_protocol,
)
from .live_ltf_collector import LiveLtfCollector
from .plugin.live_provider import PluginSymbol

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from multiprocessing.synchronize import Lock as LockType
    from .live_runner import LiveBarStreamer
    from ..types.ohlcv import OHLCV

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


# Warmup window for cross-symbol live security contexts. 500 bars matches the
# chart-side default in ``pyne run`` (``--from -500``); enough to seed any
# reasonable indicator without burning REST budget on excess history.
_DEFAULT_WARMUP_BARS = 500


def _run_rate_source_loop(
        sec_id: str,
        data_source: 'PluginSymbol',
        sync_block: SyncBlock,
        result_block: ResultBlock,
        advance_event,
        data_ready_event,
        done_event,
        stop_event,
        result_locks: 'dict[str, LockType]',
) -> None:
    """Minimal close-only loop for auto-spawned rate-source contexts.

    Skips Pine import and ``main()`` execution; every chart-side advance
    pushes the next warmup/live ``close`` into the ``ResultBlock`` so the
    :class:`CurrencyRateProvider` can read it as the live FX rate.
    """
    from .plugin import load_plugin
    from .plugin.live_provider import LiveProviderPlugin
    from .live_runner import download_warmup_in_memory, LiveBarStreamer
    from .security_shm import write_result, write_na
    from pynecore.lib.timeframe import in_seconds
    from pynecore.lib import _parse_timezone

    provider_cls = load_plugin(data_source.provider_name)
    if not issubclass(provider_cls, LiveProviderPlugin):
        raise RuntimeError(
            f"Plugin '{data_source.provider_name}' is not a live provider; "
            f"cannot drive an auto-spawned rate-source context."
        )
    # ``ohlcv_dir`` is forwarded from the chart process so the child can
    # resolve per-exchange config overrides under ``<workdir>/config/plugins``
    # (e.g. the ``[binance]`` section of ``ccxt.toml``). No OHLCV file is
    # written either way — the writer is just attached for symmetry; the
    # rate-source loop reads bars directly into memory.
    provider = provider_cls(
        symbol=data_source.symbol,
        timeframe=data_source.timeframe,
        ohlcv_dir=data_source.ohlcv_dir,
        config=data_source.config,
    )
    if data_source.syminfo is not None:
        syminfo = data_source.syminfo
        provider.syminfo = syminfo
    else:
        syminfo = provider.update_symbol_info()

    tz = _parse_timezone(syminfo.timezone)

    tf_seconds = in_seconds(data_source.timeframe)
    time_to = datetime.now(UTC)
    if data_source.time_from is not None:
        time_from = data_source.time_from
        if time_from.tzinfo is None:
            time_from = time_from.replace(tzinfo=UTC)
    else:
        time_from = time_to - timedelta(seconds=tf_seconds * _DEFAULT_WARMUP_BARS)

    # Start the WS streamer *before* the REST warmup download so the
    # subscription is open while warmup is in flight. If we downloaded
    # first, any bar that closes between the REST round-trip finishing
    # and the WS handshake completing would be lost: the warmup window
    # ends at ``time_to`` (captured above), and the not-yet-subscribed
    # WS cannot replay closes from before its connect time.
    # ``last_historical_timestamp=None`` because warmup hasn't run yet;
    # we dedupe the post-warmup catch-up below against the warmup tail.
    live_streamer = LiveBarStreamer(
        provider, data_source.symbol, data_source.timeframe,
        syminfo=syminfo,
        last_historical_timestamp=None,
    )
    live_streamer.start()
    bar_buffer = download_warmup_in_memory(provider, time_from, time_to)
    last_warmup_ts = bar_buffer[-1].timestamp if bar_buffer else None
    # Drain anything the WS already queued and dedupe against warmup.
    # Streaming continues; later ``pop_new_closed_bars()`` calls only
    # need a forward-timestamp check, which the buffer-walk already does.
    # Equal-timestamp WS closes refine the warmup tail (providers whose
    # REST window includes the currently forming candle deliver the true
    # final close via WS at the same timestamp), so we overwrite rather
    # than drop — mirroring the chart's live path which lets equal
    # timestamps through for refinement.
    if last_warmup_ts is not None:
        for _bar in live_streamer.pop_new_closed_bars():
            if _bar.timestamp == last_warmup_ts and bar_buffer:
                bar_buffer[-1] = _bar
            elif _bar.timestamp > last_warmup_ts:
                bar_buffer.append(_bar)
    else:
        bar_buffer.extend(live_streamer.pop_new_closed_bars())

    try:
        current_bar = 0
        # Track the most recently published close so multiple advance
        # cycles within a single chart bar (e.g. ``calc_on_every_tick``
        # strategies, or indicators that resignal the rate sources every
        # script invocation) can re-publish the same value instead of
        # falling back to ``na`` once ``bar_buffer`` is fully drained up
        # to the current chart timestamp. ``CurrencyRateProvider`` requires
        # a real rate for ``request.security(..., currency=…)`` conversion;
        # writing NaN on the second call would stop the conversion mid-bar.
        last_close: float | None = None
        while True:
            advance_event.wait()
            advance_event.clear()
            if stop_event.is_set():
                break

            target_time = sync_block.get_target_time(sec_id)
            # Same dedupe applies forever: streamer was subscribed before
            # warmup, so the first batch may overlap warmup; subsequent
            # batches monotonically advance and the timestamp check is a
            # no-op once ``last_warmup_ts`` is past. Equal-timestamp closes
            # refine the partial warmup tail (see comment above the initial
            # drain) — overwrite rather than drop.
            for _bar in live_streamer.pop_new_closed_bars():
                if last_warmup_ts is None:
                    bar_buffer.append(_bar)
                elif _bar.timestamp == last_warmup_ts and bar_buffer:
                    bar_buffer[-1] = _bar
                elif _bar.timestamp > last_warmup_ts:
                    bar_buffer.append(_bar)

            bars_run = False
            while current_bar < len(bar_buffer):
                bar = bar_buffer[current_bar]
                # A UTC->tz datetime roundtrip preserves the instant, so the
                # raw timestamp is already the answer
                bar_time_ms = int(bar.timestamp * 1000)
                if bar_time_ms > target_time:
                    break
                last_close = float(bar.close)
                with result_locks[sec_id]:
                    write_result(result_block, sync_block, last_close)
                current_bar += 1
                bars_run = True

            if not bars_run:
                with result_locks[sec_id]:
                    if last_close is not None:
                        # Re-publish the last known close so a second
                        # advance within the same chart bar still sees a
                        # valid rate.
                        write_result(result_block, sync_block, last_close)
                    else:
                        # No bar has been consumed yet — the rate feed has
                        # nothing to publish; signal NaN so the consumer
                        # falls back to whatever default it has.
                        write_na(result_block, sync_block)

            data_ready_event.set()
            done_event.set()
    finally:
        live_streamer.stop()
        result_block.close()
        sync_block.close()


def _collect_in_period_intrabars(
        bar_buffer: 'list[OHLCV]',
        next_idx: int,
        period_end_exclusive: int,
        ltf_span_ms: int,
        chart_developing: bool,
        grace_seconds: float,
        wait_for_bars: 'Callable[[float], list[OHLCV]]',
        ingest: 'Callable[[OHLCV], None]',
        *,
        now: 'Callable[[], float] | None' = None,
) -> 'tuple[list[OHLCV], int]':
    """Collect a chart period's closed intrabars, blocking (bounded) for a lagging
    final intrabar on a confirmed chart bar.

    Advances ``next_idx`` through ``bar_buffer`` collecting every closed intrabar
    inside ``[period_start, period_end_exclusive)`` (future-period bars are left
    for the next round). On a developing chart bar the partial collection is
    returned at once. On a confirmed chart bar the published array must be the
    FULL closed period, so when the final in-period intrabar
    (``ts >= period_end_exclusive - ltf_span_ms``) is still missing the loop waits
    up to ``grace_seconds`` for the streamer to deliver it, ingesting late bars as
    they arrive — unless a later-period bar is already buffered, which means the
    final intrabar is genuinely gone (a provider gap) and waiting would only burn
    the grace. Warmup is a no-op here (its bars are already buffered).

    The clock (``now``) and the streamer wait/ingest are injected so the loop is
    deterministically testable without real time; ``now`` defaults to
    :func:`time.monotonic`.

    :param bar_buffer: The shared, append-only intrabar buffer; ``ingest`` mutates
        this same list, so bars delivered by ``wait_for_bars`` become visible to
        the next collect pass.
    :param next_idx: Cursor into ``bar_buffer`` of the first not-yet-consumed bar.
    :param period_end_exclusive: Exclusive end of the chart period, in ms.
    :param ltf_span_ms: One LTF intrabar span, in ms (defines the final-intrabar
        open as ``period_end_exclusive - ltf_span_ms``).
    :param chart_developing: True for a forming chart bar (return the partial
        collection without waiting), False for a confirmed one.
    :param grace_seconds: Maximum seconds to block for the lagging final intrabar.
    :param wait_for_bars: ``(timeout) -> list[OHLCV]`` blocking wait on the
        streamer, up to ``timeout`` seconds.
    :param ingest: ``(bar) -> None`` appends a streamer bar into ``bar_buffer``.
    :param now: ``() -> float`` monotonic clock; defaults to ``time.monotonic``.
    :return: ``(collected, next_idx)`` — the in-period intrabars and the advanced
        cursor.
    """
    if now is None:
        now = monotonic
    collected: 'list[OHLCV]' = []
    final_open = period_end_exclusive - ltf_span_ms
    deadline = now() + grace_seconds
    while True:
        while next_idx < len(bar_buffer):
            _b = bar_buffer[next_idx]
            if int(_b.timestamp * 1000) >= period_end_exclusive:
                break
            collected.append(_b)
            next_idx += 1
        if chart_developing:
            break
        if collected and int(collected[-1].timestamp * 1000) >= final_open:
            break
        # A bar beyond this period is already buffered: the stream has advanced
        # past the period end, so the final in-period intrabar is genuinely
        # missing (a gap) — stop waiting and publish what we have rather than
        # burning the full grace.
        if next_idx < len(bar_buffer):
            logger.warning(
                "Live LTF period ending %d ms missing its final intrabar "
                "(stream advanced past it); publishing the gap as-is",
                period_end_exclusive,
            )
            break
        _remaining = deadline - now()
        if _remaining <= 0:
            logger.warning(
                "Live LTF period ending %d ms incomplete after %.0fs grace; "
                "publishing without the final intrabar",
                period_end_exclusive, grace_seconds,
            )
            break
        for _sb in wait_for_bars(_remaining):
            ingest(_sb)
    return collected, next_idx


def _heikinashi_step(prev_open: float | None, prev_close: float | None,
                     o: float, h: float, lo: float, c: float
                     ) -> tuple[float, float, float, float]:
    """One Heikin Ashi recurrence step over ordinary OHLC.

    ``prev_open``/``prev_close`` are the previous bar's Heikin Ashi open/close
    (``None`` on the seed bar, where the open falls back to ``(o + c) / 2``).

    :return: ``(ha_open, ha_high, ha_low, ha_close)``.
    """
    ha_close = (o + h + lo + c) / 4.0
    ha_open = (o + c) / 2.0 if (prev_open is None or prev_close is None) \
        else (prev_open + prev_close) / 2.0
    return ha_open, max(h, ha_open, ha_close), min(lo, ha_open, ha_close), ha_close


# noinspection PyProtectedMember
def security_process_main(
        sec_id: str,
        script_path: str,
        data_source: 'str | PluginSymbol',
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
        chart_type: 'str | None' = None,
        chart_timeframe: 'str | None' = None,
        plain_ltf: bool = False,
):
    assert result_locks is not None, "result_locks must be provided by script_runner"
    """
    Entry point for a security process (multiprocessing.Process target).

    Re-registers import hooks (needed for spawn mode on macOS/Windows),
    re-imports the script, and runs the bar loop.

    :param sec_id: This security context's unique ID
    :param script_path: Path to the script .py file
    :param data_source: Either a path to an ``.ohlcv`` file (backtest / file
                        mode) or a :class:`PluginSymbol` describing the live
                        provider, symbol, timeframe and pre-loaded config.
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
    :param chart_type: Synthetic chart type requested via ``ticker.heikinashi()``
        etc. (currently only ``"heikinashi"``). When set, the child applies the
        per-bar chart-type transform to every bar before the script reads it (so
        backtest and live both work) and flips the matching ``chart.*`` builtin.
    :param chart_timeframe: The chart's (main-series) timeframe. Stored in
        ``lib._main_timeframe`` so ``timeframe.main_period`` in this child reports
        the chart TF instead of the context's own period (the child has no
        ``lib._script`` to carry it).
    :param plain_ltf: Scalar ``request.security()`` with a timeframe finer than
        the chart's. The chart signals every bar with the bar's own period as
        the target; an empty round (no intrabar in the period — a feed gap)
        must keep the previous value (TradingView ``gaps_off`` forward-fill)
        instead of writing ``na``.
    """
    # Safety net first: exit if the parent is hard-killed (see the watchdog docstring).
    _start_parent_death_watchdog()

    # Re-register import hooks (spawn mode starts a fresh Python process)
    from . import import_hook  # noqa

    # Open shared memory blocks
    sync_block = SyncBlock(all_sec_ids, create=False, name=sync_block_name)
    result_block = ResultBlock(sec_id, create=False, version=0)

    # Rate-source-only contexts skip the Pine machinery entirely: they only
    # exist so the chart-side ``CurrencyRateProvider`` can read a fresh
    # ``close`` from the result block. Dispatch before any script import.
    if isinstance(data_source, PluginSymbol) and data_source.is_rate_source:
        _run_rate_source_loop(
            sec_id, data_source, sync_block, result_block,
            advance_event, data_ready_event, done_event, stop_event,
            result_locks,
        )
        return

    # Create protocol functions for security context
    (signal_fn, write_fn, read_fn, wait_fn, cleanup, flush_fn,
     ltf_take_value, ltf_publish, ltf_buffer_len) = create_security_protocol(
        sec_id, sync_block, result_block, all_sec_ids, result_locks, is_ltf=is_ltf,
    )

    # ── Source dispatch: file (backtest) or live provider (cross-symbol live) ──
    from .syminfo import SymInfo
    from ..types.ohlcv import OHLCV
    live_provider = None
    live_streamer: 'LiveBarStreamer | None' = None
    bar_buffer: list[OHLCV] = []
    # Watermark used by the live path to dedupe streamer bars against the
    # warmup window — the streamer is subscribed *before* the REST warmup
    # download, so the first batch from ``pop_new_closed_bars()`` may
    # overlap warmup. ``None`` in file mode or when warmup yielded zero bars.
    last_warmup_ts: int | None = None
    # Grace window (seconds) the closed-only loop blocks on the streamer
    # queue waiting for a bar at ``target_time`` that has not yet been
    # published by the upstream WS feed. Cross-exchange symbol closes can
    # land a few seconds after the chart symbol's close, and the chart
    # process already advanced ``last_confirmed`` for this period — without
    # blocking here the loop would emit ``na`` and never get re-signaled.
    # Set when the live cross-symbol path is initialised below.
    live_bar_grace_seconds: float = 0.0
    # Warmup horizon (ms): the instant the REST warmup ended. The live LTF-window
    # path uses it to drop a still-forming warmup tail bar. Set in the live branch.
    warmup_horizon_ms: int = 0

    if isinstance(data_source, PluginSymbol):
        # Live cross-symbol path: own provider + warmup download + WS stream.
        from .plugin import load_plugin
        from .plugin.live_provider import LiveProviderPlugin
        from .live_runner import download_warmup_in_memory, LiveBarStreamer
        from pynecore.lib.timeframe import in_seconds

        provider_cls = load_plugin(data_source.provider_name)
        if not issubclass(provider_cls, LiveProviderPlugin):
            raise RuntimeError(
                f"Plugin '{data_source.provider_name}' is not a live provider; "
                f"cannot drive cross-symbol live request.security."
            )
        # ``ohlcv_dir`` is forwarded from the chart process so the child can
        # resolve per-exchange config overrides under
        # ``<workdir>/config/plugins`` (e.g. the ``[binance]`` section of
        # ``ccxt.toml``). The OHLCV writer is attached but no file is
        # written — warmup goes through ``download_warmup_in_memory``.
        live_provider = provider_cls(
            symbol=data_source.symbol,
            timeframe=data_source.timeframe,
            ohlcv_dir=data_source.ohlcv_dir,
            config=data_source.config,
        )
        # The chart may have pre-fetched syminfo to keep both sides in sync
        # with a single REST round-trip; fall back to a local call if not.
        if data_source.syminfo is not None:
            syminfo = data_source.syminfo
            live_provider.syminfo = syminfo
        else:
            syminfo = live_provider.update_symbol_info()

        tf_seconds = in_seconds(data_source.timeframe)
        # Mirror ``live_runner.bar_grace`` — 15..30s tracks how long an
        # exchange typically lags between bar close and WS publish, scaled
        # to the bar period (sub-15s windows would race against jitter).
        live_bar_grace_seconds = max(15.0, min(tf_seconds * 0.5, 30.0))
        time_to = datetime.now(UTC)
        warmup_horizon_ms = int(time_to.timestamp() * 1000)
        if data_source.time_from is not None:
            time_from = data_source.time_from
            if time_from.tzinfo is None:
                time_from = time_from.replace(tzinfo=UTC)
        else:
            time_from = time_to - timedelta(seconds=tf_seconds * _DEFAULT_WARMUP_BARS)

        # Start the WS streamer *before* the REST warmup so the subscription
        # is open while warmup is in flight. If we downloaded first, any bar
        # closing in the (REST-done → WS-connected) window would be lost.
        # ``last_historical_timestamp=None`` because warmup hasn't run yet;
        # the post-warmup catch-up below dedupes against the warmup tail.
        live_streamer = LiveBarStreamer(
            live_provider, data_source.symbol, data_source.timeframe,
            syminfo=syminfo,
            last_historical_timestamp=None,
        )
        live_streamer.start()
        bar_buffer = download_warmup_in_memory(live_provider, time_from, time_to)
        last_warmup_ts = bar_buffer[-1].timestamp if bar_buffer else None
        # Equal-timestamp WS closes refine the warmup tail — providers whose
        # REST window includes the currently forming candle deliver the true
        # final close via WS at the same timestamp, so overwrite rather than
        # drop. Mirrors the chart's live path (``live_ohlcv_generator``) which
        # lets equal timestamps through for refinement. A tail refined this way
        # is authoritative *and* already consumed from the streamer queue, so the
        # LTF pop loop below must keep it (the stream will not redeliver it).
        ws_refined_tail_ts: float | None = None
        if last_warmup_ts is not None:
            for _bar in live_streamer.pop_new_closed_bars():
                if _bar.timestamp == last_warmup_ts and bar_buffer:
                    bar_buffer[-1] = _bar
                    ws_refined_tail_ts = _bar.timestamp
                elif _bar.timestamp > last_warmup_ts:
                    bar_buffer.append(_bar)
        else:
            bar_buffer.extend(live_streamer.pop_new_closed_bars())
        reader = None
        ohlcv_path = None
    else:
        # File mode (backtest, same-symbol live with HTF aggregator).
        ohlcv_path = data_source
        from .ohlcv_file import OHLCVReader
        reader = OHLCVReader(ohlcv_path)
        reader.open()

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
    # are intentionally continuous). Same-TF cross-symbol compacts too: a
    # session-bounded feed on a 24/7 chart (TVC:US10Y on BINANCE:BTCUSDT) would
    # otherwise run stateful indicators (``ta.ema``) over thousands of flat
    # weekend/overnight fills, dragging their state toward the last close while
    # TradingView's are frozen between real bars. The chart side pairs this with
    # the ``bar_opens`` clamp in ``_get_confirmed_time``: in a gap nothing new is
    # confirmed, so the prior real value forward-fills (TV ``gaps_off``) instead
    # of the child advancing into an empty window and writing ``na``.
    # ``None`` = no compaction (no gaps, or LTF).
    real_index_map: list[int] | None = None
    if reader is not None and not is_ltf:
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
    from ..types.ohlcv import OHLCV

    # Set syminfo BEFORE importing the script
    _set_lib_syminfo_properties(syminfo)

    # ``timeframe.main_period`` must report the chart TF, not this context's own
    # period — the child has no ``lib._script`` to carry it, so propagate it here.
    if chart_timeframe is not None:
        lib._main_timeframe = chart_timeframe

    # Chart-type context (``ticker.heikinashi()``): this child evaluates the
    # script on Heikin Ashi candles, so the chart-type builtins must reflect the
    # security context, not the parent's standard chart. The ``chart`` module is
    # this process's own replica, so the parent's flags stay untouched.
    if chart_type == 'heikinashi':
        from pynecore.lib import chart
        chart.is_heikinashi = True
        chart.is_standard = False

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

    # Inject security protocol into module globals. Imported library modules can
    # contain request.security() calls too, so their transformed code references
    # __active_security__ / __sec_read__ / … — every module whose main runs in
    # this child needs the protocol, exactly like the chart side (see
    # ScriptRunner's sec_modules). Injecting only the main module leaves a
    # library main raising NameError on __active_security__.
    sec_modules: list = [script_module]
    for _reg_title, _reg_main in script_mod._registered_libraries:
        _reg_mod = sys.modules.get(getattr(_reg_main, '__module__', ''))
        if _reg_mod is not None and _reg_mod is not script_module:
            sec_modules.append(_reg_mod)
    for _sec_mod in sec_modules:
        inject_protocol(_sec_mod, signal_fn, write_fn, read_fn, wait_fn,
                        active_security=sec_id)

    # Fresh per-process state: drop anything inherited (fork start method) and
    # create the root state vectors of the entry points this process drives —
    # the script's main and the registered library mains (same binding rules
    # as the runner: state-carrying entries get a root vector as hidden first
    # argument, stateless ones are called as-is)
    instance_state.reset()
    main_func = script_module.main
    root_keys: list[str] = []
    seen_keys: set[str] = set()
    bound_entries: dict[int, Callable] = {}
    # noinspection PyProtectedMember
    for entry_func in [main_func] + [f for _title, f in script_mod._registered_libraries]:
        if id(entry_func) in bound_entries:
            continue
        entry_layout = getattr(entry_func, '__pyne_layout__', None)
        if entry_layout is None:
            bound_entries[id(entry_func)] = entry_func
            continue
        entry_root_key = f'{entry_func.__module__}.{entry_func.__qualname__}'
        if entry_root_key in seen_keys:
            entry_root_key = f'{entry_root_key}#{len(root_keys)}'
        seen_keys.add(entry_root_key)
        root_keys.append(entry_root_key)
        bound_entries[id(entry_func)] = partial(
            entry_func, instance_state.create_root(entry_root_key, entry_layout))
    run_main = bound_entries[id(main_func)]

    # Set lib semaphore to suppress plot/strategy/alert side effects
    lib._lib_semaphore = True

    # noinspection PyProtectedMember
    def _run_script_main():
        """Mirror the chart's ``_run_libs_and_main``: registered library mains
        initialize their exported-function proxies, so they must run before the
        script's ``main()`` on every bar — otherwise a script that calls an
        imported library function dies here with "Exported proxy has not been
        initialized". ``lib._lib_semaphore`` stays True for both (every side
        effect is suppressed in a security child)."""
        for _title, _lib_main in script_mod._registered_libraries:
            bound_entries.get(id(_lib_main), _lib_main)()
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

    # Chart-type (Heikin Ashi) per-bar transform. The two carried HA values ride
    # a synthetic root vector whose var slots are captured/rolled back by the
    # existing ``RootVarSnapshot`` (create_root + root_keys.append) — so a
    # developing bar recomputes from the fixed prior-close baseline with no
    # bespoke logic. ``_ha_apply`` reads the baseline and returns the HA bar;
    # ``_ha_commit`` writes the new HA open/close AFTER ``_run_script_main()`` in
    # each branch (before any ``snap.save()``), so ``save()`` captures the
    # previous committed values and ``restore()`` rolls a re-tick back to them.
    # Registering the HA root also makes ``RootVarSnapshot.has_vars`` True for an
    # otherwise var-less plain-OHLCV context, activating the snapshot exactly
    # where the recurrence needs it.
    if chart_type == 'heikinashi':
        _ha_key = f'__heikinashi__{sec_id}'
        _ha_root = instance_state.create_root(_ha_key, {
            'init': (None, None), 'series': (), 'varip': (), 'children': (),
            'names': ('prevHaOpen', 'prevHaClose'),
        })
        root_keys.append(_ha_key)
        _ha_pending: 'list[float | None]' = [None, None]

        def _ha_apply(_b: OHLCV) -> OHLCV:
            if _b.volume < 0:
                # Gap-fill bar: forward-fill the last HA close flat and do NOT
                # advance the recurrence (mirrors the removed feed transform's
                # ``volume < 0`` skip + the writer's gap re-fill).
                _fill = _ha_root[1] if _ha_root[1] is not None else _b.close
                _ha_pending[0], _ha_pending[1] = _ha_root[0], _ha_root[1]
                return _b._replace(open=_fill, high=_fill, low=_fill, close=_fill)
            _ho, _hh, _hl, _hc = _heikinashi_step(
                _ha_root[0], _ha_root[1], _b.open, _b.high, _b.low, _b.close)
            _ha_pending[0], _ha_pending[1] = _ho, _hc
            return _b._replace(open=_ho, high=_hh, low=_hl, close=_hc)

        def _ha_commit() -> None:
            _ha_root[0], _ha_root[1] = _ha_pending[0], _ha_pending[1]
    else:
        def _ha_apply(_b: OHLCV) -> OHLCV:
            return _b

        def _ha_commit() -> None:
            pass

    # Set up file-based logging if PYNE_SECURITY_LOG is set
    security_log_path = os.environ.get("PYNE_SECURITY_LOG")
    if security_log_path:
        context_label = f"{syminfo.ticker} {syminfo.period}"
        from pynecore.lib.log import setup_security_file_log
        setup_security_file_log(security_log_path, context_label)

    # Snapshot of this process's root vectors' var slots — the rollback
    # baseline for live re-executions of the same HTF period.
    var_snapshot: instance_state.RootVarSnapshot | None = None

    # Companion series-slot snapshot, used ONLY by the live LTF-window path: a
    # reordered feed can make the collector replay an earlier LTF ``bar_index``
    # after a later one ran, which (unlike a same-index re-tick) would append to
    # the root price/history series instead of overwriting. ``RootVarSnapshot``
    # excludes series slots, so they get their own rollback. Captured only in the
    # live phase (the warmup path is forward-only and never restores), seeded at
    # the warmup->live transition so the first replay has a baseline.
    series_snapshot: instance_state.RootSeriesSnapshot | None = None

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

    def _ensure_snapshot() -> instance_state.RootVarSnapshot | None:
        nonlocal var_snapshot
        if var_snapshot is None:
            var_snapshot = instance_state.RootVarSnapshot(root_keys)
        return var_snapshot if var_snapshot.has_vars else None

    def _ensure_series_snapshot() -> instance_state.RootSeriesSnapshot | None:
        nonlocal series_snapshot
        if series_snapshot is None:
            series_snapshot = instance_state.RootSeriesSnapshot(root_keys)
        return series_snapshot if series_snapshot.has_series else None

    # Append a streamer bar to ``bar_buffer``, deduped against ``last_warmup_ts``:
    # the WS was subscribed before the REST warmup ran, so the initial batch may
    # overlap the warmup tail (replace it) and anything older is a duplicate.
    def _ingest_streamer_bar(_bar: 'OHLCV') -> None:
        if last_warmup_ts is None:
            bar_buffer.append(_bar)
        elif _bar.timestamp == last_warmup_ts and bar_buffer:
            bar_buffer[-1] = _bar
        elif _bar.timestamp > last_warmup_ts:
            bar_buffer.append(_bar)

    # Polymorphic bar source: file-backed reader (random access on a static
    # ``.ohlcv``) or in-memory buffer fed by a live WS streamer (append-only,
    # grows over time as new closed bars arrive).
    def _read_bar(idx: int) -> 'OHLCV | None':
        if reader is not None:
            if real_index_map is not None:
                return reader.read(real_index_map[idx]) if idx < len(real_index_map) else None
            if idx < reader.size:
                return reader.read(idx)
            return None
        if live_streamer is not None:
            for _sb in live_streamer.pop_new_closed_bars():
                _ingest_streamer_bar(_sb)
        return bar_buffer[idx] if idx < len(bar_buffer) else None

    def _current_total() -> int:
        if reader is not None:
            return len(real_index_map) if real_index_map is not None else reader.size
        return len(bar_buffer)

    # ── Live LTF-window machinery (request.security_lower_tf on a streaming
    # source) ──────────────────────────────────────────────────────────────
    # A streaming LTF context accumulates each chart period's intrabars into a
    # window (closed intrabars plus the developing intrabar as the live last
    # element) via :class:`LiveLtfCollector`. Warmup replay (confirmed chart
    # bars) and live both flow through this one path; the collector owns the
    # monotonic LTF ``bar_index`` and the snapshot discipline, while ``_ltf_round``
    # supplies bars from this process's own streamer and publishes the window.
    ltf_next_idx = 0
    ltf_span_ms = 0
    ltf_phase_live = False
    ltf_series_base_armed = False
    _ltf_round: 'Callable[[int], None] | None' = None
    if (is_ltf and live_streamer is not None
            and isinstance(data_source, PluginSymbol)):
        from pynecore.lib.timeframe import in_seconds
        assert ltf_take_value is not None and ltf_publish is not None
        _streamer = cast('LiveBarStreamer', live_streamer)
        _take_value = ltf_take_value
        _publish = ltf_publish
        ltf_span_ms = int(in_seconds(data_source.timeframe) * 1000)

        # The REST warmup may end on a still-forming tail bar. On the window
        # path the developing intrabar must come ONLY from the live stream's
        # developing slot (re-run per tick), never from the closed buffer — a
        # forming tail left here would be consumed as a closed intrabar AND
        # surface again via ``peek_developing_bar``. Drop any such tail; the live
        # stream delivers it authoritatively (developing, then closed). Time
        # eligibility (period not ended by the warmup horizon) is the primary,
        # provider-agnostic test; the ``is_closed`` flag catches providers that
        # stamp it explicitly. WS closes consumed during the catch-up drain are
        # exempt: those bars were taken off the streamer queue and will not be
        # redelivered, so popping them would permanently lose intrabars that
        # closed during warmup. That covers both the equal-timestamp refinement
        # of the REST tail (``ws_refined_tail_ts``) and any later closes appended
        # past the REST tail (``timestamp > rest_warmup_tail_ts``) whose period
        # end can fall after ``warmup_horizon_ms`` — the horizon was sampled
        # before the warmup download, so a bar that closed mid-download would
        # otherwise look future-dated and be dropped despite being authoritative.
        rest_warmup_tail_ts = last_warmup_ts
        while bar_buffer and bar_buffer[-1].timestamp != ws_refined_tail_ts and not (
                rest_warmup_tail_ts is not None
                and bar_buffer[-1].timestamp > rest_warmup_tail_ts) and (
                not bar_buffer[-1].is_closed
                or int(bar_buffer[-1].timestamp * 1000) + ltf_span_ms > warmup_horizon_ms):
            bar_buffer.pop()
        last_warmup_ts = bar_buffer[-1].timestamp if bar_buffer else None

        def _run_ltf_intrabar(intrabar, bar_index, confirmed, is_new, islast):
            _set_lib_properties(intrabar, bar_index, tz, lib, round_decimals)
            lib.last_bar_index = bar_index
            barstate.isfirst = (bar_index == 0)
            barstate.islast = islast
            barstate.isconfirmed = confirmed
            barstate.ishistory = not ltf_phase_live
            barstate.isrealtime = ltf_phase_live
            barstate.islastconfirmedhistory = False
            barstate.isnew = is_new
            _run_script_main()
            return _take_value()

        def _save_ltf_baseline():
            _snap = _ensure_snapshot()
            if _snap is not None:
                _snap.save()
            # Series rollback is needed only in the live phase, where a reordered
            # feed can trigger a backward replay; the warmup path is forward-only
            # and never restores, so copying buffers there would be wasted work.
            if ltf_phase_live:
                _ss = _ensure_series_snapshot()
                if _ss is not None:
                    _ss.save()

        def _restore_ltf_baseline():
            _snap = _ensure_snapshot()
            if _snap is not None:
                _snap.restore()
            if series_snapshot is not None and series_snapshot.saved:
                series_snapshot.restore()
            instance_state.reset()

        _ltf_collector = LiveLtfCollector(
            _run_ltf_intrabar, _save_ltf_baseline, _restore_ltf_baseline,
        )

        def _ltf_round(slot_flags: int) -> None:
            nonlocal ltf_next_idx, ltf_phase_live, ltf_series_base_armed
            period_start = sync_block.get_target_time(sec_id)
            period_end_exclusive = sync_block.get_ltf_period_end(sec_id)
            chart_developing = is_ltf_chart_developing(slot_flags)
            chart_confirmed = not chart_developing
            ltf_phase_live = is_ltf_live_phase(slot_flags)

            # Seed the series baseline once, on the warmup->live transition: the
            # warmup ran forward-only so the series now hold exactly the confirmed
            # prefix, and the first live replay's restore needs that baseline even
            # if it precedes the first live confirmed save.
            if ltf_phase_live and not ltf_series_base_armed:
                _ss = _ensure_series_snapshot()
                if _ss is not None:
                    _ss.save()
                ltf_series_base_armed = True

            for _sb in _streamer.pop_new_closed_bars():
                _ingest_streamer_bar(_sb)

            # Collect not-yet-consumed closed intrabars up to the period end
            # (future-period bars are deferred). On a confirmed chart bar the
            # published array must be the FULL closed period, so wait (bounded)
            # for the final in-period intrabar if the streamer lags — a no-op
            # during warmup, where the REST bars are already buffered.
            collected, ltf_next_idx = _collect_in_period_intrabars(
                bar_buffer, ltf_next_idx, period_end_exclusive, ltf_span_ms,
                chart_developing, live_bar_grace_seconds,
                _streamer.wait_for_bars, _ingest_streamer_bar,
            )

            developing_bar: 'OHLCV | None' = None
            if chart_developing:
                _dev = _streamer.peek_developing_bar()
                # Only a forming bar strictly after the last collected closed
                # intrabar is a live developing tail; a stale tick at/under it
                # (e.g. a late forming update for an already-closed intrabar)
                # would otherwise allocate a spurious new bar_index.
                _last_collected_ms = (
                    int(collected[-1].timestamp * 1000) if collected else period_start - 1
                )
                if _dev is not None:
                    _dev_ms = int(_dev.timestamp * 1000)
                    if _last_collected_ms < _dev_ms < period_end_exclusive:
                        developing_bar = _dev
                    else:
                        logger.debug(
                            "LTF developing tick ts=%d ms outside (%d, %d) — skipped",
                            _dev_ms, _last_collected_ms, period_end_exclusive,
                        )

            window_values = _ltf_collector.process_round(
                period_start, period_end_exclusive, collected,
                developing_bar, chart_confirmed=chart_confirmed,
            )
            _publish(window_values)

    # Fixed ``last_bar_time`` anchor for file-backed (historical) runs: Pine
    # fixes it to the security series' final bar, known up front from the
    # static file. Scan back over the writer's gap-fill tail (``volume == -1``
    # records; ``not (volume < 0)`` keeps NaN-volume real bars) — a no-op when
    # ``real_index_map`` already compacted the gaps away. Streamer-fed (live)
    # contexts keep the per-bar value from ``_set_lib_properties``: the
    # realtime bar IS the last bar there.
    file_last_bar_time_ms = 0
    if reader is not None:
        for _pos in range(_current_total() - 1, -1, -1):
            _tail_bar = _read_bar(_pos)
            if _tail_bar is not None and not (_tail_bar.volume < 0):
                file_last_bar_time_ms = int(_tail_bar.timestamp * 1000)
                break

    try:
        current_bar = 0

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

            # ── (4) Live LTF window round ──
            # Checked first: a streaming request.security_lower_tf round carries
            # its own flag and never the HTF developing/closed-override flags.
            if _ltf_round is not None and is_ltf_window(flags):
                _ltf_round(flags)
                data_ready_event.set()
                done_event.set()
                continue

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
                    instance_state.reset()

                _set_lib_properties(_ha_apply(ohlcv), current_bar, tz, lib, round_decimals)
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

                _run_script_main()
                # Commit the developing HA open/close AFTER the run and the
                # new-period ``save()`` above; a same-period re-tick's
                # ``snap.restore()`` rolls these back to the period baseline.
                _ha_commit()

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
                    instance_state.reset()
                    is_new_closed_period = False
                else:
                    if seen_live_bar:
                        current_bar += 1
                    seen_live_bar = True
                    is_new_closed_period = True

                last_dev_period_start = None

                _set_lib_properties(_ha_apply(ohlcv), current_bar, tz, lib, round_decimals)
                lib.last_bar_index = current_bar
                barstate.isfirst = (current_bar == 0)
                barstate.islast = False
                barstate.isconfirmed = True
                barstate.ishistory = False
                barstate.isrealtime = True
                barstate.islastconfirmedhistory = False
                barstate.isnew = is_new_closed_period

                _run_script_main()
                # Commit the closed HA open/close BEFORE the baseline ``save()``
                # below, so the confirmed values become the next period's seed.
                _ha_commit()

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

            # ── (1) Historical / cross-symbol-live closed bar from local source ──
            # In cross-symbol live mode (``live_streamer is not None``), the
            # upstream WS feed may publish the requested symbol's closed bar
            # a few seconds after the chart symbol's close. The chart side
            # has already advanced ``last_confirmed = target_time`` and will
            # not re-signal this period, so we must block briefly until the
            # bar arrives instead of falling through to ``write_na``.
            bars_run = False
            # File-backed LTF: values written while replaying feed bars that
            # open BEFORE the chart bar's own period (cold start mid-feed,
            # chart session gaps) are expression-state warmup, not array
            # content — TradingView arrays hold only the bar's own period.
            ltf_period_start = (
                sync_block.get_ltf_period_start(sec_id) if is_ltf else 0
            )
            ltf_prefix_len = 0
            grace_deadline: float | None = None
            if live_streamer is not None:
                grace_deadline = monotonic() + live_bar_grace_seconds
            while True:
                ohlcv_file_bar = _read_bar(current_bar)
                if ohlcv_file_bar is None:
                    if grace_deadline is not None and live_streamer is not None:
                        remaining = grace_deadline - monotonic()
                        if remaining > 0:
                            # Block on the streamer queue for the new bar.
                            # ``_read_bar`` only consults the streamer when
                            # ``current_bar >= len(bar_buffer)``, so any
                            # bars returned here are appended to the buffer
                            # via ``_read_bar`` on the next iteration.
                            new_bars = live_streamer.wait_for_bars(remaining)
                            if new_bars:
                                for _bar in new_bars:
                                    if last_warmup_ts is None:
                                        bar_buffer.append(_bar)
                                    elif (_bar.timestamp == last_warmup_ts
                                            and bar_buffer):
                                        bar_buffer[-1] = _bar
                                    elif _bar.timestamp > last_warmup_ts:
                                        bar_buffer.append(_bar)
                                continue
                    break
                # A UTC->tz datetime roundtrip preserves the instant, so the
                # raw timestamp is already the answer
                bar_time_ms = int(ohlcv_file_bar.timestamp * 1000)
                if bar_time_ms > target_time:
                    break

                total_bars = _current_total()
                _set_lib_properties(_ha_apply(ohlcv_file_bar), current_bar, tz, lib, round_decimals)
                lib.last_bar_index = total_bars - 1
                if reader is not None:
                    lib.last_bar_time = file_last_bar_time_ms
                barstate.isfirst = (current_bar == 0)
                barstate.islast = (current_bar == total_bars - 1)
                barstate.isconfirmed = True
                # In PluginSymbol mode, bars beyond the warmup tail come from
                # the live WS streamer and represent realtime closes. Without
                # a phase flip here, ``barstate.ishistory`` would stay ``True``
                # for streamed closed bars — diverging from the same-symbol
                # live path (see lines 511-513 and 568-570 above) and breaking
                # script branches that key off ``barstate.isrealtime``.
                if (live_streamer is not None
                        and last_warmup_ts is not None
                        and ohlcv_file_bar.timestamp > last_warmup_ts):
                    barstate.ishistory = False
                    barstate.isrealtime = True

                _run_script_main()
                # Advance the HA recurrence for this confirmed bar; the batch
                # ``save()`` below captures the last bar's HA as the baseline.
                _ha_commit()

                if is_ltf and bar_time_ms < ltf_period_start:
                    ltf_prefix_len = ltf_buffer_len()  # noqa - non-None when is_ltf
                current_bar += 1
                bars_run = True

            if bars_run:
                snap = _ensure_snapshot()
                if snap is not None:
                    snap.save()

            if is_ltf:
                flush_fn(ltf_prefix_len)
            elif not bars_run and not plain_ltf:
                with result_locks[sec_id]:
                    write_na(result_block, sync_block)

            data_ready_event.set()
            done_event.set()

    finally:
        cleanup()
        if reader is not None:
            reader.close()
        if live_streamer is not None:
            live_streamer.stop()
        result_block.close()
        sync_block.close()
        lib._lib_semaphore = False
