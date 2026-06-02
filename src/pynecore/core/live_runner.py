"""
Async/sync bridge for live data streaming.

Runs a LiveProviderPlugin's async watch_ohlcv() in a background thread
and yields OHLCV objects to the synchronous ScriptRunner via queue.Queue.

Serial startup: ``live_ohlcv_generator()`` starts the background thread
**and blocks until ``provider.connect()`` succeeds** before returning.
This guarantees that by the time the caller starts consuming the warmup
(local OHLCV file), the WS subscription is already active — any bar that
closes while warmup is running goes into ``bar_queue`` and cannot be
lost. A parallel-start design is appealing on paper but in practice
``provider.connect()`` (REST session + activity-cursor recovery + WS
handshake + subscribe) is slow enough that warmup of a local file
finishes first; the resulting gap means the very next bar close on the
exchange happens with no listener, and that bar is gone forever.

Catch-up: when the consumer first pulls from the live iterator (right
after the file iterator has been exhausted), any closed bars already
sitting in the queue are drained as additional warmup (the script_runner
historical loop processes them with ``barstate.ishistory=True`` and the
strategy still suppressed). Intra-bar updates queued during warmup are
dropped — they would otherwise inflate ``bar_index`` against a still-open
bar that the historical loop is not equipped to dedup. Once the queue
empties, ``LIVE_TRANSITION`` is yielded inline; the script_runner flips
to live mode and every subsequent bar runs against an unsuppressed
strategy.

Dedup: the duplicate-filter for ``last_historical_timestamp`` uses
**strict** less-than for both closed bars and intra-bar updates. The
plugin contract is that ``download_ohlcv`` returns fully-closed bars
only (the Capital.com plugin, for instance, drops the still-forming
last bar in its REST response), so under normal operation the
``ts == last_historical`` case does not arise. The strict-less filter
is kept as defense-in-depth: if a provider violates the contract or a
race serves an in-progress bar from REST, the script_runner live loop
treats the same-timestamp first live update as a continuation of the
last warmup bar — it seeds ``last_bar_timestamp`` from
``last_warmup_timestamp``, so ``bar_index`` does not double-bump and
the bar simply gets one more execution with the refined OHLC values.
"""
import asyncio
import logging
import time
import threading
from collections.abc import Coroutine, Generator
from datetime import datetime
from queue import Queue, Empty, Full
from typing import Any
from zoneinfo import ZoneInfo

from pynecore.core.syminfo import SymInfo
from pynecore.types.ohlcv import OHLCV
from pynecore.core.plugin.live_provider import LiveProviderPlugin
from pynecore.core.script_runner import LIVE_TRANSITION
from pynecore.lib.log import broker_info, broker_warning
from pynecore.lib.session import _is_in_session, _is_point_in_session
from pynecore.lib.timeframe import in_seconds

__all__ = ['live_ohlcv_generator', 'download_warmup_in_memory', 'LiveBarStreamer']


class LiveBarStreamer:
    """Non-blocking, thread-safe closed-bar source for security subprocesses.

    Wraps :func:`live_ohlcv_generator` in a background thread that drains its
    output into an internal queue. The security subprocess polls
    :meth:`pop_new_closed_bars` once per chart advance — receiving zero or
    more freshly-closed bars without ever blocking the chart-driven flow.

    Intra-bar updates and the warmup→live transition sentinel are dropped:
    cross-symbol :func:`request.security` only exposes closed bars, and the
    subprocess does not switch modes mid-run.
    """

    def __init__(self, provider: LiveProviderPlugin, symbol: str, timeframe: str,
                 *, syminfo: SymInfo | None = None,
                 last_historical_timestamp: int | None = None):
        self._provider = provider
        self._symbol = symbol
        self._timeframe = timeframe
        self._syminfo = syminfo
        self._last_historical_timestamp = last_historical_timestamp
        self._queue: Queue[OHLCV] = Queue()
        self._stopped = threading.Event()
        self._gen: Generator[OHLCV, None, None] | None = None
        self._thread: threading.Thread | None = None
        # Captures an exception raised by the upstream generator so the
        # next ``pop_new_closed_bars()`` call can surface it to the
        # security subprocess. Without this, a dead WS would just stop
        # delivering bars and the security loop would silently emit ``na``
        # forever while the chart-side liveness check never notices.
        self._drain_error: BaseException | None = None

    def start(self) -> None:
        """Start the background drain thread."""
        if self._thread is not None:
            return
        self._gen = live_ohlcv_generator(
            provider=self._provider,
            symbol=self._symbol,
            timeframe=self._timeframe,
            syminfo=self._syminfo,
            last_historical_timestamp=self._last_historical_timestamp,
        )
        self._thread = threading.Thread(
            target=self._drain, daemon=True, name=f"sec-stream-{self._symbol}",
        )
        self._thread.start()

    def _drain(self) -> None:
        assert self._gen is not None
        try:
            for bar in self._gen:
                if self._stopped.is_set():
                    break
                if bar is LIVE_TRANSITION:
                    continue
                if not getattr(bar, 'is_closed', True):
                    continue
                self._queue.put(bar)
        except Exception as exc:  # noqa: BLE001
            logger.exception("LiveBarStreamer drain raised")
            if not self._stopped.is_set():
                self._drain_error = exc

    def pop_new_closed_bars(self) -> list[OHLCV]:
        """Drain all currently-available closed bars (non-blocking).

        Re-raises any exception captured by the upstream drain thread once
        the queue is empty, so the security subprocess can fail loudly and
        the chart-side liveness check (``proc.is_alive()`` polling) catches
        the dead process instead of silently turning every bar into ``na``.
        """
        out: list[OHLCV] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except Empty:
                break
        if not out and self._drain_error is not None:
            err = self._drain_error
            # Single-shot raise so a re-tried call (e.g. after restart)
            # would not keep raising forever; ``_drain_error`` stays set
            # only as a flag for ``stop()`` cleanup.
            self._drain_error = None
            raise err
        return out

    def wait_for_bars(self, timeout: float) -> list[OHLCV]:
        """Block up to ``timeout`` seconds for at least one closed bar.

        Used by the cross-symbol live HTF security loop when the chart
        process has signaled a confirmed period whose bar has not yet been
        published by the upstream WS feed (different exchange / network
        delay). Blocking briefly on the streamer queue avoids advancing the
        chart's ``last_confirmed`` watermark past the security's actual bar
        and emitting ``na`` for that period.

        Drains any further bars that have already accumulated after the
        first one arrives so a single call still returns the full batch.
        Re-raises a captured drain error using the same single-shot
        semantics as :meth:`pop_new_closed_bars`.
        """
        out: list[OHLCV] = []
        try:
            first = self._queue.get(timeout=max(0.0, timeout))
        except Empty:
            if self._drain_error is not None:
                err = self._drain_error
                self._drain_error = None
                raise err
            return out
        out.append(first)
        while True:
            try:
                out.append(self._queue.get_nowait())
            except Empty:
                break
        return out

    def stop(self) -> None:
        """Signal the drain thread to exit and close the upstream generator.

        The drain thread is typically parked inside ``next(self._gen)`` at
        this point. Calling ``self._gen.close()`` from a *different* thread
        while the generator is suspended on a ``yield`` is supported, but
        if the runtime considers the generator to be executing on the drain
        side (race with the moment a bar is being yielded), CPython raises
        ``ValueError: generator already executing``. The drain thread will
        eventually return on its own once ``self._stopped`` is observed, so
        swallow the race here and rely on the ``join`` below to finish
        cleanup.
        """
        self._stopped.set()
        if self._gen is not None:
            try:
                self._gen.close()
            except (RuntimeError, ValueError, GeneratorExit):
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def download_warmup_in_memory(
        provider: LiveProviderPlugin,
        time_from: datetime,
        time_to: datetime,
) -> list[OHLCV]:
    """
    Download historical OHLCV data into an in-memory list (no file written).

    Used by the security subprocess to fetch warmup bars without creating
    any ``.ohlcv`` file. Captures the records by temporarily redirecting
    :meth:`ProviderPlugin.save_ohlcv_data` to an internal list — the
    plugin's own ``download_ohlcv`` implementation is otherwise unchanged.

    :param provider: A live provider instance (``ohlcv_dir`` may be ``None``).
    :param time_from: Naive UTC start of the warmup window.
    :param time_to: Naive UTC end of the warmup window.
    :return: Fully-closed warmup bars in chronological order.
    """
    captured: list[OHLCV] = []

    original_save = provider.save_ohlcv_data

    def _capture(data):
        if isinstance(data, OHLCV):
            captured.append(data)
        else:
            captured.extend(data)

    provider.save_ohlcv_data = _capture  # type: ignore[method-assign]
    try:
        tf = time_from.replace(tzinfo=None) if time_from.tzinfo else time_from
        tt = time_to.replace(tzinfo=None) if time_to.tzinfo else time_to
        provider.download_ohlcv(tf, tt)
    finally:
        provider.save_ohlcv_data = original_save  # type: ignore[method-assign]

    return captured

logger = logging.getLogger(__name__)


class _Sentinel(BaseException):
    """Marker signaling end of the live stream."""


_SENTINEL = _Sentinel()

# Soft cap for intra-bar updates queued ahead of the consumer. Closed
# bars are never dropped (they go through the blocking ``put`` path),
# but intra-bar updates are advisory and must not accumulate without
# bound when the consumer falls behind — otherwise stale ticks would
# sit ahead of newer closed bars and delay live transition.
_INTRA_BAR_SOFT_CAP = 32

# Sleep cadence during known-closed windows. Long enough to keep the
# TimeoutError handler from spinning at ``effective_timeout`` (~50ms)
# while the boundary deadline is stale, short enough that the next
# session open is noticed within ~30s. Exposed at module scope so tests
# can shrink it without spending the full 30s per gated timeout.
_CLOSED_WINDOW_SLEEP_S = 30.0


def live_ohlcv_generator(
        provider: LiveProviderPlugin,
        symbol: str,
        timeframe: str,
        syminfo: SymInfo | None = None,
        *,
        last_historical_timestamp: int | None = None,
        shutdown_timeout: float = 120.0,
        event_loop: asyncio.AbstractEventLoop | None = None,
        engine_event_stream: Coroutine[Any, Any, Any] | None = None,
        raise_on_connect_failure: bool = False,
) -> Generator[OHLCV, None, None]:
    """
    Bridge async watch_ohlcv() to a sync Generator[OHLCV, None, None].

    Spawns a background thread running asyncio, collects OHLCV objects
    via queue.Queue, and yields them including intra-bar updates.

    The background thread is started **eagerly** at call time, not on
    first ``next()`` — so the WS subscription is open during warmup and
    no bar is lost in the gap between the REST historical download
    finishing and the consumer reaching the first live update.

    The first batch of bars yielded after the consumer starts pulling is
    the catch-up: closed bars that landed in the queue while the local
    warmup loop was running. ``LIVE_TRANSITION`` is yielded inline once
    the queue empties, so the script_runner can flip to live mode at the
    correct point regardless of how long warmup took.

    :param provider: A LiveProviderPlugin instance (already configured).
    :param symbol: Symbol in provider-specific format.
    :param timeframe: Timeframe in TradingView format.
    :param syminfo: Optional symbol metadata used to gate idle-bar
                    synthesis and reconnect attempts on the trading-
                    session calendar. ``None`` (the default) or an
                    empty ``opening_hours`` preserves the legacy 24/7
                    behaviour where idle synth and reconnect fire on
                    every timeout.
    :param last_historical_timestamp: Timestamp of the last historical bar to avoid duplicates.
    :param shutdown_timeout: Max seconds to wait for graceful shutdown. 0 = wait forever.
    :param event_loop: Optional externally-owned event loop. When supplied, the background
                       thread runs the async loop on it via ``run_until_complete`` instead
                       of ``asyncio.run``. Required for broker mode so that the Order Sync
                       Engine can submit coroutines to the same loop.
    :param engine_event_stream: Optional coroutine (typically
                                ``OrderSyncEngine.run_event_stream()``) to run as a
                                long-lived task alongside the OHLCV watcher. The engine
                                receives its :class:`OrderEvent` stream this way.
    :param raise_on_connect_failure: When True, a ``provider.connect()`` that
                                fails fast during warmup is re-raised here, from
                                the construction call, instead of being buffered
                                for the first bar pull. Broker mode sets this so
                                the real connect error surfaces before
                                ``start_broker()`` can mask it with a generic
                                "live connection not established" reconcile
                                failure. Data-only callers (and the security
                                ``LiveBarStreamer``) leave it False and keep the
                                surface-through-the-iterator behaviour.
    :return: Iterator yielding OHLCV objects (both closed and intra-bar) interleaved
             with a single ``LIVE_TRANSITION`` sentinel marking the warmup→live boundary.
    """
    # Unbounded on the closed-bar path: a bounded queue would block
    # ``bar_queue.put`` when the consumer (``script_runner`` live path)
    # falls behind during heavy fill processing. Since ``_async_loop``
    # runs as an asyncio task on the same event loop as ``_listen_loop``
    # (broker mode pumps both via ``run_coroutine_threadsafe`` onto the
    # main loop), a sync ``Queue.put`` parking on ``not_full`` would
    # stall the entire event loop — listener stops draining the WS,
    # quote ticks pile up in the kernel buffer, ``_tick_volume`` stops
    # advancing, and the watchdog ends up synthesising V=0 bars on a
    # market that is actually trading. Intra-bar updates are bounded
    # by a soft ``qsize`` cap at the put site (see
    # ``_INTRA_BAR_SOFT_CAP`` below) so advisory ticks cannot pile up
    # ahead of closed bars when the consumer lags. The closed-bar
    # ``broker_warning`` in the put path below flags any consumer lag.
    bar_queue: Queue[OHLCV | BaseException] = Queue()
    stop_event = threading.Event()
    # Signalled by ``_async_loop`` once ``provider.connect()`` has either
    # succeeded (WS subscribed, ready to receive bars) or failed (with
    # the exception already pushed into ``bar_queue``). Used to make
    # ``live_ohlcv_generator`` block until the WS is up before returning
    # — see module docstring for why warmup must follow connect.
    connected_event = threading.Event()
    connect_timeout_seconds = 30.0
    # Holds the exception from a fast-failing ``provider.connect()``. Appended
    # before ``connected_event`` is set (so the construction-site read below
    # synchronises on the event and observes it), letting the caller re-raise
    # the real cause instead of returning a generator whose buffered error
    # only surfaces on the first pull — too late, by which point
    # ``start_broker()`` has masked it. See ``raise_on_connect_failure``.
    connect_failure: list[BaseException] = []

    async def _graceful_shutdown():
        """Poll can_shutdown(), then disconnect. Respects shutdown_timeout."""
        logger.info("Graceful shutdown started, polling can_shutdown()...")

        if shutdown_timeout > 0:
            deadline = time.monotonic() + shutdown_timeout
        else:
            deadline = None

        while True:
            try:
                if await provider.can_shutdown():
                    logger.info("Provider ready to shut down")
                    break
            except Exception as e:
                logger.warning("can_shutdown() raised: %s", e)
                break

            if deadline is not None and time.monotonic() >= deadline:
                logger.warning("Shutdown timeout (%.0fs reached), forcing disconnect",
                               shutdown_timeout)
                break

            await asyncio.sleep(1.0)

        try:
            await provider.disconnect()
        except (OSError, RuntimeError):
            pass

    # Idle-bar synthesis: providers whose WS feed only emits ohlc.event when
    # a bar contained at least one tick (Capital.com is the documented
    # case; Bybit/Binance/IB exhibit the same on illiquid moments) leave
    # bar_index frozen across idle TF intervals. The REST history
    # endpoint, however, returns those zero-volume bars on the same
    # calendar boundaries — so on a future restart the historical replay
    # would step through bars the live run never saw, and strategy
    # decisions on the same minute diverge between live and replay.
    # The watchdog below synthesises one zero-volume CLOSED bar
    # (O=H=L=C=last close, V=0) per missed TF boundary so the live
    # stream matches what REST will later return. This lives in the
    # framework, not in each plugin, so providers stay free of bar-rhythm
    # bookkeeping — they only emit when their feed pushes a real bar.
    tf_seconds = max(1, int(in_seconds(timeframe)))
    # Grace past close before declaring a bar missed. The minimum is set
    # to 15s so plugins that recover dropped WS bars via REST have a
    # realistic window to fetch and inject the missing bar before this
    # synth fires — exchanges typically publish a closed bar to REST 5-10s
    # past close, and the plugin watchdog needs a few extra seconds to
    # detect, fetch and inject. The cap (30s) keeps longer TFs from
    # sitting on idle gaps too long.
    bar_grace = max(15.0, min(tf_seconds * 0.5, 30.0))

    # Resolve the symbol timezone once. ``syminfo.opening_hours`` times are
    # expressed in this zone; epoch timestamps must be converted before
    # session checks.
    _sym_tz: ZoneInfo | None = None
    if syminfo is not None and syminfo.opening_hours:
        try:
            _sym_tz = ZoneInfo(syminfo.timezone)
        except Exception:  # noqa: BLE001
            logger.warning("Unknown syminfo.timezone=%r; session-gate disabled",
                           syminfo.timezone)
    _has_calendar = (
        syminfo is not None
        and bool(syminfo.opening_hours)
        and _sym_tz is not None
    )

    def _market_open_at(epoch_ts: float) -> bool:
        """Slot-aware "is this bar slot in-session?" check.

        Returns True iff the candle ``[epoch_ts, epoch_ts+tf)`` overlaps
        any ``opening_hours`` interval (or unconditionally True for the
        24/7 fallback when the symbol has no calendar). Use this for
        bar-synth decisions and any other slot-aware logic — for the
        point-in-time "is the market open right now?" question use
        :func:`_market_open_now` instead, which does not extend the
        instant by one timeframe.
        """
        if not _has_calendar:
            return True
        assert syminfo is not None and _sym_tz is not None
        local_dt = datetime.fromtimestamp(epoch_ts, tz=_sym_tz)
        return _is_in_session(syminfo.opening_hours, local_dt, tf_seconds)

    def _market_open_now() -> bool:
        """Point-in-time "is the market open right now?" check.

        Does not extend wall-clock by one timeframe, so it does not
        report a session as open one timeframe before its real start.
        Used by the reconnect gate so a long timeframe (e.g. 1h, 1D)
        cannot burn ``max_reconnect_attempts`` during the final
        timeframe of a closed window.
        """
        if not _has_calendar:
            return True
        assert syminfo is not None and _sym_tz is not None
        local_dt = datetime.fromtimestamp(time.time(), tz=_sym_tz)
        return _is_point_in_session(syminfo.opening_hours, local_dt)

    async def _async_loop():
        # Declared before ``try`` so the ``finally`` branch can reference
        # it even when ``provider.connect()`` raises before assignment.
        engine_task: asyncio.Task | None = None
        # Last CLOSED bar seen (real or synthesised). The boundary
        # watchdog only arms after the first real bar so we never
        # fabricate state without a baseline.
        last_closed_bar: OHLCV | None = None
        # Tracks whether the last observed market state was open. Flips to
        # False on the first synth-skip in a closed period (emits a single
        # INFO log) and back to True when a real bar arrives.
        market_open_state = True
        try:
            broker_info("WS connect starting (warmup blocks until subscribed)")
            try:
                await provider.connect()
            except BaseException as connect_exc:
                # Record the real cause, then unblock the caller waiting on
                # ``connected_event``. The append happens-before ``set()``,
                # which the caller synchronises on via ``.wait()``, so the
                # construction-site re-raise (when ``raise_on_connect_failure``)
                # observes it. The outer ``except`` still pushes it onto
                # ``bar_queue`` so a post-warmup failure surfaces through the
                # iterator instead of hanging the wait for the full timeout.
                connect_failure.append(connect_exc)
                connected_event.set()
                raise
            watch_symbol = provider.normalize_symbol(symbol)
            broker_info("WS connected and subscribed: %s %s@%s",
                        type(provider).__name__, symbol, timeframe)
            connected_event.set()

            # Broker mode: attach the Order Sync Engine's event stream as
            # a background task so OrderEvents land in its queue without
            # blocking the OHLCV reader.
            if engine_event_stream is not None:
                engine_task = asyncio.create_task(engine_event_stream)

            reconnect_attempts = 0

            async def _handle_connection_error(
                    err: BaseException,
                    attempts: int,
            ) -> tuple[int, bool]:
                """Drive the closed-market wait / reconnect sequence.

                Returns ``(attempts, should_break)``. Used by the
                ``except Exception`` branch below and by the
                synth-skip session-gate when a dead WS is detected
                inside the ``except asyncio.TimeoutError`` handler —
                ``raise`` from one ``except`` handler does not enter
                its sibling handlers, so the original ``raise
                ConnectionError`` pattern would escape past
                ``except Exception`` and kill the live iterator.
                """
                nonlocal market_open_state
                # Session-gate: when the market is in a known-closed
                # window (e.g. FX weekend), do not burn reconnect
                # attempts on a connection error. We sleep ~30s and
                # re-enter the loop without incrementing
                # ``reconnect_attempts`` so ``max_reconnect_attempts``
                # is never exhausted across a long closed window.
                # Logs the connection error once on the closed→still-
                # closed transition for post-mortem visibility.
                # Uses the point-in-time helper (not the slot-aware
                # ``_market_open_at``) so a long timeframe cannot report
                # the market as already open one TF before the real
                # session start and exhaust reconnect attempts.
                if not _market_open_now():
                    if market_open_state:
                        broker_info(
                            "market closed: pausing reconnect attempts "
                            "until next session open "
                            "(last error: %s)",
                            err,
                        )
                        market_open_state = False
                    slept = 0.0
                    while slept < 30.0 and not stop_event.is_set():
                        await asyncio.sleep(min(1.0, 30.0 - slept))
                        slept += 1.0
                    return attempts, stop_event.is_set()
                attempts += 1
                if attempts > provider.max_reconnect_attempts:
                    logger.error(
                        "Max reconnect attempts reached (%d), stopping",
                        provider.max_reconnect_attempts,
                    )
                    bar_queue.put(err)
                    return attempts, True
                logger.warning(
                    "Connection error (attempt %d/%d): %s",
                    attempts, provider.max_reconnect_attempts, err,
                )
                await provider.on_disconnect()
                delay = provider.reconnect_delay * (2 ** (attempts - 1))
                slept = 0.0
                while slept < delay and not stop_event.is_set():
                    await asyncio.sleep(min(0.5, delay - slept))
                    slept += 0.5
                if stop_event.is_set():
                    return attempts, True
                try:
                    await provider.disconnect()
                except Exception as disc_err:
                    logger.debug(
                        "disconnect() before reconnect raised: %s",
                        disc_err,
                    )
                try:
                    await provider.connect()
                    await provider.on_reconnect()
                    logger.info("Reconnected successfully")
                except Exception as reconn_err:
                    logger.error(
                        "Reconnect failed (attempt %d/%d): %s",
                        attempts,
                        provider.max_reconnect_attempts,
                        reconn_err,
                    )
                return attempts, False

            # Latched dead-WS signal from inside the
            # ``except asyncio.TimeoutError`` handler: ``raise`` from one
            # ``except`` does not enter sibling handlers of the same
            # ``try``, so we cannot trigger ``_handle_connection_error``
            # via ``raise``. The handler sets this flag instead and the
            # top-of-loop dispatch at the next iteration consumes it.
            pending_connection_error: BaseException | None = None

            while not stop_event.is_set():
                # Dispatch any deferred dead-WS signal from the previous
                # iteration's ``except asyncio.TimeoutError`` handler
                # (see ``pending_connection_error`` notes above).
                if pending_connection_error is not None:
                    err = pending_connection_error
                    pending_connection_error = None
                    reconnect_attempts, should_break = (
                        await _handle_connection_error(err, reconnect_attempts)
                    )
                    if should_break:
                        break
                    continue

                # Cap the per-iteration wait at 2 s for the existing
                # is_connected healthcheck cadence; if a missed-bar
                # deadline falls sooner, shorten the wait so synthesis
                # fires promptly when the WS goes idle.
                if last_closed_bar is not None:
                    boundary_deadline = (
                        last_closed_bar.timestamp + 2 * tf_seconds + bar_grace
                    )
                    boundary_remaining = boundary_deadline - time.time()
                else:
                    boundary_remaining = float("inf")
                effective_timeout = min(2.0, max(0.05, boundary_remaining))

                try:
                    bar_update = await asyncio.wait_for(
                        provider.watch_ohlcv(watch_symbol, timeframe),
                        timeout=effective_timeout,
                    )
                    reconnect_attempts = 0

                    # Filter duplicates from the historical phase. Strict
                    # ``<`` for closed bars too: see module docstring on
                    # the open-bar overlap (Capital.com et al.) — the
                    # equal-timestamp case must reach the script_runner
                    # so it can refine the partial last-warmup bar with
                    # the true close, not be silently swallowed here.
                    if last_historical_timestamp is not None:
                        ts = bar_update.timestamp
                        if ts < last_historical_timestamp:
                            continue

                    # In-stream dedup against the boundary watchdog:
                    # if a real ``ohlc.event`` arrives late (past
                    # ``bar_grace`` past close) for a slot we already
                    # synthesised, the synth is already in the queue
                    # and may have been consumed downstream — drop the
                    # late real bar so the consumer never sees two
                    # closed bars on the same TF boundary. Only applies
                    # to ``is_closed=True`` because providers' intra-bar
                    # updates legitimately reuse the last closed bar's
                    # timestamp until the next close arrives.
                    if (bar_update.is_closed
                            and last_closed_bar is not None
                            and bar_update.timestamp <= last_closed_bar.timestamp):
                        continue

                    if bar_update.is_closed:
                        last_closed_bar = bar_update
                        if not market_open_state:
                            broker_info(
                                "market reopened: resuming live stream "
                                "(first real bar ts=%d)",
                                bar_update.timestamp,
                            )
                            market_open_state = True
                        # Backpressure probe: a non-trivial qsize or put
                        # latency here is the smoking gun for consumer-side
                        # lag stalling the asyncio loop. The queue is
                        # unbounded so ``put`` cannot actually block on
                        # capacity, but cross-thread handoff + GIL pressure
                        # can still take meaningful time when the consumer
                        # is busy. Warn so the live log preserves the
                        # evidence next time a synth-laden run happens.
                        qsize_before = bar_queue.qsize()
                        put_t0 = time.monotonic()
                        bar_queue.put(bar_update)
                        put_ms = (time.monotonic() - put_t0) * 1000.0
                        if qsize_before > 50 or put_ms > 100.0:
                            broker_warning(
                                "bar_queue backpressure: qsize_before=%d "
                                "put_ms=%.1f ts=%d",
                                qsize_before, put_ms, bar_update.timestamp,
                            )
                    else:
                        # Intra-bar updates are advisory — closed bars
                        # carry authoritative state. With an unbounded
                        # queue, ``put_nowait`` would never raise
                        # ``Full``, so a lagging consumer could let stale
                        # intra-bar items pile up ahead of newer closed
                        # bars and delay live-mode transition or fill
                        # processing. Apply a soft cap based on
                        # ``qsize`` so the closed-bar path retains its
                        # unbounded guarantee while intra-bar growth
                        # stays bounded.
                        if bar_queue.qsize() < _INTRA_BAR_SOFT_CAP:
                            try:
                                bar_queue.put_nowait(bar_update)
                            except Full:
                                pass

                except asyncio.TimeoutError:
                    # Boundary watchdog: if the next-bar close has passed
                    # by more than ``bar_grace`` without a real WS bar,
                    # synthesise a zero-volume filler. Emits exactly one
                    # missed bar per timeout; the next iteration re-checks
                    # against ``time.time()`` and either fills the next
                    # gap or waits for a real push.
                    if (last_closed_bar is not None
                            and time.time()
                            >= last_closed_bar.timestamp
                            + 2 * tf_seconds + bar_grace):
                        synth_ts = last_closed_bar.timestamp + tf_seconds
                        # Session-gate: never synthesise a bar for a slot
                        # that the symbol's opening_hours calendar marks
                        # as closed. Without this gate the framework would
                        # emit V=0 bars across the whole weekend on an FX
                        # symbol whose feed quite legitimately goes silent
                        # at session close. Pine remains paused;
                        # ``bar_index`` does not advance until a real bar
                        # arrives after the next session opens.
                        if not _market_open_at(synth_ts):
                            if market_open_state:
                                broker_info(
                                    "market closed: pausing live stream "
                                    "until next session open "
                                    "(skipped synth ts=%d)",
                                    synth_ts,
                                )
                                market_open_state = False
                            # Dead WS surfaces via the
                            # ``pending_connection_error`` flag, not by
                            # ``raise``: raising from inside this
                            # ``except asyncio.TimeoutError`` handler
                            # escapes past the sibling
                            # ``except Exception`` reconnect block and
                            # would kill the live iterator. WS alive
                            # uses a coarse sleep instead of an
                            # immediate ``continue`` so the loop does
                            # not race ``wait_for`` at the 50ms floor
                            # for the whole closed window
                            # (``boundary_remaining`` is far past, so
                            # ``effective_timeout`` would otherwise pin
                            # to 0.05s → ~20 watch_ohlcv calls/s for
                            # the entire weekend).
                            if not provider.is_connected:
                                pending_connection_error = ConnectionError(
                                    "Provider reports disconnected state"
                                )
                            else:
                                slept = 0.0
                                while (slept < _CLOSED_WINDOW_SLEEP_S
                                       and not stop_event.is_set()):
                                    step = min(
                                        1.0,
                                        _CLOSED_WINDOW_SLEEP_S - slept,
                                    )
                                    await asyncio.sleep(step)
                                    slept += step
                                if stop_event.is_set():
                                    break
                            # Skip synth in both branches — either we are
                            # waiting out the closed window (WS alive) or
                            # we deferred the connection error so the next
                            # iteration's top-of-loop dispatch drives the
                            # reconnect path.
                            continue
                        last_close = last_closed_bar.close
                        synth = OHLCV(
                            timestamp=synth_ts,
                            open=last_close,
                            high=last_close,
                            low=last_close,
                            close=last_close,
                            volume=0.0,
                            extra_fields=last_closed_bar.extra_fields,
                            is_closed=True,
                        )
                        last_closed_bar = synth
                        # Explicit log marker so the V=0 in the next
                        # OHLCV line is unambiguously framework synth,
                        # not a provider-side closed bar whose
                        # ``_tick_volume`` happened to be zero. Carries
                        # the synth timestamp + frozen close so a
                        # post-mortem can see exactly which TF slot the
                        # watchdog filled and at what price.
                        broker_warning(
                            "idle-bar synth emitted: ts=%d close=%s "
                            "(no real ohlc.event for >= 2*tf+grace)",
                            synth_ts, last_close,
                        )
                        bar_queue.put(synth)
                        continue
                    if not provider.is_connected:
                        raise ConnectionError(
                            "Provider reports disconnected state"
                        )
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    reconnect_attempts, should_break = (
                        await _handle_connection_error(e, reconnect_attempts)
                    )
                    if should_break:
                        break
                    continue

        except Exception as e:
            bar_queue.put(e)
        finally:
            if engine_task is not None and not engine_task.done():
                engine_task.cancel()
                try:
                    await engine_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            await _graceful_shutdown()
            bar_queue.put(_SENTINEL)

    def _thread_target():
        if event_loop is not None:
            if event_loop.is_running():
                # Loop is already driven elsewhere (e.g. the CLI broker
                # event-loop pump). Submit our async worker onto it and
                # block this thread on the resulting future instead of
                # trying to start the loop a second time.
                future = asyncio.run_coroutine_threadsafe(_async_loop(), event_loop)
                future.result()
            else:
                asyncio.set_event_loop(event_loop)
                try:
                    event_loop.run_until_complete(_async_loop())
                finally:
                    # The caller owns the loop; don't close it here.
                    pass
        else:
            asyncio.run(_async_loop())

    # Start the provider thread, then block until ``provider.connect()``
    # has completed (or timed out / failed). The warmup that follows
    # therefore runs against an already-listening WS — bars that close
    # mid-warmup land in ``bar_queue`` and are drained as catch-up at
    # the warmup→live boundary. Without this barrier the eager-start
    # design is racy: a slow connect() can finish AFTER warmup, leaving
    # an unsubscribed window in which the next bar close is lost.
    thread = threading.Thread(target=_thread_target, daemon=True, name="live-provider")
    thread.start()
    if not connected_event.wait(timeout=connect_timeout_seconds):
        broker_info(
            "WS connect did not confirm within %.0fs — proceeding; "
            "any underlying error will surface on the first bar pull",
            connect_timeout_seconds,
        )
    elif connect_failure and raise_on_connect_failure:
        # connect() failed fast during warmup. Surface the REAL cause to the
        # caller now, before it reaches start_broker() (whose reconcile would
        # otherwise mask it). ``connected_event`` is set inside the connect
        # except BEFORE ``_async_loop``'s finally has run, so the producer
        # thread may still be draining ``_graceful_shutdown()`` (can_shutdown
        # poll + disconnect) on the shared broker loop. ``_consumer`` is never
        # created on this path, so nothing else will signal ``stop_event`` or
        # join the thread — and the caller's teardown closes the broker loop
        # right away, which would interrupt that pending shutdown and leak the
        # half-open connection. Signal and join here so the teardown finishes
        # while the loop is still alive.
        stop_event.set()
        join_timeout = (shutdown_timeout + 5.0) if shutdown_timeout > 0 else None
        thread.join(timeout=join_timeout)
        raise connect_failure[0]

    def _consumer() -> Generator[OHLCV, None, None]:
        in_warmup_catchup = True

        try:
            while True:
                if in_warmup_catchup:
                    # Drain bars buffered during warmup without blocking.
                    # An empty queue means warmup catch-up is over: emit
                    # the transition sentinel and switch to live polling.
                    try:
                        item = bar_queue.get_nowait()
                    except Empty:
                        yield LIVE_TRANSITION
                        in_warmup_catchup = False
                        continue

                    if item is _SENTINEL:
                        # Stream ended before any live bar arrived. Still
                        # emit the transition so callers that gate on it
                        # observe a clean warmup→live boundary.
                        yield LIVE_TRANSITION
                        break
                    if isinstance(item, BaseException):
                        raise item

                    # Intra-bar updates queued during warmup are dropped:
                    # the historical loop in script_runner has no intra-bar
                    # path and would treat each tick as a fresh bar, so the
                    # bar_index would inflate against a still-open bar.
                    # The latest tick state is preserved by the provider's
                    # internal _last_bar_ohlcv; new ticks after transition
                    # flow through the live path normally.
                    if not item.is_closed:
                        continue

                    yield item
                else:
                    try:
                        item = bar_queue.get(timeout=1.0)
                    except Empty:
                        if not thread.is_alive():
                            break
                        continue

                    if item is _SENTINEL:
                        break
                    if isinstance(item, BaseException):
                        raise item

                    yield item

        except KeyboardInterrupt:
            logger.info("Live streaming interrupted by user")
        finally:
            stop_event.set()
            join_timeout = (shutdown_timeout + 5.0) if shutdown_timeout > 0 else None
            thread.join(timeout=join_timeout)

    return _consumer()
