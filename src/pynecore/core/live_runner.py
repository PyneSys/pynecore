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
from queue import Queue, Empty, Full
from typing import Any

from pynecore.types.ohlcv import OHLCV
from pynecore.core.plugin.live_provider import LiveProviderPlugin
from pynecore.core.script_runner import LIVE_TRANSITION
from pynecore.lib.log import broker_info
from pynecore.lib.timeframe import in_seconds

__all__ = ['live_ohlcv_generator']

logger = logging.getLogger(__name__)


class _Sentinel(BaseException):
    """Marker signaling end of the live stream."""


_SENTINEL = _Sentinel()


def live_ohlcv_generator(
        provider: LiveProviderPlugin,
        symbol: str,
        timeframe: str,
        *,
        last_historical_timestamp: int | None = None,
        shutdown_timeout: float = 120.0,
        event_loop: asyncio.AbstractEventLoop | None = None,
        engine_event_stream: Coroutine[Any, Any, Any] | None = None,
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
    :return: Iterator yielding OHLCV objects (both closed and intra-bar) interleaved
             with a single ``LIVE_TRANSITION`` sentinel marking the warmup→live boundary.
    """
    bar_queue: Queue[OHLCV | BaseException] = Queue(maxsize=100)
    stop_event = threading.Event()
    # Signalled by ``_async_loop`` once ``provider.connect()`` has either
    # succeeded (WS subscribed, ready to receive bars) or failed (with
    # the exception already pushed into ``bar_queue``). Used to make
    # ``live_ohlcv_generator`` block until the WS is up before returning
    # — see module docstring for why warmup must follow connect.
    connected_event = threading.Event()
    connect_timeout_seconds = 30.0

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
    # Grace past close before declaring a bar missed. Floored at 2s for
    # the WS-late case on active markets, capped so longer TFs don't sit
    # on idle gaps too long.
    bar_grace = max(2.0, min(tf_seconds * 0.1, 15.0))

    async def _async_loop():
        # Declared before ``try`` so the ``finally`` branch can reference
        # it even when ``provider.connect()`` raises before assignment.
        engine_task: asyncio.Task | None = None
        # Last CLOSED bar seen (real or synthesised). The boundary
        # watchdog only arms after the first real bar so we never
        # fabricate state without a baseline.
        last_closed_bar: OHLCV | None = None
        try:
            broker_info("WS connect starting (warmup blocks until subscribed)")
            try:
                await provider.connect()
            except BaseException:
                # Unblock the caller waiting on ``connected_event`` so the
                # exception (already on bar_queue via the outer except) can
                # surface through the iterator instead of hanging the
                # ``connected_event.wait()`` for the full timeout.
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

            while not stop_event.is_set():
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
                        bar_queue.put(bar_update)
                    else:
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
                    reconnect_attempts += 1
                    if reconnect_attempts > provider.max_reconnect_attempts:
                        logger.error("Max reconnect attempts reached (%d), stopping",
                                     provider.max_reconnect_attempts)
                        bar_queue.put(e)
                        break

                    logger.warning("Connection error (attempt %d/%d): %s",
                                   reconnect_attempts, provider.max_reconnect_attempts, e)

                    await provider.on_disconnect()

                    delay = provider.reconnect_delay * (2 ** (reconnect_attempts - 1))
                    slept = 0.0
                    while slept < delay and not stop_event.is_set():
                        await asyncio.sleep(min(0.5, delay - slept))
                        slept += 0.5
                    if stop_event.is_set():
                        break

                    try:
                        await provider.disconnect()
                    except Exception as disc_err:
                        logger.debug("disconnect() before reconnect raised: %s", disc_err)

                    try:
                        await provider.connect()
                        await provider.on_reconnect()
                        logger.info("Reconnected successfully")
                    except Exception as reconn_err:
                        logger.error("Reconnect failed (attempt %d/%d): %s",
                                     reconnect_attempts,
                                     provider.max_reconnect_attempts, reconn_err)
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
