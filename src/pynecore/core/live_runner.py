"""
Async/sync bridge for live data streaming.

Runs a LiveProviderPlugin's async watch_ohlcv() in a background thread
and yields OHLCV objects to the synchronous ScriptRunner via queue.Queue.
"""
from __future__ import annotations

import asyncio
import logging
import time
import threading
from collections.abc import Iterator
from queue import Queue, Empty, Full

from pynecore.types.ohlcv import OHLCV
from pynecore.core.plugin.live_provider import LiveProviderPlugin

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
        engine_event_stream: asyncio.coroutines | None = None,
) -> Iterator[OHLCV]:
    """
    Bridge async watch_ohlcv() to a sync Iterator[OHLCV].

    Spawns a background thread running asyncio, collects OHLCV objects
    via queue.Queue, and yields them including intra-bar updates.

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
    :return: Iterator yielding OHLCV objects (both closed and intra-bar).
    """
    bar_queue: Queue[OHLCV | BaseException] = Queue(maxsize=100)
    stop_event = threading.Event()

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

    async def _async_loop():
        # Declared before ``try`` so the ``finally`` branch can reference
        # it even when ``provider.connect()`` raises before assignment.
        engine_task: asyncio.Task | None = None
        try:
            await provider.connect()
            watch_symbol = provider.normalize_symbol(symbol)
            logger.info("Live provider connected: %s %s@%s",
                        type(provider).__name__, symbol, timeframe)

            # Broker mode: attach the Order Sync Engine's event stream as
            # a background task so OrderEvents land in its queue without
            # blocking the OHLCV reader.
            if engine_event_stream is not None:
                engine_task = asyncio.create_task(engine_event_stream)

            reconnect_attempts = 0

            while not stop_event.is_set():
                try:
                    bar_update = await asyncio.wait_for(
                        provider.watch_ohlcv(watch_symbol, timeframe),
                        timeout=2.0,
                    )
                    reconnect_attempts = 0

                    # Filter duplicates from the historical phase
                    if last_historical_timestamp is not None:
                        ts = bar_update.timestamp
                        if bar_update.is_closed and ts <= last_historical_timestamp:
                            continue
                        if not bar_update.is_closed and ts < last_historical_timestamp:
                            continue

                    if bar_update.is_closed:
                        bar_queue.put(bar_update)
                    else:
                        try:
                            bar_queue.put_nowait(bar_update)
                        except Full:
                            pass

                except asyncio.TimeoutError:
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

    thread = threading.Thread(target=_thread_target, daemon=True, name="live-provider")
    thread.start()

    try:
        while True:
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
        # Wait for graceful shutdown to complete
        join_timeout = (shutdown_timeout + 5.0) if shutdown_timeout > 0 else None
        thread.join(timeout=join_timeout)
