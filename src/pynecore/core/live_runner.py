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
from queue import Queue, Empty

from pynecore.core.plugin.live_provider import LiveProviderPlugin, BarUpdate
from pynecore.types.ohlcv import OHLCV

__all__ = ['live_ohlcv_generator']

logger = logging.getLogger(__name__)

_SENTINEL = object()


def live_ohlcv_generator(
        provider: LiveProviderPlugin,
        symbol: str,
        timeframe: str,
        *,
        last_historical_timestamp: int | None = None,
        shutdown_timeout: float = 120.0,
) -> Iterator[OHLCV]:
    """
    Bridge async watch_ohlcv() to a sync Iterator[OHLCV].

    Spawns a background thread running asyncio, collects BarUpdate objects
    via queue.Queue, filters for closed bars, and yields OHLCV.

    :param provider: A LiveProviderPlugin instance (already configured).
    :param symbol: Symbol in provider-specific format.
    :param timeframe: Timeframe in TradingView format.
    :param last_historical_timestamp: Timestamp of the last historical bar to avoid duplicates.
    :param shutdown_timeout: Max seconds to wait for graceful shutdown. 0 = wait forever.
    :return: Iterator yielding OHLCV objects as bars close.
    """
    bar_queue: Queue[BarUpdate | BaseException] = Queue(maxsize=100)
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
        except Exception:
            pass

    async def _async_loop():
        try:
            await provider.connect()
            logger.info("Live provider connected: %s %s@%s",
                        type(provider).__name__, symbol, timeframe)

            reconnect_attempts = 0

            while not stop_event.is_set():
                try:
                    bar_update = await asyncio.wait_for(
                        provider.watch_ohlcv(symbol, timeframe),
                        timeout=2.0,
                    )
                    reconnect_attempts = 0

                    if not bar_update.is_closed:
                        continue

                    if (last_historical_timestamp is not None
                            and bar_update.ohlcv.timestamp <= last_historical_timestamp):
                        continue

                    bar_queue.put(bar_update)

                except asyncio.TimeoutError:
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
                    await asyncio.sleep(delay)

                    try:
                        await provider.connect()
                        await provider.on_reconnect()
                        logger.info("Reconnected successfully")
                    except Exception as reconn_err:
                        logger.error("Reconnect failed: %s", reconn_err)

        except Exception as e:
            bar_queue.put(e)
        finally:
            await _graceful_shutdown()
            bar_queue.put(_SENTINEL)

    def _thread_target():
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

            yield item.ohlcv

    except KeyboardInterrupt:
        logger.info("Live streaming interrupted by user")
    finally:
        stop_event.set()
        # Wait for graceful shutdown to complete
        join_timeout = (shutdown_timeout + 5.0) if shutdown_timeout > 0 else None
        thread.join(timeout=join_timeout)
