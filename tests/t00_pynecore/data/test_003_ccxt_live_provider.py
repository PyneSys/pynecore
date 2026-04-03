"""
Integration tests for CCXT LiveProviderPlugin websocket streaming.

These tests connect to real exchanges and require network access.
They are skipped if ccxt is not installed.
"""
import asyncio
import logging

import pytest

from pynecore.providers.ccxt import CCXTProvider
from pynecore.core.plugin.live_provider import BarUpdate
from pynecore.types.ohlcv import OHLCV

logging.getLogger("ccxt").setLevel(logging.WARNING)
logging.getLogger("ccxt.base.exchange").setLevel(logging.WARNING)

pytestmark = pytest.mark.live


def _skip_if_no_ccxt():
    try:
        import ccxt.pro  # noqa: F401
    except ImportError:
        pytest.skip("CCXT library not available")


def __test_ccxt_live_connect_disconnect__():
    """CCXTProvider can connect and disconnect via websocket"""
    _skip_if_no_ccxt()

    provider = CCXTProvider(
        symbol="BYBIT:BTC/USDT:USDT",
        timeframe="1",
        ohlv_dir=None,
    )

    async def _run():
        await provider.connect()
        assert provider.is_connected
        await provider.disconnect()
        assert not provider.is_connected

    asyncio.run(_run())


def __test_ccxt_live_watch_ohlcv__():
    """CCXTProvider receives at least one OHLCV update from Bybit websocket"""
    _skip_if_no_ccxt()

    provider = CCXTProvider(
        symbol="BYBIT:BTC/USDT:USDT",
        timeframe="1",
        ohlv_dir=None,
    )

    async def _run():
        await provider.connect()
        try:
            bar_update = await asyncio.wait_for(
                provider.watch_ohlcv("BTC/USDT:USDT", "1"),
                timeout=30.0,
            )

            assert isinstance(bar_update, BarUpdate)
            assert isinstance(bar_update.ohlcv, OHLCV)
            assert isinstance(bar_update.is_closed, bool)

            ohlcv = bar_update.ohlcv
            assert ohlcv.timestamp > 0
            assert ohlcv.open > 0
            assert ohlcv.high >= ohlcv.low
            assert ohlcv.close > 0
            assert ohlcv.volume >= 0
        finally:
            await provider.disconnect()

    asyncio.run(_run())


def __test_ccxt_live_multiple_updates__():
    """CCXTProvider receives multiple consecutive updates"""
    _skip_if_no_ccxt()

    provider = CCXTProvider(
        symbol="BYBIT:BTC/USDT:USDT",
        timeframe="1",
        ohlv_dir=None,
    )

    async def _run():
        await provider.connect()
        try:
            updates = []
            for _ in range(3):
                bar_update = await asyncio.wait_for(
                    provider.watch_ohlcv("BTC/USDT:USDT", "1"),
                    timeout=30.0,
                )
                updates.append(bar_update)

            assert len(updates) == 3
            for u in updates:
                assert isinstance(u, BarUpdate)
                assert u.ohlcv.timestamp > 0
        finally:
            await provider.disconnect()

    asyncio.run(_run())


def __test_ccxt_live_can_shutdown_default__():
    """CCXTProvider.can_shutdown() returns True by default"""
    _skip_if_no_ccxt()

    provider = CCXTProvider(
        symbol="BYBIT:BTC/USDT:USDT",
        timeframe="1",
        ohlv_dir=None,
    )

    result = asyncio.run(provider.can_shutdown())
    assert result is True


def __test_ccxt_live_generator_integration__():
    """Full live_ohlcv_generator integration: connect, receive one bar, shutdown"""
    _skip_if_no_ccxt()

    from pynecore.core.live_runner import live_ohlcv_generator

    provider = CCXTProvider(
        symbol="BYBIT:BTC/USDT:USDT",
        timeframe="1",
        ohlv_dir=None,
    )

    received = []
    for ohlcv in live_ohlcv_generator(provider, "BTC/USDT:USDT", "1",
                                       shutdown_timeout=5.0):
        received.append(ohlcv)
        assert isinstance(ohlcv, OHLCV)
        assert ohlcv.timestamp > 0
        assert ohlcv.close > 0
        if len(received) >= 1:
            break

    assert len(received) >= 1
    assert not provider.is_connected
