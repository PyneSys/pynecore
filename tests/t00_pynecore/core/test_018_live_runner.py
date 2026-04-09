"""
Tests for the live runner async/sync bridge.
"""
import asyncio
import time

from pynecore.core.live_runner import live_ohlcv_generator
from pynecore.types.ohlcv import OHLCV


def _make_ohlcv(timestamp: int, close: float = 100.0, is_closed: bool = True) -> OHLCV:
    return OHLCV(timestamp=timestamp, open=close, high=close + 1,
                 low=close - 1, close=close, volume=1000.0, is_closed=is_closed)


class MockLiveProvider:
    """Mock LiveProviderPlugin for testing the bridge."""

    def __init__(self, bar_updates: list[OHLCV]):
        self._bar_updates = bar_updates
        self._index = 0
        self._connected = False
        self.reconnect_delay = 0.01
        self.max_reconnect_attempts = 3

    async def connect(self):
        self._connected = True

    async def disconnect(self):
        self._connected = False

    @property
    def is_connected(self):
        return self._connected

    async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        if self._index >= len(self._bar_updates):
            raise asyncio.CancelledError()

        bar = self._bar_updates[self._index]
        self._index += 1
        await asyncio.sleep(0.001)
        return bar

    async def on_disconnect(self):
        pass

    async def on_reconnect(self):
        pass

    async def can_shutdown(self):
        return True


def __test_live_generator_yields_all_bar_updates__():
    """live_ohlcv_generator yields both intra-bar and closed bar updates"""
    updates = [
        _make_ohlcv(1000, is_closed=False, close=100.0),
        _make_ohlcv(1000, is_closed=True, close=101.0),
        _make_ohlcv(2000, is_closed=False, close=102.0),
        _make_ohlcv(2000, is_closed=True, close=103.0),
    ]

    provider = MockLiveProvider(updates)
    bars = list(live_ohlcv_generator(provider, "BTC/USDT", "1D"))

    assert len(bars) == 4
    assert not bars[0].is_closed
    assert bars[0].close == 100.0
    assert bars[1].is_closed
    assert bars[1].close == 101.0
    assert not bars[2].is_closed
    assert bars[3].is_closed


def __test_live_generator_filters_old_bars__():
    """live_ohlcv_generator skips bars older than last_historical_timestamp"""
    updates = [
        _make_ohlcv(1000, is_closed=True, close=100.0),
        _make_ohlcv(2000, is_closed=True, close=200.0),
        _make_ohlcv(3000, is_closed=True, close=300.0),
    ]

    provider = MockLiveProvider(updates)
    bars = list(live_ohlcv_generator(provider, "BTC/USDT", "1D",
                                     last_historical_timestamp=2000))

    assert len(bars) == 1
    assert bars[0].timestamp == 3000
    assert bars[0].close == 300.0


def __test_live_generator_yields_ohlcv_objects__():
    """live_ohlcv_generator yields OHLCV objects directly"""
    updates = [
        _make_ohlcv(1000, is_closed=True),
    ]

    provider = MockLiveProvider(updates)
    bars = list(live_ohlcv_generator(provider, "BTC/USDT", "1D"))

    assert len(bars) == 1
    assert isinstance(bars[0], OHLCV)
    assert bars[0].is_closed is True


def __test_live_generator_connects_and_disconnects__():
    """live_ohlcv_generator calls connect on start and disconnect on finish"""
    updates = [
        _make_ohlcv(1000, is_closed=True),
    ]

    provider = MockLiveProvider(updates)
    list(live_ohlcv_generator(provider, "BTC/USDT", "1D"))

    assert not provider.is_connected


def __test_live_generator_empty_stream__():
    """live_ohlcv_generator handles empty stream gracefully"""
    provider = MockLiveProvider([])
    bars = list(live_ohlcv_generator(provider, "BTC/USDT", "1D"))
    assert len(bars) == 0


class DelayedShutdownProvider(MockLiveProvider):
    """Provider that delays shutdown for a number of can_shutdown() calls."""

    def __init__(self, bar_updates: list[OHLCV], deny_count: int = 2):
        super().__init__(bar_updates)
        self._deny_count = deny_count
        self._shutdown_calls = 0

    async def can_shutdown(self):
        self._shutdown_calls += 1
        if self._shutdown_calls <= self._deny_count:
            return False
        return True


def __test_graceful_shutdown_waits_for_can_shutdown__():
    """Shutdown waits until can_shutdown() returns True"""
    updates = [_make_ohlcv(1000, is_closed=True)]
    provider = DelayedShutdownProvider(updates, deny_count=2)

    list(live_ohlcv_generator(provider, "BTC/USDT", "1D", shutdown_timeout=10.0))

    assert provider._shutdown_calls == 3
    assert not provider.is_connected


def __test_graceful_shutdown_timeout_forces_disconnect__():
    """Shutdown force-disconnects after timeout even if can_shutdown() returns False"""

    class NeverReadyProvider(MockLiveProvider):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._shutdown_calls = 0

        async def can_shutdown(self):
            self._shutdown_calls += 1
            return False

    updates = [_make_ohlcv(1000, is_closed=True)]
    provider = NeverReadyProvider(updates)

    start = time.monotonic()
    list(live_ohlcv_generator(provider, "BTC/USDT", "1D", shutdown_timeout=2.0))
    elapsed = time.monotonic() - start

    assert provider._shutdown_calls >= 1
    assert elapsed < 5.0
    assert not provider.is_connected


def __test_graceful_shutdown_zero_timeout_waits_until_ready__():
    """shutdown_timeout=0 waits indefinitely until can_shutdown() returns True"""
    updates = [_make_ohlcv(1000, is_closed=True)]
    provider = DelayedShutdownProvider(updates, deny_count=3)

    list(live_ohlcv_generator(provider, "BTC/USDT", "1D", shutdown_timeout=0))

    assert provider._shutdown_calls == 4
    assert not provider.is_connected
