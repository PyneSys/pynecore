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

    def normalize_symbol(self, symbol: str) -> str:
        return symbol

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


# --- Reconnect behavior tests ---

class ReconnectTrackingProvider(MockLiveProvider):
    """Provider that records connect/disconnect call order and fails once."""

    def __init__(self, bar_updates: list[OHLCV], fail_at_index: int = 1):
        super().__init__(bar_updates)
        self._fail_at_index = fail_at_index
        self._failed = False
        self.call_log: list[str] = []

    async def connect(self):
        self.call_log.append('connect')
        self._connected = True

    async def disconnect(self):
        self.call_log.append('disconnect')
        self._connected = False

    async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        if not self._failed and self._index == self._fail_at_index:
            self._failed = True
            raise ConnectionError("Simulated connection loss")
        return await super().watch_ohlcv(symbol, timeframe)


def __test_reconnect_calls_disconnect_before_connect__():
    """Reconnect path calls disconnect() before connect() to prevent resource leaks"""
    updates = [
        _make_ohlcv(1000, is_closed=True, close=100.0),
        _make_ohlcv(2000, is_closed=True, close=200.0),
        _make_ohlcv(3000, is_closed=True, close=300.0),
    ]

    provider = ReconnectTrackingProvider(updates, fail_at_index=1)
    bars = list(live_ohlcv_generator(provider, "BTC/USDT", "1D"))

    # Should have: connect, [fail], disconnect, connect, ..., disconnect (shutdown)
    assert provider.call_log[0] == 'connect'

    # Find the reconnect sequence: after the first connect, there should be
    # disconnect followed by connect before the final shutdown disconnect
    post_initial = provider.call_log[1:]
    assert 'disconnect' in post_initial
    disc_idx = post_initial.index('disconnect')
    assert disc_idx + 1 < len(post_initial)
    assert post_initial[disc_idx + 1] == 'connect'

    # Data should still come through after reconnect
    assert len(bars) >= 1


def __test_reconnect_max_attempts_exceeded__():
    """Generator raises after max reconnect attempts are exhausted"""

    class AlwaysFailProvider(MockLiveProvider):
        def __init__(self):
            super().__init__([])
            self.max_reconnect_attempts = 2
            self.reconnect_delay = 0.01

        async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
            raise ConnectionError("Permanent failure")

    provider = AlwaysFailProvider()
    try:
        list(live_ohlcv_generator(provider, "BTC/USDT", "1D"))
        assert False, "Should have raised"
    except ConnectionError:
        pass


# --- Queue overflow tests ---

class FloodProvider(MockLiveProvider):
    """Provider that generates a burst of intra-bar updates, then closed bars."""

    def __init__(self, intra_bar_count: int, closed_bars: list[OHLCV]):
        all_updates: list[OHLCV] = []
        # Generate many intra-bar updates (same timestamp, is_closed=False)
        for i in range(intra_bar_count):
            all_updates.append(_make_ohlcv(1000, is_closed=False, close=100.0 + i * 0.01))
        # Then the actual closed bars
        all_updates.extend(closed_bars)
        super().__init__(all_updates)


def __test_queue_overflow_preserves_closed_bars__():
    """When queue is full, intra-bar updates may be dropped but closed bars are never lost"""
    closed_bars = [
        _make_ohlcv(1000, is_closed=True, close=150.0),
        _make_ohlcv(2000, is_closed=True, close=250.0),
    ]

    # 200 intra-bar updates will overflow the 100-item queue
    provider = FloodProvider(intra_bar_count=200, closed_bars=closed_bars)
    bars = list(live_ohlcv_generator(provider, "BTC/USDT", "1D"))

    # All closed bars must be present
    closed_received = [b for b in bars if b.is_closed]
    assert len(closed_received) == 2
    assert closed_received[0].close == 150.0
    assert closed_received[1].close == 250.0


# --- normalize_symbol tests ---

class NormalizingProvider(MockLiveProvider):
    """Provider that tracks which symbol was passed to watch_ohlcv."""

    def __init__(self, bar_updates: list[OHLCV]):
        super().__init__(bar_updates)
        self.received_symbols: list[str] = []

    def normalize_symbol(self, symbol: str) -> str:
        # Strip "exchange:" prefix
        if ':' in symbol:
            return symbol.split(':', 1)[1]
        return symbol

    async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        self.received_symbols.append(symbol)
        return await super().watch_ohlcv(symbol, timeframe)


def __test_normalize_symbol_applied_to_watch_ohlcv__():
    """Framework calls normalize_symbol() before passing symbol to watch_ohlcv"""
    updates = [_make_ohlcv(1000, is_closed=True)]
    provider = NormalizingProvider(updates)

    list(live_ohlcv_generator(provider, "binance:BTC/USDT", "1D"))

    assert all(s == "BTC/USDT" for s in provider.received_symbols)


# --- Connection error from listener death tests ---

class ListenerDeathProvider(MockLiveProvider):
    """Provider that simulates WebSocket listener dying mid-stream."""

    def __init__(self, bar_updates: list[OHLCV], die_at_index: int = 2):
        super().__init__(bar_updates)
        self._die_at_index = die_at_index
        self._died = False
        self._reconnected = False

    async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        if not self._died and self._index == self._die_at_index:
            self._died = True
            raise ConnectionError("WebSocket listener disconnected")
        if self._died and not self._reconnected:
            self._reconnected = True
        return await super().watch_ohlcv(symbol, timeframe)


def __test_connection_error_triggers_reconnect__():
    """ConnectionError from watch_ohlcv triggers reconnect and resumes streaming"""
    updates = [
        _make_ohlcv(1000, is_closed=True, close=100.0),
        _make_ohlcv(2000, is_closed=True, close=200.0),
        _make_ohlcv(3000, is_closed=True, close=300.0),
        _make_ohlcv(4000, is_closed=True, close=400.0),
    ]

    provider = ListenerDeathProvider(updates, die_at_index=2)
    bars = list(live_ohlcv_generator(provider, "BTC/USDT", "1D"))

    # Should get bars from before and after the simulated death
    assert len(bars) >= 2
    assert provider._reconnected
