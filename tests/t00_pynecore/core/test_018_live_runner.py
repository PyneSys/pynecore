"""
Tests for the live runner async/sync bridge.
"""
import asyncio
import time

from pynecore.core.live_runner import live_ohlcv_generator
from pynecore.core.script_runner import LIVE_TRANSITION
from pynecore.types.ohlcv import OHLCV


def _make_ohlcv(timestamp: int, close: float = 100.0, is_closed: bool = True) -> OHLCV:
    return OHLCV(timestamp=timestamp, open=close, high=close + 1,
                 low=close - 1, close=close, volume=1000.0, is_closed=is_closed)


def _drain(provider, *args, **kwargs) -> tuple[list[OHLCV], list[OHLCV]]:
    """Consume the live iterator and split out the pre/post LIVE_TRANSITION halves.

    Returns ``(catchup_bars, live_bars)`` — the warmup catch-up batch
    (empty in tests where the producer hasn't queued anything by the
    time the consumer hits its first ``get_nowait``) and everything
    yielded after the in-band transition sentinel. Tests typically only
    care about ``live_bars``; the split makes the boundary explicit.
    """
    catchup: list[OHLCV] = []
    live: list[OHLCV] = []
    seen_transition = False
    for item in live_ohlcv_generator(provider, *args, **kwargs):
        if item is LIVE_TRANSITION:
            seen_transition = True
            continue
        if seen_transition:
            live.append(item)
        else:
            catchup.append(item)
    return catchup, live


class MockLiveProvider:
    """Mock LiveProviderPlugin for testing the bridge."""

    def __init__(self, bar_updates: list[OHLCV]):
        self._bar_updates = bar_updates
        self._index = 0
        self._connected = False
        self.reconnect_delay = 0.01
        self.max_reconnect_delay = 0.05
        self.feed_timeout_bars = 3

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
    """live_ohlcv_generator yields both intra-bar and closed bar updates after the transition"""
    updates = [
        _make_ohlcv(1000, is_closed=False, close=100.0),
        _make_ohlcv(1000, is_closed=True, close=101.0),
        _make_ohlcv(2000, is_closed=False, close=102.0),
        _make_ohlcv(2000, is_closed=True, close=103.0),
    ]

    provider = MockLiveProvider(updates)
    _, bars = _drain(provider, "BTC/USDT", "1D")

    assert len(bars) == 4
    assert not bars[0].is_closed
    assert bars[0].close == 100.0
    assert bars[1].is_closed
    assert bars[1].close == 101.0
    assert not bars[2].is_closed
    assert bars[3].is_closed


def __test_live_generator_filters_old_bars__():
    """live_ohlcv_generator skips bars strictly older than last_historical_timestamp.

    A bar at exactly ``last_historical_timestamp`` MUST pass through —
    that's the close of the still-open last-warmup bar (e.g.
    Capital.com's REST history includes the currently-forming bar, and
    the WS push for its close has the same timestamp). The script_runner
    live loop recognises the same-timestamp first live update as a
    continuation of the last warmup bar.
    """
    updates = [
        _make_ohlcv(1000, is_closed=True, close=100.0),
        _make_ohlcv(2000, is_closed=True, close=200.0),
        _make_ohlcv(3000, is_closed=True, close=300.0),
    ]

    provider = MockLiveProvider(updates)
    _, bars = _drain(provider, "BTC/USDT", "1D",
                     last_historical_timestamp=2000)

    assert len(bars) == 2
    assert bars[0].timestamp == 2000
    assert bars[0].close == 200.0
    assert bars[1].timestamp == 3000
    assert bars[1].close == 300.0


def __test_live_generator_yields_ohlcv_objects__():
    """live_ohlcv_generator yields OHLCV objects directly"""
    updates = [
        _make_ohlcv(1000, is_closed=True),
    ]

    provider = MockLiveProvider(updates)
    _, bars = _drain(provider, "BTC/USDT", "1D")

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
    """live_ohlcv_generator emits LIVE_TRANSITION even for an empty stream"""
    provider = MockLiveProvider([])
    items = list(live_ohlcv_generator(provider, "BTC/USDT", "1D"))
    assert items == [LIVE_TRANSITION]


def __test_live_generator_emits_transition_sentinel__():
    """live_ohlcv_generator yields one LIVE_TRANSITION sentinel between catch-up and live."""
    updates = [_make_ohlcv(1000, is_closed=True), _make_ohlcv(2000, is_closed=True)]
    provider = MockLiveProvider(updates)

    items = list(live_ohlcv_generator(provider, "BTC/USDT", "1D"))
    transitions = [i for i, x in enumerate(items) if x is LIVE_TRANSITION]
    assert len(transitions) == 1, f"expected exactly one LIVE_TRANSITION, got {len(transitions)}"


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
    _, bars = _drain(provider, "BTC/USDT", "1D")

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


def __test_reconnect_retries_indefinitely_and_recovers__():
    """Reconnect has no attempt cap: an outage longer than any fixed limit
    is ridden out and the stream resumes once the provider recovers.

    A live bot must never give up on a network outage — the historical
    behaviour (raise after ``max_reconnect_attempts``, default 10) killed
    the run ~17 minutes into a router restart. 15 consecutive failures
    here is comfortably past that old cap, so passing proves the limit is
    gone, and the final real bar proves the stream survives the outage.
    """

    class FlakyProvider(MockLiveProvider):
        def __init__(self, fail_count: int):
            super().__init__([_make_ohlcv(1000, is_closed=True, close=100.0)])
            self.reconnect_delay = 0.001
            self.max_reconnect_delay = 0.002
            self.connect_calls = 0
            self._remaining_failures = fail_count

        async def connect(self):
            self.connect_calls += 1
            self._connected = True

        async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
            if self._remaining_failures > 0:
                self._remaining_failures -= 1
                raise ConnectionError("Simulated long outage")
            return await super().watch_ohlcv(symbol, timeframe)

    provider = FlakyProvider(fail_count=15)
    _, bars = _drain(provider, "BTC/USDT", "1D")

    assert len(bars) == 1
    assert bars[0].close == 100.0
    # Initial connect + one reconnect per failed watch call.
    assert provider.connect_calls >= 15


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
    _, bars = _drain(provider, "BTC/USDT", "1D")

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


# --- Idle-bar synthesis (boundary watchdog) tests ---

class _IdleAfterFirstBar(MockLiveProvider):
    """Sends pre-canned bars then blocks indefinitely.

    Used to drive the boundary watchdog: once the queue is exhausted,
    ``watch_ohlcv`` never returns, so the only way a bar lands in
    ``bar_queue`` afterwards is via the framework's idle-bar synthesis.
    """

    async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        if self._index < len(self._bar_updates):
            bar = self._bar_updates[self._index]
            self._index += 1
            await asyncio.sleep(0.001)
            return bar
        # Block until the consumer cancels the loop. ``stop_event`` set
        # by the generator's finally tears the thread down via the outer
        # CancelledError path, so we don't need to react to it here.
        await asyncio.sleep(60)
        raise asyncio.CancelledError()


def __test_live_generator_synthesises_idle_bars_at_tf_boundary__():
    """The framework fills idle TF intervals with zero-volume CLOSED bars.

    Capital.com's WS only emits ``ohlc.event`` for bars that contained
    at least one tick — idle minutes produce no event, freezing
    ``bar_index`` while REST history later returns those zero-volume
    bars. The framework's boundary watchdog synthesises one filler per
    missed TF interval (O=H=L=C=last close, V=0) so live and replay
    step in lockstep.

    Setup: real bar with a stale timestamp (200 s ago, more than two
    1-minute TF intervals + grace) — the watchdog therefore fires on
    the very first iteration and immediately catches up.
    """
    base_ts = int(time.time()) - 200
    updates = [_make_ohlcv(base_ts, is_closed=True, close=100.0)]
    provider = _IdleAfterFirstBar(updates)

    bars: list[OHLCV] = []
    seen_transition = False
    for item in live_ohlcv_generator(provider, "BTC/USDT", "1"):
        if item is LIVE_TRANSITION:
            seen_transition = True
            continue
        if not seen_transition:
            continue
        bars.append(item)
        if len(bars) >= 3:
            break

    assert bars[0].timestamp == base_ts
    assert bars[0].volume == 1000.0  # real bar, untouched

    # Subsequent bars are synth fillers. The watchdog advances the
    # baseline by one TF per emission; each filler carries the previous
    # close as O=H=L=C and zero volume.
    for i in range(1, 3):
        assert bars[i].timestamp == base_ts + 60 * i, (
            f"synth bar {i} timestamp {bars[i].timestamp} "
            f"!= expected {base_ts + 60 * i}"
        )
        assert bars[i].open == 100.0
        assert bars[i].high == 100.0
        assert bars[i].low == 100.0
        assert bars[i].close == 100.0
        assert bars[i].volume == 0.0
        assert bars[i].is_closed


class _LateBarAfterIdle(MockLiveProvider):
    """Sends one bar, blocks long enough for the watchdog to synth, then sends a bar with the same boundary timestamp."""

    def __init__(self, bar_updates: list[OHLCV], block_seconds: float = 0.3):
        super().__init__(bar_updates)
        self._block_seconds = block_seconds
        self._blocked = False

    async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        if self._index == 1 and not self._blocked:
            # Stall once between bar 0 and bar 1 — long enough for the
            # watchdog to synth a filler at the next boundary.
            self._blocked = True
            await asyncio.sleep(self._block_seconds)
        return await super().watch_ohlcv(symbol, timeframe)


def __test_live_generator_drops_late_real_bar_for_already_synthesised_boundary__():
    """A late real ``ohlc.event`` for an already-synthesised slot is dropped.

    Without dedup the consumer would see two CLOSED bars on the same TF
    boundary — the synth (already published, possibly already executed
    against the script) and the late real bar with conflicting OHLCV.
    Drop the real one: the synth's flat values are now the authoritative
    live record for that minute.
    """
    base_ts = int(time.time()) - 200
    updates = [
        _make_ohlcv(base_ts, is_closed=True, close=100.0),
        # Late real bar arrives at base_ts + 60 — same boundary the
        # watchdog will synthesise while we sleep below.
        _make_ohlcv(base_ts + 60, is_closed=True, close=999.0),
    ]
    provider = _LateBarAfterIdle(updates, block_seconds=0.3)

    bars: list[OHLCV] = []
    seen_transition = False
    for item in live_ohlcv_generator(provider, "BTC/USDT", "1"):
        if item is LIVE_TRANSITION:
            seen_transition = True
            continue
        if not seen_transition:
            continue
        bars.append(item)
        if len(bars) >= 2:
            break

    assert bars[0].timestamp == base_ts
    assert bars[0].close == 100.0
    # Second bar must be the synth (V=0, close=100), NOT the late real
    # bar with close=999. The late one was dropped by the dedup.
    assert bars[1].timestamp == base_ts + 60
    assert bars[1].volume == 0.0
    assert bars[1].close == 100.0


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
    _, bars = _drain(provider, "BTC/USDT", "1D")

    # Should get bars from before and after the simulated death
    assert len(bars) >= 2
    assert provider._reconnected


# --- Session-gate tests ---

def _make_syminfo(opening_hours, timezone: str = "UTC"):
    """Build a minimal SymInfo with the given opening_hours + timezone.

    Only the fields read by ``live_ohlcv_generator``'s session gate are
    populated meaningfully; the rest get throw-away values.
    """
    from pynecore.core.syminfo import SymInfo
    return SymInfo(
        prefix="TEST",
        description="test",
        ticker="TEST",
        currency="USD",
        period="1",
        type="forex",
        mintick=0.0001,
        pricescale=10000,
        minmove=1,
        pointvalue=1.0,
        mincontract=1.0,
        opening_hours=opening_hours,
        session_starts=[],
        session_ends=[],
        timezone=timezone,
    )


class _IdleThenCancelProvider(MockLiveProvider):
    """Yields pre-canned bars, then raises CancelledError after a fixed
    number of idle ``watch_ohlcv`` calls.

    The framework's ``wait_for`` cancels any in-flight ``asyncio.sleep``
    when its timeout elapses, so a sleep-then-raise pattern never gets
    to the raise statement — counting idle invocations works regardless.
    Each idle call lives only as long as the framework's ``effective_timeout``
    (down to 0.05 s once the synth deadline is in the past), so the test
    completes within ``max_idle_calls × 0.05 s`` wall time.
    """

    def __init__(self, bar_updates: list[OHLCV], max_idle_calls: int = 30):
        super().__init__(bar_updates)
        self._idle_calls = 0
        self._max_idle_calls = max_idle_calls

    async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        if self._index < len(self._bar_updates):
            bar = self._bar_updates[self._index]
            self._index += 1
            await asyncio.sleep(0.001)
            return bar
        self._idle_calls += 1
        if self._idle_calls >= self._max_idle_calls:
            raise asyncio.CancelledError()
        # Park until ``wait_for`` cancels us so the framework synth deadline
        # (boundary_remaining < 0 → effective_timeout=0.05 s) can elapse
        # and the gate is actually exercised every iteration.
        await asyncio.sleep(10.0)
        raise asyncio.CancelledError()


def __test_synth_gate_suppresses_synth_during_known_closed_window__():
    """When syminfo.opening_hours says market closed, no idle synth is emitted.

    Setup: a SymInfo with a single 5-minute weekday interval anchored
    12 hours away from ``synth_ts`` (computed from ``base_ts + 60``).
    Pinning the open interval to ``synth_ts``'s opposite half-day keeps
    the test deterministic regardless of wall-clock time, while still
    exercising the closed-window gate for the slot the watchdog tries
    to synth. The synth deadline elapses (real bar 200s in the past on
    1m TF) but the gate intercepts before any V=0 OHLCV lands in queue.
    """
    from pynecore.core.syminfo import SymInfoInterval
    from datetime import datetime as ddatetime, time as dtime, timedelta, UTC

    base_ts = int(time.time()) - 200
    synth_ts = base_ts + 60
    synth_dt = ddatetime.fromtimestamp(synth_ts, tz=UTC)
    open_dt = synth_dt + timedelta(hours=12)
    open_time = dtime(open_dt.hour, open_dt.minute, 0)
    close_dt = open_dt + timedelta(minutes=5)
    close_time = dtime(close_dt.hour, close_dt.minute, 0)
    closed_calendar = [
        SymInfoInterval(day=d, start=open_time, end=close_time)
        for d in range(7)
    ]
    syminfo = _make_syminfo(closed_calendar, timezone="UTC")

    updates = [_make_ohlcv(base_ts, is_closed=True, close=100.0)]
    provider = _IdleThenCancelProvider(updates, max_idle_calls=20)

    # Shrink the closed-window sleep so the test does not spend the full
    # 30s production cadence per gated timeout. With ``max_idle_calls=20``
    # the unpatched value would block the suite for ~10 minutes per run.
    from pynecore.core import live_runner as _live_runner_mod
    _orig_sleep = _live_runner_mod._CLOSED_WINDOW_SLEEP_S
    _live_runner_mod._CLOSED_WINDOW_SLEEP_S = 0.05
    try:
        _, bars = _drain(provider, "TEST", "1", syminfo=syminfo)
    finally:
        _live_runner_mod._CLOSED_WINDOW_SLEEP_S = _orig_sleep

    # Only the real bar (volume=1000) was emitted. No V=0 synth filler.
    real_bars = [b for b in bars if b.volume > 0]
    synth_bars = [b for b in bars if b.volume == 0.0]
    assert len(real_bars) == 1
    assert real_bars[0].timestamp == base_ts
    assert synth_bars == [], (
        f"expected no synth bars during closed window, got {len(synth_bars)}"
    )


def __test_synth_gate_passthrough_when_opening_hours_empty__():
    """Empty syminfo.opening_hours preserves legacy 24/7 synth behaviour.

    Mirrors __test_live_generator_synthesises_idle_bars_at_tf_boundary__
    but with an explicit empty-calendar SymInfo to verify the gate does
    NOT short-circuit when there is no calendar data.
    """
    syminfo = _make_syminfo([], timezone="UTC")

    base_ts = int(time.time()) - 200
    updates = [_make_ohlcv(base_ts, is_closed=True, close=100.0)]
    provider = _IdleAfterFirstBar(updates)

    bars: list[OHLCV] = []
    seen_transition = False
    for item in live_ohlcv_generator(provider, "TEST", "1", syminfo=syminfo):
        if item is LIVE_TRANSITION:
            seen_transition = True
            continue
        if not seen_transition:
            continue
        bars.append(item)
        if len(bars) >= 3:
            break

    # First bar real, next two are V=0 synth fillers (no gate suppression).
    assert bars[0].volume == 1000.0
    assert bars[1].volume == 0.0
    assert bars[2].volume == 0.0
    assert bars[1].timestamp == base_ts + 60
    assert bars[2].timestamp == base_ts + 120


# --- Feed-liveness watchdog tests ---

class _SilentFeedProvider(MockLiveProvider):
    """Connected-looking provider whose feed goes silent after its bars.

    ``watch_ohlcv`` serves the pre-canned bars, then parks forever (each
    park is cancelled by the framework's ``wait_for`` timeout).
    ``is_connected`` stays True throughout, so only the feed-liveness
    watchdog can drive a reconnect — the dead-subscription / half-open-
    socket failure mode. Ends the run by raising ``CancelledError`` once
    ``connect()`` was called ``stop_after_connects`` times, or after
    ``max_idle_calls`` silent ``watch_ohlcv`` invocations.
    """

    def __init__(self, bar_updates: list[OHLCV], *,
                 stop_after_connects: int | None = None,
                 max_idle_calls: int | None = None):
        super().__init__(bar_updates)
        self.connect_calls = 0
        self._stop_after_connects = stop_after_connects
        self._idle_calls = 0
        self._max_idle_calls = max_idle_calls

    async def connect(self):
        self.connect_calls += 1
        self._connected = True

    async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        if (self._stop_after_connects is not None
                and self.connect_calls >= self._stop_after_connects):
            raise asyncio.CancelledError()
        if self._index < len(self._bar_updates):
            bar = self._bar_updates[self._index]
            self._index += 1
            await asyncio.sleep(0.001)
            return bar
        self._idle_calls += 1
        if (self._max_idle_calls is not None
                and self._idle_calls >= self._max_idle_calls):
            raise asyncio.CancelledError()
        await asyncio.sleep(10.0)
        raise asyncio.CancelledError()


def __test_feed_staleness_watchdog_forces_reconnect__():
    """A connected-but-silent feed is reconnected during an open session.

    The failure mode this guards: the transport looks healthy
    (``is_connected`` True) but the server-side subscription is gone or
    the socket is half-open, so ``watch_ohlcv`` never returns and idle-bar
    synthesis would otherwise run the strategy on a frozen price forever.
    With ``feed_timeout_bars=1`` on a 1-second timeframe (staleness floor
    shrunk for test speed) the threshold is ~1 s, so the watchdog must
    drive ``connect()`` again within the test's few-second budget without
    the provider ever raising a ConnectionError.
    """
    base_ts = int(time.time()) - 30
    updates = [_make_ohlcv(base_ts, is_closed=True, close=100.0)]
    provider = _SilentFeedProvider(updates, stop_after_connects=2)
    provider.feed_timeout_bars = 1

    from pynecore.core import live_runner as _live_runner_mod
    _orig_floor = _live_runner_mod._FEED_STALE_FLOOR_S
    _live_runner_mod._FEED_STALE_FLOOR_S = 0.2
    try:
        _drain(provider, "TEST", "1S")
    finally:
        _live_runner_mod._FEED_STALE_FLOOR_S = _orig_floor

    assert provider.connect_calls >= 2, (
        "feed-liveness watchdog should have forced a reconnect on a "
        "connected-but-silent feed"
    )


def __test_feed_staleness_watchdog_disabled_with_none__():
    """``feed_timeout_bars=None`` keeps a silent-but-connected feed running.

    Same silent-feed setup as the positive test, but with the watchdog
    disabled there must be no reconnect — idle-bar synthesis keeps
    filling slots on the single original connection.

    ``base_ts`` is far enough in the past that all 40 idle calls run in
    the synth catch-up phase (~0.05 s effective timeout each), keeping
    the test at ~2 s of wall clock — well past the ~1 s staleness
    threshold the positive test reconnects under.
    """
    base_ts = int(time.time()) - 200
    updates = [_make_ohlcv(base_ts, is_closed=True, close=100.0)]
    provider = _SilentFeedProvider(updates, max_idle_calls=40)
    provider.feed_timeout_bars = None

    from pynecore.core import live_runner as _live_runner_mod
    _orig_floor = _live_runner_mod._FEED_STALE_FLOOR_S
    _live_runner_mod._FEED_STALE_FLOOR_S = 0.2
    try:
        _, bars = _drain(provider, "TEST", "1S")
    finally:
        _live_runner_mod._FEED_STALE_FLOOR_S = _orig_floor

    assert provider.connect_calls == 1, (
        "watchdog disabled: no reconnect may happen on a silent feed"
    )
    # Idle synthesis must keep working without the watchdog interfering.
    assert any(b.volume == 0.0 for b in bars)


def __test_feed_staleness_fires_when_slot_pinned_in_closed_window__():
    """A feed that dies across a session close trips the watchdog after reopen.

    Regression test for the pinned-slot blind spot: when the feed dies
    near a session close, ``last_closed_bar`` stops advancing, so the
    pending synth slot stays calendar-closed forever and the closed-window
    branch runs on every iteration — even after the session has reopened.
    That branch used to rebase the staleness clock unconditionally, which
    disarmed the liveness watchdog for good (frozen strategy on an open
    market); it must instead trip the watchdog once the session is open
    again.

    Calendar: one interval covering "now" (session open) but NOT the
    pinned synth slot ~139 s in the past (slot closed). With the staleness
    floor shrunk and ``feed_timeout_bars=1`` on a 1-second timeframe the
    threshold is ~1 s, so the forced reconnect must arrive within the
    test's few-second budget.
    """
    from pynecore.core.syminfo import SymInfoInterval
    from datetime import datetime as ddatetime, time as dtime, timedelta, UTC

    now_dt = ddatetime.now(tz=UTC)
    open_start = now_dt - timedelta(seconds=60)
    if open_start.date() != now_dt.date():
        # Just after midnight: clamp the window to today so the single
        # same-day interval still covers "now".
        open_start = now_dt.replace(hour=0, minute=0, second=0)
    open_end = now_dt + timedelta(minutes=6)
    start_time = dtime(open_start.hour, open_start.minute, open_start.second)
    if open_end.date() != now_dt.date():
        # Just before midnight: split the window at the day boundary —
        # clamping the end to 23:59:59 would close the synthetic session
        # again seconds after the test starts, before the ~1 s staleness
        # trip can fire, and the run would never terminate. The post-
        # midnight segment starts at 00:00, so it cannot reach back to
        # the pinned slot ~139 s in the past (previous day).
        intervals = [
            SymInfoInterval(day=d, start=start_time, end=dtime(23, 59, 59))
            for d in range(7)
        ] + [
            SymInfoInterval(
                day=d, start=dtime(0, 0, 0),
                end=dtime(open_end.hour, open_end.minute, open_end.second),
            )
            for d in range(7)
        ]
    else:
        intervals = [
            SymInfoInterval(
                day=d, start=start_time,
                end=dtime(open_end.hour, open_end.minute, open_end.second),
            )
            for d in range(7)
        ]
    syminfo = _make_syminfo(intervals, timezone="UTC")

    # The synth slot (base_ts + 1s) lies ~79 s before the session window
    # opens -> calendar-closed, while wall-clock "now" is in-session.
    base_ts = int(time.time()) - 140
    updates = [_make_ohlcv(base_ts, is_closed=True, close=100.0)]
    provider = _SilentFeedProvider(updates, stop_after_connects=2)
    provider.feed_timeout_bars = 1

    from pynecore.core import live_runner as _live_runner_mod
    _orig_floor = _live_runner_mod._FEED_STALE_FLOOR_S
    _orig_sleep = _live_runner_mod._CLOSED_WINDOW_SLEEP_S
    _live_runner_mod._FEED_STALE_FLOOR_S = 0.2
    _live_runner_mod._CLOSED_WINDOW_SLEEP_S = 0.05
    try:
        _drain(provider, "TEST", "1S", syminfo=syminfo)
    finally:
        _live_runner_mod._FEED_STALE_FLOOR_S = _orig_floor
        _live_runner_mod._CLOSED_WINDOW_SLEEP_S = _orig_sleep

    assert provider.connect_calls >= 2, (
        "staleness watchdog must fire when the pinned synth slot is "
        "calendar-closed but the session is open now"
    )


# --- Forming-bar finalisation tests ---
#
# Providers that close a bar only when the NEXT bar's timestamp arrives
# (cTrader) never emit a close event for the last bar before a session
# boundary or a feed gap — that close simply never comes. The boundary
# watchdog must finalise the real forming bar it already received with
# its accumulated OHLCV instead of discarding it and fabricating a frozen
# V=0 synth. The no-forming-bar V=0 fallback (the watchdog's behaviour
# when no forming bar was tracked for the slot) is covered by
# ``__test_live_generator_synthesises_idle_bars_at_tf_boundary__``.


def __test_idle_watchdog_finalises_forming_bar_with_real_data__():
    """A forming bar for the boundary slot is closed with its real OHLCV.

    The provider delivers a closed bar then a forming (is_closed=False)
    bar for the next slot carrying real accumulated volume, then goes
    silent — no close event ever follows. The watchdog must emit a CLOSED
    bar at the forming slot with the forming bar's own OHLCV/volume, NOT
    a frozen V=0 filler at the previous close.
    """
    base_ts = int(time.time()) - 200
    forming = OHLCV(timestamp=base_ts + 60, open=100.0, high=106.0,
                    low=99.0, close=105.0, volume=500.0, is_closed=False)
    updates = [_make_ohlcv(base_ts, is_closed=True, close=100.0), forming]
    provider = _IdleAfterFirstBar(updates)

    bars: list[OHLCV] = []
    seen_transition = False
    for item in live_ohlcv_generator(provider, "BTC/USDT", "1"):
        if item is LIVE_TRANSITION:
            seen_transition = True
            continue
        if not seen_transition:
            continue
        bars.append(item)
        if any(b.timestamp == base_ts + 60 and b.is_closed for b in bars):
            break
        if len(bars) >= 8:
            break

    finalised = [b for b in bars if b.timestamp == base_ts + 60 and b.is_closed]
    assert len(finalised) == 1, (
        f"expected one finalised closed bar at the forming slot, "
        f"got {len(finalised)}"
    )
    assert finalised[0].close == 105.0  # real close, not the previous 100.0
    assert finalised[0].high == 106.0
    assert finalised[0].low == 99.0
    assert finalised[0].volume == 500.0  # real volume, not a V=0 synth


class _FormingThenLateClose(MockLiveProvider):
    """closed -> forming(next slot) -> stall (watchdog finalises) -> late
    real close for the same slot (conflicting values) -> exhausted.

    Models a provider whose own delayed close for a slot lands only after
    the watchdog already finalised that slot from the forming bar (e.g.
    the close event arrives after a session reopens).
    """

    def __init__(self, bar_updates: list[OHLCV], stall_seconds: float = 0.3):
        super().__init__(bar_updates)
        self._stall_seconds = stall_seconds
        self._stalled = False

    async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        # Stall once between the forming bar (index 2) and the late close
        # so the watchdog finalises the forming slot before it is served.
        if self._index == 2 and not self._stalled:
            self._stalled = True
            await asyncio.sleep(self._stall_seconds)
        return await super().watch_ohlcv(symbol, timeframe)


def __test_late_real_close_dropped_after_forming_finalisation__():
    """A provider's late close for an already-finalised slot is dropped.

    After the watchdog finalises the forming slot, the provider finally
    emits its own CLOSED bar for the same timestamp with conflicting
    values. The existing same-/older-timestamp dedup must drop it so the
    consumer keeps exactly one closed bar for the slot — the real forming
    data, not the late conflicting close.
    """
    base_ts = int(time.time()) - 200
    forming = OHLCV(timestamp=base_ts + 60, open=100.0, high=106.0,
                    low=99.0, close=105.0, volume=500.0, is_closed=False)
    late_close = OHLCV(timestamp=base_ts + 60, open=100.0, high=200.0,
                       low=50.0, close=999.0, volume=777.0, is_closed=True)
    updates = [_make_ohlcv(base_ts, is_closed=True, close=100.0),
               forming, late_close]
    provider = _FormingThenLateClose(updates, stall_seconds=0.3)

    bars: list[OHLCV] = []
    seen_transition = False
    for item in live_ohlcv_generator(provider, "BTC/USDT", "1"):
        if item is LIVE_TRANSITION:
            seen_transition = True
            continue
        if not seen_transition:
            continue
        bars.append(item)
        if len(bars) >= 8:
            break

    closed_at_slot = [b for b in bars
                      if b.timestamp == base_ts + 60 and b.is_closed]
    assert len(closed_at_slot) == 1, (
        f"late real close must be dropped; got {len(closed_at_slot)} "
        f"closed bars at the slot"
    )
    assert closed_at_slot[0].close == 105.0  # forming data wins
    assert closed_at_slot[0].volume == 500.0


def __test_forming_bar_finalised_even_when_soft_cap_drops_queue_updates__():
    """Forming tracking is independent of the intra-bar queue soft-cap.

    With the soft cap forced to zero every forming (is_closed=False)
    update is dropped from the consumer queue, so the consumer sees no
    intra-bar updates at all. The watchdog must still finalise the slot
    with the forming bar's real OHLCV, because finalisation state is
    tracked BEFORE the cap, not via queue admission.
    """
    base_ts = int(time.time()) - 200
    forming = OHLCV(timestamp=base_ts + 60, open=100.0, high=106.0,
                    low=99.0, close=105.0, volume=500.0, is_closed=False)
    updates = [_make_ohlcv(base_ts, is_closed=True, close=100.0), forming]
    provider = _IdleAfterFirstBar(updates)

    from pynecore.core import live_runner as _live_runner_mod
    _orig_cap = _live_runner_mod._INTRA_BAR_SOFT_CAP
    _live_runner_mod._INTRA_BAR_SOFT_CAP = 0
    try:
        bars: list[OHLCV] = []
        seen_transition = False
        for item in live_ohlcv_generator(provider, "BTC/USDT", "1"):
            if item is LIVE_TRANSITION:
                seen_transition = True
                continue
            if not seen_transition:
                continue
            bars.append(item)
            if any(b.timestamp == base_ts + 60 and b.is_closed for b in bars):
                break
            if len(bars) >= 8:
                break
    finally:
        _live_runner_mod._INTRA_BAR_SOFT_CAP = _orig_cap

    # The soft cap (0) dropped every forming update from the queue.
    assert all(b.is_closed for b in bars), (
        "soft cap=0 must drop all intra-bar (forming) updates from the queue"
    )
    # ...but the watchdog still finalised the slot with real data.
    finalised = [b for b in bars if b.timestamp == base_ts + 60 and b.is_closed]
    assert len(finalised) == 1
    assert finalised[0].close == 105.0
    assert finalised[0].volume == 500.0


def __test_forming_finalised_at_session_end_then_next_slot_skipped__():
    """Session-boundary case: last in-session slot finalises, next is skipped.

    Models the cTrader Friday-close bug: the last bar of the session is
    delivered as a forming bar (no close event ever follows, because the
    next bar only arrives after the weekend). The watchdog must finalise
    that slot with its real OHLCV (the session's true close), while the
    slot AFTER the session end is gated out — neither finalised nor
    frozen-synth.
    """
    from pynecore.core.syminfo import SymInfoInterval
    from datetime import datetime as ddatetime, time as dtime, UTC

    base_ts = int(time.time()) - 260
    synth_ts = base_ts + 60  # last in-session slot start
    # Session ends exactly at the slot boundary after ``synth_ts`` so
    # [synth_ts, synth_ts+60) is in-session and [synth_ts+60, +120) is not.
    start_dt = ddatetime.fromtimestamp(synth_ts - 1800, tz=UTC)
    end_dt = ddatetime.fromtimestamp(synth_ts + 60, tz=UTC)
    start_t = dtime(start_dt.hour, start_dt.minute, start_dt.second)
    end_t = dtime(end_dt.hour, end_dt.minute, end_dt.second)
    if start_dt.date() == end_dt.date():
        intervals = [SymInfoInterval(day=d, start=start_t, end=end_t)
                     for d in range(7)]
    else:
        # Window straddles UTC midnight: split at the day boundary so the
        # pre/post-midnight halves stay contiguous and the post-end slot
        # remains closed.
        intervals = [
            SymInfoInterval(day=d, start=start_t, end=dtime(23, 59, 59))
            for d in range(7)
        ] + [
            SymInfoInterval(day=d, start=dtime(0, 0, 0), end=end_t)
            for d in range(7)
        ]
    syminfo = _make_syminfo(intervals, timezone="UTC")

    forming = OHLCV(timestamp=synth_ts, open=100.0, high=106.0,
                    low=99.0, close=105.0, volume=500.0, is_closed=False)
    updates = [_make_ohlcv(base_ts, is_closed=True, close=100.0), forming]
    provider = _IdleThenCancelProvider(updates, max_idle_calls=20)

    from pynecore.core import live_runner as _live_runner_mod
    _orig_sleep = _live_runner_mod._CLOSED_WINDOW_SLEEP_S
    _live_runner_mod._CLOSED_WINDOW_SLEEP_S = 0.05
    try:
        _, bars = _drain(provider, "TEST", "1", syminfo=syminfo)
    finally:
        _live_runner_mod._CLOSED_WINDOW_SLEEP_S = _orig_sleep

    # The last in-session slot was finalised once with its real OHLCV.
    finalised = [b for b in bars if b.timestamp == synth_ts and b.is_closed]
    assert len(finalised) == 1, (
        f"last in-session slot must be finalised once, got {len(finalised)}"
    )
    assert finalised[0].close == 105.0
    assert finalised[0].volume == 500.0
    # The slot past the session end must not be emitted at all.
    assert all(b.timestamp != synth_ts + 60 for b in bars), (
        "the out-of-session slot must be neither finalised nor synthesised"
    )
    # No frozen V=0 synth was produced — the only closed-bar fill was the
    # real-data finalisation.
    assert all(b.volume > 0.0 for b in bars if b.is_closed), (
        "no frozen V=0 synth expected; only the real-data finalisation"
    )
