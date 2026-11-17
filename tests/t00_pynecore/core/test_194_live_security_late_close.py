"""
A live ``request.security`` round whose target close the venue publishes after the
stream grace. The round keeps waiting past the grace and asks the provider's history
for the closed bars after its newest one, through the stream's own closed-bar cursor:
the target from either side ends the round with its real value, a later bar instead
ends it at once as a gap, and an evidence budget bounds the wait when nothing comes.
"""
import asyncio
import threading
import time

import pytest

from pynecore.core import security_process
from pynecore.core.live_runner import LiveBarStreamer
# noinspection PyProtectedMember
from pynecore.core.security_process import (_LateCloseRetry, _close_evidence_budget,
                                            _read_round_bar)
from pynecore.types.ohlcv import OHLCV

_MINUTE_MS = 60_000
# Bar opens near the wall clock, so the stream's boundary watchdog stays far away
_T1 = (int(time.time() * 1000) // _MINUTE_MS - 1) * _MINUTE_MS
_T0 = _T1 - _MINUTE_MS
_T2 = _T1 + _MINUTE_MS

_GRACE = 0.3
_BUDGET = 1.5


def _bar(ts: int, close: float) -> OHLCV:
    return OHLCV(timestamp=ts, open=close, high=close, low=close, close=close,
                 volume=1.0, is_closed=True)


class _Venue:
    """A live provider whose socket and history the test drives.

    The socket delivers what :meth:`push` hands it; the history answers every query
    with :attr:`~_Venue.answer`.
    """

    reconnect_delay = 0.01
    max_reconnect_delay = 0.05
    feed_timeout_bars = None
    initial_connect_timeout = 5.0

    def __init__(self, answer: list[OHLCV]):
        self.answer = answer
        self.history_calls: list[int] = []
        self.history_called = threading.Event()
        self._socket: asyncio.Queue | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._connected = False

    async def connect(self):
        self._loop = asyncio.get_running_loop()
        self._socket = asyncio.Queue()
        self._connected = True

    async def disconnect(self):
        self._connected = False

    @property
    def is_connected(self):
        return self._connected

    async def watch_ohlcv(self, _symbol: str, _timeframe: str) -> OHLCV:
        assert self._socket is not None
        return await self._socket.get()

    def push(self, bar: OHLCV) -> None:
        assert self._loop is not None and self._socket is not None
        self._loop.call_soon_threadsafe(self._socket.put_nowait, bar)

    async def backfill_closed_bars(self, _symbol: str, _timeframe: str,
                                   since_ms: int) -> list[OHLCV]:
        self.history_calls.append(since_ms)
        self.history_called.set()
        return list(self.answer)

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        return symbol

    async def on_disconnect(self):
        pass

    async def on_reconnect(self):
        pass

    @staticmethod
    async def can_shutdown():
        return True


class _Context:
    """The closed-round read of a live context, wired the way the child wires it."""

    def __init__(self, venue: _Venue, *, buffer: list[OHLCV] | None = None,
                 boundary_watchdog: bool = False):
        self.venue = venue
        self.streamer = LiveBarStreamer(venue, "X", "1",  # type: ignore[arg-type]
                                        boundary_watchdog=boundary_watchdog)
        self.buffer: list[OHLCV] = [_bar(_T0, 100.0)] if buffer is None else buffer
        self.warnings: list[tuple[int, float]] = []
        self.retry = _LateCloseRetry(
            lambda: self.streamer.request_backfill(self.buffer[-1].timestamp),
            _GRACE, _BUDGET, lambda target, budget: self.warnings.append((target, budget)))

    def __enter__(self) -> '_Context':
        self.streamer.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.streamer.stop()

    def ingest(self, bar: OHLCV) -> None:
        self.retry.bar_arrived()
        self.buffer.append(bar)

    def _read(self, idx: int) -> OHLCV | None:
        for bar in self.streamer.pop_new_closed_bars():
            self.ingest(bar)
        return self.buffer[idx] if idx < len(self.buffer) else None

    def read_round(self, target: int, *, retry: bool = True,
                   stopped: 'threading.Event | None' = None) -> tuple[OHLCV | None, float]:
        """Run the round's read of the bar after the buffered ones.

        :return: ``(bar, seconds the read took)``.
        """
        started = time.monotonic()
        bar = _read_round_bar(self._read, len(self.buffer), target, None,
                              started + _GRACE, self.streamer.wait_for_bars, self.ingest,
                              retry=self.retry if retry else None,
                              stopped=None if stopped is None else stopped.is_set)
        return bar, time.monotonic() - started


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch: pytest.MonkeyPatch):
    """History queries a tenth of a second apart, so a test runs in about a second"""
    monkeypatch.setattr(security_process, '_CLOSE_RETRY_FIRST_DELAY', 0.1)
    monkeypatch.setattr(security_process, '_CLOSE_RETRY_MAX_DELAY', 0.2)


def __test_late_close_from_history_completes_the_round__():
    """The socket never delivers the target; its close comes from the history after the grace"""
    venue = _Venue([_bar(_T1, 111.0)])
    with _Context(venue) as ctx:
        bar, took = ctx.read_round(_T1)
    assert bar is not None and bar.timestamp == _T1 and bar.close == 111.0
    assert took >= _GRACE
    assert venue.history_calls == [_T0]
    assert ctx.warnings == []


def __test_without_the_retry_the_round_ends_at_the_grace__():
    """The same venue without the retry: the round has no bar and would publish ``na``"""
    venue = _Venue([_bar(_T1, 111.0)])
    with _Context(venue) as ctx:
        bar, took = ctx.read_round(_T1, retry=False)
    assert bar is None
    assert venue.history_calls == []
    assert took < _BUDGET


def __test_socket_close_during_the_retry_is_consumed_once__():
    """The history knows nothing yet; the socket's close arrives during the retry

    It ends the wait at once, and the history answering with the same slot
    afterwards does not hand it over a second time.
    """
    venue = _Venue([])
    with _Context(venue) as ctx:
        def _socket_after_first_query() -> None:
            venue.history_called.wait(5.0)
            venue.push(_bar(_T1, 111.0))
            venue.push(_bar(_T2, 112.0))

        pusher = threading.Thread(target=_socket_after_first_query)
        pusher.start()
        bar, took = ctx.read_round(_T1)
        pusher.join()
        assert bar is not None and bar.timestamp == _T1 and bar.close == 111.0
        assert took < _BUDGET
        settle_by = time.monotonic() + 5.0
        while len(ctx.buffer) < 3 and time.monotonic() < settle_by:
            for streamed in ctx.streamer.wait_for_bars(0.5):
                ctx.ingest(streamed)
        # A later history answer with the same slots reaches nobody
        venue.answer = [_bar(_T1, 999.0), _bar(_T2, 999.0)]
        follow_up = ctx.streamer.request_backfill(_T0)
        assert follow_up is not None and follow_up.result(timeout=5.0) == 0
        assert ctx.streamer.wait_for_bars(0.5) == []
    assert [b.timestamp for b in ctx.buffer] == [_T0, _T1, _T2]
    assert [b.close for b in ctx.buffer[1:]] == [111.0, 112.0]


def __test_later_bar_without_the_target_is_a_gap_at_once__():
    """The history has the next bar but not the target: no trades in that period

    The round gets the later bar right after the first query and ends there (the
    caller treats a bar past its target as the gap), without waiting further.
    """
    venue = _Venue([_bar(_T2, 112.0)])
    with _Context(venue) as ctx:
        bar, took = ctx.read_round(_T1)
    assert bar is not None and bar.timestamp == _T2
    assert bar.timestamp > _T1
    assert venue.history_calls == [_T0]
    assert took < _BUDGET
    assert ctx.warnings == []


def __test_nothing_arrives_the_budget_bounds_the_wait__():
    """Neither side ever has the target: the round ends after the budget, with one warning

    The next round of the same hole does not wait the budget again, and does not
    warn again.
    """
    venue = _Venue([])
    with _Context(venue) as ctx:
        bar, took = ctx.read_round(_T1)
        assert bar is None
        assert _BUDGET <= took < _BUDGET + 1.0
        assert ctx.warnings == [(_T1, _BUDGET)]
        assert len(venue.history_calls) >= 2

        queries = len(venue.history_calls)
        bar, took = ctx.read_round(_T2)
        assert bar is None
        assert took < _GRACE + 0.5
        assert ctx.warnings == [(_T1, _BUDGET)]
        assert len(venue.history_calls) == queries


def __test_shutdown_ends_the_wait__(monkeypatch: pytest.MonkeyPatch):
    """A stop during the retry ends the round within one wait slice"""
    monkeypatch.setattr(security_process, '_ROUND_WAIT_SLICE_SECONDS', 0.05)
    monkeypatch.setattr(security_process, '_CLOSE_RETRY_FIRST_DELAY', 10.0)
    venue = _Venue([])
    stop = threading.Event()
    with _Context(venue) as ctx:
        def _stop_after_first_query() -> None:
            venue.history_called.wait(5.0)
            stop.set()

        stopper = threading.Thread(target=_stop_after_first_query)
        stopper.start()
        bar, took = ctx.read_round(_T1, stopped=stop)
        stopper.join()
    assert bar is None
    assert took < _GRACE + 0.5
    assert ctx.warnings == []


def __test_helper_overdue_target(boundary_watchdog: bool) -> OHLCV | None:
    """The round's bar for a target long overdue, after the previous close came via the socket.

    The socket's close arms the stream's closed-bar cursor, and the target's close
    lies minutes in the past, so a stream with the boundary watchdog settles it at once.
    """
    s0 = _T0 - 4 * _MINUTE_MS
    s1 = s0 + _MINUTE_MS
    venue = _Venue([_bar(s1, 111.0)])
    with _Context(venue, buffer=[], boundary_watchdog=boundary_watchdog) as ctx:
        venue.push(_bar(s0, 100.0))
        previous, _took = ctx.read_round(s0)
        assert previous is not None and previous.timestamp == s0
        bar, _took = ctx.read_round(s1)
    assert bar is not None and bar.timestamp == s1
    return bar


def __test_security_stream_leaves_a_late_close_to_the_history__():
    """No filler stands in for a target the stream is late with: the history's close does"""
    bar = __test_helper_overdue_target(boundary_watchdog=False)
    assert bar is not None and (bar.close, bar.volume) == (111.0, 1.0)


def __test_boundary_watchdog_settles_the_late_close_from_the_history__():
    """The same feed through a stream with the boundary watchdog: its probe gets the close"""
    bar = __test_helper_overdue_target(boundary_watchdog=True)
    assert bar is not None and (bar.close, bar.volume) == (111.0, 1.0)


def __test_arriving_bar_ends_the_hole__():
    """A closed bar after a spent budget re-arms the full budget for the next miss"""
    retry = _LateCloseRetry(lambda: None, _GRACE, _BUDGET, lambda *_args: None)
    assert retry.deadline(10.0) == 10.0 - _GRACE + _BUDGET
    # The same hole keeps its start
    assert retry.deadline(20.0) == 10.0 - _GRACE + _BUDGET
    retry.bar_arrived()
    assert retry.deadline(30.0) == 30.0 - _GRACE + _BUDGET


def __test_evidence_budget_follows_the_timeframe__():
    """A quarter of the bar, never under two minutes and never over ten"""
    assert _close_evidence_budget("1") == 120.0
    assert _close_evidence_budget("5") == 120.0
    assert _close_evidence_budget("15") == 225.0
    assert _close_evidence_budget("30") == 450.0
    assert _close_evidence_budget("60") == 600.0
    assert _close_evidence_budget("1D") == 600.0
