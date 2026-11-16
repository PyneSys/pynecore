"""
A live lower-timeframe window whose final intrabar the venue publishes after the
stream grace. The window of a closed chart bar is history once published, so the
collect keeps waiting past the grace and asks the provider's history for the closed
intrabars after its newest one, through the stream's own closed-bar cursor: the final
intrabar from either side completes the window, a later intrabar instead publishes it
at once as a gap, and the evidence budget bounds the wait when nothing comes.
"""
import asyncio
import logging
import threading
import time

import pytest

from pynecore.core import security_process
from pynecore.core.live_runner import LiveBarStreamer
# noinspection PyProtectedMember
from pynecore.core.security_process import _LateCloseRetry, _collect_in_period_intrabars
from pynecore.types.ohlcv import OHLCV

_MINUTE_MS = 60_000
# A three-intrabar chart period of one-minute intrabars, its final intrabar opening
# near the wall clock, so the stream's boundary watchdog stays far away
_FINAL = (int(time.time() * 1000) // _MINUTE_MS - 1) * _MINUTE_MS
_PERIOD_START = _FINAL - 2 * _MINUTE_MS
_PERIOD_END = _FINAL + _MINUTE_MS
_NEXT = _PERIOD_END

_GRACE = 0.3
_BUDGET = 1.5


def _bar(ts: int, close: float) -> OHLCV:
    return OHLCV(timestamp=ts, open=close, high=close, low=close, close=close,
                 volume=1.0, is_closed=True)


_FIRST = _bar(_PERIOD_START, 100.0)
_SECOND = _bar(_PERIOD_START + _MINUTE_MS, 101.0)


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
    """The window collect of a live LTF context, wired the way the child wires it."""

    def __init__(self, venue: _Venue):
        self.venue = venue
        self.streamer = LiveBarStreamer(venue, "X", "1")  # type: ignore[arg-type]
        self.buffer: list[OHLCV] = [_FIRST, _SECOND]
        self.next_idx = 0
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

    def collect(self, period_end: int, *, retry: bool = True, developing: bool = False,
                stopped: 'threading.Event | None' = None) -> tuple[list[OHLCV], float]:
        """Run the collect of the window ending at ``period_end``, as the round does.

        :return: ``(window, seconds the collect took)``.
        """
        started = time.monotonic()
        for bar in self.streamer.pop_new_closed_bars():
            self.ingest(bar)
        window, self.next_idx = _collect_in_period_intrabars(
            self.buffer, self.next_idx, period_end, _MINUTE_MS, developing, _GRACE,
            self.streamer.wait_for_bars, self.ingest,
            retry=self.retry if retry else None,
            stopped=None if stopped is None else stopped.is_set)
        return window, time.monotonic() - started


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch: pytest.MonkeyPatch):
    """History queries a tenth of a second apart, so a test runs in about a second"""
    monkeypatch.setattr(security_process, '_CLOSE_RETRY_FIRST_DELAY', 0.1)
    monkeypatch.setattr(security_process, '_CLOSE_RETRY_MAX_DELAY', 0.2)


def __test_late_final_intrabar_from_history_completes_the_window__():
    """The socket never delivers the final intrabar; it comes from the history after the grace"""
    venue = _Venue([_bar(_FINAL, 102.0)])
    with _Context(venue) as ctx:
        window, took = ctx.collect(_PERIOD_END)
    assert [b.timestamp for b in window] == [_PERIOD_START, _PERIOD_START + _MINUTE_MS, _FINAL]
    assert window[-1].close == 102.0
    assert took >= _GRACE
    assert venue.history_calls == [_SECOND.timestamp]
    assert ctx.warnings == []


def __test_without_the_retry_the_window_ends_at_the_grace__():
    """The same venue without the retry: the window is published without its final intrabar"""
    venue = _Venue([_bar(_FINAL, 102.0)])
    with _Context(venue) as ctx:
        window, took = ctx.collect(_PERIOD_END, retry=False)
    assert window == [_FIRST, _SECOND]
    assert venue.history_calls == []
    assert took < _BUDGET


def __test_socket_final_intrabar_during_the_retry_is_consumed_once__():
    """The history knows nothing yet; the socket's close arrives during the retry

    It completes the window at once, the next period's intrabar stays for the next
    round, and the history answering with the same slots afterwards does not hand
    them over a second time.
    """
    venue = _Venue([])
    with _Context(venue) as ctx:
        def _socket_after_first_query() -> None:
            venue.history_called.wait(5.0)
            venue.push(_bar(_FINAL, 102.0))
            venue.push(_bar(_NEXT, 103.0))

        pusher = threading.Thread(target=_socket_after_first_query)
        pusher.start()
        window, took = ctx.collect(_PERIOD_END)
        pusher.join()
        assert [b.timestamp for b in window] == [_PERIOD_START, _PERIOD_START + _MINUTE_MS,
                                                 _FINAL]
        assert window[-1].close == 102.0
        assert took < _BUDGET
        settle_by = time.monotonic() + 5.0
        while len(ctx.buffer) < 4 and time.monotonic() < settle_by:
            for streamed in ctx.streamer.wait_for_bars(0.5):
                ctx.ingest(streamed)
        # A later history answer with the same slots reaches nobody
        venue.answer = [_bar(_FINAL, 999.0), _bar(_NEXT, 999.0)]
        follow_up = ctx.streamer.request_backfill(_SECOND.timestamp)
        assert follow_up is not None and follow_up.result(timeout=5.0) == 0
        assert ctx.streamer.wait_for_bars(0.5) == []
    assert [b.timestamp for b in ctx.buffer] == [_PERIOD_START, _PERIOD_START + _MINUTE_MS,
                                                 _FINAL, _NEXT]
    assert [b.close for b in ctx.buffer[2:]] == [102.0, 103.0]
    assert ctx.buffer[ctx.next_idx].timestamp == _NEXT


def __test_later_intrabar_without_the_final_one_is_a_gap_at_once__(caplog):
    """The history has the next period's intrabar but not the final one: no trades there

    The window is published without it right after the first query, without waiting
    further, and the later intrabar stays for the next round.
    """
    venue = _Venue([_bar(_NEXT, 103.0)])
    with _Context(venue) as ctx:
        with caplog.at_level(logging.WARNING, logger="pynecore.core.security_process"):
            window, took = ctx.collect(_PERIOD_END)
    assert window == [_FIRST, _SECOND]
    assert ctx.buffer[ctx.next_idx].timestamp == _NEXT
    assert venue.history_calls == [_SECOND.timestamp]
    assert took < _BUDGET
    assert ctx.warnings == []
    assert "missing its final intrabar" in caplog.text


def __test_nothing_arrives_the_budget_bounds_the_wait__():
    """Neither side ever has the final intrabar: published after the budget, with one warning

    The next window of the same hole does not wait the budget again, and does not
    warn again.
    """
    venue = _Venue([])
    with _Context(venue) as ctx:
        window, took = ctx.collect(_PERIOD_END)
        assert window == [_FIRST, _SECOND]
        assert _BUDGET <= took < _BUDGET + 1.0
        assert ctx.warnings == [(_FINAL, _BUDGET)]
        assert len(venue.history_calls) >= 2

        queries = len(venue.history_calls)
        window, took = ctx.collect(_PERIOD_END + 3 * _MINUTE_MS)
        assert window == []
        assert took < _GRACE + 0.5
        assert ctx.warnings == [(_FINAL, _BUDGET)]
        assert len(venue.history_calls) == queries


def __test_shutdown_ends_the_wait__(monkeypatch: pytest.MonkeyPatch):
    """A stop during the retry ends the collect within one wait slice"""
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
        window, took = ctx.collect(_PERIOD_END, stopped=stop)
        stopper.join()
    assert window == [_FIRST, _SECOND]
    assert took < _GRACE + 0.5
    assert ctx.warnings == []


def __test_developing_window_never_asks_the_history__():
    """A forming chart bar publishes its partial window at once"""
    venue = _Venue([_bar(_FINAL, 102.0)])
    with _Context(venue) as ctx:
        window, took = ctx.collect(_PERIOD_END, developing=True)
    assert window == [_FIRST, _SECOND]
    assert took < _GRACE
    assert venue.history_calls == []
