"""
Closed-bar recovery of the chart's live stream, owned by the framework.

The chart's stream has the close its socket missed asked from the provider's history
once the venue has published it, the gap of an outage filled after every reconnect
before the new subscription's bars, and the subscription reconnected once closes keep
coming only from the history. A forming bar seen before an outage is not taken for the
slot's close: the history or a fresh socket close settles it. A history query that
fails is never taken for "no bar": the stream waits for an answer instead of going on
over the gap. A slot that had no trades gets the bar the venue's history has for it: a
zero-volume filler, or none on a venue whose history skips such slots.
"""
import asyncio
import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from types import SimpleNamespace

from pynecore.core import live_runner
from pynecore.core.live_runner import LiveStop, live_ohlcv_generator
from pynecore.core.script_runner import LIVE_TRANSITION
from pynecore.core.syminfo import SymInfo, SymInfoInterval
from pynecore.types.ohlcv import OHLCV

_TF_MS = 60_000


def _make_syminfo(opening_hours: list[SymInfoInterval]) -> SymInfo:
    """A UTC symbol with the given weekly session calendar; the rest are throw-away values."""
    return SymInfo(prefix="TEST", description="test", ticker="TEST", currency="USD",
                   period="1", type="crypto", mintick=0.01, pricescale=100, minmove=1,
                   pointvalue=1.0, mincontract=1.0, opening_hours=opening_hours,
                   session_starts=[], session_ends=[], timezone="UTC")


def _bar(timestamp: int, close: float = 100.0, *, volume: float = 10.0,
         is_closed: bool = True) -> OHLCV:
    return OHLCV(timestamp=timestamp, open=close, high=close + 1.0, low=close - 1.0,
                 close=close, volume=volume, is_closed=is_closed)


def _now_ms() -> int:
    return int(time.time() * 1000)


class _Venue:
    """A venue whose socket replays one script per connection and whose history the test sets.

    The socket hands out its script's items in order (an exception is raised from
    ``watch_ohlcv``), then stays silent while the connection looks healthy. The history
    returns the bars of :attr:`~_Venue.history` after the requested open, and fails while
    :attr:`~_Venue.history_failures` is positive. With ``publish_delay`` a bar shows up
    in the history only that many seconds after its close, like a venue that publishes
    its closes late. With ``stream_closes_final`` false the socket's candle is not the
    history's
    (:attr:`~LiveProviderPlugin.stream_closes_match_history`),
    with ``revises`` the history revises a fresh bar within the lag
    (:attr:`~LiveProviderPlugin.history_revises_before_lag`).
    """

    reconnect_delay = 0.01
    max_reconnect_delay = 0.05
    feed_timeout_bars = None

    def __init__(self, scripts: list[list[OHLCV | BaseException]], *, idle_bars: bool = True,
                 publish_lag: float = 0.0, publish_delay: float | None = None,
                 stream_closes_final: bool = True, revises: bool = False):
        self.history_has_idle_bars = idle_bars
        self.history_publish_lag = publish_lag
        self.stream_closes_match_history = stream_closes_final
        self.history_revises_before_lag = revises
        self.publish_delay = publish_delay
        self.scripts = scripts
        self.history: list[OHLCV] = []
        self.history_failures = 0
        self.history_calls: list[tuple[float, int]] = []
        self.connects = 0
        self.watch_calls = 0
        self._cond = threading.Condition()
        self._connected = False
        self._socket: asyncio.Queue | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def connect(self):
        self._loop = asyncio.get_running_loop()
        socket: asyncio.Queue = asyncio.Queue()
        with self._cond:
            self.connects += 1
            script = self.scripts[self.connects - 1] if self.connects <= len(self.scripts) else []
            self._cond.notify_all()
        for item in script:
            socket.put_nowait(item)
        self._socket = socket
        self._connected = True

    async def disconnect(self):
        self._connected = False

    @property
    def is_connected(self):
        return self._connected

    async def watch_ohlcv(self, _symbol: str, _timeframe: str) -> OHLCV:
        assert self._socket is not None
        with self._cond:
            self.watch_calls += 1
            self._cond.notify_all()
        item = await self._socket.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def backfill_closed_bars(self, _symbol: str, _timeframe: str,
                                   since_ms: int) -> list[OHLCV]:
        with self._cond:
            self.history_calls.append((time.time(), since_ms))
            failing = self.history_failures > 0
            if failing:
                self.history_failures -= 1
            answer = [bar for bar in self.history if bar.timestamp > since_ms]
            if self.publish_delay is not None:
                now = time.time()
                answer = [bar for bar in answer
                          if (bar.timestamp + _TF_MS) / 1000.0 + self.publish_delay <= now]
            self._cond.notify_all()
        if failing:
            raise ConnectionError("history endpoint down")
        return answer

    def set_history(self, bars: list[OHLCV], *, failures: int = 0) -> None:
        with self._cond:
            self.history = bars
            self.history_failures = failures

    def push(self, item: OHLCV) -> None:
        assert self._loop is not None and self._socket is not None
        self._loop.call_soon_threadsafe(self._socket.put_nowait, item)

    def wait_for(self, predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
        with self._cond:
            return self._cond.wait_for(predicate, timeout)

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


class _Consumer:
    """Pulls a live stream on its own thread and records the updates and closed bars it yields."""

    def __init__(self, venue: _Venue, timeframe: str = "1", **kwargs):
        self.stop = LiveStop()
        self.closed: list[OHLCV] = []
        self.updates: list[OHLCV] = []
        self.live = False
        self.error: BaseException | None = None
        self._cond = threading.Condition()
        self._stream = live_ohlcv_generator(venue, "BTC/USDT", timeframe, stop=self.stop,
                                            shutdown_timeout=5.0, **kwargs)
        self._thread = threading.Thread(target=self._pull, daemon=True)
        self._thread.start()

    def _pull(self) -> None:
        try:
            for item in self._stream:
                if item is LIVE_TRANSITION:
                    with self._cond:
                        self.live = True
                        self._cond.notify_all()
                    continue
                with self._cond:
                    self.updates.append(item)
                    if item.is_closed:
                        self.closed.append(item)
                    self._cond.notify_all()
        except BaseException as exc:  # noqa: BLE001 - surfaced by the test
            self.error = exc

    def wait_for(self, predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
        with self._cond:
            return self._cond.wait_for(predicate, timeout)

    def stamps(self) -> list[int]:
        with self._cond:
            return [bar.timestamp for bar in self.closed]

    def update_keys(self) -> list[tuple[int, float, bool]]:
        """Timestamp, close and closed flag of every update yielded so far."""
        with self._cond:
            return [(bar.timestamp, bar.close, bar.is_closed) for bar in self.updates]

    def close(self) -> None:
        self.stop.set()
        self._thread.join(timeout=10.0)
        assert not self._thread.is_alive(), "the stream did not stop"
        assert self.error is None, f"the stream raised: {self.error!r}"


def __test_missed_close_comes_from_history_after_the_publish_lag__():
    """The close the socket missed is asked once the lag past its close is over

    The forming bar of the slot is dropped for the history's close, and the query
    waits for the publish lag instead of the boundary deadline half a minute later.
    """
    lag = 0.4
    last = _now_ms() - 2 * _TF_MS + 300
    slot = last + _TF_MS
    venue = _Venue([[_bar(last), _bar(slot, 150.0, volume=3.0, is_closed=False)]],
                   publish_lag=lag)
    venue.set_history([_bar(slot, 155.0, volume=12.0)])
    consumer = _Consumer(venue)
    try:
        assert consumer.wait_for(lambda: len(consumer.closed) >= 2, timeout=10.0)
        recovered = consumer.closed[1]
        assert (recovered.timestamp, recovered.close, recovered.volume) == (slot, 155.0, 12.0)
        due = (slot + _TF_MS) / 1000.0 + lag
        first_call, since = venue.history_calls[0]
        assert since == last
        assert due - 0.05 <= first_call <= due + 1.5, (
            f"history asked {first_call - due:+.2f}s from its due time")
    finally:
        consumer.close()


def __test_failed_query_holds_the_slot_until_the_history_answers__():
    """A failed query is not a "no bar" answer: the overdue slot is neither filled nor skipped

    Once the history answers, its bar is the slot's close; the next overdue slot,
    answered empty on a venue with idle bars, gets the zero-volume filler.
    """
    last = (_now_ms() // _TF_MS) * _TF_MS - 4 * _TF_MS
    venue = _Venue([[_bar(last)]])
    venue.set_history([], failures=1_000_000)
    consumer = _Consumer(venue)
    try:
        assert venue.wait_for(lambda: len(venue.history_calls) >= 4)
        assert consumer.stamps() == [last], "a slot was settled without a history answer"
        venue.set_history([_bar(last + _TF_MS, 120.0)])
        assert consumer.wait_for(lambda: len(consumer.closed) >= 3)
        recovered, filler = consumer.closed[1], consumer.closed[2]
        assert (recovered.timestamp, recovered.close, recovered.volume) == (
            last + _TF_MS, 120.0, 10.0)
        assert (filler.timestamp, filler.close, filler.volume) == (
            last + 2 * _TF_MS, 120.0, 0.0)
    finally:
        consumer.close()


def __test_a_later_socket_close_waits_behind_an_unanswered_slot__():
    """The socket's close of a later slot does not move the cursor past a slot still unanswered

    Handed on at once, it would make the slot's history close a bar behind the cursor,
    which could then never be spliced in.
    """
    last = (_now_ms() // _TF_MS) * _TF_MS - 4 * _TF_MS
    s1, s2 = last + _TF_MS, last + 2 * _TF_MS
    venue = _Venue([[_bar(last)]])
    venue.set_history([], failures=1_000_000)
    consumer = _Consumer(venue)
    try:
        assert venue.wait_for(lambda: len(venue.history_calls) >= 2)
        venue.push(_bar(s2, 102.0))
        assert venue.wait_for(lambda: len(venue.history_calls) >= 5)
        assert consumer.stamps() == [last], "a later close went on over the unanswered slot"
        venue.set_history([_bar(s1, 101.0)])
        assert consumer.wait_for(lambda: len(consumer.closed) >= 3)
        assert [(bar.timestamp, bar.close) for bar in consumer.closed[:3]] == [
            (last, 100.0), (s1, 101.0), (s2, 102.0)]
        assert all(since == last for _, since in venue.history_calls[:5])
    finally:
        consumer.close()


def __test_a_later_forming_bar_waits_behind_an_unanswered_slot__():
    """An update of a later slot is held too: the consumer's clock would pass the slot"""
    last = (_now_ms() // _TF_MS) * _TF_MS - 4 * _TF_MS
    s1, s2 = last + _TF_MS, last + 2 * _TF_MS
    venue = _Venue([[_bar(last)]])
    venue.set_history([], failures=1_000_000)
    consumer = _Consumer(venue)
    try:
        assert venue.wait_for(lambda: len(venue.history_calls) >= 2)
        venue.push(_bar(s2, 102.0, is_closed=False))
        venue.push(_bar(s2, 103.0, is_closed=False))
        assert venue.wait_for(lambda: len(venue.history_calls) >= 5)
        assert [key[0] for key in consumer.update_keys()] == [last], \
            "a later forming bar went on over the unanswered slot"
        venue.set_history([_bar(s1, 101.0)])
        assert consumer.wait_for(lambda: len(consumer.updates) >= 3)
        # Only the newest update of the held forming slot goes on
        assert consumer.update_keys()[:3] == [(last, 100.0, True), (s1, 101.0, True), (s2, 103.0, False)]
    finally:
        consumer.close()


def __test_a_held_close_after_a_session_break_goes_on_once_the_gap_is_filled__():
    """A held close past a session break is replayed when the last in-session slot is settled

    The slot after it is in the closed session, so it is never settled; with no further
    socket update the held close still has to go on.
    """
    last = (_now_ms() // _TF_MS) * _TF_MS - 4 * _TF_MS
    s1, s2, s3 = last + _TF_MS, last + 2 * _TF_MS, last + 3 * _TF_MS
    # A daily one-minute break at s2: each day's session runs from s3 overnight to s2
    reopen = datetime.fromtimestamp(s3 / 1000.0, tz=UTC).time()
    break_start = datetime.fromtimestamp(s2 / 1000.0, tz=UTC).time()
    syminfo = _make_syminfo([SymInfoInterval(day=d, start=reopen, end=break_start)
                             for d in range(7)])
    venue = _Venue([[_bar(last)]])
    venue.set_history([], failures=1_000_000)
    consumer = _Consumer(venue, syminfo=syminfo)
    try:
        assert venue.wait_for(lambda: len(venue.history_calls) >= 2)
        venue.push(_bar(s3, 103.0))
        assert venue.wait_for(lambda: len(venue.history_calls) >= 4)
        assert consumer.stamps() == [last], "a post-break close went on over the unanswered slot"
        venue.set_history([_bar(s1, 101.0)])
        assert consumer.wait_for(lambda: len(consumer.closed) >= 3), \
            "the held post-break close stayed stuck behind the closed-session slot"
        assert [(bar.timestamp, bar.close) for bar in consumer.closed] == [
            (last, 100.0), (s1, 101.0), (s3, 103.0)]
    finally:
        consumer.close()


def __test_idle_slots_stay_empty_on_a_venue_without_idle_bars__():
    """No filler where the venue's history has no bar: the cursor waits for the next real close

    Each overdue slot costs at most one query, and the watchdog does not spin on a
    deadline it cannot settle.
    """
    last = _now_ms() - 4 * _TF_MS - 40_000
    next_real = last + 4 * _TF_MS
    venue = _Venue([[_bar(last)]], idle_bars=False)
    consumer = _Consumer(venue)
    try:
        assert venue.wait_for(lambda: len(venue.history_calls) >= 3)
        # The three overdue slots are settled; the fourth closes later
        baseline = venue.watch_calls
        assert not venue.wait_for(lambda: venue.watch_calls > baseline + 10, timeout=1.0), \
            "the watchdog spins on an overdue slot it left empty"
        venue.push(_bar(next_real, 140.0))
        assert consumer.wait_for(lambda: len(consumer.closed) >= 2)
        assert consumer.stamps() == [last, next_real]
        assert all(bar.volume > 0.0 for bar in consumer.closed)
        assert len(venue.history_calls) == 3
        assert all(since == last for _, since in venue.history_calls)
    finally:
        consumer.close()


# A publish lag longer than the one-minute timeframe's 30 s close grace, and a slot
# whose close the history publishes ``_LATE_PUBLISH_IN`` seconds from now: its
# deadline is over, the end of the publish lag is just ahead.
_SLOW_LAG = 45.0
_LATE_PUBLISH_IN = 1.5


def _slow_venue_slot() -> tuple[int, float]:
    """Open of the newest closed bar and the moment the next slot's lag ends, in epoch s."""
    s1_close = _now_ms() - int((_SLOW_LAG - _LATE_PUBLISH_IN) * 1000)
    return s1_close - 2 * _TF_MS, s1_close / 1000.0 + _SLOW_LAG


def _until(moment: float) -> float:
    """Seconds left until ``moment`` (epoch s), with a margin for the stream's own pace."""
    return max(0.0, moment - time.time() - 0.2)


def __test_slow_history_bar_is_awaited_and_later_updates_wait_behind_it__():
    """A slot without a forming bar waits out the publish lag, not just the close grace

    The history answers empty at the deadline, before the venue published the close.
    That is not "no bar": the slot waits, the later slot's updates are held behind it,
    and once the lag is over the history's bar goes on, then the held update.
    """
    last, lag_end = _slow_venue_slot()
    s1, s2 = last + _TF_MS, last + 2 * _TF_MS
    venue = _Venue([[_bar(last), _bar(s2, 102.0, is_closed=False),
                     _bar(s2, 103.0, is_closed=False)]],
                   idle_bars=False, publish_lag=_SLOW_LAG, publish_delay=_SLOW_LAG - 1.0)
    venue.set_history([_bar(s1, 101.0)])
    consumer = _Consumer(venue)
    try:
        assert venue.wait_for(lambda: len(venue.history_calls) >= 1)
        assert not consumer.wait_for(lambda: len(consumer.updates) > 1,
                                     timeout=_until(lag_end)), \
            "the slot was settled on an answer asked before the venue published its close"
        assert consumer.wait_for(lambda: len(consumer.updates) >= 3)
        assert consumer.update_keys() == [
            (last, 100.0, True), (s1, 101.0, True), (s2, 103.0, False)]
        assert venue.history_calls[-1][0] >= lag_end - 0.05
        assert len(venue.history_calls) <= 3, "the history is flooded while the slot waits"
    finally:
        consumer.close()


def __test_slow_history_without_the_bar_after_the_lag_leaves_a_hole__():
    """An empty answer asked after the publish lag is final: the slot stays empty

    On a venue without idle bars the held update of the next slot then goes on, and
    so does that slot's close.
    """
    last, lag_end = _slow_venue_slot()
    s2 = last + 2 * _TF_MS
    venue = _Venue([[_bar(last), _bar(s2, 103.0, is_closed=False)]],
                   idle_bars=False, publish_lag=_SLOW_LAG)
    consumer = _Consumer(venue)
    try:
        assert venue.wait_for(lambda: len(venue.history_calls) >= 1)
        assert not consumer.wait_for(lambda: len(consumer.updates) > 1,
                                     timeout=_until(lag_end)), \
            "the slot was left empty before the venue published its close"
        assert consumer.wait_for(lambda: len(consumer.updates) >= 2)
        assert consumer.update_keys() == [(last, 100.0, True), (s2, 103.0, False)]
        venue.push(_bar(s2, 104.0))
        assert consumer.wait_for(lambda: len(consumer.closed) >= 2)
        assert consumer.stamps() == [last, s2]
        assert len(venue.history_calls) <= 3, "the history is flooded while the slot waits"
    finally:
        consumer.close()


def __test_slow_history_still_finalizes_a_forming_bar_at_the_deadline__():
    """A slot with a forming bar is closed from its ticks at the deadline, not the lag's end"""
    s1_close = _now_ms() - 31_000
    last, s1 = s1_close - 2 * _TF_MS, s1_close - _TF_MS
    venue = _Venue([[_bar(last), _bar(s1, 150.0, volume=3.0, is_closed=False)]],
                   idle_bars=False, publish_lag=_SLOW_LAG, publish_delay=_SLOW_LAG - 1.0)
    venue.set_history([_bar(s1, 155.0, volume=12.0)])
    consumer = _Consumer(venue)
    try:
        assert consumer.wait_for(lambda: len(consumer.closed) >= 2)
        finalized = consumer.closed[1]
        assert (finalized.timestamp, finalized.close, finalized.volume) == (s1, 150.0, 3.0)
        assert time.time() < s1_close / 1000.0 + _SLOW_LAG - 5.0
        assert len(venue.history_calls) == 1
    finally:
        consumer.close()


def __test_a_later_history_bar_proves_the_hole_before_the_lag__():
    """The history's bar of a later slot settles the awaited slot as empty at once

    With a publish lag longer than the timeframe, the venue publishing the next slot's
    close is the evidence that the awaited slot has no bar, long before its own lag ends.
    """
    lag = 90.0
    s1_close = _now_ms() - 65_000
    last, s2 = s1_close - 2 * _TF_MS, s1_close
    venue = _Venue([[_bar(last)]], idle_bars=False, publish_lag=lag)
    venue.set_history([_bar(s2, 102.0)])
    consumer = _Consumer(venue)
    try:
        assert consumer.wait_for(lambda: len(consumer.closed) >= 2)
        assert consumer.stamps() == [last, s2]
        assert venue.history_calls[0][0] < s1_close / 1000.0 + lag - 10.0
    finally:
        consumer.close()


def __test_startup_awaits_a_bar_the_history_has_not_published_yet__():
    """A bar the startup fill cannot get yet is awaited, not skipped by the first live update"""
    warmup_last, lag_end = _slow_venue_slot()
    s1, s2 = warmup_last + _TF_MS, warmup_last + 2 * _TF_MS
    venue = _Venue([[_bar(s2, 102.0, is_closed=False)]],
                   idle_bars=False, publish_lag=_SLOW_LAG, publish_delay=_SLOW_LAG - 1.0)
    venue.set_history([_bar(s1, 101.0)])
    consumer = _Consumer(venue, last_historical_timestamp=warmup_last)
    try:
        assert venue.wait_for(lambda: len(venue.history_calls) >= 2)
        assert not consumer.wait_for(lambda: len(consumer.updates) > 0,
                                     timeout=_until(lag_end)), \
            "a live update went on over a slot the history had not published yet"
        assert consumer.wait_for(lambda: len(consumer.updates) >= 2)
        assert consumer.update_keys() == [(s1, 101.0, True), (s2, 102.0, False)]
        assert all(since == warmup_last for _, since in venue.history_calls)
    finally:
        consumer.close()


def __test_reconnect_fills_the_outage_before_the_new_subscription__():
    """The bars closed during the outage go on before the new socket's first close"""
    last = _now_ms() - 2 * _TF_MS
    s1, s2, s3 = last + _TF_MS, last + 2 * _TF_MS, last + 3 * _TF_MS
    venue = _Venue([[_bar(last), ConnectionError("socket dropped")], [_bar(s3, 103.0)]],
                   publish_lag=60.0)
    venue.set_history([_bar(s1, 101.0), _bar(s2, 102.0)])
    consumer = _Consumer(venue)
    try:
        assert consumer.wait_for(lambda: len(consumer.closed) >= 4)
        assert [(bar.timestamp, bar.close) for bar in consumer.closed] == [
            (last, 100.0), (s1, 101.0), (s2, 102.0), (s3, 103.0)]
        assert venue.connects == 2
        assert [since for _, since in venue.history_calls] == [last]
    finally:
        consumer.close()


def __test_failed_reconnect_fill_is_retried_on_the_new_connection__():
    """A reconnect whose gap query fails does not go live: it asks again, without reconnecting"""
    # The outage's slot closes just now and the next one is forming: no slot is
    # overdue yet, so only the reconnect asks the history
    last = _now_ms() - 2 * _TF_MS
    s1, s2 = last + _TF_MS, last + 2 * _TF_MS
    forming = _bar(s2, 102.0, is_closed=False)
    venue = _Venue([[_bar(last), ConnectionError("socket dropped")], [forming]],
                   publish_lag=60.0)
    venue.set_history([_bar(s1, 101.0)], failures=3)
    consumer = _Consumer(venue)
    try:
        assert consumer.wait_for(lambda: len(consumer.updates) >= 3)
        assert consumer.update_keys() == [
            (last, 100.0, True), (s1, 101.0, True), (s2, 102.0, False)]
        assert venue.connects == 2, "a working connection was renewed for a failed history query"
        assert [since for _, since in venue.history_calls] == [last] * 4
    finally:
        consumer.close()


def __test_reconnect_fill_on_a_connection_lost_again_is_another_reconnect_round__():
    """The connection drops under the reconnect's gap query: it reconnects, then fills"""
    last = _now_ms() - 2 * _TF_MS
    s1, s2 = last + _TF_MS, last + 2 * _TF_MS
    forming = _bar(s2, 102.0, is_closed=False)
    venue = _DroppingVenue(
        [[_bar(last), ConnectionError("socket dropped")], [forming], [forming]],
        publish_lag=60.0, drop_on_connect=2)
    venue.set_history([_bar(s1, 101.0)])
    consumer = _Consumer(venue)
    try:
        assert consumer.wait_for(lambda: len(consumer.updates) >= 3)
        assert consumer.update_keys() == [
            (last, 100.0, True), (s1, 101.0, True), (s2, 102.0, False)]
        assert venue.connects == 3
    finally:
        consumer.close()


def __test_forming_bar_from_before_an_outage_is_not_the_slots_close__():
    """An outage inside a bar: the slot waits for the history, not the partial bar

    The forming bar the socket showed before it dropped holds only the ticks seen until
    then. Past the deadline the slot is not finalized from it: it waits out the publish
    lag like a slot without a forming bar, and the venue's bar is its close.
    """
    last, lag_end = _slow_venue_slot()
    s1 = last + _TF_MS
    venue = _Venue([[_bar(last), _bar(s1, 150.0, volume=3.0, is_closed=False),
                     ConnectionError("socket dropped")], []],
                   publish_lag=_SLOW_LAG, publish_delay=_SLOW_LAG - 1.0)
    venue.set_history([_bar(s1, 155.0, volume=12.0)])
    consumer = _Consumer(venue)
    try:
        assert venue.wait_for(lambda: venue.connects >= 2 and len(venue.history_calls) >= 2)
        assert not consumer.wait_for(lambda: len(consumer.closed) > 1,
                                     timeout=_until(lag_end)), \
            "the slot was closed from the forming bar seen before the outage"
        assert consumer.wait_for(lambda: len(consumer.closed) >= 2)
        recovered = consumer.closed[1]
        assert (recovered.timestamp, recovered.close, recovered.volume) == (s1, 155.0, 12.0)
    finally:
        consumer.close()


def __test_forming_bar_after_the_reconnect_is_finalized_at_the_deadline__():
    """A forming bar the new subscription delivers is tracked again and closes the slot"""
    s1_close = _now_ms() - 31_000
    last, s1 = s1_close - 2 * _TF_MS, s1_close - _TF_MS
    venue = _Venue([[_bar(last), _bar(s1, 150.0, volume=3.0, is_closed=False),
                     ConnectionError("socket dropped")],
                    [_bar(s1, 151.0, volume=5.0, is_closed=False)]],
                   publish_lag=_SLOW_LAG, publish_delay=_SLOW_LAG - 1.0)
    venue.set_history([_bar(s1, 155.0, volume=12.0)])
    consumer = _Consumer(venue)
    try:
        assert consumer.wait_for(lambda: len(consumer.closed) >= 2)
        finalized = consumer.closed[1]
        assert (finalized.timestamp, finalized.close, finalized.volume) == (s1, 151.0, 5.0)
        assert time.time() < s1_close / 1000.0 + _SLOW_LAG - 5.0
        assert venue.connects == 2
    finally:
        consumer.close()


# A publish lag shorter than the one-minute timeframe's 30 s close grace, as on a venue
# whose history has a close before the stream gives up on the socket's.
_FAST_LAG = 10.0


def _two_answer_slots() -> tuple[int, int, int]:
    """Opens of the newest closed bar and of the two slots after it, both closed.

    The history has published the first slot's close; the second one's lag ends
    ``_LATE_PUBLISH_IN`` seconds from now. With ``publish_delay`` just under the lag
    the history hands the two closes on in two answers.
    """
    s2_close = _now_ms() - int((_FAST_LAG - _LATE_PUBLISH_IN) * 1000)
    s2 = s2_close - _TF_MS
    return s2 - 2 * _TF_MS, s2 - _TF_MS, s2


def __test_closes_only_from_history_reconnect_the_subscription__():
    """Two closes with trades in a row from the history, none from the socket: a reconnect"""
    last, s1, s2 = _two_answer_slots()
    venue = _Venue([[_bar(last)]], publish_lag=_FAST_LAG, publish_delay=_FAST_LAG - 1.0)
    venue.set_history([_bar(s1, 101.0), _bar(s2, 102.0)])
    consumer = _Consumer(venue)
    try:
        assert consumer.wait_for(lambda: len(consumer.closed) >= 3)
        assert consumer.stamps() == [last, s1, s2]
        assert venue.wait_for(lambda: venue.connects >= 2)
    finally:
        consumer.close()


def __test_forming_updates_do_not_prove_the_close_subscription_alive__():
    """The socket shows every slot forming but never closes one: still a reconnect

    A venue can feed the forming bar from another subscription than the closes, so
    only a close from the socket tells that the close subscription works.
    """
    last, s1, s2 = _two_answer_slots()
    venue = _Venue([[_bar(last), _bar(s1, 101.0, is_closed=False),
                     _bar(s2, 102.0, is_closed=False)]],
                   publish_lag=_FAST_LAG, publish_delay=_FAST_LAG - 1.0)
    venue.set_history([_bar(s1, 101.0), _bar(s2, 102.5)])
    consumer = _Consumer(venue)
    try:
        assert consumer.wait_for(lambda: len(consumer.closed) >= 3)
        assert [(bar.timestamp, bar.close) for bar in consumer.closed] == [
            (last, 100.0), (s1, 101.0), (s2, 102.5)]
        assert venue.wait_for(lambda: venue.connects >= 2)
    finally:
        consumer.close()


def __test_late_socket_closes_do_not_count_as_a_dead_subscription__():
    """The socket delivers a close after the history did: the subscription is alive"""
    last, s1, s2 = _two_answer_slots()
    venue = _Venue([[_bar(last)]], publish_lag=_FAST_LAG, publish_delay=_FAST_LAG - 1.0)
    venue.set_history([_bar(s1, 101.0), _bar(s2, 102.0)])
    consumer = _Consumer(venue)
    try:
        assert consumer.wait_for(lambda: len(consumer.closed) >= 2)
        venue.push(_bar(s1, 101.0))
        assert consumer.wait_for(lambda: len(consumer.closed) >= 3)
        assert consumer.stamps() == [last, s1, s2]
        assert not venue.wait_for(lambda: venue.connects >= 2, timeout=1.0)
    finally:
        consumer.close()


def __test_a_backlog_in_one_history_answer_is_one_missed_delivery__():
    """Several closes fetched in one answer, then a socket close: no reconnect"""
    last = _now_ms() - 3 * _TF_MS - 40_000
    s1, s2, s3 = last + _TF_MS, last + 2 * _TF_MS, last + 3 * _TF_MS
    venue = _Venue([[_bar(last)]])
    venue.set_history([_bar(s1, 101.0), _bar(s2, 102.0)])
    consumer = _Consumer(venue)
    try:
        assert consumer.wait_for(lambda: len(consumer.closed) >= 3)
        venue.push(_bar(s3, 103.0))
        assert consumer.wait_for(lambda: len(consumer.closed) >= 4)
        assert consumer.stamps() == [last, s1, s2, s3]
        assert venue.connects == 1
    finally:
        consumer.close()


def __test_first_close_after_a_session_break_comes_from_history__():
    """The socket misses the first close after the reopen: the history has it

    The slot after the last in-session bar is in the closed session and has no close
    to wait for; once the session has reopened, its first slot is the awaited one.
    """
    last = (_now_ms() // _TF_MS) * _TF_MS - 4 * _TF_MS
    s1, s2, s3 = last + _TF_MS, last + 2 * _TF_MS, last + 3 * _TF_MS
    # A daily one-minute break at s1: each day's session runs from s2 overnight to s1
    reopen = datetime.fromtimestamp(s2 / 1000.0, tz=UTC).time()
    break_start = datetime.fromtimestamp(s1 / 1000.0, tz=UTC).time()
    syminfo = _make_syminfo([SymInfoInterval(day=d, start=reopen, end=break_start)
                             for d in range(7)])
    venue = _Venue([[_bar(last)]], idle_bars=False)
    venue.set_history([_bar(s2, 102.0)])
    consumer = _Consumer(venue, syminfo=syminfo)
    try:
        assert consumer.wait_for(lambda: len(consumer.closed) >= 2), \
            "the first close after the reopen was never asked from the history"
        venue.push(_bar(s3, 103.0))
        assert consumer.wait_for(lambda: len(consumer.closed) >= 3)
        assert [(bar.timestamp, bar.close) for bar in consumer.closed] == [
            (last, 100.0), (s2, 102.0), (s3, 103.0)]
    finally:
        consumer.close()


def __test_close_of_a_warmup_bar_forming_at_download_comes_from_history__():
    """The warmup's last bar was still forming when downloaded and the socket misses its close

    Its slot is the first one the stream closes: the history's bar of it is handed on
    under the warmup bar's own timestamp, before any later update.
    """
    tail = _now_ms() - _TF_MS - 40_000
    s1 = tail + _TF_MS
    venue = _Venue([[_bar(s1, 102.0, is_closed=False)]])
    venue.set_history([_bar(tail, 101.0)])
    consumer = _Consumer(venue, last_historical_timestamp=tail,
                         history_as_of=tail + _TF_MS // 2)
    try:
        assert consumer.wait_for(lambda: len(consumer.updates) >= 2)
        assert consumer.update_keys() == [(tail, 101.0, True), (s1, 102.0, False)]
        assert all(since == tail - _TF_MS for _, since in venue.history_calls)
    finally:
        consumer.close()


def __test_a_warmup_bar_complete_at_download_is_not_asked_again__():
    """The warmup's last bar had closed before the download: the history starts after it"""
    tail = _now_ms() - 2 * _TF_MS - 40_000
    s1 = tail + _TF_MS
    venue = _Venue([[]])
    venue.set_history([_bar(tail, 101.0), _bar(s1, 102.0)])
    consumer = _Consumer(venue, last_historical_timestamp=tail,
                         history_as_of=tail + _TF_MS + 1_000)
    try:
        assert consumer.wait_for(lambda: len(consumer.closed) >= 1)
        assert consumer.stamps()[0] == s1
        assert all(since >= tail for _, since in venue.history_calls)
    finally:
        consumer.close()


class _DroppingVenue(_Venue):
    """A venue one of whose connections goes down under a history query on it."""

    def __init__(self, scripts: list[list[OHLCV | BaseException]], *,
                 drop_on_connect: int = 1, **kwargs):
        super().__init__(scripts, **kwargs)
        self.drop_on_connect = drop_on_connect

    async def backfill_closed_bars(self, symbol: str, timeframe: str,
                                   since_ms: int) -> list[OHLCV]:
        if self.connects == self.drop_on_connect:
            self._connected = False
            raise ConnectionError("connection lost under the history query")
        return await super().backfill_closed_bars(symbol, timeframe, since_ms)


def __test_startup_gap_reconnects_when_the_connection_drops_under_the_query__():
    """A connection lost during the startup gap query is renewed, then the gap is filled"""
    last = _now_ms() - 3 * _TF_MS
    s1 = last + _TF_MS
    venue = _DroppingVenue([[], []])
    venue.set_history([_bar(s1, 101.0)])
    consumer = _Consumer(venue, last_historical_timestamp=last)
    try:
        assert consumer.wait_for(lambda: len(consumer.closed) >= 1), \
            "the startup gap query was retried on a dead connection forever"
        assert consumer.stamps()[0] == s1
        assert venue.connects == 2
    finally:
        consumer.close()


def __test_startup_gap_failure_waits_until_stopped__():
    """A startup gap query that keeps failing never lets a live bar through, and a stop ends it"""
    last = _now_ms() - 3 * _TF_MS
    venue = _Venue([[_bar(last + 3 * _TF_MS)]])
    venue.set_history([], failures=1_000_000)
    consumer = _Consumer(venue, last_historical_timestamp=last)
    assert venue.wait_for(lambda: len(venue.history_calls) >= 3)
    started = time.monotonic()
    consumer.close()
    assert time.monotonic() - started < 5.0
    assert consumer.closed == []
    assert not venue.is_connected


def _utc_ms(year: int, month: int, day: int, hour: int = 0) -> int:
    return int(datetime(year, month, day, hour, tzinfo=UTC).timestamp() * 1000)


def __test_monthly_stream_awaits_the_next_calendar_month__(monkeypatch):
    """A monthly stream awaits the slot opening on the 1st of the next month

    The average month the timeframe's length stands for ends on the 31st of a long
    month, so a slot computed from it would hold every update of the real next month
    behind a slot that never comes. Checked after a closed bar and after the warmup.
    """
    now_ms = _utc_ms(2026, 11, 17, 12)
    clock_offset = now_ms / 1000.0 - time.time()
    monkeypatch.setattr(live_runner, "time", SimpleNamespace(
        time=lambda: time.time() + clock_offset, monotonic=time.monotonic))
    october, november = _utc_ms(2026, 10, 1), _utc_ms(2026, 11, 1)
    for script, warmup_last in (([_bar(october)], None), ([], october)):
        venue = _Venue([script])
        consumer = _Consumer(venue, "1M", last_historical_timestamp=warmup_last)
        try:
            # Pushed once live: a forming bar queued during the warmup catch-up is dropped
            assert consumer.wait_for(lambda: consumer.live and len(consumer.closed) == len(script))
            venue.push(_bar(november, 101.0, is_closed=False))
            assert consumer.wait_for(lambda: (november, 101.0, False) in consumer.update_keys()), \
                "the next month's forming bar was held behind a slot that does not exist"
            assert venue.history_calls == []
        finally:
            consumer.close()


def __test_a_socket_close_not_matching_the_history_is_replaced_by_the_history_bar__(caplog):
    """On a venue whose socket candle is not the history's, the close comes from the history

    The socket's close of the slot never reaches the consumer as a closed bar; the
    slot's closed bar is the history's, asked once the publish lag past the close is
    over. It is the expected close, not a missed one.
    """
    lag = 0.4
    last = _now_ms() - 2 * _TF_MS + 300
    s1 = last + _TF_MS
    venue = _Venue([[_bar(last), _bar(s1, 150.0, volume=3.0)]], publish_lag=lag,
                   stream_closes_final=False)
    venue.set_history([_bar(s1, 155.0, volume=12.0)])
    with caplog.at_level(logging.INFO, logger="pyne_core_logger"):
        consumer = _Consumer(venue)
        try:
            assert consumer.wait_for(lambda: len(consumer.closed) >= 2, timeout=10.0)
            closed = consumer.closed[1]
            assert (closed.timestamp, closed.close, closed.volume) == (s1, 155.0, 12.0)
            assert (s1, 150.0, True) not in consumer.update_keys()
            due = (s1 + _TF_MS) / 1000.0 + lag
            assert venue.history_calls[0][0] >= due - 0.05, "the history was asked before the lag"
        finally:
            consumer.close()
    assert "missed close" not in caplog.text


def __test_the_socket_close_stands_in_when_the_history_does_not_answer__(caplog):
    """A slot whose history query keeps failing is closed with the socket's close at the deadline

    The stream does not stand on a slot the socket closed: the socket's values are a
    real close, only possibly not the history's, and the operator is told so.
    """
    last = (_now_ms() // _TF_MS) * _TF_MS - 4 * _TF_MS
    s1 = last + _TF_MS
    venue = _Venue([[_bar(last), _bar(s1, 150.0, volume=3.0)]], stream_closes_final=False)
    venue.set_history([], failures=1_000_000)
    with caplog.at_level(logging.INFO, logger="pyne_core_logger"):
        consumer = _Consumer(venue)
        try:
            assert consumer.wait_for(lambda: len(consumer.closed) >= 2, timeout=10.0)
            closed = consumer.closed[1]
            assert (closed.timestamp, closed.close, closed.volume) == (s1, 150.0, 3.0)
        finally:
            consumer.close()
    assert f"close of ts={s1} not in the history by its deadline" in caplog.text


def __test_a_history_bar_asked_before_the_publish_lag_is_asked_again__():
    """On a venue that revises a fresh bar, one returned before the lag past its close is not taken

    The venue may still change it: the reconnect gap fill right after the close gets
    the slot with its early values, and the bar handed on is the one asked for once
    the lag is over.
    """
    lag = 1.5
    last = _now_ms() - 2 * _TF_MS - 50
    s1 = last + _TF_MS
    venue = _Venue([[_bar(last), ConnectionError("socket dropped")], []], publish_lag=lag,
                   revises=True)
    venue.set_history([_bar(s1, 140.0, volume=5.0)])
    revise = threading.Timer(
        max(0.0, (s1 + _TF_MS) / 1000.0 + 0.7 - time.time()),
        lambda: venue.set_history([_bar(s1, 145.0, volume=11.0)]))
    revise.start()
    consumer = _Consumer(venue)
    try:
        assert venue.wait_for(lambda: venue.connects >= 2)
        assert venue.wait_for(lambda: len(venue.history_calls) >= 1)
        assert venue.history_calls[0][0] < (s1 + _TF_MS) / 1000.0 + 0.7, \
            "the reconnect gap was not asked before the revision"
        assert consumer.wait_for(lambda: len(consumer.closed) >= 2, timeout=10.0)
        closed = consumer.closed[1]
        assert (closed.timestamp, closed.close, closed.volume) == (s1, 145.0, 11.0)
    finally:
        revise.cancel()
        consumer.close()
