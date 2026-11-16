"""
A live ``request.security`` context's REST warmup: where it starts (the earlier of
the chart's history start and the context's own default depth), and that its bar
still forming at the download never enters the closed buffer. And a closed round's
bar read: it waits for a lagging stream close only while a bar opening at or before
its target can still come.
"""
from datetime import datetime, timedelta, UTC
from zoneinfo import ZoneInfo

from pynecore.core.security import BarCalendar, actual_bar_close
# noinspection PyProtectedMember
from pynecore.core.security_process import (_DEFAULT_WARMUP_BARS, _drop_forming_rest_tail,
                                            _read_round_bar, _warmup_start)
from pynecore.types.ohlcv import OHLCV

_NOW = datetime(2026, 11, 16, 15, 38, tzinfo=UTC)
# A 50-bar hourly chart
_CHART_FROM = _NOW - timedelta(hours=50)

_HOUR_MS = 3_600_000
_MINUTE_MS = 60_000
# The download's end: 15:38, inside the 15:00 hourly bar
_HORIZON_MS = int(_NOW.timestamp() * 1000)
_FORMING_OPEN = _HORIZON_MS - 38 * 60_000
_CAL = BarCalendar(tz=ZoneInfo("UTC"))


def _buffer(*opens: int) -> list[OHLCV]:
    return [OHLCV(timestamp=ts, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)
            for ts in opens]


def __test_htf_context_gets_its_own_depth__():
    """A daily context on a short hourly chart warms up over its own default depth"""
    start = _warmup_start(_CHART_FROM, "1D", _NOW)
    assert start == _NOW - timedelta(days=_DEFAULT_WARMUP_BARS)


def __test_ltf_context_covers_the_chart_history__():
    """A lower-timeframe context reaches back to the chart's first bar"""
    long_chart = _NOW - timedelta(days=30)
    assert _warmup_start(long_chart, "1", _NOW) == long_chart


def __test_naive_chart_start_is_utc__():
    """A naive chart start is read as UTC and the result is timezone-aware"""
    naive = (_NOW - timedelta(days=30)).replace(tzinfo=None)
    start = _warmup_start(naive, "5", _NOW)
    assert start.tzinfo is not None
    assert start == _NOW - timedelta(days=30)


def __test_no_chart_start_uses_the_default_depth__():
    """Without a chart start the context's own default depth applies"""
    assert _warmup_start(None, "240", _NOW) == _NOW - timedelta(hours=4 * _DEFAULT_WARMUP_BARS)


def __test_forming_rest_tail_is_dropped__():
    """The bar still forming at the download leaves the buffer; its stream close replaces it"""
    closed = _FORMING_OPEN - _HOUR_MS
    buffer = _buffer(closed, _FORMING_OPEN)
    tail = _drop_forming_rest_tail(buffer, _FORMING_OPEN, None,
                                   _HORIZON_MS, _CAL, "60")
    assert [bar.timestamp for bar in buffer] == [closed]
    assert tail == closed


def __test_closed_rest_tail_stays__():
    """A tail whose close the download already reached is final"""
    closed = _FORMING_OPEN - _HOUR_MS
    buffer = _buffer(closed - _HOUR_MS, closed)
    tail = _drop_forming_rest_tail(buffer, closed, None,
                                   _HORIZON_MS, _CAL, "60")
    assert len(buffer) == 2 and tail == closed


def __test_stream_refined_tail_stays__():
    """A tail the catch-up drain replaced with its stream close is final"""
    buffer = _buffer(_FORMING_OPEN)
    tail = _drop_forming_rest_tail(buffer, _FORMING_OPEN,
                                   _FORMING_OPEN,
                                   _HORIZON_MS, _CAL, "60")
    assert len(buffer) == 1 and tail == _FORMING_OPEN


def __test_daily_tail_closed_by_its_session_stays__():
    """A daily bar is closed by its calendar's close, not by the next day's open"""
    # The UTC daily bar of 2026-11-15 closed at midnight, before the 15:38 download
    day_open = int(datetime(2026, 11, 15, tzinfo=UTC).timestamp() * 1000)
    buffer = _buffer(day_open)
    assert _drop_forming_rest_tail(buffer, day_open, None, _HORIZON_MS,
                                   _CAL, "1D") == day_open
    # Today's daily bar is still forming
    today = day_open + 24 * _HOUR_MS
    buffer = _buffer(day_open, today)
    assert _drop_forming_rest_tail(buffer, today, None, _HORIZON_MS,
                                   _CAL, "1D") == day_open
    assert len(buffer) == 1


class __test_helper_Stream:
    """A streamer stand-in: records its waits and delivers a fixed batch."""

    def __init__(self, batch: list[OHLCV]):
        self.batch = batch
        self.waits: list[float] = []

    def wait_for_bars(self, timeout: float) -> list[OHLCV]:
        self.waits.append(timeout)
        batch, self.batch = self.batch, []
        return batch


def __test_consumed_target_ends_the_round_without_waiting__():
    """With the target run and its close the last buffered bar, nothing is waited for"""
    target = _FORMING_OPEN - _HOUR_MS
    buffer = _buffer(target)
    stream = __test_helper_Stream(_buffer(_FORMING_OPEN))
    bar = _read_round_bar(lambda i: buffer[i] if i < len(buffer) else None, 1, target,
                          actual_bar_close(target, 0, _CAL, "60"), 30.0,
                          stream.wait_for_bars, buffer.append, now=lambda: 0.0)
    assert bar is None
    assert stream.waits == []


def __test_lower_timeframe_cutoff_ends_the_round_without_waiting__():
    """A 5-minute chart bar's 1-minute round ends after its fifth intrabar, without a wait

    The live scalar lower-timeframe target is the chart period's last millisecond,
    which no intrabar opens at.
    """
    chart_open = _FORMING_OPEN - _HOUR_MS
    cutoff = chart_open + 5 * _MINUTE_MS - 1
    buffer = _buffer(*(chart_open + i * _MINUTE_MS for i in range(5)))
    stream = __test_helper_Stream(_buffer(chart_open + 5 * _MINUTE_MS))
    run_close = None
    idx = 0
    while (bar := _read_round_bar(lambda i: buffer[i] if i < len(buffer) else None, idx,
                                  cutoff, run_close, 30.0, stream.wait_for_bars,
                                  buffer.append, now=lambda: 0.0)) is not None:
        next_open = buffer[idx + 1].timestamp if idx + 1 < len(buffer) else 0
        run_close = actual_bar_close(bar.timestamp, next_open, _CAL, "1")
        idx += 1
    assert idx == 5
    assert stream.waits == []


def __test_missing_target_waits_for_the_stream_close__():
    """While the target has not run, a missing bar is waited for and ingested"""
    buffer = _buffer(_FORMING_OPEN - _HOUR_MS)
    stream = __test_helper_Stream(_buffer(_FORMING_OPEN))
    bar = _read_round_bar(lambda i: buffer[i] if i < len(buffer) else None, 1, _FORMING_OPEN,
                          actual_bar_close(_FORMING_OPEN - _HOUR_MS, 0, _CAL, "60"), 30.0,
                          stream.wait_for_bars, buffer.append, now=lambda: 10.0)
    assert bar is not None and bar.timestamp == _FORMING_OPEN
    assert stream.waits == [20.0]


def __test_spent_grace_or_no_stream_does_not_wait__():
    """A passed deadline or a context without a stream returns the gap at once"""
    stream = __test_helper_Stream(_buffer(_FORMING_OPEN))
    assert _read_round_bar(lambda _i: None, 0, _FORMING_OPEN, None, 30.0, stream.wait_for_bars,
                           lambda _b: None, now=lambda: 30.0) is None
    assert stream.waits == []
    assert _read_round_bar(lambda _i: None, 0, _FORMING_OPEN, None, None, None,
                           lambda _b: None) is None
