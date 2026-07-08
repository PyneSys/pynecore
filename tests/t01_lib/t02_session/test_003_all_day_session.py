"""
Full-day ("0000-0000") session handling in ``lib._is_bar_in_session``.

Pine's ``time(timeframe, session)`` treats a session whose start equals its end
as the full 24-hour day -- ``"0000-0000"`` is the canonical all-day session and
the default of ``input.session``. A corpus strategy gating entries on
``na(time(timeframe.period, "0000-0000:" + days)) == false`` placed zero trades
because every bar was wrongly reported out of session.
"""
from datetime import datetime, UTC

from pynecore.lib import _parse_session_string, _is_bar_in_session


def _ms(year: int, month: int, day: int, hour: int, minute: int = 0) -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1000)


def __test_all_day_session_includes_every_hour__():
    """"0000-0000" (start == end) covers the whole day on an allowed weekday"""
    # 2021-01-04 is a Monday.
    session_info = _parse_session_string("0000-0000:1234567", "UTC")
    for hour in (0, 3, 12, 18, 23):
        assert _is_bar_in_session(_ms(2021, 1, 4, hour), session_info, "60") is True, hour


def __test_all_day_session_still_honours_day_filter__():
    """The all-day span does not override the day-of-week gate"""
    # Days "23456" = Monday..Friday (TV numbering, 1=Sun).
    session_info = _parse_session_string("0000-0000:23456", "UTC")
    assert _is_bar_in_session(_ms(2021, 1, 3, 12), session_info, "60") is False  # Sunday
    assert _is_bar_in_session(_ms(2021, 1, 4, 12), session_info, "60") is True   # Monday


def __test_bounded_session_unaffected__():
    """A normal HHMM-HHMM session keeps excluding out-of-range bars"""
    session_info = _parse_session_string("0930-1600:1234567", "UTC")
    assert _is_bar_in_session(_ms(2021, 1, 4, 12), session_info, "60") is True
    assert _is_bar_in_session(_ms(2021, 1, 4, 20), session_info, "60") is False
