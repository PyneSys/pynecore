"""
Session bar bounds behind ``time(timeframe, session)`` / ``time_close(...)``.

A session does not filter the requested timeframe's grid, it replaces it with a
series of session bars: a daily request reports the session's own open and
close, and weekly/monthly requests run from one period's first session open to
the next one's. Measured against TradingView on BINANCE:BTCUSDT@30 with the
"0300-1200", "1700-0200" and "0930-1600:23456" New York sessions.
"""
from datetime import date, datetime, UTC

from pynecore.lib import (_parse_session_string, _is_bar_in_session, _session_occurrence,
                          _session_period_anchor, _session_bar_bounds, syminfo)

NY = "America/New_York"
HALF_HOUR = 1800


def _ms(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1000)


def __test_day_session_reports_its_own_bounds__():
    """A day session's occurrence opens and closes at the session, not at midnight"""
    infos = _parse_session_string("0300-1200:1234567", NY)
    # 2025-01-06 is a Monday; 03:00 New York is 08:00 UTC in winter.
    bounds = _session_occurrence(_ms(2025, 1, 6, 10), infos, HALF_HOUR)
    assert bounds == (_ms(2025, 1, 6, 8), _ms(2025, 1, 6, 17))
    # The bar before the open and the bar at the close are both outside.
    assert _session_occurrence(_ms(2025, 1, 6, 7, 30), infos, HALF_HOUR) is None
    assert _session_occurrence(_ms(2025, 1, 6, 17), infos, HALF_HOUR) is None


def __test_overnight_occurrence_is_anchored_on_its_opening_day__():
    """A bar after midnight belongs to the run that opened the previous evening"""
    infos = _parse_session_string("1700-0200:1234567", NY)
    # 17:00 New York on 2025-01-06 is 22:00 UTC, the run closes 02:00 the next day.
    expected = (_ms(2025, 1, 6, 22), _ms(2025, 1, 7, 7))
    assert _session_occurrence(_ms(2025, 1, 6, 23), infos, HALF_HOUR) == expected
    assert _session_occurrence(_ms(2025, 1, 7, 3), infos, HALF_HOUR) == expected


def __test_dst_crossing_session_keeps_its_wall_clock_endpoints__():
    """A run spanning a daylight-saving change keeps its clock times, not its length

    MEASURED on BINANCE:BTCUSDT@30: across the 2025-03-09 spring change
    "1900-0400" ran 19:00 EST -> 04:00 EDT (eight hours instead of nine) and
    "0100-0500" ran 01:00 EST -> 05:00 EDT (three instead of four); across the
    2025-11-02 fall-back the same two ran ten and five hours.
    """
    evening = _parse_session_string("1900-0400:1234567", NY)
    night = _parse_session_string("0100-0500:1234567", NY)
    # Spring forward: the closing clock time is reached an hour of real time early.
    assert _session_occurrence(_ms(2025, 3, 9, 3), evening, HALF_HOUR) == \
           (_ms(2025, 3, 9, 0), _ms(2025, 3, 9, 8))
    assert _session_occurrence(_ms(2025, 3, 9, 7), night, HALF_HOUR) == \
           (_ms(2025, 3, 9, 6), _ms(2025, 3, 9, 9))
    # Fall back: the repeated hour stretches the run instead.
    assert _session_occurrence(_ms(2025, 11, 2, 3), evening, HALF_HOUR) == \
           (_ms(2025, 11, 1, 23), _ms(2025, 11, 2, 9))
    assert _session_occurrence(_ms(2025, 11, 2, 7), night, HALF_HOUR) == \
           (_ms(2025, 11, 2, 5), _ms(2025, 11, 2, 10))
    # The bar starting at the close is already out of session.
    assert _is_bar_in_session(_ms(2025, 11, 2, 8, 30), evening, "30") is True
    assert _is_bar_in_session(_ms(2025, 11, 2, 9), evening, "30") is False


def __test_session_closing_on_the_changing_hour_ends_on_its_last_minute__():
    """A run closing at 02:00 keeps its span across both daylight-saving changes

    MEASURED on BINANCE:BTCUSDT@30 with daily requests: "1700-0200" ran nine
    hours on both 2025 transition nights, closing 03:00 EDT on 2025-03-09 and
    01:00 EST on 2025-11-02. The last minute is what closes the run: on the
    fall-back date 01:59 is still the first, EDT one, an hour before the
    unambiguous 02:00 EST the closing wall clock alone would name.
    """
    infos = _parse_session_string("1700-0200:1234567", NY)
    # Spring forward: 17:00 EST is 22:00 UTC, and 02:00 does not exist.
    assert _session_occurrence(_ms(2025, 3, 9, 0), infos, HALF_HOUR) == \
           (_ms(2025, 3, 8, 22), _ms(2025, 3, 9, 7))
    # Fall back: 17:00 EDT is 21:00 UTC, and 02:00 is reached an hour early.
    assert _session_occurrence(_ms(2025, 11, 2, 0), infos, HALF_HOUR) == \
           (_ms(2025, 11, 1, 21), _ms(2025, 11, 2, 6))
    assert _is_bar_in_session(_ms(2025, 11, 2, 5, 30), infos, "30") is True
    assert _is_bar_in_session(_ms(2025, 11, 2, 6), infos, "30") is False


def __test_period_anchor_is_the_first_session_open_of_the_period__():
    """Weekly and monthly session bars open at the period's first session open"""
    infos = _parse_session_string("0300-1200:1234567", NY)
    # Week of Monday 2024-12-30: 03:00 New York on that Monday.
    assert _session_period_anchor(date(2024, 12, 30), infos) == _ms(2024, 12, 30, 8)
    # Month of January 2025 opens on the 1st, a Wednesday.
    assert _session_period_anchor(date(2025, 1, 1), infos) == _ms(2025, 1, 1, 8)


def __test_period_anchor_walks_past_masked_days__():
    """A period starting on a masked-out day anchors on its first trading day"""
    infos = _parse_session_string("0930-1600:23456", NY)
    # 2025-02-01 is a Saturday: the month's first session runs on Monday the 3rd.
    assert _session_period_anchor(date(2025, 2, 1), infos) == _ms(2025, 2, 3, 14, 30)


def __test_day_mask_names_the_closing_weekday__():
    """An overnight occurrence is filed under the weekday its last minute is on

    MEASURED on BINANCE:BTCUSDT@60 (exchange time is UTC): "1700-0200:23456"
    ran Sunday 17:00 -> Monday 02:00 up to Thursday 17:00 -> Friday 02:00, so
    the Sunday evening is in session and the Friday evening is not.
    """
    infos = _parse_session_string("1700-0200:23456", "UTC")
    # 2025-01-05 is a Sunday, 2025-01-10 a Friday.
    sunday_night = (_ms(2025, 1, 5, 17), _ms(2025, 1, 6, 2))
    assert _session_occurrence(_ms(2025, 1, 5, 18), infos, 3600) == sunday_night
    assert _session_occurrence(_ms(2025, 1, 6, 1), infos, 3600) == sunday_night
    # The Friday evening opens a Saturday-closing run, which the mask drops.
    assert _session_occurrence(_ms(2025, 1, 10, 18), infos, 3600) is None
    assert _session_occurrence(_ms(2025, 1, 11, 1), infos, 3600) is None


def __test_day_mask_of_a_full_day_wrap_stays_on_the_opening_day__():
    """A session closing exactly at midnight belongs to the day it ran through

    Same measurement: "0000-0000:23456" covered Monday through Friday, while
    "1700-1700:23456" ran Sunday 17:00 -> Monday 17:00 up to Thursday -> Friday.
    """
    all_day = _parse_session_string("0000-0000:23456", "UTC")
    assert _is_bar_in_session(_ms(2025, 1, 6, 12), all_day, "60") is True   # Monday
    assert _is_bar_in_session(_ms(2025, 1, 10, 12), all_day, "60") is True  # Friday
    assert _is_bar_in_session(_ms(2025, 1, 11, 12), all_day, "60") is False  # Saturday
    assert _is_bar_in_session(_ms(2025, 1, 5, 12), all_day, "60") is False  # Sunday

    wrap = _parse_session_string("1700-1700:23456", "UTC")
    assert _is_bar_in_session(_ms(2025, 1, 5, 18), wrap, "60") is True    # Sunday evening
    assert _is_bar_in_session(_ms(2025, 1, 10, 12), wrap, "60") is True   # Friday noon
    assert _is_bar_in_session(_ms(2025, 1, 10, 18), wrap, "60") is False  # Friday evening


def __test_period_anchor_of_an_overnight_session_opens_the_evening_before__():
    """A period anchors on the run whose LAST minute is its first session day

    MEASURED on BINANCE:BTCUSDT@30: the weekly bar of "1700-0200:23456" New York
    opened Sunday 17:00 -- the run that closes Monday 02:00 -- and its monthly
    bar opened 2024-12-31 17:00 for January 2025 and 2025-02-02 17:00 for
    February, whose 1st is a Saturday.
    """
    infos = _parse_session_string("1700-0200:23456", NY)
    # 2025-01-06 is a Monday; 17:00 New York is 22:00 UTC in winter.
    assert _session_period_anchor(date(2025, 1, 6), infos) == _ms(2025, 1, 5, 22)
    assert _session_period_anchor(date(2025, 1, 1), infos) == _ms(2024, 12, 31, 22)
    assert _session_period_anchor(date(2025, 2, 1), infos) == _ms(2025, 2, 2, 22)

    period = syminfo.period
    timezone = syminfo.timezone
    try:
        syminfo.period = "30"
        syminfo.timezone = "UTC"
        week = _ms(2025, 1, 6)
        previous_week = _ms(2024, 12, 30)
        # The Sunday evening open already belongs to the new weekly bar.
        assert _session_bar_bounds(_ms(2025, 1, 5, 21, 30), infos, 'W', 1,
                                   previous_week, previous_week) == \
               (_ms(2024, 12, 29, 22), _ms(2025, 1, 5, 22))
        assert _session_bar_bounds(_ms(2025, 1, 5, 22), infos, 'W', 1,
                                   previous_week, previous_week) == \
               (_ms(2025, 1, 5, 22), _ms(2025, 1, 12, 22))
        assert _session_bar_bounds(_ms(2025, 1, 8, 12), infos, 'W', 1, week, week) == \
               (_ms(2025, 1, 5, 22), _ms(2025, 1, 12, 22))
    finally:
        syminfo.period = period
        syminfo.timezone = timezone


def __test_period_anchor_of_a_midnight_closing_session_stays_on_its_day__():
    """A run closing at midnight anchors the period on its own opening day

    MEASURED on BINANCE:BTCUSDT@30: the weekly bar of "0000-0000:23456" New York
    opened Monday 00:00, and its monthly bar opened 2025-01-01 00:00 for January
    and 2025-02-03 00:00 for February, whose first two days are a masked-out
    weekend. Without the day mask February opened on the 1st.
    """
    infos = _parse_session_string("0000-0000:23456", NY)
    # Midnight New York is 05:00 UTC in winter.
    assert _session_period_anchor(date(2025, 1, 6), infos) == _ms(2025, 1, 6, 5)
    assert _session_period_anchor(date(2025, 1, 1), infos) == _ms(2025, 1, 1, 5)
    assert _session_period_anchor(date(2025, 2, 1), infos) == _ms(2025, 2, 3, 5)
    every_day = _parse_session_string("0000-0000", NY)
    assert _session_period_anchor(date(2025, 2, 1), every_day) == _ms(2025, 2, 1, 5)

    period = syminfo.period
    timezone = syminfo.timezone
    try:
        syminfo.period = "30"
        syminfo.timezone = "UTC"
        week = _ms(2025, 1, 6)
        # The weekly bar turns over at the Monday open, not the Sunday before it.
        assert _session_bar_bounds(_ms(2025, 1, 6, 4, 30), infos, 'W', 1, week, week) == \
               (_ms(2024, 12, 30, 5), _ms(2025, 1, 6, 5))
        assert _session_bar_bounds(_ms(2025, 1, 6, 5), infos, 'W', 1, week, week) == \
               (_ms(2025, 1, 6, 5), _ms(2025, 1, 13, 5))
    finally:
        syminfo.period = period
        syminfo.timezone = timezone


def __test_timeframe_bars_back_walks_the_session_bar_series__():
    """A positive offset steps session bars, crossing into the previous run

    MEASURED (BINANCE:BTCUSDT@30, requested "60"): with "0900-1130" -- three
    buckets a run -- the offsets 1 and 3 taken on the 09:00 bucket reported the
    previous day's 11:00 and 09:00 buckets. Out of session the walk starts from
    the last bucket of the run that has already closed and reports whichever run
    it lands in by that run's OPENING bucket: offsets 1 and 3 gave the current
    day's and the previous day's 09:00 bucket.
    """
    period = syminfo.period
    timezone = syminfo.timezone
    try:
        syminfo.period = "30"
        syminfo.timezone = "UTC"
        infos = _parse_session_string("0900-1130", "UTC")
        assert _session_bar_bounds(_ms(2025, 1, 6, 9), infos, '', 60, 0, 0, 1) == \
               (_ms(2025, 1, 5, 11), _ms(2025, 1, 5, 11, 30))
        assert _session_bar_bounds(_ms(2025, 1, 6, 9), infos, '', 60, 0, 0, 3) == \
               (_ms(2025, 1, 5, 9), _ms(2025, 1, 5, 10))
        assert _session_bar_bounds(_ms(2025, 1, 6, 9), infos, '', 60, 0, 0, 4) == \
               (_ms(2025, 1, 4, 11), _ms(2025, 1, 4, 11, 30))
        # Out of session the offset is answered, and always by an opening bucket.
        assert _session_bar_bounds(_ms(2025, 1, 6, 13), infos, '', 60, 0, 0) is None
        assert _session_bar_bounds(_ms(2025, 1, 6, 13), infos, '', 60, 0, 0, 1) == \
               (_ms(2025, 1, 6, 9), _ms(2025, 1, 6, 10))
        assert _session_bar_bounds(_ms(2025, 1, 6, 13), infos, '', 60, 0, 0, 3) == \
               (_ms(2025, 1, 5, 9), _ms(2025, 1, 5, 10))
        # A daily request walks whole occurrences, and out of session it counts
        # from the run the chart bar is heading into.
        assert _session_bar_bounds(_ms(2025, 1, 6, 10), infos, 'D', 1, 0, 0, 1) == \
               (_ms(2025, 1, 5, 9), _ms(2025, 1, 5, 11, 30))
        assert _session_bar_bounds(_ms(2025, 1, 6, 13), infos, 'D', 1, 0, 0, 1) == \
               (_ms(2025, 1, 6, 9), _ms(2025, 1, 6, 11, 30))
        assert _session_bar_bounds(_ms(2025, 1, 6, 13), infos, 'D', 1, 0, 0, 2) == \
               (_ms(2025, 1, 5, 9), _ms(2025, 1, 5, 11, 30))
    finally:
        syminfo.period = period
        syminfo.timezone = timezone


def __test_intraday_request_tiles_the_occurrence__():
    """Intraday session bars are counted from the session open and cut at its close

    MEASURED on BINANCE:BTCUSDT@30 with a "60" request: "0930-1600" ran
    09:30-10:30, 10:30-11:30, ... 15:30-16:00, and "0900-1130" closed its 11:00
    bucket at 11:30 instead of at the 12:00 the plain grid would give.
    """
    period = syminfo.period
    try:
        syminfo.period = "30"
        infos = _parse_session_string("0930-1600", "UTC")
        # The off-grid open shifts the whole series onto the half hours.
        assert _session_bar_bounds(_ms(2025, 1, 6, 9, 30), infos, '', 60, 0, 0) == \
               (_ms(2025, 1, 6, 9, 30), _ms(2025, 1, 6, 10, 30))
        assert _session_bar_bounds(_ms(2025, 1, 6, 10), infos, '', 60, 0, 0) == \
               (_ms(2025, 1, 6, 9, 30), _ms(2025, 1, 6, 10, 30))
        # The last bucket ends at the session close, not a full hour later.
        assert _session_bar_bounds(_ms(2025, 1, 6, 15, 30), infos, '', 60, 0, 0) == \
               (_ms(2025, 1, 6, 15, 30), _ms(2025, 1, 6, 16))
        # A chart bar before the open is out of session.
        assert _session_bar_bounds(_ms(2025, 1, 6, 9), infos, '', 60, 0, 0) is None

        short = _parse_session_string("0900-1130", "UTC")
        assert _session_bar_bounds(_ms(2025, 1, 6, 11), short, '', 60, 0, 0) == \
               (_ms(2025, 1, 6, 11), _ms(2025, 1, 6, 11, 30))
    finally:
        syminfo.period = period
