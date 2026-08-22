"""
Comma-separated multi-range sessions in ``lib._parse_session_string``.

Pine's session grammar is ``HHMM-HHMM[,HHMM-HHMM...][:days]`` -- a specification
may list several ranges, and the optional day suffix applies to all of them. The
parser only ever read a single range, so ``time(tf, "0400-0700,0900-1300")``
returned na on every bar and a corpus strategy gating its entries on that session
placed zero trades while TradingView placed 870.
"""
from datetime import datetime, UTC

from pynecore.lib import _parse_session_string, _is_bar_in_session


def _ms(year: int, month: int, day: int, hour: int, minute: int = 0) -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1000)


def __test_multi_range_parses_every_range__():
    """Each comma-separated range becomes its own SessionInfo, in source order"""
    infos = _parse_session_string("0400-0700,0900-1300", "UTC")
    assert [(i.start_time.hour, i.end_time.hour) for i in infos] == [(4, 7), (9, 13)]


def __test_multi_range_covers_both_windows__():
    """A bar in either range is in session, one between them is not"""
    # 2021-01-04 is a Monday.
    infos = _parse_session_string("0400-0700,0900-1300", "UTC")
    assert _is_bar_in_session(_ms(2021, 1, 4, 5), infos) is True    # first range
    assert _is_bar_in_session(_ms(2021, 1, 4, 8), infos) is False   # gap
    assert _is_bar_in_session(_ms(2021, 1, 4, 11), infos) is True   # second range
    assert _is_bar_in_session(_ms(2021, 1, 4, 15), infos) is False  # after both


def __test_day_suffix_applies_to_all_ranges__():
    """The trailing ``:days`` filter gates every range, not just the last one"""
    # Days "23456" = Monday..Friday (TV numbering, 1=Sun).
    infos = _parse_session_string("0400-0700,0900-1300:23456", "UTC")
    assert all(i.days == {2, 3, 4, 5, 6} for i in infos)
    assert _is_bar_in_session(_ms(2021, 1, 3, 5), infos) is False   # Sunday
    assert _is_bar_in_session(_ms(2021, 1, 3, 11), infos) is False  # Sunday
    assert _is_bar_in_session(_ms(2021, 1, 4, 11), infos) is True   # Monday


def __test_overnight_range_beside_daytime_range__():
    """An overnight range keeps its midnight-spanning semantics inside a list"""
    infos = _parse_session_string("2200-0200,0900-1300", "UTC")
    assert _is_bar_in_session(_ms(2021, 1, 4, 23), infos) is True
    assert _is_bar_in_session(_ms(2021, 1, 4, 1), infos) is True
    assert _is_bar_in_session(_ms(2021, 1, 4, 5), infos) is False
    assert _is_bar_in_session(_ms(2021, 1, 4, 11), infos) is True


def __test_overnight_range_takes_only_bars_opening_inside_it__():
    """An overnight session follows the same bar-open rule as a same-day one

    TradingView admits a bar only when its OPENING time lies in the run, so a
    bar that merely reaches into the session is out no matter how much of it
    overlaps.
    """
    infos = _parse_session_string("2200-0600", "UTC")
    # 05:30 opens inside -- the last bar of the night.
    assert _is_bar_in_session(_ms(2021, 1, 4, 5, 30), infos) is True
    # 21:30 opens before the session and only reaches into it.
    assert _is_bar_in_session(_ms(2021, 1, 4, 21, 30), infos) is False
    # 22:00 opens exactly at the session open.
    assert _is_bar_in_session(_ms(2021, 1, 4, 22), infos) is True
    # Fully outside on both ends stays out; 06:00 is the exclusive close.
    assert _is_bar_in_session(_ms(2021, 1, 4, 6), infos) is False
    assert _is_bar_in_session(_ms(2021, 1, 4, 12), infos) is False


def __test_malformed_range_still_rejected__():
    """A broken range anywhere in the list invalidates the whole specification"""
    for session in ("0400-0700,0900", "0400-0700,09000-1300", "0400-0700,2500-2600"):
        try:
            _parse_session_string(session, "UTC")
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {session!r}")
