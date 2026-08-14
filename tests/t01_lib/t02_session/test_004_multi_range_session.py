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
    assert _is_bar_in_session(_ms(2021, 1, 4, 5), infos, "60") is True    # first range
    assert _is_bar_in_session(_ms(2021, 1, 4, 8), infos, "60") is False   # gap
    assert _is_bar_in_session(_ms(2021, 1, 4, 11), infos, "60") is True   # second range
    assert _is_bar_in_session(_ms(2021, 1, 4, 15), infos, "60") is False  # after both


def __test_day_suffix_applies_to_all_ranges__():
    """The trailing ``:days`` filter gates every range, not just the last one"""
    # Days "23456" = Monday..Friday (TV numbering, 1=Sun).
    infos = _parse_session_string("0400-0700,0900-1300:23456", "UTC")
    assert all(i.days == {2, 3, 4, 5, 6} for i in infos)
    assert _is_bar_in_session(_ms(2021, 1, 3, 5), infos, "60") is False   # Sunday
    assert _is_bar_in_session(_ms(2021, 1, 3, 11), infos, "60") is False  # Sunday
    assert _is_bar_in_session(_ms(2021, 1, 4, 11), infos, "60") is True   # Monday


def __test_overnight_range_beside_daytime_range__():
    """An overnight range keeps its midnight-spanning semantics inside a list"""
    infos = _parse_session_string("2200-0200,0900-1300", "UTC")
    assert _is_bar_in_session(_ms(2021, 1, 4, 23), infos, "60") is True
    assert _is_bar_in_session(_ms(2021, 1, 4, 1), infos, "60") is True
    assert _is_bar_in_session(_ms(2021, 1, 4, 5), infos, "60") is False
    assert _is_bar_in_session(_ms(2021, 1, 4, 11), infos, "60") is True


def __test_overnight_range_covers_its_boundary_bars__():
    """An overnight session follows the same overlap rule as a same-day one

    A bar counts when any part of it falls inside the session, so the last bar
    of the night (starting before the end, ending after it) and the first one
    (starting before the start, ending after it) are both in session.
    """
    infos = _parse_session_string("2200-0600", "UTC")
    # 05:30 starts inside, ends at 06:30 -- the last bar of the session.
    assert _is_bar_in_session(_ms(2021, 1, 4, 5, 30), infos, "60") is True
    # 21:30 starts before the session, ends at 22:30 -- the first bar.
    assert _is_bar_in_session(_ms(2021, 1, 4, 21, 30), infos, "60") is True
    # 20:00 with a 4h bar runs to midnight, well into the session.
    assert _is_bar_in_session(_ms(2021, 1, 4, 20), infos, "240") is True
    # Fully outside on both ends stays out.
    assert _is_bar_in_session(_ms(2021, 1, 4, 6), infos, "60") is False
    assert _is_bar_in_session(_ms(2021, 1, 4, 12), infos, "60") is False


def __test_malformed_range_still_rejected__():
    """A broken range anywhere in the list invalidates the whole specification"""
    for session in ("0400-0700,0900", "0400-0700,09000-1300", "0400-0700,2500-2600"):
        try:
            _parse_session_string(session, "UTC")
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {session!r}")
