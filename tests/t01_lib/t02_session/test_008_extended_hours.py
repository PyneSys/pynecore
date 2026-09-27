"""
Bars of the extended trading hours: ``session.*`` and the "regular" / "extended" names.

Every expectation is MEASURED on TradingView (2026-09-27) on the NASDAQ:AAPL 60-minute
chart, once on the regular hours (09:30-16:00 New York) and once on the extended
hours (04:00-20:00), 18000 bars each. The bars of the extended chart are cut from
04:00, so the regular open never falls on a bar edge: the regular hours are read by
the bar's open, while the chart's own open and close are the bars holding them.
"""
from datetime import date, datetime, time as dt_time
from zoneinfo import ZoneInfo

import pynecore.lib as lib
from pynecore.lib import session, syminfo, time, time_close
from pynecore.core.syminfo import SymInfoInterval, SymInfoSession

NY = ZoneInfo("America/New_York")
WEEKDAYS = (0, 1, 2, 3, 4)
REGULAR = [SymInfoInterval(day=d, start=dt_time(9, 30), end=dt_time(16)) for d in WEEKDAYS]
EXTENDED = [SymInfoInterval(day=d, start=dt_time(4), end=dt_time(20)) for d in WEEKDAYS]
# The day after Thanksgiving 2024 closes at 13:00, its extended hours at 17:00
EARLY_CLOSE = date(2024, 11, 29)
REGULAR_CORRECTIONS = {EARLY_CLOSE: (SymInfoInterval(day=4, start=dt_time(9, 30), end=dt_time(13)),)}
EXTENDED_CORRECTIONS = {EARLY_CLOSE: (SymInfoInterval(day=4, start=dt_time(4), end=dt_time(17)),)}


def __test_helper_bar(monkeypatch, wall: datetime, extended: bool) -> None:
    """Install the state the script runner sets for one 60-minute bar opening at ``wall``"""
    bar_ms = int(wall.timestamp()) * 1000
    monkeypatch.setattr(lib, "_script_timeframe", None)
    monkeypatch.setattr(lib, "_main_timeframe", None)
    monkeypatch.setattr(lib, "_time", bar_ms)
    monkeypatch.setattr(lib, "_datetime", wall)
    monkeypatch.setattr(lib, "_dg_mode", "")
    monkeypatch.setattr(lib, "_dg_tz", NY)
    monkeypatch.setattr(lib, "_dg_day", None)
    monkeypatch.setattr(syminfo, "period", "60")
    monkeypatch.setattr(syminfo, "type", "stock")
    monkeypatch.setattr(syminfo, "timezone", "America/New_York")
    own = EXTENDED if extended else REGULAR
    monkeypatch.setattr(syminfo, "session", session.extended if extended else session.regular)
    monkeypatch.setattr(syminfo, "_opening_hours", list(own))
    monkeypatch.setattr(syminfo, "_session_starts", [SymInfoSession(day=oh.day, time=oh.start) for oh in own])
    monkeypatch.setattr(syminfo, "_session_ends", [SymInfoSession(day=oh.day, time=oh.end) for oh in own])
    monkeypatch.setattr(syminfo, "session_corrections",
                        EXTENDED_CORRECTIONS if extended else REGULAR_CORRECTIONS, raising=False)
    monkeypatch.setattr(syminfo, "_regular_hours", list(REGULAR))
    monkeypatch.setattr(syminfo, "_regular_corrections", REGULAR_CORRECTIONS)
    monkeypatch.setattr(syminfo, "_extended_hours", list(EXTENDED))


def __test_helper_flags() -> tuple[bool, ...]:
    """(isfirstbar, isfirstbar_regular, islastbar, islastbar_regular, ismarket, ispremarket, ispostmarket)"""
    return (session.isfirstbar(), session.isfirstbar_regular(), session.islastbar(),
            session.islastbar_regular(), session.ismarket(), session.ispremarket(),
            session.ispostmarket())


def __test_extended_chart_flags_by_bar_open__(monkeypatch):
    """ 2025-01-06: the 04:00 bar opens the day, 10:00 opens the regular hours, 15:00 holds
    the close, 19:00 holds the extended close; pre/market/post by the bar's open """
    expected = {
        4: (True, False, False, False, False, True, False),
        9: (False, False, False, False, False, True, False),   # holds 09:30, opens before it
        10: (False, True, False, False, True, False, False),
        12: (False, False, False, False, True, False, False),
        15: (False, False, False, True, True, False, False),   # holds the 16:00 close
        16: (False, False, False, False, False, False, True),
        19: (False, False, True, False, False, False, True),   # holds the 20:00 close
    }
    for hour, flags in expected.items():
        __test_helper_bar(monkeypatch, datetime(2025, 1, 6, hour, tzinfo=NY), extended=True)
        assert __test_helper_flags() == flags, hour


def __test_extended_chart_early_close__(monkeypatch):
    """ 2024-11-29 closes at 13:00 (extended 17:00): 12:00 holds the regular close, the
    13:00 bar is post-market, 16:00 holds the extended close """
    expected = {
        12: (False, False, False, True, True, False, False),
        13: (False, False, False, False, False, False, True),
        16: (False, False, True, False, False, False, True),
    }
    for hour, flags in expected.items():
        __test_helper_bar(monkeypatch, datetime(2024, 11, 29, hour, tzinfo=NY), extended=True)
        assert __test_helper_flags() == flags, hour


def __test_regular_chart_flags__(monkeypatch):
    """ On the regular chart every bar is a market bar; 09:30 opens and 15:30 holds the
    close, 12:30 holds the 13:00 early close """
    __test_helper_bar(monkeypatch, datetime(2025, 1, 6, 9, 30, tzinfo=NY), extended=False)
    assert __test_helper_flags() == (True, True, False, False, True, False, False)
    __test_helper_bar(monkeypatch, datetime(2025, 1, 6, 15, 30, tzinfo=NY), extended=False)
    assert __test_helper_flags() == (False, False, True, True, True, False, False)
    __test_helper_bar(monkeypatch, datetime(2024, 11, 29, 12, 30, tzinfo=NY), extended=False)
    assert __test_helper_flags() == (False, False, True, True, True, False, False)


def __test_session_names_pick_the_template__(monkeypatch):
    """ "regular" is the regular hours and "extended" the extended ones on both charts,
    the chart's own hours being what an unnamed session gives """
    ten = datetime(2025, 1, 6, 10, tzinfo=NY)
    ms = lambda h, m=0: int(datetime(2025, 1, 6, h, m, tzinfo=NY).timestamp()) * 1000
    for extended in (True, False):
        __test_helper_bar(monkeypatch, ten, extended=extended)
        assert time("60", "regular") == ms(9, 30)
        assert time_close("60", "regular") == ms(10, 30)
        assert time("60", "extended") == ms(10)
        assert time("D", "regular") == ms(9, 30)
        assert time_close("D", "regular") == ms(16)
        assert time("D", "extended") == ms(4)
        assert time_close("D", "extended") == ms(20)
        assert time("D", "invalid") == (ms(4) if extended else ms(9, 30))
        assert time("D") == (ms(4) if extended else ms(9, 30))
