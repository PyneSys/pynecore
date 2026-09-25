"""
Session strings of ``time(timeframe, session, timezone)`` / ``time_close(...)``.

Every expectation is MEASURED on TradingView (2026-09-25) with probes on the
CAPITALCOM:AAPL, BTCUSD and GOLD 10-minute charts (New York exchange timezone):
which strings run on the symbol's own session, which ones halt the script (na
here, so a running bot keeps going), how hours and minutes past their range are
read, the day sets of the specification forms, and the daily bar of a multi-range
session. The exceptions are PyneCore's own conventions, which have no TradingView
counterpart: the 23:59:59 end-of-day marker of the round-the-clock providers and a
symbol without opening hours.
"""
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

import pynecore.lib as lib
from pynecore.lib import _parse_session_string, _symbol_session_infos, time, time_close, syminfo, na
from pynecore.core.syminfo import SymInfoInterval, SymInfoSession

NY = "America/New_York"


def __test_helper_symbol(monkeypatch, bar_ms: int, days: tuple[int, ...], start: dt_time,
                         end: dt_time, sym_type: str = "stock") -> None:
    """Install the state the script runner sets for one 10-minute bar of a New York symbol"""
    monkeypatch.setattr(lib, "_script_timeframe", None)
    monkeypatch.setattr(lib, "_main_timeframe", None)
    monkeypatch.setattr(lib, "_time", bar_ms)
    monkeypatch.setattr(lib, "_datetime", datetime.fromtimestamp(bar_ms / 1000, ZoneInfo(NY)))
    monkeypatch.setattr(lib, "_dg_mode", "")
    monkeypatch.setattr(lib, "_dg_tz", ZoneInfo(NY))
    monkeypatch.setattr(lib, "_dg_day", None)
    monkeypatch.setattr(syminfo, "period", "10")
    monkeypatch.setattr(syminfo, "type", sym_type)
    monkeypatch.setattr(syminfo, "timezone", NY)
    monkeypatch.setattr(syminfo, "session_corrections", None, raising=False)
    # Fresh lists on every call: lib caches its per-template tables by list identity
    monkeypatch.setattr(syminfo, "_opening_hours", [
        SymInfoInterval(day=d, start=start, end=end) for d in days])
    monkeypatch.setattr(syminfo, "_session_starts", [
        SymInfoSession(day=d, time=start) for d in days])


def __test_helper_aapl(monkeypatch, bar_ms: int) -> None:
    """CAPITALCOM:AAPL: 09:30-16:00 New York, Monday to Friday"""
    __test_helper_symbol(monkeypatch, bar_ms, (0, 1, 2, 3, 4), dt_time(9, 30), dt_time(16))


def __test_helper_btcusd(monkeypatch, bar_ms: int) -> None:
    """CAPITALCOM:BTCUSD: 17:00 -> 17:00 New York, opening every day"""
    __test_helper_symbol(monkeypatch, bar_ms, (0, 1, 2, 3, 4, 5, 6), dt_time(17), dt_time(17),
                         "crypto")


def __test_helper_ranges(session: str, timezone: str = "UTC") -> list[tuple[str, str, set[int]]]:
    """The parsed ranges as ("HH:MM", "HH:MM", days) triples"""
    return [(info.start_time.strftime("%H:%M"), info.end_time.strftime("%H:%M"), set(info.days))
            for info in _parse_session_string(session, timezone)]


def __test_empty_and_named_sessions_are_the_symbol_session__(monkeypatch):
    """ An empty string or one not starting with a digit runs on the symbol's session """
    # AAPL@10, 2026-09-22: equal to time("60") / time_close("60") on every bar
    for session in ("", " ", "invalid", "regular", "abc-def", "invalid:23456", "abc:23456",
                    "a0930-1600", ",1100-1400", "|1100-1400", "-1400", ":23456", "x"):
        __test_helper_aapl(monkeypatch, 1790083800000)  # 13:30 UTC, the 09:30 open
        assert time("60", session) == 1790083800000, session
        assert time_close("60", session) == 1790087400000, session  # 14:30 UTC
        __test_helper_aapl(monkeypatch, 1790105400000)  # 19:30 UTC, the last bucket
        assert time("60", session) == 1790105400000, session
        assert time_close("60", session) == 1790107200000, session  # 20:00 UTC, the close

    # A timezone argument reads the symbol's session wall clocks in that zone:
    # 09:30-16:00 UTC
    for session in ("", "invalid"):
        __test_helper_aapl(monkeypatch, 1790092200000)  # 15:50 UTC
        assert time("60", session, "UTC") == 1790091000000, session  # 15:30 UTC
        assert time_close("60", session, "UTC") == 1790092800000, session  # 16:00 UTC
        __test_helper_aapl(monkeypatch, 1790096400000)  # 17:00 UTC
        assert na(time("60", session, "UTC")), session
        assert na(time_close("60", session, "UTC")), session


def __test_symbol_session_is_its_opening_hours__(monkeypatch):
    """ Each opening-hours interval is one run, named by the day of its last minute """
    # AAPL, BTCUSD and GOLD: time("D", "") reported the runs 09:30-16:00 Monday to
    # Friday, 17:00 -> 17:00 every day and 18:00 -> 17:00 opening Sunday to Thursday
    __test_helper_aapl(monkeypatch, 1790083800000)
    assert _symbol_session_infos(NY) == _parse_session_string("0930-1600:23456", NY)
    __test_helper_btcusd(monkeypatch, 1790083800000)
    assert _symbol_session_infos(NY) == _parse_session_string("1700-1700:1234567", NY)
    __test_helper_symbol(monkeypatch, 1790083800000, (6, 0, 1, 2, 3), dt_time(18), dt_time(17))
    assert _symbol_session_infos("UTC") == _parse_session_string("1800-1700:23456", "UTC")
    # 23:59:59 is the end-of-day marker of the round-the-clock providers
    __test_helper_symbol(monkeypatch, 1790083800000, (0, 1, 2, 3, 4, 5, 6), dt_time(0),
                         dt_time(23, 59, 59), "crypto")
    assert _symbol_session_infos(NY) == _parse_session_string("0000-0000", NY)
    # Without opening hours the symbol is a continuous market
    monkeypatch.setattr(syminfo, "_opening_hours", [])
    assert _parse_session_string("", NY) == _parse_session_string("0000-0000", NY)


def __test_weekly_and_daily_bars_of_an_evening_open__(monkeypatch):
    """ The symbol's week opens with the run that closes on its Monday """
    # BTCUSD@10: the week of 2026-05-04 opens Sunday 05-03 17:00 New York (21:00 UTC)
    for bar_ms in (1777842000000, 1778359800000):  # Sunday 21:00 UTC, Saturday 20:50 UTC
        __test_helper_btcusd(monkeypatch, bar_ms)
        assert time("W", "") == 1777842000000
        assert time_close("W", "") == 1778446800000  # Sunday 05-10 21:00 UTC
    # Saturday's run: Friday 17:00 -> Saturday 17:00 New York
    assert time("D", "") == 1778274000000
    assert time_close("D", "") == 1778360400000


def __test_malformed_specifications_are_na__(monkeypatch):
    """ A specification that halts the script on TradingView gives na """
    # Measured halts: "bad session", "incorrect entry syntax", "bad session section",
    # "duplicated default section", "Invalid days specification" and "Incorrect value
    # of the `session` parameter"
    __test_helper_btcusd(monkeypatch, 1790083800000)
    for session in ("0930", "1", "0930-16", "0930-1600a", "0930-abcd", "930-1600", "09300-1600",
                    "0930--1600", "0930-1600-1700", "1100-", "1100-1400,,1500-1600",
                    "1100-1400, 1500-1530", "1100-1400 :23456", "1100 -1400", "09:30-16:00",
                    "0930-1600|1700-1800", "1100-1400:|1200-1300", "1100-1400||1200-1300:7",
                    "0930-1600:8", "0930-1600:0", "0930-1600:9", "0930-1600:abc",
                    "0930-1600:23456a", "0930-1600:23456,", "0930-1600:2345 6", "0930-1600F",
                    "2500-2600", "2400-2500", "2500-0100", "2500-0059", "4000-0100",
                    "4700-2300", "4700-1200", "4800-1200", "4800-4800", "5000-4000",
                    "9999-9999"):
        assert na(time("60", session, "UTC")), session
        assert na(time_close("60", session, "UTC")), session
        assert na(time("D", session)), session


def __test_hours_and_minutes_are_counted_modulo_a_day__(monkeypatch):
    """ HHMM is a plain minute count: past 24:00 the range runs into the next day """
    assert __test_helper_ranges("0930-2500") == [("09:30", "01:00", set(range(1, 8)))]
    assert __test_helper_ranges("0930-2459")[0][:2] == ("09:30", "00:59")
    assert __test_helper_ranges("0930-3400")[0][:2] == ("09:30", "10:00")  # same day
    assert __test_helper_ranges("0930-4800")[0][:2] == ("09:30", "00:00")
    assert __test_helper_ranges("0930-9959")[0][:2] == ("09:30", "03:59")
    assert __test_helper_ranges("0960-1600")[0][:2] == ("10:00", "16:00")
    assert __test_helper_ranges("0930-1660")[0][:2] == ("09:30", "17:00")
    assert __test_helper_ranges("0099-0200")[0][:2] == ("01:39", "02:00")
    # A start at 24:00 or later, moved back a day, opens a run of at most a day
    assert __test_helper_ranges("2500-1200")[0][:2] == ("01:00", "12:00")
    assert __test_helper_ranges("2400-0000")[0][:2] == ("00:00", "00:00")  # all day
    assert __test_helper_ranges("2500-2500")[0][:2] == ("01:00", "01:00")  # all day
    assert __test_helper_ranges("4759-4759")[0][:2] == ("23:59", "23:59")  # all day
    assert __test_helper_ranges("2500-0101")[0][:2] == ("01:00", "01:01")
    assert __test_helper_ranges("3000-2900")[0][:2] == ("06:00", "05:00")
    assert __test_helper_ranges("4700-2400")[0][:2] == ("23:00", "00:00")

    # BTCUSD@10, "0930-2500" in UTC: the 00:40 UTC bar is in the last, cut bucket and the
    # 01:00 UTC bar is out of session
    __test_helper_btcusd(monkeypatch, 1790210400000)  # 2026-09-24 00:40 UTC
    assert time("60", "0930-2500", "UTC") == 1790209800000  # 00:30 UTC
    assert time_close("60", "0930-2500", "UTC") == 1790211600000  # 01:00 UTC
    __test_helper_btcusd(monkeypatch, 1790211600000)
    assert na(time("60", "0930-2500", "UTC"))


def __test_day_sets_of_the_specification_forms__():
    """ A lone range runs every day, every other form defaults to the weekdays """
    all_days, weekdays = set(range(1, 8)), set(range(2, 7))
    for session in ("1100-1400", " 1100-1400", "1100-1400 ", "\t1100-1400"):
        assert __test_helper_ranges(session) == [("11:00", "14:00", all_days)], session
    for session in ("1100-1400,", "1100-1400:", "1100-1400|"):
        assert __test_helper_ranges(session) == [("11:00", "14:00", weekdays)], session
    assert __test_helper_ranges("1100-1400,1500-1600") == [
        ("11:00", "14:00", weekdays), ("15:00", "16:00", weekdays)]
    assert __test_helper_ranges("1100-1400:223") == [("11:00", "14:00", {2, 3})]
    assert __test_helper_ranges("24x7") == [("00:00", "00:00", all_days)]


def __test_sections_split_the_days__():
    """ A later section replaces the earlier ones on its days; the default takes the rest """
    assert __test_helper_ranges("1100-1400|1200-1300:7") == [
        ("11:00", "14:00", set(range(2, 7))), ("12:00", "13:00", {7})]
    assert __test_helper_ranges("1100-1400|1200-1300:1") == [
        ("11:00", "14:00", set(range(2, 7))), ("12:00", "13:00", {1})]
    assert __test_helper_ranges("1100-1400:2|1200-1300:2") == [("12:00", "13:00", {2})]
    assert __test_helper_ranges("1100-1400:1234567|1200-1300:7") == [
        ("11:00", "14:00", set(range(1, 7))), ("12:00", "13:00", {7})]
    assert __test_helper_ranges("1100-1400:2|1200-1300:3|1500-1600") == [
        ("11:00", "14:00", {2}), ("12:00", "13:00", {3}), ("15:00", "16:00", {4, 5, 6})]
    assert __test_helper_ranges("1100-1400:23456|1200-1300") == [
        ("11:00", "14:00", set(range(2, 7)))]
    assert __test_helper_ranges("1100-1400:7|1200-1300:") == [
        ("11:00", "14:00", {7}), ("12:00", "13:00", set(range(2, 7)))]


def __test_daily_bar_of_a_multi_range_session_spans_the_trading_day__(monkeypatch):
    """ "D" runs from the day's first range open to its last range close, gaps included """
    # BTCUSD@10, "1100-1400,1500-1600" in UTC, Monday 2026-05-04
    __test_helper_btcusd(monkeypatch, 1777905000000)  # 14:30 UTC, between the ranges
    assert na(time("60", "1100-1400,1500-1600", "UTC"))
    assert time("D", "1100-1400,1500-1600", "UTC") == 1777892400000  # 11:00 UTC
    assert time_close("D", "1100-1400,1500-1600", "UTC") == 1777910400000  # 16:00 UTC
    __test_helper_btcusd(monkeypatch, 1777902000000)  # 13:40 UTC
    assert time("60", "1100-1400,1500-1600", "UTC") == 1777899600000  # 13:00 UTC
    assert time_close("60", "1100-1400,1500-1600", "UTC") == 1777903200000  # 14:00 UTC
    assert time_close("D", "1100-1400,1500-1600", "UTC") == 1777910400000
