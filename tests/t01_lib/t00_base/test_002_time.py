"""
``time()`` and ``time_close()`` with a timeframe, session and timezone argument.

Covers the chart bar's close; the bar of a requested timeframe (intraday, D, nD, W, nW, M,
nM) and how its close depends on the chart's timeframe; an intraday request on a D, W or M
chart; an na, empty or unparsable timeframe; Pine's keyword names; ``bars_back`` over the
chart's bar history and into future bars; ``timeframe_bars_back`` on an intraday, daily and
monthly grid, forward on an intraday one; and the session argument: gating and hourly buckets
in the exchange timezone or in the ``timezone`` argument, the day digits, an overnight
session, the symbol's own session for an empty or non-numeric string, hours past 24 and
malformed specifications.

Every expected value is a TradingView measurement; the comment above each block names the
probe's symbol, chart and bar. The installed session templates are the ones the
TradingView provider builds for the measured CAPITALCOM symbols, all in America/New_York.
"""
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

import pynecore.lib as lib
from pynecore.lib import syminfo, time, time_close, na
from pynecore.core.syminfo import SymInfoInterval, SymInfoSession
from pynecore.types.na import NA

__test_helper_NY = ZoneInfo("America/New_York")
__test_helper_UTC = ZoneInfo("UTC")
__test_helper_HOUR_MS = 3_600_000


def __test_helper_ny(y: int, mo: int, d: int, h: int, mi: int = 0) -> int:
    """Epoch milliseconds of a New York wall-clock time"""
    return int(datetime(y, mo, d, h, mi, tzinfo=__test_helper_NY).timestamp() * 1000)


def __test_helper_utc(y: int, mo: int, d: int, h: int, mi: int = 0) -> int:
    """Epoch milliseconds of a UTC wall-clock time"""
    return int(datetime(y, mo, d, h, mi, tzinfo=__test_helper_UTC).timestamp() * 1000)


def __test_helper_bar(monkeypatch, period: str, bar_ms: int, sym_type: str,
                      session_days: tuple[int, ...], start: dt_time, end: dt_time) -> None:
    """Install the state the script runner sets for one chart bar of a New York symbol"""
    monkeypatch.setattr(lib, "_script_timeframe", None)
    monkeypatch.setattr(lib, "_main_timeframe", None)
    monkeypatch.setattr(lib, "_time", bar_ms)
    monkeypatch.setattr(lib, "_datetime", datetime.fromtimestamp(bar_ms / 1000, __test_helper_NY))
    # The multi-period grid tracker is fed by the runner; without data it resolves arithmetically
    monkeypatch.setattr(lib, "_dg_mode", "")
    monkeypatch.setattr(lib, "_dg_tz", __test_helper_NY)
    monkeypatch.setattr(lib, "_dg_day", None)
    monkeypatch.setattr(syminfo, "period", period)
    monkeypatch.setattr(syminfo, "type", sym_type)
    monkeypatch.setattr(syminfo, "timezone", "America/New_York")
    monkeypatch.setattr(syminfo, "session_corrections", None, raising=False)
    # Fresh lists on every call: lib caches its per-template tables by list identity
    monkeypatch.setattr(syminfo, "_opening_hours", [
        SymInfoInterval(day=d, start=start, end=end) for d in session_days])
    monkeypatch.setattr(syminfo, "_session_starts", [
        SymInfoSession(day=d, time=start) for d in session_days])


def __test_helper_history(monkeypatch, *opens: int) -> None:
    """Install the chart bar history the script runner records, the current bar last"""
    monkeypatch.setattr(lib, "_bar_opens", list(opens))
    monkeypatch.setattr(lib, "bar_index", float(len(opens) - 1))


def __test_helper_eurusd(monkeypatch, period: str, bar_ms: int) -> None:
    """CAPITALCOM:EURUSD: 17:00 -> 17:00, opening Sunday to Thursday"""
    __test_helper_bar(monkeypatch, period, bar_ms, "forex", (6, 0, 1, 2, 3),
                      dt_time(17), dt_time(17))


def __test_helper_btcusd(monkeypatch, period: str, bar_ms: int) -> None:
    """CAPITALCOM:BTCUSD: 17:00 -> 17:00, opening every day"""
    __test_helper_bar(monkeypatch, period, bar_ms, "crypto", (0, 1, 2, 3, 4, 5, 6),
                      dt_time(17), dt_time(17))


def __test_helper_gold(monkeypatch, period: str, bar_ms: int) -> None:
    """CAPITALCOM:GOLD: 18:00 -> 17:00, opening Sunday to Thursday"""
    __test_helper_bar(monkeypatch, period, bar_ms, "cfd", (6, 0, 1, 2, 3),
                      dt_time(18), dt_time(17))


def __test_helper_aapl(monkeypatch, period: str, bar_ms: int) -> None:
    """CAPITALCOM:AAPL: 09:30-16:00, Monday to Friday"""
    __test_helper_bar(monkeypatch, period, bar_ms, "stock", (0, 1, 2, 3, 4),
                      dt_time(9, 30), dt_time(16))


def __test_time_close_is_the_chart_bar_close__(monkeypatch):
    """ ``time_close()`` is the chart bar's close on every chart timeframe """
    ny = __test_helper_ny
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:EURUSD@60): the week's last hourly bar,
    # Friday 2025-11-21 16:00
    __test_helper_eurusd(monkeypatch, "60", ny(2025, 11, 21, 16))
    assert time() == ny(2025, 11, 21, 16)
    assert time_close() == ny(2025, 11, 21, 17)
    assert type(time_close()) is float  # a Pine int is a double at runtime
    # MEASURED (EURUSD@D): the bar opening Thursday 2025-11-20 17:00 is Friday's trading day
    __test_helper_eurusd(monkeypatch, "D", ny(2025, 11, 20, 17))
    assert time_close() == ny(2025, 11, 21, 17)
    # MEASURED (EURUSD@W): the week opening Sunday 2025-11-16 17:00 closes at Friday's
    # session end, not seven days later
    __test_helper_eurusd(monkeypatch, "W", ny(2025, 11, 16, 17))
    assert time_close() == ny(2025, 11, 21, 17)


def __test_intraday_and_daily_requests_resolve_the_containing_bar__(monkeypatch):
    """ "60", "240" and "1D" report the requested bar that contains the chart bar """
    ny = __test_helper_ny
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:EURUSD@60), Friday 2025-11-21 16:00
    __test_helper_eurusd(monkeypatch, "60", ny(2025, 11, 21, 16))
    assert time("60") == ny(2025, 11, 21, 16)
    assert time_close("60") == ny(2025, 11, 21, 17)
    assert time("240") == ny(2025, 11, 21, 13)  # 4-hour grid anchored at the 17:00 open
    assert time_close("240") == ny(2025, 11, 21, 17)
    assert time("1D") == ny(2025, 11, 20, 17)  # the daily bar opens at the session open
    assert time_close("1D") == ny(2025, 11, 21, 17)
    # MEASURED (EURUSD@60): the week's first bar, Sunday 2025-11-23 17:00, opens Monday's
    # trading day
    __test_helper_eurusd(monkeypatch, "60", ny(2025, 11, 23, 17))
    assert time("240") == ny(2025, 11, 23, 17)
    assert time_close("240") == ny(2025, 11, 23, 21)
    assert time("1D") == ny(2025, 11, 23, 17)
    assert time_close("1D") == ny(2025, 11, 24, 17)
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:BTCUSD@60), Friday 2026-02-13 03:00
    __test_helper_btcusd(monkeypatch, "60", ny(2026, 2, 13, 3))
    assert time("60") == ny(2026, 2, 13, 3)
    assert time_close("60") == ny(2026, 2, 13, 4)
    assert time("240") == ny(2026, 2, 13, 1)
    assert time_close("240") == ny(2026, 2, 13, 5)
    assert time("1D") == ny(2026, 2, 12, 17)
    assert time_close("1D") == ny(2026, 2, 13, 17)


def __test_weekly_and_monthly_bars_open_at_the_session_open__(monkeypatch):
    """ W/M bars open at the session open of the period's first trading day """
    ny = __test_helper_ny
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:EURUSD@60), Friday 2025-11-21 16:00: the
    # week opened Sunday 17:00; November's first trading day (Monday 11-03) opened Sunday
    # 11-02 17:00
    __test_helper_eurusd(monkeypatch, "60", ny(2025, 11, 21, 16))
    assert time("1W") == ny(2025, 11, 16, 17)
    assert time("1M") == ny(2025, 11, 2, 17)
    # MEASURED (EURUSD@W): the requested weekly bar is the chart bar itself
    __test_helper_eurusd(monkeypatch, "W", ny(2025, 11, 16, 17))
    assert time("1W") == ny(2025, 11, 16, 17)
    assert time_close("1W") == ny(2025, 11, 21, 17)
    assert time("1M") == ny(2025, 11, 2, 17)
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:BTCUSD@60), Friday 2026-02-13 03:00:
    # February's first trading day opens Saturday 01-31 17:00
    __test_helper_btcusd(monkeypatch, "60", ny(2026, 2, 13, 3))
    assert time("1W") == ny(2026, 2, 8, 17)
    assert time_close("1W") == ny(2026, 2, 15, 17)
    assert time("1M") == ny(2026, 1, 31, 17)
    assert time_close("1M") == ny(2026, 2, 28, 17)


def __test_weekly_and_monthly_close_depends_on_the_chart__(monkeypatch):
    """ On an intraday chart a W/M bar closes when the next period opens """
    ny = __test_helper_ny
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:EURUSD@60), Friday 2025-11-21 16:00: next
    # Sunday 17:00, and the Sunday 17:00 open of December's first trading day
    __test_helper_eurusd(monkeypatch, "60", ny(2025, 11, 21, 16))
    assert time_close("1W") == ny(2025, 11, 23, 17)
    assert time_close("1M") == ny(2025, 11, 30, 17)
    # MEASURED (EURUSD@D), same week: the last session end (Friday 17:00)
    __test_helper_eurusd(monkeypatch, "D", ny(2025, 11, 20, 17))
    assert time_close("1W") == ny(2025, 11, 21, 17)
    assert time_close("1M") == ny(2025, 11, 28, 17)


def __test_session_weekly_and_monthly_close_depends_on_the_chart__(monkeypatch):
    """ With a session, a W/M bar closes at its last run's end on a D, W or M chart """
    ny = __test_helper_ny
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:EURUSD@D), Thursday 2025-11-20 17:00:
    # the symbol's session ends the week at Friday 17:00, "0000-2359" -- a lone range, so it
    # runs every day -- at Sunday 23:59
    __test_helper_eurusd(monkeypatch, "D", ny(2025, 11, 20, 17))
    assert time("W", "") == ny(2025, 11, 16, 17)
    assert time_close("W", "") == ny(2025, 11, 21, 17)
    assert time_close("M", "") == ny(2025, 11, 28, 17)
    assert time("W", "0000-2359") == ny(2025, 11, 17, 0)
    assert time_close("W", "0000-2359") == ny(2025, 11, 23, 23, 59)
    assert time_close("M", "0000-2359") == ny(2025, 11, 30, 23, 59)
    assert time_close("W", "", timeframe_bars_back=1) == ny(2025, 11, 14, 17)
    # MEASURED (EURUSD@W), the week opening Sunday 2025-11-16 17:00
    __test_helper_eurusd(monkeypatch, "W", ny(2025, 11, 16, 17))
    assert time_close("W", "") == ny(2025, 11, 21, 17)
    assert time_close("W", "0000-2359") == ny(2025, 11, 16, 23, 59)
    # MEASURED (EURUSD@60), Friday 2025-11-21 16:00: the next period's first session open
    __test_helper_eurusd(monkeypatch, "60", ny(2025, 11, 21, 16))
    assert time_close("W", "") == ny(2025, 11, 23, 17)
    assert time_close("M", "") == ny(2025, 11, 30, 17)
    assert time_close("W", "0000-2359") == ny(2025, 11, 24, 0)
    assert time_close("M", "0000-2359") == ny(2025, 12, 1, 0)


def __test_multi_period_close_depends_on_the_chart__(monkeypatch):
    """ nD/nW/nM bars close at the next period's open on an intraday chart """
    ny = __test_helper_ny
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:EURUSD@60), Wednesday 2026-09-23 18:00:
    # the 2D bar holds Thursday and Friday
    __test_helper_eurusd(monkeypatch, "60", ny(2026, 9, 23, 18))
    assert time("2D") == ny(2026, 9, 23, 17)
    assert time_close("2D") == ny(2026, 9, 27, 17)
    assert time("3D") == ny(2026, 9, 22, 17)
    assert time_close("3D") == ny(2026, 9, 27, 17)
    assert time("2W") == ny(2026, 9, 13, 17)
    assert time_close("2W") == ny(2026, 9, 27, 17)
    assert time("3M") == ny(2026, 6, 30, 17)
    assert time_close("3M") == ny(2026, 9, 30, 17)
    assert time("12M") == ny(2025, 12, 31, 17)
    assert time_close("12M") == ny(2026, 12, 31, 17)
    # MEASURED (EURUSD@D), Thursday 2025-11-20 17:00: the 2D bar (Friday + Monday) closes
    # at Monday's session end
    __test_helper_eurusd(monkeypatch, "D", ny(2025, 11, 20, 17))
    assert time("2D") == ny(2025, 11, 20, 17)
    assert time_close("2D") == ny(2025, 11, 24, 17)
    # MEASURED (EURUSD@2D): the same bar as the chart bar
    __test_helper_eurusd(monkeypatch, "2D", ny(2025, 11, 20, 17))
    assert time_close() == ny(2025, 11, 24, 17)
    assert time_close("2D") == ny(2025, 11, 24, 17)


def __test_period_close_across_a_session_break__(monkeypatch):
    """ A break between sessions separates a period's last session end from the next open """
    ny = __test_helper_ny
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:GOLD@60), Friday 2025-11-21 16:00
    __test_helper_gold(monkeypatch, "60", ny(2025, 11, 21, 16))
    assert time_close() == ny(2025, 11, 21, 17)
    assert time("D") == ny(2025, 11, 20, 18)
    assert time_close("D") == ny(2025, 11, 21, 17)
    assert time("2D") == ny(2025, 11, 20, 18)
    assert time_close("2D") == ny(2025, 11, 24, 18)
    assert time("W") == ny(2025, 11, 16, 18)
    assert time_close("W") == ny(2025, 11, 23, 18)
    assert time("M") == ny(2025, 11, 2, 18)
    assert time_close("M") == ny(2025, 11, 30, 18)
    # MEASURED (GOLD@D), Thursday 2025-11-20 18:00: every period closes at its last
    # session end
    __test_helper_gold(monkeypatch, "D", ny(2025, 11, 20, 18))
    assert time_close() == ny(2025, 11, 21, 17)
    assert time_close("2D") == ny(2025, 11, 24, 17)
    assert time_close("W") == ny(2025, 11, 21, 17)
    assert time_close("M") == ny(2025, 11, 28, 17)
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:AAPL@60), Friday 2025-08-29 15:30: the
    # next week and month open on Labor Day, a holiday the session template does not know
    __test_helper_aapl(monkeypatch, "60", ny(2025, 8, 29, 15, 30))
    assert time_close() == ny(2025, 8, 29, 16)
    assert time("D") == ny(2025, 8, 29, 9, 30)
    assert time_close("D") == ny(2025, 8, 29, 16)
    assert time("W") == ny(2025, 8, 25, 9, 30)
    assert time_close("W") == ny(2025, 9, 1, 9, 30)
    assert time("M") == ny(2025, 8, 1, 9, 30)
    assert time_close("M") == ny(2025, 9, 1, 9, 30)


def __test_daily_close_is_the_trading_day_end__(monkeypatch):
    """ On a clock-change day the daily bar is shorter and cuts the 4-hour bar """
    ny = __test_helper_ny
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:BTCUSD@60), Sunday 2026-03-08 15:00, the
    # day New York moves from EST to EDT
    __test_helper_btcusd(monkeypatch, "60", ny(2026, 3, 8, 15))
    assert time_close() == ny(2026, 3, 8, 16)
    assert time("1D") == ny(2026, 3, 7, 17)  # 17:00 EST
    assert time_close("1D") == ny(2026, 3, 8, 17)  # 17:00 EDT
    assert time_close("1D") - time("1D") == 23 * __test_helper_HOUR_MS
    # The 4-hour grid shifted by the clock change: its 14:00 bar is cut at the 17:00 roll
    assert time("240") == ny(2026, 3, 8, 14)
    assert time_close("240") == ny(2026, 3, 8, 17)


def __test_requested_bar_contains_the_chart_bar__(monkeypatch):
    """ time(tf) <= time, time_close <= time_close(tf) and time(tf) <= time_close(tf) """
    ny = __test_helper_ny
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:EURUSD@60): the 121 hourly bars from
    # Sunday 2025-11-16 17:00 to Friday 16:00 plus the next week's first bar are the chart's
    # bars, and TradingView's values satisfy the three relations on every one of them (and on
    # all 23240 bars of the probe, and of the BTCUSD, GOLD and AAPL 60-minute probes)
    start = ny(2025, 11, 16, 17)
    bars = [start + i * __test_helper_HOUR_MS for i in range(120)] + [ny(2025, 11, 23, 17)]
    for bar in bars:
        __test_helper_eurusd(monkeypatch, "60", bar)
        for tf in ("60", "240", "1D", "1W", "1M"):
            assert time(tf) <= time(), (tf, bar)
            assert time_close() <= time_close(tf), (tf, bar)
            assert time(tf) <= time_close(tf), (tf, bar)


def __test_intraday_request_on_a_dwm_chart_resolves_as_daily__(monkeypatch):
    """ A daily, weekly or monthly chart resolves an intraday request as "D" """
    ny = __test_helper_ny
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:EURUSD@D), Thursday 2025-11-20 17:00
    # (Friday's trading day)
    __test_helper_eurusd(monkeypatch, "D", ny(2025, 11, 20, 17))
    for tf in ("1", "30", "60", "240"):
        assert time(tf) == ny(2025, 11, 20, 17), tf
        assert time_close(tf) == ny(2025, 11, 21, 17), tf
    # A session filters the daily bar: the 00:00-23:59 session of the chart bar's date
    assert time("60", "0000-2359") == ny(2025, 11, 20, 0)
    assert time_close("60", "0000-2359", "UTC") == ny(2025, 11, 20, 18, 59)
    assert na(time("60", "1000-1400"))
    # MEASURED (EURUSD@W), the week opening Sunday 2025-11-16 17:00: the trading day the
    # weekly bar opens with
    __test_helper_eurusd(monkeypatch, "W", ny(2025, 11, 16, 17))
    for tf in ("1", "30", "60", "240"):
        assert time(tf) == ny(2025, 11, 16, 17), tf
        assert time_close(tf) == ny(2025, 11, 17, 17), tf
    assert time("60", "0000-2359") == ny(2025, 11, 16, 0)
    # MEASURED (EURUSD@M), November 2025, opening Sunday 11-02 17:00
    __test_helper_eurusd(monkeypatch, "M", ny(2025, 11, 2, 17))
    assert time_close() == ny(2025, 11, 28, 17)
    for tf in ("1", "60", "240"):
        assert time(tf) == ny(2025, 11, 2, 17), tf
        assert time_close(tf) == ny(2025, 11, 3, 17), tf


def __test_keyword_arguments_use_pine_names__(monkeypatch):
    """ Pine's keyword names select the timeframe, the session and its timezone """
    ny = __test_helper_ny
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:EURUSD@60), Tuesday 2025-11-18 18:00 New
    # York = 23:00 UTC: time(timeframe="60", session="0000-2359") and
    # time_close(timeframe="60", session="0000-2359", timezone="UTC")
    __test_helper_eurusd(monkeypatch, "60", ny(2025, 11, 18, 18))
    assert time(timeframe="60", session="0000-2359") == ny(2025, 11, 18, 18)
    # Read in UTC the session ends at 23:59 UTC and cuts the 23:00 UTC bar
    close = time_close(timeframe="60", session="0000-2359", timezone="UTC")
    assert close == ny(2025, 11, 18, 18, 59)


def __test_intraday_offset_skips_the_session_break__(monkeypatch):
    """ ``timeframe_bars_back`` on an intraday grid steps over the time between sessions """
    ny = __test_helper_ny
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:EURUSD@60): time("60",
    # timeframe_bars_back=1) on the week's first bar is Friday's last hourly bar
    __test_helper_eurusd(monkeypatch, "60", ny(2025, 11, 16, 17))
    assert time("60", timeframe_bars_back=1) == ny(2025, 11, 14, 16)
    assert time_close("60", timeframe_bars_back=1) == ny(2025, 11, 14, 17)
    __test_helper_eurusd(monkeypatch, "60", ny(2025, 11, 16, 18))
    assert time("60", timeframe_bars_back=1) == ny(2025, 11, 16, 17)
    assert time_close("60", timeframe_bars_back=1) == ny(2025, 11, 16, 18)


def __test_negative_intraday_offset_walks_the_session_schedule__(monkeypatch):
    """ A negative ``timeframe_bars_back`` on an intraday grid steps over the session break """
    ny = __test_helper_ny
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:EURUSD@60): on the week's first bar,
    # Sunday 2023-01-08 17:00, bars_back 1 is Friday's 16:00 bar, and the hourly bar after
    # that one is the Sunday 17:00 open
    __test_helper_eurusd(monkeypatch, "60", ny(2023, 1, 8, 17))
    __test_helper_history(monkeypatch, ny(2023, 1, 6, 16), ny(2023, 1, 8, 17))
    assert time("60", bars_back=1, timeframe_bars_back=-1) == ny(2023, 1, 8, 17)
    assert time_close("60", bars_back=1, timeframe_bars_back=-1) == ny(2023, 1, 8, 18)


def __test_bars_back_evaluates_on_an_earlier_chart_bar__(monkeypatch):
    """ ``bars_back`` evaluates the call on the chart bar that many bars back """
    ny = __test_helper_ny
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:EURUSD@60): the week's first bar, Sunday
    # 2023-01-08 17:00, looks back over the weekend to Friday's last bars
    __test_helper_eurusd(monkeypatch, "60", ny(2023, 1, 8, 17))
    __test_helper_history(monkeypatch, ny(2023, 1, 6, 14), ny(2023, 1, 6, 15),
                          ny(2023, 1, 6, 16), ny(2023, 1, 8, 17))
    assert time("", 1) == ny(2023, 1, 6, 16)
    assert time_close("", 1) == ny(2023, 1, 6, 17)
    assert time("", 3) == ny(2023, 1, 6, 14)
    assert time("60", 1) == ny(2023, 1, 6, 16)
    assert time("D", 1) == ny(2023, 1, 5, 17)
    assert time_close("D", 1) == ny(2023, 1, 6, 17)
    assert time("D", bars_back=1, timeframe_bars_back=1) == ny(2023, 1, 4, 17)
    assert time("60", bars_back=2, timeframe_bars_back=1) == ny(2023, 1, 6, 14)
    # MEASURED (EURUSD@60): the data has no 18:00 bar on its first Sunday, 2023-01-01, so
    # the 19:00 bar's previous one is 17:00, and no bar lies three back
    __test_helper_eurusd(monkeypatch, "60", ny(2023, 1, 1, 19))
    __test_helper_history(monkeypatch, ny(2023, 1, 1, 17), ny(2023, 1, 1, 19))
    assert time("", 1) == ny(2023, 1, 1, 17)
    assert time_close("", 1) == ny(2023, 1, 1, 18)
    assert na(time("", 3))
    # MEASURED (EURUSD@60): the first bar has no previous one
    __test_helper_eurusd(monkeypatch, "60", ny(2023, 1, 1, 17))
    __test_helper_history(monkeypatch, ny(2023, 1, 1, 17))
    assert na(time("", 1))
    assert na(time_close("", 1))


def __test_negative_bars_back_is_the_next_scheduled_chart_bar__(monkeypatch):
    """ A negative ``bars_back`` walks the session schedule, blind to holidays and gaps """
    ny = __test_helper_ny
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:EURUSD@60): the week's last bar, Friday
    # 2023-01-06 16:00, is followed by the Sunday 17:00 open
    __test_helper_eurusd(monkeypatch, "60", ny(2023, 1, 6, 16))
    assert time("", -1) == ny(2023, 1, 8, 17)
    assert time_close("", -1) == ny(2023, 1, 8, 18)
    assert time("60", -3) == ny(2023, 1, 8, 19)
    assert time("D", -1) == ny(2023, 1, 8, 17)
    assert time_close("D", -1) == ny(2023, 1, 9, 17)
    # MEASURED (EURUSD@60): the data's first Sunday has no 18:00 bar, the schedule has one
    __test_helper_eurusd(monkeypatch, "60", ny(2023, 1, 1, 17))
    assert time("", -1) == ny(2023, 1, 1, 18)
    # MEASURED (EURUSD@60): Christmas Day has no bars, yet the schedule opens its trading
    # day on Christmas Eve at 17:00
    __test_helper_eurusd(monkeypatch, "60", ny(2024, 12, 24, 16))
    assert time("", -1) == ny(2024, 12, 24, 17)
    assert time_close("D", -1) == ny(2024, 12, 25, 17)


def __test_daily_and_monthly_offset_steps_to_the_previous_trading_day__(monkeypatch):
    """ A D/M ``timeframe_bars_back`` step lands on the session template's previous day """
    ny = __test_helper_ny
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:AAPL@60): Labor Day, a template day
    # without data, is the day before Tuesday 2025-09-02, and a Monday steps over the weekend
    __test_helper_aapl(monkeypatch, "60", ny(2025, 9, 2, 10, 30))
    assert time("D", timeframe_bars_back=1) == ny(2025, 9, 1, 9, 30)
    assert time_close("D", timeframe_bars_back=1) == ny(2025, 9, 1, 16)
    assert time("D", timeframe_bars_back=2) == ny(2025, 8, 29, 9, 30)
    __test_helper_aapl(monkeypatch, "60", ny(2025, 11, 17, 10, 30))
    assert time("D", timeframe_bars_back=1) == ny(2025, 11, 14, 9, 30)
    assert time("M", timeframe_bars_back=1) == ny(2025, 10, 1, 9, 30)
    # MEASURED (EURUSD@60): the week's first bar, Sunday 2025-11-16 17:00, steps to Friday's
    # trading day, not into the Sunday before the open
    __test_helper_eurusd(monkeypatch, "60", ny(2025, 11, 16, 17))
    assert time("D", timeframe_bars_back=1) == ny(2025, 11, 13, 17)
    assert time_close("D", timeframe_bars_back=1) == ny(2025, 11, 14, 17)
    assert time("D", timeframe_bars_back=2) == ny(2025, 11, 12, 17)
    # MEASURED (EURUSD@D), the same bar: an intraday request resolves as "D" there
    __test_helper_eurusd(monkeypatch, "D", ny(2025, 11, 16, 17))
    assert time("60", timeframe_bars_back=1) == ny(2025, 11, 13, 17)
    assert time_close("60", timeframe_bars_back=1) == ny(2025, 11, 14, 17)
    # MEASURED (GOLD@60): October opens Tuesday 2025-09-30 18:00, an hour after the 17:00
    # close; one month back is September, opening Sunday 08-31 18:00
    __test_helper_gold(monkeypatch, "60", ny(2025, 9, 30, 18))
    assert time("D", timeframe_bars_back=1) == ny(2025, 9, 29, 18)
    assert time_close("D", timeframe_bars_back=1) == ny(2025, 9, 30, 17)
    assert time("M", timeframe_bars_back=1) == ny(2025, 8, 31, 18)


def __test_na_and_empty_timeframe_are_the_chart_timeframe__(monkeypatch):
    """ An na or empty timeframe string selects the chart's timeframe """
    ny = __test_helper_ny
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:EURUSD@60, 50 bars): time(na) ==
    # time("") == time and time_close(na) == time_close("") == time_close on every bar; here
    # the probe's first bar, Wednesday 2026-09-23 05:00
    __test_helper_eurusd(monkeypatch, "60", ny(2026, 9, 23, 5))
    assert time(NA(str)) == ny(2026, 9, 23, 5)
    assert time("") == ny(2026, 9, 23, 5)
    assert time_close(NA(str)) == ny(2026, 9, 23, 6)
    assert time_close("") == ny(2026, 9, 23, 6)


def __test_invalid_timeframe_is_na__(monkeypatch):
    """ An unparsable timeframe gives na instead of halting the script """
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:EURUSD@60): time("invalid"), and time()
    # and time_close() of a series string "invalid", halt the script on bar 0 with "Cannot
    # parse resolution 'invalid'"; PyneCore returns na so a running script keeps going
    __test_helper_eurusd(monkeypatch, "60", __test_helper_ny(2026, 9, 23, 3))
    assert na(time("invalid"))
    assert na(time_close("invalid"))


def __test_session_gates_and_tiles_in_the_exchange_timezone__(monkeypatch):
    """ Without a timezone argument the session is read in the exchange timezone """
    utc = __test_helper_utc
    session = "0930-1600:23456"
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:BTCUSD@10, exchange America/New_York),
    # Monday 2026-09-21: 09:30-16:00 EDT is 13:30-20:00 UTC, tiled in hours from the open.
    # 10:10 UTC (06:10 EDT) is before the open
    __test_helper_btcusd(monkeypatch, "10", utc(2026, 9, 21, 10, 10))
    assert na(time("60", session))
    assert na(time_close("60", session))
    __test_helper_btcusd(monkeypatch, "10", utc(2026, 9, 21, 15, 10))
    assert time("60", session) == utc(2026, 9, 21, 14, 30)
    assert time_close("60", session) == utc(2026, 9, 21, 15, 30)
    __test_helper_btcusd(monkeypatch, "10", utc(2026, 9, 21, 17, 10))
    assert time("60", session) == utc(2026, 9, 21, 16, 30)
    assert time_close("60", session) == utc(2026, 9, 21, 17, 30)


def __test_session_timezone_argument_overrides_the_exchange_timezone__(monkeypatch):
    """ A timezone argument reads the session's wall clocks in that timezone """
    utc = __test_helper_utc
    session = "0930-1600:23456"
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:BTCUSD@10, exchange America/New_York),
    # Monday 2026-09-21, the session read in UTC
    __test_helper_btcusd(monkeypatch, "10", utc(2026, 9, 21, 10, 10))
    assert time("60", session, "UTC") == utc(2026, 9, 21, 9, 30)
    assert time_close("60", session, "UTC") == utc(2026, 9, 21, 10, 30)
    __test_helper_btcusd(monkeypatch, "10", utc(2026, 9, 21, 17, 10))  # after 16:00 UTC
    assert na(time("60", session, "UTC"))
    assert na(time_close("60", session, "UTC"))
    # A session opening at midnight tiles its hours from midnight
    __test_helper_btcusd(monkeypatch, "10", utc(2026, 9, 21, 15, 10))
    assert time("60", "0000-2359:23456", "UTC") == utc(2026, 9, 21, 15)


def __test_session_last_bucket_ends_at_the_session_close__(monkeypatch):
    """ The last hourly bucket of a session is cut at the session's close """
    utc = __test_helper_utc
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:BTCUSD@10), Monday 2026-09-21, both with
    # and without the day digits
    for session in ("0000-2359:1234567", "0000-2359"):
        __test_helper_btcusd(monkeypatch, "10", utc(2026, 9, 21, 15, 10))
        assert time("60", session, "UTC") == utc(2026, 9, 21, 15), session
        assert time_close("60", session, "UTC") == utc(2026, 9, 21, 16), session
        __test_helper_btcusd(monkeypatch, "10", utc(2026, 9, 21, 23, 10))
        assert time("60", session, "UTC") == utc(2026, 9, 21, 23), session
        assert time_close("60", session, "UTC") == utc(2026, 9, 21, 23, 59), session


def __test_session_day_digits_run_from_sunday__(monkeypatch):
    """ Session day digits run 1 = Sunday ... 7 = Saturday """
    utc = __test_helper_utc
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:BTCUSD@10, sessions read in UTC, 1500 bars
    # from 2026-09-14): "0930-1600:<days>" on the 15:10 UTC bar, in the 14:30 bucket or na
    in_session = {
        utc(2026, 9, 20, 15, 10): ("1", "17", "1234567"),  # Sunday
        utc(2026, 9, 21, 15, 10): ("2", "23456", "1234567"),  # Monday
        utc(2026, 9, 18, 15, 10): ("67", "23456", "1234567"),  # Friday
        utc(2026, 9, 19, 15, 10): ("67", "17", "1234567"),  # Saturday
    }
    for bar_ms, days_in in in_session.items():
        __test_helper_btcusd(monkeypatch, "10", bar_ms)
        for days in ("1", "2", "67", "17", "23456", "1234567"):
            result = time("60", "0930-1600:" + days, "UTC")
            if days in days_in:
                assert result == bar_ms - 40 * 60_000, (bar_ms, days)
            else:
                assert na(result), (bar_ms, days)
    # MEASURED: the close of the Friday and Saturday bucket
    for bar_ms in (utc(2026, 9, 18, 15, 10), utc(2026, 9, 19, 15, 10)):
        __test_helper_btcusd(monkeypatch, "10", bar_ms)
        assert time_close("60", "0930-1600:67", "UTC") == bar_ms + 20 * 60_000
    __test_helper_btcusd(monkeypatch, "10", utc(2026, 9, 18, 15, 10))
    assert time_close("60", "0930-1600:23456", "UTC") == utc(2026, 9, 18, 15, 30)


def __test_overnight_session__(monkeypatch):
    """ An overnight session runs past midnight and is na between its close and next open """
    utc = __test_helper_utc
    session = "2200-0600:1234567"
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:BTCUSD@10, session read in UTC)
    __test_helper_btcusd(monkeypatch, "10", utc(2026, 9, 21, 15, 10))
    assert na(time("60", session, "UTC"))
    assert na(time_close("60", session, "UTC"))
    __test_helper_btcusd(monkeypatch, "10", utc(2026, 9, 21, 21, 50))
    assert na(time("60", session, "UTC"))
    __test_helper_btcusd(monkeypatch, "10", utc(2026, 9, 21, 22, 0))
    assert time("60", session, "UTC") == utc(2026, 9, 21, 22)
    assert time_close("60", session, "UTC") == utc(2026, 9, 21, 23)
    __test_helper_btcusd(monkeypatch, "10", utc(2026, 9, 21, 23, 10))
    assert time("60", session, "UTC") == utc(2026, 9, 21, 23)
    assert time_close("60", session, "UTC") == utc(2026, 9, 22, 0)
    __test_helper_btcusd(monkeypatch, "10", utc(2026, 9, 22, 5, 10))
    assert time("60", session, "UTC") == utc(2026, 9, 22, 5)
    assert time_close("60", session, "UTC") == utc(2026, 9, 22, 6)
    __test_helper_btcusd(monkeypatch, "10", utc(2026, 9, 22, 6, 10))
    assert na(time("60", session, "UTC"))


def __test_empty_or_non_numeric_session_is_the_symbol_session__(monkeypatch):
    """ An empty session or one not starting with a digit runs on the symbol's session """
    utc = __test_helper_utc
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:AAPL@10, 09:30-16:00 New York), Tuesday
    # 2026-09-22 13:30 UTC, the 09:30 EDT open: "" and "invalid" equal time("60"), while
    # the all-day "0000-0000" tiles its hours from midnight
    __test_helper_aapl(monkeypatch, "10", utc(2026, 9, 22, 13, 30))
    for session in ("", "invalid"):
        assert time("60", session) == time("60") == utc(2026, 9, 22, 13, 30), session
        assert time_close("60", session) == time_close("60") == utc(2026, 9, 22, 14, 30), session
    assert time("60", "0000-0000") == utc(2026, 9, 22, 13)
    assert time_close("60", "0000-0000") == utc(2026, 9, 22, 14)
    assert time("60", "0000-0000", "UTC") == utc(2026, 9, 22, 13)
    # MEASURED (AAPL@10): with a timezone argument the symbol's 09:30-16:00 is read in UTC
    assert time("60", "invalid", "UTC") == utc(2026, 9, 22, 13, 30)
    __test_helper_aapl(monkeypatch, "10", utc(2026, 9, 22, 19, 30))
    assert na(time("60", "invalid", "UTC"))

    # MEASURED (TradingView 2026-09-25, CAPITALCOM:BTCUSD@10), Monday 2026-09-21: the
    # symbol's 17:00 -> 17:00 run read in UTC tiles its hours from 17:00 UTC and, unlike
    # "0000-2359", is not cut before midnight
    __test_helper_btcusd(monkeypatch, "10", utc(2026, 9, 21, 10, 10))
    assert time("60", "", "UTC") == utc(2026, 9, 21, 10)
    assert time_close("60", "", "UTC") == utc(2026, 9, 21, 11)
    __test_helper_btcusd(monkeypatch, "10", utc(2026, 9, 21, 23, 10))
    assert time("60", "", "UTC") == utc(2026, 9, 21, 23)
    assert time_close("60", "", "UTC") == utc(2026, 9, 22, 0)


def __test_session_end_hour_past_24_runs_into_the_next_day__(monkeypatch):
    """ An end hour of 25 is 01:00 of the next day, not an invalid session """
    utc = __test_helper_utc
    session = "0930-2500"
    # MEASURED (TradingView 2026-09-25, CAPITALCOM:BTCUSD@10, session read in UTC, 300 bars
    # from 2026-09-23 08:40 UTC): hourly buckets from 09:30 to 01:00 of the next day, na
    # from 01:00 to 09:20 UTC
    __test_helper_btcusd(monkeypatch, "10", utc(2026, 9, 23, 15, 10))
    assert time("60", session, "UTC") == utc(2026, 9, 23, 14, 30)
    assert time_close("60", session, "UTC") == utc(2026, 9, 23, 15, 30)
    __test_helper_btcusd(monkeypatch, "10", utc(2026, 9, 24, 0, 10))
    assert time("60", session, "UTC") == utc(2026, 9, 23, 23, 30)
    assert time_close("60", session, "UTC") == utc(2026, 9, 24, 0, 30)
    __test_helper_btcusd(monkeypatch, "10", utc(2026, 9, 24, 0, 50))
    assert time("60", session, "UTC") == utc(2026, 9, 24, 0, 30)
    assert time_close("60", session, "UTC") == utc(2026, 9, 24, 1)
    __test_helper_btcusd(monkeypatch, "10", utc(2026, 9, 24, 5))
    assert na(time("60", session, "UTC"))
    assert na(time_close("60", session, "UTC"))
    __test_helper_btcusd(monkeypatch, "10", utc(2026, 9, 24, 9, 30))
    assert time("60", session, "UTC") == utc(2026, 9, 24, 9, 30)
    assert time_close("60", session, "UTC") == utc(2026, 9, 24, 10, 30)


def __test_malformed_numeric_session_is_na__(monkeypatch):
    """ A malformed session starting with a digit gives na instead of halting the script """
    # MEASURED (TradingView 2026-09-25): the script halts on bar 0 with "bad session - 0930"
    # (CAPITALCOM:AAPL@10) and "Invalid days specification - 8" (CAPITALCOM:BTCUSD@10).
    # PyneCore returns na so a running script keeps going; the bar is inside 09:30-16:00
    # both in New York and in UTC
    __test_helper_btcusd(monkeypatch, "10", __test_helper_utc(2026, 9, 21, 15, 10))
    for session in ("0930", "0930-1600:8"):
        assert na(time("60", session)), session
        assert na(time_close("60", session)), session
        assert na(time("60", session, "UTC")), session
        assert na(time_close("60", session, "UTC")), session
