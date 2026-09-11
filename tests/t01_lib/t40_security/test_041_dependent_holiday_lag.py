"""
@pyne
"""
from pynecore.lib import close, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="Holiday Weekly Lag", shorttitle="HWL")
def main():
    # A weekly context read from a DAILY equity chart. The weekly bar's
    # scheduled close is Friday 16:00; a Thursday chart bar's as-of only reaches
    # Friday's 09:30 open, so on a week whose Friday is a holiday the weekly bar
    # first appears on the following Monday. The schedule cannot know about the
    # holiday, and guessing it from the bar grid would mean reading data from
    # past the consumer bar's own close.
    w: Series[float] = request.security(syminfo.tickerid, "W", close)
    dep: Series[float] = request.security(syminfo.tickerid, "W", w + 0.25)
    plot(w, "w")
    plot(dep, "dep")


# Every timestamp here is Unix MILLISECONDS. 2025-01-06 is a Monday.
_T0 = 1_736_173_800_000  # 2025-01-06T14:30:00Z == 09:30 New York (winter, UTC-5)
_DAY = 86_400_000
_N_WEEKS = 5
# Week 2's Friday is a market holiday: the chart simply has no bar for it.
_HOLIDAY_WEEK = 2


def __test_helper_weekly_close(week):
    return 2000.0 + week


def __test_helper_equity_syminfo(period):
    from datetime import time
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    return SymInfo(
        prefix="NYSE", description="Equity", ticker="HWL",
        currency="USD", period=period, type="stock",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=1,
        timezone="America/New_York", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(9, 30), end=time(16, 0))
                       for i in range(5)],
        session_starts=[SymInfoSession(day=i, time=time(9, 30)) for i in range(5)],
        session_ends=[SymInfoSession(day=i, time=time(16, 0)) for i in range(5)],
    )


def __test_helper_write_weekly(tmp_dir):
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "WEEKLY.ohlcv"
    with OHLCVWriter(path, "1W") as w:
        for week in range(_N_WEEKS):
            c = __test_helper_weekly_close(week)
            w.write(OHLCV(timestamp=_T0 + week * 7 * _DAY, open=c, high=c, low=c,
                          close=c, volume=1.0))
    __test_helper_equity_syminfo("1W").save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_helper_chart_days():
    """``(week, weekday)`` pairs the chart has a daily bar for."""
    out = []
    for week in range(_N_WEEKS):
        for weekday in range(5):
            if week == _HOLIDAY_WEEK and weekday == 4:
                continue  # holiday Friday
            out.append((week, weekday))
    return out


def __test_helper_bar_ms(week, weekday):
    return _T0 + (week * 7 + weekday) * _DAY


def __test_weekly_bar_appears_a_bar_late_after_a_holiday_friday__(runner, log):
    """A holiday Friday delays the weekly bar to Monday instead of leaking it on Thursday.

    A normal week's Friday chart bar closes at 16:00, in the scheduled break, so
    it already sees the weekly bar closing at that same instant. When Friday is
    a holiday the last chart bar of the week is Thursday, whose close only
    reaches Friday's 09:30 open — before the weekly close. The value therefore
    arrives on the next Monday. One consumer bar late is the price of never
    reading past one's own close.
    """
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.types.na import NA
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(Path(__file__).stem, None)

    si = __test_helper_equity_syminfo("1D")
    rows = {}
    with tempfile.TemporaryDirectory() as td:
        feed = __test_helper_write_weekly(Path(td))
        bars = [OHLCV(timestamp=__test_helper_bar_ms(week, weekday), open=1.0,
                      high=1.0, low=1.0, close=1.0, volume=1.0)
                for week, weekday in __test_helper_chart_days()]
        r = runner(bars, syminfo_override=dict(
            prefix="NYSE", ticker="HWL", type="stock",
            timezone="America/New_York", period="1D",
            opening_hours=si.opening_hours, session_starts=si.session_starts,
            session_ends=si.session_ends),
            security_data={"W": feed})
        for candle, pv in r.run_iter():
            rows[candle.timestamp] = (pv.get("w"), pv.get("dep"))

    def value(week, weekday):
        return rows[__test_helper_bar_ms(week, weekday)][0]

    # A normal week: Thursday still holds the previous week's bar, Friday brings
    # this week's.
    assert value(1, 3) == __test_helper_weekly_close(0), \
        f"week 1 Thursday: w={value(1, 3)} != week-0 close"
    assert value(1, 4) == __test_helper_weekly_close(1), \
        f"week 1 Friday: w={value(1, 4)} != week-1 close"

    # The holiday week: no Friday bar, so Thursday must NOT anticipate the
    # weekly close, and the value arrives on the next Monday.
    assert value(_HOLIDAY_WEEK, 3) == __test_helper_weekly_close(_HOLIDAY_WEEK - 1), \
        (f"holiday week Thursday: w={value(_HOLIDAY_WEEK, 3)} != "
         f"week-{_HOLIDAY_WEEK - 1} close (lookahead!)")
    assert value(_HOLIDAY_WEEK + 1, 0) == __test_helper_weekly_close(_HOLIDAY_WEEK), \
        (f"Monday after the holiday week: w={value(_HOLIDAY_WEEK + 1, 0)} != "
         f"week-{_HOLIDAY_WEEK} close")

    # The dependent context pairs identically.
    for ts, (direct, dependent) in rows.items():
        if direct is None or isinstance(direct, NA):
            continue
        assert dependent == direct + 0.25, \
            f"ts={ts}: dependent={dependent} != direct+0.25"

    log.info("a holiday Friday delays the weekly bar by exactly one consumer bar")
