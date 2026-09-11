"""
@pyne
"""
from pynecore.lib import close, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="DWM Scheduled Close", shorttitle="DWMC")
def main():
    # A daily context on an hourly equity chart. The pairing runs on the daily
    # bar's SCHEDULED close (16:00), and an hourly bar whose own close falls in
    # the scheduled break already sees it.
    d: Series[float] = request.security(syminfo.tickerid, "D", close)
    plot(d, "d")


# Every timestamp here is Unix MILLISECONDS. 2025-01-06 is a Monday.
_T0 = 1_736_173_800_000  # 2025-01-06T14:30:00Z == 09:30 New York (winter, UTC-5)
_MIN = 60_000
_DAY = 86_400_000
_N_DAYS = 15  # three Mon-Fri weeks
# Hourly bars of a 09:30-16:00 session anchored on the session open. The last
# one (15:30) is shortened: it closes at 16:00 with the session.
_HOUR_OFFSETS_MIN = (0, 60, 120, 180, 240, 300, 360)


def __test_helper_weekday_days():
    """Calendar-day offsets from ``_T0`` that are Mon-Fri."""
    out = []
    for day in range(_N_DAYS + 4):
        if day % 7 < 5:
            out.append(day)
        if len(out) == _N_DAYS:
            break
    return out


def __test_helper_equity_syminfo(period):
    from datetime import time
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    return SymInfo(
        prefix="NYSE", description="Equity", ticker="DWMC",
        currency="USD", period=period, type="stock",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=1,
        timezone="America/New_York", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(9, 30), end=time(16, 0))
                       for i in range(5)],
        session_starts=[SymInfoSession(day=i, time=time(9, 30)) for i in range(5)],
        session_ends=[SymInfoSession(day=i, time=time(16, 0)) for i in range(5)],
    )


def __test_helper_daily_close(day_index):
    return 1000.0 + day_index


def __test_helper_write_daily(tmp_dir):
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "DAILY.ohlcv"
    with OHLCVWriter(path, "1D") as w:
        for i, day in enumerate(__test_helper_weekday_days()):
            c = __test_helper_daily_close(i)
            w.write(OHLCV(timestamp=_T0 + day * _DAY, open=c, high=c, low=c,
                          close=c, volume=1.0))
    __test_helper_equity_syminfo("1D").save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_helper_chart_bars(from_day_index=0):
    from pynecore.types.ohlcv import OHLCV
    out = []
    for i, day in enumerate(__test_helper_weekday_days()):
        if i < from_day_index:
            continue
        for minute in _HOUR_OFFSETS_MIN:
            out.append(OHLCV(timestamp=_T0 + day * _DAY + minute * _MIN,
                             open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0))
    return out


def __test_helper_run(runner, from_day_index, feed):
    si = __test_helper_equity_syminfo("60")
    r = runner(__test_helper_chart_bars(from_day_index),
               syminfo_override=dict(
                   prefix="NYSE", ticker="DWMC", type="stock",
                   timezone="America/New_York", period="60",
                   opening_hours=si.opening_hours,
                   session_starts=si.session_starts,
                   session_ends=si.session_ends),
               security_data={"D": feed})
    rows = {}
    for candle, pv in r.run_iter():
        rows[candle.timestamp] = pv.get("d")
    return rows


def __test_helper_bar_ms(day_index, minute):
    return _T0 + __test_helper_weekday_days()[day_index] * _DAY + minute * _MIN


def __test_hourly_consumer_sees_the_daily_bar_at_the_session_close__(runner, log):
    """An hourly bar sees the day's daily bar only once its own close reaches the break.

    The 09:30 bar closes at 10:30, inside the session, so it still reports
    YESTERDAY's daily close. The 15:30 bar is shortened to the 16:00 session end,
    which lies in the scheduled break the daily bar closes with, so it reports
    TODAY's. The rule runs off the schedule, never off the next existing record.
    """
    import sys
    import tempfile
    from pathlib import Path

    sys.modules.pop(Path(__file__).stem, None)

    with tempfile.TemporaryDirectory() as td:
        feed = __test_helper_write_daily(Path(td))
        rows = __test_helper_run(runner, 0, feed)

    for day in range(1, _N_DAYS):
        first = rows[__test_helper_bar_ms(day, 0)]
        last = rows[__test_helper_bar_ms(day, 360)]
        assert first == __test_helper_daily_close(day - 1), \
            f"day {day} 09:30 bar: d={first} != previous day {__test_helper_daily_close(day - 1)}"
        assert last == __test_helper_daily_close(day), \
            f"day {day} 15:30 bar: d={last} != same day {__test_helper_daily_close(day)}"
    log.info("hourly consumer picks up the daily bar exactly at the session close")


def __test_warmup_and_chart_window_agree__(runner, log):
    """A chart starting mid-feed reproduces the full run's values bar for bar.

    The child replays every daily bar up to the chart's first one in a SINGLE
    warmup round. Without a per-bar publication the consumer would only ever see
    that round's last value, and warmup would differ from the chart window.
    """
    import sys
    import tempfile
    from pathlib import Path

    sys.modules.pop(Path(__file__).stem, None)

    with tempfile.TemporaryDirectory() as td:
        feed = __test_helper_write_daily(Path(td))
        full = __test_helper_run(runner, 0, feed)
        sys.modules.pop(Path(__file__).stem, None)
        late = __test_helper_run(runner, 8, feed)

    assert late, "the late-start run produced no bars"
    for ts, value in late.items():
        assert full[ts] == value, \
            f"ts={ts}: warmup run gave {value}, full run {full[ts]}"
    log.info("warmup and chart window agree on %d bars", len(late))


def __test_scheduled_dwm_closes_are_calendar_ends_not_nominal_spans__(log):
    """``actual_bar_close`` ends a D/W/M bar at its last scheduled session, never nominally.

    An equity weekly bar opening Monday closes Friday 16:00 New York — not the
    next Monday's open; the monthly bar closes on the month's last trading day,
    so its length is not a nominal 30 days. A 24/5 forex week, whose sessions
    open Sunday-Thursday at 17:00, closes Friday 17:00 New York.
    """
    from datetime import datetime, time, timezone
    from zoneinfo import ZoneInfo
    from pynecore.core.security import BarCalendar, actual_bar_close
    from pynecore.core.syminfo import SymInfoInterval, SymInfoSession

    ny = ZoneInfo("America/New_York")

    def ms(dt):
        return int(dt.timestamp() * 1000)

    equity = BarCalendar(
        tz=ny,
        opening_hours=tuple(SymInfoInterval(day=i, start=time(9, 30), end=time(16, 0))
                            for i in range(5)),
        session_starts=tuple(SymInfoSession(day=i, time=time(9, 30)) for i in range(5)),
    )
    # Week of Monday 2025-01-06.
    week_open = ms(datetime(2025, 1, 6, 9, 30, tzinfo=ny))
    friday_close = ms(datetime(2025, 1, 10, 16, 0, tzinfo=ny))
    assert actual_bar_close(week_open, 0, equity, "1W") == friday_close, \
        "an equity weekly bar must close on Friday's session end"

    # January 2025: the last trading day is Friday the 31st.
    month_open = ms(datetime(2025, 1, 2, 9, 30, tzinfo=ny))
    month_close = actual_bar_close(month_open, 0, equity, "1M")
    assert month_close == ms(datetime(2025, 1, 31, 16, 0, tzinfo=ny)), \
        f"monthly close {datetime.fromtimestamp(month_close / 1000, timezone.utc)} is not 01-31 16:00"
    assert month_close - month_open != 30 * 86_400_000, \
        "the monthly length must not be a nominal 30 days"

    # Forex 24/5: sessions open Sunday-Thursday 17:00 and run 24 hours, so the
    # week's last session ends Friday 17:00.
    fx_days = (6, 0, 1, 2, 3)
    fx = BarCalendar(
        tz=ny,
        opening_hours=tuple(SymInfoInterval(day=d, start=time(17, 0), end=time(17, 0))
                            for d in fx_days),
        session_starts=tuple(SymInfoSession(day=d, time=time(17, 0)) for d in fx_days),
    )
    fx_week_open = ms(datetime(2025, 1, 5, 17, 0, tzinfo=ny))
    assert actual_bar_close(fx_week_open, 0, fx, "1W") == \
        ms(datetime(2025, 1, 10, 17, 0, tzinfo=ny)), \
        "a 24/5 forex weekly bar must close Friday 17:00 New York"
    log.info("D/W/M closes follow the schedule, not nominal spans")
