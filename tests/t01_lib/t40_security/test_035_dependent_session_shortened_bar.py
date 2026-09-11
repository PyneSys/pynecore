"""
@pyne
"""
from pynecore.lib import close, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="Session Shortened Bar Peer", shorttitle="SSBP")
def main():
    # A 20-minute peer read from a 60-minute chart whose last bar of the session
    # is SHORTENED (12:30 opens, 13:00 closes). The pairing runs on scheduled
    # closes, so that bar sees the 12:50 intrabar (closing 13:00 with the
    # session) and never the 13:10 record, which the NOMINAL 13:30 span would
    # have admitted — data from after the chart bar closed.
    p20: Series[float] = request.security(syminfo.tickerid, "20", close)
    plot(p20, "p20")


# Every timestamp here is Unix MILLISECONDS.
_T0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC
_MIN = 60_000
_DAY = 86_400_000
_N_DAYS = 3
# Session 08:30 -> 13:00. The 60-minute grid anchors on the session open, so the
# day's bars are 08:30, 09:30, 10:30, 11:30 and the shortened 12:30 one.
_SESSION_START_MIN = 8 * 60 + 30
_CHART_OPEN_MINS = (510, 570, 630, 690, 750)
# 20-minute intrabars from the session open, plus one straddling record at 13:10
# that no chart bar of this session may ever see.
_LTF_OPEN_MINS = (510, 530, 550, 570, 590, 610, 630, 650, 670, 690, 710, 730, 750, 770, 790)


def __test_helper_encode(minute_of_day):
    """Close price = the bar's open as HHMM, so every value names its own bar."""
    return float((minute_of_day // 60) * 100 + minute_of_day % 60)


def __test_helper_syminfo(period):
    from datetime import time
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    return SymInfo(
        prefix="EXCH", description="Shortened session", ticker="SSBP",
        currency="USD", period=period, type="stock",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=1,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(8, 30), end=time(13, 0))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(8, 30)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(13, 0)) for i in range(7)],
    )


def __test_helper_write_ltf(tmp_dir):
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "LTF20.ohlcv"
    with OHLCVWriter(path, "20") as w:
        for day in range(_N_DAYS):
            for minute in _LTF_OPEN_MINS:
                c = __test_helper_encode(minute)
                w.write(OHLCV(timestamp=_T0 + day * _DAY + minute * _MIN,
                              open=c, high=c, low=c, close=c, volume=1.0))
    __test_helper_syminfo("20").save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_helper_chart_bars():
    from pynecore.types.ohlcv import OHLCV
    out = []
    for day in range(_N_DAYS):
        for minute in _CHART_OPEN_MINS:
            out.append(OHLCV(timestamp=_T0 + day * _DAY + minute * _MIN,
                             open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0))
    return out


def __test_shortened_session_bar_reads_its_own_period__(runner, log):
    """A session-shortened chart bar reads the intrabar closing WITH the session.

    The 12:30 chart bar spans 30 minutes, not the nominal 60: its scheduled
    close is the session end at 13:00. The last 20-minute intrabar closing at or
    before that is the 12:50 one (also clamped to the session end); the 13:10
    record lies past the chart bar's close and must not appear.
    """
    import sys
    import tempfile
    from pathlib import Path

    sys.modules.pop(Path(__file__).stem, None)

    rows = {}
    with tempfile.TemporaryDirectory() as td:
        ltf = __test_helper_write_ltf(Path(td))
        r = runner(__test_helper_chart_bars(),
                   syminfo_path=None,
                   syminfo_override=dict(
                       period="60",
                       opening_hours=__test_helper_syminfo("60").opening_hours,
                       session_starts=__test_helper_syminfo("60").session_starts,
                       session_ends=__test_helper_syminfo("60").session_ends),
                   security_data={"20": ltf})
        for candle, pv in r.run_iter():
            offset = candle.timestamp - _T0
            rows[(offset // _DAY, (offset % _DAY) // _MIN)] = pv.get("p20")

    # ``lookahead_off`` on a finer timeframe returns the LAST intrabar of the
    # chart bar itself: three intrabars per full 60-minute bar, two in the
    # shortened one.
    for day in range(1, _N_DAYS):
        assert rows[(day, 570)] == __test_helper_encode(610), \
            f"day {day} 09:30 bar: p20={rows[(day, 570)]} != 1010"
        assert rows[(day, 690)] == __test_helper_encode(730), \
            f"day {day} 11:30 bar: p20={rows[(day, 690)]} != 1210"
        # The shortened bar: 12:50 closes with the session at 13:00, 13:10 does not.
        assert rows[(day, 750)] == __test_helper_encode(770), \
            f"day {day} 12:30 shortened bar: p20={rows[(day, 750)]} != 1250"
        assert rows[(day, 750)] != __test_helper_encode(790), \
            f"day {day} 12:30 bar leaked the 13:10 record"

    log.info("shortened session bar pairs on the scheduled close, not the nominal span")
