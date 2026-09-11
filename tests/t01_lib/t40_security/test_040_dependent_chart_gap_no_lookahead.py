"""
@pyne
"""
from pynecore.lib import close, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="Chart Gap No Lookahead", shorttitle="CGNL")
def main():
    # A daily peer on a 24/7 hourly chart that has a long hole in its data. The
    # as-of rule runs on the SCHEDULE, never on the chart's next existing
    # record, so the last chart bar before the hole must not receive the daily
    # bar that only closes hours later.
    d: Series[float] = request.security(syminfo.tickerid, "D", close)
    dep: Series[float] = request.security(syminfo.tickerid, "D", d + 0.5)
    plot(d, "d")
    plot(dep, "dep")


# Every timestamp here is Unix MILLISECONDS.
_T0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, aligned to the day and hour grids
_HOUR = 3_600_000
_DAY = 86_400_000
_N_DAYS = 5
# Day 2 is truncated: only its hours 0 and 1 are on the chart, so the chart jumps
# from 01:00 straight to the next day's 00:00.
_GAP_DAY = 2


def __test_helper_daily_close(day):
    return 500.0 + day


def __test_helper_write_daily(tmp_dir):
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "DAILY.ohlcv"
    with OHLCVWriter(path, "1D") as w:
        for day in range(_N_DAYS):
            c = __test_helper_daily_close(day)
            w.write(OHLCV(timestamp=_T0 + day * _DAY, open=c, high=c, low=c,
                          close=c, volume=1.0))
    SymInfo(
        prefix="EXCH", description="Gappy chart daily peer", ticker="CGNL",
        currency="USD", period="1D", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_helper_chart_bars():
    from pynecore.types.ohlcv import OHLCV
    out = []
    for day in range(_N_DAYS):
        hours = (0, 1) if day == _GAP_DAY else range(24)
        for hour in hours:
            out.append(OHLCV(timestamp=_T0 + day * _DAY + hour * _HOUR,
                             open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0))
    return out


def __test_chart_data_gap_does_not_advance_the_daily_peer__(runner, log):
    """A hole in the chart feed must not pull a daily peer's bar forward.

    The chart's last bar before the hole closes at 02:00, while the day's daily
    bar only closes at midnight. Riding the next EXISTING chart record (the next
    day's 00:00 bar) would hand that whole daily bar to the 01:00 bar — future
    data, and indistinguishable from a holiday. The schedule is the bound, so
    the daily bar first appears on the chart bar whose own close reaches it.
    The chart's read and the dependent child's read must agree.
    """
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.types.na import NA

    sys.modules.pop(Path(__file__).stem, None)

    rows = {}
    with tempfile.TemporaryDirectory() as td:
        feed = __test_helper_write_daily(Path(td))
        r = runner(__test_helper_chart_bars(), syminfo_override={"period": "60"},
                   security_data={"D": feed})
        for candle, pv in r.run_iter():
            offset = candle.timestamp - _T0
            rows[(offset // _DAY, (offset % _DAY) // _HOUR)] = (pv.get("d"), pv.get("dep"))

    # A normal day: the previous day's bar is reported until the day's own daily
    # bar closes, which happens exactly with the 23:00 chart bar (both close at
    # the next midnight).
    for hour in (0, 5, 22):
        value = rows[(1, hour)][0]
        assert value == __test_helper_daily_close(0), \
            f"day 1 hour {hour}: d={value} != day-0 close"
    assert rows[(1, 23)][0] == __test_helper_daily_close(1), \
        f"day 1 hour 23: d={rows[(1, 23)][0]} != day-1 close (same closing instant)"

    # The bar right before the hole: it closes at 02:00, so it still reports the
    # PREVIOUS day's bar — never day 2's, which closes at midnight.
    before_gap = rows[(_GAP_DAY, 1)][0]
    assert before_gap == __test_helper_daily_close(_GAP_DAY - 1), \
        f"bar before the chart gap: d={before_gap} != day-{_GAP_DAY - 1} close (lookahead!)"

    # The first bar after the hole closes at 01:00 the next day, which is past
    # day 2's midnight close, so it reports it.
    after_gap = rows[(_GAP_DAY + 1, 0)][0]
    assert after_gap == __test_helper_daily_close(_GAP_DAY), \
        f"first bar after the chart gap: d={after_gap} != day-{_GAP_DAY} close"

    # The dependent context pairs identically — one rule for the chart and the child.
    for key, (direct, dependent) in rows.items():
        if direct is None or isinstance(direct, NA):
            continue
        assert dependent == direct + 0.5, \
            f"{key}: dependent={dependent} != direct+0.5={direct + 0.5}"

    log.info("a chart data gap never advances the daily peer")
