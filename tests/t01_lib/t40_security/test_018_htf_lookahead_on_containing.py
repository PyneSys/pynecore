"""
@pyne
"""
from pynecore.lib import barmerge, close, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="HTF Lookahead On Containing", shorttitle="HLOC")
def main():
    # The canonical daily-pivot idiom: ``lookahead_on`` steps into the CONTAINING
    # daily bar, and the inner ``close[1]`` reads the prior (just-closed) day.
    prev_on: Series[float] = request.security(
        syminfo.tickerid, "D", close[1], lookahead=barmerge.lookahead_on)
    # Bare ``close`` under ``lookahead_on`` exposes the containing day's own final
    # close — TV's classical historical future-leak.
    cur_on: Series[float] = request.security(
        syminfo.tickerid, "D", close, lookahead=barmerge.lookahead_on)
    # ``lookahead_off`` sees only the last CLOSED daily bar, so ``close[1]`` there
    # is one further day back than the ``lookahead_on`` result.
    prev_off: Series[float] = request.security(
        syminfo.tickerid, "D", close[1], lookahead=barmerge.lookahead_off)
    plot(prev_on, "prev_on")
    plot(cur_on, "cur_on")
    plot(prev_off, "prev_off")


# Dense hourly 24/7 feed on the day grid. ``close = (day + 1) * 1000 + hour`` so
# every day's final (h23) close is distinct: day ``d`` closes at ``(d+1)*1000+23``.
_T0 = 1735689600  # 2025-01-01T00:00:00 UTC, aligned to the day grid
_N_DAYS = 6


def _close(day, hour):
    return float((day + 1) * 1000 + hour)


def _daily_close(day):
    return _close(day, 23)


def _bars():
    from pynecore.types.ohlcv import OHLCV
    out = []
    for day in range(_N_DAYS):
        for hour in range(24):
            c = _close(day, hour)
            out.append(OHLCV(timestamp=_T0 + (day * 24 + hour) * 3600,
                             open=c, high=c, low=c, close=c, volume=1.0))
    return out


def _write_feed(tmp_dir):
    from datetime import time
    from pynecore.core.ohlcv_file import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "FEED.ohlcv"
    with OHLCVWriter(path) as w:
        for bar in _bars():
            w.write(bar)
    SymInfo(
        prefix="EXCH", description="Lookahead containing", ticker="HLOC",
        currency="USD", period="60", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_htf_lookahead_on_steps_into_containing_period__(runner, log):
    """``lookahead_on`` HTF steps into the CONTAINING period in backtest mode.

    Regression for the wild ORB ``04da620e`` daily-pivot divergence. Historical
    ``lookahead_on`` on a same-symbol HTF used to fall back to ``lookahead_off``
    (last-closed) semantics, so the standard daily-pivot idiom
    ``request.security(sym, "D", close[1], lookahead_on)`` read the day-before-
    yesterday's close (D-2) instead of yesterday's (D-1). The fix steps into the
    containing daily bar: the inner ``close[1]`` then reads D-1, a bare ``close``
    reads the containing day's own final close (TV's future-leak), and
    ``lookahead_off`` stays one day further back (D-2).
    """
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.types.na import NA

    sys.modules.pop(Path(__file__).stem, None)

    rows = {}
    with tempfile.TemporaryDirectory() as td:
        feed = _write_feed(Path(td))
        r = runner(_bars(), syminfo_override={"period": "60"},
                   security_data={"D": feed})
        for candle, pv in r.run_iter():
            mins = (candle.timestamp - _T0) // 60
            day = mins // (60 * 24)
            hour = (mins // 60) % 24
            rows[(day, hour)] = (pv.get("prev_on"), pv.get("cur_on"), pv.get("prev_off"))

    # Assert at mid-day (h12): the confirmed value is stable there, past any
    # hour-23 period-boundary confirm and before the next day.
    for day in range(_N_DAYS):
        prev_on, cur_on, prev_off = rows[(day, 12)]

        # cur_on: the containing day's own final close, available all day (leak).
        assert cur_on == _daily_close(day), \
            f"day {day} h12: cur_on={cur_on} != {_daily_close(day)} (containing-day close)"

        # prev_on: yesterday's close via the inner close[1]; day 0 has no prior.
        if day == 0:
            assert isinstance(prev_on, NA), f"day 0 h12: prev_on={prev_on} should be na"
        else:
            assert prev_on == _daily_close(day - 1), \
                f"day {day} h12: prev_on={prev_on} != {_daily_close(day - 1)} (D-1 close)"

        # prev_off: last-closed daily context, so close[1] is D-2.
        if day < 2:
            assert isinstance(prev_off, NA), f"day {day} h12: prev_off={prev_off} should be na"
        else:
            assert prev_off == _daily_close(day - 2), \
                f"day {day} h12: prev_off={prev_off} != {_daily_close(day - 2)} (D-2 close)"

    # The whole point of the fix: lookahead_on's close[1] is exactly one day ahead
    # of lookahead_off's close[1].
    for day in range(2, _N_DAYS):
        assert rows[(day, 12)][0] == rows[(day, 12)][2] + 1000.0, \
            f"day {day}: prev_on should be one day (=+1000) ahead of prev_off"
