"""
@pyne
"""
from pynecore.lib import barmerge, close, input, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="Gaps From Local", shorttitle="GFL")
def main(gap=input.bool(True, "Gaps on")):
    # Pine takes a "simple" ``gaps``, so the mode may be chosen by an input and
    # held in a local. The module-level ``__security_contexts__`` dict cannot
    # name that local; the mode reaches the runtime on the context's first
    # ``__sec_signal__`` instead.
    gapout = barmerge.gaps_on if gap else barmerge.gaps_off
    k: Series[float] = request.security(syminfo.tickerid, "D", close, gaps=gapout)
    plot(k, "k")


# Dense hourly 24/7 feed on the day grid. ``close = (day + 1) * 1000 + hour`` so
# every day's final (h23) close is distinct.
# Every timestamp here is Unix MILLISECONDS.
_T0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, aligned to the day grid
_HOUR = 3_600_000
_N_DAYS = 5


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
            out.append(OHLCV(timestamp=_T0 + (day * 24 + hour) * _HOUR,
                             open=c, high=c, low=c, close=c, volume=1.0))
    return out


def _write_feed(tmp_dir):
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession

    path = tmp_dir / "FEED.ohlcv"
    with OHLCVWriter(path, "60") as w:
        for bar in _bars():
            w.write(bar)
    SymInfo(
        prefix="EXCH", description="Gaps from local", ticker="GFL",
        currency="USD", period="60", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def _run(runner, feed, gap):
    """``k`` per (day, hour) for one value of the ``gap`` input."""
    import sys
    from pathlib import Path

    sys.modules.pop(Path(__file__).stem, None)
    rows = {}
    r = runner(_bars(), syminfo_override={"period": "60"},
               security_data={"D": feed}, inputs={"gap": gap})
    for candle, pv in r.run_iter():
        mins = (candle.timestamp - _T0) // 60_000
        rows[(mins // (60 * 24), (mins // 60) % 24)] = pv.get("k")
    return rows


def __test_gaps_mode_held_in_a_local_resolves_on_the_first_bar__(runner, log):
    """A ``gaps`` chosen by an input and held in a local is honoured at runtime.

    Regression: the security transformer copied the ``gaps`` expression into
    the module-level ``__security_contexts__`` dict, where the local ``gapout``
    does not exist, so the script raised ``NameError: name 'gapout' is not
    defined`` when it loaded. The mode now travels with the context's first
    ``__sec_signal__``: ``gaps_on`` emits ``na`` between daily closes and
    ``gaps_off`` holds the last daily close through the day.
    """
    import tempfile
    from pathlib import Path
    from pynecore.types.na import isna_num

    with tempfile.TemporaryDirectory() as td:
        feed = _write_feed(Path(td))
        gaps_on = _run(runner, feed, True)
        gaps_off = _run(runner, feed, False)

    assert len(gaps_on) == len(gaps_off) == _N_DAYS * 24

    for day in range(_N_DAYS):
        # The day's close lands on the chart bar that confirms it (h23) in both modes.
        assert gaps_on[(day, 23)] == _daily_close(day), \
            f"day {day} h23: gaps_on={gaps_on[(day, 23)]} != {_daily_close(day)}"
        assert gaps_off[(day, 23)] == _daily_close(day), \
            f"day {day} h23: gaps_off={gaps_off[(day, 23)]} != {_daily_close(day)}"
        for hour in range(23):
            # gaps_on: na on every bar between two daily closes.
            assert isna_num(gaps_on[(day, hour)]), \
                f"day {day} h{hour}: gaps_on={gaps_on[(day, hour)]} should be na"
            # gaps_off: the previous day's close is held until the next one
            # lands; nothing has closed yet on day 0.
            if day == 0:
                assert isna_num(gaps_off[(day, hour)]), \
                    f"day 0 h{hour}: gaps_off={gaps_off[(day, hour)]} should be na"
            else:
                assert gaps_off[(day, hour)] == _daily_close(day - 1), \
                    f"day {day} h{hour}: gaps_off={gaps_off[(day, hour)]} " \
                    f"!= {_daily_close(day - 1)}"

    log.info("gaps mode held in a local: gaps_on gaps, gaps_off holds")
