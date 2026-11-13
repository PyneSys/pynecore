"""
@pyne
"""
from pynecore.lib import close, high, plot, request, script, syminfo, ta


def pack():
    # ``pack() => [ta.sma(close, 3), ta.highest(high, 2)]``: the tuple expression
    # of the security call is a CALL of this function, not a literal, so the call
    # site cannot show the transformer the tuple's width.
    return ta.sma(close, 3), ta.highest(high, 2)


def get(tf):
    # ``get(tf) => request.security(syminfo.tickerid, tf, pack())``: the security
    # call is a function's return, so the unpack target below is out of its sight.
    return request.security(syminfo.tickerid, tf, pack())


@script.indicator("HTF pack", shorttitle="HP")
def main():
    a, b = get("W")
    plot(a, "a")
    plot(b, "b")


# Daily 24/7 bars from a Monday, ``close = high = day + 1``: week ``w`` covers days
# ``7w .. 7w+6`` and closes at ``7w + 7``. Every timestamp here is Unix MILLISECONDS.
_T0 = 1_736_121_600_000  # 2025-01-06T00:00:00 UTC, a Monday
_DAY = 86_400_000
_N_DAYS = 300


def _bars():
    from pynecore.types.ohlcv import OHLCV
    return [OHLCV(timestamp=_T0 + i * _DAY, open=float(i + 1), high=float(i + 1),
                  low=float(i + 1), close=float(i + 1), volume=1.0)
            for i in range(_N_DAYS)]


def _write_feed(tmp_dir):
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession

    path = tmp_dir / "FEED.ohlcv"
    with OHLCVWriter(path, "1D") as w:
        for bar in _bars():
            w.write(bar)
    SymInfo(
        prefix="EXCH", description="HTF pack", ticker="HP",
        currency="USD", period="1D", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def _closed_weeks(day):
    """How many weeks have closed by the end of chart day ``day``."""
    return (day + 1) // 7


def __test_tuple_read_through_a_function_runs_from_the_first_bar__(runner, log):
    """A read whose expression calls a tuple-returning function unpacks on every bar.

    The transformer sees a tuple's width in a tuple literal expression, or in the
    call itself unpacked into names. Here the expression is ``pack()`` and the
    security call is ``get()``'s return, so the read defaulted to one scalar ``na``
    and ``a, b = get("W")`` raised ``TypeError: na is not iterable`` before the
    first weekly bar closed. The width of the tuple ``pack()`` returns is now found
    from the function itself, and the read defaults to a tuple of ``na`` of that
    width: the script runs from the first bar with ``na`` until the first week
    closes, then ``a`` is the 3-week mean of weekly closes and ``b`` the highest of
    the last two weekly highs.
    """
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.types.na import isna_num

    sys.modules.pop(Path(__file__).stem, None)

    rows = {}
    with tempfile.TemporaryDirectory() as td:
        feed = _write_feed(Path(td))
        r = runner(_bars(), syminfo_override={"period": "1D"}, security_data={"W": feed})
        for candle, pv in r.run_iter():
            day = (candle.timestamp - _T0) // _DAY
            rows[day] = (pv.get("a"), pv.get("b"))
    assert len(rows) == _N_DAYS

    for day in range(_N_DAYS):
        a, b = rows[day]
        weeks = _closed_weeks(day)
        w = weeks - 1  # the latest closed week, -1 before the first close
        # a = sma(weekly close, 3) = mean(7w+7, 7w, 7w-7) = 7w from the third week on
        if weeks < 3:
            assert isna_num(a), f"day {day}: a={a} should be na ({weeks} weeks closed)"
        else:
            assert a == 7 * w, f"day {day}: a={a} != {7 * w}"
        # b = highest(weekly high, 2) = max(7w+7, 7w) = 7w+7 from the second week on
        if weeks < 2:
            assert isna_num(b), f"day {day}: b={b} should be na ({weeks} weeks closed)"
        else:
            assert b == 7 * w + 7, f"day {day}: b={b} != {7 * w + 7}"

    log.info("tuple read through a function: na until the first week closes, then values")
