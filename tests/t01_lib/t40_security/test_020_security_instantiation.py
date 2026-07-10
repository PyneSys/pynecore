"""
@pyne
"""
from pynecore.lib import close, plot, request, script


@script.indicator(title="Security Instantiation Test", shorttitle="SECINST")
def main():
    # Pine instantiation semantics: each call of f() is a SEPARATE
    # request.security context bound to its own timeframe. Before the
    # per-call-site instantiation pass, both calls silently shared the first
    # call's binding ("1"), so ``b`` would repeat ``a``.
    def f(tf):
        return request.security("EXCH:LTFSYM", tf, close)
    a = f("1")
    b = f("15")
    plot(a, "a")
    plot(b, "b")


# --- Synthetic feed geometry -------------------------------------------------
# Chart: 10 bars at 300s (the default test syminfo period "5"), opens TS0+i*300.
# 1-minute feed: 50 bars, opens TS0+j*60, close = j+1 (covers every chart bar).
# 15-minute feed: 3 bars, opens TS0+k*900, close = 100+k.
_TS0 = 1735689600  # 2025-01-01T00:00:00 UTC
_CHART_STEP = 300

# tf="1" is a plain-LTF context: the bar's LAST intrabar close = 5*(i+1).
_EXPECTED_A = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
# tf="15" is an HTF context (lookahead_off): a 15-minute bar becomes visible
# on the chart bar that CLOSES at the same instant the period closes (chart
# bar 2 spans [600, 900) and the first 15m period closes at 900) — bars 0..1
# na, 2..4 -> 100, 5..7 -> 101, 8..9 -> 102.
_EXPECTED_B = [None, None, 100.0, 100.0, 100.0, 101.0, 101.0, 101.0, 102.0, 102.0]


def _write_feed(path, step, closes, first_ts=_TS0):
    from pynecore.core.ohlcv_file import OHLCVWriter
    from pynecore.types.ohlcv import OHLCV

    with OHLCVWriter(path) as w:
        for j, c in enumerate(closes):
            w.write(OHLCV(
                timestamp=first_ts + j * step,
                open=float(c), high=float(c), low=float(c), close=float(c),
                volume=1.0,
            ))


def _write_syminfo(path, period):
    from datetime import time
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession

    SymInfo(
        prefix="EXCH", description="Instantiation Test Symbol", ticker="LTFSYM",
        currency="USD", period=period, type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base", taker_fee=0.1, maker_fee=0.1,
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))


def _chart_bars():
    from pynecore.types.ohlcv import OHLCV
    return [
        OHLCV(timestamp=_TS0 + i * _CHART_STEP,
              open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0)
        for i in range(10)
    ]


def _assert_column(rows, key, expected, log):
    import math
    from pynecore.types.na import NA

    assert len(rows) == len(expected), f"bar count {len(rows)} != {len(expected)}"
    for i, row in enumerate(rows):
        value = row[key]
        want = expected[i]
        if want is None:
            assert isinstance(value, NA), f"{key} bar {i}: expected na, got {value!r}"
        else:
            assert not isinstance(value, NA), f"{key} bar {i}: expected {want}, got na"
            assert math.isclose(float(value), want, abs_tol=1e-9), (
                f"{key} bar {i}: {value} != {want}"
            )
    log.info("column %s matches its own timeframe's series", key)


def __test_per_call_site_contexts__(runner, log):
    """Two calls of one security-bearing function bind to their OWN
    timeframes: ``a`` reads the 1-minute series (plain-LTF last-intrabar
    merge), ``b`` the 15-minute HTF series."""
    import tempfile
    from pathlib import Path

    rows = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p1 = td / "EXCH_LTFSYM_1.ohlcv"
        _write_feed(p1, 60, [j + 1 for j in range(50)])
        _write_syminfo(p1, "1")
        p15 = td / "EXCH_LTFSYM_15.ohlcv"
        _write_feed(p15, 900, [100 + k for k in range(3)])
        _write_syminfo(p15, "15")

        r = runner(_chart_bars(), security_data={
            "EXCH:LTFSYM:1": str(p1),
            "EXCH:LTFSYM:15": str(p15),
        })
        for _candle, pv in r.run_iter():
            rows.append(dict(pv))  # run_iter reuses the plot dict per bar

    _assert_column(rows, "a", _EXPECTED_A, log)
    _assert_column(rows, "b", _EXPECTED_B, log)
