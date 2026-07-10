"""
@pyne
"""
from pynecore.lib import barmerge, close, plot, request, script, ta


@script.indicator(title="Plain LTF Security Test", shorttitle="PLTF")
def main():
    # Scalar request.security with a timeframe FINER than the chart's.
    # TradingView merge rule: lookahead_off returns the expression's value on
    # the LAST intrabar of each chart bar, lookahead_on on the FIRST; before
    # the LTF series begins the result is na.
    last_c = request.security("EXCH:LTFSYM", "1", close)
    first_c = request.security("EXCH:LTFSYM", "1", close,
                               lookahead=barmerge.lookahead_on)
    sma_c = request.security("EXCH:LTFSYM", "1", ta.sma(close, 2))
    plot(last_c, "last")
    plot(first_c, "first")
    plot(sma_c, "sma")


# --- Synthetic feed geometry -------------------------------------------------
# Chart: 10 bars at 300s (the default test syminfo period "5"), opens TS0+i*300.
# LTF:   25 bars at 60s, opens TS0+1500+j*60 (j=0..24), close = j+1.
# The LTF feed's first open lands exactly on chart bar 5, so chart bars 0..4
# precede the series entirely (na), and bars 5..9 contain 5 intrabars each.
_TS0 = 1735689600  # 2025-01-01T00:00:00 UTC
_CHART_STEP = 300
_LTF_STEP = 60
_LTF_FIRST = _TS0 + 1500  # == chart bar 5 open

# lookahead_off -> the bar's LAST intrabar: closes 5, 10, 15, 20, 25.
_EXPECTED_LAST = [None, None, None, None, None, 5.0, 10.0, 15.0, 20.0, 25.0]
# lookahead_on -> the bar's FIRST intrabar: closes 1, 6, 11, 16, 21.
_EXPECTED_FIRST = [None, None, None, None, None, 1.0, 6.0, 11.0, 16.0, 21.0]
# The expression evaluates on the LTF bars themselves: sma(close, 2) on the
# bar's last intrabar averages the last TWO intrabar closes.
_EXPECTED_SMA = [None, None, None, None, None, 4.5, 9.5, 14.5, 19.5, 24.5]


def _build_ltf_file(tmp_path):
    from datetime import time
    from pynecore.core.ohlcv_file import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_path / "EXCH_LTFSYM_1.ohlcv"
    with OHLCVWriter(path) as w:
        for j in range(25):
            c = float(j + 1)
            w.write(OHLCV(
                timestamp=_LTF_FIRST + j * _LTF_STEP,
                open=c, high=c, low=c, close=c, volume=1.0,
            ))

    SymInfo(
        prefix="EXCH", description="LTF Test Symbol", ticker="LTFSYM",
        currency="USD", period="1", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base", taker_fee=0.1, maker_fee=0.1,
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))

    return str(path)


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
    log.info("column %s matches the TradingView merge rule", key)


def __test_plain_ltf_merge__(runner, log):
    """End-to-end plain-LTF semantics: last-intrabar (OFF), first-intrabar
    (ON), na prefix before the feed begins, and expression evaluation on the
    LTF series itself."""
    import tempfile
    from pathlib import Path

    rows = []
    with tempfile.TemporaryDirectory() as td:
        ltf_path = _build_ltf_file(Path(td))
        r = runner(_chart_bars(), security_data={"EXCH:LTFSYM:1": ltf_path})
        for _candle, pv in r.run_iter():
            rows.append(dict(pv))  # run_iter reuses the plot dict per bar

    _assert_column(rows, "last", _EXPECTED_LAST, log)
    _assert_column(rows, "first", _EXPECTED_FIRST, log)
    _assert_column(rows, "sma", _EXPECTED_SMA, log)
