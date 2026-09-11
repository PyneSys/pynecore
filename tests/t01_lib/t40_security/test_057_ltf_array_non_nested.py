"""
@pyne
"""
from pynecore.lib import array, na, open, plot, request, script


@script.indicator(title="LTF Array Non-Nested", shorttitle="LANN")
def main():
    # Two lower timeframes that do NOT nest into the 3-minute chart: 2 minutes
    # (finer, but off-grid) and 4 minutes (coarser than the chart bar). Each
    # intrabar's ``open`` price IS its open offset in minutes, so a plotted
    # array element identifies the intrabar it came from and the test can check
    # its scheduled close against the chart bar's own close.
    v2 = request.security_lower_tf("EXCH:LTF2", "2", open)
    v4 = request.security_lower_tf("EXCH:LTF4", "4", open)
    n2 = array.size(v2)
    n4 = array.size(v4)
    f2 = na
    l2 = na
    f4 = na
    if n2 > 0:
        f2 = array.first(v2)
        l2 = array.last(v2)
    if n4 > 0:
        f4 = array.first(v4)
    plot(n2, "n2")
    plot(f2, "f2")
    plot(l2, "l2")
    plot(n4, "n4")
    plot(f4, "f4")


_MIN = 60_000  # one minute in milliseconds
_T0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, on every minute grid
_CHART_TF_MIN = 3
_CHART_BARS = 8


def __test_helper_write_ltf(tmp_dir, ticker, span_min, opens_min):
    """Write an ``.ohlcv`` feed at ``span_min`` minutes whose every price is the
    bar's open offset in minutes, plus a 24/7 UTC ``.toml`` sidecar."""
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / f"EXCH_{ticker}_{span_min}.ohlcv"
    with OHLCVWriter(path, str(span_min)) as w:
        for m in opens_min:
            v = float(m)
            w.write(OHLCV(timestamp=_T0 + m * _MIN, open=v, high=v, low=v,
                          close=v, volume=1.0))
    SymInfo(
        prefix="EXCH", description="LTF non-nested", ticker=ticker,
        currency="USD", period=str(span_min), type="crypto",
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
    for i in range(_CHART_BARS):
        out.append(OHLCV(timestamp=_T0 + i * _CHART_TF_MIN * _MIN,
                         open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0))
    return out


def __test_ltf_array_non_nested_no_straddle__(runner, log):
    """No intrabar closing AFTER the chart bar's close ever enters its array.

    The 2- and 4-minute grids do not nest into the 3-minute chart, so on almost
    every chart bar an intrabar straddles its close. Such an intrabar belongs to
    the NEXT chart bar's round — its close is a price from after this bar ended.
    An intrabar pushed over that way is not lost: the chart bar its close falls
    into collects it, so the two feeds' arrays still tile their intrabars exactly
    once.
    """
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.types.na import NA

    sys.modules.pop(Path(__file__).stem, None)

    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        p2 = __test_helper_write_ltf(tmp, "LTF2", 2, range(0, 24, 2))
        p4 = __test_helper_write_ltf(tmp, "LTF4", 4, range(0, 24, 4))
        r = runner(__test_helper_chart_bars(),
                   syminfo_override={"period": str(_CHART_TF_MIN)},
                   security_data={"EXCH:LTF2:2": p2, "EXCH:LTF4:4": p4})
        # ``run_iter`` hands out the SAME plot dict every bar, so copy at once.
        for _candle, pv in r.run_iter():
            rows.append(dict(pv))

    def _num(v):
        return None if v is None or isinstance(v, NA) or v != v else float(v)

    # Per chart bar: (n2, f2, l2, n4, f4). The 2-minute arrays hold every
    # intrabar exactly once across the 8 chart bars (12 intrabars), the
    # 4-minute ones likewise (6 intrabars); chart bar 4 sees no 4-minute close
    # at all, which is an EMPTY array, never the straddling bar.
    expected = [
        (1, 0.0, 0.0, 0, None),
        (2, 2.0, 4.0, 1, 0.0),
        (1, 6.0, 6.0, 1, 4.0),
        (2, 8.0, 10.0, 1, 8.0),
        (1, 12.0, 12.0, 0, None),
        (2, 14.0, 16.0, 1, 12.0),
        (1, 18.0, 18.0, 1, 16.0),
        (2, 20.0, 22.0, 1, 20.0),
    ]
    assert len(rows) == len(expected), f"bar count {len(rows)} != {len(expected)}"
    for i, (n2, f2, l2, n4, f4) in enumerate(expected):
        got = (int(_num(rows[i]["n2"]) or 0), _num(rows[i]["f2"]), _num(rows[i]["l2"]),
               int(_num(rows[i]["n4"]) or 0), _num(rows[i]["f4"]))
        assert got == (n2, f2, l2, n4, f4), f"bar {i}: {got} != {(n2, f2, l2, n4, f4)}"

    # The structural invariant behind those numbers, asserted directly: the
    # chart bar closes at ``(i + 1) * 3`` minutes and an intrabar opening at
    # ``m`` closes at ``m + span``.
    for i, row in enumerate(rows):
        chart_close = (i + 1) * _CHART_TF_MIN
        for key, span in (("f2", 2), ("l2", 2), ("f4", 4)):
            m = _num(row[key])
            if m is None:
                continue
            assert m + span <= chart_close, \
                f"bar {i}: intrabar {m}+{span} closes after the chart bar's {chart_close}"

    total2 = sum(int(_num(r_["n2"]) or 0) for r_ in rows)
    total4 = sum(int(_num(r_["n4"]) or 0) for r_ in rows)
    assert total2 == 12, f"2-minute intrabars delivered {total2} times, not 12"
    assert total4 == 6, f"4-minute intrabars delivered {total4} times, not 6"
    log.info("non-nested LTF arrays tile their intrabars with no straddling close")
