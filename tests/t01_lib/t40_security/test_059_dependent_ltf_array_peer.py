"""
@pyne
"""
from pynecore.lib import array, close, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="Dependent LTF Array Peer", shorttitle="DLAP")
def main():
    # A ``request.security_lower_tf`` context used as a DEPENDENCY: the
    # 10-minute context reads the 1-minute array peer and gets the intrabars of
    # its OWN period, not the chart bar's window.
    v1 = request.security_lower_tf(syminfo.tickerid, "1", close)
    n: Series[float] = request.security(syminfo.tickerid, "10", array.size(v1))
    s: Series[float] = request.security(syminfo.tickerid, "10", array.sum(v1))
    plot(n, "n")
    plot(s, "s")


_MIN = 60_000  # one minute in milliseconds
_T0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, on every minute grid
_CHART_TF_MIN = 5
_CHART_BARS = 6
_FEED_MINUTES = 30


def __test_helper_write_feed(tmp_dir):
    """A 1-minute feed whose every price is the bar's open offset in minutes,
    with a 24/7 UTC ``.toml`` sidecar."""
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "BASE.ohlcv"
    with OHLCVWriter(path, "1") as w:
        for m in range(_FEED_MINUTES):
            v = float(m)
            w.write(OHLCV(timestamp=_T0 + m * _MIN, open=v, high=v, low=v,
                          close=v, volume=1.0))
    SymInfo(
        prefix="EXCH", description="Dependent LTF peer", ticker="BASE",
        currency="USD", period="1", type="crypto",
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


def __test_dependent_ltf_array_peer__(runner, log):
    """A lower-timeframe array context read as a peer yields the CONSUMER's period.

    The 10-minute consumer must see the ten 1-minute intrabars closing inside
    its own bar — not the five the 5-minute chart bar collects, and never one
    closing after its own close. The producer appends every intrabar to its
    ring and the consumer takes the slice ``(bar_open, as-of]``.
    """
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.types.na import NA

    sys.modules.pop(Path(__file__).stem, None)

    rows = []
    with tempfile.TemporaryDirectory() as td:
        feed = __test_helper_write_feed(Path(td))
        r = runner(__test_helper_chart_bars(),
                   syminfo_override={"period": str(_CHART_TF_MIN)},
                   security_data={"1": feed, "10": feed})
        # ``run_iter`` reuses the plot dict, so copy each bar's values at once.
        for _candle, pv in r.run_iter():
            rows.append((pv.get("n"), pv.get("s")))

    def _num(v):
        return None if v is None or isinstance(v, NA) or v != v else float(v)

    # The 10-minute bars open at 0, 10 and 20 minutes; the chart confirms each
    # on the chart bar whose close reaches the 10-minute close, and holds it
    # (``gaps_off``) until the next one lands.
    expected = [
        (None, None),
        (10.0, 45.0),    # minutes 0..9
        (10.0, 45.0),
        (10.0, 145.0),   # minutes 10..19
        (10.0, 145.0),
        (10.0, 245.0),   # minutes 20..29
    ]
    assert len(rows) == len(expected), f"bar count {len(rows)} != {len(expected)}"
    for i, exp in enumerate(expected):
        got = tuple(_num(v) for v in rows[i])
        assert got == exp, f"bar {i}: {got} != {exp}"
    log.info("a lower-timeframe array peer delivers the consumer's own period")
