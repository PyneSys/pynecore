"""
@pyne
"""
from pynecore.lib import open, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="Dependent Non-Nested TF", shorttitle="DNNT")
def main():
    # A 3-minute chart with three contexts whose grids do not nest into it:
    # 2 minutes (finer than the chart), 4 minutes (coarser), and a 3-minute
    # cross-symbol peer. The dependency chains are 4 -> 2 and 3 -> 4 -> 2.
    # Every bar's ``open`` price IS its own open offset in minutes, so each
    # value names the exact peer bar it came from.
    v2: Series[float] = request.security(syminfo.tickerid, "2", open)
    v4: Series[float] = request.security(syminfo.tickerid, "4", v2)
    v3: Series[float] = request.security("EXCH:PEER3", "3", v4)
    plot(v2, "v2")
    plot(v4, "v4")
    plot(v3, "v3")


_MIN = 60_000  # one minute in milliseconds
_T0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, on every minute grid
_CHART_TF_MIN = 3
_CHART_BARS = 8
_FEED_MINUTES = 24


def __test_helper_write_feed(tmp_dir, name, ticker, span_min):
    """Write a ``span_min``-minute feed whose every price is the bar's open
    offset in minutes, plus a 24/7 UTC ``.toml`` sidecar."""
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / f"{name}.ohlcv"
    with OHLCVWriter(path, str(span_min)) as w:
        for m in range(0, _FEED_MINUTES, span_min):
            v = float(m)
            w.write(OHLCV(timestamp=_T0 + m * _MIN, open=v, high=v, low=v,
                          close=v, volume=1.0))
    SymInfo(
        prefix="EXCH", description="Dependent non-nested", ticker=ticker,
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


def __test_dependent_non_nested_tf__(runner, log):
    """Dependent chains across non-nesting grids: no deadlock, no post-close data.

    Each consumer sees the peer's last bar whose scheduled close reaches its
    own close, so the 4-minute bar opening at ``m`` reads the 2-minute bar
    closing at ``m + 4`` (open ``m + 2``) — never the one still open at its
    close. The 3-minute peer reads the 4-minute one under the same rule, which
    makes the whole 3 -> 4 -> 2 chain lag by exactly one peer close instead of
    borrowing a future price. Nothing waits on a bar the chart does not confirm
    in the same round, so the run completes.
    """
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.types.na import NA

    sys.modules.pop(Path(__file__).stem, None)

    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        base = __test_helper_write_feed(tmp, "BASE", "BASE", 1)
        peer3 = __test_helper_write_feed(tmp, "PEER3", "PEER3", 3)
        r = runner(__test_helper_chart_bars(),
                   syminfo_override={"period": str(_CHART_TF_MIN)},
                   security_data={"2": base, "4": base, "EXCH:PEER3:3": peer3})
        # ``run_iter`` reuses the plot dict, so copy each bar's values at once.
        for _candle, pv in r.run_iter():
            rows.append((pv.get("v2"), pv.get("v4"), pv.get("v3")))

    def _num(v):
        return None if v is None or isinstance(v, NA) or v != v else float(v)

    # v2: the chart bar's last 2-minute intrabar whose close reaches the chart
    #     bar's close (chart bar i closes at ``(i + 1) * 3`` minutes).
    # v4: the last 4-minute bar confirmed by that chart close, carrying the
    #     2-minute open two minutes past its own — bar 4 repeats bar 3's value
    #     because no 4-minute bar closes inside minutes 12..15 (``gaps_off``).
    # v3: the 3-minute peer, one step further down the same chain.
    expected = [
        (0.0, None, None),
        (4.0, 2.0, 2.0),
        (6.0, 6.0, 6.0),
        (10.0, 10.0, 10.0),
        (12.0, 10.0, 10.0),
        (16.0, 14.0, 14.0),
        (18.0, 18.0, 18.0),
        (22.0, 22.0, 22.0),
    ]
    assert len(rows) == len(expected), f"bar count {len(rows)} != {len(expected)}"
    for i, exp in enumerate(expected):
        got = tuple(_num(v) for v in rows[i])
        assert got == exp, f"bar {i}: {got} != {exp}"

    # The structural invariant: no value is a price from after the consumer
    # bar's own close. The 2-minute intrabar behind ``v4`` closes at ``v4 + 2``,
    # which must not pass the 4-minute bar's close, and the same for ``v2``
    # against its chart bar.
    for i, (r2, r4, _r3) in enumerate(rows):
        chart_close = (i + 1) * _CHART_TF_MIN
        m2 = _num(r2)
        if m2 is not None:
            assert m2 + 2 <= chart_close, \
                f"bar {i}: 2-minute intrabar {m2} closes after the chart bar"
        m4 = _num(r4)
        if m4 is not None:
            # The 4-minute bar carrying ``m4`` opens at ``m4 - 2`` and closes
            # at ``m4 + 2``; the 2-minute bar it read closes exactly there.
            assert (m4 + 2) % 4 == 0, f"bar {i}: v4={m4} is off the 4-minute grid"
    log.info("non-nested dependent chains resolve with no post-close data")
