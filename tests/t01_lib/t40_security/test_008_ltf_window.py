"""
@pyne
"""
from pynecore.lib import array, close, plot, request, script


@script.indicator(title="LTF Window Test", shorttitle="LWT")
def main():
    # ``n`` is the per-chart-bar intrabar count, ``sum`` the sum of the intrabar
    # closes — together they fully characterise the LTF array for that bar.
    vals = request.security_lower_tf("EXCH:LTFSYM", "1", close)
    plot(array.size(vals), "n")
    plot(array.sum(vals), "sum")


# --- Shared geometry ---------------------------------------------------------
# Chart bars run at 300s (the default test syminfo period "5"); each chart bar
# returns the intrabars of its OWN period ``[T, T+tf)``. The LTF feed is 60s.
_TS0 = 1735689600  # 2025-01-01T00:00:00 UTC, aligned to both the 300s and 60s grids
_CHART_STEP = 300


def _write_ltf(tmp_dir, bars):
    """Write a 1-minute LTF feed from ``(timestamp_s, close)`` pairs (+ a 24/7
    UTC ``.toml`` sidecar the subprocess loads on startup); return the path.
    """
    from datetime import time
    from pynecore.core.ohlcv_file import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "EXCH_LTFSYM_1.ohlcv"
    with OHLCVWriter(path) as w:
        for ts, c in bars:
            c = float(c)
            w.write(OHLCV(timestamp=ts, open=c, high=c, low=c, close=c, volume=1.0))

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


def _chart_bars(n):
    """``n`` chart bars at 300s; OHLCV body is irrelevant to the LTF result."""
    from pynecore.types.ohlcv import OHLCV
    return [
        OHLCV(timestamp=_TS0 + i * _CHART_STEP,
              open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0)
        for i in range(n)
    ]


def _run(runner, chart_n, ltf_bars):
    """One end-to-end run; returns the per-bar ``(n, sum)`` plot rows.

    Drops the script's stem-key import cache first so each geometry re-imports a
    clean module (the ``runner`` fixture only deletes pytest's dotted key once,
    and a single module may host only one runner-using test).
    """
    import sys
    import tempfile
    from pathlib import Path

    sys.modules.pop(Path(__file__).stem, None)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        ltf_path = _write_ltf(Path(td), ltf_bars)
        r = runner(_chart_bars(chart_n), security_data={"EXCH:LTFSYM:1": ltf_path})
        for _candle, pv in r.run_iter():
            rows.append((pv["n"], pv["sum"]))
    return rows


def _assert(rows, expected_size, expected_sum):
    import math
    from pynecore.types.na import NA

    assert len(rows) == len(expected_size), f"bar count {len(rows)} != {len(expected_size)}"
    for i, (n, s) in enumerate(rows):
        assert int(n) == expected_size[i], f"bar {i}: size {n} != {expected_size[i]}"
        exp = expected_sum[i]
        if exp is None:
            assert isinstance(s, NA), f"bar {i}: expected na, got {s!r}"
        else:
            assert not isinstance(s, NA), f"bar {i}: expected {exp}, got na"
            assert math.isclose(float(s), exp, abs_tol=1e-9), f"bar {i}: sum {s} != {exp}"


def __test_ltf_window_edges__(runner, log):
    """The historical LTF window ``[T, T+tf)`` across four edge geometries. A
    single test hosts all four because the ``runner`` fixture may be used by only
    one test per module (it deletes the module's import key once); ``_run``
    re-imports a clean script between geometries.
    """
    # --- Partial first period: the feed begins partway through a chart bar, so
    # that boundary bar gets a PARTIAL array (not skipped). The old
    # open-as-target predicate wrongly skipped it (chart open < feed first open).
    # Feed opens at TS0+720 (inside chart bar 2 = [600, 900)), step 60, closes 1..8.
    rows = _run(runner, 5, [(_TS0 + 720 + j * 60, j + 1) for j in range(8)])
    # bar 2 [600,900): 720,780,840 -> 1,2,3 ; bar 3 [900,1200): 900..1140 -> 4..8 ;
    # bar 4 [1200,1500): feed already ended -> empty (signalled, not skipped).
    _assert(rows, [0, 0, 3, 5, 0], [None, None, 6, 30, None])
    log.info("LTF partial-first-period window correct")

    # --- Boundary exclusivity: an LTF bar opening exactly at a chart bar's open
    # ``T`` belongs to the bar STARTING at ``T`` (``[T, T+tf)``), never the
    # previous bar ending at ``T``. Distinguishes ``[T, T+tf)`` from ``(T-tf, T]``.
    # LTF bars exactly on chart boundaries: TS0+300 (bar 1 open), TS0+600 (bar 2 open).
    rows = _run(runner, 3, [(_TS0 + 300, 10), (_TS0 + 600, 20)])
    # bar 0 [0,300): skipped ; bar 1 [300,600): open 300 only (600 excluded) ;
    # bar 2 [600,900): open 600.
    _assert(rows, [0, 1, 1], [None, 10, 20])
    log.info("LTF boundary-exclusive window correct")

    # --- Trailing partial period at EOF: the last chart bar collects every
    # remaining intrabar even when its period extends past the feed's end (no
    # next bar to consume them). The old code dropped this trailing period.
    rows = _run(runner, 3,
                [(_TS0 + j * 60, j + 1) for j in range(10)]          # bars 0,1: 1..10
                + [(_TS0 + 600 + k * 60, 11 + k) for k in range(3)])  # bar 2: 600,660,720 -> 11,12,13
    _assert(rows, [5, 5, 3], [15, 40, 36])
    log.info("LTF trailing-partial-at-eof window correct")

    # --- Source gap is forward-filled: a stored ``.ohlcv`` cannot represent
    # missing intrabars (the writer fills gaps with the prior close), so a chart
    # bar spanning a source gap still gets a full, correctly-windowed array of
    # flat intrabars rather than an empty one. The window itself stays right.
    rows = _run(runner, 4,
                [(_TS0 + j * 60, j + 1) for j in range(10)]           # bars 0,1: 1..10
                + [(_TS0 + 900 + k * 60, 11 + k) for k in range(5)])  # bar 3: 900.. -> 11..15
    # bar 2 [600,900): 600,660,720,780,840 forward-filled with the prior close 10
    # -> 5 intrabars, sum 50.
    _assert(rows, [5, 5, 5, 5], [15, 40, 50, 65])
    log.info("LTF source-gap forward-filled and windowed correctly")
