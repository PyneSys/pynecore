"""
@pyne
"""
from pynecore.lib import array, open, high, low, close, plot, request, script


@script.indicator(title="LTF OHLCV Tuple Test", shorttitle="LOTT")
def main():
    # A tuple expression of only raw price series takes the plain-OHLCV fast
    # path (the security child serves it without running this main()); the four
    # returned arrays must stay column-major — array ``i`` holds the per-intrabar
    # values of the ``i``-th tuple element. Distinct O/H/L/C magnitudes make a
    # transposition error (e.g. all columns equal) impossible to miss.
    o, h, l, c = request.security_lower_tf("EXCH:LTFSYM", "1", [open, high, low, close])
    plot(array.size(o), "n")
    plot(array.sum(o), "so")
    plot(array.sum(h), "sh")
    plot(array.sum(l), "sl")
    plot(array.sum(c), "sc")


# --- Shared geometry ---------------------------------------------------------
# Chart bars run at 300s (default test syminfo period "5"); each returns the
# intrabars of its OWN period ``[T, T+tf)``. The LTF feed is 60s. Each intrabar
# carries DISTINCT O/H/L/C built from a base value ``b``: low=b, open=b+1000,
# close=b+2000, high=b+3000 (a valid low<=open<=close<=high ordering), so the
# per-column sums are b-sum offset by size*{0,1000,2000,3000}.
_TS0 = 1735689600  # 2025-01-01T00:00:00 UTC, aligned to both the 300s and 60s grids
_CHART_STEP = 300


def _write_ltf(tmp_dir, bars):
    """Write a 1-minute LTF feed from ``(timestamp_s, base)`` pairs, expanding
    each base into distinct O/H/L/C (+ a 24/7 UTC ``.toml`` sidecar the
    subprocess loads on startup); return the path."""
    from datetime import time
    from pynecore.core.ohlcv_file import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "EXCH_LTFSYM_1.ohlcv"
    with OHLCVWriter(path) as w:
        for ts, b in bars:
            b = float(b)
            w.write(OHLCV(timestamp=ts, open=b + 1000.0, high=b + 3000.0,
                          low=b, close=b + 2000.0, volume=1.0))

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
    """One end-to-end run; returns the per-bar ``(n, so, sh, sl, sc)`` plot rows.

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
            rows.append((pv["n"], pv["so"], pv["sh"], pv["sl"], pv["sc"]))
    return rows


def _assert(rows, expected_size, expected_base_sum):
    """Each chart bar's four column sums must be the base-sum offset by
    ``size * {open:1000, high:3000, low:0, close:2000}`` — confirming the arrays
    stay column-major. An empty window yields size 0 and na sums."""
    import math
    from pynecore.types.na import isna_num

    assert len(rows) == len(expected_size), f"bar count {len(rows)} != {len(expected_size)}"
    for i, (n, so, sh, sl, sc) in enumerate(rows):
        size = expected_size[i]
        assert int(n) == size, f"bar {i}: size {n} != {size}"
        base_sum = expected_base_sum[i]
        if base_sum is None:
            for label, s in (("open", so), ("high", sh), ("low", sl), ("close", sc)):
                assert isna_num(s), f"bar {i} {label}: expected na, got {s!r}"
            continue
        for label, s, offset in (
                ("low", sl, 0.0), ("open", so, 1000.0),
                ("close", sc, 2000.0), ("high", sh, 3000.0)):
            expected = base_sum + size * offset
            assert not isna_num(s), f"bar {i} {label}: expected {expected}, got na"
            assert math.isclose(float(s), expected, abs_tol=1e-9), \
                f"bar {i} {label}: sum {s} != {expected}"


def __test_ltf_ohlcv_tuple__(runner, log):
    """A tuple ``request.security_lower_tf(sym, tf, [open, high, low, close])``
    served by the plain-OHLCV fast path returns the four price columns,
    column-major, identical to the full-main() path."""
    # --- Trailing partial period at EOF: every chart bar has intrabars, so all
    # four columns are populated and the per-column offsets are checked.
    rows = _run(runner, 3,
                [(_TS0 + j * 60, j + 1) for j in range(10)]            # bars 0,1: base 1..10
                + [(_TS0 + 600 + k * 60, 11 + k) for k in range(3)])    # bar 2: base 11,12,13
    # bar 0 [0,300): base 1..5 -> sum 15 ; bar 1 [300,600): base 6..10 -> sum 40 ;
    # bar 2 [600,900): base 11,12,13 -> sum 36.
    _assert(rows, [5, 5, 3], [15, 40, 36])
    log.info("LTF tuple OHLCV columns correct (populated windows)")

    # --- Empty window: a chart bar with no intrabars yields four empty arrays
    # (na sums), proving the column split survives the no-data path too.
    # Feed opens at TS0+720 (inside chart bar 2 = [600, 900)); bars 0,1 and the
    # trailing bar 4 see no intrabars.
    rows = _run(runner, 5, [(_TS0 + 720 + j * 60, j + 1) for j in range(8)])
    # bar 2 [600,900): 720,780,840 -> base 1,2,3 sum 6 ;
    # bar 3 [900,1200): 900..1140 -> base 4..8 sum 30 ; bar 4: empty.
    _assert(rows, [0, 0, 3, 5, 0], [None, None, 6, 30, None])
    log.info("LTF tuple OHLCV empty-window na columns correct")
