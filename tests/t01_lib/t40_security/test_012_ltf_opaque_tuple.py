"""
@pyne
"""
from pynecore.lib import array, close, plot, request, script


@script.indicator(title="LTF Opaque Tuple Test", shorttitle="LOPT")
def main():
    # The tuple arity is invisible in the expression itself (an opaque function
    # call, not a tuple literal) — the transformer must take it from the LHS
    # unpack target and still wrap the read in ``__ltf_unzip__``. The +1/-1
    # offsets keep the two columns distinct, so a transposition or a scalar
    # leak (the pre-fix failure mode) cannot pass.
    def up_dn():
        u = close + 1.0
        d = close - 1.0
        return u, d

    up, dn = request.security_lower_tf("EXCH:LTFSYM", "1", up_dn())
    plot(array.size(up), "n")
    plot(array.sum(up), "su")
    plot(array.sum(dn), "sd")


# --- Shared geometry ---------------------------------------------------------
# Same layout as test_011: 300s chart bars, 60s LTF feed, each intrabar's close
# is ``base + 2000`` — so per chart bar ``sum(up) = base_sum + size * 2001`` and
# ``sum(dn) = base_sum + size * 1999``.
_TS0 = 1735689600  # 2025-01-01T00:00:00 UTC, aligned to both the 300s and 60s grids
_CHART_STEP = 300


def _write_ltf(tmp_dir, bars):
    """Write a 1-minute LTF feed from ``(timestamp_s, base)`` pairs (close is
    ``base + 2000``, matching test_011's O/H/L/C spread), plus the ``.toml``
    sidecar the subprocess loads on startup; return the path."""
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


def __test_ltf_opaque_tuple__(runner, log):
    """``a, b = request.security_lower_tf(sym, tf, fn())`` returns two
    column arrays sized/summed per the LHS-derived arity."""
    import math
    import tempfile
    from pathlib import Path
    from pynecore.types.na import isna_num

    with tempfile.TemporaryDirectory() as td:
        ltf_path = _write_ltf(
            Path(td),
            [(_TS0 + j * 60, j + 1) for j in range(10)]           # bars 0,1: base 1..10
            + [(_TS0 + 600 + k * 60, 11 + k) for k in range(3)])  # bar 2: base 11,12,13
        rows = []
        r = runner(_chart_bars(4), security_data={"EXCH:LTFSYM:1": ltf_path})
        for _candle, pv in r.run_iter():
            rows.append((pv["n"], pv["su"], pv["sd"]))

    # bar 0 [0,300): base 1..5 -> sum 15 ; bar 1 [300,600): base 6..10 -> sum 40 ;
    # bar 2 [600,900): base 11,12,13 -> sum 36 ; bar 3: no intrabars -> empty.
    expected = [(5, 15), (5, 40), (3, 36), (0, None)]
    assert len(rows) == len(expected)
    for i, ((n, su, sd), (size, base_sum)) in enumerate(zip(rows, expected)):
        assert int(n) == size, f"bar {i}: size {n} != {size}"
        if base_sum is None:
            assert isna_num(su), f"bar {i} up: expected na, got {su!r}"
            assert isna_num(sd), f"bar {i} dn: expected na, got {sd!r}"
            continue
        exp_up = base_sum + size * 2001.0
        exp_dn = base_sum + size * 1999.0
        assert not isna_num(su) and math.isclose(float(su), exp_up, abs_tol=1e-9), \
            f"bar {i} up: sum {su} != {exp_up}"
        assert not isna_num(sd) and math.isclose(float(sd), exp_dn, abs_tol=1e-9), \
            f"bar {i} dn: sum {sd} != {exp_dn}"
    log.info("LTF opaque-tuple columns correct (LHS-derived arity)")
