"""
@pyne
"""
from pynecore.lib import array, close, plot, request, script


@script.indicator(title="LTF Prefix Skip Test", shorttitle="LPS")
def main():
    # Same-symbol-style lower-timeframe read against a feed that begins partway
    # through the chart. ``n`` is the intrabar count, ``sum`` the sum of the
    # intrabar closes — both fully characterise the per-bar LTF array.
    vals = request.security_lower_tf("EXCH:LTFSYM", "1", close)
    plot(array.size(vals), "n")
    plot(array.sum(vals), "sum")


# --- Synthetic feed geometry -------------------------------------------------
# Chart: 10 bars at 300s (the default test syminfo period "5"), opens TS0+i*300.
# LTF:   25 bars at 60s, opens TS0+1500+j*60 (j=0..24), close = j+1.
# The LTF feed's first open lands exactly on chart bar 5, so chart bars 0..4
# open strictly before it (prefix → skip → empty array) and bar 5 is the first
# bar that can contain an intrabar.
_TS0 = 1735689600  # 2025-01-01T00:00:00 UTC, aligned to both the 300s and 60s grids
_CHART_STEP = 300
_LTF_STEP = 60
_LTF_FIRST = _TS0 + 1500  # == chart bar 5 open

# Analytic expectation, identical for the optimised (prefix-skip) and the
# unoptimised (every-bar-signal) path — see __test_ltf_prefix_skip__ below.
_EXPECTED_SIZE = [0, 0, 0, 0, 0, 1, 5, 5, 5, 5]
_EXPECTED_SUM = [None, None, None, None, None, 1, 20, 45, 70, 95]


def _build_ltf_file(tmp_path):
    """Write the 25-bar 1-minute LTF feed (+ a 24/7 UTC ``.toml`` sidecar the
    subprocess loads on startup) and return the ``.ohlcv`` path.
    """
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
    """10 chart bars at 300s; OHLCV body is irrelevant to the LTF result."""
    from pynecore.types.ohlcv import OHLCV
    return [
        OHLCV(timestamp=_TS0 + i * _CHART_STEP,
              open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0)
        for i in range(10)
    ]


def _assert_matches_expected(rows, log):
    """Assert the collected (n, sum) plot rows match the analytic expectation."""
    import math
    from pynecore.types.na import NA

    assert len(rows) == len(_EXPECTED_SIZE), (
        f"bar count {len(rows)} != {len(_EXPECTED_SIZE)}"
    )
    for i, (n, s) in enumerate(rows):
        assert int(n) == _EXPECTED_SIZE[i], (
            f"bar {i}: size {n} != {_EXPECTED_SIZE[i]}"
        )
        expected_sum = _EXPECTED_SUM[i]
        if expected_sum is None:
            assert isinstance(s, NA), f"bar {i}: expected na, got {s!r}"
        else:
            assert not isinstance(s, NA), f"bar {i}: expected {expected_sum}, got na"
            assert math.isclose(float(s), expected_sum, abs_tol=1e-9), (
                f"bar {i}: sum {s} != {expected_sum}"
            )
    log.info("LTF prefix-skip output matches analytic expectation")


def __test_load_ltf_first_ms__(tmp_path):
    """``load_ltf_first_ms`` records the feed's first bar open in ms for LTF
    contexts, and is a no-op for non-LTF contexts (optimization stays off).
    """
    from zoneinfo import ZoneInfo
    from pynecore.core.security import SecurityState, load_ltf_first_ms

    data_path = _build_ltf_file(tmp_path)

    ltf_state = SecurityState(
        sec_id="s", timeframe="1", gaps_on=False, same_timeframe=False,
        resampler=None, tz=ZoneInfo("UTC"), is_ltf=True,
    )
    load_ltf_first_ms(ltf_state, data_path)
    assert ltf_state.ltf_first_ms == _LTF_FIRST * 1000

    # Non-LTF context: must remain disabled (None) so HTF/same-TF paths are
    # never affected by the prefix-skip.
    htf_state = SecurityState(
        sec_id="s", timeframe="1D", gaps_on=False, same_timeframe=False,
        resampler=None, tz=ZoneInfo("UTC"), is_ltf=False,
    )
    load_ltf_first_ms(htf_state, data_path)
    assert htf_state.ltf_first_ms is None

    # Empty feed (no first bar): the optimization cleanly disables itself
    # (ltf_first_ms stays None) instead of skipping every chart bar.
    from pynecore.core.ohlcv_file import OHLCVWriter
    empty_path = str(tmp_path / "empty.ohlcv")
    with OHLCVWriter(empty_path):
        pass
    empty_state = SecurityState(
        sec_id="s", timeframe="1", gaps_on=False, same_timeframe=False,
        resampler=None, tz=ZoneInfo("UTC"), is_ltf=True,
    )
    load_ltf_first_ms(empty_state, empty_path)
    assert empty_state.ltf_first_ms is None


def _run_collect(runner):
    """One end-to-end run; returns the per-bar ``(n, sum)`` plot rows."""
    import tempfile
    from pathlib import Path

    rows = []
    with tempfile.TemporaryDirectory() as td:
        ltf_path = _build_ltf_file(Path(td))
        r = runner(_chart_bars(), security_data={"EXCH:LTFSYM:1": ltf_path})
        for _candle, pv in r.run_iter():
            rows.append((pv["n"], pv["sum"]))
    return rows


def __test_ltf_prefix_skip__(runner, monkeypatch, log):
    """End-to-end behaviour-equivalence of the LTF prefix-skip.

    Run 1 (optimised, default): chart bars before the LTF feed return empty
    arrays without a cross-process handshake; the boundary bar and the bars
    after it carry the correct intrabar counts and close sums.

    Run 2 (unoptimised): with ``load_ltf_first_ms`` neutralised the chart
    signals every bar (the original behaviour). Asserting both runs against the
    same analytic table proves the prefix-skip changes no output.

    Both runs share one ``runner`` fixture (its ``del sys.modules`` runs once);
    the stem module is dropped between runs so ``import_script`` re-imports a
    clean module for the second run.
    """
    import sys
    from pathlib import Path
    import pynecore.core.security as sec_mod

    # Run 1 — optimisation active.
    optimised = _run_collect(runner)
    _assert_matches_expected(optimised, log)

    # Force a fresh re-import for the second run (the runner fixture only drops
    # pytest's dotted module key; import_script caches under the file stem).
    sys.modules.pop(Path(__file__).stem, None)

    # Run 2 — optimisation neutralised (every chart bar signals, as before).
    # Both runs matching the same analytic table is the equivalence proof.
    monkeypatch.setattr(sec_mod, "load_ltf_first_ms", lambda state, path: None)
    unoptimised = _run_collect(runner)
    _assert_matches_expected(unoptimised, log)
