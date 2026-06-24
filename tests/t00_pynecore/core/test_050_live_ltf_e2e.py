"""
@pyne

End-to-end live ``request.security_lower_tf`` window test through a real spawned
security child driven by the built-in ``replay`` provider. See the test function
docstring for what it proves and the scope boundary.
"""
from pynecore.lib import array, close, plot, request, script


@script.indicator(title="Live LTF E2E", shorttitle="LLE")
def main():
    # ``n`` = intrabar count, ``sum`` = sum of intrabar closes: together they
    # fully characterise the LTF array for that chart bar.
    vals = request.security_lower_tf("EXCH:LTFSYM", "1", close)
    plot(array.size(vals), "n")
    plot(array.sum(vals), "sum")


# --- Geometry ----------------------------------------------------------------
# Chart bars run at 300s (the default test syminfo period "5"); the LTF feed is
# 60s, so every chart bar's period [T, T+300) holds exactly 5 LTF intrabars.
_TS0 = 1735689600  # 2025-01-01T00:00:00 UTC, aligned to both the 300s and 60s grids
_CHART_STEP = 300
_LTF_STEP = 60
_N_HIST = 3        # historical (warmup) chart bars
_N_LIVE = 2        # live chart bars after LIVE_TRANSITION
_LTF_PER_BAR = _CHART_STEP // _LTF_STEP  # 5


def _chart_ohlcv(ts, is_closed=True):
    from pynecore.types.ohlcv import OHLCV
    return OHLCV(timestamp=ts, open=100.0, high=100.0, low=100.0, close=100.0,
                 volume=1.0, is_closed=is_closed)


def _ltf_syminfo():
    """A 24/7 UTC SymInfo for the LTF security symbol (period "1")."""
    from datetime import time
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    return SymInfo(
        prefix="EXCH", description="LTF Test Symbol", ticker="LTFSYM",
        currency="USD", basecurrency="LTFSYM", period="1", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base", taker_fee=0.0, maker_fee=0.0,
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    )


def __test_live_ltf_e2e_warmup_and_closed_windows__(script_path, module_key, syminfo, log):
    """Drive a real spawned LTF security child via the ``replay`` provider and
    assert the per-chart-bar ``[T, T+300)`` window for warmup and live bars.

    What this proves (the integration boundary):
      * warmup windows are correct — the warmup-replay path flows through the
        same LTF-window code as live (Option 2), end to end across the process
        boundary;
      * the confirmed closed-period array is correct for every chart bar,
        including a live boundary where the eager replay feed has already queued
        the next period's intrabars and the child must defer them to the
        following signal;
      * the full wiring works: parent ``__sec_signal__`` -> SHM signal ->
        spawned child ``load_plugin('replay')`` -> ``LiveBarStreamer`` ->
        ``LiveLtfCollector`` -> window publish -> parent reads the security array.

    Scope: the bar-by-bar developing-tail "live last element" growth is NOT
    asserted here — the replay feed is a deterministic CLOSED-window replay, not
    a lockstep forming-update replay (the streamer keeps only the latest forming
    snapshot). That micro-semantics is covered deterministically by
    ``test_049_live_ltf_collector`` (the same ``LiveLtfCollector`` the child runs).
    """
    import sys
    import json
    import tempfile
    import itertools
    from pathlib import Path
    from datetime import datetime, UTC

    from pynecore import lib
    from pynecore.core.script_runner import ScriptRunner, LIVE_TRANSITION
    from pynecore.core.plugin.live_provider import PluginSymbol
    from pynecore.providers.replay import ReplayConfig

    # Each test re-imports a clean script module (the spawned child also imports
    # this file by path).
    sys.modules.pop(Path(__file__).stem, None)
    sys.modules.pop(module_key, None)

    n_warmup_ltf = _N_HIST * _LTF_PER_BAR   # 15
    n_live_ltf = _N_LIVE * _LTF_PER_BAR     # 10

    # LTF close at intrabar index k (0-based across the whole feed) is k+1, so
    # each chart bar's 5-intrabar sum is deterministic and distinct.
    warmup_rows = [
        [_TS0 + j * _LTF_STEP, float(j + 1), float(j + 1), float(j + 1), float(j + 1), 1.0]
        for j in range(n_warmup_ltf)
    ]
    live_rows = [
        [_TS0 + (n_warmup_ltf + j) * _LTF_STEP,
         float(n_warmup_ltf + j + 1), float(n_warmup_ltf + j + 1),
         float(n_warmup_ltf + j + 1), float(n_warmup_ltf + j + 1), 1.0, True]
        for j in range(n_live_ltf)
    ]
    fixture = {"warmup": warmup_rows, "live": live_rows}

    # Per chart bar: 5 intrabars, sum of (k+1) over the bar's 5 indices.
    expected_size = [_LTF_PER_BAR] * (_N_HIST + _N_LIVE)
    expected_sum = []
    for bar in range(_N_HIST + _N_LIVE):
        base = bar * _LTF_PER_BAR
        expected_sum.append(sum(base + i + 1 for i in range(_LTF_PER_BAR)))

    with tempfile.TemporaryDirectory() as td:
        fixture_path = Path(td) / "ltf_fixture.json"
        fixture_path.write_text(json.dumps(fixture), encoding="utf-8")

        ps = PluginSymbol(
            provider_name="replay",
            symbol="LTFSYM",
            timeframe="1",
            config=ReplayConfig(fixture_path=str(fixture_path)),
            syminfo=_ltf_syminfo(),
            time_from=datetime.fromtimestamp(_TS0, UTC),
        )
        security_data = {"EXCH:LTFSYM:1": ps}

        historical = [_chart_ohlcv(_TS0 + i * _CHART_STEP) for i in range(_N_HIST)]
        live_chart = [_chart_ohlcv(_TS0 + (_N_HIST + i) * _CHART_STEP) for i in range(_N_LIVE)]

        setattr(lib, '_is_live', True)
        try:
            runner = ScriptRunner(
                script_path,
                itertools.chain(historical, [LIVE_TRANSITION], live_chart),
                syminfo,
                security_data=security_data,
            )
            rows = []
            for _candle, pv in runner.run_iter():
                rows.append((pv["n"], pv["sum"]))
        finally:
            setattr(lib, '_is_live', False)

    _assert_rows(rows, expected_size, expected_sum, log)


def _assert_rows(rows, expected_size, expected_sum, log):
    import math
    from pynecore.types.na import NA

    assert len(rows) == len(expected_size), \
        f"bar count {len(rows)} != {len(expected_size)}"
    for i, (n, s) in enumerate(rows):
        assert int(n) == expected_size[i], f"bar {i}: size {n} != {expected_size[i]}"
        exp = expected_sum[i]
        if exp is None:
            assert isinstance(s, NA), f"bar {i}: expected na, got {s!r}"
        else:
            assert not isinstance(s, NA), f"bar {i}: expected {exp}, got na"
            assert math.isclose(float(s), exp, abs_tol=1e-9), \
                f"bar {i}: sum {s} != {exp}"
    log.info("live LTF e2e warmup+closed windows correct: %s", rows)
