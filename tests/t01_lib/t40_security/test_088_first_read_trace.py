"""
@pyne
"""
from pynecore.lib import bar_index, close, high, plot, request, script, syminfo


def later_read(tf):
    """A helper defined ahead of ``main()`` but called last."""
    return request.security(syminfo.tickerid, tf, close)


@script.indicator(title="First Read Trace", shorttitle="FRT")
def main():
    # The signals of all three contexts are hoisted to the top of their scope;
    # the READS stay where the calls are written, and that is the order traced.
    first = request.security(syminfo.tickerid, "60", high)
    own = request.security(syminfo.tickerid, "5", close)
    gated = 0.0
    if bar_index >= 12:
        gated = later_read("120")
    plot(first, "first")
    plot(own, "own")
    plot(gated, "gated")


__test_helper_t0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, Unix MILLISECONDS
__test_helper_step = 300_000  # the chart's 5 minutes


def __test_helper_chart_bars():
    """Four hours of 5-minute chart bars."""
    from pynecore.types.ohlcv import OHLCV
    bars = []
    for i in range(48):
        c = 50.0 + (i % 7)
        bars.append(OHLCV(timestamp=__test_helper_t0 + i * __test_helper_step,
                          open=c, high=c + 1.0, low=c - 1.0, close=c, volume=1.0))
    return bars


def __test_helper_write_feed(tmp_dir, period, minutes):
    """Write one higher-timeframe feed with its 24/7 UTC ``.toml`` sidecar.

    :param tmp_dir: Directory to write into.
    :param period: The feed's timeframe string.
    :param minutes: The feed's bar length in minutes.
    :return: The feed path as a string.
    """
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / ("FRT" + period + ".ohlcv")
    with OHLCVWriter(path, period) as w:
        for i in range(240 // minutes):
            c = 100.0 + i
            w.write(OHLCV(timestamp=__test_helper_t0 + i * minutes * 60_000,
                          open=c, high=c + 1.0, low=c - 1.0, close=c, volume=1.0))
    SymInfo(
        prefix="PYTEST", description="First Read Trace", ticker="TEST",
        currency="USD", period=period, type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_first_reads_are_traced_in_call_site_order__(runner, tmp_path, log):
    """``PYNE_SECURITY_TRACE`` names every context once, resolved, as first read

    The chart-timeframe context is flagged, the runtime timeframe of the helper
    call is the resolved one, and the context behind the branch appears last
    although its helper is defined first.
    """
    import json
    import os
    import sys
    from pathlib import Path

    trace = tmp_path / "trace.jsonl"
    sys.modules.pop(Path(__file__).stem, None)
    os.environ['PYNE_SAVE_SCRIPT_TOML'] = '0'
    os.environ['PYNE_SECURITY_TRACE'] = str(trace)
    try:
        feeds = {"60": __test_helper_write_feed(tmp_path, "60", 60),
                 "120": __test_helper_write_feed(tmp_path, "120", 120)}
        for _candle, _plot in runner(__test_helper_chart_bars(),
                                     security_data=feeds).run_iter():
            pass
    finally:
        os.environ.pop('PYNE_SECURITY_TRACE', None)

    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert [row["timeframe"] for row in rows] == ["60", "5", "120"]
    assert [row["order"] for row in rows] == [0, 1, 2]
    assert [row["same_context"] for row in rows] == [False, True, False]
    assert len({row["sec_id"] for row in rows}) == 3
    log.info("three contexts traced once each, in the order main() reads them")


def __test_a_traced_run_names_every_missing_feed__(runner, tmp_path, log):
    """With no feed provisioned the traced run still completes, reading na

    Each unprovisioned context is logged once with ``"missing": true``, so one
    pass tells a provisioning tool everything the script needs; the chart's own
    context is never missing.
    """
    import json
    import os
    import sys
    from pathlib import Path

    trace = tmp_path / "trace.jsonl"
    sys.modules.pop(Path(__file__).stem, None)
    os.environ['PYNE_SAVE_SCRIPT_TOML'] = '0'
    os.environ['PYNE_SECURITY_TRACE'] = str(trace)
    rows = []
    try:
        for _candle, plots in runner(__test_helper_chart_bars()).run_iter():
            rows.append(dict(plots))
    finally:
        os.environ.pop('PYNE_SECURITY_TRACE', None)

    assert len(rows) == 48
    from pynecore.types.na import NA
    assert all(isinstance(row["first"], NA) for row in rows)
    traced = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert [(row["timeframe"], row["missing"]) for row in traced] == \
           [("60", True), ("5", False), ("120", True)]
    log.info("both unprovisioned contexts named in one pass, the run read na for them")

