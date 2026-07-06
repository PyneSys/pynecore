"""
@pyne
"""
from pynecore.lib import plot, request, script, timeframe, volume
from pynecore.types import Persistent, Series


def _accum_volume():
    # A ``var`` accumulator that resets at the start of every CHART bar and sums
    # the intrabar volumes within it. The reset key is ``timeframe.main_period``,
    # which must report the CHART timeframe even though this expression runs in a
    # lower-timeframe ``request.security`` child (where ``lib._script`` is None).
    # If ``main_period`` wrongly reported the child's own period, the reset would
    # fire on every intrabar and the accumulator would only ever hold the last
    # intrabar's volume. This mirrors ``TradingView/ta``'s ``upAndDownVolumeCalc``.
    acc: Persistent[float] = 0.0
    if timeframe.change(timeframe.main_period):
        acc = 0.0
    acc += volume
    return acc


@script.indicator(title="LTF main_period reset", shorttitle="LMPR")
def main():
    v: Series[float] = request.security("EXCH:LMPR", "1", _accum_volume())
    plot(v, "acc")


# Chart runs at the default test period "5" (300s); the LTF feed is 60s, so each
# chart bar contains exactly five 1-minute intrabars. Intrabar volumes cycle
# 1,2,3,4,5 within each chart bar, so the per-bar SUM is 15 and the LAST intrabar
# alone is 5 -- a value that cleanly separates the correct sum from the bug.
_TS0 = 1735689600  # 2025-01-01T00:00:00 UTC, aligned to both the 300s and 60s grids
_CHART_STEP = 300


def _write_ltf(tmp_dir, n_chart_bars):
    from datetime import time
    from pynecore.core.ohlcv_file import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "TEST_LMPR_1.ohlcv"
    with OHLCVWriter(path) as w:
        for j in range(n_chart_bars * 5):
            vol = float(j % 5 + 1)  # 1,2,3,4,5 within each chart bar
            w.write(OHLCV(timestamp=_TS0 + j * 60,
                          open=100.0, high=100.0, low=100.0, close=100.0, volume=vol))
    SymInfo(
        prefix="EXCH", description="LMPR", ticker="LMPR",
        currency="USD", period="1", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def _chart_bars(n):
    from pynecore.types.ohlcv import OHLCV
    return [
        OHLCV(timestamp=_TS0 + i * _CHART_STEP,
              open=100.0, high=100.0, low=100.0, close=100.0, volume=5.0)
        for i in range(n)
    ]


def __test_main_period_reset_in_ltf_child__(runner, log):
    """``timeframe.main_period`` reports the chart TF inside a request.security LTF child.

    A ``var`` accumulator in the security expression resets on
    ``timeframe.change(timeframe.main_period)`` and sums the chart bar's intrabar
    volumes. With ``main_period`` correctly propagated (chart "5"), each chart bar
    returns the full sum (15); the pre-fix bug -- ``main_period`` resolving to the
    child's own period "1" -- reset every intrabar and would return only the last
    intrabar's volume (5).
    """
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.types.na import NA

    sys.modules.pop(Path(__file__).stem, None)

    n = 6
    rows = []
    with tempfile.TemporaryDirectory() as td:
        ltf = _write_ltf(Path(td), n)
        r = runner(_chart_bars(n), security_data={"EXCH:LMPR:1": ltf})
        for _candle, pv in r.run_iter():
            rows.append(pv.get("acc"))

    # Every settled chart bar must expose the full intrabar sum (15), never the
    # last-intrabar-only value (5) that the reset-every-intrabar bug produced.
    settled = [v for v in rows if not isinstance(v, NA)]
    assert settled, "no settled request.security values were produced"
    assert all(float(v) == 15.0 for v in settled), \
        f"expected every chart bar's accumulator to sum to 15, got {settled}"
    log.info("timeframe.main_period reports the chart TF inside an LTF request.security child")
