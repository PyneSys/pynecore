"""
@pyne
"""
from pynecore.lib import array, close, na, plot, request, script, syminfo, time


@script.indicator(title="LTF Invalid Timeframe", shorttitle="LTFIT")
def main():
    # Above the 60-minute chart: na, every element of a tuple too
    t_120, c_120 = request.security_lower_tf(syminfo.tickerid, "120", (time, close),
                                             ignore_invalid_timeframe=True)
    c_d = request.security_lower_tf(syminfo.tickerid, "D", close, ignore_invalid_timeframe=True)
    # The chart's own timeframe: one intrabar, the chart bar itself
    c_60 = request.security_lower_tf(syminfo.tickerid, "60", close)
    c_15 = request.security_lower_tf(syminfo.tickerid, "15", close)
    # Computed, so its child runs main() (unsliced: the whole of it)
    htf = request.security(syminfo.tickerid, "240", close + 0.0)
    last_d = 0.0
    # The guard scripts write around such a request (converter corpus
    # 1a04048648): an empty array in place of na passes it, and the call raises
    if not na(c_d):
        last_d = array.last(c_d)
    plot(last_d + htf, "x")
    return {"t_120": t_120, "c_120": c_120, "c_d": c_d, "c_60": c_60, "c_15": c_15}


def __test_helper_run(runner, no_slice):
    """Run the script once.

    :param runner: The ``runner`` fixture.
    :param no_slice: Whether to switch the security slice off, so the ``240``
        child replays the whole script and reads the na contexts too.
    :return: The values per chart bar.
    """
    import os
    import sys
    import tempfile
    from pathlib import Path

    sys.modules.pop(Path(__file__).stem, None)
    rows: list[dict] = []
    if no_slice:
        os.environ['PYNE_NO_SECURITY_SLICE'] = '1'
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            security_data = {
                "15": __test_helper_write_feed(tmp, "15", __test_helper_quarter),
                "240": __test_helper_write_feed(tmp, "240", 4 * __test_helper_step),
            }
            r = runner(__test_helper_bars(__test_helper_step), {"period": "60"},
                       security_data=security_data)
            for _candle, values in r.run_iter():
                rows.append(dict(values))
    finally:
        os.environ.pop('PYNE_NO_SECURITY_SLICE', None)
    return rows


def __test_an_ltf_above_the_script_timeframe_answers_na__(runner, log):
    """``ignore_invalid_timeframe=true`` above the chart's timeframe gives ``na``.

    TradingView, on an hourly chart: ``120`` and ``D`` answer ``na`` (each
    element of a tuple as well), ``60`` a one-element array and ``15`` four
    elements. PyneCore answered an empty array, which passes a ``not na(...)``
    guard, so the script's own ``array.last()`` raised on the chart and in every
    child replaying the script.
    """
    from pynecore.types.na import NA

    closes = [b.close for b in __test_helper_bars(__test_helper_step)]
    quarters = [b.close for b in __test_helper_bars(__test_helper_quarter)]
    per_bar = __test_helper_step // __test_helper_quarter
    for no_slice in (False, True):
        rows = __test_helper_run(runner, no_slice)
        assert len(rows) == len(closes), f"no_slice={no_slice}: {len(rows)} bars"
        for i, row in enumerate(rows):
            for key in ("t_120", "c_120", "c_d"):
                assert row[key].__class__ is NA, f"bar {i} {key}: {row[key]!r}"
            assert list(row["c_60"]) == [closes[i]], f"bar {i} 60: {row['c_60']}"
            want = quarters[i * per_bar:(i + 1) * per_bar]
            assert list(row["c_15"]) == want, f"bar {i} 15: {row['c_15']} != {want}"
    log.info("%d chart bars, sliced and unsliced", len(closes))


# Every timestamp here is Unix MILLISECONDS.
__test_helper_t0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC
__test_helper_quarter = 900_000
__test_helper_step = 3_600_000  # 60 minutes, the chart
__test_helper_span = 2 * 24 * __test_helper_step


def __test_helper_bars(span_ms):
    """Bars of ``span_ms`` over the test's days, with a moving close.

    :param span_ms: The bar span in ms.
    :return: The bar list.
    """
    from pynecore.types.ohlcv import OHLCV
    bars = []
    for i in range(__test_helper_span // span_ms):
        c = 100.0 + (i % 11) * 0.5
        bars.append(OHLCV(timestamp=__test_helper_t0 + i * span_ms, open=c - 0.25,
                          high=c + 1.0, low=c - 1.0, close=c, volume=1.0 + i))
    return bars


def __test_helper_write_feed(tmp_dir, timeframe, span_ms):
    """Write one feed covering the chart's days.

    :param tmp_dir: Directory to write into.
    :param timeframe: The feed's timeframe string.
    :param span_ms: That timeframe's period length in ms.
    :return: Path to the written ``.ohlcv`` file, as a string.
    """
    from datetime import time as dtime
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession

    path = tmp_dir / ("LTFIT" + timeframe + ".ohlcv")
    with OHLCVWriter(path, timeframe) as w:
        for bar in __test_helper_bars(span_ms):
            w.write(bar)
    SymInfo(
        prefix="PYTEST", description="LTF invalid timeframe", ticker="TEST",
        currency="USD", period=timeframe, type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=dtime(0, 0), end=dtime(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=dtime(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=dtime(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)
