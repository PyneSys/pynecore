"""
@pyne
"""
from pynecore.lib import close, input, plot, request, script, syminfo, ta
from pynecore.types import Series


@script.indicator(title="Dependent Input Timeframe", shorttitle="DIT")
def main():
    # The context timeframe comes from an input, so the dependency analysis has
    # to hoist the binding: a dependent context's timeframe must be resolvable
    # ahead of every write block, or the chain could not be signalled in time.
    htf = input.timeframe("60", "HTF")
    base: Series[float] = request.security(syminfo.tickerid, htf, close)
    dep: Series[float] = request.security(syminfo.tickerid, htf, ta.sma(base, 3))
    plot(base, "base")
    plot(dep, "dep")


# Every timestamp here is Unix MILLISECONDS.
_T0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, aligned to the 1h and 5m grids
_HOUR = 3_600_000
_CHART_STEP = 300_000  # 5 minutes
_N_HOURS = 9


def __test_helper_htf_close(hour):
    return 100.0 + hour


def __test_helper_write_feed(tmp_dir):
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "HTF60.ohlcv"
    with OHLCVWriter(path, "60") as w:
        for hour in range(_N_HOURS):
            c = __test_helper_htf_close(hour)
            w.write(OHLCV(timestamp=_T0 + hour * _HOUR, open=c, high=c, low=c,
                          close=c, volume=1.0))
    SymInfo(
        prefix="EXCH", description="Input TF HTF", ticker="DIT",
        currency="USD", period="60", type="crypto",
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
    return [OHLCV(timestamp=_T0 + i * _CHART_STEP, open=1.0, high=1.0, low=1.0,
                  close=1.0, volume=1.0)
            for i in range(_N_HOURS * 12)]


def __test_helper_run_with_timeout(fn, seconds=120):
    """Run ``fn`` on a daemon thread; a deadlock fails the test instead of hanging."""
    import threading
    box = {}

    def target():
        try:
            box['result'] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            box['error'] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise AssertionError(f"deadlock: the run did not finish within {seconds}s")
    if 'error' in box:
        raise box['error']
    return box.get('result')


def __test_dependent_chain_on_an_input_timeframe__(runner, log):
    """A dependent chain works when the context timeframe comes from an input.

    The timeframe is a Pine "simple" value, so the binding hoists to the top
    block and both contexts are signalled before any write block runs. Had the
    dependency not been hoistable the transform would have refused the script;
    here it must simply produce the same values a literal ``"60"`` would.
    """
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.types.na import NA

    sys.modules.pop(Path(__file__).stem, None)

    def scenario():
        with tempfile.TemporaryDirectory() as td:
            feed = __test_helper_write_feed(Path(td))
            r = runner(__test_helper_chart_bars(), security_data={"60": feed})
            rows = {}
            for candle, pv in r.run_iter():
                minute = (candle.timestamp - _T0) // 60_000
                rows[(minute // 60, minute % 60)] = (pv.get("base"), pv.get("dep"))
            return rows

    rows = __test_helper_run_with_timeout(scenario)

    for hour in range(3, _N_HOURS):
        base, dep = rows[(hour, 0)]
        j = hour - 1  # the last CLOSED hourly bar
        assert base == __test_helper_htf_close(j), f"hour {hour}: base={base}"
        expected = __test_helper_htf_close(j) - 1.0  # mean of closes j-2..j
        assert not isinstance(dep, NA) and abs(dep - expected) < 1e-9, \
            f"hour {hour}: dep={dep} != {expected}"
    log.info("dependent chain on an input-driven timeframe matches the literal form")
