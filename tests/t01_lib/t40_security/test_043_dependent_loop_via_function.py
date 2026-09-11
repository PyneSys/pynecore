"""
@pyne
"""
from pynecore.lib import close, plot, request, script, syminfo
from pynecore.types import Series


def doubled(base):
    """One call site, reached from a loop body: its sid is marked ``in_loop`` too."""
    return request.security(syminfo.tickerid, "60", base * 2.0)


@script.indicator(title="Dependent Loop Via Function", shorttitle="DLVF")
def main():
    # The security call sits in a FUNCTION, and the loop calls that function.
    # Instantiation clones per call site, not per iteration, so the clone's sid
    # writes several times per bar and must carry the ``in_loop`` marking
    # through the call chain — otherwise the second write is an internal error.
    base: Series[float] = request.security(syminfo.tickerid, "60", close)
    looped = 0.0
    for _i in range(3):
        looped = doubled(base)
    plot(base, "base")
    plot(looped, "looped")


# Every timestamp here is Unix MILLISECONDS.
_T0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, aligned to the 1h and 5m grids
_HOUR = 3_600_000
_CHART_STEP = 300_000  # 5 minutes
_N_HOURS = 8


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
        prefix="EXCH", description="Loop HTF", ticker="DLVF",
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


def __test_loop_calling_a_function_with_a_dependent_security__(runner, log):
    """A dependent security reached through a function called from a loop still works.

    ``SecurityInstantiation`` clones a called function per CALL SITE, so one
    clone's write block runs once per loop iteration. The ``in_loop`` marking has
    to travel the call chain: without it the repeat write is an internal error,
    and with it the identical repeats are no-ops.
    """
    import sys
    import tempfile
    from pathlib import Path

    sys.modules.pop(Path(__file__).stem, None)

    def scenario():
        with tempfile.TemporaryDirectory() as td:
            feed = __test_helper_write_feed(Path(td))
            r = runner(__test_helper_chart_bars(), security_data={"60": feed})
            rows = {}
            for candle, pv in r.run_iter():
                minute = (candle.timestamp - _T0) // 60_000
                rows[(minute // 60, minute % 60)] = (pv.get("base"), pv.get("looped"))
            return rows

    rows = __test_helper_run_with_timeout(scenario)

    for hour in range(1, _N_HOURS):
        base, looped = rows[(hour, 0)]
        assert base == __test_helper_htf_close(hour - 1), f"hour {hour}: base={base}"
        assert looped == base * 2.0, f"hour {hour}: looped={looped} != 2*base={base * 2.0}"
    log.info("loop-invariant dependent security matches the plain form")
