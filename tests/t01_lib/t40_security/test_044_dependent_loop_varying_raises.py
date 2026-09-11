"""
@pyne
"""
from pynecore.lib import close, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="Dependent Loop Varying", shorttitle="DLV")
def main():
    # The loop index enters the requested expression, so the sid's writes carry
    # DIFFERENT values within one bar. The chart may already have read the first
    # one, so there is no single answer: the run must fail loudly rather than
    # silently keep the last iteration's value.
    base: Series[float] = request.security(syminfo.tickerid, "60", close)
    looped = 0.0
    for i in range(3):
        looped = request.security(syminfo.tickerid, "60", base + i)
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
        prefix="EXCH", description="Loop HTF", ticker="DLV",
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


def __test_loop_varying_security_expression_raises__(runner, log):
    """A loop-VARYING security expression fails instead of silently keeping the last write.

    Before the ring the child overwrote its slot on every iteration and the
    chart read whichever value happened to be last. That is unanswerable once
    the first write has already published, so a differing repeat is an error.
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

    raised = None
    try:
        __test_helper_run_with_timeout(scenario)
    except AssertionError:
        raise
    except BaseException as exc:  # noqa: BLE001 - the failure mode is the assertion
        raised = exc
    assert raised is not None, "a loop-varying security expression must not run silently"
    # The child raises ``RuntimeError("loop-varying security expression is not
    # supported")``; the chart surfaces its death as a RuntimeError of its own.
    assert isinstance(raised, RuntimeError), \
        f"expected a RuntimeError, got {type(raised).__name__}: {raised}"
    log.info("loop-varying security expression rejected: %s: %s",
             type(raised).__name__, raised)
