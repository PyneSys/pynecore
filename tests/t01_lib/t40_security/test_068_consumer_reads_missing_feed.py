"""
@pyne
"""
from pynecore.lib import close, input, plot, request, script, syminfo, ta
from pynecore.types import Series


@script.indicator(title="Consumer Reads Missing Feed", shorttitle="CRMF")
def main(htf=input.timeframe(defval="60", title="HTF")):
    # The inner context is resolved at RUNTIME (its timeframe is an input), so a
    # missing feed is tolerated until something reads it. Nothing on the chart
    # does — the only reader is the outer context, which runs in a child process
    # and reaches the inner one over the peer protocol.
    dep: Series[float] = request.security(
        "EXCH:RRP", "60", ta.sma(request.security(syminfo.tickerid, htf, close), 3))
    plot(dep, "dep")


# Every timestamp here is Unix MILLISECONDS.
_T0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, aligned to the 1h and 5m grids
_HOUR = 3_600_000
_CHART_STEP = 300_000  # 5 minutes
_N_HOURS = 12


def __test_helper_write_feed(tmp_dir):
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "RRP60.ohlcv"
    with OHLCVWriter(path, "60") as w:
        for hour in range(_N_HOURS):
            c = 100.0 + hour
            w.write(OHLCV(timestamp=_T0 + hour * _HOUR, open=c, high=c, low=c,
                          close=c, volume=1.0))
    SymInfo(
        prefix="EXCH", description="Consumer feed", ticker="RRP",
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


def __test_helper_run_with_timeout(fn, seconds):
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


def __test_consumer_read_reports_the_missing_feed__(runner, log):
    """A context read only by another context's child still fails the run

    The chart never reads the inner context, so the missing feed can only
    surface on the peer path: the consumer child is told the feed is missing
    and fails, instead of being handed ``na`` and computing a whole column of
    silently wrong numbers.
    """
    import os
    import sys
    import tempfile
    from pathlib import Path

    sys.modules.pop(Path(__file__).stem, None)

    def scenario():
        os.environ['PYNE_SAVE_SCRIPT_TOML'] = '0'
        with tempfile.TemporaryDirectory() as td:
            feed = __test_helper_write_feed(Path(td))
            rows = {}
            error = None
            r = runner(__test_helper_chart_bars(),
                       security_data={"EXCH:RRP:60": feed})
            try:
                for i, (_candle, pv) in enumerate(r.run_iter()):
                    rows[i] = dict(pv)
            except Exception as exc:  # noqa: BLE001
                error = exc
            return rows, error

    rows, error = __test_helper_run_with_timeout(scenario, seconds=120)

    assert error is not None, \
        f"the consumer read of an unprovisioned context stayed silent ({len(rows)} bars)"

    log.info("a consumer child reports the missing feed instead of answering na")
