"""
@pyne
"""
from pynecore.lib import close, plot, request, script, syminfo, ta
from pynecore.types import Series


@script.indicator(title="Dependent Ring GC", shorttitle="DRG")
def main():
    # A long warmup: the child replays hundreds of hourly bars in ONE round, so
    # its ring passes its initial capacity and has to grow (and collect the
    # entries every consumer has moved past) while the consumer is reading it.
    base: Series[float] = request.security(syminfo.tickerid, "60", close)
    dep: Series[float] = request.security(syminfo.tickerid, "60", ta.sma(base, 5))
    plot(base, "base")
    plot(dep, "dep")


# Every timestamp here is Unix MILLISECONDS.
_T0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, aligned to the 1h and 5m grids
_HOUR = 3_600_000
_CHART_STEP = 300_000  # 5 minutes
# Far more producer bars than the ring's initial capacity, and the chart starts
# deep inside the feed so most of them are replayed in the warmup round.
_N_HOURS = 500
_CHART_START_HOUR = 460


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
        prefix="EXCH", description="Ring GC HTF", ticker="DRG",
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
    first = _CHART_START_HOUR * 12
    return [OHLCV(timestamp=_T0 + i * _CHART_STEP, open=1.0, high=1.0, low=1.0,
                  close=1.0, volume=1.0)
            for i in range(first, _N_HOURS * 12)]


def __test_helper_run_with_timeout(fn, seconds=180):
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


def __test_ring_growth_and_gc_keep_the_dependent_values_exact__(runner, log):
    """Hundreds of producer bars in one round: the ring grows and collects, values hold.

    The warmup round appends far more entries than the ring's initial capacity.
    Growth (reallocation with a version bump) and garbage collection below the
    consumers' watermark both run while the consumer is pairing against the ring,
    so a bug in either shows up directly as a wrong ``ta.sma`` value.
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

    checked = 0
    for hour in range(_CHART_START_HOUR + 1, _N_HOURS):
        base, dep = rows[(hour, 0)]
        j = hour - 1  # the last CLOSED hourly bar
        assert base == __test_helper_htf_close(j), f"hour {hour}: base={base}"
        # sma over the hourly closes j-4..j, whose mean is close(j) - 2.
        expected = __test_helper_htf_close(j) - 2.0
        assert not isinstance(dep, NA) and abs(dep - expected) < 1e-9, \
            f"hour {hour}: dep={dep} != {expected}"
        checked += 1
    assert checked > 30, f"only {checked} bars checked — the fixture is too short"
    log.info("ring growth + GC across a %d-bar warmup left every value exact", _N_HOURS)
