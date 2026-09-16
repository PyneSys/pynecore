"""
@pyne
"""
from pynecore.lib import barmerge, close, input, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="Deferred Producer Start", shorttitle="DPS")
def main(htf=input.timeframe(defval="60", title="HTF")):
    # A producer whose timeframe is only known at runtime: its symbol and
    # timeframe are the helper's parameters, so the chart resolves it at its
    # first signal. The lift moves that signal to the top of ``main()``, BEHIND
    # the consumer's own.
    def fetch(tf, src):
        return request.security(syminfo.tickerid, tf, src)

    producer: Series[float] = fetch(htf, close)
    # A ``lookahead_on`` consumer of that producer. A developing-bar context
    # that depends on another is started at its first signal, which comes
    # before the producer's: the producer cannot be spawned yet, and it has to
    # start once its own signal resolves it.
    consumer: Series[float] = request.security(
        syminfo.tickerid, "60", producer * 2.0, lookahead=barmerge.lookahead_on)
    plot(producer, "producer")
    plot(consumer, "consumer")


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

    path = tmp_dir / "HTF60.ohlcv"
    with OHLCVWriter(path, "60") as w:
        for hour in range(_N_HOURS):
            c = 100.0 + hour
            w.write(OHLCV(timestamp=_T0 + hour * _HOUR, open=c, high=c, low=c,
                          close=c, volume=1.0))
    SymInfo(
        prefix="EXCH", description="Deferred HTF", ticker="DPS",
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


def __test_consumer_started_before_its_runtime_resolved_producer__(runner, log):
    """A consumer started before its runtime-resolved producer does not deadlock.

    Starting the consumer starts its producers first. This producer has no data
    path until its own signal resolves it, so starting it at that moment spawns
    nothing — and a producer marked started without a child is never started
    again: the chart's first read of it would wait forever on a process that
    does not exist. The producer has to start at its own signal instead, and
    the consumer has to read the values it produces.
    """
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.types.na import NA

    sys.modules.pop(Path(__file__).stem, None)

    def scenario():
        rows = {}
        with tempfile.TemporaryDirectory() as td:
            feed = __test_helper_write_feed(Path(td))
            r = runner(__test_helper_chart_bars(), security_data={"60": feed})
            for i, (_candle, pv) in enumerate(r.run_iter()):
                rows[i] = dict(pv)
        return rows

    rows = __test_helper_run_with_timeout(scenario, seconds=60)

    assert len(rows) == _N_HOURS * 12
    # From the second hour on the producer answers the previous closed hour and
    # the consumer, reading the producer inside its own child, doubles a real
    # value; neither may be stuck at ``na``.
    later = [rows[i] for i in range(24, len(rows))]
    assert all(not isinstance(v["producer"], NA) for v in later), "producer stuck at na"
    assert all(not isinstance(v["consumer"], NA) for v in later), "consumer stuck at na"
    assert all(v["consumer"] >= 2.0 * 100.0 for v in later), \
        f"consumer did not read the producer: {[v['consumer'] for v in later[:5]]}"
    log.info("consumer started ahead of its runtime-resolved producer without a deadlock")
