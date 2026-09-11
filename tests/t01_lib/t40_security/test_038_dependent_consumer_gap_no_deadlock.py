"""
@pyne
"""
from pynecore.lib import close, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="Consumer Gap No Deadlock", shorttitle="CGND")
def main():
    # ``gappy`` is an hourly context whose own feed is missing an hour, and it
    # consumes the dense hourly ``base``. A missing consumer bar must not push
    # the consumer's as-of past what the chart targets in this round — the
    # chart-as-of cap is what keeps the wait bounded.
    base: Series[float] = request.security(syminfo.tickerid, "60", close)
    gappy: Series[float] = request.security("EXCH:GAPPY", "60", base)
    plot(base, "base")
    plot(gappy, "gappy")


# Every timestamp here is Unix MILLISECONDS.
_T0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, aligned to the 1h and 5m grids
_HOUR = 3_600_000
_CHART_STEP = 300_000  # 5 minutes
_N_HOURS = 9
_MISSING_HOUR = 4  # absent from the CONSUMER's feed only


def __test_helper_htf_close(hour):
    return 100.0 + hour


def __test_helper_syminfo(ticker, period):
    from datetime import time
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    return SymInfo(
        prefix="EXCH", description=ticker, ticker=ticker,
        currency="USD", period=period, type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    )


def __test_helper_write_feed(tmp_dir, ticker, skip_hour):
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / f"{ticker}.ohlcv"
    with OHLCVWriter(path, "60") as w:
        for hour in range(_N_HOURS):
            if hour == skip_hour:
                continue
            c = __test_helper_htf_close(hour)
            w.write(OHLCV(timestamp=_T0 + hour * _HOUR, open=c, high=c, low=c,
                          close=c, volume=1.0))
    __test_helper_syminfo(ticker, "60").save_toml(path.with_suffix(".toml"))
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


def __test_a_gap_in_the_consumer_feed_does_not_deadlock__(runner, log):
    """A missing bar in the CONSUMER's feed must not strand it on an untargeted peer bar.

    The consumer's bar before the hole closes an hour early (its next bar opens
    two hours later), so its own as-of does not stretch across the gap. Were it
    to, it would ask for a producer bar the chart does not confirm in this round
    and the frontier would never reach it. The values also stay exact: the
    consumer's hour-3 bar carries the producer's hour-3 value, and the chart
    holds it until the consumer's hour-5 bar closes.
    """
    import sys
    import tempfile
    from pathlib import Path

    sys.modules.pop(Path(__file__).stem, None)

    def scenario():
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            dense = __test_helper_write_feed(tmp, "DENSE", -1)
            gappy = __test_helper_write_feed(tmp, "GAPPY", _MISSING_HOUR)
            r = runner(__test_helper_chart_bars(),
                       security_data={"60": dense, "EXCH:GAPPY:60": gappy})
            rows = {}
            for candle, pv in r.run_iter():
                minute = (candle.timestamp - _T0) // 60_000
                rows[(minute // 60, minute % 60)] = (pv.get("base"), pv.get("gappy"))
            return rows

    rows = __test_helper_run_with_timeout(scenario)

    assert len(rows) == _N_HOURS * 12, "the run did not reach the last chart bar"

    # Before the hole the consumer tracks the producer one bar behind the chart.
    assert rows[(3, 0)][1] == __test_helper_htf_close(2), f"hour 3: gappy={rows[(3, 0)][1]}"
    # The consumer's hour-3 bar closes at 04:00 (its next bar only opens at
    # 05:00), so hours 4 AND 5 report it — the hole adds no value of its own.
    for hour in (4, 5):
        assert rows[(hour, 0)][1] == __test_helper_htf_close(3), \
            f"hour {hour}: gappy={rows[(hour, 0)][1]} != hour-3 value"
    # The consumer's hour-5 bar closes at 06:00 and carries the producer's
    # hour-5 value, which the chart picks up on its own hour-6 bars.
    assert rows[(6, 0)][1] == __test_helper_htf_close(5), \
        f"hour 6: gappy={rows[(6, 0)][1]} != hour-5 value"
    # The dense producer is unaffected by the consumer's hole.
    for hour in range(1, _N_HOURS):
        assert rows[(hour, 0)][0] == __test_helper_htf_close(hour - 1), \
            f"hour {hour}: base={rows[(hour, 0)][0]}"
    log.info("a consumer-side data gap neither deadlocks nor shifts a value")
