"""
@pyne
"""
from pynecore.lib import close, high, plot, request, script, syminfo, timeframe
from pynecore.types import Series


@script.indicator(title="Live Chart Context Producer", shorttitle="LCCP")
def main():
    # ``own`` is a CHART-CONTEXT producer: same symbol and timeframe, so the
    # chart evaluates it itself and owns its ring. The 120-minute child consumes
    # it, which means the chart has to publish that ring entry at its own write
    # (and raise the frontier at the bar end) before the child can pair — live
    # rounds included.
    own: Series[float] = request.security(syminfo.tickerid, timeframe.period, close)
    dep: Series[float] = request.security(syminfo.tickerid, "120", own + 1000.0)
    plot(own, "a")
    plot(dep, "b")
    plot(own, "v")


# Every timestamp here is Unix MILLISECONDS.
_T0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, aligned to the 1h and 2h grids
_HOUR = 3_600_000
_N_HIST = 10
_N_LIVE = 6
_N_HOURS = _N_HIST + _N_LIVE


def __test_helper_close(hour):
    return 100.0 + hour


def __test_helper_high(hour):
    return 200.0 + hour


def __test_helper_bar(hour):
    from pynecore.types.ohlcv import OHLCV
    return OHLCV(timestamp=_T0 + hour * _HOUR, open=__test_helper_close(hour),
                 high=__test_helper_high(hour), low=__test_helper_close(hour),
                 close=__test_helper_close(hour), volume=1.0, is_closed=True)


def __test_helper_write_feed(tmp_dir):
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession

    path = tmp_dir / "FEED.ohlcv"
    with OHLCVWriter(path, "60") as w:
        for hour in range(_N_HOURS):
            w.write(__test_helper_bar(hour))
    SymInfo(
        prefix="EXCH", description="Live two HTF", ticker="LCCP",
        currency="USD", period="60", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


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


def __test_live_chart_context_producer_feeds_its_child_consumer__(script_path, module_key,
                                                                   syminfo, log):
    """A chart-context producer publishes for its child consumer, live rounds included.

    The chart itself evaluates a same-symbol same-timeframe context, so it is
    the producer: it appends to that context's ring at its own write and raises
    the frontier from the bar-end hook. A developing chart bar never appends —
    its re-runs would scatter entries sharing one open — so the consumer pairs
    only against closed chart bars, exactly as the historical path does.
    """
    import sys
    import itertools
    import tempfile
    from pathlib import Path

    from pynecore import lib
    from pynecore.core.script_runner import ScriptRunner, LIVE_TRANSITION
    from pynecore.types.na import NA

    sys.modules.pop(Path(__file__).stem, None)
    sys.modules.pop(module_key, None)

    def scenario():
        with tempfile.TemporaryDirectory() as td:
            feed = __test_helper_write_feed(Path(td))
            historical = [__test_helper_bar(h) for h in range(_N_HIST)]
            live = [__test_helper_bar(h) for h in range(_N_HIST, _N_HOURS)]
            setattr(lib, '_is_live', True)
            try:
                r = ScriptRunner(
                    script_path,
                    itertools.chain(historical, [LIVE_TRANSITION], live),
                    syminfo, security_data={"120": feed})
                return [(c.timestamp, pv.get("a"), pv.get("b"), pv.get("v"))
                        for c, pv in r.run_iter()]
            finally:
                setattr(lib, '_is_live', False)

    syminfo.period = "60"
    rows = __test_helper_run_with_timeout(scenario)

    assert len(rows) == _N_HOURS, f"bar count {len(rows)} != {_N_HOURS}"
    checked = 0
    for i, (ts, a, b, v) in enumerate(rows):
        hour = (ts - _T0) // _HOUR
        assert hour == i
        # 120-minute bar ``k`` covers hours 2k and 2k+1 and closes at hour 2k+2;
        # the chart bar of hour h closes at h+1, so the last CLOSED one is
        # ``(h - 1) // 2``.
        k = (hour - 1) // 2
        if k < 0:
            continue
        if hour == _N_HIST:
            # MEASURED: the first bar after LIVE_TRANSITION reports ``na`` for a
            # same-symbol HTF context even without any dependency (reproduced
            # with a single plain context). That belongs to the live HTF
            # transport, not to the chart-context producer this file guards.
            continue
        # The chart-context producer is the chart's own close.
        assert a == __test_helper_close(hour), f"hour {hour}: own={a}"
        # The 120-minute consumer's bar k closes at hour 2k+2 and pairs against
        # the last chart entry closing at or before that: the hour-2k+1 bar.
        assert b == __test_helper_close(2 * k + 1) + 1000.0, \
            f"hour {hour}: dep={b} != {__test_helper_close(2 * k + 1) + 1000.0}"
        checked += 1
    assert checked >= _N_HOURS - 3, f"only {checked} bars checked"
    log.info("live chart-context producer fed its child consumer on %d bars", checked)
