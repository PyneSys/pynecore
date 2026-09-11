"""
@pyne
"""
from pynecore.lib import barmerge, close, high, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="Gaps On Live Rerun", shorttitle="GOLR")
def main():
    # A ``gaps_on`` dependent consumer driven over DEVELOPING chart bars. The
    # read state is keyed by consumer bar AND as-of, so re-running the same
    # developing bar must give the same answer instead of turning a held value
    # into ``na`` (or the other way round) on the second pass.
    a: Series[float] = request.security(syminfo.tickerid, "120", close)
    b: Series[float] = request.security(syminfo.tickerid, "120", a + 1000.0,
                                        gaps=barmerge.gaps_on)
    plot(a, "a")
    plot(b, "b")
    plot(a, "v")


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


def __test_helper_bar(hour, is_closed=True):
    from pynecore.types.ohlcv import OHLCV
    return OHLCV(timestamp=_T0 + hour * _HOUR, open=__test_helper_close(hour),
                 high=__test_helper_high(hour), low=__test_helper_close(hour),
                 close=__test_helper_close(hour), volume=1.0, is_closed=is_closed)


def __test_helper_write_feed(tmp_dir):
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession

    path = tmp_dir / "FEED.ohlcv"
    with OHLCVWriter(path, "60") as w:
        for hour in range(_N_HOURS):
            w.write(__test_helper_bar(hour))
    SymInfo(
        prefix="EXCH", description="Live two HTF", ticker="GOLR",
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


def __test_gaps_on_consumer_is_stable_across_developing_reruns__(script_path, module_key,
                                                                 syminfo, log):
    """A ``gaps_on`` dependent consumer answers the same on every re-run of one bar.

    ``gaps_on`` emits ``na`` on a consumer bar that brings no new producer bar,
    so it is exactly the mode a re-run can corrupt: running the same developing
    bar twice must not make the second pass believe a new entry arrived. The
    value is therefore present precisely on the chart bars that open a fresh
    120-minute period, historical and live alike.
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
            # Each live hour arrives as two DEVELOPING ticks before it closes,
            # so every live bar's ``main()`` runs three times.
            live = []
            for h in range(_N_HIST, _N_HOURS):
                live.append(__test_helper_bar(h, False))
                live.append(__test_helper_bar(h, False))
                live.append(__test_helper_bar(h, True))
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

    assert len(rows) == _N_HOURS, \
        f"bar count {len(rows)} != {_N_HOURS} — the run stalled on a developing tick"
    checked = 0
    for ts, a, b, _v in rows:
        hour = (ts - _T0) // _HOUR
        if hour == _N_HIST:
            # MEASURED: the first bar after LIVE_TRANSITION reports ``na`` for a
            # same-symbol HTF context even without any dependency (reproduced
            # with a single plain context). That belongs to the live HTF
            # transport, not to the gaps_on read state this file guards.
            continue
        if isinstance(a, NA):
            assert isinstance(b, NA), f"hour {hour}: a is na but b={b}"
            continue
        if hour % 2 == 1:
            # This chart bar opens a fresh 120-minute period: the producer bar
            # that just closed is new, so gaps_on yields it.
            assert b == a + 1000.0, f"hour {hour}: b={b} != a+1000={a + 1000.0}"
            checked += 1
        else:
            assert isinstance(b, NA), \
                f"hour {hour}: gaps_on returned {b} on a bar with no new producer bar"
    assert checked > 4, f"only {checked} gaps_on values seen"
    log.info("gaps_on dependent consumer stayed stable across developing re-runs")
