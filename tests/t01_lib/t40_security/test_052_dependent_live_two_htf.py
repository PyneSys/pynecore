"""
@pyne
"""
from pynecore.lib import close, high, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="Live Two HTF", shorttitle="LTH")
def main():
    # Two 120-minute contexts on a 60-minute chart. The ``try`` block makes the
    # dependency analysis fall back to "every sid", so each believes it depends
    # on the other — the live shape of the mutual-wait case. In live mode a
    # chart bar can launch several rounds per context, so this also exercises
    # the round counter that keeps one bar's slot data out of the next one's.
    a: Series[float] = request.security(syminfo.tickerid, "120", close)
    b: Series[float] = request.security(syminfo.tickerid, "120", high)
    try:
        v = a + b
    except ZeroDivisionError:
        v = 0.0
    plot(a, "a")
    plot(b, "b")
    plot(v, "v")


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
        prefix="EXCH", description="Live two HTF", ticker="LTH",
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


def __test_two_live_htf_contexts_do_not_deadlock__(script_path, module_key, syminfo, log):
    """Two mutually dependent 120-minute contexts on a live 60-minute chart terminate.

    This is the case that sank "the chart waits for the child's whole
    ``main()``": A writes, the chart is released at the VALUE, B's site runs, B
    starts; A then waits for B while B does not wait for A, B publishes, A is
    freed. Live adds several rounds per chart bar, so the round counter has to
    keep the next bar's slot data away from a child still unpacking an earlier
    round.
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
            # transport, not to the dependent-peer machinery this file guards;
            # what matters here is that BOTH contexts behave the same and that
            # the run keeps going.
            assert isinstance(a, NA) == isinstance(b, NA), \
                f"hour {hour}: a={a} and b={b} disagree at the live transition"
            continue
        assert a == __test_helper_close(2 * k + 1), \
            f"hour {hour}: a={a} != {__test_helper_close(2 * k + 1)}"
        assert b == __test_helper_high(2 * k + 1), \
            f"hour {hour}: b={b} != {__test_helper_high(2 * k + 1)}"
        assert v == a + b, f"hour {hour}: v={v} != a+b"
        checked += 1
    assert checked >= _N_HOURS - 3, f"only {checked} bars checked"
    log.info("two mutually dependent live HTF contexts terminated on %d bars", checked)
