"""
@pyne
"""
from pynecore.lib import close, high, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="Live Developing Consumer", shorttitle="LDC")
def main():
    # A dependent 120-minute chain driven over DEVELOPING chart bars. On a
    # developing round the consumer's as-of is the round's fixed tick instant,
    # not its own scheduled close (which lies in the future): it sees the peer's
    # bars closed so far and never waits for a bar the round does not target.
    a: Series[float] = request.security(syminfo.tickerid, "120", close)
    b: Series[float] = request.security(syminfo.tickerid, "120", a + 1000.0)
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
        prefix="EXCH", description="Live two HTF", ticker="LDC",
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


def __test_developing_consumer_does_not_wait_for_its_own_close__(script_path, module_key,
                                                                 syminfo, log):
    """A developing live bar pairs on the round's tick, not on the consumer's own close.

    Taking the scheduled close as the as-of on a developing round would ask for
    a peer bar closing in the future — nothing publishes it in that round and
    the chart would freeze on every tick. The as-of is the round's fixed tick
    instant instead, so each developing tick resolves against the peer bars
    closed so far and the run walks through all of them.
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
            # Each live hour arrives as two DEVELOPING ticks and then closes.
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

    # ``run_iter`` yields once per CLOSED chart bar; the two developing ticks of
    # each live hour are processed in between and are what this test feeds in.
    assert len(rows) == _N_HOURS, \
        f"bar count {len(rows)} != {_N_HOURS} — the run stalled on a developing tick"
    checked = 0
    for ts, a, b, v in rows:
        hour = (ts - _T0) // _HOUR
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
        if isinstance(a, NA):
            # A developing round inside an unclosed 120-minute period can leave
            # the context without a confirmed value; the dependent one must then
            # be ``na`` too, never a stale pairing.
            assert isinstance(b, NA), f"hour {hour}: a is na but b={b}"
            continue
        assert b == a + 1000.0, f"hour {hour}: b={b} != a+1000={a + 1000.0}"
        checked += 1
    assert checked > _N_HIST, f"only {checked} rows carried a value"
    log.info("developing consumer resolved on %d rows without waiting for its own close",
             checked)
