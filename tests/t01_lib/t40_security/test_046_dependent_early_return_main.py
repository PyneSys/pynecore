"""
@pyne
"""
from pynecore.lib import bar_index, close, plot, request, script, syminfo, timeframe
from pynecore.types import Series


@script.indicator(title="Early Return Main", shorttitle="ERM")
def main():
    # An early ``return`` skips every write block and ``__sec_wait__`` after it.
    # The chart-context producer ``own`` therefore publishes nothing on those
    # bars, yet its consumer must not stall: the runner's bar-end hook raises
    # the producer's frontier whatever ``main()`` did.
    if bar_index % 3 == 2:
        return
    own: Series[float] = request.security(syminfo.tickerid, timeframe.period, close)
    dep: Series[float] = request.security("EXCH:PEER", timeframe.period, own + 1000.0)
    plot(own, "own")
    plot(dep, "dep")


# Every timestamp here is Unix MILLISECONDS.
_T0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, aligned to the 5m grid
_STEP = 300_000  # 5 minutes
_N = 45


def __test_helper_write_peer(tmp_dir):
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "PEER.ohlcv"
    with OHLCVWriter(path, "5") as w:
        for i in range(_N):
            c = 300.0 + i
            w.write(OHLCV(timestamp=_T0 + i * _STEP, open=c, high=c, low=c,
                          close=c, volume=1.0))
    SymInfo(
        prefix="EXCH", description="Peer", ticker="PEER",
        currency="USD", period="5", type="crypto",
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
    return [OHLCV(timestamp=_T0 + i * _STEP, open=100.0 + i, high=100.0 + i,
                  low=100.0 + i, close=100.0 + i, volume=1.0)
            for i in range(_N)]


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


def __test_early_return_does_not_strand_a_consumer__(runner, log):
    """An early ``return`` in ``main()`` leaves no producer frontier behind.

    On every third bar ``main()`` returns before any write block runs, so the
    chart-context producer appends nothing and emits no ``__sec_wait__``. Only
    the runner's bar-end hook — which runs whatever the script did — can raise
    the frontier there. Without it a consumer waiting on that context would wait
    forever, and the run would never reach the last bar.
    """
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.types.na import NA

    sys.modules.pop(Path(__file__).stem, None)

    def scenario():
        with tempfile.TemporaryDirectory() as td:
            peer = __test_helper_write_peer(Path(td))
            r = runner(__test_helper_chart_bars(),
                       security_data={"EXCH:PEER": peer})
            return [(c.timestamp, pv.get("own"), pv.get("dep"))
                    for c, pv in r.run_iter()]

    rows = __test_helper_run_with_timeout(scenario)

    assert len(rows) == _N, f"bar count {len(rows)} != {_N} — the run did not complete"
    checked = 0
    for i, (ts, own, dep) in enumerate(rows):
        assert ts == _T0 + i * _STEP
        if i % 3 == 2:
            continue  # the bar ``main()`` returned early on: no plot values
        assert own == 100.0 + i, f"bar {i}: own={own}"
        # The peer context reads the chart-context producer's value for its own
        # bar, so ``dep`` is the chart close plus 1000.
        assert not isinstance(dep, NA) and dep == 100.0 + i + 1000.0, \
            f"bar {i}: dep={dep} != {100.0 + i + 1000.0}"
        checked += 1
    assert checked > 20, f"only {checked} bars checked"
    log.info("early return in main() left no consumer stranded (%d bars checked)", checked)
