"""
@pyne
"""
from pynecore.lib import bar_index, close, request, script, syminfo, timeframe
from pynecore.types import Series


@script.indicator(title="Early Return Offset Peer", shorttitle="EROP")
def main():
    # The chart skips its write on every third bar. The peer's feed starts one
    # bar before the chart, so its own early returns fall on DIFFERENT bars:
    # on a bar the chart skips, the peer still reads ``own`` at that bar's close.
    if bar_index % 3 == 2:
        return
    own: Series[float] = request.security(syminfo.tickerid, timeframe.period, close)
    dep: Series[float] = request.security("EXCH:PEER", timeframe.period, own + 1000.0)
    return {"own": own, "dep": dep}


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
        # One bar ahead of the chart: the peer's bar_index runs one above the
        # chart's for the same instant.
        for i in range(-1, _N):
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


def __test_skipped_chart_write_releases_a_consumer_reading_that_bar__(runner, log):
    """A confirmed chart bar without a write still releases its own close.

    On a bar where ``main()`` returns early the chart-context producer appends
    nothing, while the peer — whose early returns fall on other bars — reads
    ``own`` as of that bar's close. The bar-end hook must publish a frontier
    reaching that close; one ms below it leaves the peer waiting on the chart
    while the next chart bar waits on the peer.
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
            assert own is None and dep is None, f"bar {i}: early return published output"
            continue
        assert own == 100.0 + i, f"bar {i}: own={own}"
        if i % 3 == 0:
            # The peer's bar for this instant evaluates its expression, reading
            # the chart-context producer's value for the same bar.
            assert not isinstance(dep, NA) and dep == 100.0 + i + 1000.0, \
                f"bar {i}: dep={dep} != {100.0 + i + 1000.0}"
            checked += 1
    assert checked > 10, f"only {checked} bars checked"
    log.info("skipped chart write released the peer (%d bars checked)", checked)
