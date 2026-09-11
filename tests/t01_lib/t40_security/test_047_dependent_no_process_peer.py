"""
@pyne
"""
from pynecore.lib import close, na, nz, plot, request, script, timeframe
from pynecore.types import Series


@script.indicator(title="No Process Peer", shorttitle="NPP")
def main():
    # ``ignore_invalid_symbol`` downgrades this context to "no process": nothing
    # ever publishes for it, so its frontier can never move. A consumer must
    # answer with the default straight away instead of waiting for it.
    missing: Series[float] = request.security(
        "EXCH:MISSING", timeframe.period, close, ignore_invalid_symbol=True)
    dep: Series[float] = request.security(
        "EXCH:PEER", timeframe.period, close + nz(missing, 0.0))
    plot(missing, "missing")
    plot(dep, "dep")
    plot(1.0 if na(missing) else 0.0, "missing_is_na")


# Every timestamp here is Unix MILLISECONDS.
_T0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, aligned to the 5m grid
_STEP = 300_000  # 5 minutes
_N = 40


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
    return [OHLCV(timestamp=_T0 + i * _STEP, open=1.0, high=1.0, low=1.0,
                  close=1.0, volume=1.0)
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


def __test_peer_without_a_producer_answers_with_the_default__(runner, log):
    """A peer that no process produces for is answered with the default, without waiting.

    ``ignore_invalid_symbol=True`` leaves the context without a subprocess, so
    its ring frontier stays put forever. The registry marks it "no producer" and
    the consumer's read returns the default immediately — otherwise the whole
    run would hang on the first dependent bar.
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
            return [(c.timestamp, pv.get("missing"), pv.get("dep"),
                     pv.get("missing_is_na")) for c, pv in r.run_iter()]

    rows = __test_helper_run_with_timeout(scenario)

    assert len(rows) == _N, f"bar count {len(rows)} != {_N} — the run did not complete"
    for i, (ts, missing, dep, is_na) in enumerate(rows):
        assert ts == _T0 + i * _STEP
        assert isinstance(missing, NA) or missing != missing, \
            f"bar {i}: the invalid-symbol context returned {missing}, expected na"
        assert is_na == 1.0, f"bar {i}: the chart-side read did not see na"
        # The consumer got ``na`` for the missing peer, so ``nz`` contributes 0.
        assert dep == 300.0 + i, f"bar {i}: dep={dep} != {300.0 + i}"
    log.info("a producer-less peer answers with the default and never blocks")
