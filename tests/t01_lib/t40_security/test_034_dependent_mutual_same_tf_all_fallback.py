"""
@pyne
"""
from pynecore.lib import close, plot, request, script, timeframe
from pynecore.types import Series


@script.indicator(title="Mutual Same TF All Fallback", shorttitle="MSTF")
def main():
    # Two same-timeframe cross-symbol contexts. The ``try`` block is an
    # unmodelled construct for the dependency analysis, so BOTH contexts fall
    # back to "depends on every other sid" — each one waits for the other.
    # The run must still terminate and produce the time-aligned values.
    a: Series[float] = request.security("EXCH:SYMA", timeframe.period, close)
    b: Series[float] = request.security("EXCH:SYMB", timeframe.period, close)
    try:
        v = a + b
    except ZeroDivisionError:
        v = 0.0
    plot(a, "a")
    plot(b, "b")
    plot(v, "v")


# Every timestamp here is Unix MILLISECONDS.
_T0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, aligned to the 5m grid
_STEP = 300_000  # 5 minutes
_N = 60


def __test_helper_write_feed(tmp_dir, ticker, base):
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / f"{ticker}.ohlcv"
    with OHLCVWriter(path, "5") as w:
        for i in range(_N):
            c = base + i
            w.write(OHLCV(timestamp=_T0 + i * _STEP, open=c, high=c, low=c,
                          close=c, volume=1.0))
    SymInfo(
        prefix="EXCH", description=ticker, ticker=ticker,
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


def __test_mutual_all_sids_fallback_terminates__(runner, log):
    """Two mutually dependent same-timeframe contexts terminate with correct values.

    The ``try`` block makes the dependency analysis fall back to "every sid", so
    each context believes it depends on the other. Deadlock-freedom does not
    rest on the precision of that set: a producer publishes at its write, and a
    consumer can only be ahead of a producer whose write site precedes its own
    read. Over-approximating ``depends`` costs waiting, never liveness.
    """
    import sys
    import tempfile
    from pathlib import Path

    sys.modules.pop(Path(__file__).stem, None)

    def scenario():
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            security_data = {
                "EXCH:SYMA": __test_helper_write_feed(tmp, "SYMA", 200.0),
                "EXCH:SYMB": __test_helper_write_feed(tmp, "SYMB", 300.0),
            }
            r = runner(__test_helper_chart_bars(), security_data=security_data)
            return [(c.timestamp, pv.get("a"), pv.get("b"), pv.get("v"))
                    for c, pv in r.run_iter()]

    rows = __test_helper_run_with_timeout(scenario)

    assert len(rows) == _N, f"bar count {len(rows)} != {_N}"
    for i, (ts, a, b, v) in enumerate(rows):
        assert ts == _T0 + i * _STEP
        assert a == 200.0 + i, f"bar {i}: a={a} != {200.0 + i}"
        assert b == 300.0 + i, f"bar {i}: b={b} != {300.0 + i}"
        assert v == a + b, f"bar {i}: v={v} != a+b"
    log.info("mutual all-sids fallback terminated with time-aligned values")
