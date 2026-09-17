"""
@pyne
"""
from pynecore.lib import close, input, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="Independent Child Missing Feed", shorttitle="ICMF")
def main(htf=input.timeframe(defval="60", title="HTF")):
    # The chart's closes never exceed 50, so the chart never reads the
    # unprovisioned runtime-resolved context; the ternary still resolves it on
    # every bar. The independent context's child replays the same script on its
    # own closes (all above 50) and does take the branch — but its own
    # expression never reads that context.
    m: Series[float] = request.security(syminfo.tickerid, htf, close) if close > 50.0 else close
    # A computed expression: a bare OHLCV field is shipped without replaying
    # the script, so it would never reach the branch at all.
    other: Series[float] = request.security("EXCH:RRP", "60", close * 2.0)
    plot(m, "m")
    plot(other, "other")


# Every timestamp here is Unix MILLISECONDS.
_T0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, aligned to the 1h and 5m grids
_HOUR = 3_600_000
_CHART_STEP = 300_000  # 5 minutes
_N_HOURS = 12


def __test_helper_write_feed(tmp_dir, ticker, base):
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / f"{ticker}60.ohlcv"
    with OHLCVWriter(path, "60") as w:
        for hour in range(_N_HOURS):
            c = base + hour
            w.write(OHLCV(timestamp=_T0 + hour * _HOUR, open=c, high=c, low=c,
                          close=c, volume=1.0))
    SymInfo(
        prefix="EXCH", description="Consumer feed", ticker=ticker,
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


def __test_independent_child_ignores_unread_missing_feed__(runner, log):
    """A missing feed outside a child's own expression does not fail the child

    The independent context's child takes a branch the chart never does and
    reaches the unprovisioned context's read there. Nothing it writes draws on
    that read, so the run must complete with the provisioned values instead of
    failing on a feed nobody needs.
    """
    import os
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.types.na import NA

    sys.modules.pop(Path(__file__).stem, None)

    def scenario():
        os.environ['PYNE_SAVE_SCRIPT_TOML'] = '0'
        with tempfile.TemporaryDirectory() as td:
            feed = __test_helper_write_feed(Path(td), "RRP", 100.0)
            r = runner(__test_helper_chart_bars(),
                       security_data={"EXCH:RRP:60": feed})
            return [(pv.get("m"), pv.get("other")) for _candle, pv in r.run_iter()]

    rows = __test_helper_run_with_timeout(scenario, seconds=120)

    assert len(rows) == _N_HOURS * 12
    assert all(m == 1.0 for m, _other in rows), "the chart took the unprovisioned branch"
    others = [o for _m, o in rows if o is not None and not isinstance(o, NA) and o == o]
    assert len(others) >= len(rows) // 2, f"only {len(others)} bars carry the peer value"
    assert set(others) <= {2.0 * (100.0 + hour) for hour in range(_N_HOURS)}

    log.info("the independent child completed on %d bars", len(rows))
