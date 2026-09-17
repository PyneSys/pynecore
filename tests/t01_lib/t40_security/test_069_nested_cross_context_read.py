"""
@pyne
"""
from pynecore.lib import close, plot, request, script
from pynecore.types import Series


@script.indicator(title="Nested Cross Context", shorttitle="NCC")
def main():
    # The inner context stands inside the outer one's expression, so only the
    # outer context's child reads it. That read has to pair the inner value,
    # not answer it with ``na``.
    dep: Series[float] = request.security(
        "EXCH:OUT", "60", request.security("EXCH:INN", "60", close))
    plot(dep, "dep")


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


def __test_nested_read_pairs_the_inner_value__(runner, log):
    """A nested cross-context read in the outer child returns the inner value

    The outer context's child replays the whole script and reads the inner
    context over the peer protocol. The inner write block has to come first,
    so the inner context is a dependency the child waits for and pairs; a read
    answered with the default would turn the whole plot into ``na``.
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
            outer = __test_helper_write_feed(Path(td), "OUT", 100.0)
            inner = __test_helper_write_feed(Path(td), "INN", 7.0)
            r = runner(__test_helper_chart_bars(),
                       security_data={"EXCH:OUT:60": outer, "EXCH:INN:60": inner})
            return [pv.get("dep") for _candle, pv in r.run_iter()]

    values = __test_helper_run_with_timeout(scenario, seconds=120)

    inner_closes = {7.0 + hour for hour in range(_N_HOURS)}
    paired = [v for v in values
              if v is not None and not isinstance(v, NA) and v == v]
    assert len(paired) >= len(values) // 2, \
        f"only {len(paired)} of {len(values)} bars carry the nested value"
    assert set(paired) <= inner_closes, f"unexpected values: {sorted(set(paired))}"
    assert len(set(paired)) > 5, f"degenerate pairing: {sorted(set(paired))}"

    log.info("the outer child pairs the nested inner value on %d bars", len(paired))
