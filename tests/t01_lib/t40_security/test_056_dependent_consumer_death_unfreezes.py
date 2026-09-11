"""
@pyne
"""
from pynecore.lib import bar_index, close, plot, request, script, timeframe
from pynecore.types import Series


@script.indicator(title="Consumer Death Unfreezes", shorttitle="CDU")
def main():
    # Two same-timeframe cross-symbol contexts that depend on each other (the
    # ``try`` block makes the dependency analysis fall back to "every sid").
    # The test kills one child mid-run: the survivor then blocks on the dead
    # one's ring, so the chart can only notice through the OTHER child's
    # liveness — which is why the liveness check watches every child, not just
    # the one it is waiting on.
    a: Series[float] = request.security("EXCH:SYMA", timeframe.period, close)
    # The SYMB child dies partway through: the expression divides by zero on one
    # bar, and only that child evaluates it.
    b: Series[float] = request.security(
        "EXCH:SYMB", timeframe.period,
        close + (1 // 0 if bar_index == _DEATH_BAR else 0))
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
_N = 200
# The bar the SYMB child dies on.
_DEATH_BAR = 20


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


def __test_a_dying_consumer_child_stops_the_chart_with_an_error__(runner, log):
    """A security child dying mid-run stops the chart with an error instead of freezing it.

    The surviving child is blocked on the dead one's ring, so the chart may well
    be waiting on a child that is alive but will never answer. Watching only the
    child it waits on would freeze the run forever; watching every child's
    liveness turns the death into an error. The bounded timeout here is what
    separates "raised" from "hung".

    The child is killed by an exception rather than a signal on purpose: a
    SIGKILLed process can leave a shared ``multiprocessing`` lock held, which no
    liveness scheme can recover from and which is not what this guards.
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
            bars = 0
            for _candle, _pv in r.run_iter():
                bars += 1
            return bars

    raised = None
    try:
        __test_helper_run_with_timeout(scenario, seconds=60)
    except AssertionError:
        raise
    except BaseException as exc:  # noqa: BLE001 - the failure mode is the assertion
        raised = exc

    assert raised is not None, "the chart finished normally after a child died"
    assert isinstance(raised, RuntimeError), \
        f"expected a RuntimeError about the dead child, got {type(raised).__name__}: {raised}"
    assert "died" in str(raised), f"unexpected error message: {raised}"
    log.info("a dead security child surfaced as: %s", raised)
