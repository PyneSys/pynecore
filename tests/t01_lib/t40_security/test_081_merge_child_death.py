"""
@pyne
"""
from pynecore.lib import bar_index, close, plot, request, script, syminfo, ta
from pynecore.types import Series


# The higher-timeframe bar the shared child dies on.
_DEATH_BAR = 5


@script.indicator(title="Merge Child Death", shorttitle="MCD")
def main():
    # Three calls on ONE context, so ONE child serves all three. The second
    # member's expression divides by zero on a single higher-timeframe bar and
    # takes that child down with it — together with the two innocent members,
    # which is exactly what the chart has to notice: waiting on any of the three
    # sids must end in an error, not in a freeze.
    alive: Series[float] = request.security(syminfo.tickerid, "60", close)
    doomed: Series[float] = request.security(
        syminfo.tickerid, "60",
        close + (1 // 0 if bar_index == _DEATH_BAR else 0))
    third: Series[float] = request.security(syminfo.tickerid, "60",
                                            ta.sma(close, 3))
    # A second, independent context with a child of its own. It keeps running,
    # so the chart cannot conclude anything from "some child is gone" — it has
    # to unblock the sids the DEAD child served.
    other: Series[float] = request.security(syminfo.tickerid, "120", close)
    plot(alive, "alive")
    plot(doomed, "doomed")
    plot(third, "third")
    plot(other, "other")


def __test_helper_run(runner, no_merge):
    """Run the script to completion or to the error the dead child causes.

    :param runner: The ``runner`` fixture.
    :param no_merge: Whether to force one child process per context.
    :return: ``(raised, spawned)`` — the exception the run ended with (or None)
        and one tuple of served sids per child process.
    """
    import multiprocessing
    import os
    import sys
    import tempfile
    from pathlib import Path

    sys.modules.pop(Path(__file__).stem, None)

    spawned: list = []
    original = multiprocessing.Process

    def _recording_process(*args, **kwargs):
        sec_args = kwargs.get('args') or ()
        if sec_args:
            spawned.append(tuple(sec_args[0]))
        return original(*args, **kwargs)

    raised = None
    multiprocessing.Process = _recording_process
    if no_merge:
        os.environ['PYNE_NO_SECURITY_MERGE'] = '1'
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            security_data = {
                "60": __test_helper_write_feed(tmp, "60", __test_helper_hour, "MCD"),
                "120": __test_helper_write_feed(tmp, "120", 2 * __test_helper_hour,
                                                "MCD"),
            }
            r = runner(__test_helper_chart_bars(), security_data=security_data)
            try:
                for _candle, _pv in r.run_iter():
                    pass
            except BaseException as exc:  # noqa: BLE001 - the failure IS the result
                raised = exc
    finally:
        multiprocessing.Process = original
        if no_merge:
            os.environ.pop('PYNE_NO_SECURITY_MERGE', None)
    return raised, spawned


def __test_a_dead_merged_child_unblocks_every_context_it_served__(runner, log):
    """A merged child dying stops the chart with an error instead of freezing it.

    One child now stands behind THREE contexts, so its death strands three
    waiting sids at once while an unrelated child keeps answering normally. The
    bounded timeout is what separates "raised" from "hung": if the liveness
    check unblocked only the sid whose own process died — or only one member of
    the group — this run would never end.

    The child is killed by an exception rather than a signal on purpose: a
    SIGKILLed process can leave a shared ``multiprocessing`` lock held, which no
    liveness scheme recovers from and which is not what this guards.
    """
    raised, spawned = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, no_merge=False), seconds=90)

    log.info("merged children: %s", spawned)
    assert any(len(served) == 3 for served in spawned), \
        f"the three contexts did not share one child: {spawned}"
    assert raised is not None, "the chart finished normally after a child died"
    assert isinstance(raised, RuntimeError), (
        f"expected a RuntimeError about the dead child, got "
        f"{type(raised).__name__}: {raised}")
    assert "died" in str(raised), f"unexpected error message: {raised}"
    log.info("a dead merged security child surfaced as: %s", raised)


def __test_the_unmerged_run_fails_the_same_way__(runner, log):
    """With merging off the same script fails identically, one child per context.

    The comparison the rest of this family makes on values is made here on the
    failure: merging must not turn a clean error into a hang, nor a hang into
    something else.
    """
    raised, spawned = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, no_merge=True), seconds=90)

    assert len(spawned) == 4, f"unmerged children: {spawned}"
    assert all(len(served) == 1 for served in spawned), \
        f"unmerged child serves several contexts: {spawned}"
    assert isinstance(raised, RuntimeError), (
        f"expected a RuntimeError about the dead child, got "
        f"{type(raised).__name__}: {raised}")
    assert "died" in str(raised), f"unexpected error message: {raised}"
    log.info("unmerged: %s", raised)


# Every timestamp here is Unix MILLISECONDS.
__test_helper_t0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, on the 5m/1h/2h grids
__test_helper_hour = 3_600_000
__test_helper_step = 300_000  # 5 minutes
__test_helper_hours = 24
__test_helper_bars = __test_helper_hours * 12


def __test_helper_write_feed(tmp_dir, timeframe, span_ms, tag):
    """Write one higher-timeframe feed covering the chart's range.

    :param tmp_dir: Directory to write into.
    :param timeframe: The feed's timeframe string.
    :param span_ms: That timeframe's period length in ms.
    :param tag: File name prefix.
    :return: Path to the written ``.ohlcv`` file, as a string.
    """
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / (tag + timeframe + ".ohlcv")
    total = (__test_helper_hours * __test_helper_hour) // span_ms
    with OHLCVWriter(path, timeframe) as w:
        for i in range(total):
            c = 100.0 + (i % 11) * 2.5
            w.write(OHLCV(timestamp=__test_helper_t0 + i * span_ms, open=c,
                          high=c + 1.0, low=c - 1.0, close=c, volume=1.0 + i))
    SymInfo(
        prefix="PYTEST", description="Merge", ticker="TEST",
        currency="USD", period=timeframe, type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_helper_chart_bars():
    """The chart's own 5-minute bars, with a moving close.

    :return: The bar list.
    """
    from pynecore.types.ohlcv import OHLCV
    bars = []
    for i in range(__test_helper_bars):
        c = 50.0 + (i % 19) * 0.25
        bars.append(OHLCV(timestamp=__test_helper_t0 + i * __test_helper_step,
                          open=c - 0.1, high=c + 0.5, low=c - 0.5, close=c,
                          volume=1.0 + i))
    return bars


def __test_helper_is_na(value):
    """Whether a read answered Pine ``na``, in either of its two shapes.

    :param value: The read value.
    :return: Whether it is ``na``.
    """
    from pynecore.types.na import NA
    return value is None or isinstance(value, NA) or (
        isinstance(value, float) and value != value)


def __test_helper_same(a, b):
    """Pine-style equality for two read values: ``na`` matches ``na``.

    :param a: One value.
    :param b: The other.
    :return: Whether the two reads answered the same.
    """
    if __test_helper_is_na(a) or __test_helper_is_na(b):
        return __test_helper_is_na(a) and __test_helper_is_na(b)
    return a == b


def __test_helper_run_with_timeout(fn, seconds=240):
    """Run ``fn`` on a daemon thread; a deadlock fails the test instead of hanging.

    :param fn: The callable to run.
    :param seconds: How long to wait for it.
    :return: Whatever ``fn`` returned.
    """
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


def __test_helper_compare(merged_rows, plain_rows, log, label):
    """Assert two runs produced the very same plot rows.

    :param merged_rows: Rows of the merged run.
    :param plain_rows: Rows of the ``PYNE_NO_SECURITY_MERGE=1`` run.
    :param log: The ``log`` fixture.
    :param label: Scenario name for the log line.
    """
    assert len(merged_rows) == len(plain_rows) == __test_helper_bars, (
        f"{label}: bar counts differ: {len(merged_rows)} vs {len(plain_rows)}")
    for i in range(len(merged_rows)):
        got = merged_rows[i]
        want = plain_rows[i]
        assert got.keys() == want.keys(), f"{label} bar {i}: plot columns differ"
        for key in want:
            assert __test_helper_same(got[key], want[key]), (
                f"{label} bar {i} '{key}': merged={got[key]!r} "
                f"unmerged={want[key]!r}")
    log.info("%s: %d bars x %d plots identical", label, len(merged_rows),
             len(merged_rows[0]))
