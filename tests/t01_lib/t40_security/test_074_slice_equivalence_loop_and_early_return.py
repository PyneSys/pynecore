"""
@pyne
"""
from pynecore.lib import close, math, nz, plot, request, script, syminfo
from pynecore.types import Persistent, Series


@script.indicator(title="Slice Loop", shorttitle="SLL")
def main():
    state: Persistent[float] = 0.0
    looped: Series[float] = 0.0
    # The ``request.security()`` call sits in a LOOP BODY, so its write block
    # runs several times per bar. The expression is loop-invariant, so the first
    # write publishes and the repeats are no-ops — but the surrounding loop has
    # to stay whole in any slice, or the write count changes.
    for _i in range(3):
        looped = request.security(syminfo.tickerid, "60", state * 0.001 + close)
    # Tail: a ``math.sum`` accumulator feeding the ``var`` the expression reads
    # on the NEXT bar. It stands behind the loop and still decides its value.
    running = math.sum(close, 5)
    state = nz(state) + nz(running) * 0.001
    plot(looped, "looped")
    plot(state, "state")
    plot(running, "running")


# The early-return side script: ``main()`` leaves before its tail on two bars
# out of three, so the tail's writes land on an irregular subset of the bars.
__test_helper_early_return_script = '''"""
@pyne
"""
from pynecore.lib import bar_index, close, plot, request, script, syminfo
from pynecore.types import Persistent, Series


@script.indicator(title="Slice Early Return", shorttitle="SER")
def main():
    acc: Persistent[float] = 1.0
    x: Series[float] = request.security(syminfo.tickerid, "60", acc * 0.01 + close)
    if bar_index % 3 == 2:
        return
    acc = acc + close * 0.001
    plot(x, "x")
    plot(acc, "acc")
'''

# Every timestamp here is Unix MILLISECONDS.
__test_helper_t0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, on the 1h and 5m grids
__test_helper_hour = 3_600_000
__test_helper_step = 300_000  # 5 minutes
__test_helper_hours = 26
__test_helper_bars = __test_helper_hours * 12


def __test_helper_write_feed(tmp_dir):
    """Write the ``60`` feed the higher-timeframe context reads.

    :param tmp_dir: Directory to write into.
    :return: Path to the written ``.ohlcv`` file, as a string.
    """
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "SLL60.ohlcv"
    with OHLCVWriter(path, "60") as w:
        for hour in range(__test_helper_hours):
            c = 100.0 + (hour % 9) * 1.75
            w.write(OHLCV(timestamp=__test_helper_t0 + hour * __test_helper_hour,
                          open=c, high=c + 1.0, low=c - 1.0, close=c, volume=1.0))
    SymInfo(
        prefix="PYTEST", description="Slice Loop", ticker="TEST",
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
    """The chart's own 5-minute bars, with a moving close.

    :return: The bar list.
    """
    from pynecore.types.ohlcv import OHLCV
    bars = []
    for i in range(__test_helper_bars):
        c = 80.0 + (i % 31) * 0.5
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


def __test_helper_set_slice_mode(no_slice):
    """Arm ``PYNE_NO_SECURITY_SLICE`` and drop everything cached for the old mode.

    The switch is read while the module is TRANSFORMED, so flipping it only
    takes effect on a fresh import: the script module goes out of
    ``sys.modules``, its cached bytecode is dropped, and the pipeline digest —
    memoized per process — is recomputed so a ``.pyc`` produced in the other
    mode can never be accepted.

    :param no_slice: Whether slicing must be disabled for the next import.
    """
    import importlib.util
    import os
    import sys
    from pathlib import Path

    import pynecore.core.import_hook as import_hook

    sys.modules.pop(Path(__file__).stem, None)
    sys.modules.pop("pynecore.transformers.security_slice", None)
    import_hook._transform_pipeline_hash = None
    try:
        Path(importlib.util.cache_from_source(__file__)).unlink()
    except OSError:
        pass
    os.environ['PYNE_SAVE_SCRIPT_TOML'] = '0'
    if no_slice:
        os.environ['PYNE_NO_SECURITY_SLICE'] = '1'
    else:
        os.environ.pop('PYNE_NO_SECURITY_SLICE', None)


def __test_helper_run(runner, no_slice):
    """Run the loop script once in the given slicing mode.

    :param runner: The ``runner`` fixture.
    :param no_slice: Whether to force the whole-``main()`` child path.
    :return: The plot values per chart bar.
    """
    import os
    import tempfile
    from pathlib import Path

    __test_helper_set_slice_mode(no_slice)
    rows = []
    try:
        with tempfile.TemporaryDirectory() as td:
            feed = __test_helper_write_feed(Path(td))
            r = runner(__test_helper_chart_bars(), security_data={"60": feed})
            for _candle, pv in r.run_iter():
                rows.append(dict(pv))
    finally:
        os.environ.pop('PYNE_NO_SECURITY_SLICE', None)
    return rows


def __test_helper_run_early_return(runner, no_slice):
    """Run the early-return side script once in the given slicing mode.

    :param runner: The ``runner`` fixture, used for its chart symbol info.
    :param no_slice: Whether to force the whole-``main()`` child path.
    :return: The plot values per chart bar.
    """
    import os
    import sys
    import tempfile
    from pathlib import Path

    from pynecore.core.script_runner import ScriptRunner

    __test_helper_set_slice_mode(no_slice)
    rows = []
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            feed = __test_helper_write_feed(tmp)
            path = tmp / "slice_early_return_reference.py"
            path.write_text(__test_helper_early_return_script)
            chart = runner(__test_helper_chart_bars())
            sys.modules.pop(path.stem, None)
            sys.path.insert(0, str(tmp))
            try:
                sr = ScriptRunner(path, iter(__test_helper_chart_bars()),
                                  chart.syminfo, security_data={"60": feed})
                for _candle, pv in sr.run_iter():
                    rows.append(dict(pv))
            finally:
                sys.path.remove(str(tmp))
                sys.modules.pop(path.stem, None)
    finally:
        os.environ.pop('PYNE_NO_SECURITY_SLICE', None)
    return rows


def __test_helper_run_with_timeout(fn, seconds=180):
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


def __test_helper_compare(sliced, plain, log, what):
    """Assert two runs produced bit-identical plot rows.

    :param sliced: Rows from the sliced run.
    :param plain: Rows from the ``PYNE_NO_SECURITY_SLICE=1`` run.
    :param log: The ``log`` fixture.
    :param what: Name for the log line.
    """
    assert len(sliced) == len(plain), \
        f"{what}: bar counts differ ({len(sliced)} vs {len(plain)})"
    assert sliced, f"{what}: the run produced no bars"
    for i in range(len(sliced)):
        got = sliced[i]
        want = plain[i]
        assert got.keys() == want.keys(), f"{what} bar {i}: plot columns differ"
        for key in want:
            assert __test_helper_same(got[key], want[key]), (
                f"{what} bar {i} '{key}': sliced={got[key]!r} "
                f"whole-main={want[key]!r}")
    log.info("%s: %d bars x %d plots identical in both slicing modes",
             what, len(sliced), len(sliced[0]))


def __test_slice_keeps_the_loop_and_the_accumulator__(runner, log):
    """A write in a loop body and a ``math.sum`` tail survive slicing

    The context call stands inside a ``for`` body, so control dependence keeps
    the whole loop; behind it a ``math.sum`` accumulator feeds the ``var`` the
    expression reads on the next bar. Slicing either one away — the loop, or the
    accumulator statement — changes the child's value, so the plot rows must
    match value for value in both modes.
    """
    sliced = __test_helper_run_with_timeout(lambda: __test_helper_run(runner, False))
    plain = __test_helper_run_with_timeout(lambda: __test_helper_run(runner, True))
    __test_helper_compare(sliced, plain, log, "in-loop write + math.sum tail")


def __test_slice_keeps_an_early_returning_tail__(runner, log):
    """A tail behind an early ``return`` survives slicing

    ``main()`` leaves on two bars out of three, so the accumulator the
    expression reads is advanced on an irregular subset of the bars. The slice
    must keep both the guard and the statement behind it, otherwise the child's
    accumulator runs at a different rate than the chart's.
    """
    sliced = __test_helper_run_with_timeout(
        lambda: __test_helper_run_early_return(runner, False))
    plain = __test_helper_run_with_timeout(
        lambda: __test_helper_run_early_return(runner, True))
    __test_helper_compare(sliced, plain, log, "early return")


def __test_loop_and_early_return_scenarios_are_not_degenerate__(runner, log):
    """Both scripts actually carry moving values

    The loop script's context column has to move and its accumulator to grow;
    the early-return script has to publish on the bars it reaches and to leave
    the other bars unplotted — that asymmetry is the whole point of the case.
    """
    loop_rows = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, False))
    early_rows = __test_helper_run_with_timeout(
        lambda: __test_helper_run_early_return(runner, False))

    looped = []
    for row in loop_rows:
        v = row.get("looped")
        if not __test_helper_is_na(v):
            looped.append(v)
    assert len(looped) >= len(loop_rows) // 2, \
        f"only {len(looped)} of {len(loop_rows)} bars carry a value"
    assert len(set(looped)) > 20, f"degenerate context column: {len(set(looped))}"
    assert loop_rows[-1]["state"] > loop_rows[0]["state"], \
        "the math.sum accumulator did not grow"

    plotted = 0
    for row in early_rows:
        if not __test_helper_is_na(row.get("acc")):
            plotted += 1
    assert plotted > 0, "the early-return script never reached its tail"
    assert plotted < len(early_rows), "the early return never fired"

    log.info("loop column carries %d distinct values; the early-return script "
             "plotted %d of %d bars", len(set(looped)), plotted, len(early_rows))
