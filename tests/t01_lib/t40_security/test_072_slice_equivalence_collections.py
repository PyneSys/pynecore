"""
@pyne
"""
from dataclasses import dataclass

from pynecore.lib import array, close, plot, request, script, syminfo
from pynecore.types import Persistent, Series


@dataclass(slots=True)
class Slot:
    """A one-field UDT the security expression reads and the tail writes."""
    acc: float = 0.0


def bump_second(arr, v):
    """Mutate an array received as a PARAMETER.

    The slice has no way to see this write at the call site, so the call itself
    has to be pulled in as a mutation of ``a``.

    :param arr: The array, aliased through the parameter.
    :param v: The chart close driving the bump.
    """
    array.set(arr, 1, array.get(arr, 1) + v * 0.0001)


@script.indicator(title="Slice Collections", shorttitle="SLC")
def main():
    a: Persistent[list] = array.new_float(3, 1.0)
    u: Persistent[Slot] = Slot(0.0)
    # The expression reads the CONTAINER and the UDT field. Everything that can
    # reach either one — through the name, an alias, or a parameter — is an
    # input of the next bar's expression.
    htf: Series[float] = request.security(
        syminfo.tickerid, "60",
        array.get(a, 0) + array.sum(a) * 0.01 + u.acc * 0.001)
    # --- tail: mutations only, all of them behind the expression ---
    array.push(a, close * 0.01)          # direct: grows the array
    array.shift(a)                       # direct: drops the head
    b = a                                # alias: same object under a new name
    array.set(b, 0, array.get(b, 0) + 0.25)
    bump_second(a, close)                # mutation inside a user function
    u.acc = u.acc + close * 0.001        # UDT field write
    plot(htf, "htf")
    plot(array.get(a, 0), "a0")
    plot(array.sum(a), "asum")
    plot(u.acc, "acc")


# Every timestamp here is Unix MILLISECONDS.
__test_helper_t0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, on the 1h and 5m grids
__test_helper_hour = 3_600_000
__test_helper_step = 300_000  # 5 minutes
__test_helper_hours = 30
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

    path = tmp_dir / "SLC60.ohlcv"
    with OHLCVWriter(path, "60") as w:
        for hour in range(__test_helper_hours):
            c = 100.0 + (hour % 13) * 2.0
            w.write(OHLCV(timestamp=__test_helper_t0 + hour * __test_helper_hour,
                          open=c, high=c + 1.0, low=c - 1.0, close=c, volume=1.0))
    SymInfo(
        prefix="PYTEST", description="Slice Collections", ticker="TEST",
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
        c = 60.0 + (i % 29) * 0.75
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
    """Run the script once in the given slicing mode.

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


def __test_slice_keeps_container_mutations_behind_the_expression__(runner, log):
    """Array and UDT mutations after the expression survive slicing

    The expression reads an array and a UDT field; behind it the array is
    mutated directly, through an alias bound to the same object, and inside a
    user function that received it as a parameter, and the UDT field is
    reassigned. None of those writes is visible at the expression's own site,
    yet every one of them decides what the NEXT bar's expression reads — so a
    slice that stops at the expression diverges within a bar.
    """
    sliced = __test_helper_run_with_timeout(lambda: __test_helper_run(runner, False))
    plain = __test_helper_run_with_timeout(lambda: __test_helper_run(runner, True))
    __test_helper_compare(sliced, plain, log, "array + alias + parameter + UDT")


def __test_collection_scenario_is_not_degenerate__(runner, log):
    """Every mutation path actually moves the compared values

    Each of the four writes lands in a different place, and the run has to prove
    they are live: the array head moves (direct push/shift plus the alias
    write), the sum moves (the function's element bump is in it) and the UDT
    accumulator grows. Without that, the equivalence assertion above would hold
    over constants.
    """
    rows = __test_helper_run_with_timeout(lambda: __test_helper_run(runner, False))

    htf = []
    for row in rows:
        v = row.get("htf")
        if not __test_helper_is_na(v):
            htf.append(v)
    assert len(htf) >= len(rows) // 2, \
        f"only {len(htf)} of {len(rows)} bars carry a value"
    assert len(set(htf)) > 20, f"degenerate context column: {len(set(htf))} values"
    assert rows[-1]["a0"] != rows[0]["a0"], "the array head never moved"
    assert rows[-1]["asum"] != rows[0]["asum"], "the array sum never moved"
    assert rows[-1]["acc"] > rows[0]["acc"], "the UDT field never grew"

    log.info("scenario carries %d distinct context values over %d bars",
             len(set(htf)), len(rows))
