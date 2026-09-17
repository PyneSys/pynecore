"""
@pyne
"""
from pynecore.lib import bar_index, close, input, nz, plot, request, script, ta
from pynecore.types import Persistent, Series


@script.indicator(title="Merge Equivalence Basic", shorttitle="MEB")
def main(tf=input.string(defval="60", title="HTF")):
    # Three calls on ONE context: the same symbol (the chart's own, spelled as
    # the empty string) and the same INPUT-derived timeframe, so the security
    # transformer proves they resolve to the same feed and puts them in one
    # group. Their expressions differ, which is the whole point — one child has
    # to serve all three.
    rsi: Series[float] = request.security("", tf, ta.rsi(close, 5))
    # (a) A dependent sibling: inside a merged child this read never goes
    # through the ring, it takes the value the group's own write produced on
    # this very bar.
    smoothed: Series[float] = request.security("", tf, ta.sma(rsi, 3))
    # (b) A persistent accumulator the child carries across its own bars.
    acc: Persistent[float] = 0.0
    acc = acc + 1.0
    accSec: Series[float] = request.security("", tf, close * 0.5 + acc)
    # (c) Conditional CONSUMPTION: the value is only used on every second chart
    # bar, and the other bars keep the previous one. The call itself stays
    # unconditional — a ``request.security()`` sitting in a branch signals
    # inline instead of in the hoisted top block, and an inline signal never
    # joins a group.
    gated: Series[float] = accSec if bar_index % 2 == 0 else nz(gated[1])
    plot(rsi, "rsi")
    plot(smoothed, "smoothed")
    plot(accSec, "accSec")
    plot(gated, "gated")
    plot(acc, "acc")


# Every timestamp here is Unix MILLISECONDS.
__test_helper_t0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, on the 1h and 5m grids
__test_helper_hour = 3_600_000
__test_helper_step = 300_000  # 5 minutes
__test_helper_hours = 24
__test_helper_bars = __test_helper_hours * 12


def __test_helper_write_feed(tmp_dir):
    """Write the ``60`` feed the higher-timeframe contexts read.

    :param tmp_dir: Directory to write into.
    :return: Path to the written ``.ohlcv`` file, as a string.
    """
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "MEB60.ohlcv"
    with OHLCVWriter(path, "60") as w:
        for hour in range(__test_helper_hours):
            c = 100.0 + (hour % 11) * 2.5
            w.write(OHLCV(timestamp=__test_helper_t0 + hour * __test_helper_hour,
                          open=c, high=c + 1.0, low=c - 1.0, close=c, volume=1.0))
    SymInfo(
        prefix="PYTEST", description="Merge Equivalence", ticker="TEST",
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


def __test_helper_run(runner, no_merge):
    """Run the script once, recording which sids each child process serves.

    ``PYNE_NO_SECURITY_MERGE`` is a RUNTIME switch — it changes which processes
    the runner spawns, never the emitted tree — so unlike the slicing switch it
    needs no module or bytecode eviction between the two runs. The script module
    is still dropped from ``sys.modules`` so the two runs start from the same
    state the rest of this directory's tests do.

    ``script_runner`` imports ``Process`` from :mod:`multiprocessing` inside the
    run, so patching the module attribute up front is what the spawn resolves —
    and the first entry of the spawn's ``args`` is the served sid list.

    :param runner: The ``runner`` fixture.
    :param no_merge: Whether to force one child process per context.
    :return: ``(rows, spawned)``, the plot values per chart bar and one tuple of
        served sids per child process.
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

    rows: list[dict] = []
    multiprocessing.Process = _recording_process
    if no_merge:
        os.environ['PYNE_NO_SECURITY_MERGE'] = '1'
    try:
        with tempfile.TemporaryDirectory() as td:
            feed = __test_helper_write_feed(Path(td))
            r = runner(__test_helper_chart_bars(), security_data={"60": feed})
            for _candle, pv in r.run_iter():
                rows.append(dict(pv))
    finally:
        multiprocessing.Process = original
        if no_merge:
            os.environ.pop('PYNE_NO_SECURITY_MERGE', None)
    return rows, spawned


def __test_helper_run_with_timeout(fn, seconds=180):
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


def __test_merged_group_equals_one_child_per_context__(runner, log):
    """Three calls on one context answer identically merged and unmerged.

    The script's three ``request.security()`` calls share a symbol and an
    input-derived timeframe, so they form ONE compile-time group: merged, a
    single child process serves all three; with ``PYNE_NO_SECURITY_MERGE=1``
    each gets its own, which is what the two runs are compared against.

    What the comparison covers is exactly what merging changes: a sibling read
    inside the shared child (``smoothed`` reads ``rsi``), a ``var`` accumulator
    the child carries across bars, and a conditional write that leaves a member
    unpublished on half the chart bars.
    """
    merged_rows, merged = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, no_merge=False))
    plain_rows, plain = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, no_merge=True))

    assert len(merged) == 1, f"the group did not share one process: {merged}"
    assert len(merged[0]) == 3, f"served contexts: {merged[0]}"
    assert len(set(merged[0])) == 3, f"duplicate member: {merged[0]}"
    assert len(plain) == 3, f"unmerged children: {plain}"
    assert all(len(served) == 1 for served in plain), \
        f"unmerged child serves several contexts: {plain}"

    assert len(merged_rows) == len(plain_rows) == __test_helper_bars
    for i in range(len(merged_rows)):
        got = merged_rows[i]
        want = plain_rows[i]
        assert got.keys() == want.keys(), f"bar {i}: plot columns differ"
        for key in want:
            assert __test_helper_same(got[key], want[key]), (
                f"bar {i} '{key}': merged={got[key]!r} unmerged={want[key]!r}")

    log.info("%d bars x %d plots identical: 1 child merged, %d unmerged",
             len(merged_rows), len(merged_rows[0]), len(plain))
