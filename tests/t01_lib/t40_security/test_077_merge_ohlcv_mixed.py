"""
@pyne
"""
from pynecore.lib import close, high, plot, request, script, syminfo, ta, volume
from pynecore.types import Series


@script.indicator(title="Merge OHLCV Mixed", shorttitle="MOM")
def main():
    # (a) A MIXED group on the ``60`` context: one plain ``close`` passthrough
    # plus two computed expressions. The passthrough alone would be served by
    # the child's OHLCV fast path; sharing a child with sliced members means the
    # child has to run ``main()`` for the group and still answer the passthrough
    # exactly as it did alone.
    plain: Series[float] = request.security(syminfo.tickerid, "60", close)
    rsi: Series[float] = request.security(syminfo.tickerid, "60", ta.rsi(close, 5))
    ratio: Series[float] = request.security(
        syminfo.tickerid, "60", close * 2.0 - ta.sma(close, 3))
    # (b) An ALL-OHLCV group on a different context (``120``), so it forms its
    # own group: every member is a bare field passthrough, which is what keeps
    # the child on the ``main()``-free fast path even when it serves three sids.
    c2: Series[float] = request.security(syminfo.tickerid, "120", close)
    h2: Series[float] = request.security(syminfo.tickerid, "120", high)
    v2: Series[float] = request.security(syminfo.tickerid, "120", volume)
    plot(plain, "plain")
    plot(rsi, "rsi")
    plot(ratio, "ratio")
    plot(c2, "c2")
    plot(h2, "h2")
    plot(v2, "v2")


def __test_helper_run(runner, no_merge):
    """Run the script once, recording which sids each child process serves.

    ``PYNE_NO_SECURITY_MERGE`` is a RUNTIME switch — it changes which processes
    the runner spawns, never the emitted tree — so the two runs share one
    bytecode. The first entry of a spawn's ``args`` is the served sid list.

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
            tmp = Path(td)
            security_data = {
                "60": __test_helper_write_feed(tmp, "60", __test_helper_hour, "MOM"),
                "120": __test_helper_write_feed(tmp, "120", 2 * __test_helper_hour,
                                                "MOM"),
            }
            r = runner(__test_helper_chart_bars(), security_data=security_data)
            for _candle, pv in r.run_iter():
                rows.append(dict(pv))
    finally:
        multiprocessing.Process = original
        if no_merge:
            os.environ.pop('PYNE_NO_SECURITY_MERGE', None)
    return rows, spawned


def __test_mixed_and_all_ohlcv_groups_each_share_one_child__(runner, log):
    """A mixed group and an all-OHLCV group each collapse to a single child.

    Two contexts, three calls each. The ``60`` group mixes a bare ``close``
    passthrough with two sliced expressions, the ``120`` group is nothing but
    field passthroughs. Merged, that is two children; with
    ``PYNE_NO_SECURITY_MERGE=1`` it is six, and every plotted value has to come
    out the same either way — including the passthroughs, whose answer must not
    change just because they now share a child with a sliced member.
    """
    merged_rows, merged = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, no_merge=False))
    plain_rows, plain = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, no_merge=True))

    assert len(merged) == 2, f"the two groups did not share two processes: {merged}"
    for served in merged:
        assert len(served) == 3, f"served contexts: {served}"
        assert len(set(served)) == 3, f"duplicate member: {served}"
    assert len(plain) == 6, f"unmerged children: {plain}"
    assert all(len(served) == 1 for served in plain), \
        f"unmerged child serves several contexts: {plain}"

    __test_helper_compare(merged_rows, plain_rows, log, "ohlcv-mixed")


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
