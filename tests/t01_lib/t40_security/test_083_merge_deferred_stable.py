"""
@pyne
"""
from pynecore.lib import close, high, input, na, plot, request, script, syminfo, ta
from pynecore.types import Series


def pick(tf, mode, length):
    """One higher-timeframe value, picked between two moving averages.

    The gcBksf shape: both ``request.security()`` calls sit in a ternary, so
    neither is reached on every run and both signal INLINE. Their symbol and
    timeframe nevertheless come from ``main``'s inputs through this function's
    never-rebound parameters, whichever call site is running, so both read the
    same feed and may share a child with everything else on it.
    """
    return (request.security(syminfo.tickerid, tf, ta.sma(close, length))
            if mode == 'SMA'
            else request.security(syminfo.tickerid, tf, ta.ema(close, length)))


@script.indicator(title="Merge Deferred Stable", shorttitle="MDS")
def main(res=input.string(defval="60", title="HTF"),
         mode=input.string(defval='SMA', title='Type', options=('SMA', 'EMA')),
         htf=input.bool(defval=True, title="Use HTF")):
    # (a) The helper, called three times: SIX contexts (the instantiation pass
    # gives every call site its own copy), all on the same feed.
    fast: Series[float] = pick(res, mode, 3)
    mid: Series[float] = pick(res, mode, 5)
    slow: Series[float] = pick(res, mode, 8)
    # (b) The LuxAlgo shape: a call that only runs when an input flag is on, so
    # its signal stays inline too — and an input timeframe it cannot change.
    gated: Series[float] = request.security(syminfo.tickerid, res, high) if htf else na
    gated2: Series[float] = request.security(syminfo.tickerid, res, close) if htf else na
    plot(fast, "fast")
    plot(mid, "mid")
    plot(slow, "slow")
    plot(gated, "gated")
    plot(gated2, "gated2")


# Every timestamp here is Unix MILLISECONDS.
__test_helper_t0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, on the 1h and 5m grids
__test_helper_hour = 3_600_000
__test_helper_step = 300_000  # 5 minutes
__test_helper_hours = 24
__test_helper_bars = __test_helper_hours * 12


def __test_helper_write_feed(tmp_dir):
    """Write the ``60`` feed every context of this script reads.

    :param tmp_dir: Directory to write into.
    :return: Path to the written ``.ohlcv`` file, as a string.
    """
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "MDS60.ohlcv"
    with OHLCVWriter(path, "60") as w:
        for hour in range(__test_helper_hours):
            c = 100.0 + (hour % 7) * 3.0
            w.write(OHLCV(timestamp=__test_helper_t0 + hour * __test_helper_hour,
                          open=c, high=c + 2.0, low=c - 2.0, close=c, volume=1.0))
    SymInfo(
        prefix="PYTEST", description="Merge Deferred Stable", ticker="TEST",
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
        c = 50.0 + (i % 23) * 0.25
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


def __test_helper_run_with_timeout(fn, seconds=60):
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


def __test_inline_signalled_contexts_share_one_child__(runner, log):
    """Every call on one stable feed shares a child, however its signal stands.

    None of this script's EIGHT contexts signals in the hoisted top block: six
    are the two arms of a helper's ternary, instantiated once per call site, and
    two sit behind an input flag. All of them take their symbol and timeframe
    from expressions that cannot change during a bar — ``syminfo.tickerid`` and
    ``main``'s own inputs, carried through never-rebound parameters — so the
    transformer keys them by those arguments alone and the chart serves all
    eight from ONE child, resolving the members that have not signalled yet from
    the one that did.

    These are the two shapes the corpus is full of and that the earlier,
    position-scoped key left entirely unmerged (gcBksf: 18 contexts, 0 groups;
    LuxAlgo: 9 contexts, 0 groups).
    """
    merged_rows, merged = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, no_merge=False))
    plain_rows, plain = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, no_merge=True))

    log.info("merged children: %s", merged)
    assert len(merged) == 1, f"merged children: {merged}"
    assert len(merged[0]) == 8, f"served contexts: {merged[0]}"
    assert len(set(merged[0])) == 8, f"duplicate member: {merged[0]}"
    # Unmerged, only the FIVE contexts this run actually reads get a child: the
    # three EMA arms are never taken with ``mode == 'SMA'``. Merged they are
    # carried by the child their group needed anyway, which is what TradingView
    # does with them too — it evaluates every ``request.security()`` call
    # whatever conditional stands around it.
    assert len(plain) == 5, f"unmerged children: {plain}"
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

    for column in ("fast", "mid", "slow", "gated", "gated2"):
        assert any(not __test_helper_is_na(row[column]) for row in merged_rows), \
            f"'{column}' never resolved a value"

    log.info("%d bars x %d plots identical; 1 child for 8 contexts merged, "
             "%d children unmerged", len(merged_rows), len(merged_rows[0]),
             len(plain))


def __test_an_adopted_member_checks_its_own_signal__(log):
    """The adoption guard has no constructible script, so it is not run here.

    The chart resolves a stable group's not-yet-signalled members from the seed,
    which is sound exactly while the compile-time key really does prove they
    name the same feed. When such a member's own signal finally runs, the chart
    compares what it names with what it was resolved as and raises a
    ``RuntimeError`` naming both contexts and both values
    (``__sec_signal__`` in ``core/security.py``), rather than running the script
    against a feed it never asked for.

    A script that reaches that branch cannot be written: producing one would
    mean finding a pair of contexts the stable key calls identical while their
    signals evaluate differently, which is precisely the bug the guard exists to
    surface — and if such a script existed the right fix would be the key rule,
    not the test. Reaching the branch from the outside is no cheaper: the
    adoption bookkeeping lives in the protocol closure, so injecting a mismatch
    would mean rebuilding ``_start``'s whole path (spawn callbacks, states,
    shared memory) around a fake, which tests the fake rather than the guard.
    Left deliberately unexercised, and said so here.
    """
    import pytest
    pytest.skip("no constructible mismatch: see this test's docstring")
