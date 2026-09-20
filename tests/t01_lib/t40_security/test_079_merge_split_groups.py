"""
@pyne
"""
from pynecore.lib import (
    barmerge, close, input, plot, request, script, syminfo
)
from pynecore.types import Series


def _htf15(mult):
    """One higher-timeframe read, made twice from ``main()``.

    Both call sites live in THIS function's top block with the same literal
    timeframe, so their signal arguments are syntactically identical at the same
    program point and the two contexts group. ``mult`` is not part of what
    decides the feed, so it does not split them.

    :param mult: Factor applied to the higher-timeframe close.
    :return: The context's value.
    """
    v: Series[float] = request.security(syminfo.tickerid, "15", close * mult)
    return v


@script.indicator(title="Merge Split Groups", shorttitle="MSG")
def main(tfA=input.string(defval="240", title="TF A"),
         tfB=input.string(defval="240", title="TF B"),
         useCond=input.bool(defval=True, title="Read the conditional context")):
    # (a) Same symbol, same timeframe, DIFFERENT lookahead: two feeds read two
    # different ways, so the child's round protocol differs — they may not share
    # one. Two contexts, two children, merged or not.
    laOff: Series[float] = request.security(syminfo.tickerid, "120", close)
    laOn: Series[float] = request.security(syminfo.tickerid, "120", close,
                                           lookahead=barmerge.lookahead_on)
    # (b) Same feed, different ``gaps``: ``gaps`` only shapes how the CHART side
    # reads the published result, not what the child loads, so these do share a
    # child.
    gapOff: Series[float] = request.security(syminfo.tickerid, "60", close)
    gapOn: Series[float] = request.security(syminfo.tickerid, "60", close * 1.1,
                                            gaps=barmerge.gaps_on)
    # (c) Two INPUT-derived timeframes that happen to hold the same string. The
    # group key is syntactic, and ``tfA`` is not ``tfB`` as syntax, so these
    # never merge — but they resolve to one feed, so their values match.
    inA: Series[float] = request.security(syminfo.tickerid, tfA, close)
    inB: Series[float] = request.security(syminfo.tickerid, tfB, close)
    # (d) A call behind an input-controlled branch. It signals INLINE, not in
    # the top block, so it is never a group member and always runs alone.
    cond: Series[float] = 0.0
    if useCond:
        cond = request.security(syminfo.tickerid, "30", close * 0.5)
    # (e) One helper, called twice: a group formed across call sites.
    help1: Series[float] = _htf15(1.0)
    help2: Series[float] = _htf15(2.0)
    plot(laOff, "laOff")
    plot(laOn, "laOn")
    plot(gapOff, "gapOff")
    plot(gapOn, "gapOn")
    plot(inA, "inA")
    plot(inB, "inB")
    plot(cond, "cond")
    plot(help1, "help1")
    plot(help2, "help2")


def __test_helper_run(runner, no_merge):
    """Run the script once, recording which sids each child process serves.

    :param runner: The ``runner`` fixture.
    :param no_merge: Whether to force one child process per context.
    :return: ``(rows, spawned)``, the plot values per chart bar and one tuple of
        served sids per child process.
    """
    import os
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.core import security_mp

    sys.modules.pop(Path(__file__).stem, None)

    spawned: list = []
    context = security_mp.mp_context
    original = context.Process

    def _recording_process(*args, **kwargs):
        sec_args = kwargs.get('args') or ()
        if sec_args:
            spawned.append(tuple(sec_args[0]))
        return original(*args, **kwargs)

    rows: list[dict] = []
    context.Process = _recording_process
    if no_merge:
        os.environ['PYNE_NO_SECURITY_MERGE'] = '1'
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            minute = __test_helper_hour // 60
            security_data = {
                "15": __test_helper_write_feed(tmp, "15", 15 * minute, "MSG"),
                "30": __test_helper_write_feed(tmp, "30", 30 * minute, "MSG"),
                "60": __test_helper_write_feed(tmp, "60", __test_helper_hour, "MSG"),
                "120": __test_helper_write_feed(tmp, "120", 2 * __test_helper_hour,
                                                "MSG"),
                "240": __test_helper_write_feed(tmp, "240", 4 * __test_helper_hour,
                                                "MSG"),
            }
            r = runner(__test_helper_chart_bars(), security_data=security_data)
            for _candle, pv in r.run_iter():
                rows.append(dict(pv))
    finally:
        del context.Process
        if no_merge:
            os.environ.pop('PYNE_NO_SECURITY_MERGE', None)
    return rows, spawned


def __test_groups_split_exactly_where_the_feed_or_the_syntax_differs__(runner, log):
    """Which contexts merge, which stay apart — and identical values either way.

    Nine contexts covering the five shapes the split rules have to get right: a
    lookahead difference (apart), a ``gaps`` difference (together), two
    input-derived timeframes holding the same string (apart, because the key is
    syntactic), a conditional inline call (always alone) and a helper called
    twice (together). Whatever the split, the plotted values must equal the
    one-child-per-context run.
    """
    merged_rows, merged = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, no_merge=False))
    plain_rows, plain = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, no_merge=True))

    log.info("merged children: %s", merged)
    assert len(plain) == 9, f"unmerged children: {plain}"
    assert all(len(served) == 1 for served in plain), \
        f"unmerged child serves several contexts: {plain}"

    # Contexts are numbered in the order the transformer meets them, and the
    # helper is defined before ``main()``: 0-1 are the helper's two call sites
    # (e), 2-3 the lookahead pair (a), 4-5 the ``gaps`` pair (b), 6-7 the
    # input-derived pair (c) and 8 the conditional call (d).
    shared = sorted(tuple(sorted(int(sid.rsplit('\u00b7', 1)[1]) for sid in served))
                    for served in merged if len(served) > 1)
    assert shared == [(0, 1), (4, 5)], (
        f"the wrong contexts shared a child: {merged}")
    assert len(merged) == 7, f"merged children: {merged}"

    __test_helper_compare(merged_rows, plain_rows, log, "split-groups")
    for i in range(len(merged_rows)):
        row = merged_rows[i]
        assert __test_helper_same(row['inA'], row['inB']), (
            f"bar {i}: the two input-derived timeframes resolved to the same "
            f"feed but answered differently: {row['inA']!r} vs {row['inB']!r}")


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
