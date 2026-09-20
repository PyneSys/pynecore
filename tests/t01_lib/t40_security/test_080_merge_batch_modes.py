"""
@pyne
"""
from pynecore.lib import barmerge, close, plot, request, script, syminfo, ta
from pynecore.types import Series


@script.indicator(title="Merge Batch Modes", shorttitle="MBM")
def main():
    # (a) A ``lookahead_on`` group on the chart's own symbol. This is the shape
    # that runs as a DEVELOPING BATCH: the whole historical phase is one round,
    # planned by the chart and replayed by the child. Merged, one child replays
    # that plan for all three members at once.
    devA: Series[float] = request.security(syminfo.tickerid, "60", close,
                                           lookahead=barmerge.lookahead_on)
    devB: Series[float] = request.security(syminfo.tickerid, "60",
                                           ta.sma(close, 3),
                                           lookahead=barmerge.lookahead_on)
    devC: Series[float] = request.security(syminfo.tickerid, "60", close * 1.5,
                                           lookahead=barmerge.lookahead_on)
    # (b) A ``lookahead_off`` group on another context: closed-only rounds, so
    # the child publishes a period's value only once that period is over and the
    # chart carries an ``na`` prefix.
    offA: Series[float] = request.security(syminfo.tickerid, "120", close)
    offB: Series[float] = request.security(syminfo.tickerid, "120",
                                           ta.sma(close, 2))
    plot(devA, "devA")
    plot(devB, "devB")
    plot(devC, "devC")
    plot(offA, "offA")
    plot(offB, "offB")


# The strategy side script: two merged higher-timeframe values drive the
# entries and exits, so a merge that shifted either value by one round would
# move the trades.
__test_helper_strategy_script = '''"""
@pyne
"""
from pynecore.lib import close, na, request, script, strategy, syminfo, ta
from pynecore.types import Series


@script.strategy("Merge Batch Strategy", overlay=True, initial_capital=100000,
                 default_qty_type=strategy.fixed, default_qty_value=1)
def main():
    fast: Series[float] = request.security(syminfo.tickerid, "60", ta.sma(close, 2))
    slow: Series[float] = request.security(syminfo.tickerid, "60", ta.sma(close, 5))
    if not na(fast) and not na(slow):
        if fast > slow:
            strategy.entry("L", strategy.long)
        else:
            strategy.close("L")
'''


def __test_helper_chart_window(tmp_dir, timeframe, bars):
    """Write the chart's own bars to an ``.ohlcv`` file and window over it.

    A developing batch is replayed in the CHILD from the window it is handed, so
    the chart's stream has to be a static feed here too.

    :param tmp_dir: Directory to write into.
    :param timeframe: The chart's timeframe.
    :param bars: The chart bars to write.
    :return: A :class:`ChartBarWindow` over the whole file.
    """
    from pynecore.core.ohlcv import ChartBarWindow, OHLCVWriter

    path = tmp_dir / ("chart" + timeframe + ".ohlcv")
    with OHLCVWriter(path, timeframe) as w:
        for bar in bars:
            w.write(bar)
    return ChartBarWindow(path, bars[0].timestamp, bars[-1].timestamp)


def __test_helper_run(runner, no_merge):
    """Run the indicator once, recording the children and which ones were batched.

    :param runner: The ``runner`` fixture.
    :param no_merge: Whether to force one child process per context.
    :return: ``(rows, spawned, batched)`` — plot values per chart bar, one tuple
        of served sids per child, and the sids whose child got a planned
        developing sequence.
    """
    import os
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.core import security_mp

    sys.modules.pop(Path(__file__).stem, None)

    spawned: list = []
    batched: list = []
    context = security_mp.mp_context
    original = context.Process

    def _recording_process(*args, **kwargs):
        sec_args = kwargs.get('args') or ()
        if sec_args:
            spawned.append(tuple(sec_args[0]))
            if sec_args[-1]:
                batched.extend(sec_args[0])
        return original(*args, **kwargs)

    rows: list[dict] = []
    context.Process = _recording_process
    if no_merge:
        os.environ['PYNE_NO_SECURITY_MERGE'] = '1'
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            security_data = {
                "60": __test_helper_write_feed(tmp, "60", __test_helper_hour, "MBM"),
                "120": __test_helper_write_feed(tmp, "120", 2 * __test_helper_hour,
                                                "MBM"),
            }
            bars = __test_helper_chart_bars()
            window = __test_helper_chart_window(tmp, "5", bars)
            r = runner(window.bars(), security_data=security_data,
                       last_bar_index=len(bars) - 1,
                       last_bar_time=bars[-1].timestamp,
                       chart_bar_window=window)
            for _candle, pv in r.run_iter():
                rows.append(dict(pv))
    finally:
        del context.Process
        if no_merge:
            os.environ.pop('PYNE_NO_SECURITY_MERGE', None)
    return rows, spawned, batched


def __test_helper_run_strategy(runner, no_merge):
    """Run the side strategy once and collect its closed trades.

    :param runner: The ``runner`` fixture (used only for its chart ``syminfo``).
    :param no_merge: Whether to force one child process per context.
    :return: ``(trades, spawned)`` — one comparable tuple per closed trade and
        one tuple of served sids per child process.
    """
    import os
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.core import security_mp

    spawned: list = []
    context = security_mp.mp_context
    original = context.Process

    def _recording_process(*args, **kwargs):
        sec_args = kwargs.get('args') or ()
        if sec_args:
            spawned.append(tuple(sec_args[0]))
        return original(*args, **kwargs)

    trades: list = []
    context.Process = _recording_process
    if no_merge:
        os.environ['PYNE_NO_SECURITY_MERGE'] = '1'
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            feed = __test_helper_write_feed(tmp, "60", __test_helper_hour, "MBS")
            path = tmp / "merge_batch_strategy.py"
            path.write_text(__test_helper_strategy_script)
            chart = runner(__test_helper_chart_bars())
            sys.modules.pop(path.stem, None)
            sys.path.insert(0, str(tmp))
            try:
                sr = ScriptRunner(path, iter(__test_helper_chart_bars()),
                                  chart.syminfo, security_data={"60": feed})
                for _candle, _pv, new_closed in sr.run_iter():
                    for trade in new_closed:
                        trades.append((
                            trade.entry_id, trade.entry_bar_index,
                            trade.entry_price, trade.exit_id,
                            trade.exit_bar_index, trade.exit_price,
                            trade.size, trade.profit))
            finally:
                sys.path.remove(str(tmp))
                sys.modules.pop(path.stem, None)
    finally:
        del context.Process
        if no_merge:
            os.environ.pop('PYNE_NO_SECURITY_MERGE', None)
    return trades, spawned


def __test_batched_and_closed_only_groups_match_the_unmerged_run__(runner, log):
    """A developing-batch group and a closed-only group both stay value-exact.

    The ``lookahead_on`` trio runs its whole historical phase as one planned
    batch; merged, a single child replays that plan for all three sids at once,
    which is the case where a mis-dispatched write would be silent. The
    ``lookahead_off`` pair is the closed-only counterpart, ``na`` prefix
    included.
    """
    merged_rows, merged, merged_batched = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, no_merge=False))
    plain_rows, plain, plain_batched = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, no_merge=True))

    log.info("merged children: %s, batched: %s", merged, merged_batched)
    assert len(merged) == 2, f"the two groups did not share two processes: {merged}"
    assert sorted(len(served) for served in merged) == [2, 3], \
        f"served contexts: {merged}"
    assert len(plain) == 5, f"unmerged children: {plain}"
    assert all(len(served) == 1 for served in plain), \
        f"unmerged child serves several contexts: {plain}"
    assert merged_batched, "no merged context ran as a developing batch"
    assert sorted(merged_batched) == sorted(plain_batched), (
        f"merging changed which contexts were batched: {merged_batched} vs "
        f"{plain_batched}")

    __test_helper_compare(merged_rows, plain_rows, log, "batch-modes")


def __test_strategy_trades_are_unchanged_by_merging__(runner, log):
    """Two merged higher-timeframe values drive a strategy to the same trades.

    Plot equality is value equality on the bar the value is read; a strategy
    turns the same values into orders, so a one-round shift that a plot diff
    might absorb moves an entry bar instead. The two runs must produce the same
    closed trades, field for field.
    """
    merged_trades, merged = __test_helper_run_with_timeout(
        lambda: __test_helper_run_strategy(runner, no_merge=False))
    plain_trades, plain = __test_helper_run_with_timeout(
        lambda: __test_helper_run_strategy(runner, no_merge=True))

    assert len(merged) == 1, f"the group did not share one process: {merged}"
    assert len(merged[0]) == 2, f"served contexts: {merged[0]}"
    assert len(plain) == 2, f"unmerged children: {plain}"
    assert merged_trades, "the strategy closed no trades at all"
    assert len(merged_trades) == len(plain_trades), (
        f"trade counts differ: merged={len(merged_trades)} "
        f"unmerged={len(plain_trades)}")
    for i in range(len(merged_trades)):
        assert merged_trades[i] == plain_trades[i], (
            f"trade {i}: merged={merged_trades[i]!r} "
            f"unmerged={plain_trades[i]!r}")
    log.info("%d closed trades identical, merged into %d child",
             len(merged_trades), len(merged))


# Every timestamp here is Unix MILLISECONDS.
__test_helper_t0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, on the 5m/1h/2h grids
__test_helper_hour = 3_600_000
__test_helper_step = 300_000  # 5 minutes
__test_helper_hours = 72
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
