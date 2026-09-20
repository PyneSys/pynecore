"""
@pyne
"""
from pynecore.lib import (
    barmerge, close, high, low, na, open, plot, request, script, syminfo, ta
)
from pynecore.types import Series


def deferred_symbol():
    """
    The chart's own symbol, behind a CALL.

    A call is not evaluable at module level, so the context whose symbol argument
    this is cannot be resolved at setup: it is deferred, and the runtime resolver
    both resolves and PREPARES it mid-run — arming its developing batch itself.

    :return: The chart instrument's ticker id.
    """
    return syminfo.tickerid


@script.indicator(title="Dev Skip Batch", shorttitle="DSB")
def main():
    # The expressions run in the CHILD, so these two series are its own.
    sma: Series[float] = ta.sma(close, 5)
    summ: Series[float] = close + open
    # Two shifted contexts on one feed: ONE compile-time group, one child, and
    # with the skip on one developing round per hourly period instead of one per
    # chart bar.
    prev_close: Series[float] = request.security(
        syminfo.tickerid, "60", close[1], lookahead=barmerge.lookahead_on)
    prev_sma: Series[float] = request.security(
        syminfo.tickerid, "60", sma[1], lookahead=barmerge.lookahead_on)
    # A daily tuple and a daily shifted sum: the ratio of chart bars to periods
    # is at its largest here, which is where the skip removes the most.
    prev_high, prev_low = request.security(
        syminfo.tickerid, "D", [high[1], low[2]], lookahead=barmerge.lookahead_on)
    prev_sum: Series[float] = request.security(
        syminfo.tickerid, "D", summ[1], lookahead=barmerge.lookahead_on)
    # A DEFERRED context: its timeframe comes from an input, so the chart cannot
    # resolve it at setup — the runtime resolver does, mid-run, and that resolver
    # PREPARES the context itself (arming its developing batch). The skip has to
    # be decided before that, or this context's plan silently keeps every round.
    prev_dyn: Series[float] = request.security(
        deferred_symbol(), "60", close[1], lookahead=barmerge.lookahead_on)
    # A context that must keep every developing round, so the two classes are
    # planned and replayed side by side in the same run.
    mixed: Series[float] = request.security(
        syminfo.tickerid, "60", close[1] + close, lookahead=barmerge.lookahead_on)
    plot(prev_dyn, "prev_dyn")
    plot(prev_close, "prev_close")
    plot(prev_sma, "prev_sma")
    plot(prev_high, "prev_high")
    plot(prev_low, "prev_low")
    plot(prev_sum, "prev_sum")
    plot(mixed, "mixed")
    plot(na if na(prev_close) else prev_close - close, "spread")


# Every timestamp here is Unix MILLISECONDS.
__test_helper_t0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, on the 5m/1h/1D grid
__test_helper_step = 300_000  # 5 minutes
__test_helper_day_ms = 86_400_000
__test_helper_days = 1
__test_helper_bars = __test_helper_days * 288


def __test_helper_write_feed(tmp_dir, timeframe, span_ms):
    """Write one security feed at ``timeframe`` over the chart's range.

    :param tmp_dir: Directory to write into.
    :param timeframe: The feed's timeframe string.
    :param span_ms: That timeframe's period length in ms.
    :return: Path to the written ``.ohlcv`` file, as a string.
    """
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / f"DSB{timeframe}.ohlcv"
    total = (__test_helper_days * __test_helper_day_ms) // span_ms
    with OHLCVWriter(path, timeframe) as w:
        for i in range(total):
            c = 100.0 + i
            w.write(OHLCV(timestamp=__test_helper_t0 + i * span_ms, open=c,
                          high=c + 1.0, low=c - 1.0, close=c, volume=10.0))
    SymInfo(
        prefix="PYTEST", description="Dev Skip Batch", ticker="TEST",
        currency="USD", period=timeframe, type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


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

    path = tmp_dir / f"chart{timeframe}.ohlcv"
    with OHLCVWriter(path, timeframe) as w:
        for bar in bars:
            w.write(bar)
    return ChartBarWindow(path, bars[0].timestamp, bars[-1].timestamp)


def __test_helper_chart_bars():
    """The chart's own 5-minute bars, with a moving close.

    :return: The bar list.
    """
    from pynecore.types.ohlcv import OHLCV
    bars = []
    for i in range(__test_helper_bars):
        c = 50.0 + (i % 37) * 0.25
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


def __test_helper_run(runner, no_skip):
    """Run the script with the developing BATCH armed, skip on or off.

    The batch is the path where the child produces its own round sequence from
    the spec (``iter_dev_batch_records``), so the skip has to be in the spec: the
    specs the runner built are captured here and replayed through that very
    generator, while the chart's bar file still exists.

    :param runner: The ``runner`` fixture.
    :param no_skip: Whether to force a developing round per chart bar.
    :return: ``(rows, batched, plans)`` — the plot values per chart bar, the sids
        whose child was spawned with a planned sequence, and one
        ``(first_dev_only, developing records, records, closed_shift)`` tuple per
        plan.
    """
    import os
    import sys
    import tempfile
    from pathlib import Path

    from pynecore.core import security_mp
    from pynecore.core.script_runner import ScriptRunner

    sys.modules.pop(Path(__file__).stem, None)

    batched: list[str] = []
    specs: list = []
    context = security_mp.mp_context
    original_process = context.Process
    original_spec = ScriptRunner._dev_batch_spec

    def _recording_process(*args, **kwargs):
        sec_args = kwargs.get('args') or ()
        if sec_args and sec_args[-1]:
            batched.extend(sec_args[0])
        return original_process(*args, **kwargs)

    def _recording_spec(self, sec_state):
        spec = original_spec(self, sec_state)
        # ``closed_shift`` is the compile-time half of the skip; pairing it with
        # the spec is what lets the plan assertions below check that every
        # context the flag was emitted for actually got a skipping plan.
        specs.append((spec, bool(sec_state.closed_shift)))
        return spec

    rows: list[dict] = []
    context.Process = _recording_process
    ScriptRunner._dev_batch_spec = _recording_spec
    if no_skip:
        os.environ['PYNE_NO_SECURITY_DEV_SKIP'] = '1'
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            feeds = {
                "60": __test_helper_write_feed(tmp, "60", 3_600_000),
                "D": __test_helper_write_feed(tmp, "1D", __test_helper_day_ms),
            }
            bars = __test_helper_chart_bars()
            window = __test_helper_chart_window(tmp, "5", bars)
            r = runner(window.bars(), security_data=feeds,
                       last_bar_index=len(bars) - 1,
                       last_bar_time=bars[-1].timestamp,
                       chart_bar_window=window)
            for _candle, pv in r.run_iter():
                rows.append(dict(pv))
            plans = __test_helper_plans(specs)
    finally:
        del context.Process
        ScriptRunner._dev_batch_spec = original_spec
        if no_skip:
            os.environ.pop('PYNE_NO_SECURITY_DEV_SKIP', None)
    return rows, batched, plans


def __test_helper_plans(specs):
    """Replay every captured plan and count what it holds.

    :param specs: The captured ``(DevBatchSpec, closed_shift)`` pairs.
    :return: One ``(first_dev_only, developing records, records, closed_shift)``
        tuple per spec.
    """
    from pynecore.core.security import DEV_BATCH_DEVELOPING, iter_dev_batch_records

    plans = []
    for spec, closed_shift in specs:
        developing = 0
        total = 0
        for record in iter_dev_batch_records(spec):
            total += 1
            if record[0] == DEV_BATCH_DEVELOPING:
                developing += 1
        plans.append((spec.first_dev_only, developing, total, closed_shift))
    return plans


def __test_helper_run_with_timeout(fn, seconds=600):
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


def __test_helper_both_runs(runner):
    """Both halves of the A/B, skip first.

    :param runner: The ``runner`` fixture.
    :return: ``(skip result, no-skip result)``.
    """
    skip = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, no_skip=False))
    plain = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, no_skip=True))
    return skip, plain


def __test_batched_dev_skip_matches_a_round_per_bar__(runner, log):
    """A batch that drops the re-ticks answers like one that replays them all

    Both halves of this A/B run the DEVELOPING BATCH, so what is compared is the
    planned sequence with and without the skip — the per-bar path of the same
    shapes is compared in test_085. The values have to be identical, and the
    ``na`` prefixes with them.
    """
    (skip_rows, batched, _ss), (plain_rows, plain_batched, _ps) = (
        __test_helper_both_runs(runner))

    assert batched, "no context ran as a developing batch"
    assert sorted(batched) == sorted(plain_batched), (
        f"the switch changed which contexts batched: {batched} vs {plain_batched}")
    assert len(skip_rows) == len(plain_rows) == __test_helper_bars
    for i in range(len(skip_rows)):
        got = skip_rows[i]
        want = plain_rows[i]
        assert got.keys() == want.keys(), f"bar {i}: plot columns differ"
        for key in want:
            assert __test_helper_same(got[key], want[key]), (
                f"bar {i} '{key}': skip={got[key]!r} per-bar={want[key]!r}")

    log.info("%d bars x %d plots identical, %d contexts batched",
             len(skip_rows), len(skip_rows[0]), len(batched))


def __test_the_plan_holds_one_developing_record_per_period__(runner, log):
    """The skip is in the plan itself, not only in the chart's step building

    The records are produced by the very generator the child replays, so counting
    them is counting the child's rounds.
    """
    (_skip_rows, _batched, skip_plans), (_plain_rows, _pb, plain_plans) = (
        __test_helper_both_runs(runner))

    assert skip_plans, "no developing batch was planned"
    assert len(skip_plans) == len(plain_plans)
    assert any(first for first, _dev, _total, _cs in skip_plans), \
        "no plan asked for the first developing bar only"
    assert not any(first for first, _dev, _total, _cs in plain_plans), \
        "the switch left a plan skipping"

    skip_dev = sum(dev for _first, dev, _total, _cs in skip_plans)
    plain_dev = sum(dev for _first, dev, _total, _cs in plain_plans)
    skip_total = sum(total for _first, _dev, total, _cs in skip_plans)
    plain_total = sum(total for _first, _dev, total, _cs in plain_plans)
    assert plain_dev > skip_dev, "the plan kept every developing record"
    assert skip_total < plain_total
    # A skipping plan is down to one developing record per HTF period: 24 hourly
    # periods and one daily one on this chart, against one record per chart bar.
    periods = __test_helper_bars // 12 + 2
    for first, dev, _total, _cs in skip_plans:
        if first:
            assert dev <= periods, f"a skipping plan kept {dev} developing records"

    log.info("%d developing records planned instead of %d (%d vs %d in total)",
             skip_dev, plain_dev, skip_total, plain_total)


def __test_every_closed_shift_context_gets_a_skipping_plan__(runner, log):
    """The DEFERRED context's plan skips too, not only the statically resolved ones

    A deferred symbol/timeframe is resolved mid-run, and the resolver prepares
    the context — arming its developing batch — itself. Deciding the skip anywhere
    after that hands the child a plan built from a stale ``False``, which is
    invisible in the values (they are identical either way) and only shows up as
    every developing round still being replayed. Every context the compile-time
    ``closed_shift`` flag was emitted for is skip-eligible on this chart, so the
    two have to agree plan by plan.
    """
    (_skip_rows, _batched, skip_plans), (_plain_rows, _pb, plain_plans) = (
        __test_helper_both_runs(runner))

    assert skip_plans, "no developing batch was planned"
    for first, _dev, _total, closed_shift in skip_plans:
        assert first == closed_shift, (
            f"a plan of a closed_shift={closed_shift} context has "
            f"first_dev_only={first}")
    assert sum(1 for _f, _d, _t, cs in skip_plans if cs) >= 5, (
        "fewer closed_shift contexts were planned than the script declares")
    # The toggle proves the counts above come from the decision and not from the
    # plan builder: with it off every plan keeps its rounds.
    for first, _dev, _total, _cs in plain_plans:
        assert not first

    log.info("%d of %d plans skip, matching closed_shift exactly",
             sum(1 for f, _d, _t, _cs in skip_plans if f), len(skip_plans))
