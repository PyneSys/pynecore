"""
@pyne
"""
from pynecore.lib import (
    barmerge, close, na, plot, request, script, syminfo, ta
)
from pynecore.types import IBPersistent, Series


def __test_helper_tick_count():
    """Count every execution of this expression, re-ticks included.

    A ``varip`` slot is the one piece of child state a developing re-tick does
    NOT roll back, so this counter is a direct fingerprint of the ROUND
    SEQUENCE the child ran. A batch that dropped, merged or reordered a
    same-period re-tick would show up here and nowhere else.

    :return: The number of executions so far in this context.
    """
    n: IBPersistent[int] = 0
    n += 1
    return n


@script.indicator(title="Developing Batch Equality", shorttitle="DBE")
def main():
    # Plain higher-timeframe developing read, with an ``na`` warmup prefix.
    hourly: Series[float] = request.security(
        syminfo.tickerid, "60", ta.sma(close, 3),
        lookahead=barmerge.lookahead_on)
    # ``gaps_on``: a value only on the chart bars that open a fresh period.
    gapped: Series[float] = request.security(
        syminfo.tickerid, "60", close, gaps=barmerge.gaps_on,
        lookahead=barmerge.lookahead_on)
    # A daily context on an intraday chart, and the previous daily close — ``na``
    # for the whole first day, which is the chart's own prefix.
    daily: Series[float] = request.security(
        syminfo.tickerid, "D", close, lookahead=barmerge.lookahead_on)
    prev_daily: Series[float] = request.security(
        syminfo.tickerid, "D", close[1], lookahead=barmerge.lookahead_on)
    # The round-sequence fingerprint.
    ticks: Series[float] = request.security(
        syminfo.tickerid, "60", __test_helper_tick_count(),
        lookahead=barmerge.lookahead_on)
    # A dependent pair: the producer has a consumer and the consumer has a
    # dependency, so NEITHER may be batched — both keep the per-bar path, and
    # their values have to be identical either way too.
    producer: Series[float] = request.security(
        syminfo.tickerid, "60", close, lookahead=barmerge.lookahead_on)
    consumer: Series[float] = request.security(
        syminfo.tickerid, "60", ta.sma(producer, 2),
        lookahead=barmerge.lookahead_on)
    plot(hourly, "hourly")
    plot(gapped, "gapped")
    plot(daily, "daily")
    plot(prev_daily, "prev_daily")
    plot(ticks, "ticks")
    plot(producer, "producer")
    plot(consumer, "consumer")
    plot(na if na(hourly) else hourly - close, "spread")


# Every timestamp here is Unix MILLISECONDS.
__test_helper_t0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, on the 5m/1h/1D grid
__test_helper_step = 300_000  # 5 minutes
__test_helper_day_ms = 86_400_000
# Two whole days of 5-minute bars. The LAST chart bar (23:55) closes at midnight,
# so it completes an hourly AND a daily period — the period-closing-last-bar case.
__test_helper_days = 2
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

    path = tmp_dir / f"DBE{timeframe}.ohlcv"
    total = (__test_helper_days * __test_helper_day_ms) // span_ms
    with OHLCVWriter(path, timeframe) as w:
        for i in range(total):
            c = 100.0 + i
            w.write(OHLCV(timestamp=__test_helper_t0 + i * span_ms, open=c,
                          high=c + 1.0, low=c - 1.0, close=c, volume=10.0))
    SymInfo(
        prefix="PYTEST", description="Dev Batch", ticker="TEST",
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

    The developing batch is replayed in the CHILD, which reproduces the chart's
    bars from the window it is handed — so the chart's stream has to be a static
    feed here too, and the bar loop takes its bars from the same window.

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

    A plot value is ``na`` as an :class:`NA` marker (nothing was published) or
    as a float NaN (the expression itself evaluated to ``na``). Both paths
    produce both shapes, so the comparison has to see them as one.

    :param value: The read value.
    :return: Whether it is ``na``.
    """
    from pynecore.types.na import NA
    return isinstance(value, NA) or (isinstance(value, float) and value != value)


def __test_helper_same(a, b):
    """Pine-style equality for two read values: ``na`` matches ``na``.

    :param a: One value.
    :param b: The other.
    :return: Whether the two reads answered the same.
    """
    if __test_helper_is_na(a) or __test_helper_is_na(b):
        return __test_helper_is_na(a) and __test_helper_is_na(b)
    return a == b


def __test_helper_run(runner, no_batch):
    """Run the script once, with the batch rounds on or off.

    ``no_batch`` flips :data:`pynecore.core.security.NO_BATCH`, the switch
    ``PYNE_NO_SECURITY_BATCH`` sets: every context then keeps the per-bar round
    path. The chart's bars are handed to the runner as a window over a static
    feed, which is what a developing batch's child reproduces its rounds from —
    and what the bar loop itself takes its bars from.

    :param runner: The ``runner`` fixture.
    :param no_batch: Whether to force the per-bar path.
    :return: ``(rows, batched_sids)`` — the plot values per chart bar, and the
        sids whose child was spawned with a planned developing sequence (one
        child can serve a whole context group).
    """
    import sys
    import tempfile
    from pathlib import Path

    import pynecore.core.security as security_module
    from pynecore.core import security_mp

    sys.modules.pop(Path(__file__).stem, None)

    batched: list[str] = []
    context = security_mp.mp_context
    original = context.Process

    def _recording_process(*args, **kwargs):
        sec_args = kwargs.get('args') or ()
        if sec_args and sec_args[-1]:
            batched.extend(sec_args[0])
        return original(*args, **kwargs)

    rows: list[dict] = []
    previous = security_module.NO_BATCH
    security_module.NO_BATCH = no_batch
    context.Process = _recording_process
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
    finally:
        del context.Process
        security_module.NO_BATCH = previous
    return rows, batched


def __test_developing_batch_matches_per_bar_path__(runner, log):
    """A batched ``lookahead_on`` context answers exactly like the per-bar one.

    The whole historical phase of such a context runs as ONE round: the chart
    plans every push its per-bar path would have made and the child replays the
    plan while running ahead. That is an optimization and nothing else, so the
    two paths must produce the same plot output value for value — including the
    ``na`` prefixes, the ``gaps_on`` holes, the daily context on this intraday
    chart, the daily ``close[1]`` that is ``na`` for the whole first day, the
    dependent pair that keeps the per-bar path, and the ``varip`` counter that
    fingerprints the round sequence.
    """
    batch_rows, batched = __test_helper_run(runner, no_batch=False)
    plain_rows, unbatched = __test_helper_run(runner, no_batch=True)

    assert batched, "no context ran as a developing batch"
    assert not unbatched, f"batch planned with the switch off: {unbatched}"
    # The dependent pair must NOT be batched: 5 of the 7 contexts are.
    assert len(batched) == 5, f"batched contexts: {batched}"
    assert len(set(batched)) == 5, f"duplicate batched context: {batched}"
    assert len(batch_rows) == len(plain_rows) == __test_helper_bars

    for i, (got, want) in enumerate(zip(batch_rows, plain_rows)):
        assert got.keys() == want.keys(), f"bar {i}: plot columns differ"
        for key in want:
            assert __test_helper_same(got[key], want[key]), (
                f"bar {i} '{key}': batch={got[key]!r} per-bar={want[key]!r}")

    log.info("developing batch matched the per-bar path on %d bars x %d plots, "
             "%d contexts batched", len(batch_rows), len(batch_rows[0]), len(batched))


def __test_developing_batch_replays_every_re_tick__(runner, log):
    """The batch runs one round per chart bar, re-ticks included.

    The ``varip`` counter in the security expression survives a developing
    re-tick's rollback, so it counts the child's EXECUTIONS. The per-bar path
    runs one per chart bar (plus the period closes), and the plan must
    reproduce that exactly — a compressed same-period re-tick would leave the
    counter behind.
    """
    batch_rows, _batched = __test_helper_run(runner, no_batch=False)
    plain_rows, _ = __test_helper_run(runner, no_batch=True)

    last_batch = batch_rows[-1]["ticks"]
    last_plain = plain_rows[-1]["ticks"]
    assert not __test_helper_is_na(last_batch), "ticks stayed na"
    assert last_batch == last_plain, (
        f"round count differs: batch={last_batch} per-bar={last_plain}")
    # Far more than one execution per hourly PERIOD: the count has to be on the
    # chart's scale, which is what proves the re-ticks are in there at all.
    periods = __test_helper_bars // 12
    assert last_batch > periods * 10, (
        f"only {last_batch} executions for {periods} hourly periods")

    log.info("batch replayed %d executions, same as the per-bar path", last_batch)
