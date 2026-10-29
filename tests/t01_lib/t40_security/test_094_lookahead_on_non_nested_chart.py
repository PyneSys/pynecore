"""
@pyne
"""
from pynecore.lib import barmerge, close, open, plot, request, script, syminfo, ta, time, volume
from pynecore.types import Series


@script.indicator(title="Lookahead On Non-Nested Chart", shorttitle="LONN")
def main():
    # The expressions run in the CHILD, so the moving average is the context's own:
    # it accumulates over the closed HTF bars the context is handed.
    sma: Series[float] = ta.sma(close, 3)
    bar_time: Series[int] = time
    c5, o5, v5, t5, s5 = request.security(
        syminfo.tickerid, "5", [close[1], open[1], volume[1], bar_time[1], sma[1]],
        lookahead=barmerge.lookahead_on)
    c15, o15, v15, t15, s15 = request.security(
        syminfo.tickerid, "15", [close[1], open[1], volume[1], bar_time[1], sma[1]],
        lookahead=barmerge.lookahead_on)
    plot(c5, "c5")
    plot(o5, "o5")
    plot(v5, "v5")
    plot(t5, "t5")
    plot(s5, "s5")
    plot(c15, "c15")
    plot(o15, "o15")
    plot(v15, "v15")
    plot(t15, "t15")
    plot(s15, "s15")


# Every timestamp here is Unix MILLISECONDS. Every feed is derived from ONE minute
# series, like a real exchange feed: the minute ``m`` (counted from ``t0``) closes
# at ``100 + m``, so a ``tf``-minute bar opening at minute ``p`` opens at ``100 + p``,
# closes at ``100 + p + tf - 1`` and carries volume ``tf``. A 5-minute bar folded
# from 3-minute chart bars therefore shows up in its close AND its volume.
__test_helper_t0 = 1_785_024_000_000  # 2026-07-26T00:00:00 UTC
__test_helper_minute = 60_000
__test_helper_minutes = 6 * 60


def __test_helper_bars(tf, skip=()):
    """The ``tf``-minute bars of the shared minute series.

    :param tf: Bar span in minutes.
    :param skip: Open minutes of the bars the feed does not have.
    :return: The bar list.
    """
    from pynecore.types.ohlcv import OHLCV
    out = []
    for p in range(0, __test_helper_minutes, tf):
        if p in skip:
            continue
        out.append(OHLCV(timestamp=__test_helper_t0 + p * __test_helper_minute,
                         open=100.0 + p, high=100.0 + p + tf - 1, low=100.0 + p,
                         close=100.0 + p + tf - 1, volume=float(tf)))
    return out


def __test_helper_write_feed(tmp_dir, tf, skip=(), legacy=False):
    """Write the ``tf``-minute feed with its symbol info.

    :param tmp_dir: Directory to write into.
    :param tf: Bar span in minutes.
    :param skip: Open minutes of the bars the feed does not have.
    :param legacy: Write a legacy v1 file, which cannot express a gap: every
        skipped bar becomes a flat ``volume == -1`` gap fill at the previous close.
    :return: Path to the written ``.ohlcv`` file, as a string.
    """
    import struct
    from datetime import time as dtime
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession

    path = tmp_dir / f"LONN{tf}.ohlcv"
    if legacy:
        records = []
        prev_close = 100.0
        for bar in __test_helper_bars(tf):
            minute = (bar.timestamp - __test_helper_t0) // __test_helper_minute
            if minute in skip:
                records.append(struct.pack("Ifffff", bar.timestamp // 1000, prev_close,
                                           prev_close, prev_close, prev_close, -1.0))
            else:
                records.append(struct.pack("Ifffff", bar.timestamp // 1000, bar.open,
                                           bar.high, bar.low, bar.close, bar.volume))
                prev_close = bar.close
        path.write_bytes(b"".join(records))
    else:
        with OHLCVWriter(path, str(tf)) as w:
            for bar in __test_helper_bars(tf, skip):
                w.write(bar)
    SymInfo(
        prefix="PYTEST", description="Lookahead On Non-Nested", ticker="TEST",
        currency="USD", period=str(tf), type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=dtime(0, 0), end=dtime(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=dtime(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=dtime(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_helper_run(runner, batched, skip=None, legacy=()):
    """Run the script on the 3-minute chart, per bar or as a developing batch.

    :param runner: The ``runner`` fixture.
    :param batched: Hand the runner a chart bar window, which arms the
        historical developing batch (``iter_dev_batch_records``).
    :param skip: Per timeframe in minutes, the open minutes of the bars that
        feed (the chart's included) does not have.
    :param legacy: The context timeframes, in minutes, whose feed is a legacy v1
        file (see :func:`__test_helper_write_feed`).
    :return: ``(rows, planned)`` — ``(chart bar open, plot values)`` per chart bar,
        and how many developing batches the runner planned.
    """
    import sys
    import tempfile
    from pathlib import Path

    from pynecore.core.ohlcv import ChartBarWindow, OHLCVWriter
    from pynecore.core.script_runner import ScriptRunner

    sys.modules.pop(Path(__file__).stem, None)
    skip = skip or {}

    planned = []
    original_spec = ScriptRunner._dev_batch_spec

    def _recording_spec(self, sec_state):
        spec = original_spec(self, sec_state)
        if spec is not None:
            planned.append(spec)
        return spec

    rows = []
    ScriptRunner._dev_batch_spec = _recording_spec
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            feeds = {str(tf): __test_helper_write_feed(tmp, tf, skip.get(tf, ()), tf in legacy)
                     for tf in (5, 15)}
            bars = __test_helper_bars(3, skip.get(3, ()))
            kwargs = {"syminfo_override": {"period": "3"}, "security_data": feeds}
            if batched:
                path = tmp / "chart3.ohlcv"
                with OHLCVWriter(path, "3") as w:
                    for bar in bars:
                        w.write(bar)
                window = ChartBarWindow(path, bars[0].timestamp, bars[-1].timestamp)
                kwargs.update(last_bar_index=len(bars) - 1,
                              last_bar_time=bars[-1].timestamp,
                              chart_bar_window=window)
                bars = window.bars()
            r = runner(bars, **kwargs)
            for candle, pv in r.run_iter():
                rows.append((candle.timestamp, dict(pv)))
    finally:
        ScriptRunner._dev_batch_spec = original_spec
    return rows, len(planned)


def __test_helper_check(rows, skip=None):
    """Every closed HTF bar a context reads must be the requested timeframe's own bar.

    ``lookahead_on`` steps into the latest bar of the feed opening at or before
    the chart bar, and every ``[1]`` reads the feed bar before that one, which is
    closed. A bar the feed does not have is stepped over (measured on
    TradingView: CAPITALCOM:BTCUSD has no 5-minute bar at 21:00 UTC, and on the
    3-minute chart ``time[1]`` of the 21:06 bar is 20:55).

    :param rows: The rows of one run.
    :param skip: Per timeframe in minutes, the open minutes the feed does not have.
    """
    skip = skip or {}
    checked = 0
    for chart_time, pv in rows:
        minute = (chart_time - __test_helper_t0) // __test_helper_minute
        for tf in (5, 15):
            opens = [p for p in range(0, minute + 1, tf) if p not in skip.get(tf, ())]
            if len(opens) < 4:
                continue  # the moving average has not seen three bars yet
            prev, closes = opens[-2], [100.0 + p + tf - 1 for p in opens[-4:-1]]
            got = (pv[f"c{tf}"], pv[f"o{tf}"], pv[f"v{tf}"], pv[f"t{tf}"], pv[f"s{tf}"])
            want = (100.0 + prev + tf - 1, 100.0 + prev, float(tf),
                    float(__test_helper_t0 + prev * __test_helper_minute),
                    sum(closes) / 3)
            assert got == want, (
                f"3m bar at minute {minute}, '{tf}' context: "
                f"(close, open, volume, time, sma)[1] = {got}, expected {want} "
                f"(the {tf}m bar opening at minute {prev})")
            checked += 1
    assert checked > 100, f"only {checked} rows were checked"


def __test_lookahead_on_closed_bars_on_non_nested_chart__(runner, log):
    """A ``lookahead_on`` context's closed bars are its timeframe's own bars on any chart.

    Issue #100. The chart folds its own bars into the developing HTF bar, and it
    used to deliver that fold as the CLOSED bar too. On a chart whose timeframe
    does not divide the context's ("5" on a 3-minute chart) a chart bar straddles
    the period boundary, so the fold is not the HTF bar: the 3m bar opening at
    02:03 runs to 02:06, gave the 02:00 5m bar its close and volume, and the 02:05
    bar lost the minute it covered. Every expression evaluated in the context ran
    on those bars, ``ta.*`` state included. A closed bar is complete past data, so
    the context's own file supplies it. The "15" context nests and is the control.
    """
    rows, _ = __test_helper_run(runner, batched=False)
    __test_helper_check(rows)


def __test_lookahead_on_closed_bars_on_non_nested_chart_batched__(runner, log):
    """The developing batch replays the same closed bars as the per-bar path.

    The batch builds its records with its own aggregator (``iter_dev_batch_records``),
    so it must mark the straddled periods exactly as the per-bar path does.
    """
    rows, planned = __test_helper_run(runner, batched=True)
    assert planned, "the runner planned no developing batch"
    __test_helper_check(rows)


def __test_lookahead_on_steps_over_a_period_the_feed_lacks__(runner, log):
    """A closed period the feed has no bar for adds no bar to the context's series.

    Measured on TradingView (CAPITALCOM:BTCUSD, 3-minute chart): the venue's daily
    maintenance leaves no 3-minute bar at 21:00 UTC and no 5-minute bar at 21:00,
    while trading resumes at 21:03. The 21:03 chart bar opens and completes the
    21:00 period on its own, and TradingView's ``[1]`` steps over that period.
    The chart's fold of the 21:03 bar must not stand in for the missing bar.
    """
    skip = {3: (120,), 5: (120,)}
    for batched in (False, True):
        rows, _ = __test_helper_run(runner, batched=batched, skip=skip)
        __test_helper_check(rows, skip)


def __test_lookahead_on_steps_over_a_legacy_feed_gap_fill__(runner, log):
    """A legacy feed's gap fill is a period the feed lacks, per bar and batched alike.

    A legacy v1 file cannot express a gap, so the missing 02:00 5-minute bar is
    stored as a flat ``volume == -1`` record. The chart's feed oracle rides the
    real bars only, so the developing batch, which asks the child's own file, has
    to skip the fill too, or it delivers the chart's fold as a bar the feed lacks.
    """
    skip = {3: (120,), 5: (120,)}
    for batched in (False, True):
        rows, _ = __test_helper_run(runner, batched=batched, skip=skip, legacy=(5,))
        __test_helper_check(rows, skip)


def __test_lookahead_on_reads_feed_bars_the_chart_never_reached__(runner, log):
    """A feed bar in a period no chart bar falls in is part of the context's series.

    Measured on TradingView (CAPITALCOM:EURUSD, 3-minute chart): the chart has no
    bars at 21:12 and 21:15 while the 5-minute feed has its 21:10 bar, and on the
    21:21 chart bar ``time[1]`` is 21:10. The 3m bars 02:12 and 02:15 are left out
    here the same way, so no chart bar opens the 02:10 period of the "5" context.
    """
    skip = {3: (132, 135)}
    for batched in (False, True):
        rows, _ = __test_helper_run(runner, batched=batched, skip=skip)
        __test_helper_check(rows, skip)


def __test_lookahead_on_closes_both_periods_after_a_chart_gap__(runner, log):
    """A chart bar after a gap closes the gap-left period and completes its own.

    Without the 02:03 and 02:06 bars the 02:00 period of the "5" context is left
    with the 02:00 bar alone, and the 02:09 bar both closes it and runs to the end
    of its own 02:05 period. Measured on CAPITALCOM:EURUSD, where the feed then
    also lacks the next period: the developing batch replayed the late close with
    no entry inside its chart bar's window, and the run stopped.
    """
    skip = {3: (123, 126), 5: (130,)}
    for batched in (False, True):
        rows, _ = __test_helper_run(runner, batched=batched, skip=skip)
        __test_helper_check(rows, skip)
