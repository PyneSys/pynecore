"""
@pyne
"""
from pynecore.lib import barmerge, plot, request, script, syminfo, time
from pynecore.types import Series


@script.indicator(title="Lookahead On Multi-Period Calendar", shorttitle="LOMP")
def main():
    bar_time: Series[int] = time
    t3d = request.security(syminfo.tickerid, "3D", bar_time[1], lookahead=barmerge.lookahead_on)
    t2m = request.security(syminfo.tickerid, "2M", bar_time[1], lookahead=barmerge.lookahead_on)
    plot(t3d, "t3d")
    plot(t2m, "t2m")


# A 24/7 crypto feed whose session opens at midnight: ``resolve_session_anchor``
# anchors nothing, so a multi-period grid inferred from the session anchor would
# count weekdays. Its bar closes run on the symbol's 'calendar' grid, so every
# HTF period here is calendar days and months, counted from the year's first day.
# Every timestamp is Unix MILLISECONDS; the chart is hourly over 2026-01-01 ..
# 2026-03-21, inside one year, so no period straddles the year reset.
__test_helper_t0 = 1_767_225_600_000  # 2026-01-01T00:00:00 UTC
__test_helper_hour = 3_600_000
__test_helper_day = 24 * __test_helper_hour
__test_helper_hours = 79 * 24
__test_helper_mar1 = __test_helper_t0 + 59 * __test_helper_day  # 2026-03-01


def __test_helper_symbol(tf):
    """The symbol info of a 24/7 crypto feed with a midnight session open.

    :param tf: The feed's timeframe.
    :return: The :class:`SymInfo`.
    """
    from datetime import time as dtime
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession

    return SymInfo(
        prefix="PYTEST", description="Lookahead On Multi-Period", ticker="TEST",
        currency="USD", period=tf, type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=dtime(0, 0), end=dtime(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=dtime(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=dtime(23, 59, 59)) for i in range(7)],
    )


def __test_helper_bar(ts, span):
    """One bar of a series whose price is its hour index from ``t0``.

    :param ts: Bar open time in ms.
    :param span: Bar span in ms.
    :return: The bar.
    """
    from pynecore.types.ohlcv import OHLCV
    first = (ts - __test_helper_t0) // __test_helper_hour
    last = first + span // __test_helper_hour - 1
    return OHLCV(timestamp=ts, open=100.0 + first, high=100.0 + last, low=100.0 + first,
                 close=100.0 + last, volume=float(last - first + 1))


def __test_helper_feeds(tmp_dir):
    """Write the 3D and 2M feeds of the hourly series, on the calendar grid.

    :param tmp_dir: Directory to write into.
    :return: ``security_data`` for the runner.
    """
    from pynecore.core.ohlcv import OHLCVWriter

    end = __test_helper_t0 + __test_helper_hours * __test_helper_hour
    periods = {
        "3D": [(t, 3 * __test_helper_day)
               for t in range(__test_helper_t0, end, 3 * __test_helper_day)],
        "2M": [(__test_helper_t0, 59 * __test_helper_day),
               (__test_helper_mar1, end - __test_helper_mar1)],
    }
    feeds = {}
    for tf, bars in periods.items():
        path = tmp_dir / f"LOMP{tf}.ohlcv"
        with OHLCVWriter(path, tf) as w:
            for ts, span in bars:
                w.write(__test_helper_bar(ts, span))
        __test_helper_symbol(tf).save_toml(path.with_suffix(".toml"))
        feeds[tf] = str(path)
    return feeds


def __test_helper_run(runner, batched):
    """Run the script on the hourly chart, per bar or as a developing batch.

    :param runner: The ``runner`` fixture.
    :param batched: Hand the runner a chart bar window, which arms the
        historical developing batch (``iter_dev_batch_records``).
    :return: ``(chart bar open, plot values)`` per chart bar.
    """
    import sys
    import tempfile
    from pathlib import Path

    from pynecore.core.ohlcv import ChartBarWindow, OHLCVWriter

    sys.modules.pop(Path(__file__).stem, None)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        bars = [__test_helper_bar(__test_helper_t0 + h * __test_helper_hour, __test_helper_hour)
                for h in range(__test_helper_hours)]
        kwargs = {"syminfo_override": {"period": "60"},
                  "security_data": __test_helper_feeds(tmp)}
        if batched:
            path = tmp / "chart60.ohlcv"
            with OHLCVWriter(path, "60") as w:
                for bar in bars:
                    w.write(bar)
            window = ChartBarWindow(path, bars[0].timestamp, bars[-1].timestamp)
            kwargs.update(last_bar_index=len(bars) - 1, last_bar_time=bars[-1].timestamp,
                          chart_bar_window=window)
            bars = window.bars()
        r = runner(bars, **kwargs)
        for candle, pv in r.run_iter():
            rows.append((candle.timestamp, dict(pv)))
    return rows


def __test_helper_check(rows):
    """``time[1]`` of each context is the previous calendar period's open.

    :param rows: The rows of one run.
    """
    checked = 0
    for chart_time, pv in rows:
        k = (chart_time - __test_helper_t0) // (3 * __test_helper_day)
        if k >= 1:
            want = float(__test_helper_t0 + (k - 1) * 3 * __test_helper_day)
            assert pv["t3d"] == want, (
                f"hourly bar {chart_time}: 3D time[1] = {pv['t3d']}, expected {want}")
            checked += 1
        if chart_time >= __test_helper_mar1:
            assert pv["t2m"] == float(__test_helper_t0), (
                f"hourly bar {chart_time}: 2M time[1] = {pv['t2m']}, "
                f"expected {__test_helper_t0}")
    assert checked > 1000, f"only {checked} rows were checked"


def __test_lookahead_on_multi_period_calendar_grid__(runner, log):
    """A 24/7 market's ``3D`` / ``2M`` periods are calendar days and months.

    The chart folds its own bars into the developing bar of a ``lookahead_on``
    context with an aggregator, which took no grid mode: a midnight session anchors
    nothing, so its multi-period grid was inferred as weekdays while the context's
    bar closes ran on the calendar grid.
    """
    __test_helper_check(__test_helper_run(runner, batched=False))


def __test_lookahead_on_multi_period_calendar_grid_batched__(runner, log):
    """The developing batch walks the same calendar grid as the per-bar path.

    With the weekday grid its records skipped the rounds the context's calendar
    closes expected, and the read stopped with "the developing batch published no
    round".
    """
    __test_helper_check(__test_helper_run(runner, batched=True))
