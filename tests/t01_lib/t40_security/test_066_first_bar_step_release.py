"""
@pyne
"""
from pynecore.core.series import inline_series
from pynecore.lib import barmerge, plot, request, script, syminfo, ta
from pynecore.types import Series


@script.indicator(title="First Bar Step Release", shorttitle="FBSR")
def main():
    # The chart starts inside the daily feed, so its FIRST bar runs two rounds of
    # each context: the replay of the days before the chart, then the developing
    # day. The developing value differs from the previous day's, which is what
    # tells the two rounds apart.
    atr: Series[float] = request.security(
        syminfo.tickerid, "D", ta.atr(3), lookahead=barmerge.lookahead_on)
    prev_atr: Series[float] = request.security(
        syminfo.tickerid, "D", inline_series(ta.atr(3), 1),
        lookahead=barmerge.lookahead_on)
    # Work standing AFTER the writes: the child releases the chart at its write
    # and keeps running this, so a second release at the end of the round lands
    # while the chart is already waiting for the next round.
    spin = 0
    for _i in range(_SPIN):
        spin += 1
    plot(atr, "atr")
    plot(prev_atr, "prev_atr")


# Iterations of the work standing after the writes.
_SPIN = 20_000

# Every timestamp here is Unix MILLISECONDS.
__test_helper_day_ms = 86_400_000
__test_helper_step_ms = 1_800_000  # 30 minutes
__test_helper_chart_t0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC
__test_helper_hist_days = 20
__test_helper_chart_days = 3


def __test_helper_price(i):
    return 100.0 + (i % 17) * 0.9 - (i % 5) * 0.4


def __test_helper_all_bars():
    """30-minute bars from the first history day to the chart's last bar.

    :return: The bar list.
    """
    from pynecore.types.ohlcv import OHLCV
    t0 = __test_helper_chart_t0 - __test_helper_hist_days * __test_helper_day_ms
    days = __test_helper_hist_days + __test_helper_chart_days
    bars = []
    for i in range(days * 48):
        o = __test_helper_price(i)
        c = __test_helper_price(i + 1)
        bars.append(OHLCV(timestamp=t0 + i * __test_helper_step_ms, open=o,
                          high=max(o, c) + 0.5 + (i % 7) * 0.2,
                          low=min(o, c) - 0.5 - (i % 3) * 0.3,
                          close=c, volume=1.0))
    return bars


def __test_helper_syminfo(period):
    from datetime import time
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    return SymInfo(
        prefix="PYTEST", description="First Bar Release", ticker="TEST",
        currency="USD", period=period, type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    )


def __test_helper_write_feeds(tmp_dir):
    """Write the daily feed (history included) and the chart's own window.

    :param tmp_dir: Directory to write into.
    :return: ``(daily_path, chart_window, chart_bars)``.
    """
    from pynecore.core.ohlcv import ChartBarWindow, OHLCVWriter
    from pynecore.types.ohlcv import OHLCV

    bars = __test_helper_all_bars()
    days = __test_helper_hist_days + __test_helper_chart_days
    daily = tmp_dir / "FBSRD.ohlcv"
    with OHLCVWriter(daily, "1D") as w:
        for d in range(days):
            chunk = bars[d * 48:(d + 1) * 48]
            w.write(OHLCV(timestamp=chunk[0].timestamp, open=chunk[0].open,
                          high=max(b.high for b in chunk),
                          low=min(b.low for b in chunk),
                          close=chunk[-1].close, volume=float(len(chunk))))
    __test_helper_syminfo("1D").save_toml(daily.with_suffix(".toml"))

    chart_bars = [b for b in bars if b.timestamp >= __test_helper_chart_t0]
    chart = tmp_dir / "FBSR30.ohlcv"
    with OHLCVWriter(chart, "30") as w:
        for bar in chart_bars:
            w.write(bar)
    window = ChartBarWindow(chart, chart_bars[0].timestamp, chart_bars[-1].timestamp)
    return str(daily), window, chart_bars


def __test_helper_run(runner, no_batch):
    """Run the script once, with the batch rounds on or off.

    :param runner: The ``runner`` fixture.
    :param no_batch: Whether to force the per-bar round path.
    :return: The plotted rows, one per chart bar.
    """
    import sys
    import tempfile
    from pathlib import Path

    import pynecore.core.security as security_module

    sys.modules.pop(Path(__file__).stem, None)

    rows: list[dict] = []
    previous = security_module.NO_BATCH
    security_module.NO_BATCH = no_batch
    try:
        with tempfile.TemporaryDirectory() as td:
            daily, window, bars = __test_helper_write_feeds(Path(td))
            r = runner(window.bars(), {"period": "30"}, security_data={"D": daily},
                       last_bar_index=len(bars) - 1,
                       last_bar_time=bars[-1].timestamp,
                       chart_bar_window=window)
            for _candle, pv in r.run_iter():
                rows.append(dict(pv))
    finally:
        security_module.NO_BATCH = previous
    return rows


def __test_per_bar_first_bar_reads_the_developing_round__(runner, log):
    """The per-bar path answers the chart's first bar from its LAST round.

    The child releases the chart at the round's final write and wakes it once
    more when the round ends. That second wake-up must not answer the chart's
    wait for the NEXT round: the chart would then read the replay round's value
    on the first bar instead of the developing day's. The planned batch builds
    the same rounds without that wait, so the two paths have to agree on every
    bar.
    """
    batch_rows = __test_helper_run(runner, no_batch=False)
    plain_rows = __test_helper_run(runner, no_batch=True)

    assert len(batch_rows) == len(plain_rows) == __test_helper_chart_days * 48
    first = batch_rows[0]
    assert first["atr"] != first["prev_atr"], \
        "fixture: the developing day must move the ATR on the first bar"
    for i, (got, want) in enumerate(zip(plain_rows, batch_rows)):
        for key in ("atr", "prev_atr"):
            assert got[key] == want[key], (
                f"bar {i} '{key}': per-bar={got[key]!r} batch={want[key]!r}")
    log.info("per-bar first bar matched the developing batch on %d bars", len(plain_rows))
