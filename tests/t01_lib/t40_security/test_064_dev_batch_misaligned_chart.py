"""
@pyne
"""
from pynecore.lib import (
    barmerge, close, na, plot, request, script, syminfo
)
from pynecore.types import Series


@script.indicator(title="Developing Batch Misaligned Chart", shorttitle="DBM")
def main():
    # A 60-minute context on a 45-MINUTE chart: the two grids never align, so an
    # hourly period completes in the MIDDLE of a chart bar. The chart bar that
    # completes it publishes only the closed record, whose ring entry carries the
    # period's own scheduled close (01:00) and not the bar's tick (01:30).
    hourly: Series[float] = request.security(
        syminfo.tickerid, "60", close, lookahead=barmerge.lookahead_on)
    gapped: Series[float] = request.security(
        syminfo.tickerid, "60", close, gaps=barmerge.gaps_on,
        lookahead=barmerge.lookahead_on)
    plot(hourly, "hourly")
    plot(gapped, "gapped")
    plot(na if na(hourly) else hourly - close, "spread")


# Every timestamp here is Unix MILLISECONDS.
__test_helper_t0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC
__test_helper_chart_step = 2_700_000  # 45 minutes
__test_helper_day_ms = 86_400_000
__test_helper_days = 2
__test_helper_bars = (__test_helper_days * __test_helper_day_ms) // __test_helper_chart_step


def __test_helper_write_feed(tmp_dir):
    """Write the 60-minute security feed over the chart's range.

    :param tmp_dir: Directory to write into.
    :return: Path to the written ``.ohlcv`` file, as a string.
    """
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    span_ms = 3_600_000
    path = tmp_dir / "DBM60.ohlcv"
    total = (__test_helper_days * __test_helper_day_ms) // span_ms
    with OHLCVWriter(path, "60") as w:
        for i in range(total):
            c = 100.0 + i
            w.write(OHLCV(timestamp=__test_helper_t0 + i * span_ms, open=c,
                          high=c + 1.0, low=c - 1.0, close=c, volume=10.0))
    SymInfo(
        prefix="PYTEST", description="Dev Batch Misaligned", ticker="TEST",
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
    """The chart's own 45-minute bars, with a moving close.

    :return: The bar list.
    """
    from pynecore.types.ohlcv import OHLCV
    bars = []
    for i in range(__test_helper_bars):
        c = 50.0 + (i % 17) * 0.25
        bars.append(OHLCV(timestamp=__test_helper_t0 + i * __test_helper_chart_step,
                          open=c - 0.1, high=c + 0.5, low=c - 0.5, close=c,
                          volume=1.0 + i))
    return bars


def __test_helper_chart_window(tmp_dir, bars):
    """Write the chart's bars to an ``.ohlcv`` file and window over it.

    :param tmp_dir: Directory to write into.
    :param bars: The chart bars to write.
    :return: A :class:`ChartBarWindow` over the whole file.
    """
    from pynecore.core.ohlcv import ChartBarWindow, OHLCVWriter

    path = tmp_dir / "chart45.ohlcv"
    with OHLCVWriter(path, "45") as w:
        for bar in bars:
            w.write(bar)
    return ChartBarWindow(path, bars[0].timestamp, bars[-1].timestamp)


def __test_helper_is_na(value):
    """Whether a read answered Pine ``na``, in either of its two shapes.

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
    """Run the script once on the 45-minute chart, batch rounds on or off.

    :param runner: The ``runner`` fixture.
    :param no_batch: Whether to force the per-bar path.
    :return: ``(rows, batched_sids)`` — the plot values per chart bar, and the
        sids whose child was spawned with a planned developing sequence.
    """
    import multiprocessing
    import sys
    import tempfile
    from pathlib import Path

    import pynecore.core.security as security_module

    sys.modules.pop(Path(__file__).stem, None)

    batched: list[str] = []
    original = multiprocessing.Process

    def _recording_process(*args, **kwargs):
        sec_args = kwargs.get('args') or ()
        if sec_args and sec_args[-1]:
            batched.append(sec_args[0])
        return original(*args, **kwargs)

    rows: list[dict] = []
    previous = security_module.NO_BATCH
    security_module.NO_BATCH = no_batch
    multiprocessing.Process = _recording_process
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            feeds = {"60": __test_helper_write_feed(tmp)}
            bars = __test_helper_chart_bars()
            window = __test_helper_chart_window(tmp, bars)
            r = runner(window.bars(), {"period": "45"}, security_data=feeds,
                       last_bar_index=len(bars) - 1,
                       last_bar_time=bars[-1].timestamp,
                       chart_bar_window=window)
            for _candle, pv in r.run_iter():
                rows.append(dict(pv))
    finally:
        multiprocessing.Process = original
        security_module.NO_BATCH = previous
    return rows, batched


def __test_developing_batch_on_a_misaligned_chart__(runner, log):
    """A batched context whose periods close mid-bar answers like the per-bar path.

    On a 45-minute chart an hourly period is completed by a chart bar whose own
    tick lies HALF AN HOUR past the period close, and that bar's only record is
    the closed one — its ring entry closes on the security's grid, not on the
    chart's. The batch read has to pair it all the same, so this is the case an
    exact tick-equality replay check rejected.
    """
    batch_rows, batched = __test_helper_run(runner, no_batch=False)
    plain_rows, unbatched = __test_helper_run(runner, no_batch=True)

    assert batched, "no context ran as a developing batch"
    assert not unbatched, f"batch planned with the switch off: {unbatched}"
    assert len(batch_rows) == len(plain_rows) == __test_helper_bars

    for i, (got, want) in enumerate(zip(batch_rows, plain_rows)):
        assert got.keys() == want.keys(), f"bar {i}: plot columns differ"
        for key in want:
            assert __test_helper_same(got[key], want[key]), (
                f"bar {i} '{key}': batch={got[key]!r} per-bar={want[key]!r}")

    log.info("misaligned developing batch matched the per-bar path on %d bars",
             len(batch_rows))
