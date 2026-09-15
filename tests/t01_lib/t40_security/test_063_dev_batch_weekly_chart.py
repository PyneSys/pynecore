"""
@pyne
"""
from pynecore.lib import barmerge, close, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="Weekly Developing Batch", shorttitle="WDB")
def main():
    # A WEEKLY context on a DAILY chart: the calendar frontier path, where a
    # period's close comes from the trading calendar rather than from an
    # arithmetic span (a D/W/M chart bar has no fixed span, so ``chart_off`` is
    # 0 and the aggregator closes the period on the first bar of the next one).
    weekly: Series[float] = request.security(
        syminfo.tickerid, "W", close, lookahead=barmerge.lookahead_on)
    # The previous weekly close — ``na`` for the chart's whole first week.
    prev_weekly: Series[float] = request.security(
        syminfo.tickerid, "W", close[1], lookahead=barmerge.lookahead_on)
    # ``gaps_on`` on the same grid: a value only where a fresh week starts.
    gapped: Series[float] = request.security(
        syminfo.tickerid, "W", close, gaps=barmerge.gaps_on,
        lookahead=barmerge.lookahead_on)
    plot(weekly, "weekly")
    plot(prev_weekly, "prev_weekly")
    plot(gapped, "gapped")


# Every timestamp here is Unix MILLISECONDS.
# 2025-01-06 is a Monday, so the chart starts on a week boundary and the last
# bar below is a Sunday — a chart bar that closes its weekly period.
__test_helper_t0 = 1_736_121_600_000
__test_helper_day_ms = 86_400_000
__test_helper_weeks = 5
__test_helper_bars = __test_helper_weeks * 7


def __test_helper_write_feed(tmp_dir):
    """Write the weekly context's own feed, one bar per Monday.

    :param tmp_dir: Directory to write into.
    :return: Path to the written ``.ohlcv`` file, as a string.
    """
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "WDBW.ohlcv"
    with OHLCVWriter(path, "1W") as w:
        for week in range(__test_helper_weeks):
            c = 200.0 + week
            w.write(OHLCV(timestamp=__test_helper_t0 + week * 7 * __test_helper_day_ms,
                          open=c, high=c + 2.0, low=c - 2.0, close=c, volume=100.0))
    SymInfo(
        prefix="PYTEST", description="Weekly Dev Batch", ticker="TEST",
        currency="USD", period="1W", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_helper_chart_bars():
    """The chart's own daily bars.

    :return: The bar list.
    """
    from pynecore.types.ohlcv import OHLCV
    bars = []
    for i in range(__test_helper_bars):
        c = 150.0 + (i % 11) * 0.5
        bars.append(OHLCV(timestamp=__test_helper_t0 + i * __test_helper_day_ms,
                          open=c - 0.2, high=c + 1.0, low=c - 1.0, close=c,
                          volume=5.0 + i))
    return bars


def __test_helper_is_na(value):
    """Whether a read answered Pine ``na``, in either of its two shapes.

    :param value: The read value.
    :return: Whether it is ``na`` (an :class:`NA` marker or a float NaN).
    """
    from pynecore.types.na import NA
    return isinstance(value, NA) or (isinstance(value, float) and value != value)


def __test_helper_run(runner, no_batch):
    """Run the script once on a daily chart, with the batch rounds on or off.

    :param runner: The ``runner`` fixture.
    :param no_batch: Whether to force the per-bar round path.
    :return: ``(rows, batched_sids)``.
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
            feed = __test_helper_write_feed(Path(td))
            bars = __test_helper_chart_bars()
            r = runner(list(bars), {"period": "1D"}, security_data={"W": feed},
                       last_bar_index=len(bars) - 1,
                       last_bar_time=bars[-1].timestamp,
                       chart_bar_source=lambda: iter(bars))
            for _candle, pv in r.run_iter():
                rows.append(dict(pv))
    finally:
        multiprocessing.Process = original
        security_module.NO_BATCH = previous
    return rows, batched


def __test_weekly_batch_matches_per_bar_path__(runner, log):
    """A weekly batch on a daily chart answers exactly like the per-bar path.

    The daily/weekly/monthly shape takes a different frontier route than an
    intraday one: the chart bar has no fixed arithmetic span, so the period
    closes on the first chart bar of the next week and the round's as-of is
    extended to the end of the scheduled break it falls in. The planned
    sequence has to reproduce all of it, the chart's first (``na``) week and
    the last bar closing a week included.
    """
    batch_rows, batched = __test_helper_run(runner, no_batch=False)
    plain_rows, unbatched = __test_helper_run(runner, no_batch=True)

    assert batched, "no context ran as a developing batch"
    assert not unbatched, f"batch planned with the switch off: {unbatched}"
    assert len(batch_rows) == len(plain_rows) == __test_helper_bars

    for i, (got, want) in enumerate(zip(batch_rows, plain_rows)):
        assert got.keys() == want.keys(), f"bar {i}: plot columns differ"
        for key in want:
            same = (__test_helper_is_na(got[key]) and __test_helper_is_na(want[key])
                    ) or got[key] == want[key]
            assert same, (
                f"bar {i} '{key}': batch={got[key]!r} per-bar={want[key]!r}")

    # The point of the fixture: the first week really is ``na`` and later weeks
    # really are not, so the comparison above is not comparing ``na`` to ``na``.
    assert __test_helper_is_na(batch_rows[0]["prev_weekly"]), "first week not na"
    assert not __test_helper_is_na(batch_rows[-1]["weekly"]), "last week stayed na"

    log.info("weekly developing batch matched the per-bar path on %d daily bars, "
             "%d contexts batched", len(batch_rows), len(batched))
