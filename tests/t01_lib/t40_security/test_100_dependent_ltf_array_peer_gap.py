"""
@pyne
"""
from pynecore.lib import array, close, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="Dependent LTF Array Peer Gap", shorttitle="DLAPG")
def main():
    # The 10-minute contexts read the 1-minute array as a peer while the
    # 1-minute feed has holes: one whole 10-minute period has no intrabar at
    # all, another lacks three of its ten, and the feed ends halfway through
    # the last one. The chart's own read of the array is the non-peer
    # reference the peer read has to agree with.
    v1 = request.security_lower_tf(syminfo.tickerid, "1", close)
    n: Series[float] = request.security(syminfo.tickerid, "10", array.size(v1))
    s: Series[float] = request.security(syminfo.tickerid, "10", array.sum(v1))
    plot(array.size(v1), "chart n")
    plot(array.sum(v1), "chart s")
    plot(n, "n")
    plot(s, "s")


# Every timestamp here is Unix MILLISECONDS.
__test_helper_t0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, on every minute grid
__test_helper_minute = 60_000
__test_helper_feed_minutes = 40
# Minutes absent from the 1-minute feed: the whole 10-minute period 10..19,
# three intrabars of the period 20..29, and the feed's tail from minute 35 on.
__test_helper_missing = frozenset(range(10, 20)) | {25, 26, 27} | frozenset(range(35, 40))


def __test_helper_syminfo(period):
    from datetime import time
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    return SymInfo(
        prefix="EXCH", description="Dependent LTF peer gap", ticker="BASE",
        currency="USD", period=period, type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    )


def __test_helper_write_feed(tmp_dir, name, span_min, skip):
    """Write a feed at ``span_min`` minutes whose every price is the bar's open
    offset in minutes, leaving out the bars opening at the minutes in ``skip``,
    plus a 24/7 UTC ``.toml`` sidecar."""
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / f"{name}.ohlcv"
    with OHLCVWriter(path, str(span_min)) as w:
        for m in range(0, __test_helper_feed_minutes, span_min):
            if m in skip:
                continue
            v = float(m)
            w.write(OHLCV(timestamp=__test_helper_t0 + m * __test_helper_minute,
                          open=v, high=v, low=v, close=v, volume=1.0))
    __test_helper_syminfo(str(span_min)).save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_helper_chart_bars(chart_tf_min):
    from pynecore.types.ohlcv import OHLCV
    return [OHLCV(timestamp=__test_helper_t0 + m * __test_helper_minute,
                  open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0)
            for m in range(0, __test_helper_feed_minutes, chart_tf_min)]


def __test_helper_run_with_timeout(fn, seconds=120):
    """Run ``fn`` on a daemon thread; a hang fails the test instead of stalling it."""
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
        raise AssertionError(f"hang: the run did not finish within {seconds}s")
    if 'error' in box:
        raise box['error']
    return box.get('result')


def __test_helper_run(runner, chart_tf_min):
    """Run the script on a ``chart_tf_min``-minute chart, bounded in time.

    :return: Per chart bar, ``(chart n, chart s, n, s)`` with ``na`` as ``None``.
    """
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.types.na import NA

    def num(v):
        return None if v is None or isinstance(v, NA) or v != v else float(v)

    def scenario():
        sys.modules.pop(Path(__file__).stem, None)
        rows = []
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ltf = __test_helper_write_feed(tmp, "LTF1", 1, __test_helper_missing)
            htf = __test_helper_write_feed(tmp, "HTF10", 10, frozenset())
            r = runner(__test_helper_chart_bars(chart_tf_min),
                       syminfo_override={"period": str(chart_tf_min)},
                       security_data={"1": ltf, "10": htf})
            # ``run_iter`` reuses the plot dict, so copy each bar's values at once.
            for _candle, pv in r.run_iter():
                rows.append(tuple(num(pv.get(k)) for k in ("chart n", "chart s", "n", "s")))
        return rows

    return __test_helper_run_with_timeout(scenario)


def __test_ltf_array_peer_across_a_feed_gap__(runner, log):
    """A lower-timeframe array peer answers across a feed gap instead of hanging.

    The 10-minute consumer's bar 10..19 has no intrabar in the 1-minute feed,
    and its bar 30..39 outlasts the feed. Its read waits until the producer has
    published everything closing at or before its as-of, and the producer's
    rounds over a hole run no feed bar, so the wait must still be released.
    The answer is what the feed holds for the consumer's own period — nothing
    for 10..19, seven intrabars for 20..29, five for 30..39 — exactly what the
    chart's own (non-peer) read gives on a 10-minute chart.
    """
    reference = __test_helper_run(runner, 10)
    # The non-peer read on a 10-minute chart: each chart bar's own intrabars.
    assert [r[:2] for r in reference] == [
        (10.0, 45.0),    # minutes 0..9
        (0.0, None),     # the hole: an empty array, whose sum is na
        (7.0, 167.0),    # minutes 20..29 without 25, 26, 27
        (5.0, 160.0),    # minutes 30..34, where the feed ends
    ], f"non-peer reference: {reference}"

    rows = __test_helper_run(runner, 5)
    assert len(rows) == 8, f"bar count {len(rows)} != 8"
    # The chart's own read on the 5-minute chart: the five intrabars of each
    # chart bar, none over a hole, two in 25..29.
    assert [r[:2] for r in rows] == [
        (5.0, 10.0), (5.0, 35.0), (0.0, None), (0.0, None),
        (5.0, 110.0), (2.0, 57.0), (5.0, 160.0), (0.0, None),
    ], f"chart reads: {[r[:2] for r in rows]}"
    # The consumer confirms each 10-minute bar on the chart bar reaching its
    # close and holds it (``gaps_off``) until the next one lands; each value is
    # the non-peer read of that same period.
    expected = [(None, None)]
    for period in reference:
        expected.extend([period[:2], period[:2]])
    expected = expected[:8]
    peer = [r[2:] for r in rows]
    assert peer == expected, f"peer reads: {peer} != {expected}"
    log.info("a lower-timeframe array peer answers across a feed gap")
