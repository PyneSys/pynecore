"""
@pyne
"""
from pynecore.lib import close, high, plot, request, script, syminfo, ta
from pynecore.types import Series


@script.indicator(title="Merge Member Frontier", shorttitle="MMF")
def main():
    # Two calls on ONE context, so the transformer groups them and one child
    # serves both. ``lead`` signals first and becomes the group's primary.
    lead: Series[float] = request.security(syminfo.tickerid, "240", close)
    member: Series[float] = request.security(syminfo.tickerid, "240", high)
    # A context OUTSIDE the group, on a FINER timeframe, that consumes the
    # NON-primary member: its child pairs ``member`` out of ``member``'s ring
    # and waits on that ring's frontier for ITS own hourly bar. Three hours out
    # of four the chart closes no fresh 4-hour period and runs no round for the
    # group at all — so the chart itself has to publish how far every member is
    # settled, or this consumer parks on a ring nothing moves while the chart is
    # parked on this consumer.
    consumer: Series[float] = request.security(syminfo.tickerid, "60",
                                               ta.sma(member, 2))
    plot(lead, "lead")
    plot(member, "member")
    plot(consumer, "consumer")


# Every timestamp here is Unix MILLISECONDS.
__test_helper_t0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, on the 4h/1h/5m grids
__test_helper_hour = 3_600_000
__test_helper_step = 300_000  # 5 minutes
__test_helper_hours = 24
__test_helper_bars = __test_helper_hours * 12


def __test_helper_write_feed(tmp_dir, name, timeframe, span_ms):
    """Write one security feed covering the chart's whole range.

    :param tmp_dir: Directory to write into.
    :param name: File stem.
    :param timeframe: The feed's timeframe string.
    :param span_ms: One bar's span in milliseconds.
    :return: Path to the written ``.ohlcv`` file, as a string.
    """
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / f"{name}.ohlcv"
    count = (__test_helper_hours * __test_helper_hour) // span_ms
    with OHLCVWriter(path, timeframe) as w:
        for i in range(count):
            c = 100.0 + (i % 13) * 1.5
            w.write(OHLCV(timestamp=__test_helper_t0 + i * span_ms,
                          open=c, high=c + 2.0, low=c - 2.0, close=c, volume=1.0))
    SymInfo(
        prefix="PYTEST", description="Merge Member Frontier", ticker="TEST",
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
        c = 50.0 + (i % 17) * 0.5
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


def __test_helper_run(runner, no_merge):
    """Run the script once, recording which sids each child process serves.

    :param runner: The ``runner`` fixture.
    :param no_merge: Whether to force one child process per context.
    :return: ``(rows, spawned)``, the plot values per chart bar and one tuple of
        served sids per child process.
    """
    import multiprocessing
    import os
    import sys
    import tempfile
    from pathlib import Path

    sys.modules.pop(Path(__file__).stem, None)

    spawned: list = []
    original = multiprocessing.Process

    def _recording_process(*args, **kwargs):
        sec_args = kwargs.get('args') or ()
        if sec_args:
            spawned.append(tuple(sec_args[0]))
        return original(*args, **kwargs)

    rows: list[dict] = []
    multiprocessing.Process = _recording_process
    if no_merge:
        os.environ['PYNE_NO_SECURITY_MERGE'] = '1'
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            feeds = {
                "60": __test_helper_write_feed(tmp, "MMF60", "60",
                                               __test_helper_hour),
                "240": __test_helper_write_feed(tmp, "MMF240", "240",
                                                4 * __test_helper_hour),
            }
            r = runner(__test_helper_chart_bars(), security_data=feeds)
            for _candle, pv in r.run_iter():
                rows.append(dict(pv))
    finally:
        multiprocessing.Process = original
        if no_merge:
            os.environ.pop('PYNE_NO_SECURITY_MERGE', None)
    return rows, spawned


def __test_helper_run_with_timeout(fn, seconds=60):
    """Run ``fn`` on a daemon thread; a deadlock fails the test instead of hanging.

    :param fn: The callable to run.
    :param seconds: How long to wait for it. A passing run takes under a
        second, so this only bounds how long a regression costs.
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


def __test_a_groups_non_primary_member_publishes_its_frontier__(runner, log):
    """A consumer of a merged group's NON-primary member never parks.

    On a chart bar that closes no fresh 4-hour period the chart launches no
    round for the group: the child publishes nothing, so the chart itself has to
    raise the producers' ring frontiers. Doing that only for the primary — the
    one context whose slot the chart schedules — leaves the other members'
    frontiers where the last round left them, and the outside consumer of such a
    member waits for an instant nothing will ever reach while the chart is
    waiting for that consumer's value.

    Reproduces the shape that hung the Traders Reality corpus script (11 call
    sites over BTCUSDT D/W/M): four children parked on the non-primary member's
    ring while its own child sat idle.
    """
    merged_rows, merged = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, no_merge=False))
    plain_rows, plain = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, no_merge=True))

    assert sorted(len(child) for child in merged) == [1, 2], \
        f"unexpected grouping: {merged}"
    assert sorted(len(child) for child in plain) == [1, 1, 1], \
        f"unmerged children: {plain}"

    assert len(merged_rows) == len(plain_rows) == __test_helper_bars
    for i in range(len(merged_rows)):
        got = merged_rows[i]
        want = plain_rows[i]
        assert got.keys() == want.keys(), f"bar {i}: plot columns differ"
        for key in want:
            assert __test_helper_same(got[key], want[key]), (
                f"bar {i} '{key}': merged={got[key]!r} unmerged={want[key]!r}")

    # The consumer really did read the member: an all-``na`` column would make
    # the comparison above vacuous.
    assert any(not __test_helper_is_na(row["consumer"]) for row in merged_rows), \
        "the outside consumer never resolved a value"

    log.info("%d bars x %d plots identical; group of 2 plus its outside consumer",
             len(merged_rows), len(merged_rows[0]))
