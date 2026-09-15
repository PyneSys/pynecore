"""
@pyne
"""
from pynecore.lib import (
    bar_index, close, high, input, na, plot, request, script, syminfo, ta
)
from pynecore.types import Series


@script.indicator(title="Lazy Security Start", shorttitle="LSS")
def main(matype=input.string(defval='SMA', title='Type', options=('SMA', 'EMA'))):
    # The branch the run never takes: its ``request.security()`` is signalled on
    # every bar (the transformer hoists the signal) but never read, so it must
    # never get a child process.
    branch: Series[float] = (
        request.security(syminfo.tickerid, "60", close) if matype == 'SMA'
        else request.security(syminfo.tickerid, "60", high)
    )
    # Two identical requests, one read from the first bar and one only later:
    # from that bar on they must agree to the last bit.
    eager: Series[float] = request.security(syminfo.tickerid, "60", close)
    late: Series[float] = (request.security(syminfo.tickerid, "60", close)
                           if bar_index >= _LATE_FROM else na)
    # A dependent pair. ``producer`` is read from the first bar, so the late
    # consumer has to pair itself against a producer that has been running all
    # along — off its ring, for every one of the consumer's own bars.
    producer: Series[float] = request.security(syminfo.tickerid, "60", close)
    consumer: Series[float] = (
        request.security(syminfo.tickerid, "60", ta.sma(producer, 2))
        if bar_index >= _LATE_FROM else na
    )
    # The same pair the other way round: this producer's own read comes much
    # later than its consumer's, so the consumer's start is what has to start
    # it — and its own values must still be right once the chart reads it.
    late_producer: Series[float] = (request.security(syminfo.tickerid, "60", close)
                                    if bar_index >= _PROD_FROM else na)
    late_consumer: Series[float] = (
        request.security(syminfo.tickerid, "60", ta.sma(late_producer, 2))
        if bar_index >= _LATE_FROM else na
    )
    plot(branch, "branch")
    plot(eager, "eager")
    plot(late, "late")
    plot(consumer, "consumer")
    plot(late_producer, "late_producer")
    plot(late_consumer, "late_consumer")


# Chart bar index from which the late contexts are read.
_LATE_FROM = 40
# Chart bar index from which ``late_producer`` is read.
_PROD_FROM = 80

# Every timestamp here is Unix MILLISECONDS.
_T0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, aligned to the 1h and 5m grids
_HOUR = 3_600_000
_CHART_STEP = 300_000  # 5 minutes
_N_HOURS = 12


def __test_helper_htf_close(hour):
    return 100.0 + hour


def __test_helper_write_feed(tmp_dir):
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "HTF60.ohlcv"
    with OHLCVWriter(path, "60") as w:
        for hour in range(_N_HOURS):
            c = __test_helper_htf_close(hour)
            w.write(OHLCV(timestamp=_T0 + hour * _HOUR, open=c,
                          high=1000.0 + hour, low=c, close=c, volume=1.0))
    SymInfo(
        prefix="EXCH", description="Lazy HTF", ticker="LSS",
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
    from pynecore.types.ohlcv import OHLCV
    return [OHLCV(timestamp=_T0 + i * _CHART_STEP, open=1.0, high=1.0, low=1.0,
                  close=1.0, volume=1.0)
            for i in range(_N_HOURS * 12)]


def __test_helper_same(a, b):
    """Pine-style equality for two read values: ``na`` matches ``na``.

    :param a: One value.
    :param b: The other.
    :return: Whether the two reads answered the same.
    """
    from pynecore.types.na import NA
    if isinstance(a, NA) or isinstance(b, NA):
        return isinstance(a, NA) and isinstance(b, NA)
    return a == b


def __test_helper_run(runner):
    """Run the script once, recording which sids got a child process.

    ``script_runner`` imports ``Process`` from :mod:`multiprocessing` inside the
    run, so patching the module attribute up front is what the spawn resolves —
    and the first entry of the spawn's ``args`` is the sid, which is all the
    test needs.

    :param runner: The ``runner`` fixture.
    :return: ``(rows, spawned_sids)``, rows keyed by chart bar index.
    """
    import multiprocessing
    import sys
    import tempfile
    from pathlib import Path

    sys.modules.pop(Path(__file__).stem, None)

    spawned: list[str] = []
    original = multiprocessing.Process

    def _recording_process(*args, **kwargs):
        sec_args = kwargs.get('args') or ()
        if sec_args:
            spawned.append(sec_args[0])
        return original(*args, **kwargs)

    rows = {}
    multiprocessing.Process = _recording_process
    try:
        with tempfile.TemporaryDirectory() as td:
            feed = __test_helper_write_feed(Path(td))
            r = runner(__test_helper_chart_bars(), security_data={"60": feed})
            for i, (_candle, pv) in enumerate(r.run_iter()):
                rows[i] = dict(pv)
    finally:
        multiprocessing.Process = original
    return rows, spawned


def __test_unread_branch_gets_no_process__(runner, log):
    """A security context in a branch this run never takes gets no child process.

    The transformer hoists every context's ``__sec_signal__`` to the top of
    ``main()``, so the untaken branch's context is signalled on every bar. Only
    a READ may start it: of the EIGHT contexts the script declares (both branch
    arms and the six named ones) the seven that are reached get a child, and the
    untaken arm gets none.
    """
    rows, spawned = __test_helper_run(runner)

    assert len(spawned) == 7, f"spawned children: {spawned}"
    assert len(set(spawned)) == 7, f"duplicate spawn: {spawned}"
    # The TAKEN arm is the same request as ``eager``, so it answers the same —
    # the missing child belongs to the other arm.
    for i in (0, 10, 100, 140):
        assert __test_helper_same(rows[i]["branch"], rows[i]["eager"]), \
            f"bar {i}: branch={rows[i]['branch']} != eager={rows[i]['eager']}"

    log.info("untaken branch context ran without a child process")


def __test_late_first_read_matches_eager__(runner, log):
    """A context first read at bar N answers exactly like an eagerly read one.

    ``late`` and ``eager`` are the same request; the first is read from bar
    ``_LATE_FROM`` on, the second from the first bar. The one late round the
    start runs brings the child up to the same target the per-bar rounds would
    have left it at, so every value from that bar on has to be identical.
    """
    rows, _spawned = __test_helper_run(runner)

    from pynecore.types.na import NA
    compared = 0
    for i, values in rows.items():
        if i < _LATE_FROM:
            continue
        eager = values["eager"]
        assert not isinstance(eager, NA), f"bar {i}: eager is na"
        assert __test_helper_same(values["late"], eager), \
            f"bar {i}: late={values['late']} != eager={eager}"
        compared += 1
    assert compared > 0

    log.info("late first read matched the eager context on %d bars", compared)


def __test_late_consumer_reads_producer_history__(runner, log):
    """A lazily started consumer pairs against its producer's whole history.

    ``consumer`` reads ``producer`` inside its own child, off the producer's
    ring. Its start is the first read at bar ``_LATE_FROM``, and the round that
    start runs replays every one of the consumer's own bars — so the producer's
    earlier entries have to be there, not just the current bar's.
    """
    rows, _spawned = __test_helper_run(runner)

    from pynecore.types.na import NA
    # ``ta.sma`` of the last two hourly closes. The chart's last CLOSED hourly
    # bar is ``hour - 1``, so the mean is ``close(hour - 1) - 0.5``.
    checked = 0
    for hour in range(_LATE_FROM // 12 + 1, _N_HOURS):
        consumer = rows[hour * 12]["consumer"]
        expected = __test_helper_htf_close(hour - 1) - 0.5
        assert not isinstance(consumer, NA) and abs(consumer - expected) < 1e-9, \
            f"hour {hour}: consumer={consumer} != {expected}"
        checked += 1
    assert checked > 0

    log.info("lazily started consumer resolved its producer on %d hourly bars", checked)


def __test_consumer_start_starts_its_producer__(runner, log):
    """A consumer's start starts the producer it depends on, values intact.

    ``late_consumer`` is read at bar ``_LATE_FROM`` and ``late_producer`` only
    at ``_PROD_FROM``, so the producer's child exists solely because the
    consumer's start brought it up. Once the chart reads it, it must answer
    exactly like the eagerly read context on the same request.
    """
    rows, _spawned = __test_helper_run(runner)

    from pynecore.types.na import NA
    compared = 0
    for i, values in rows.items():
        if i < _PROD_FROM:
            continue
        eager = values["eager"]
        assert not isinstance(eager, NA), f"bar {i}: eager is na"
        assert __test_helper_same(values["late_producer"], eager), \
            f"bar {i}: late_producer={values['late_producer']} != eager={eager}"
        compared += 1
    assert compared > 0

    log.info("consumer-started producer matched the eager context on %d bars", compared)
