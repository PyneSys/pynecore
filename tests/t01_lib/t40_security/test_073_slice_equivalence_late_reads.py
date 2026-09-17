"""
@pyne
"""
from pynecore.lib import close, nz, plot, request, script, syminfo
from pynecore.types import Persistent, Series


@script.indicator(title="Slice Late Reads", shorttitle="SLR")
def main():
    prev: Persistent[float] = 1.0
    # A BACK-EDGE consumer: its expression reads a value produced by a context
    # whose own write block stands LATER in ``main()``. The edge therefore lands
    # in ``late_reads`` instead of ``depends`` — the child cannot wait for it, so
    # the peer read answers ``na`` there — but the read itself is MANDATORY and
    # no slice may drop it. ``nz`` keeps the rest of the expression alive so the
    # column still moves and the comparison is not over a constant.
    late: Series[float] = request.security(
        "EXCH:LATE", "60", nz(prev) * 2.0 + close * 0.01)
    producer: Series[float] = request.security(syminfo.tickerid, "60", close)
    # A plain forward dependency: this one's expression reads a context that
    # already wrote, so the edge is a ``depends``.
    forward: Series[float] = request.security("EXCH:FWD", "60", producer + 5.0)
    # The tail that closes the back edge. Drop it and the late consumer freezes
    # on its seed value from the first bar on.
    prev = producer
    plot(late, "late")
    plot(producer, "producer")
    plot(forward, "forward")
    plot(prev, "prev")


# A side script for the missing-feed case: the context is resolved at RUNTIME
# from an input, read on every bar, and never provisioned.
__test_helper_missing_feed_script = '''"""
@pyne
"""
from pynecore.lib import close, input, plot, request, script, syminfo
from pynecore.types import Persistent, Series


@script.indicator(title="Slice Missing Feed", shorttitle="SMF")
def main(htf=input.timeframe(defval="60", title="HTF")):
    acc: Persistent[float] = 0.0
    x: Series[float] = request.security(syminfo.tickerid, htf, close * 2.0 + acc)
    acc = acc + 1.0
    plot(x, "x")
'''

# Every timestamp here is Unix MILLISECONDS.
__test_helper_t0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, on the 1h and 5m grids
__test_helper_hour = 3_600_000
__test_helper_step = 300_000  # 5 minutes
__test_helper_hours = 24
__test_helper_bars = __test_helper_hours * 12


def __test_helper_write_feed(tmp_dir, prefix, ticker, base):
    """Write one hourly feed.

    :param tmp_dir: Directory to write into.
    :param prefix: Exchange prefix.
    :param ticker: Ticker name.
    :param base: First close; the feed walks up from here.
    :return: Path to the written ``.ohlcv`` file, as a string.
    """
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / f"{prefix}_{ticker}60.ohlcv"
    with OHLCVWriter(path, "60") as w:
        for hour in range(__test_helper_hours):
            c = base + (hour % 11) * 1.25
            w.write(OHLCV(timestamp=__test_helper_t0 + hour * __test_helper_hour,
                          open=c, high=c + 1.0, low=c - 1.0, close=c, volume=1.0))
    SymInfo(
        prefix=prefix, description="Slice Late Reads", ticker=ticker,
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
    """The chart's own 5-minute bars, with a moving close.

    :return: The bar list.
    """
    from pynecore.types.ohlcv import OHLCV
    bars = []
    for i in range(__test_helper_bars):
        c = 70.0 + (i % 19) * 0.5
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


def __test_helper_set_slice_mode(no_slice):
    """Arm ``PYNE_NO_SECURITY_SLICE`` and drop everything cached for the old mode.

    The switch is read while the module is TRANSFORMED, so flipping it only
    takes effect on a fresh import: the script module goes out of
    ``sys.modules``, its cached bytecode is dropped, and the pipeline digest —
    memoized per process — is recomputed so a ``.pyc`` produced in the other
    mode can never be accepted.

    :param no_slice: Whether slicing must be disabled for the next import.
    """
    import importlib.util
    import os
    import sys
    from pathlib import Path

    import pynecore.core.import_hook as import_hook

    sys.modules.pop(Path(__file__).stem, None)
    sys.modules.pop("pynecore.transformers.security_slice", None)
    import_hook._transform_pipeline_hash = None
    try:
        Path(importlib.util.cache_from_source(__file__)).unlink()
    except OSError:
        pass
    os.environ['PYNE_SAVE_SCRIPT_TOML'] = '0'
    if no_slice:
        os.environ['PYNE_NO_SECURITY_SLICE'] = '1'
    else:
        os.environ.pop('PYNE_NO_SECURITY_SLICE', None)


def __test_helper_run(runner, no_slice):
    """Run the script once in the given slicing mode.

    :param runner: The ``runner`` fixture.
    :param no_slice: Whether to force the whole-``main()`` child path.
    :return: ``(rows, error)`` — plot values per chart bar and the ``ValueError``
        the run stopped on, if any.
    """
    import os
    import tempfile
    from pathlib import Path

    __test_helper_set_slice_mode(no_slice)
    rows = []
    error = None
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            feeds = {
                "60": __test_helper_write_feed(tmp, "PYTEST", "TEST", 100.0),
                "EXCH:LATE:60": __test_helper_write_feed(tmp, "EXCH", "LATE", 200.0),
                "EXCH:FWD:60": __test_helper_write_feed(tmp, "EXCH", "FWD", 300.0),
            }
            r = runner(__test_helper_chart_bars(), security_data=feeds)
            try:
                for _candle, pv in r.run_iter():
                    rows.append(dict(pv))
            except ValueError as exc:
                error = exc
    finally:
        os.environ.pop('PYNE_NO_SECURITY_SLICE', None)
    return rows, error


def __test_helper_run_with_timeout(fn, seconds=180):
    """Run ``fn`` on a daemon thread; a deadlock fails the test instead of hanging."""
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


def __test_helper_compare(sliced, plain, log, what):
    """Assert two runs produced bit-identical plot rows.

    :param sliced: Rows from the sliced run.
    :param plain: Rows from the ``PYNE_NO_SECURITY_SLICE=1`` run.
    :param log: The ``log`` fixture.
    :param what: Name for the log line.
    """
    assert len(sliced) == len(plain), \
        f"{what}: bar counts differ ({len(sliced)} vs {len(plain)})"
    assert sliced, f"{what}: the run produced no bars"
    for i in range(len(sliced)):
        got = sliced[i]
        want = plain[i]
        assert got.keys() == want.keys(), f"{what} bar {i}: plot columns differ"
        for key in want:
            assert __test_helper_same(got[key], want[key]), (
                f"{what} bar {i} '{key}': sliced={got[key]!r} "
                f"whole-main={want[key]!r}")
    log.info("%s: %d bars x %d plots identical in both slicing modes",
             what, len(sliced), len(sliced[0]))


def __test_slice_keeps_both_peer_edge_directions__(runner, log):
    """A ``depends`` edge and a ``late_reads`` back edge both survive slicing

    One consumer stands AFTER its producer (a forward ``depends``) and one
    stands BEFORE it, reading the producer's value through a ``var`` the tail
    assigns (a ``late_reads`` back edge). Both peer reads are mandatory in every
    child, and the tail statement that closes the back edge is an input of the
    next bar — so a slice may drop neither.
    """
    sliced, sliced_error = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, False))
    plain, plain_error = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, True))
    assert sliced_error is None, f"the sliced run raised: {sliced_error}"
    assert plain_error is None, f"the whole-main run raised: {plain_error}"
    __test_helper_compare(sliced, plain, log, "depends + late_reads")


def __test_peer_edges_are_not_degenerate__(runner, log):
    """Both peer columns actually carry moving values

    The back edge in particular would look fine while frozen: if the tail
    assignment were dropped the late consumer would publish one constant. The
    scenario pins that all three contexts publish and that the back-edge column
    moves with its producer.
    """
    rows, error = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, False))
    assert error is None, f"the run raised: {error}"

    for key in ("late", "producer", "forward"):
        values = []
        for row in rows:
            v = row.get(key)
            if not __test_helper_is_na(v):
                values.append(v)
        assert len(values) >= len(rows) // 2, \
            f"'{key}': only {len(values)} of {len(rows)} bars carry a value"
        assert len(set(values)) > 3, \
            f"'{key}' is frozen: {sorted(set(values))}"
    assert rows[-1]["prev"] == rows[-1]["producer"], \
        "the tail did not carry the producer into the back edge"

    log.info("all three contexts publish moving values over %d bars", len(rows))


def __test_helper_run_missing_feed(runner, no_slice):
    """Run the unprovisioned side script once in the given slicing mode.

    A separate script, because the context has to be RUNTIME-RESOLVED for the
    read to reach the raising path: a statically named symbol with no data is
    reported as an unprovisioned context and answers ``na``, while a resolved
    one that is actually read has nothing to answer with and raises. This is the
    same shape as the taken-branch case in ``test_067``.

    :param runner: The ``runner`` fixture, used for its chart symbol info.
    :param no_slice: Whether to force the whole-``main()`` child path.
    :return: ``(bars, error)`` — the bars that completed and the ``ValueError``.
    """
    import os
    import sys
    import tempfile
    from pathlib import Path

    from pynecore.core.script_runner import ScriptRunner

    __test_helper_set_slice_mode(no_slice)
    done = 0
    error = None
    try:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "slice_missing_feed_reference.py"
            path.write_text(__test_helper_missing_feed_script)
            chart = runner(__test_helper_chart_bars())
            sys.modules.pop(path.stem, None)
            sys.path.insert(0, str(td))
            try:
                sr = ScriptRunner(path, iter(__test_helper_chart_bars()),
                                  chart.syminfo, security_data={})
                try:
                    for _candle, _pv in sr.run_iter():
                        done += 1
                except ValueError as exc:
                    error = exc
            finally:
                sys.path.remove(str(td))
                sys.modules.pop(path.stem, None)
    finally:
        os.environ.pop('PYNE_NO_SECURITY_SLICE', None)
    return done, error


def __test_missing_feed_still_raises_with_slicing__(runner, log):
    """A read of an unprovisioned resolved context raises in both modes

    Slicing may only remove statements the expression cannot depend on; the
    ``__sec_read__`` that fetches a context's own value never qualifies. With no
    data behind the resolved context the run has to stop with the same
    ``ValueError`` whether the child runs a slice or the whole ``main()`` — a
    dropped read would turn the failure into a quiet ``na``.
    """
    sliced_done, sliced_error = __test_helper_run_with_timeout(
        lambda: __test_helper_run_missing_feed(runner, False))
    plain_done, plain_error = __test_helper_run_with_timeout(
        lambda: __test_helper_run_missing_feed(runner, True))

    assert sliced_error is not None, "the sliced run swallowed the missing feed"
    assert plain_error is not None, "the whole-main run swallowed the missing feed"
    for error in (sliced_error, plain_error):
        assert "No OHLCV data found for security context" in str(error), \
            f"unexpected error message: {error}"
        assert "PYTEST:TEST" in str(error), \
            f"the error does not name the symbol: {error}"
    assert sliced_done == plain_done, (
        f"raised at different bars: sliced={sliced_done} whole-main={plain_done}")

    log.info("the unprovisioned read raised at bar %d in both modes", sliced_done)
