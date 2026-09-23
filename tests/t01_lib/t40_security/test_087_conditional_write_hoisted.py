"""
@pyne
"""
from pynecore.lib import (
    array, bar_index, close, high, input, low, na, open, plot, request, script, syminfo,
)
from pynecore.types import Series


def gated_read():
    """A security call standing in a helper the caller reaches conditionally."""
    return request.security(syminfo.tickerid, "60", close * 2.0)


def ternary_read():
    """A security call standing in a helper only one ternary branch calls."""
    return request.security(syminfo.tickerid, "60", close * 4.0)


def bool_read():
    """A security call standing in a helper called behind a short circuit."""
    return request.security(syminfo.tickerid, "60", close * 8.0 > open * 8.0)


def guarded_inside(flag):
    """A helper every bar calls, with the guard around the call INSIDE it."""
    value = na(float)
    if flag:
        value = request.security(syminfo.tickerid, "60", close * 32.0)
    return value


def forwarded_read(exp, use, res):
    """A security call on an argument, in a helper each branch statement calls."""
    value = request.security(syminfo.tickerid, res, exp)
    return value if use else exp


@script.indicator(title="Conditional Security Write", shorttitle="CSW")
def main(use_res=input.bool(True, "Use Alt Res"), res=input.timeframe("60", "Alt Res"),
         res_first=input.timeframe("120", "First Res")):
    # TradingView hoists a ``request.security()`` call to global scope: the
    # branch it is written in decides where its RESULT lands, never whether the
    # requested series is computed. The gate below is a function of
    # ``bar_index``, so a child that re-evaluated it against ITS OWN bars would
    # publish on a different set of bars than the chart asks about — and the
    # gated reads would fall a whole security period behind the ungated ones.
    #
    # Every gated expression is a power-of-two multiple of its reference, so the
    # two are distinct contexts whose values still compare exactly.
    ref_close: Series[float] = request.security(syminfo.tickerid, "60", close)
    ref_high: Series[float] = request.security(syminfo.tickerid, "60", high)
    ref_low: Series[float] = request.security(syminfo.tickerid, "60", low)
    ref_up = request.security(syminfo.tickerid, "60", close > open)
    ref_ltf = request.security_lower_tf("EXCH:LTFSYM", "1", close)

    gate = bar_index % 3 != 1

    in_if: Series[float] = 0.0
    in_if_fn: Series[float] = 0.0
    if gate:
        in_if = request.security(syminfo.tickerid, "60", close * 16.0)
        in_if_fn = gated_read()
    else:
        in_if = in_if[1]
        in_if_fn = in_if_fn[1]

    # One context in each branch: both series are computed on every bar
    either = 0.0
    if gate:
        either = request.security(syminfo.tickerid, "60", high * 2.0)
    else:
        either = request.security(syminfo.tickerid, "60", low * 2.0)

    in_ternary = ternary_read() if gate else na(float)
    in_helper = guarded_inside(gate)
    in_bool = gate and bool_read()

    in_ltf = array.new_float(0)
    if gate:
        in_ltf = request.security_lower_tf("EXCH:LTFSYM", "1", close * 2.0)

    # Two writing statements in one branch, both handing input values to the
    # helper, on two feeds so each context gets a clone of its own: the first
    # statement's call must not pin the second one under the guard in the
    # clone serving the second context.
    first_of_two: Series[float] = 0.0
    second_of_two: Series[float] = 0.0
    if gate:
        first_of_two = forwarded_read(close * 64.0, use_res, res_first)
        second_of_two = forwarded_read(close * 128.0, use_res, res)
    else:
        first_of_two = first_of_two[1]
        second_of_two = second_of_two[1]

    plot(gate, "gate")
    plot(ref_close, "ref_close")
    plot(ref_high, "ref_high")
    plot(ref_low, "ref_low")
    plot(1.0 if ref_up else 0.0, "ref_up")
    plot(array.sum(ref_ltf), "ref_ltf")
    plot(in_if / 16.0, "in_if")
    plot(in_if_fn / 2.0, "in_if_fn")
    plot(either / 2.0, "either")
    plot(in_ternary / 4.0, "in_ternary")
    plot(in_helper / 32.0, "in_helper")
    plot(1.0 if in_bool else 0.0, "in_bool")
    plot(array.sum(in_ltf) / 2.0, "in_ltf")
    plot(first_of_two / 64.0, "first_of_two")
    plot(second_of_two / 128.0, "second_of_two")


# Every timestamp here is Unix MILLISECONDS.
__test_helper_t0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, on the 1h, 5m and 1m grids
__test_helper_hour = 3_600_000
__test_helper_step = 300_000  # 5 minutes
__test_helper_minute = 60_000
__test_helper_hours = 30
__test_helper_bars = __test_helper_hours * 12


def __test_helper_write_syminfo(path, prefix, ticker, period):
    """Write the 24/7 UTC ``.toml`` sidecar a security child loads on startup."""
    from datetime import time
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession

    SymInfo(
        prefix=prefix, description="Conditional Security Write", ticker=ticker,
        currency="USD", period=period, type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))


def __test_helper_write_feeds(tmp_dir):
    """Write the ``60`` and ``120`` feeds and the 1-minute feed the contexts read.

    :param tmp_dir: Directory to write into.
    :return: The ``security_data`` mapping for the runner.
    """
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.types.ohlcv import OHLCV

    htf = tmp_dir / "CSW60.ohlcv"
    with OHLCVWriter(htf, "60") as w:
        for hour in range(__test_helper_hours):
            c = 100.0 + (hour % 19) * 1.25
            o = c - 0.5 if hour % 4 else c + 0.5
            w.write(OHLCV(timestamp=__test_helper_t0 + hour * __test_helper_hour,
                          open=o, high=c + 1.0 + hour % 5, low=c - 1.0 - hour % 3,
                          close=c, volume=1.0))
    __test_helper_write_syminfo(htf, "PYTEST", "TEST", "60")

    htf2 = tmp_dir / "CSW120.ohlcv"
    with OHLCVWriter(htf2, "120") as w:
        for block in range(__test_helper_hours // 2):
            c = 200.0 + (block % 7) * 2.5
            w.write(OHLCV(timestamp=__test_helper_t0 + block * 2 * __test_helper_hour,
                          open=c, high=c + 1.0, low=c - 1.0, close=c, volume=1.0))
    __test_helper_write_syminfo(htf2, "PYTEST", "TEST", "120")

    ltf = tmp_dir / "EXCH_LTFSYM_1.ohlcv"
    with OHLCVWriter(ltf, "1") as w:
        for minute in range(__test_helper_bars * 5):
            c = 10.0 + (minute % 31) * 0.25
            w.write(OHLCV(timestamp=__test_helper_t0 + minute * __test_helper_minute,
                          open=c, high=c, low=c, close=c, volume=1.0))
    __test_helper_write_syminfo(ltf, "EXCH", "LTFSYM", "1")
    return {"60": str(htf), "120": str(htf2), "EXCH:LTFSYM:1": str(ltf)}


def __test_helper_chart_bars():
    """The chart's own 5-minute bars, with a moving close.

    :return: The bar list.
    """
    from pynecore.types.ohlcv import OHLCV
    bars = []
    for i in range(__test_helper_bars):
        c = 50.0 + (i % 23) * 0.5
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


def __test_helper_run(runner, no_slice=False):
    """Run the script once and collect its plot rows.

    :param runner: The ``runner`` fixture.
    :param no_slice: Whether the backward slicing is switched off for the run.
    :return: The plot values per chart bar.
    """
    import os
    import tempfile
    from pathlib import Path

    __test_helper_set_slice_mode(no_slice)
    rows = []
    try:
        with tempfile.TemporaryDirectory() as td:
            feeds = __test_helper_write_feeds(Path(td))
            r = runner(__test_helper_chart_bars(), security_data=feeds)
            for _candle, pv in r.run_iter():
                rows.append(dict(pv))
    finally:
        os.environ.pop('PYNE_NO_SECURITY_SLICE', None)
    return rows


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


# The gated column and the ungated reference it must equal while the gate is open.
__test_helper_open_pairs = (
    ("in_if", "ref_close"), ("in_if_fn", "ref_close"), ("either", "ref_high"),
    ("in_ternary", "ref_close"), ("in_helper", "ref_close"),
    ("in_bool", "ref_up"), ("in_ltf", "ref_ltf"),
    ("second_of_two", "ref_close"),
)


def __test_helper_assert_hoisted(rows):
    """Assert every gated read equals its ungated reference.

    :param rows: The plot values per chart bar.
    :return: How many open-gate and closed-gate bars were compared.
    """
    opened = 0
    closed = 0
    for i in range(len(rows)):
        row = rows[i]
        if __test_helper_is_na(row["ref_close"]):
            continue
        if row["gate"]:
            opened += 1
            for gated, ref in __test_helper_open_pairs:
                assert row[gated] == row[ref], (
                    f"bar {i}: {gated}={row[gated]!r} {ref}={row[ref]!r}")
        else:
            closed += 1
            assert row["either"] == row["ref_low"], (
                f"bar {i}: either={row['either']!r} ref_low={row['ref_low']!r}")
    assert opened > 0 and closed > 0, "the gate never took one of its two values"
    return opened, closed


def __test_conditional_call_reads_the_current_period__(runner, log):
    """A gated ``request.security`` answers what an ungated one answers

    On every bar the gate is open the two reads are of the same context at the
    same instant, so they must be the same value — for a call in an ``if``, in
    either branch of an ``if`` / ``else`` holding a context each, through a
    helper called from a branch, from one side of a ternary or behind ``and``,
    through a helper holding the guard itself, for two helper calls in one
    branch that each write their own context, and for
    ``request.security_lower_tf``. A child that carried the chart's
    guard into its own bar loop would answer with the previous security period
    on a third of them.
    """
    rows = __test_helper_run_with_timeout(lambda: __test_helper_run(runner))
    opened, closed = __test_helper_assert_hoisted(rows)
    log.info("%d open-gate and %d closed-gate bars agree with the ungated reads",
             opened, closed)


def __test_conditional_call_is_hoisted_without_the_slice__(runner, log):
    """The same holds while ``PYNE_NO_SECURITY_SLICE`` switches the slicing off

    Forcing the write is a matter of what the child computes, so it may not
    depend on the optimization that usually carries it.
    """
    rows = __test_helper_run_with_timeout(
        lambda: __test_helper_run(runner, no_slice=True))
    opened, closed = __test_helper_assert_hoisted(rows)
    log.info("%d open-gate and %d closed-gate bars agree with slicing off",
             opened, closed)


def __test_conditional_scenario_is_not_degenerate__(runner, log):
    """The compared contexts move, and the gate really closes

    An equivalence assertion over a constant column would pass whatever the
    child published, so the scenario itself is pinned here: the gate has to
    close on a third of the bars and every reference has to take many distinct
    values across them.
    """
    rows = __test_helper_run_with_timeout(lambda: __test_helper_run(runner))

    closed = 0
    for row in rows:
        if not row["gate"]:
            closed += 1
    assert closed >= len(rows) // 4, f"the gate only closed on {closed} bars"
    for ref, least in (("ref_close", 10), ("ref_high", 10), ("ref_low", 10),
                       ("ref_up", 2), ("ref_ltf", 10)):
        values = set()
        for row in rows:
            if not __test_helper_is_na(row[ref]):
                values.add(row[ref])
        assert len(values) >= least, f"degenerate {ref} column: {len(values)} values"

    log.info("gate closed on %d of %d bars", closed, len(rows))
