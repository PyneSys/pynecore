"""
@pyne
"""
import pytest
from pynecore.lib import array, close, matrix, plot, request, script, syminfo, ta, timeframe
from pynecore.types import Matrix, Persistent


@script.indicator(title="LTF Slice", shorttitle="LTFS")
def main():
    # Chart-side state the context does not depend on: the row exists only on
    # the chart's own 5-minute bars, so reading it on the context's 1-minute
    # bars is out of range. TradingView never evaluates it in the context.
    boxes: Persistent[Matrix[float]] = matrix.new(0, 1, 0.0)
    vals = request.security_lower_tf(syminfo.tickerid, "1", ta.sma(close, 3))
    if timeframe.period == "5" and matrix.rows(boxes) == 0:
        matrix.add_row(boxes, 0)
    first = matrix.get(boxes, 0, 0)
    plot(array.size(vals) + first, "size")
    return {"vals": vals}


def __test_helper_run(runner, no_slice):
    """Run the script once.

    :param runner: The ``runner`` fixture.
    :param no_slice: Whether to switch the security slice off.
    :return: The context's array per chart bar.
    """
    import os
    import sys
    import tempfile
    from pathlib import Path

    sys.modules.pop(Path(__file__).stem, None)
    rows: list = []
    if no_slice:
        os.environ['PYNE_NO_SECURITY_SLICE'] = '1'
    try:
        with tempfile.TemporaryDirectory() as td:
            security_data = {"1": __test_helper_write_feed(Path(td))}
            r = runner(__test_helper_bars(__test_helper_step), security_data=security_data)
            for _candle, values in r.run_iter():
                rows.append(list(values["vals"]))
    finally:
        os.environ.pop('PYNE_NO_SECURITY_SLICE', None)
    return rows


@pytest.mark.aot_unsupported("switches the security slice off; "
                             "a compiled run has no unsliced child")
def __test_the_unsliced_child_reaches_the_chart_side_statement__(runner, log):
    """Unsliced, the LTF child evaluates the chart's statements that raise there.

    The child evaluates ``matrix.get(boxes, 0, 0)`` on its own bars, where the
    chart's row was never added, and dies with an ``IndexError`` (converter corpus
    1a04048648, Volume Analysis - Heatmap and Volume Profile).
    """
    try:
        __test_helper_run(runner, no_slice=True)
    except RuntimeError as exc:
        assert "IndexError" in str(exc), f"unexpected failure: {exc}"
    else:
        raise AssertionError("the unsliced child did not reach the chart-side statement")
    log.info("the unsliced child died on the chart-side statement")


def __test_an_ltf_context_skips_the_chart_side_statements__(runner, log):
    """The LTF child runs its slice, not the chart's statements that raise there.

    Sliced, every chart bar carries its five intrabars' ``ta.sma(close, 3)``.
    """
    rows = __test_helper_run(runner, no_slice=False)
    sma = __test_helper_sma([b.close for b in __test_helper_bars(__test_helper_minute)], 3)
    per_bar = __test_helper_step // __test_helper_minute
    for i, vals in enumerate(rows):
        want = sma[i * per_bar:(i + 1) * per_bar]
        assert len(vals) == per_bar, f"chart bar {i}: {len(vals)} intrabars"
        for got, exp in zip(vals, want):
            assert got == exp or (got != got and exp != exp), \
                f"chart bar {i}: {vals} != {want}"
    log.info("%d chart bars, each with its %d intrabar values", len(rows), per_bar)


# Every timestamp here is Unix MILLISECONDS.
__test_helper_t0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC
__test_helper_minute = 60_000
__test_helper_step = 300_000  # 5 minutes, the chart
__test_helper_span = 6 * 3_600_000


def __test_helper_sma(values, length):
    """``ta.sma`` over a plain list, ``na`` (NaN) until ``length`` values.

    :param values: The source values.
    :param length: The window.
    :return: The averages.
    """
    out = []
    for i in range(len(values)):
        if i + 1 < length:
            out.append(float('nan'))
        else:
            out.append(sum(values[i + 1 - length:i + 1]) / length)
    return out


def __test_helper_bars(span_ms):
    """Bars of ``span_ms`` over the test's hours, with a moving close.

    :param span_ms: The bar span in ms.
    :return: The bar list.
    """
    from pynecore.types.ohlcv import OHLCV
    bars = []
    for i in range(__test_helper_span // span_ms):
        c = 100.0 + (i % 13) * 0.25
        bars.append(OHLCV(timestamp=__test_helper_t0 + i * span_ms, open=c - 0.5,
                          high=c + 1.0, low=c - 1.0, close=c, volume=1.0 + i))
    return bars


def __test_helper_write_feed(tmp_dir):
    """Write the one-minute feed covering the chart's hours.

    :param tmp_dir: Directory to write into.
    :return: Path to the written ``.ohlcv`` file, as a string.
    """
    from datetime import time as dtime
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession

    path = tmp_dir / "LTFS1.ohlcv"
    with OHLCVWriter(path, "1") as w:
        for bar in __test_helper_bars(__test_helper_minute):
            w.write(bar)
    SymInfo(
        prefix="PYTEST", description="LTF slice", ticker="TEST",
        currency="USD", period="1", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=dtime(0, 0), end=dtime(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=dtime(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=dtime(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)
