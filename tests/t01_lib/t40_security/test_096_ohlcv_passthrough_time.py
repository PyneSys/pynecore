"""
@pyne
"""
from pynecore.lib import open, plot, request, script, syminfo, time
from pynecore.types import Series


@script.indicator(title="OHLCV Passthrough Time", shorttitle="OPT")
def main():
    # ``time`` alone, and inside a tuple of bar fields, is served by the child's
    # main()-free fast path; ``time + 0`` and ``open + 0.0`` are computed, so their
    # child runs the sliced main(). Both have to answer the same.
    t_pass: Series[float] = request.security(syminfo.tickerid, "60", time)
    t_calc: Series[float] = request.security(syminfo.tickerid, "60", time + 0)
    o_pass, t_ltf_pass = request.security_lower_tf(syminfo.tickerid, "1", (open, time))
    o_calc, t_ltf_calc = request.security_lower_tf(syminfo.tickerid, "1",
                                                   (open + 0.0, time + 0))
    plot(t_pass, "t_pass")
    plot(t_calc, "t_calc")
    return {"o_pass": o_pass, "t_ltf_pass": t_ltf_pass,
            "o_calc": o_calc, "t_ltf_calc": t_ltf_calc}


def __test_helper_run(runner):
    """Run the script with one child per context.

    :param runner: The ``runner`` fixture.
    :return: ``(rows, contexts)``, the values per chart bar and the script
        module's security contexts.
    """
    import os
    import sys
    import tempfile
    from pathlib import Path

    sys.modules.pop(Path(__file__).stem, None)
    rows: list[dict] = []
    # One child per context: the passthrough contexts must not share a child
    # with a computed one, which would run the clone for them as well
    os.environ['PYNE_NO_SECURITY_MERGE'] = '1'
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            security_data = {
                "60": __test_helper_write_feed(tmp, "60", __test_helper_hour),
                "1": __test_helper_write_feed(tmp, "1", __test_helper_minute),
            }
            r = runner(__test_helper_bars(__test_helper_step), security_data=security_data)
            for _candle, values in r.run_iter():
                rows.append({k: (list(v) if isinstance(v, list) else v)
                             for k, v in values.items()})
            contexts = dict(getattr(r.script_module, '__security_contexts__'))
    finally:
        os.environ.pop('PYNE_NO_SECURITY_MERGE', None)
    return rows, contexts


def __test_time_takes_the_passthrough_and_answers_as_main_does__(runner, log):
    """``time`` is a passthrough field, and its fast path equals the computed one.

    A one-minute ``request.security_lower_tf`` of ``(open, high, low, close,
    volume, time)`` ran the whole sliced ``main()`` once per intrabar because
    ``time`` was not a passthrough field, which made a script with loops after
    the call (Mxwll Price Action Suite) run for hours.
    """
    rows, contexts = __test_helper_run(runner)

    fields = sorted(tuple(ctx.get('ohlcv_fields') or ()) for ctx in contexts.values())
    assert fields == [(), (), ('open', 'time'), ('time',)], \
        f"the passthrough contexts: {fields}"

    checked = 0
    for row in rows:
        assert __test_helper_same(row["t_pass"], row["t_calc"]), \
            f"60: {row['t_pass']} != {row['t_calc']}"
        assert row["t_ltf_pass"] == row["t_ltf_calc"], \
            f"1: {row['t_ltf_pass']} != {row['t_ltf_calc']}"
        assert row["o_pass"] == row["o_calc"], f"1: {row['o_pass']} != {row['o_calc']}"
        if row["t_ltf_pass"]:
            assert len(row["t_ltf_pass"]) == __test_helper_step // __test_helper_minute
            checked += 1
    assert checked > 200, f"only {checked} chart bars carried intrabars"
    log.info("%d chart bars compared, %d with intrabars", len(rows), checked)


def __test_helper_same(a, b):
    """Whether two plotted values are the same, ``na`` (a NaN) included.

    :param a: One value.
    :param b: The other.
    :return: Whether they are equal or both ``na``.
    """
    return a == b or (a != a and b != b)


# Every timestamp here is Unix MILLISECONDS.
__test_helper_t0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, on the 1m/5m/1h grids
__test_helper_minute = 60_000
__test_helper_step = 300_000  # 5 minutes, the chart
__test_helper_hour = 3_600_000
__test_helper_span = 24 * __test_helper_hour


def __test_helper_bars(span_ms):
    """Bars of ``span_ms`` over the test's day, with a moving price.

    :param span_ms: The bar span in ms.
    :return: The bar list.
    """
    from pynecore.types.ohlcv import OHLCV
    bars = []
    for i in range(__test_helper_span // span_ms):
        c = 100.0 + (i % 17) * 0.5
        bars.append(OHLCV(timestamp=__test_helper_t0 + i * span_ms, open=c - 0.25,
                          high=c + 1.0, low=c - 1.0, close=c, volume=1.0 + i))
    return bars


def __test_helper_write_feed(tmp_dir, timeframe, span_ms):
    """Write one feed covering the chart's day.

    :param tmp_dir: Directory to write into.
    :param timeframe: The feed's timeframe string.
    :param span_ms: That timeframe's period length in ms.
    :return: Path to the written ``.ohlcv`` file, as a string.
    """
    from datetime import time as dtime
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession

    path = tmp_dir / ("OPT" + timeframe + ".ohlcv")
    with OHLCVWriter(path, timeframe) as w:
        for bar in __test_helper_bars(span_ms):
            w.write(bar)
    SymInfo(
        prefix="PYTEST", description="Passthrough", ticker="TEST",
        currency="USD", period=timeframe, type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=dtime(0, 0), end=dtime(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=dtime(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=dtime(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)
