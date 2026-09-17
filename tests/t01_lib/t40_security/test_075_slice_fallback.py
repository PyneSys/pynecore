"""
@pyne
"""
from pynecore.lib import close, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="Slice Plain OHLCV", shorttitle="SPO")
def main():
    # A bare OHLCV field: the degenerate case of the slice, already served by
    # the ``ohlcv_fields`` fast path — the child ships the field without
    # replaying the script at all. Slicing must leave that path alone.
    x: Series[float] = request.security(syminfo.tickerid, "60", close)
    plot(x, "x")


# A side script whose ``main()`` contains a ``try`` statement: an unmodelled
# node, so no backward slice can be proved sound and the context must fall back
# to running the whole ``main()`` in the child.
__test_helper_unsliceable_script = '''"""
@pyne
"""
from pynecore.lib import close, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="Slice Fallback", shorttitle="SFB")
def main():
    v: Series[float] = close
    try:
        v = close * 2.0
    except ValueError:
        v = close
    x: Series[float] = request.security(syminfo.tickerid, "60", v + 1.0)
    plot(x, "x")
'''

# Every timestamp here is Unix MILLISECONDS.
__test_helper_t0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, on the 1h and 5m grids
__test_helper_hour = 3_600_000
__test_helper_step = 300_000  # 5 minutes
__test_helper_hours = 12
__test_helper_bars = __test_helper_hours * 12


def __test_helper_write_feed(tmp_dir):
    """Write the ``60`` feed the higher-timeframe context reads.

    :param tmp_dir: Directory to write into.
    :return: Path to the written ``.ohlcv`` file, as a string.
    """
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "SPO60.ohlcv"
    with OHLCVWriter(path, "60") as w:
        for hour in range(__test_helper_hours):
            c = 100.0 + hour
            w.write(OHLCV(timestamp=__test_helper_t0 + hour * __test_helper_hour,
                          open=c, high=c + 1.0, low=c - 1.0, close=c, volume=1.0))
    SymInfo(
        prefix="PYTEST", description="Slice Fallback", ticker="TEST",
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
    """The chart's own 5-minute bars.

    :return: The bar list.
    """
    from pynecore.types.ohlcv import OHLCV
    bars = []
    for i in range(__test_helper_bars):
        c = 40.0 + (i % 17) * 0.25
        bars.append(OHLCV(timestamp=__test_helper_t0 + i * __test_helper_step,
                          open=c - 0.1, high=c + 0.5, low=c - 0.5, close=c,
                          volume=1.0))
    return bars


def __test_helper_is_na(value):
    """Whether a read answered Pine ``na``, in either of its two shapes.

    :param value: The read value.
    :return: Whether it is ``na``.
    """
    from pynecore.types.na import NA
    return value is None or isinstance(value, NA) or (
        isinstance(value, float) and value != value)


def __test_helper_contexts(path):
    """Import a script through the Pyne hook and return its context metadata.

    :param path: The script's path.
    :return: The module's ``__security_contexts__`` dict.
    """
    import os
    import sys

    from pynecore.core.script_runner import import_script

    os.environ['PYNE_SAVE_SCRIPT_TOML'] = '0'
    sys.modules.pop(path.stem, None)
    module = import_script(path)
    contexts = getattr(module, '__security_contexts__', None)
    assert contexts, f"{path.name} declared no security contexts"
    return contexts


def __test_helper_only_context(contexts):
    """Return the single context of a one-context script.

    :param contexts: The ``__security_contexts__`` dict.
    :return: Its only value.
    """
    assert len(contexts) == 1, f"expected one context, got {sorted(contexts)}"
    for sid in contexts:
        return contexts[sid]
    return None


def __test_plain_ohlcv_expression_keeps_the_field_fast_path__(runner, log):
    """A bare OHLCV expression still ships its field list, not a slice

    ``ohlcv_fields`` is the existing degenerate slice: the child answers from
    the feed without running the script. Slicing must not take that case over —
    the context keeps its field list — and the values it publishes must still be
    the higher timeframe's closes.
    """
    import os
    import tempfile
    from pathlib import Path

    contexts = __test_helper_contexts(Path(__file__))
    ctx = __test_helper_only_context(contexts)
    assert ctx.get('ohlcv_fields') == ['close'], \
        f"the field fast path was lost: {ctx}"
    assert not ctx.get('slice_main'), \
        f"a bare OHLCV context must not carry a slice clone: {ctx}"

    os.environ['PYNE_SAVE_SCRIPT_TOML'] = '0'
    with tempfile.TemporaryDirectory() as td:
        feed = __test_helper_write_feed(Path(td))
        r = runner(__test_helper_chart_bars(), security_data={"60": feed})
        rows = [dict(pv) for _candle, pv in r.run_iter()]

    values = []
    for row in rows:
        v = row.get("x")
        if not __test_helper_is_na(v):
            values.append(v)
    assert len(values) >= len(rows) // 2, \
        f"only {len(values)} of {len(rows)} bars carry a value"
    expected = set()
    for hour in range(__test_helper_hours):
        expected.add(100.0 + hour)
    assert set(values) <= expected, f"unexpected values: {sorted(set(values))}"

    log.info("the bare OHLCV context kept its field fast path over %d bars",
             len(rows))


def __test_unmodelled_main_falls_back_to_the_whole_main__(log):
    """A ``main()`` with a ``try`` gets no slice clone

    The backward slice is only sound while every statement can be classified.
    A ``try`` cannot be, so the transformer has to give up on that script and
    leave the context on today's behaviour: the child runs the whole ``main()``
    and ``slice_main`` is absent.
    """
    import sys
    import tempfile
    from pathlib import Path

    import pytest

    pytest.importorskip(
        "pynecore.transformers.security_slice",
        reason="per-context slicing is not implemented yet")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        path = tmp / "slice_fallback_reference.py"
        path.write_text(__test_helper_unsliceable_script)
        sys.path.insert(0, str(tmp))
        try:
            ctx = __test_helper_only_context(__test_helper_contexts(path))
        finally:
            sys.path.remove(str(tmp))
            sys.modules.pop(path.stem, None)

    assert not ctx.get('slice_main'), \
        f"an unmodelled main() was sliced anyway: {ctx}"

    log.info("the unsliceable script fell back to the whole main()")


def __test_unsliceable_script_still_runs__(runner, log):
    """The fallback script produces values whether or not slicing exists

    The fallback is a performance decision, never a correctness one: the same
    script with a ``try`` in it has to run and publish exactly as it does today.
    """
    import os
    import sys
    import tempfile
    from pathlib import Path

    from pynecore.core.script_runner import ScriptRunner

    os.environ['PYNE_SAVE_SCRIPT_TOML'] = '0'
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        feed = __test_helper_write_feed(tmp)
        path = tmp / "slice_fallback_reference.py"
        path.write_text(__test_helper_unsliceable_script)
        chart = runner(__test_helper_chart_bars())
        sys.modules.pop(path.stem, None)
        sys.path.insert(0, str(tmp))
        try:
            sr = ScriptRunner(path, iter(__test_helper_chart_bars()),
                              chart.syminfo, security_data={"60": feed})
            rows = [dict(pv) for _candle, pv in sr.run_iter()]
        finally:
            sys.path.remove(str(tmp))
            sys.modules.pop(path.stem, None)

    values = []
    for row in rows:
        v = row.get("x")
        if not __test_helper_is_na(v):
            values.append(v)
    assert len(values) >= len(rows) // 2, \
        f"only {len(values)} of {len(rows)} bars carry a value"
    expected = set()
    for hour in range(__test_helper_hours):
        expected.add((100.0 + hour) * 2.0 + 1.0)
    assert set(values) <= expected, f"unexpected values: {sorted(set(values))}"

    log.info("the unsliceable script published on %d of %d bars",
             len(values), len(rows))
