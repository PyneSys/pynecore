"""
@pyne
"""
from pynecore.lib import bar_index, barmerge, close, input, plot, request, script, syminfo


@script.indicator(title="Runtime Gaps", shorttitle="RG")
def main(gap=input.bool(True, "Gaps")):
    # A Pine "simple" gaps mode held in a local: input-derived (hoisted to the
    # top signal block) and per-bar (signalled inline, before its write).
    mode = barmerge.gaps_on if gap else barmerge.gaps_off
    mode_inline = barmerge.gaps_on if gap and bar_index >= 0 else barmerge.gaps_off
    rt = request.security(syminfo.tickerid, "15", close, gaps=mode)
    rt_inline = request.security(syminfo.tickerid, "15", close, gaps=mode_inline)
    on = request.security(syminfo.tickerid, "15", close, gaps=barmerge.gaps_on)
    off = request.security(syminfo.tickerid, "15", close, gaps=barmerge.gaps_off)
    # Consumers reading a 15-minute context from a 5-minute child: the peer
    # path answers with the producer's gaps mode
    rt_peer = request.security(syminfo.tickerid, "5", rt)
    on_peer = request.security(syminfo.tickerid, "5", on)
    off_peer = request.security(syminfo.tickerid, "5", off)
    plot(rt, "rt")
    plot(rt_inline, "rt_inline")
    plot(on, "on")
    plot(off, "off")
    plot(rt_peer, "rt_peer")
    plot(on_peer, "on_peer")
    plot(off_peer, "off_peer")


__test_helper_t0 = 1_785_024_000_000  # 2026-07-26T00:00:00 UTC
__test_helper_minute = 60_000
__test_helper_minutes = 6 * 60


def __test_helper_bars(tf):
    """The ``tf``-minute bars of one minute series closing at ``100 + minute``.

    :param tf: Bar span in minutes.
    :return: The bar list.
    """
    from pynecore.types.ohlcv import OHLCV
    return [OHLCV(timestamp=__test_helper_t0 + p * __test_helper_minute,
                  open=100.0 + p, high=100.0 + p + tf - 1, low=100.0 + p,
                  close=100.0 + p + tf - 1, volume=float(tf))
            for p in range(0, __test_helper_minutes, tf)]


def __test_helper_write_feed(tmp_dir, tf):
    """Write the ``tf``-minute feed with its symbol info.

    :param tmp_dir: Directory to write into.
    :param tf: Bar span in minutes.
    :return: Path to the written ``.ohlcv`` file, as a string.
    """
    from datetime import time as dtime
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession

    path = tmp_dir / f"RG{tf}.ohlcv"
    with OHLCVWriter(path, str(tf)) as w:
        for bar in __test_helper_bars(tf):
            w.write(bar)
    SymInfo(
        prefix="PYTEST", description="Runtime Gaps", ticker="TEST",
        currency="USD", period=str(tf), type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=dtime(0, 0), end=dtime(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=dtime(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=dtime(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_helper_run(runner, gap):
    """Run the script on the 1-minute chart.

    :param runner: The ``runner`` fixture.
    :param gap: The ``Gaps`` input.
    :return: The plot values of every chart bar.
    """
    import sys
    import tempfile
    from pathlib import Path

    sys.modules.pop(Path(__file__).stem, None)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        feeds = {str(tf): __test_helper_write_feed(tmp, tf) for tf in (5, 15)}
        r = runner(__test_helper_bars(1), {"period": "1"}, security_data=feeds,
                   inputs={"gap": gap})
        for _candle, pv in r.run_iter():
            rows.append(dict(pv))
    return rows


def __test_helper_is_na(value):
    """Whether a plotted value is ``na`` (plots carry it as a float NaN)."""
    from math import isnan
    return isinstance(value, float) and isnan(value)


def __test_helper_same(a, b):
    """Equality that treats two ``na`` values as equal."""
    if __test_helper_is_na(a) or __test_helper_is_na(b):
        return __test_helper_is_na(a) and __test_helper_is_na(b)
    return a == b


def __test_helper_check(rows, mode):
    """Every runtime-gaps plot must match the constant-gaps plot of ``mode``.

    :param rows: The rows of one run.
    :param mode: ``"on"`` or ``"off"``.
    """
    assert len(rows) == __test_helper_minutes
    for i, pv in enumerate(rows):
        for got, want in (("rt", mode), ("rt_inline", mode), ("rt_peer", f"{mode}_peer")):
            assert __test_helper_same(pv[got], pv[want]), (
                f"bar {i}: {got} = {pv[got]!r}, {want} = {pv[want]!r}")
    # The two modes differ on this data, so the comparison above can tell them apart
    assert any(__test_helper_is_na(pv["on"]) and not __test_helper_is_na(pv["off"])
               for pv in rows)
    assert any(__test_helper_is_na(pv["on_peer"]) and not __test_helper_is_na(pv["off_peer"])
               for pv in rows)


def __test_runtime_gaps_on__(runner, log):
    """A ``gaps`` mode held in a local loads, and ``gaps_on`` is applied.

    Before the fix the module-level ``__security_contexts__`` literal named the
    local, and importing the script raised ``NameError``.
    """
    __test_helper_check(__test_helper_run(runner, True), "on")


def __test_runtime_gaps_off__(runner, log):
    """The same script with the input off behaves as ``gaps_off``."""
    __test_helper_check(__test_helper_run(runner, False), "off")
