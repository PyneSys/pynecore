"""
@pyne
"""
from pynecore.lib import close, high, open, plot, request, script, syminfo, ta


def pack():
    return ta.sma(close, 3), ta.highest(high, 2)


def forward():
    # Returns another function's tuple: no tuple literal in its own ``return``
    return pack()


def flags():
    return close > open, close


def get(tf):
    # The read stands in a ``return``: its unpack target is out of the transform's sight
    return request.security(syminfo.tickerid, tf, pack())


def get_forward(tf):
    return request.security(syminfo.tickerid, tf, forward())


def get_flags(tf):
    return request.security(syminfo.tickerid, tf, flags())


@script.indicator(title="Tuple Read Through Function", shorttitle="TRF")
def main():
    a, b = get("15")
    c, d = get_forward("15")
    up, price = get_flags("15")
    ref_a, ref_b = request.security(syminfo.tickerid, "15", (ta.sma(close, 3), ta.highest(high, 2)))
    ref_up, ref_price = request.security(syminfo.tickerid, "15", (close > open, close))
    plot(a, "a")
    plot(b, "b")
    plot(c, "c")
    plot(d, "d")
    plot(1 if up else 0, "up")
    plot(price, "price")
    plot(ref_a, "ref_a")
    plot(ref_b, "ref_b")
    plot(1 if ref_up else 0, "ref_up")
    plot(ref_price, "ref_price")


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

    path = tmp_dir / f"TRF{tf}.ohlcv"
    with OHLCVWriter(path, str(tf)) as w:
        for bar in __test_helper_bars(tf):
            w.write(bar)
    SymInfo(
        prefix="PYTEST", description="Tuple Read Through Function", ticker="TEST",
        currency="USD", period=str(tf), type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=dtime(0, 0), end=dtime(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=dtime(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=dtime(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_helper_is_na(value):
    """Whether a plotted value is ``na`` (plots carry it as a float NaN)."""
    from math import isnan
    return isinstance(value, float) and isnan(value)


def __test_helper_same(a, b):
    """Equality that treats two ``na`` values as equal."""
    if __test_helper_is_na(a) or __test_helper_is_na(b):
        return __test_helper_is_na(a) and __test_helper_is_na(b)
    return a == b


def __test_tuple_read_through_a_function_matches_the_literal_read__(runner):
    """A read whose expression calls a tuple-returning function unpacks from the first bar.

    The security transform sees a tuple's width only in a tuple literal expression or
    in the read itself unpacked into names. Here the expression is a call and the read
    stands in a ``return``, so its default was one scalar ``na`` and the unpack raised
    ``TypeError: na is not iterable`` before the first 15-minute bar closed. The type
    pass knows the width, also through a function that returns another one's tuple,
    and the missing value is one ``na`` per element. Every bar matches the tuple
    literal read of the same expression.
    """
    import sys
    import tempfile
    from pathlib import Path

    sys.modules.pop(Path(__file__).stem, None)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        feeds = {"15": __test_helper_write_feed(Path(td), 15)}
        r = runner(__test_helper_bars(1), {"period": "1"}, security_data=feeds)
        for _candle, pv in r.run_iter():
            rows.append(dict(pv))

    assert len(rows) == __test_helper_minutes
    pairs = (("a", "ref_a"), ("b", "ref_b"), ("c", "ref_a"), ("d", "ref_b"),
             ("up", "ref_up"), ("price", "ref_price"))
    for i, pv in enumerate(rows):
        for got, want in pairs:
            assert __test_helper_same(pv[got], pv[want]), (
                f"bar {i}: {got} = {pv[got]!r}, {want} = {pv[want]!r}")
    # The default path ran (na before the first 15-minute value), and values followed
    assert __test_helper_is_na(rows[0]["a"]) and __test_helper_is_na(rows[0]["price"])
    assert not __test_helper_is_na(rows[-1]["a"])
    assert not __test_helper_is_na(rows[-1]["price"])
