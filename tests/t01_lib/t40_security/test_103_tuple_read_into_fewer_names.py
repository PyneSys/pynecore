"""
@pyne
"""
from pynecore.lib import array, close, high, low, na, plot, request, script, syminfo


def triple(x):
    return close * x, high * x, low * x


def get(tf):
    # The read stands in a ``return``: it is unpacked at the call site
    return request.security(syminfo.tickerid, tf, triple(1))


def get_ltf(tf):
    # The lower-timeframe read stands in a ``return`` as well
    return request.security_lower_tf(syminfo.tickerid, tf, triple(1))


@script.indicator(title="Tuple Read Into Fewer Names", shorttitle="TRFN")
def main():
    a, b = request.security(syminfo.tickerid, "15", triple(1))
    c, d = get("15")
    la, lb = request.security_lower_tf(syminfo.tickerid, "1", triple(1))
    lc, ld = get_ltf("1")
    xa, xb, xc = get_ltf("1")
    ref_close = request.security(syminfo.tickerid, "15", close)
    ref_high = request.security(syminfo.tickerid, "15", high)
    ltf_close = request.security_lower_tf(syminfo.tickerid, "1", close)
    ltf_high = request.security_lower_tf(syminfo.tickerid, "1", high)
    plot(a, "a")
    plot(b, "b")
    plot(c, "c")
    plot(d, "d")
    plot(ref_close, "ref_close")
    plot(ref_high, "ref_high")
    plot(array.get(la, 0) if array.size(la) > 0 else na, "la0")
    plot(array.get(lb, 0) if array.size(lb) > 0 else na, "lb0")
    plot(array.get(ltf_close, 0) if array.size(ltf_close) > 0 else na, "ltf_close0")
    plot(array.get(ltf_high, 0) if array.size(ltf_high) > 0 else na, "ltf_high0")
    plot(array.get(lc, 0) if array.size(lc) > 0 else na, "lc0")
    plot(array.get(ld, 0) if array.size(ld) > 0 else na, "ld0")
    plot(array.get(xb, 0) if array.size(xb) > 0 else na, "xb0")
    plot(array.size(la), "la_size")
    plot(array.size(lc), "lc_size")
    plot(array.size(xc), "xc_size")
    plot(array.size(ltf_close), "ltf_size")


__test_helper_t0 = 1_785_024_000_000  # 2026-07-26T00:00:00 UTC
__test_helper_minute = 60_000
__test_helper_minutes = 6 * 60


def __test_helper_bars(tf):
    """The ``tf``-minute bars of one minute series: open and low ``100 + minute``,
    high and close ``100 + minute + tf - 1``, so ``close``, ``high`` and ``low`` of
    a bar are distinct values except ``close == high``. The ``triple`` elements are
    told apart by ``low`` and by the order of the two reference reads.

    :param tf: Bar span in minutes.
    :return: The bar list.
    """
    from pynecore.types.ohlcv import OHLCV
    return [OHLCV(timestamp=__test_helper_t0 + p * __test_helper_minute,
                  open=100.0 + p, high=100.0 + p + tf - 1 + 0.5, low=100.0 + p - 0.5,
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

    path = tmp_dir / f"TRFN{tf}.ohlcv"
    with OHLCVWriter(path, str(tf)) as w:
        for bar in __test_helper_bars(tf):
            w.write(bar)
    SymInfo(
        prefix="PYTEST", description="Tuple Read Into Fewer Names", ticker="TEST",
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


def __test_tuple_read_into_fewer_names_takes_the_leading_values__(runner):
    """A tuple read unpacked into fewer names than the tuple has takes its first values.

    TradingView compiles ``[a, b] = request.security(s, tf, f(1))`` with ``f`` giving
    three values, also when the read stands in another function's ``return`` and for
    ``request.security_lower_tf``, and assigns the first two (TV probe 2026-11-13,
    every bar of CAPITALCOM:EURUSD 1D against the single-value reads). The unpack
    raised ``ValueError: too many values to unpack``. Every bar matches the reads of
    the first two values on their own, before the first 15-minute value as well. A
    lower-timeframe read in a ``return`` gives one array per element too, unpacked
    into fewer names or into all of them; it gave the intrabar rows instead.
    """
    import sys
    import tempfile
    from pathlib import Path

    sys.modules.pop(Path(__file__).stem, None)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        feeds = {tf: __test_helper_write_feed(Path(td), int(tf)) for tf in ("1", "15")}
        r = runner(__test_helper_bars(5), {"period": "5"}, security_data=feeds)
        for _candle, pv in r.run_iter():
            rows.append(dict(pv))

    assert len(rows) == __test_helper_minutes // 5
    pairs = (("a", "ref_close"), ("b", "ref_high"), ("c", "ref_close"), ("d", "ref_high"),
             ("la0", "ltf_close0"), ("lb0", "ltf_high0"), ("la_size", "ltf_size"),
             ("lc0", "ltf_close0"), ("ld0", "ltf_high0"), ("xb0", "ltf_high0"),
             ("lc_size", "ltf_size"), ("xc_size", "ltf_size"))
    for i, pv in enumerate(rows):
        for got, want in pairs:
            assert __test_helper_same(pv[got], pv[want]), (
                f"bar {i}: {got} = {pv[got]!r}, {want} = {pv[want]!r}")
    # The default path ran (na before the first 15-minute value), and values followed
    assert __test_helper_is_na(rows[0]["a"]) and __test_helper_is_na(rows[0]["c"])
    assert not __test_helper_is_na(rows[-1]["a"]) and not __test_helper_is_na(rows[-1]["d"])
    assert rows[-1]["la_size"] == 5 and rows[-1]["lc_size"] == 5
