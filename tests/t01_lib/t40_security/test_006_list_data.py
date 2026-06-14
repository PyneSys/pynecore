"""
@pyne
"""
from pynecore.lib import close, format, plot, request, script, syminfo, timeframe


@script.indicator(title="List Data Test", shorttitle="LDT", format=format.price)
def main():
    # Same symbol + same timeframe -> chart / main data bucket
    a = request.security(syminfo.ticker, timeframe.period, close)
    # Same symbol + different timeframe -> same-symbol other-TF bucket
    b = request.security(syminfo.ticker, "1D", close)
    # Cross-symbol literal -> cross-symbol bucket
    c = request.security("NASDAQ:AAPL", "1D", close)
    # Cross-symbol with ignore_invalid_symbol -> cross-symbol bucket, tolerated
    d = request.security("NASDAQ:MSFT", "60", close, ignore_invalid_symbol=True)
    # Cross-symbol lower timeframe -> cross-symbol bucket, LTF
    e = request.security_lower_tf("NASDAQ:GOOG", "1", close)

    # Runtime symbol (function parameter) -> dynamic bucket
    def _dyn(sym):
        return request.security(sym, timeframe.period, close)

    f = _dyn("NYSE:XYZ")

    plot(a + b + c + d + f, title="Out")
    plot(e[0] if e else close, title="Ltf")


def __test_list_data__(runner, dummy_ohlcv_iter, log):
    """list_data_requirements buckets each request.security context statically"""
    r = runner(dummy_ohlcv_iter)

    chart_symbol = f"{r.syminfo.prefix}:{r.syminfo.ticker}"
    chart_tf = str(r.syminfo.period)

    req = r.list_data_requirements(
        chart_symbol=chart_symbol,
        chart_tf=chart_tf,
        security_keys={"NASDAQ:AAPL:1D"},
    )

    assert req.chart_symbol == chart_symbol
    assert req.chart_tf == chart_tf

    # Chart / main data: same symbol + same TF
    assert len(req.chart_main) == 1
    cm = req.chart_main[0]
    assert cm.symbol == chart_symbol and cm.timeframe == chart_tf
    assert not cm.is_ltf

    # Same symbol, other timeframe
    assert len(req.same_symbol_other_tf) == 1
    ss = req.same_symbol_other_tf[0]
    assert ss.symbol == chart_symbol and ss.timeframe == "1D"
    assert not ss.has_security_mapping  # no "1D"/symbol/"symbol:1D" key provided

    # Cross-symbol: AAPL (mapped), MSFT (ignore_invalid), GOOG (LTF)
    cross = {c.symbol: c for c in req.cross_symbol}
    assert set(cross) == {"NASDAQ:AAPL", "NASDAQ:MSFT", "NASDAQ:GOOG"}

    aapl = cross["NASDAQ:AAPL"]
    assert aapl.timeframe == "1D" and aapl.has_security_mapping
    assert not aapl.is_ltf and not aapl.ignore_invalid_symbol

    msft = cross["NASDAQ:MSFT"]
    assert msft.timeframe == "60" and msft.ignore_invalid_symbol
    assert not msft.has_security_mapping

    goog = cross["NASDAQ:GOOG"]
    assert goog.timeframe == "1" and goog.is_ltf

    # Dynamic: the function-parameter symbol cannot be resolved statically
    assert len(req.dynamic) == 1
    assert req.dynamic[0].symbol is None

    log.info("list_data_requirements buckets verified")


def __test_ltf_unzip__():
    """__ltf_unzip__ transposes the row-major intrabar buffer of a tuple
    security_lower_tf() result into column arrays; an empty buffer (a chart bar
    with no intrabars) yields N empty arrays so the tuple-unpack still succeeds.
    """
    from pynecore.core.security import __ltf_unzip__

    # Three intrabars, arity 2: row-major (h, l) pairs -> two column arrays.
    cols = __ltf_unzip__([(1, 10), (2, 20), (3, 30)], 2)
    assert cols == ([1, 2, 3], [10, 20, 30])
    assert all(isinstance(c, list) for c in cols)  # mutable arrays, not tuples

    # No intrabars -> N empty arrays; the unpack must not raise.
    a, b, c = __ltf_unzip__([], 3)
    assert a == [] and b == [] and c == []
