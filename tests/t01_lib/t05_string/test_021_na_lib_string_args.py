"""
@pyne

A typeless ``na`` passed to a lib function's string parameter is the string na, the
empty string; passed as compile-time text of the study it is the text ``na``.

MEASURED on TradingView (CAPITALCOM:EURUSD 60): ``label.new(x, y, na, tooltip=na)`` and
``label.set_text(l, na)`` leave ``label.get_text`` at ``""``; ``array.join(a, na)`` joins
with nothing; ``ticker.new(na, na)`` is ``:`` and ``ticker.standard(na)`` / ``modify(na)`` /
``inherit(na, na)`` are ``""``; ``request.currency_rate(na, "USD")`` is 1 like
``currency_rate("", "USD")``; ``log.info(na)`` logs an empty message. In the study's metadata
``plot(title=na)`` is titled ``na`` (``title=""`` stays empty), ``plotchar(char=na,
text=na)`` draws ``n`` with the text ``na`` (only the first character of ``char`` is drawn,
``char="ab"`` draws ``a``; the default is ``★``), and ``plotshape(text=na)`` has the
text ``na``.
"""
from pynecore.lib import array, bar_index, close, label, na, open, plot, plotchar, plotshape, script
from pynecore.lib import string, ticker


@script.indicator("Na Lib String Args", overlay=True)
def main():
    plot(close, title=na)
    plot(open, title="")
    plotchar(close > open, title="c1", char=na, text=na)
    plotchar(close > open, title="c2", char="ab")
    plotchar(close > open, title="c3")
    plotshape(close > open, title="s1", text=na)

    created = label.new(bar_index, close, na, tooltip=na)
    first = label.get_text(created)
    label.set_text(created, "x")
    label.set_text(created, na)

    return {
        "label": string.format("[{0}] [{1}] [{2}]", first, label.get_text(created),
                               created.tooltip),
        "join": array.join(array.from_items("a", "b"), na),
        "ticker": string.format("[{0}] [{1}] [{2}] [{3}]", ticker.new(na, na),
                                ticker.standard(na), ticker.modify(na), ticker.inherit(na, na)),
    }


def __test_na_lib_string_args__(runner, dummy_ohlcv_iter):
    """na is the empty string in run-time string parameters and the text na in metadata"""
    from pynecore import lib
    from pynecore.core.viz import serialize_meta
    result = None
    for _candle, plot_data in runner(dummy_ohlcv_iter).run_iter():
        result = plot_data
        break
    assert result is not None
    assert result["label"] == "[] [] []"
    assert result["join"] == "ab"
    assert result["ticker"] == "[:] [] [] []"
    assert "na" in result and "" in result
    chars = {meta.title: meta for meta in lib._plot_meta.values()}
    assert chars["c1"].char == "n" and chars["c1"].text == "na"
    assert chars["c2"].char == "a"
    assert chars["c3"].char is None
    assert serialize_meta(chars["c1"])["char"] == "n"
    assert serialize_meta(chars["c3"])["char"] == "★"
    assert chars["s1"].text == "na"
