"""
@pyne

``text.format_*`` is a set of text styles, combined with ``+`` (union) and ``-``
(difference), and the visual-data stream names the styles a value holds.

The expected values were measured on TradingView: the drawing data of a chart
carries each label's, box's and table cell's formatting as a bitmask (``tfrm``).
"""
from pynecore.lib import script, label, box, table, text, position, bar_index, high, low, barstate


@script.indicator("Text format set", overlay=True)
def main():
    if barstate.islast:
        label.new(bar_index, high, "bold+italic", text_formatting=text.format_bold + text.format_italic)
        label.new(bar_index, high, "bold+bold", text_formatting=text.format_bold + text.format_bold)
        label.new(bar_index, high, "none", text_formatting=text.format_none)
        lbl = label.new(bar_index, high, "set")
        label.set_text_formatting(lbl, text.format_bold + text.format_italic - text.format_italic)
        box.new(bar_index - 1, high, bar_index, low, text="box",
                text_formatting=text.format_italic + text.format_bold)
        t = table.new(position.top_right, 1, 1)
        table.cell(t, 0, 0, "cell", text_formatting=text.format_bold + text.format_italic)


def __test_text_format_bits_match_tradingview__():
    """each constant carries TradingView's bit value"""
    from pynecore.types.text import FormatEnum
    assert int(text.format_none) == 0
    assert int(text.format_bold) == 1
    assert int(text.format_italic) == 2
    assert isinstance(text.format_bold + text.format_italic, FormatEnum)
    assert isinstance(text.format_bold - text.format_italic, FormatEnum)


def __test_text_format_arithmetic_is_set_arithmetic__():
    """+ adds a style at most once, - removes only a style that is present"""
    assert int(text.format_bold + text.format_italic) == 3
    assert int(text.format_bold + text.format_bold) == 1
    assert int(text.format_none + text.format_bold) == 1
    assert int(text.format_bold + text.format_italic + text.format_bold) == 3
    assert int(text.format_bold + text.format_italic - text.format_italic) == 1
    assert int(text.format_bold - text.format_italic) == 1
    assert int(text.format_bold - text.format_bold) == 0


def __test_text_format_combinations_reach_drawings__(runner):
    """labels, boxes and table cells serialize their formatting as style names"""
    from pynecore.types.ohlcv import OHLCV

    bars = [OHLCV(timestamp=1_704_067_200_000 + i * 300_000, open=100.0, high=101.0,
                  low=99.0, close=100.5, volume=1.0) for i in range(2)]
    r = runner(iter(bars))
    list(r.run_iter())
    snap = r.drawings()

    labels = {lb["text"]: lb for lb in snap["labels"]}
    assert labels["bold+italic"]["text_formatting"] == ["bold", "italic"]
    assert labels["bold+bold"]["text_formatting"] == ["bold"]
    assert labels["none"]["text_formatting"] == []
    assert labels["set"]["text_formatting"] == ["bold"]
    assert snap["boxes"][0]["text_formatting"] == ["bold", "italic"]
    assert snap["tables"][0]["cells"][0]["text_formatting"] == ["bold", "italic"]
