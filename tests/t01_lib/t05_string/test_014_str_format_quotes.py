"""
@pyne

How str.format reads its pattern.

Every expected string below is measured TradingView output, not inference. The
pattern is scanned the way Java's MessageFormat does it: single quotes open a
literal section in which braces lose their meaning, ``''`` is one literal quote,
an unterminated quote makes the rest of the pattern literal, and an unquoted
``}`` outside a placeholder is literal as well -- only ``{`` opens one. This is
what lets a webhook script carry a JSON body through str.format:
``"'{'\\"stop\\": {0}'}'"`` prints real braces around a formatted number.

Inside a placeholder the segments are handed on almost untouched. The argument
index is parsed strictly (``{0 ,number,#}`` is an error on TradingView, not
index 0) but an index nobody passed an argument for is echoed rather than raised,
so a misnumbered placeholder does not stop the script. The type and any style
KEYWORD are matched trimmed and lowercased, while a style that is a DecimalFormat
pattern keeps its spaces and quotes -- so ``{0, number, #.#}`` prints a leading
space and ``'#'.##`` a leading hash.
"""
import pytest

from pynecore.lib.string import format as str_format


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass


@pytest.mark.parametrize("fmt,args,expected", [
    # Quoting in the literal text
    ("{0}", (1,), "1"),
    ("'{'", (1,), "{"),
    ("'}'", (1,), "}"),
    ("'{0}'", (1,), "{0}"),
    ("'{'0'}'", (1,), "{0}"),
    ("'{{ticker}}'", (1,), "{{ticker}}"),
    ("a'{'b'}'c", (1,), "a{b}c"),
    ("'abc'", (1,), "abc"),
    ("''", (1,), "'"),
    ("it''s {0}", (1,), "it's 1"),
    ("'{'''}'", (1,), "{'}"),
    ("'{0}' {0}", (1,), "{0} 1"),
    ("'{0}'{0}'{0}'", (1,), "{0}1{0}"),
    # An unterminated quote runs to the end of the pattern
    ("{0} 'unterminated {0}", (1,), "1 unterminated {0}"),
    ("x'", (1,), "x"),
    # Arguments are never rescanned for quotes
    ("{0}", ("a'b",), "a'b"),
    # The whole webhook idiom the wild corpus runs into
    ("'{'\"price\": {0,number,#.##}, \"tag\": '\"{{ticker}}\"}'", (1.234,),
     '{"price": 1.23, "tag": "{{ticker}}"}'),
    # An index with no argument behind it prints as its own placeholder
    ("{1}", (1,), "{1}"),
    ("{0} {2}", (1, 2), "1 {2}"),
])
def __test_format_quoting__(fmt: str, args: tuple, expected: str):
    """str.format resolves quotes the way TradingView does"""
    assert str_format(fmt, *args) == expected


@pytest.mark.parametrize("fmt,args,expected", [
    # A style keyword is matched trimmed and lowercased ...
    ("{0,NUMBER,#.#}", (1.234,), "1.2"),
    ("{0, number, integer}", (1.34,), "1"),
    ("{0, number, currency}", (1340000,), "$1,340,000.00"),
    ("{0, number, percent}", (0.1,), "10%"),
    # ... but a DecimalFormat pattern keeps every character it was given
    ("{0, number, #.#}", (1.234,), " 1.2"),
    ("{0,number,#.# }", (1.234,), "1.2 "),
    ("{0, number, #.# }", (1.234,), " 1.2 "),
    ("{0} != {0, number, #.#}", (1.34,), "1.34 !=  1.3"),
    ("{0,number,#.#}", (1.34,), "1.3"),
    # ... including its own quoted literals
    ("{0,number,'#'.##}", (1.234,), "#1.23"),
])
def __test_format_style_segment__(fmt: str, args: tuple, expected: str):
    """str.format trims a style keyword but not a number pattern"""
    assert str_format(fmt, *args) == expected


@pytest.mark.parametrize("fmt,args", [
    ("a{", (1,)),                   # unmatched braces
    ("{{0}}", (1,)),                # the index segment reads "{0}"
    ("{0 ,number,#.#}", (1.234,)),  # the index segment reads "0 ", and is not trimmed
])
def __test_format_rejects_malformed__(fmt: str, args: tuple):
    """str.format raises where TradingView raises"""
    with pytest.raises(ValueError):
        str_format(fmt, *args)
