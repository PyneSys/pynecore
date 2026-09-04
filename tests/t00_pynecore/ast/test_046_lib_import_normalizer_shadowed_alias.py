"""
@pyne

A lib-module alias rebound later in the module denotes that binding from
there on: the uses before the rebinding are rewritten to the lib path, the
uses after it are left alone.
"""
import pynecore.lib.ta as ta
import pynecore.lib.ta as x
from pynecore.lib import close, plot, script


def smoothed():
    return ta.sma(close, 3)


def first():
    return x.sma(close, 2)


import pynecore.lib.math as x  # noqa: E402


def second():
    return x.max(close, 2.0)


def ta():
    return 1.0


@script.indicator("shadowed alias")
def main():
    plot(smoothed() + ta() + first() + second(), 'x')


def __test_import_normalizer_shadowed_alias__(ast_transformed_code):
    """The alias is rewritten before its rebinding and kept after a foreign one;
    ``x`` is ta before its second import and math after it"""
    assert ast_transformed_code.count('lib.ta.sma(') == 2
    assert 'lib.ta()' not in ast_transformed_code
    assert 'def ta():' in ast_transformed_code
    assert 'lib.math.max(' in ast_transformed_code
    assert 'lib.math.sma(' not in ast_transformed_code
    assert 'lib.math.sma(' not in ast_transformed_code
