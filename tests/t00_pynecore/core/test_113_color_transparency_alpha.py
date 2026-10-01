"""
@pyne

color.new / color.rgb transparency -> alpha byte, and color.t read back.

Measured on TradingView (CAPITALCOM:EURUSD 1D, a 0.1-step transparency sweep of
-10..110 through both color.new and color.rgb, read back from the packed bgcolor
export and from color.t):

- an out of range transparency is clipped into 0-100 instead of rejected,
- the alpha byte is ``(1 - transp / 100) * 255`` rounded half up,
- color.t returns the alpha byte converted back and rounded to a whole number.
"""
from pynecore.lib import color


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass


def __test_color_new_out_of_range_transparency_is_clipped__():
    """110, 150, 1000 behave as 100 and -10 as 0 instead of raising."""
    for transp in (100.5, 101, 110, 150, 255, 1000):
        assert color.new(color.red, transp).a == 0
        assert color.t(color.new(color.red, transp)) == 100
    for transp in (-0.4, -10):
        assert color.new(color.red, transp).a == 255
        assert color.t(color.new(color.red, transp)) == 0


def __test_alpha_is_rounded_half_up__():
    """Ties round up (30 -> 179), 90 gives 25 because its product is 25.4999..."""
    assert color.new(color.red, 10).a == 230
    assert color.new(color.red, 30).a == 179
    assert color.new(color.red, 50).a == 128
    assert color.new(color.red, 70).a == 77
    assert color.new(color.red, 90).a == 25
    assert color.new(color.red, 0.4).a == 254
    assert color.new(color.red, 50.3).a == 127
    assert color.new(color.red, 99.6).a == 1
    assert color.rgb(242, 54, 69, 30).a == 179


def __test_color_t_is_a_whole_number__():
    """color.t reads the alpha byte back and rounds it."""
    assert color.t(color.new(color.red, 30)) == 30
    assert color.t(color.new(color.red, 50.3)) == 50
    assert color.t(color.new(color.red, 0.4)) == 0
    assert color.t(color.new(color.red, 0.7)) == 1
    assert color.t(color.new(color.red, 2.5)) == 2
    assert color.t(color.new(color.new(color.red, 77), 30)) == 30
