"""
@pyne

color.rgb — Pine-compatible handling of fractional, out of range and na arguments.

The implementation used to demand integers in the 0-255 range: a fractional
channel died with ``ValueError: Unknown format code 'X' for object of type
'float'`` and anything outside the range raised as well. TradingView accepts all
of it, so the values below are the measured ground truth (BINANCE:BTCUSDT 1D):
a fractional channel is TRUNCATED and not rounded, an out of range argument is
clipped, and an na argument gives a solid color -- na channel 0, na
transparency 100.
"""
from pynecore.lib import color
from pynecore.types.na import NA


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass


#
# Fractional channels
#

def __test_fractional_channel_is_truncated__():
    """127.4, 127.5 and 127.6 all give 127 -- a rounding implementation gives 128 for two of them."""
    assert color.r(color.rgb(127.4, 0, 0)) == 127
    assert color.r(color.rgb(127.5, 0, 0)) == 127
    assert color.r(color.rgb(127.6, 0, 0)) == 127
    assert color.r(color.rgb(126.5, 0, 0)) == 126
    assert color.r(color.rgb(0.5, 0, 0)) == 0
    assert color.g(color.rgb(0, 254.7, 0)) == 254


#
# Out of range arguments
#

def __test_out_of_range_channel_is_clipped__():
    """Above 255 and below 0 clip to the ends instead of raising."""
    assert color.r(color.rgb(255.6, 0, 0)) == 255
    assert color.r(color.rgb(300, 0, 0)) == 255
    assert color.r(color.rgb(-0.5, 0, 0)) == 0
    assert color.r(color.rgb(-20, 0, 0)) == 0


def __test_out_of_range_transparency_is_clipped__():
    """Transparency clips into 0-100, so the alpha byte never overflows."""
    assert color.t(color.rgb(0, 0, 0, 110)) == 100
    assert color.t(color.rgb(0, 0, 0, -10)) == 0


#
# na arguments
#

def __test_na_channel_reads_back_as_zero__():
    """An na channel is 0, and the other channels keep their own values."""
    c = color.rgb(NA(float), 10, 10)
    assert color.r(c) == 0
    assert color.g(c) == 10
    assert color.t(c) == 0


def __test_na_transparency_is_fully_transparent__():
    """An na transparency gives 100, while the channels stay intact."""
    c = color.rgb(10, 10, 10, NA(float))
    assert color.r(c) == 10
    assert color.t(c) == 100
