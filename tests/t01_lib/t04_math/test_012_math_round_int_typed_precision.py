"""
``math.round`` tells its two overloads apart by the PRESENCE of a precision.

Pine's ``int`` is a static type only, so an int-TYPED precision reaches PyneCore
as a Python float whenever it came out of a division: ``R / 14`` is int-typed
with the value 1.0. Discriminating on ``isinstance(precision, int)`` read that
as "no precision given" and dropped the rounding -- a silent wrong result rather
than a crash, and reachable from any divided precision.

MEASURED on TradingView (FX:EURUSD@60, ``math.round(2.34567 + z, ...)``,
``R = input.int(14)``, ``z = bar_index >= 0 ? 0 : 1``):

| precision expression   | value | TradingView  |
|------------------------|-------|--------------|
| `(R + z) / 14`         | 1.0   | 2.3          |
| `(R + z) / 14 + 1`     | 2.0   | 2.35         |
| `2` (literal)          | 2     | 2.35         |
| `1 + z`                | 1     | 2.3          |
| `(R + z) / 8`          | 1.75  | 2.3378730879 |
| `(R + z) * 199 / 1400` | 1.99  | 2.3479196174 |

The last two rows are an OPEN reverse-engineering question: TradingView uses a
fractional precision continuously, and the formula is not
``round(x * 10 ** p) / 10 ** p`` -- that gives 2.34734 at p = 1.75. Truncating a
fractional precision is provisional, so those rows are not asserted here.
"""
from pynecore.lib import math
from pynecore.types.na import NA


def __test_integral_float_precision_rounds__():
    """A precision carried as a float rounds exactly like the same int"""
    assert math.round(2.34567, 1.0) == math.round(2.34567, 1) == 2.3
    assert math.round(2.34567, 2.0) == math.round(2.34567, 2) == 2.35
    assert math.round(2.34567, 0.0) == math.round(2.34567, 0) == 2.0
    # The int-typed division that produced the divergence in the first place
    assert math.round(2.34567, 14 / 14) == 2.3
    assert math.round(2.34567, 14 / 14 + 1) == 2.35


def __test_negative_float_precision_scales_up__():
    """A negative precision keeps working when it arrives as a float"""
    assert math.round(1234.5678, -2.0) == math.round(1234.5678, -2) == 1200.0


def __test_missing_precision_keeps_the_int_contract__():
    """Without a precision the one-argument overload still returns an int"""
    assert math.round(2.34567) == 2
    # A Pine int is a double at runtime: the integral result is a float
    assert math.round(2.5) == 3 and isinstance(math.round(2.5), float)
    # An explicit na precision is the same "not given"
    assert math.round(2.34567, NA(int)) == 2
