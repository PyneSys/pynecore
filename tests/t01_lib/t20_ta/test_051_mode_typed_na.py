"""
@pyne

ta.mode typed-na regression: the non-na window filter must be isinstance-based.
The old identity filter (``source[i] is not NA(float)``) let ``NA(int)``
elements of an int series slip into the candidate list, and the na branches
returned ``NA(float)`` under an int contract (typed sentinels are interned per
type, so that breaks identity checks downstream).
"""
from pynecore.lib import script, bar_index, ta
from pynecore.types.na import NA
from pynecore.types.series import Series


@script.indicator(title="ta.mode typed na test")
def main():
    v: Series[int] = NA(int) if bar_index % 2 == 0 else 1
    return {
        "mode": ta.mode(v, 4),
    }


def __test_mode_int_series_with_na_holes__(csv_reader, runner):
    """NA(int) elements never win the mode; na results carry the int type"""
    with csv_reader('ma.csv', subdir="data") as cr:
        for i, (candle, plot) in enumerate(runner(cr).run_iter()):
            value = plot["mode"]
            if i % 2 == 0:
                # Current bar's source is na -> na result, typed after the series
                assert isinstance(value, NA), f"bar {i}: expected na, got {value!r}"
                assert value.type is int, f"bar {i}: expected NA(int), got {value!r}"
            elif i < 3:
                # Warm-up: typed after the source's runtime type
                assert isinstance(value, NA) and value.type is int
            else:
                # An NA(int) hole must not be selected as mode
                assert value == 1, f"bar {i}: expected 1, got {value!r}"
            if i > 20:
                break
