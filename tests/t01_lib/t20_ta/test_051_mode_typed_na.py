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
            if i < 7:
                # Warm-up: the na bars never join the window, so four non-na
                # values first exist on bar 7 (every other bar carries one)
                assert value != value, f"bar {i}: expected na, got {value!r}"
            else:
                # An NA(int) hole must not be selected as mode, and an na bar
                # answers from the window it does not join
                assert value == 1, f"bar {i}: expected 1, got {value!r}"
            if i > 20:
                break
