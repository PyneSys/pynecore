"""
@pyne
"""
from pynecore.lib import script, plot, bar_index, na, nz, fixnan


@script.indicator(title="Na", shorttitle="na")
def main():
    val = bar_index if bar_index % 2 == 0 else na(float)

    plot(val, "val")
    plot(0 if na(val) else 1, "na(val)")
    plot(nz(val), "nz(val)")
    plot(nz(val, 2), "nz(val, 2)")
    plot(fixnan(val), "fixnan(val)")


def __test_time__(csv_reader, runner, dict_comparator, log):
    """ NA """
    with csv_reader('na.csv', subdir="data") as cr:
        for i, (candle, _plot) in enumerate(runner(cr).run_iter()):
            dict_comparator(_plot, candle.extra_fields)
            if i > 300:
                break


def __test_nz_default_by_type__():
    """ one-argument nz answers the zero of the source's type """
    # Measured on TradingView: color.r/g/b of nz(color(na)) are 0, color.t is 100
    from pynecore.lib import color
    from pynecore.types.color import Color
    from pynecore.types.na import NA

    zero = nz(float('nan'))
    assert zero == 0.0 and type(zero) is float
    assert nz(NA(bool)) is False
    transparent = nz(NA(Color))
    assert isinstance(transparent, Color)
    assert (color.r(transparent), color.g(transparent), color.b(transparent),
            color.t(transparent)) == (0, 0, 0, 100)
    assert nz(NA(Color), color.red) is color.red
