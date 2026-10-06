"""
@pyne
"""
from pynecore.lib import script, timeframe, plot


@script.indicator(title="Timeframe in_seconds()", shorttitle="tf_is")
def main():
    plot(timeframe.in_seconds("1"), "in_seconds(1)")
    plot(timeframe.in_seconds("5"), "in_seconds(5)")
    plot(timeframe.in_seconds("240"), "in_seconds(240)")
    plot(timeframe.in_seconds("D"), "in_seconds(D)")
    plot(timeframe.in_seconds("4D"), "in_seconds(4D)")
    plot(timeframe.in_seconds("W"), "in_seconds(W)")
    plot(timeframe.in_seconds("5W"), "in_seconds(5W)")
    plot(timeframe.in_seconds("M"), "in_seconds(M)")
    plot(timeframe.in_seconds("2M"), "in_seconds(2M)")


def __test_timeframe_in_seconds__(csv_reader, runner, dict_comparator):
    """ timeframe.in_seconds() """
    import pytest

    with csv_reader('timeframe_is.csv', subdir="data") as cr:
        for candle, _plot in runner(cr).run_iter():
            dict_comparator(_plot, candle.extra_fields)

    pytest.raises(ValueError, timeframe.in_seconds, "5T").match("Not supported timeframe!")
    pytest.raises(AssertionError, timeframe.in_seconds, "5Q").match("Invalid timeframe: wrong modifier!")
