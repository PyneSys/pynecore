"""
@pyne
"""
from pynecore import Series
from pynecore.lib import script, bar_index, na, ta


@script.indicator(title="Cross Bool Na", shorttitle="cross_bool_na", overlay=False, na_bool=True)
def main():
    # Under the three-state bool a cross over an na source is a bool na, not
    # false; the na gap leaves the remembered relation alone. Measured on
    # TradingView (BINANCE:BTCUSDT 60m, v4 and v5): crossover / crossunder /
    # cross export na on exactly the bars where a source is na, and 0 on the
    # first bar where both are defined.

    # below 3..4, na at bar 5, above from bar 6: the gap keeps the up-cross armed
    up: Series[float] = na if bar_index < 3 or bar_index == 5 else (0.0 if bar_index < 5 else 2.0)
    # above 3..4, na at bar 5, below from bar 6
    down: Series[float] = na if bar_index < 3 or bar_index == 5 else (2.0 if bar_index < 5 else 0.0)
    level = 1.0

    over = ta.crossover(up, level)
    under = ta.crossunder(down, level)
    cross = ta.cross(up, level)
    return {
        "over": 1.0 if over else 0.0,
        "over_na": 1.0 if na(over) else 0.0,
        "under": 1.0 if under else 0.0,
        "under_na": 1.0 if na(under) else 0.0,
        "cross": 1.0 if cross else 0.0,
        "cross_na": 1.0 if na(cross) else 0.0,
    }


def __test_cross_over_an_na_source_is_a_bool_na__(runner, dummy_ohlcv_iter):
    """ A cross over an na source is na under the three-state bool and keeps its state """
    keys = ("over", "over_na", "under", "under_na", "cross", "cross_na")
    fired = {k: [] for k in keys}
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    for i in range(8):
        _, plot = next(run_iter)
        for k in fired:
            if plot[k]:
                fired[k].append(i)

    assert fired["over_na"] == [0, 1, 2, 5]
    assert fired["under_na"] == [0, 1, 2, 5]
    assert fired["cross_na"] == [0, 1, 2, 5]
    assert fired["over"] == [6]
    assert fired["under"] == [6]
    assert fired["cross"] == [6]
