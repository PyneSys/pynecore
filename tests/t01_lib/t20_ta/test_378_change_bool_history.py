"""
@pyne
"""
from pynecore.lib import script, bar_index, ta


@script.indicator(title="Change Bool History", shorttitle="change_bool_history", overlay=False)
def main():
    # Under the two-state bool a history read before the bar exists is false, the
    # same false a stored value is. ta.change still knows the missing bars: measured
    # on TradingView (CAPITALCOM:EURUSD 60, v6), ta.change(cond) is false on the first
    # bar although cond is true there and cond[1] reads false, and ta.change(cond, 2)
    # is false on the first two bars; from then on it is cond != cond[length].
    cond = bar_index % 4 < 2
    return {
        "chg": 1.0 if ta.change(cond) else 0.0,
        "chg2": 1.0 if ta.change(cond, 2) else 0.0,
    }


def __test_change_of_a_bool_is_false_before_its_history_exists__(runner, dummy_ohlcv_iter):
    """ ta.change(bool) is false until the source has ``length`` bars of history """
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    chg = []
    chg2 = []
    for _ in range(8):
        _, plot = next(run_iter)
        chg.append(plot["chg"])
        chg2.append(plot["chg2"])

    cond = [i % 4 < 2 for i in range(8)]
    assert chg == [0.0] + [1.0 if cond[i] != cond[i - 1] else 0.0 for i in range(1, 8)]
    assert chg2 == [0.0, 0.0] + [1.0 if cond[i] != cond[i - 2] else 0.0 for i in range(2, 8)]
