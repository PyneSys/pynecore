"""
@pyne
"""
from pynecore.lib import script, bar_index, na, ta


@script.indicator(title="Change Bool History Na", shorttitle="change_bool_history_na",
                  overlay=False, na_bool=True)
def main():
    # Under the three-state bool the missing history is a real bool na: measured on
    # TradingView (CAPITALCOM:EURUSD 60, v5), ta.change(cond) is na on the first bar
    # and cond != cond[length] afterwards.
    cond = bar_index % 4 < 2
    chg = ta.change(cond)
    chg2 = ta.change(cond, 2)
    return {
        "chg": 1.0 if chg else 0.0,
        "chg_na": 1.0 if na(chg) else 0.0,
        "chg2": 1.0 if chg2 else 0.0,
        "chg2_na": 1.0 if na(chg2) else 0.0,
    }


def __test_change_of_a_bool_is_na_before_its_history_exists__(runner, dummy_ohlcv_iter):
    """ Under the three-state bool ta.change(bool) is na until the history exists """
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    rows = [next(run_iter)[1] for _ in range(8)]

    cond = [i % 4 < 2 for i in range(8)]
    assert [r["chg_na"] for r in rows] == [1.0] + [0.0] * 7
    assert [r["chg2_na"] for r in rows] == [1.0, 1.0] + [0.0] * 6
    assert [r["chg"] for r in rows][1:] == [1.0 if cond[i] != cond[i - 1] else 0.0
                                           for i in range(1, 8)]
    assert [r["chg2"] for r in rows][2:] == [1.0 if cond[i] != cond[i - 2] else 0.0
                                            for i in range(2, 8)]
