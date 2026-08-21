"""
@pyne
"""
from pynecore import Series
from pynecore.lib import script, bar_index, ta


@script.indicator(title="Cross Plateau", shorttitle="cross_plateau", overlay=False)
def main():
    # ``ta.cross`` is NOT ``crossover or crossunder``: it remembers the last
    # tolerantly-unequal relation and stays UNARMED until one exists, so a
    # from-the-start equality plateau never fires either way, while an equality
    # run keeps the direction a prior strict relation armed. Expected values
    # measured on TradingView (BINANCE:BTCUSDT 30m, 2026-08): the crossover /
    # crossunder legs, which start armed, DO fire on the from-the-start plateau.

    # U: equal 0..5, then source1 jumps ABOVE at bar 6
    u1: Series[float] = 2.0 if bar_index >= 6 else 1.0
    u2 = 1.0
    # D: equal 0..5, then source1 jumps BELOW at bar 6
    d1: Series[float] = 0.0 if bar_index >= 6 else 1.0
    d2 = 1.0
    # F: strictly below 0..2, equal 3..5, above at bar 6 (prior below arms the up-cross)
    f1: Series[float] = 0.0 if bar_index < 3 else (1.0 if bar_index < 6 else 2.0)
    f2 = 1.0
    # G: strictly above 0..2, equal 3..5, below at bar 6 (prior above arms the down-cross)
    g1: Series[float] = 2.0 if bar_index < 3 else (1.0 if bar_index < 6 else 0.0)
    g2 = 1.0
    # B: strictly ABOVE 0..2, equal 3..5, above again at bar 6. crossover fires
    # (only the immediate previous bar matters), but ta.cross stays armed for a
    # DOWN cross so it does NOT fire.
    b1: Series[float] = 2.0 if bar_index < 3 else (1.0 if bar_index < 6 else 2.0)
    b2 = 1.0
    # W: mirror of B for crossunder -- below, equal, below again
    w1: Series[float] = 0.0 if bar_index < 3 else (1.0 if bar_index < 6 else 0.0)
    w2 = 1.0

    return {
        "u_cross": 1.0 if ta.cross(u1, u2) else 0.0,
        "u_over": 1.0 if ta.crossover(u1, u2) else 0.0,
        "d_cross": 1.0 if ta.cross(d1, d2) else 0.0,
        "d_under": 1.0 if ta.crossunder(d1, d2) else 0.0,
        "f_cross": 1.0 if ta.cross(f1, f2) else 0.0,
        "g_cross": 1.0 if ta.cross(g1, g2) else 0.0,
        "b_over": 1.0 if ta.crossover(b1, b2) else 0.0,
        "b_cross": 1.0 if ta.cross(b1, b2) else 0.0,
        "w_under": 1.0 if ta.crossunder(w1, w2) else 0.0,
        "w_cross": 1.0 if ta.cross(w1, w2) else 0.0,
    }


def __test_cross_plateau__(runner, dummy_ohlcv_iter):
    """ ta.cross stays unarmed through a from-the-start equality plateau """
    keys = ("u_cross", "u_over", "d_cross", "d_under", "f_cross", "g_cross",
            "b_over", "b_cross", "w_under", "w_cross")
    fired = {k: [] for k in keys}
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    for i in range(8):
        _, plot = next(run_iter)
        for k in fired:
            if plot[k]:
                fired[k].append(i)

    # cross does NOT fire on a from-the-start equality plateau, either direction
    assert fired["u_cross"] == []
    assert fired["d_cross"] == []
    # crossover / crossunder start armed, so they DO fire on the same plateau jump
    assert fired["u_over"] == [6]
    assert fired["d_under"] == [6]
    # a strict relation before an equality plateau arms the jump after it
    assert fired["f_cross"] == [6]
    assert fired["g_cross"] == [6]
    # crossover only reads the immediate previous bar: an equality plateau entered
    # from ABOVE still fires the up-cross...
    assert fired["b_over"] == [6]
    assert fired["w_under"] == [6]
    # ...but ta.cross stays armed for the opposite direction, so it does NOT fire
    assert fired["b_cross"] == []
    assert fired["w_cross"] == []
