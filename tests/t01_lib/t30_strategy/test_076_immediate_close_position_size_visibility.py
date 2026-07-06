"""
@pyne
"""
from pynecore.lib import bar_index, na, script, strategy, plot


@script.strategy("Immediate Close Visibility", overlay=True)
def main():
    if bar_index == 3:
        strategy.entry('L', strategy.long, qty=1)
    if bar_index == 6:
        strategy.close_all('X', immediately=True)
    # Plotted AFTER the close_all call: on the close bar TradingView keeps these
    # at their pre-close values for the rest of the bar; they only go flat next bar.
    plot(strategy.position_size, "pos_size")
    plot(strategy.position_avg_price, "avg_price")


# noinspection PyShadowingNames
def __test_immediate_close_position_size_visibility__(csv_reader, runner):
    """ strategy.close_all(immediately=True) settles at the bar close, but the
    script-visible position series stay at their pre-close values for the rest of
    that bar — matching TradingView (and PyneCore's own broker mode). The
    simulator used to fill inline mid-body and zero position_size one bar early,
    breaking any position_size-gated plot on the close bar. """
    plots = []
    with csv_reader('ohlcv.csv', subdir="data") as cr:
        for i, (_candle, plot_data, _new_closed_trades) in enumerate(runner(cr).run_iter()):
            plots.append((plot_data["pos_size"], plot_data["avg_price"]))
            if i >= 7:
                break

    # Entry (bar 3) fills at bar 4 open, so the position is open from bar 4 on.
    assert plots[5][0] != 0            # sanity: position open on the bar before the close
    # bar 6 = the immediate-close bar: position must still read as open in the body
    assert plots[6][0] != 0            # position_size > 0 during the close bar (the fix)
    assert not na(plots[6][1])         # position_avg_price still the entry price
    # bar 7: the close has settled — position is flat
    assert plots[7][0] == 0
    assert na(plots[7][1])
