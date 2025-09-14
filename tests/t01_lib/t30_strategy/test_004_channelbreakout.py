# All Pyne code must start with "@pyne" magic comment.
"""
@pyne
"""
# Import Series type
from pynecore import Series
# You can import Pine Script compatible functions and properties from pynecore.lib
from pynecore.lib import script, strategy, input, ta, high, low, close, na, syminfo


# You can define a strategy or indicator with decorator
@script.strategy("ChannelBreakOutStrategy", overlay=True)
# Every Pyne code must have a main function
def main(
        # Inputs are defined in main function arguments
        length: int = input.int(title="Length", minval=1, maxval=1000, defval=5)
):
    # Calculate the upper and lower bounds
    upBound = ta.highest(high, length)
    downBound = ta.lowest(low, length)

    # Entry conditions with stop orders
    # Only place long entry if we have enough historical data
    if not na(close[length]):
        strategy.entry('ChBrkLE', strategy.long,
                       stop=upBound + syminfo.mintick,
                       comment='ChBrkLE')

    # Always place short entry
    strategy.entry('ChBrkSE', strategy.short,
                   stop=downBound - syminfo.mintick,
                   comment='ChBrkSE')


# noinspection PyShadowingNames
def __test_channelbreakout__(csv_reader, runner, strat_equity_comparator):
    """ Channel Breakout Strategy """
    with csv_reader('ohlcv.csv', subdir="data") as cr, \
            csv_reader('channelbreakout_trades.csv', subdir="data") as cr_equity:
        r = runner(cr, syminfo_override=dict(timezone="US/Eastern"))
        equity_iter = iter(cr_equity)
        for i, (candle, plot, new_closed_trades) in enumerate(r.run_iter()):
            for trade in new_closed_trades:
                good_entry = next(equity_iter)
                good_exit = next(equity_iter)
                strat_equity_comparator(trade, good_entry.extra_fields, good_exit.extra_fields)
