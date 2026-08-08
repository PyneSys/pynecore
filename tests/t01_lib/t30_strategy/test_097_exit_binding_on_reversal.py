"""
@pyne

Which position a `from_entry`-less `strategy.exit` binds to.
"""
# TradingView binds an exit without `from_entry` to the OPEN position; a still-pending
# entry order is only its target when the strategy is flat. The two cases separate on a
# REVERSAL, where both exist at once under different ids: the bracket covers the position
# being reversed out of, and the position the reversal opens gets its own bracket from the
# NEXT bar's script run -- so its stop cannot fire on the bar the reversal filled.
#
# MEASURED on TradingView (CAPITALCOM:EURUSD 60, "Technical Ratings Strategy", 580 trades,
# `strategy.exit(loss = 3 * atr(14) / mintick)`). Of the entries whose stop level was
# breached on their own fill bar, all 5 opened from FLAT exited on that bar and the 1
# opened by a reversal held: a clean 6/6 split.
#
# The bars below are built so a single run shows both branches and the re-arming that
# follows: bar 2 breaches a flat entry's stop on its fill bar, bar 7 breaches a reversal
# entry's stop on its fill bar, and bar 8 breaches it again once the bracket is live.
from pynecore.lib import bar_index, plot, script, strategy


@script.strategy(
    "Exit binding on reversal",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    margin_long=0,
    margin_short=0,
)
def main():
    if bar_index == 1:
        strategy.entry('L1', strategy.long)
    if bar_index == 4:
        strategy.entry('L2', strategy.long)
    if bar_index == 6:
        strategy.entry('S', strategy.short)

    # 100 ticks at a 0.01 mintick: a 1.00 stop distance from the entry price.
    strategy.exit('x', loss=100)

    plot(strategy.position_size, "psize")
    plot(strategy.closedtrades, "closed")


BASE_TS = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
DAY_MS = 86_400_000

# Every entry fills at 100.0, so a long stops at 99.0 and a short at 101.0.
CHART_BARS = [
    #  open,   high,    low,  close
    (100.0, 100.0, 100.0, 100.0),  # 0
    (100.0, 100.0, 100.0, 100.0),  # 1 signal: enter L1 while FLAT
    (100.0, 100.0, 98.0, 100.0),   # 2 L1 fills at 100, low 98 breaches its 99 stop
    (100.0, 100.0, 100.0, 100.0),  # 3
    (100.0, 100.0, 100.0, 100.0),  # 4 signal: enter L2 while FLAT
    (100.0, 100.0, 100.0, 100.0),  # 5 L2 fills at 100
    (100.0, 100.0, 100.0, 100.0),  # 6 signal: enter S -- a REVERSAL, L2 is open
    (100.0, 102.0, 100.0, 100.0),  # 7 S fills at 100, high 102 breaches its 101 stop
    (100.0, 102.0, 100.0, 100.0),  # 8 same breach, now with the bracket armed
    (100.0, 100.0, 100.0, 100.0),  # 9
]


def _run(runner) -> list[dict]:
    """Run the strategy over CHART_BARS and collect the plot row of each bar."""
    from pynecore.types.ohlcv import OHLCV

    bars = []
    for i, (o, h, low, c) in enumerate(CHART_BARS):
        bars.append(OHLCV(timestamp=BASE_TS + i * DAY_MS, open=o, high=h, low=low, close=c,
                          volume=100.0))
    r = runner(iter(bars), syminfo_override={'mintick': 0.01, 'pricescale': 100,
                                             'mincontract': 1.0})
    rows = []
    for _candle, plot_values, _closed in r.run_iter():
        rows.append(dict(plot_values))
    return rows


def __test_flat_entry_stop_fires_on_its_fill_bar__(runner):
    """With nothing open, the bracket binds to the pending entry and is live at the fill"""
    rows = _run(runner)
    assert rows[2]['psize'] == 0.0
    assert rows[2]['closed'] == 1


def __test_reversal_entry_stop_cannot_fire_on_its_fill_bar__(runner):
    """The bracket covered the position being reversed out of, not the new one"""
    rows = _run(runner)
    assert rows[6]['psize'] == 1.0     # L2 still open, reversal only queued
    assert rows[7]['psize'] == -1.0    # reversal filled and the short survived its bar
    assert rows[7]['closed'] == 2      # only L2 closed on that bar


def __test_reversal_entry_stop_arms_from_the_next_bar__(runner):
    """The bracket issued at the fill bar's close covers the short from then on"""
    rows = _run(runner)
    assert rows[8]['psize'] == 0.0
    assert rows[8]['closed'] == 3
