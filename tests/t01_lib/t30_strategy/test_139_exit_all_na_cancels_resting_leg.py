"""
@pyne

Regression test for ``strategy.exit`` with ALL triggers na after a same-id leg rests.

Such a call arms no trigger, but it is not a no-op either: it replaces the resting
leg of the same id with an order that can never fill, which cancels the leg.
MEASURED on TradingView (CAPITALCOM:EURUSD 60): after a trailing exit, a same-id
``strategy.exit(id, from_entry, stop = na)`` yields the trade list of
``strategy.cancel(id)``, with or without ``qty_percent``.

The cancelling call here asks for ``qty_percent=50`` of a one-contract trade on a
one-contract lot grid, so its own slice rounds to nothing -- the leg must still go.
"""
from pynecore.lib import bar_index, na, script, strategy


# noinspection PyTypeChecker
@script.strategy(
    "Exit All-NA Cancels Resting Leg",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=0,
)
def main():
    if bar_index == 0:
        strategy.entry('E', strategy.long)
    if bar_index == 1:
        strategy.exit('X', from_entry='E', limit=110.0)
    if bar_index == 2:
        strategy.exit('X', from_entry='E', qty_percent=50, stop=na)


def __test_helper_make_syminfo(period: str = '1'):
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period=period, type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=1,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames
def __test_all_na_exit_cancels_the_resting_leg__(script_path, module_key):
    """
    The limit leg placed on bar 1 must not fire once the all-na call replaced it.

    * bar 0: entry signal.
    * bar 1: E fills @100; exit X rests at limit 110.
    * bar 2: the all-na exit X cancels the resting leg.
    * bar 3: high 115 would fill the limit 110 if the leg were still there.

    Keeping the leg closes the trade @110 on bar 3; cancelling it leaves the
    position open to the last bar.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = __test_helper_make_syminfo(period='1')
    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms

    rows = [
        # open,  high,   low,   close
        (100.0, 100.5, 99.5, 100.0),  # bar 0 - entry signal
        (100.0, 100.5, 99.5, 100.0),  # bar 1 - E fills @100, X rests at limit 110
        (100.0, 100.5, 99.5, 100.0),  # bar 2 - all-na X cancels the leg
        (100.0, 115.0, 99.5, 112.0),  # bar 3 - high 115 would fill limit 110
        (112.0, 112.5, 111.5, 112.0),  # bar 4 - tail
    ]
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert trades == [], (
        f"the cancelled limit leg still closed {len(trades)} trade(s): "
        f"{[(t.exit_bar_index, t.exit_price) for t in trades]}"
    )
