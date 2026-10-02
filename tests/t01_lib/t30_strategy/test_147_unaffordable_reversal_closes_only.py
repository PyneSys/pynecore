"""
@pyne

A market ``strategy.entry`` that cannot be margined when it is placed still closes
the opposite position.

MEASURED on TradingView (BINANCE:BTCUSDT 30m, a long held at 100 % of equity and a
short ``strategy.entry`` with an explicit ``qty`` worth 1.0002x, 1.02x and 5x the
equity at the placement close): in every cycle the long is closed at the next open
by a fill named after the short entry, and no short opens.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Unaffordable Reversal Closes Only",
    overlay=True,
    initial_capital=1000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
)
def main():
    if bar_index == 0:
        strategy.entry('L', strategy.long, qty=1)
    if bar_index == 2:
        strategy.entry('S', strategy.short, qty=20)
    return {"psize": strategy.position_size}


def __test_helper_make_syminfo():
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period='1', type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.0001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


def __test_helper_run(script_path, module_key, rows):
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]
    runner = ScriptRunner(Path(script_path), iter(bars), __test_helper_make_syminfo())
    trades = []
    sizes = []
    for _candle, plot, new_closed in runner.run_iter():
        trades.extend(new_closed)
        sizes.append(round(plot["psize"], 6))
    return trades, sizes


# noinspection PyShadowingNames
def __test_unaffordable_reversal_only_closes__(script_path, module_key):
    """
    The 20-contract short (2000 against 1000 of equity) flattens the long and stays out.

    * bar 1: the long fills at the open (100.00).
    * bar 2: the short entry is placed; its own 20 contracts cannot be margined.
    * bar 3: the long closes at the open (100.00) under the id ``S``; no short opens.
    """
    rows = [
        # open,   high,   low,    close
        (100.00, 100.05, 99.95, 100.00),  # bar 0 - long placed
        (100.00, 100.05, 99.95, 100.00),  # bar 1 - long fills
        (100.00, 100.05, 99.95, 100.00),  # bar 2 - oversized short placed
        (100.00, 100.05, 99.95, 100.00),  # bar 3 - long closed, nothing opened
        (100.00, 100.05, 99.95, 100.00),  # bar 4 - flat
        (100.00, 100.05, 99.95, 100.00),  # bar 5 - flat
    ]
    trades, sizes = __test_helper_run(script_path, module_key, rows)

    shape = [(t.entry_id, t.exit_id, t.entry_bar_index, t.exit_bar_index) for t in trades]
    assert shape == [('L', 'S', 1, 3)], shape
    assert abs(trades[0].exit_price - 100.00) < 1e-9, trades[0].exit_price
    assert sizes == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0], sizes
