"""
@pyne

A ``strategy.order`` whose resulting net position cannot be margined is rejected whole.

MEASURED on TradingView (BINANCE:BTCUSDT 30m, 100 % margin): a ``strategy.order``
that would leave a net position worth more than the equity never fills -- from flat,
stacked on a position of its own direction, and as a flip through zero, as a market
and as a limit order. An order that only reduces the position is not judged.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Order Margin Reject",
    overlay=True,
    initial_capital=1000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
)
def main():
    if bar_index == 0:
        strategy.order('TooBig', strategy.long, qty=20)
    if bar_index == 2:
        strategy.order('Fits', strategy.long, qty=5)
    if bar_index == 4:
        strategy.order('Stack', strategy.long, qty=6)
    if bar_index == 6:
        strategy.order('Flip', strategy.short, qty=30)
    if bar_index == 8:
        strategy.order('Reduce', strategy.short, qty=3)
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
def __test_order_exceeding_margin_is_rejected_whole__(script_path, module_key):
    """
    Only the orders whose net position fits 1000 of equity at 100.00 change the position.

    * ``TooBig``: 20 contracts from flat need 2000 -- rejected.
    * ``Fits``: 5 contracts need 500 -- filled.
    * ``Stack``: 5 + 6 contracts need 1100 -- rejected, the 5 stay.
    * ``Flip``: 5 - 30 leaves 25 short, needing 2500 -- rejected, the long is not closed.
    * ``Reduce``: 5 - 3 only shrinks the position -- filled.
    """
    rows = [(100.00, 100.05, 99.95, 100.00)] * 11
    trades, sizes = __test_helper_run(script_path, module_key, rows)

    assert sizes == [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 2.0, 2.0], sizes
    shape = [(t.entry_id, t.exit_id, abs(t.size)) for t in trades]
    assert shape == [('Fits', 'Reduce', 3.0)], shape
