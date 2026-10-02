"""
@pyne

A ``from_entry``-less exit of a margin-rejected entry is re-aimed at the next bare position.

MEASURED on TradingView (BINANCE:BTCUSDT 30m, a 100 % long rejected at its fill with
a ``from_entry``-less stop/limit exit, then a half-size entry five bars later, 3
rejected cycles per probe): a bare long is exited at the orphan's stop and a bare
short is closed by it on its own fill bar, while the same entries issued with a
``from_entry`` bracket of their own never see the orphan, in either direction.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Orphan Exit Re-aimed",
    overlay=True,
    initial_capital=1000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
)
def main():
    if bar_index == 0:
        strategy.entry('L', strategy.long, qty=10)
        strategy.exit('XL', stop=95.0, limit=110.0)
    if bar_index == 3:
        strategy.entry('L2', strategy.long, qty=5)
    if bar_index == 6:
        strategy.close_all()
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
def __test_orphan_exit_closes_bare_position__(script_path, module_key):
    """
    ``L2`` has no exit of its own, so the rejected ``L``'s stop closes it.

    * bar 1: ``L`` needs 1010 at the open (101.00) against 1000 of equity -- rejected.
    * bar 4: ``L2`` fills at the open (101.00).
    * bar 5: the low passes 95.00 and ``XL`` closes ``L2`` there.
    """
    rows = [
        # open,   high,   low,    close
        (100.00, 100.05, 99.95, 100.00),  # bar 0 - L and its exit placed
        (101.00, 101.05, 100.95, 101.00),  # bar 1 - L rejected at the open
        (101.00, 101.05, 100.95, 101.00),  # bar 2 - flat
        (101.00, 101.05, 100.95, 101.00),  # bar 3 - L2 placed
        (101.00, 101.05, 100.95, 101.00),  # bar 4 - L2 fills
        (101.00, 101.05, 94.00, 96.00),  # bar 5 - the orphan's stop is passed
        (96.00, 96.05, 95.95, 96.00),  # bar 6 - close_all placed
        (96.00, 96.05, 95.95, 96.00),  # bar 7
        (96.00, 96.05, 95.95, 96.00),  # bar 8
    ]
    trades, sizes = __test_helper_run(script_path, module_key, rows)

    assert sizes == [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0], sizes
    shape = [(t.entry_id, t.exit_id, t.entry_bar_index, t.exit_bar_index) for t in trades]
    assert shape == [('L2', 'XL', 4, 5)], shape
    assert abs(trades[0].exit_price - 95.00) < 1e-9, trades[0].exit_price
