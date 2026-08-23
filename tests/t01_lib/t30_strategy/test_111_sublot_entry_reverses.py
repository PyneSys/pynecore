"""
@pyne

Regression test: an entry whose quantity snaps to ZERO lots still reverses.

TradingView floors an explicit ``strategy.entry`` quantity onto the instrument's
mincontract grid. A quantity below one lot floors to nothing -- but the order is
not dropped: it still closes an opposite open position and simply opens nothing.

Measured on BINANCE:BTCUSDT 30m (mincontract 1e-5), probe ``subq``: a 9e-6 short
against a 0.001 long books a closed trade and leaves ``strategy.position_size``
at 0, while the same 1.9e-5 short issued from flat opens exactly one lot
(-1e-5). PyneCore used to drop the sub-lot order outright, so the long stayed
open and every later signal ran against a position TradingView had already
closed (wild corpus "Built-in Kelly ratio for dynamic position sizing": 366
trades against TradingView's 284).
"""
from pynecore.lib import bar_index, plot, script, strategy


@script.strategy(
    "Sub-lot Entry Reversal",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=0,
)
def main():
    if bar_index == 0:
        strategy.entry('L', strategy.long, qty=0.001)
    if bar_index == 2:
        strategy.entry('S', strategy.short, qty=0.000009)
    if bar_index == 4:
        strategy.entry('S2', strategy.short, qty=0.000019)
    plot(strategy.position_size, 'psize')


def _make_syminfo():
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period='1', type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.00001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames
def __test_sublot_entry_closes_the_opposite_position__(script_path, module_key):
    """
    The sub-lot short closes the long and opens nothing; from flat it opens one lot.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo()
    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms

    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=100.0, high=100.5, low=99.5,
              close=100.0, volume=100.0)
        for i in range(8)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)
    sizes = []
    trades = []
    for _candle, plot_data, new_closed in runner.run_iter():
        trades.extend(new_closed)
        sizes.append(plot_data['psize'])

    assert len(trades) == 1, f"Expected 1 closed trade, got {len(trades)}"
    assert abs(abs(trades[0].size) - 0.001) < 1e-12, f"closed size {trades[0].size}"
    # bar 1: the long fills; bar 3: the sub-lot short closes it and opens nothing;
    # bar 5: the 1.9e-5 short from flat floors to exactly one lot.
    assert abs(sizes[1] - 0.001) < 1e-12, f"after long fill: {sizes[1]}"
    assert sizes[3] == 0.0, f"after sub-lot reversal: {sizes[3]} (expected flat)"
    assert abs(sizes[5] + 0.00001) < 1e-12, f"after flat sub-lot-floor short: {sizes[5]}"
