"""
@pyne

Regression test: re-issuing a pending order with a quantity that floors to ZERO
lots removes that order.

TradingView treats the re-issue as a modification of the resting order with the
same id, so a quantity below one lot leaves nothing resting. Measured on
CAPITALCOM:EURUSD 30m (mincontract 0.1): a resting 0.1 buy limit re-issued with
qty 0.05 never fills afterwards (0 fills against 285 without the re-issue), with
the position flat, with a long already open, on the placing bar itself, for a
default-sized re-issue that floors to zero lots, and for ``strategy.order``.
PyneCore used to drop only the new call and leave the old order resting at its
old qty and price (wild corpus "Robot WhiteBox MultiMA": 30 trades against
TradingView's 7).
"""
from pynecore.lib import bar_index, plot, script, strategy


@script.strategy(
    "Sub-lot Re-issue Cancels Pending",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.cash,
    default_qty_value=0.0005,
    pyramiding=5,
)
def main():
    if bar_index == 0:
        strategy.entry('L', strategy.long, qty=0.001, limit=99.0)
        strategy.entry('B', strategy.long, qty=0.001, limit=99.0)
        strategy.entry('B', strategy.long, qty=0.000005, limit=99.0)
        strategy.entry('D', strategy.long, qty=0.001, limit=99.0)
        strategy.order('O', strategy.long, qty=0.001, limit=99.0)
        strategy.entry('C', strategy.long, qty=0.001, limit=99.0)
    if bar_index == 1:
        strategy.entry('L', strategy.long, qty=0.000009, limit=99.0)
        strategy.entry('D', strategy.long, limit=99.0)
        strategy.order('O', strategy.long, qty=0.000009, limit=99.0)
    plot(strategy.position_size, 'psize')
    plot(strategy.opentrades, 'opentrades')


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
def __test_sublot_reissue_cancels_the_pending_order__(script_path, module_key):
    """
    Only the control order, never re-issued, fills once price reaches the limit.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo()
    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms

    # The limit (99.0) is out of reach until bar 3, whose low touches 98.5.
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=100.0, high=100.5,
              low=98.5 if i == 3 else 99.5, close=100.0, volume=100.0)
        for i in range(6)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)
    sizes = []
    opentrades = []
    for _candle, plot_data, _new_closed in runner.run_iter():
        sizes.append(plot_data['psize'])
        opentrades.append(plot_data['opentrades'])

    assert sizes[2] == 0.0, f"nothing may fill before the limit is reached: {sizes[2]}"
    assert abs(sizes[-1] - 0.001) < 1e-12, f"only the control order may fill: {sizes[-1]}"
    assert opentrades[-1] == 1, f"expected 1 open trade, got {opentrades[-1]}"
