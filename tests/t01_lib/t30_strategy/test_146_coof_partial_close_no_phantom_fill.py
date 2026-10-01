"""
@pyne

A partial ``strategy.close()`` that fired its whole slice stays in the exit book
as a zero-size tombstone while its entry is still open, but it no longer fills:
with ``calc_on_order_fills`` a later bar without a real fill runs no re-execution.

Measured on TradingView (CAPITALCOM:EURUSD 60, 467 events): the partial close
placed by the re-execution after the entry fill fills on the entry bar, and a
``close_all`` issued two bars later fills at the open of the NEXT bar, not on its
own bar.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "COOF Partial Close No Phantom Fill",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=10,
    calc_on_order_fills=True,
)
def main():
    if bar_index == 1:
        strategy.entry('G', strategy.short)
    if bar_index == 2 and strategy.position_size <= -10:
        strategy.close('G', comment='part', qty=3)
    if bar_index == 4:
        strategy.close_all('flat')


def __test_helper_make_syminfo(period: str = '1'):
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period=period, type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.0001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames,PyProtectedMember
def __test_close_all_after_partial_close_fills_next_open__(script_path, module_key):
    """The close_all of bar 4 fills at the open of bar 5."""
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=100.0 + i, high=101.5 + i,
              low=98.5 + i, close=100.5 + i, volume=100.0)
        for i in range(7)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), __test_helper_make_syminfo())
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    got = sorted((t.exit_comment, abs(t.size), t.exit_bar_index, t.exit_price) for t in trades)
    assert got == [('flat', 7.0, 5, 105.0), ('part', 3.0, 2, 102.0)], got
