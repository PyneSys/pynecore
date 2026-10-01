"""
@pyne

A ``calc_on_order_fills`` re-execution that replaces its own ``immediately=True``
close with a plain close of the same id (``close_by_id=True``) keeps the plain
close.

The re-execution after the entry fill enqueues an immediate close of 3 and then
replaces it with a full close. The immediate close a trial run enqueued is undone
before the next order-processing pass, but the replacement that now holds its
order slot is a regular order of that run and fills on the same bar, at the next
node of the bar's path.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "COOF Replaced Immediate Close",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=10,
    calc_on_order_fills=True,
    close_by_id=True,
)
def main():
    if bar_index == 1:
        strategy.entry('G', strategy.short)
    if bar_index == 2 and strategy.position_size == -10:
        strategy.close('G', 'ipart', qty=3, immediately=True)
        strategy.close('G', 'full')
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
def __test_coof_replacement_of_immediate_close_fills_same_bar__(script_path, module_key):
    """The plain close replacing a trial run's immediate close fills on the entry bar."""
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=100.0, high=101.0, low=98.5,
              close=100.0, volume=100.0)
        for i in range(6)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), __test_helper_make_syminfo())
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    sheds = sorted((t.exit_comment, abs(t.size)) for t in trades if t.entry_id == 'G')
    assert sheds == [('flat', 3.0), ('full', 7.0)], sheds
    full = [t for t in trades if t.exit_comment == 'full']
    # Market close placed by the re-execution at the entry fill: fills at that node.
    assert [(t.entry_bar_index, t.exit_bar_index) for t in full] == [(2, 2)], full
