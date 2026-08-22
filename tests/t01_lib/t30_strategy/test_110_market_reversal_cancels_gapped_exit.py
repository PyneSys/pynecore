"""
@pyne

A MARKET entry that reverses the position cancels a gapped ``strategy.exit``.

MEASURED on TradingView (BINANCE:BTCUSDT 30m, 6/6 events): with a live exit stop
sitting above the next open, a market reversal takes the position to exactly the
new entry's quantity -- the exit leg does NOT fill on top of it. This is the
counterpart of :mod:`test_108_gap_batch_exit_after_reversal`, where the reversal
arrives as a stop order inside the same bar-open gap batch and the exit DOES
fill: only an order queued on an earlier bar cancels the leg.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Market Reversal Cancels Gapped Exit",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
)
def main():
    if bar_index == 0:
        strategy.entry('Long', strategy.long, qty=1.0)
    if bar_index == 2:
        strategy.entry('Short', strategy.short, qty=2.0)
    if strategy.position_size > 0 and bar_index >= 2:
        strategy.exit(id='L Stop', stop=99.0)


def _make_syminfo(period: str = '1'):
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


def _bars():
    from pynecore.types.ohlcv import OHLCV
    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    # 0-2 flat at 100, 3 gaps down past the 99.0 exit stop, 4 flat
    ohlc = [
        (100.0, 100.0, 100.0, 100.0),
        (100.0, 100.0, 100.0, 100.0),
        (100.0, 100.0, 100.0, 100.0),
        (90.0, 90.5, 89.0, 90.0),
        (90.0, 90.5, 89.5, 90.0),
    ]
    return [
        OHLCV(timestamp=base_ts + i * 60_000, open=o, high=h, low=lo, close=c, volume=100.0)
        for i, (o, h, lo, c) in enumerate(ohlc)
    ]


# noinspection PyShadowingNames
def __test_market_reversal_leaves_the_gapped_exit_unfilled__(script_path, module_key):
    """The position stops at the reversal's own -2, with no exit-opened leg.

    Buggy code path: arming the gap batch before the bar's queued market orders
    fill lets the exit survive the reversal and sell its 1 again, giving -3.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner

    sys.modules.pop(module_key, None)

    runner = ScriptRunner(Path(script_path), iter(_bars()), _make_syminfo())
    rows = []
    sizes = []
    for _candle, _plot, new_closed in runner.run_iter():
        sizes.append(runner.script.position.size)
        rows.extend((t.entry_id, t.exit_id, abs(t.size)) for t in new_closed)

    assert sizes[3] == -2.0, f"Expected the market reversal's own -2, got {sizes[3]}."
    assert rows == [('Long', 'Short', 1.0)], rows
