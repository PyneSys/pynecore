"""
@pyne

Regression test: an order whose quantity is not a positive finite number is dropped.

Default sizing has nothing to compute from when ``default_qty_value`` is na (or
when the equity itself has gone NaN), so ``_default_entry_qty`` returns NaN. Both
NaN and infinity pass a ``qty <= 0.0`` guard, so they used to flow into the size
rounding, where ``round()`` killed the whole run with "cannot convert float NaN
to integer" / "cannot convert float infinity to integer".

TradingView cannot produce this state -- ``default_qty_value`` is a const float
and rejects na at compile time (measured: CE10034) -- so this is PyneCore
robustness for hand-written Pyne code, not a TradingView compatibility rule.
"""
from pynecore.lib import bar_index, low, na, plot, script, strategy


# noinspection PyTypeChecker
@script.strategy(
    "Unsizable default qty",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.percent_of_equity,
    default_qty_value=na,
    pyramiding=2,
)
def main():
    # All three placement paths size themselves from the na default value: the
    # market entry sizes at the placement close, the limit entry at its resting
    # price, and strategy.order takes the same route as entry
    if bar_index == 0:
        strategy.entry('M', strategy.long)
    if bar_index == 1:
        strategy.entry('L', strategy.long, limit=low - 5.0)
    if bar_index == 2:
        strategy.order('O', strategy.long)
    # An infinite explicit qty is unsizable the same way: it used to reach the
    # lot rounding, whose round() raised OverflowError
    if bar_index == 3:
        strategy.entry('I', strategy.long, qty=float('inf'))
    if bar_index == 4:
        strategy.order('OI', strategy.long, qty=float('inf'))

    plot(strategy.position_size, "psize")
    plot(strategy.opentrades, "ot")


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


# noinspection PyShadowingNames
def __test_unsizable_qty_places_no_order__(script_path, module_key):
    """
    Every unsizable placement is skipped, and the run survives it.

    Before the fix the very first entry raised ValueError from the size rounding
    inside ``_judge_money_entry``, so the strategy never reached bar 1; the
    infinite quantities raised OverflowError from the same rounding.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='1')
    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=100.0, high=101.0, low=99.0,
              close=100.0, volume=100.0)
        for i in range(6)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)
    position_sizes = []
    open_trades = []
    closed_trades = []
    for _candle, plot_values, new_closed in runner.run_iter():
        position_sizes.append(plot_values['psize'])
        open_trades.append(plot_values['ot'])
        closed_trades.extend(new_closed)

    assert len(position_sizes) == len(bars), "the run must reach the last bar"
    assert all(size == 0.0 for size in position_sizes), f"positions opened: {position_sizes}"
    assert all(count == 0 for count in open_trades), f"trades opened: {open_trades}"
    assert not closed_trades, f"trades closed: {closed_trades}"
