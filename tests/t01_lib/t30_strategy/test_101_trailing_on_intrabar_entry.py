"""
@pyne

Regression test for a `trail_points=` bracket whose entry fills partway through a bar.

The trailing walk runs before the price walk and took its first tick from the bar
open, where the bracket was still inactive — so a bracket its own entry activated
mid-bar sat out the whole bar and could not fill until the next one.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Trailing On Intrabar Entry",
    overlay=True,
    initial_capital=1000000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=0,
)
def main():
    if bar_index == 0:
        strategy.entry('L1', strategy.long, stop=100.50)
        strategy.exit('L1X', from_entry='L1', trail_points=50, trail_offset=10)


def _make_syminfo():
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


# noinspection PyShadowingNames
def __test_trailing_arms_and_fills_on_its_entry_bar__(script_path, module_key):
    """
    A trailing bracket activated mid-bar rides the rest of that bar's path.

    * bar 0: a long stop entry at 100.50 and its trailing exit are placed.
    * bar 1: O=100.00 H=101.50 L=99.90 C=100.20 -> |H-O| > |O-L|, so the assumed
      path is open -> low -> high -> close. The rise from the low fills the entry
      at 100.50; the trail arms at 100.50 + 50 ticks = 101.00 further up the SAME
      leg, the water mark rides to the bar high 101.50, and the closing leg pulls
      back through 101.50 - 10 ticks = 101.40.

    Measured on TradingView (CAPITALCOM:EURUSD 30m, 20149 bars, stop entry with
    trail_points=1 / trail_offset=1): all 13750 same-bar exits of intrabar-filled
    long entries landed at exactly `high - trail_offset`.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    rows = [
        # open,   high,   low,    close
        (100.00, 100.10, 99.95, 100.00),  # bar 0 - orders placed
        (100.00, 101.50, 99.90, 100.20),  # bar 1 - entry @100.50, trail fill @101.40
        (100.20, 100.30, 100.10, 100.15),  # bar 2 - tail
        (100.15, 100.25, 100.05, 100.10),  # bar 3 - tail
    ]
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert len(trades) == 1, f"expected exactly one closed trade, got {len(trades)}"
    t = trades[0]
    assert t.entry_bar_index == 1, f"entry_bar_index={t.entry_bar_index}"
    assert abs(t.entry_price - 100.50) < 1e-9, f"entry_price={t.entry_price}"
    assert t.exit_bar_index == 1, (
        f"the trail arms and fills on the entry's own bar, got bar {t.exit_bar_index}"
    )
    assert abs(t.exit_price - 101.40) < 1e-9, (
        f"the water mark starts at the fill price and rides to 101.50, so the stop "
        f"sits at 101.40, got {t.exit_price}"
    )
