"""
@pyne

A ``strategy.exit`` leg carried across a ``close_all`` + same-id re-entry.

When ``close_all`` flattens a position and the SAME entry id is re-entered in the
same script body, the pending exit leg does not die with the binding the flatten
spent: TradingView hands it to the order still waiting to fill and walks it on the
entry bar itself -- but only for a LONG re-entry. A short one loses the leg.

MEASURED on the wild script `Gap Filling Strategy` (NASDAQ:AAPL 30m, 906 trades):
prev long -> new long carried on all 15 reachable bars, prev short -> new short on
none of 19. Reproduced on synthetic BINANCE:BTCUSDT 30m probes for limit legs
(long 338/338, short 0/359) and stop legs (long 192/192, short 0/725).

The direction asymmetry has the shape of a TradingView bookkeeping bug -- a
``position_size > 0`` guard where ``!= 0`` was meant -- but a bug in the reference
engine is still the reference.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Exit Leg Carries Over close_all",
    overlay=True,
    initial_capital=1000000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=1,
)
def main():
    if bar_index == 0:
        strategy.entry('L', strategy.long)
    if bar_index == 2:
        strategy.close_all()
        strategy.entry('L', strategy.long)
    if bar_index == 6:
        strategy.entry('S', strategy.short)
    if bar_index == 8:
        strategy.close_all()
        strategy.entry('S', strategy.short)
    if bar_index == 11:
        strategy.close_all()
    strategy.exit('XL', 'L', limit=100.50 if bar_index >= 2 else 200.0)
    strategy.exit('XS', 'S', limit=99.50 if bar_index >= 8 else 50.0)


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
def __test_carried_exit_leg_is_long_only__(script_path, module_key):
    """
    The re-entered long exits on its own entry bar, the re-entered short does not.

    * bar 2: ``close_all`` + a second ``L`` entry, ``XL`` re-issued at 100.50.
    * bar 3: both market orders fill at the open (100.00); the carried ``XL`` is
      walked up to 101.00 and fills the fresh long at 100.50.
    * bar 8: the same shape on the short side, ``XS`` re-issued at 99.50.
    * bar 9: the market orders fill at the open (100.00) and the bar reaches 99.00,
      but the carried ``XS`` is gone, so the fresh short survives to the bar-11
      ``close_all``.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    rows = [
        # open,   high,   low,    close
        (100.00, 100.00, 100.00, 100.00),  # bar 0  - long entry signal
        (100.00, 100.05, 99.95, 100.00),   # bar 1  - long fills, XL out of reach
        (100.00, 100.05, 99.95, 100.00),   # bar 2  - close_all + re-entry, XL=100.50
        (100.00, 101.00, 99.90, 100.30),   # bar 3  - flatten, re-fill, carried XL fills
        (100.30, 100.35, 100.25, 100.30),  # bar 4  - flat
        (100.00, 100.05, 99.95, 100.00),   # bar 5  - flat
        (100.00, 100.05, 99.95, 100.00),   # bar 6  - short entry signal
        (100.00, 100.05, 99.95, 100.00),   # bar 7  - short fills, XS out of reach
        (100.00, 100.05, 99.95, 100.00),   # bar 8  - close_all + re-entry, XS=99.50
        (100.00, 100.10, 99.00, 99.50),    # bar 9  - flatten, re-fill, XS must not fill
        (99.60, 99.70, 99.55, 99.60),      # bar 10 - the short stays open
        (99.60, 99.70, 99.55, 99.60),      # bar 11 - close_all
        (99.60, 99.70, 99.55, 99.60),      # bar 12 - the close fills
    ]
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    shape = [(t.entry_bar_index, t.exit_bar_index, round(t.exit_price, 2)) for t in trades]
    assert len(trades) == 4, f"expected four closed trades, got {shape}"

    flattened_long, carried_long, flattened_short, orphaned_short = trades

    assert (flattened_long.entry_bar_index, flattened_long.exit_bar_index) == (1, 3), shape
    assert abs(flattened_long.exit_price - 100.00) < 1e-9, shape

    assert carried_long.entry_bar_index == 3, shape
    assert carried_long.exit_bar_index == 3, (
        f"the carried long leg must fill on the entry bar, got {shape}"
    )
    assert abs(carried_long.exit_price - 100.50) < 1e-9, shape

    assert (flattened_short.entry_bar_index, flattened_short.exit_bar_index) == (7, 9), shape
    assert abs(flattened_short.exit_price - 100.00) < 1e-9, shape

    assert orphaned_short.entry_bar_index == 9, shape
    assert orphaned_short.exit_bar_index == 12, (
        f"the short leg must not carry over the flatten, got {shape}"
    )
    assert abs(orphaned_short.exit_price - 99.60) < 1e-9, shape
