"""
@pyne

Regression test for a tick-based bracket whose entry fills partway through a bar.

A short stop entry at 100.00 and its `strategy.exit(profit=50)` take-profit are
placed together on bar 0. Bar 1's assumed path is open -> high -> low -> close: the
descent reaches the entry at 100.00 first and the take-profit at 99.50 further down
the SAME leg, so TradingView closes the trade on that bar.

Before the fix `profit`/`loss` offsets became concrete prices only in
`_process_at_bar_open`, so an entry filling inside the walk left its bracket with no
price level at all — indexed nowhere, yielded by no leg. The level appeared at the
next bar's open and the exit landed on bar 4.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Tick Bracket On Intrabar Entry",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=0,
)
def main():
    if bar_index == 0:
        strategy.entry('S1', strategy.short, stop=100.00)
        strategy.exit('S1X', from_entry='S1', profit=50)  # ticks -> 100.00 - 0.50


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
def __test_tick_bracket_fills_on_its_entry_bar__(script_path, module_key):
    """
    A `profit=` take-profit reached after its stop entry on the same leg fills on that bar.

    * bar 0: short stop entry at 100.00 plus a 50-tick take-profit are placed.
    * bar 1: O=100.50 H=100.60 L=99.30 C=99.80 -> path open -> high -> low -> close.
      The descent fills the entry at 100.00 and then the take-profit at 99.50.
    * bars 2-6 exist only to prove the exit is not deferred: bar 4 is the next bar
      whose range reaches 99.50, which is where the exit used to land.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='1')
    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms

    rows = [
        # open,   high,   low,    close
        (101.00, 101.20, 100.80, 101.00),  # bar 0 - orders placed
        (100.50, 100.60, 99.30, 99.80),    # bar 1 - entry @100.00 then TP @99.50
        (99.80, 99.90, 99.70, 99.85),      # bar 2
        (99.85, 99.90, 99.60, 99.65),      # bar 3
        (99.60, 99.65, 99.40, 99.45),      # bar 4 - where the deferred exit used to land
        (99.45, 99.50, 98.90, 99.10),      # bar 5
        (99.10, 99.20, 99.00, 99.15),      # bar 6
    ]
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    # Measured on TradingView (CAPITALCOM:EURUSD 30, 20145 bars): 62 of 76
    # intrabar-filled stop entries exit on their own fill bar, and the
    # `profit=<ticks>` and `limit=<price>` spellings give identical trade lists.
    assert len(trades) == 1, f"Expected 1 closed trade, got {len(trades)}"
    trade = trades[0]
    assert trade.entry_bar_index == 1, f"entry bar {trade.entry_bar_index}"
    assert abs(trade.entry_price - 100.0) < 1e-9, f"entry price {trade.entry_price}"
    assert trade.exit_bar_index == 1, (
        f"take-profit deferred to bar {trade.exit_bar_index} — it is reached on the "
        f"entry bar's own descent, after the entry filled"
    )
    assert abs(trade.exit_price - 99.5) < 1e-9, f"exit price {trade.exit_price}"
