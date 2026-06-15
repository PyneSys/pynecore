"""
@pyne

Regression test for the sub-lot margin-call shortfall minimum liquidation.

On fractional-lot symbols TradingView liquidates 4x the cover amount in lot units.
When the shortfall is smaller than one lot's notional (the cover truncates to
zero lots), it closes one whole contract, capped by the current position size.
For sub-1.0 BTC positions that still wipes the whole trade.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Margin Call Sub-Lot Wipe",
    overlay=True,
    initial_capital=40,
    default_qty_type=strategy.percent_of_equity,
    default_qty_value=100,
    margin_long=100,
    margin_short=100,
    pyramiding=1,
)
def main():
    if bar_index == 0 and strategy.position_size == 0:
        strategy.entry('S', strategy.short)


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
def __test_sublot_shortfall_wipes_whole_position__(script_path, module_key):
    """
    A margin shortfall under one lot's notional closes this entire sub-1.0 short.

    * bar 0: entry signal (market) -> fills bar 1 open at 100.00 with
      qty = 100% equity = 40 / 100.00 = 0.4 (available funds exactly zero).
    * bar 2: the high ticks one mintick against the short (100.01). Available
      funds at the high: equity 39.996 - margin 40.004 = -0.008, less than one
      lot's notional (0.0001 * 100.01 ~ 0.01), so the cover truncates to zero
      lots. TradingView closes the WHOLE 0.4 position at the high in a single
      margin-call fill.

    The pre-fix behavior liquidated max(1, 0*4) = 1 lot = 0.0001 and kept the
    rest of the position open.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1704067200  # 2024-01-01 00:00:00 UTC
    rows = [
        # open,   high,   low,    close
        (100.00, 100.00, 100.00, 100.00),  # bar 0 - entry signal
        (100.00, 100.00, 99.95, 100.00),   # bar 1 - entry fill, AF stays >= 0
        (100.00, 100.01, 99.95, 99.96),    # bar 2 - sub-lot shortfall at high
        (99.96, 100.00, 99.90, 99.95),     # bar 3 - tail
    ]
    bars = [
        OHLCV(timestamp=base_ts + i * 60, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert len(trades) == 1, (
        f"expected the margin call to close the whole position in ONE fill, "
        f"got {len(trades)} closed trades: {trades}"
    )
    t = trades[0]
    assert t.exit_comment == 'Margin call', f"exit_comment={t.exit_comment!r}"
    assert t.exit_bar_index == 2, f"exit_bar_index={t.exit_bar_index}"
    assert abs(abs(t.size) - 0.4) < 1e-9, (
        f"the margin call should liquidate the full 0.4, got {t.size}"
    )
    assert abs(t.exit_price - 100.01) < 1e-9, f"exit_price={t.exit_price}"
