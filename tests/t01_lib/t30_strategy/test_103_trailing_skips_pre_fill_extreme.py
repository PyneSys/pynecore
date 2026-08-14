"""
@pyne

Regression test: the water mark of a mid-bar bracket starts at the entry fill price.

A buy limit fills on a FALLING leg, so on an open -> high -> low -> close bar the
high is already behind the fill. That high belongs to a stretch the bracket was
not live for and must stay out of its water mark.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Trailing Skips Pre Fill Extreme",
    overlay=True,
    initial_capital=1000000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=0,
)
def main():
    if bar_index == 0:
        strategy.entry('L1', strategy.long, limit=99.50)
        strategy.exit('L1X', from_entry='L1', trail_points=10, trail_offset=10)


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
def __test_high_before_the_fill_stays_out_of_the_water_mark__(script_path, module_key):
    """
    An extreme reached before the entry filled must not arm or ratchet the trail.

    * bar 0: a long limit entry at 99.50 and its trailing exit are placed.
    * bar 1: O=100.00 H=100.20 L=99.00 C=99.80 -> |H-O| < |O-L|, so the assumed
      path is open -> high -> low -> close. The descent fills the entry at 99.50
      AFTER the high; only the drop to 99.00 and the rise to the close remain.
      The trail arms on that closing rise at 99.60 and ratchets to 99.70, but the
      bar has no further retrace, so nothing fills here.
    * bar 2: the ride to 99.85 lifts the stop to 99.75, and the drop to 99.40
      fills it.

    Anchoring the water mark to the bar open instead would have armed at the
    pre-fill high 100.20 and filled at 100.10 on bar 1. TradingView does not:
    measured on CAPITALCOM:EURUSD 30m with a buy-limit entry, 0 of 2063 trades
    entered on such a bar closed on it, while 1497 of 1526 entered on an
    open -> low -> high -> close bar did (the high comes after the fill there).
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
        (100.00, 100.20, 99.00, 99.80),   # bar 1 - entry @99.50 after the high
        (99.80, 99.85, 99.40, 99.50),     # bar 2 - ratchet to 99.75, fill there
        (99.50, 99.60, 99.40, 99.45),     # bar 3 - tail
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
    assert abs(t.entry_price - 99.50) < 1e-9, f"entry_price={t.entry_price}"
    assert t.exit_bar_index == 2, (
        f"the high at 100.20 precedes the fill, so bar 1 offers no retrace to fill "
        f"on, got bar {t.exit_bar_index}"
    )
    assert abs(t.exit_price - 99.75) < 1e-9, (
        f"bar 2 rides to 99.85 and the stop follows 10 ticks under it, got {t.exit_price}"
    )
