"""
@pyne

Regression test: ``trail_points`` WITHOUT ``trail_offset`` arms an offset-0 trail.

TradingView requires the offset only at COMPILE time and only when the trailing
pair would be the exit's sole trigger (verified live: both trailing-only shapes
are rejected with "strategy.exit must have at least one of ... profit, limit,
loss, stop or one of the following pairs: trail_offset and trail_price /
trail_points", so no runtime behavior exists for them). Alongside
``stop``/``limit`` the call compiles, and the TV reference trade exports behind
the pynecomp bracket trail probes (88-91) prove the trailing stop IS live with
an offset of 0 ticks: the trade exits at the activation level on the piercing
bar, ahead of the far stop/limit legs. A previous fix misread the compile rule
as a runtime rule and dropped the trailing arguments whenever ``trail_offset``
was missing, pushing these exits onto the stop/limit legs bars later.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Trail Without Offset Arms At Activation",
    overlay=True,
    initial_capital=1000000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=1,
)
def main():
    if bar_index == 0 and strategy.position_size == 0:
        strategy.entry('L', strategy.long)
    if strategy.position_size > 0:
        # No trail_offset: with stop/limit siblings this is the TV-legal shape;
        # the trailing leg must arm with offset 0, not be ignored.
        strategy.exit('X', 'L', stop=95.00, limit=110.00, trail_points=100)


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
def __test_trail_points_without_offset_arms_offset0__(script_path, module_key):
    """
    The offset-less trailing leg must fill at the activation level.

    * bar 0: entry signal (market) -> fills bar 1 open at 100.00.
    * bar 1: exit issued with stop=95.00, limit=110.00, trail_points=100
      (activation 100.00 + 100 * 0.01 = 101.00) and NO trail_offset.
    * bar 2: runs to 101.50 through the 101.00 activation: the offset-0 trail
      arms and fills right at 101.00. Were the trailing arguments dropped,
      nothing could fill here (stop/limit are far) and the position would
      ride until one of them is hit.
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
        (100.00, 100.20, 99.90, 100.10),   # bar 1 - entry fill, exit issued
        (100.50, 101.50, 100.40, 100.60),  # bar 2 - pierces 101.00: arm + fill
        (100.60, 100.70, 100.50, 100.60),  # bar 3 - tail
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
        f"expected the offset-0 trail to close the trade, got {len(trades)} "
        f"closed trade(s) (0 means the trailing arguments were dropped)"
    )
    t = trades[0]
    assert t.exit_bar_index == 2, (
        f"the trail must arm and fill on the piercing bar 2, got bar {t.exit_bar_index}"
    )
    assert abs(t.exit_price - 101.00) < 1e-9, (
        f"offset-0 fill should land at the 101.00 activation level, got {t.exit_price}"
    )
