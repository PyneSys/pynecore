"""
@pyne

Regression test for ``strategy.exit`` with ``trail_price`` but no ``trail_offset``.

TradingView only creates a trailing stop when ``trail_offset`` is supplied
alongside the activation level — the Pine reference is explicit: "a
strategy.exit() call must specify a trail_offset argument and either a
trail_price or trail_points argument". A call carrying only ``trail_price``
therefore arms NOTHING: the trailing arguments are ignored and (with no other
trigger present) the whole call is a no-op, exactly like the all-na case
(test_054).

Before the fix PyneCore normalized the missing offset to 0 and created an
offset-0 trailing stop that armed and filled at the activation level — closing
positions TradingView keeps open.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Trail Without Offset Ignored",
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
        # No trail_offset: TradingView ignores the trailing arguments entirely.
        strategy.exit('X', 'L', trail_price=101.00)


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
def __test_trail_price_without_offset_is_ignored__(script_path, module_key):
    """
    The offset-less trailing exit must neither fill nor market-close anything.

    * bar 0: entry signal (market) -> fills bar 1 open at 100.00.
    * bar 1: exit issued with trail_price=101.00 and NO trail_offset.
    * bar 2: runs to 101.50 (through the 101.00 activation) and retraces to
      100.60 — an offset-0 trail would arm and fill here. Nothing may fill.
    * bar 3: tail; the position must still be open at the end of the run.
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
        (100.50, 101.50, 100.40, 100.60),  # bar 2 - pierces 101.00, retraces deep
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

    assert len(trades) == 0, (
        f"the offset-less trailing exit must be a no-op, but {len(trades)} "
        f"trade(s) closed (an offset-0 trailing stop was created)"
    )
