"""
@pyne

Regression test for the take-profit limit leg beating the trailing stop.

TradingView's broker emulator moves the price along the assumed intrabar path
and fills whichever exit leg the path reaches first. A take-profit ``limit``
crossed on the rising leg fires BEFORE a trailing fill on the later retrace,
so the trade must exit at the limit level — not at ``watermark - offset``.
Verified against TradingView references on BINANCE:ETHUSDT.P (15m, 174-trade
probe combining ``limit`` + ``trail_price`` + ``trail_offset``): TV filled the
limit at its level on every such bar; PyneCore previously pre-empted it with
the trailing fill at the ratcheted watermark minus the offset.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Limit Beats Trailing Same Bar",
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
        strategy.exit('X', 'L', limit=100.60, trail_price=100.10, trail_offset=30)


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
def __test_limit_fills_before_trailing_retrace__(script_path, module_key):
    """
    The limit is crossed on the rising leg, the trail would fill on the retrace.

    * bar 0: entry signal (market) -> fills bar 1 open at 100.00.
    * bar 1: quiet bar, the exit is issued (limit=100.60, trail_price=100.10,
      offset=30 ticks = 0.30); the high (100.05) stays below the activation
      level, so nothing arms.
    * bar 2: open 100.00 -> low 99.90 -> high 101.00 -> close 100.30. On the
      rising leg the trail arms at 100.10 and the limit is crossed at 100.60;
      the closing leg retraces through 100.70 = high - offset. The limit fired
      earlier on the path, so the fill must land at 100.60.

    The pre-fix behavior rode the trail to the bar high and filled the
    trailing stop at 101.00 - 0.30 = 100.70, pre-empting the limit.
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
        (100.00, 100.05, 99.95, 100.00),   # bar 1 - entry fill, exit issued
        (100.00, 101.00, 99.90, 100.30),   # bar 2 - limit crossed, then retrace
        (100.30, 100.50, 100.20, 100.40),  # bar 3 - tail
    ]
    bars = [
        OHLCV(timestamp=base_ts + i * 60, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert len(trades) == 1, f"expected exactly one closed trade, got {len(trades)}"
    t = trades[0]
    assert t.entry_bar_index == 1, f"entry_bar_index={t.entry_bar_index}"
    assert abs(t.entry_price - 100.00) < 1e-9, f"entry_price={t.entry_price}"
    assert t.exit_bar_index == 2, (
        f"exit should land on bar 2, got {t.exit_bar_index}"
    )
    assert abs(t.exit_price - 100.60) < 1e-9, (
        f"the limit crossed on the rising leg must win over the trailing "
        f"retrace fill at 100.70, got {t.exit_price}"
    )
