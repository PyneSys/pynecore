"""
@pyne

Regression test for a ``trail_points`` activation nearer than ``trail_price``.

The trailing pair follows the same v6 rule as the take-profit and stop-loss
pairs: both activation levels stay live and the one the path reaches first arms
the stop. MEASURED on TradingView (BINANCE:BTCUSDT 30m, 28915 bars, probes
``ep6_K``/``ep6_L``/``ep6_M``): the pair closes 140 trades where ``trail_price``
alone closes 65 and ``trail_points`` alone 117 — neither level alone accounts
for the exits the pair produces.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Exit Trail Points Undercuts Trail Price",
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
        strategy.exit('X', 'L', trail_price=100.50, trail_points=10, trail_offset=30)


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
def __test_trail_points_arms_below_the_trail_price__(script_path, module_key):
    """
    The tick distance arms the stop on a bar whose high never reaches
    ``trail_price``.

    * bar 0: entry signal (market) -> fills bar 1 open at 100.00.
    * bar 1: quiet bar, the exit is issued (trail_price=100.50, trail_points=10
      ticks = 0.10 above the fill, i.e. 100.10, trail_offset=30 ticks = 0.30).
    * bar 2: an up bar, so the path runs open 100.00 -> low 99.90 -> high
      100.40 -> close. The rising leg arms at 100.10, the water mark ratchets
      to 100.40 and the closing leg retraces into 100.40 - 0.30 = 100.10.

    ``trail_price`` alone never arms here: the bar's high stays below 100.50.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    rows = [
        # open,   high,   low,    close
        (100.00, 100.00, 100.00, 100.00),  # bar 0 - entry signal
        (100.00, 100.05, 99.95, 100.00),   # bar 1 - entry fill, exit issued
        (100.00, 100.40, 99.90, 100.10),   # bar 2 - arms and fills
        (100.10, 100.20, 100.00, 100.10),  # bar 3 - tail
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
    assert abs(t.entry_price - 100.00) < 1e-9, f"entry_price={t.entry_price}"
    assert t.exit_bar_index == 2, (
        f"the trail_points activation must arm and fill on bar 2, got "
        f"{t.exit_bar_index}"
    )
    assert abs(t.exit_price - 100.10) < 1e-9, (
        f"the trailing stop must fill at high - offset = 100.10, got {t.exit_price}"
    )
