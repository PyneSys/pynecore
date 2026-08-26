"""
@pyne

Regression test for a ``profit`` distance nearer than the ``limit`` beside it.

Pine v6 keeps both take-profit levels of a ``strategy.exit`` live and fills
whichever the price path reaches first; before v6 the absolute ``limit`` shut
the relative ``profit`` out whenever it held a real price. MEASURED on
TradingView (BINANCE:BTCUSDT 30m, 28915 bars, probes ``ep6_C``/``ep6_D`` vs.
``ep6_E``): with the limit twice as far out the v6 trades are bit-identical to
the same script written without a limit at all, and with the limit nearer 30 of
140 exits move onto it. A pre-v6 source keeps its own meaning through PyneComp
(``converter/semantics.py``), so the runtime carries the v6 rule alone.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Exit Profit Undercuts Limit",
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
        strategy.exit('X', 'L', limit=100.60, profit=20)


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
def __test_profit_distance_fills_before_the_farther_limit__(script_path, module_key):
    """
    The tick distance resolves below the absolute level, so it fills first.

    * bar 0: entry signal (market) -> fills bar 1 open at 100.00.
    * bar 1: quiet bar, the exit is issued (limit=100.60, profit=20 ticks =
      0.20 above the fill, i.e. 100.20).
    * bar 2: open 100.00 -> low 99.90 -> high 101.00. The rising leg crosses
      100.20 before 100.60, so the fill lands at 100.20.

    The pre-v6 rule ignored ``profit`` outright and filled at 100.60.
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
        (100.00, 101.00, 99.90, 100.30),   # bar 2 - both levels crossed
        (100.30, 100.50, 100.20, 100.40),  # bar 3 - tail
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
    assert t.exit_bar_index == 2, f"exit should land on bar 2, got {t.exit_bar_index}"
    assert abs(t.exit_price - 100.20) < 1e-9, (
        f"the profit distance at 100.20 must fill before the limit at 100.60, "
        f"got {t.exit_price}"
    )
