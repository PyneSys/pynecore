"""
@pyne

Regression test for a ``loss`` distance nearer than the ``stop`` beside it.

The stop-loss side of the v6 rule verified in
``test_118_exit_profit_undercuts_limit``: both levels stay live and the path
decides. MEASURED on TradingView (BINANCE:BTCUSDT 30m, 28915 bars, probe
``ep6_H``): with ``stop`` the nearer of the pair 36 of 141 exits fill on it,
the rest on the ``loss`` distance because the stop expression evaluated to
``na`` on the bar whose order was live when the entry filled.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Exit Loss Undercuts Stop",
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
        strategy.exit('X', 'L', stop=99.40, loss=30)


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
def __test_loss_distance_fills_before_the_farther_stop__(script_path, module_key):
    """
    The tick distance resolves above the absolute level, so it fills first.

    * bar 0: entry signal (market) -> fills bar 1 open at 100.00.
    * bar 1: quiet bar, the exit is issued (stop=99.40, loss=30 ticks = 0.30
      below the fill, i.e. 99.70).
    * bar 2: a down bar, so the path runs open 100.00 -> high 100.05 -> low
      99.00. The falling leg reaches 99.70 before 99.40, so the fill lands at
      99.70.

    The pre-v6 rule ignored ``loss`` outright and filled at 99.40.
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
        (100.00, 100.05, 99.00, 99.20),    # bar 2 - both levels crossed
        (99.20, 99.40, 99.10, 99.30),      # bar 3 - tail
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
    assert abs(t.exit_price - 99.70) < 1e-9, (
        f"the loss distance at 99.70 must fill before the stop at 99.40, "
        f"got {t.exit_price}"
    )
