"""
@pyne

Regression test for slippage on a limit order the bar gapped through.

TradingView applies slippage to stop legs and to genuine market orders, never
to a limit leg — a limit fills at its own level or better, and a bar that gaps
past it fills at the open. Measured on CME_MINI:ES1! 30m over 30575 bars with
two run pairs (limit exit and limit entry, each ``slippage=0`` vs
``slippage=1``): all 3057 gapped exit fills and 3058 gapped entry fills landed
exactly on the bar open, and the slippage setting moved none of them.

PyneCore reclassifies such an order to a market order at bar open, and used to
slip it like any other market fill, landing one tick on the adverse side.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Gap Limit Fill Takes No Slippage",
    overlay=True,
    initial_capital=1000000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=1,
    slippage=1,
)
def main():
    if bar_index == 0 and strategy.position_size == 0:
        strategy.entry('L', strategy.long)
    if strategy.position_size > 0:
        strategy.exit('X', 'L', limit=100.50)


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
def __test_gapped_limit_exit_fills_at_the_open__(script_path, module_key):
    """
    The take-profit limit is 100.50 and bar 2 opens above it at 101.00.

    * bar 0: entry signal (market) -> fills bar 1 open at 100.00 plus one tick
      of slippage, because a genuine market order IS slipped: 100.01.
    * bar 1: quiet bar, the exit is issued with limit=100.50.
    * bar 2: opens at 101.00, i.e. above the limit. The order is reclassified
      to a market order at bar open and must fill at max(limit, open) = 101.00.

    The pre-fix behavior charged the market-order slippage to the limit leg and
    filled at 101.00 - 0.01 = 100.99.
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
        (101.00, 101.20, 100.90, 101.10),  # bar 2 - gaps above the limit
        (101.10, 101.20, 101.00, 101.10),  # bar 3 - tail
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
    assert abs(t.entry_price - 100.01) < 1e-9, (
        f"a market entry keeps its slippage, expected 100.01, got {t.entry_price}"
    )
    assert t.exit_bar_index == 2, f"exit should land on bar 2, got {t.exit_bar_index}"
    assert abs(t.exit_price - 101.00) < 1e-9, (
        f"a limit the bar gapped through fills at the open with no slippage, "
        f"expected 101.00, got {t.exit_price}"
    )
