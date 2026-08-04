"""
@pyne

Control test for the gap-fill slippage rule: the STOP leg still gets slipped.

The companion test (081) asserts that a limit order the bar gapped through
fills at the open with no slippage. The rule is leg-specific, not gap-specific:
a stop leg reached by the same gap keeps its slippage, exactly as the intrabar
``_check_low_stop`` walk would charge it. This test pins that half so the
limit-side fix cannot drift into suppressing stop slippage as well.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Gap Stop Fill Keeps Slippage",
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
        strategy.exit('X', 'L', stop=99.50)


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
def __test_gapped_stop_exit_keeps_its_slippage__(script_path, module_key):
    """
    The protective stop is 99.50 and bar 2 opens below it at 99.00.

    * bar 0: entry signal (market) -> fills bar 1 open at 100.00 + 0.01.
    * bar 1: quiet bar, the exit is issued with stop=99.50.
    * bar 2: opens at 99.00, i.e. below the stop. The order is reclassified to
      a market order at bar open and fills at min(stop, open) = 99.00 minus one
      tick of slippage: 98.99.
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
        (99.00, 99.10, 98.80, 98.90),      # bar 2 - gaps below the stop
        (98.90, 99.00, 98.80, 98.90),      # bar 3 - tail
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
    assert abs(t.entry_price - 100.01) < 1e-9, f"entry_price={t.entry_price}"
    assert t.exit_bar_index == 2, f"exit should land on bar 2, got {t.exit_bar_index}"
    assert abs(t.exit_price - 98.99) < 1e-9, (
        f"a stop the bar gapped through keeps its slippage, expected 98.99, "
        f"got {t.exit_price}"
    )
