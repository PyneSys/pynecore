"""
@pyne
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Percent Commission Entry",
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    commission_type=strategy.commission.percent,
    commission_value=0.055,
)
def main():
    if bar_index == 0:
        strategy.entry("Long", strategy.long)

    if bar_index == 2:
        strategy.close("Long")


def __test_percent_commission_uses_entry_and_exit_notional__(script_path, module_key, syminfo):
    """Percent commission charges both entry and exit notional, not just the entry leg."""
    import math
    import sys
    from pathlib import Path

    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1704067200
    bars = [
        OHLCV(
            timestamp=base_ts + 0 * 60,
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1.0,
        ),
        OHLCV(
            timestamp=base_ts + 1 * 60,
            open=50000.0,
            high=50000.0,
            low=50000.0,
            close=50000.0,
            volume=1.0,
        ),
        OHLCV(
            timestamp=base_ts + 2 * 60,
            open=50000.0,
            high=51000.0,
            low=50000.0,
            close=51000.0,
            volume=1.0,
        ),
        OHLCV(
            timestamp=base_ts + 3 * 60,
            open=51000.0,
            high=51000.0,
            low=51000.0,
            close=51000.0,
            volume=1.0,
        ),
        OHLCV(
            timestamp=base_ts + 4 * 60,
            open=51000.0,
            high=51000.0,
            low=51000.0,
            close=51000.0,
            volume=1.0,
        ),
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)

    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert len(trades) == 1

    trade = trades[0]
    assert trade.entry_price == 50000.0
    assert trade.exit_price == 51000.0
    assert trade.size == 1.0

    expected_commission = (50000.0 + 51000.0) * 0.055 * 0.01
    assert math.isclose(trade.commission, expected_commission, rel_tol=1e-12, abs_tol=1e-12)
