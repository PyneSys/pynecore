"""
@pyne
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Futures Pointvalue PnL",
    initial_capital=1000000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
)
def main():
    if bar_index == 0:
        strategy.entry("Long", strategy.long)

    if bar_index == 2:
        strategy.close("Long")


def __test_strategy_pnl_uses_syminfo_pointvalue__(script_path, module_key, syminfo):
    import math
    import sys
    from pathlib import Path

    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo.type = "futures"
    syminfo.pointvalue = 20.0
    syminfo.mintick = 0.25
    syminfo.pricescale = 4

    base_ts = 1704067200
    bars = [
        OHLCV(timestamp=base_ts + 0 * 60, open=18000.0, high=18000.0, low=18000.0, close=18000.0, volume=1.0),
        OHLCV(timestamp=base_ts + 1 * 60, open=18000.0, high=18000.0, low=18000.0, close=18000.0, volume=1.0),
        OHLCV(timestamp=base_ts + 2 * 60, open=18000.0, high=18010.0, low=18000.0, close=18010.0, volume=1.0),
        OHLCV(timestamp=base_ts + 3 * 60, open=18010.0, high=18010.0, low=18010.0, close=18010.0, volume=1.0),
        OHLCV(timestamp=base_ts + 4 * 60, open=18010.0, high=18010.0, low=18010.0, close=18010.0, volume=1.0),
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)

    trades = []
    open_profits = []
    for _candle, _plot, new_closed in runner.run_iter():
        open_profits.append(runner.script.position.openprofit)
        trades.extend(new_closed)

    assert len(trades) == 1

    trade = trades[0]
    assert trade.entry_price == 18000.0
    assert trade.exit_price == 18010.0
    assert trade.size == 1.0

    expected_profit = (18010.0 - 18000.0) * 1.0 * syminfo.pointvalue
    expected_profit_percent = expected_profit / (18000.0 * syminfo.pointvalue) * 100.0

    assert math.isclose(open_profits[2], expected_profit, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(trade.profit, expected_profit, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(trade.profit_percent, expected_profit_percent, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(runner.script.position.netprofit, expected_profit, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(runner.script.position.equity, 1000000.0 + expected_profit, rel_tol=1e-12, abs_tol=1e-12)
