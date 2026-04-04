"""
@pyne

Bar magnifier real-data test against TradingView reference.

This strategy uses SMA(3)/SMA(10) crossover with limit/stop exits on BINANCE:BTCUSDT 1D chart.
The 60-minute sub-bar data is used for magnified order fills (TV mapping: 1D → 60m).

The reference trades were exported from TradingView with use_bar_magnifier=true.
"""
from pynecore.lib import close, script, strategy, ta


@script.strategy(
    "Bar Mag 1D v2 ON",
    overlay=True,
    initial_capital=10000,
    default_qty_type=strategy.fixed,
    default_qty_value=0.01,
    use_bar_magnifier=True,
)
def main():
    fast = ta.sma(close, 3)
    slow = ta.sma(close, 10)

    if ta.crossover(fast, slow):
        strategy.entry('Long', strategy.long)
        strategy.exit('XL', from_entry='Long', limit=close + 1500, stop=close - 1000)

    if ta.crossunder(fast, slow):
        strategy.entry('Short', strategy.short)
        strategy.exit('XS', from_entry='Short', limit=close - 1500, stop=close + 1000)


# noinspection PyShadowingNames
def __test_bar_magnifier_matches_tradingview__(script_path, module_key, csv_reader,
                                               strat_equity_comparator):
    """
    Bar magnifier with real OHLCV data matches TradingView's magnifier results.

    Uses 60-minute BINANCE:BTCUSDT data aggregated to 1D chart bars.
    TradingView uses 60m as the magnifier timeframe for 1D charts.
    """
    import sys
    from pathlib import Path
    from pynecore.core.ohlcv_file import OHLCVReader
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.core.syminfo import SymInfo

    sys.modules.pop(module_key, None)

    data_dir = Path(script_path).parent / 'data'
    ohlcv_path = data_dir / 'bar_magnifier_60m.ohlcv'
    toml_path = data_dir / 'bar_magnifier_60m.toml'

    syminfo = SymInfo.load_toml(toml_path)
    syminfo.period = '1D'

    with OHLCVReader(ohlcv_path) as reader, \
            csv_reader('bar_magnifier_tv_trades.csv', subdir='data') as cr_trades:
        ohlcv_iter = reader.read_from(reader.start_timestamp, reader.end_timestamp)

        runner = ScriptRunner(
            Path(script_path),
            iter([]),
            syminfo,
            magnifier_iter=ohlcv_iter,
        )

        trade_iter = iter(cr_trades)
        trade_count = 0

        for candle, plot, new_closed in runner.run_iter():
            for trade in new_closed:
                good_entry = next(trade_iter)
                good_exit = next(trade_iter)
                strat_equity_comparator(trade, good_entry.extra_fields, good_exit.extra_fields)
                trade_count += 1

    assert trade_count > 0, "Expected at least one trade"
